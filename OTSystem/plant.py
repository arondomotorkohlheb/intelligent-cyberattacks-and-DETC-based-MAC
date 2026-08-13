import numpy as np
from scipy.linalg import expm
from abc import ABC, abstractmethod
from pprint import pprint


class LTImodel:
    def __init__(self, A, B, C, D, ts):
        # continous
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.ts = ts
        self.x = np.zeros(A.shape[0]) # initial state, not really used but can be helpful for debugging
        self.name = "LTI_model"

        #discrete
        self.Ad = expm(A * self.ts)
        self.Bd = (self.Ad - np.eye(A.shape[0])) @ np.linalg.inv(A) @ B
        self.Cd = C
        self.Dd = D

    def predict(self, past_x, u, time_elapsed = None):
        if time_elapsed is None:
            time_elapsed = self.ts
        return expm(self.A * time_elapsed) @ past_x + (expm(self.A * time_elapsed) - np.eye(self.A.shape[0])) @ np.linalg.inv(self.A) @ self.B @ u

    def make_new_LTI(self, ts):
        return LTImodel(self.A, self.B, self.C, self.D, ts)

class Plant(ABC):
    def __init__(self, x0, ts):
        super().__init__()
        self.x = x0
        self.ts = ts
        self.n = self.x.shape[0] 
        self.m: int
        self.l: int
        self.unstable_state_index: int
        self.t = 0
        self.control_limit = 0.05
        self.controller_on = True
        self.name = self.__class__.__name__

    @property
    @abstractmethod
    def info(self) -> dict:
        pass

    def __str__(self):
        return str(self.info)

    @property
    def unstable_state(self):
        return self.x[self.unstable_state_index]

    @abstractmethod
    def update(self, ts = None) -> None:
        pass

    @abstractmethod
    def set_control_input(self, u) -> None:
        pass

    @abstractmethod
    def get_LTI(self, linearization_point = None) -> LTImodel:
        pass

    @abstractmethod
    def reset(self, x0) -> None:
        pass


    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        pass
    

class Qube(Plant):
    def __init__(self, x0 = np.array([0, 0, 0, 0]), ts = 0.002, noise_std = 10):
        super().__init__(x0, ts)
        self.unstable_state_index = 3 # alpha is the unstable state
        self.measurement_noise_std = 0.0001
        self.L0 = 1
        self.L1 = 4
        self.r = self.L0/self.L1
        self.inertia_ratio = 1/3 *self.L0**3/self.L1**3 
        self.gravity_coefficient = 9.81*10/2
        self.ts = ts
        self.voltage2torque = 800
        self.damping_theta = 0.8
        self.damping_alpha = 0.8
        self.spring = 0.01
        self.noise_mean = 0
        self.noise_std = noise_std
        self.voltage = 0
        self.stable = True
        self.LTI = self.get_LTI()
        self.l = 2
        self.m = 1
        self.n = self.x.shape[0]
        self.t = 0
        self.name = "Qube"
        

    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "state dimension": self.n,
            "input dimension": self.m,
            "output dimension": self.l,
            "sampling time step": self.ts,
            "unstable_state_index": self.unstable_state_index
        }

    @property
    def theta_dot(self):
        return self.x[0]
    
    @property
    def alpha_dot(self):
        return self.x[1]
    
    @property
    def theta(self):
        return self.x[2]
    
    @property
    def alpha(self):
        return self.x[3]
    
    @property
    def angular_velocity_vector(self):
        return np.array([self.theta_dot, self.alpha_dot])
    
    @property
    def endpoints(self):
        return np.array([self.get_point(0), self.get_point(self.L1)])

    
    def get_point(self, s):
        return self.L0 * np.array([
                np.cos(self.theta),
                np.sin(self.theta),
                0
            ]) + s * np.array([
                np.sin(self.alpha) * np.sin(self.theta),
                -np.sin(self.alpha) * np.cos(self.theta),
                np.cos(self.alpha)
            ])


    @property
    def C0(self):
        return np.array([
            -np.sin(self.theta),
            np.cos(self.theta),
            0
        ])
    
    @property
    def C1(self):
        return np.array([
            np.sin(self.alpha) * np.cos(self.theta),
            np.sin(self.alpha) * np.sin(self.theta),
            0
        ])
    
    @property
    def C2(self):
        return np.array([
            np.cos(self.alpha) * np.sin(self.theta),
            -np.cos(self.alpha) * np.cos(self.theta),
            -np.sin(self.alpha)
        ])
    
    @property
    def M(self):
        return np.array([
            [
                self.r**2 * (self.C0.T @ self.C0) + self.r * (self.C0.T @ self.C1) + (1/3) * (self.C1.T @ self.C1),
                (1/2) *self.r * (self.C0.T @ self.C2) + (1/3) * (self.C1.T @ self.C2)
            ],
            [
                (1/2) * self.r * (self.C0.T @ self.C2) + (1/3) * (self.C1.T @ self.C2),
                (1/3) * (self.C2.T @ self.C2)
            ]
        ])
    
    @property
    def dC1_dalpha(self):
        return np.array([
            np.cos(self.alpha) * np.cos(self.theta),
            np.cos(self.alpha) * np.sin(self.theta),
            0
            ])
    
    @property
    def dC2_dalpha(self):
        return np.array([
                -np.sin(self.alpha) * np.sin(self.theta),
                np.sin(self.alpha) * np.cos(self.theta),
                -np.cos(self.alpha)
            ])

    @property
    def dM_dalpha(self):
        M_2 = np.array([[self.r, 0], [0, 0]])
        M_3 = np.array([[0, 0.5*self.r], [0.5*self.r, 0]])
        M_4 = np.array([[1/3, 0], [0, 0]])
        M_5 = np.array([[0, 1/3], [1/3, 0]])
        M_6 = np.array([[0, 0], [0, 1/3]])


        return (
                    (                     self.C0.T @ self.dC1_dalpha) * M_2 +
                    (                     self.C0.T @ self.dC2_dalpha) * M_3 +
                    (self.dC1_dalpha.T @ self.C1 + self.C1.T @ self.dC1_dalpha) * M_4 +
                    (self.dC1_dalpha.T @ self.C2 + self.C1.T @ self.dC2_dalpha) * M_5 +
                    (self.dC2_dalpha.T @ self.C2 + self.C2.T @ self.dC2_dalpha) * M_6
        )                                       
    @property
    def y(self):
        return self.LTI.C @ self.x + np.random.normal(0, self.measurement_noise_std, size=self.l)
    
    @property
    def Mbar_inverse(self):
        M11 = self.M[0, 0]
        M12 = self.M[0, 1]
        M22 = self.M[1, 1]

        det = (M22 * (M11+self.inertia_ratio) - M12**2)

        if det == 0:
            return np.array([[0, 0], [0, 0]])
        else:
            return np.array([
                [M22, -M12],
                [-M12, M11 + self.inertia_ratio]]) / det

    def set_control_input(self, u):
        if self.controller_on:
            self.voltage = min(u[0], self.control_limit) if u[0] > 0 else max(u[0], -self.control_limit)
        else:
            self.voltage = 0

    def reset(self, x0 = np.array([0, 0, 0, 0])):
        self.x = x0
        self.t = 0
        self.voltage = 0
        self.stable = True
        self.controller_on = True

    def update(self, ts = None):
        if ts is None:
            ts = self.ts

        # Malpha_term =  np.array([[-1*self.alpha_dot, 0], [0.5*self.theta_dot, -0.5*self.alpha_dot]]) @ self.dM_dalpha @ self.angular_velocity_vector
        
        gravity_term = np.array([
            0,
            +self.gravity_coefficient * np.sin(self.alpha)])
        
        tau = np.array([
        self.voltage2torque * self.voltage - self.damping_theta * self.theta_dot - self.spring * self.theta, 
        -self.damping_alpha * self.alpha_dot])
        
        angular_acceleration_vector = self.Mbar_inverse @ (tau + gravity_term) # + Malpha_term)
        # add noise on angular acceleration to simulate unmodeled dynamics and sensor noise
        noise = np.concatenate([np.zeros(1), np.random.normal(self.noise_mean, self.noise_std, size=1)])
        xdot = np.concatenate([angular_acceleration_vector + noise, self.angular_velocity_vector])
        self.x = self.x + xdot * ts
        self.t += ts

        #check if theta and alpha are within -pi to pi limits, if not wrap around

        if np.abs(self.theta_dot) > 1e5 or np.abs(self.alpha_dot) > 1e5:
            self.controller_on = False

        if np.abs(self.theta) > 2*np.pi:
            self.x[2] = self.theta % (2*np.pi)

        if self.theta > np.pi:
            self.x[2] = self.theta - 2*np.pi
        elif self.theta < -np.pi:
            self.x[2] = self.theta + 2*np.pi
        
        if np.abs(self.alpha) > 2*np.pi:
            self.x[3] = self.alpha % (2*np.pi)

        if self.alpha > np.pi:
            self.x[3] = self.alpha - 2*np.pi
        elif self.alpha < -np.pi:
            self.x[3] = self.alpha + 2*np.pi

    def state_contain(self, states: np.ndarray) -> np.ndarray:
        for i in range(states.shape[0]):
            if np.abs(states[i, 2]) > np.pi:
                states[i, 2] = states[i, 2] % (2*np.pi)

            if states[i, 2] > np.pi:
                states[i, 2] = states[i, 2] - 2*np.pi
            elif states[i, 2] < -np.pi:
                states[i, 2] = states[i, 2] + 2*np.pi
            
            if np.abs(states[i, 3]) > np.pi:
                states[i, 3] = states[i, 3] % (2*np.pi)

            if states[i, 3] > np.pi:
                states[i, 3] = states[i, 3] - 2*np.pi
            elif states[i, 3] < -np.pi:
                states[i, 3] = states[i, 3] + 2*np.pi

        return states

    def get_LTI(self, linearization_point = None):
        # compute mass-inverse at current operating point

        # damping on velocities -> A11
        A11 = self.Mbar_inverse @ np.array([
            [-self.damping_theta, 0],
            [0, -self.damping_alpha]
        ])

        # position terms -> A12
        # first equation: -spring * theta
        # second equation: gravity linearized ~ 0.5 * gravity_term * alpha
        A12 = self.Mbar_inverse @ np.array([
            [-self.spring, 0],
            [0, +self.gravity_coefficient]
        ])

        # input gain -> B1 (voltage -> torque on first joint)
        B1 = self.Mbar_inverse @ np.array([[self.voltage2torque], [0]])

        # assemble full continuous-time state-space (states: [dot_theta, dot_alpha, theta, alpha])
        A = np.block([
            [A11, A12],
            [np.eye(2), np.zeros((2, 2))]
        ])

        B = np.vstack([B1, np.zeros((2, 1))])

        # output matrices (measure angular velocities and positions as before)
        C = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        D = np.zeros((2, 1))

        # discrete approximation using the simulation timestep

        return LTImodel(A, B, C, D, self.ts)


if __name__ == "__main__":
    qube_instance = Qube()
    print(qube_instance.get_LTI())
    for _ in range(10):
        qube_instance.set_control_input([0])
        qube_instance.update()
        print(qube_instance.x)