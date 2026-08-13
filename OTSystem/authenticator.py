import numpy as np
from abc import ABC, abstractmethod
import sys
from scipy.signal import place_poles
import matplotlib.pyplot as plt
sys.path.append("..")
import galois #type: ignore
GF = galois.GF(256)
import numpy.typing as npt
import copy



class DetcMac(ABC):
    def __init__(self):
        self.name: str
        self.tag_size: int
    
    @abstractmethod
    def __call__(self, message: npt.NDArray[np.float32]) -> int:
        pass

    @abstractmethod
    def call_without_update(self, message: npt.NDArray[np.float32]) -> int:
        pass

    def authentication_check(self, tag: int, message: npt.NDArray[np.float32]) -> bool:
        tag_generated = self.call_without_update(message)
        if tag_generated == tag:
            self(message)
            return True
        else:
            return False
    
    def copy(self):
        return copy.deepcopy(self)

class DetcMacFloat(DetcMac):
    def __init__(self, n = 2, m = 2, l = 1):
        super().__init__()
        self.name = "DetcMacFloat"
        self.n = n
        self.m = m
        self.l = l
        self.tag_size: int = l
        self.etc_A = np.array([[1, 0], [1, 1]])
        self.etc_B = np.array([[1], [1]])
        self.etc_C =  np.array([[1, 1]])
        controller_poles = np.array([0.1, 0.3]) # type: ignore
        self.etc_K = - place_poles(self.etc_A, self.etc_B, controller_poles).gain_matrix # type: ignore
        self.M = np.array([[1, 0], [0, 1]]) * 0.5

        self.x = np.random.rand(2, 1)
        self.x_hat = self.x.copy()
        self.x_hat2nu = np.array([[[1, 1]]])
        self.updated_x_hat = True
        
        self.nu = 1
        self.alpha = 0.9

    @property
    def u(self):
        return self.etc_K @ self.x_hat
    
    def update_nu(self):
        self.nu = self.alpha * self.nu + np.abs((1 - self.alpha) * self.x_hat2nu @ self.x_hat).squeeze()
    
    def trigger_condition(self)-> bool:
        self.updated_x_hat = bool(np.all(np.linalg.norm(self.x - self.x_hat) > self.nu * np.linalg.norm(self.x)))
        return self.updated_x_hat
    
    def update_states(self, dist):
        if self.trigger_condition():
            self.x_hat = self.x.copy()
        self.x = self.etc_A @ self.x + self.etc_B @ self.u + self.M @ dist

    def __call__(self, message: npt.NDArray[np.float32]):
        self.update_states(message)
        self.update_nu()
        return int((self.etc_C @ self.x)[0,0]*16)

    def call_without_update(self, message: npt.NDArray[np.float32]) -> int:
        nu_temp = self.alpha * self.nu + np.abs((1 - self.alpha) * self.x_hat2nu @ self.x_hat).squeeze()

        if bool(np.all(np.linalg.norm(self.x - self.x_hat) > nu_temp)):
            x_hat_temp = self.x.copy()
        else:
            x_hat_temp = self.x_hat.copy()      

        x_temp = self.etc_A @ self.x + self.etc_B @ (self.etc_K @ x_hat_temp) + self.M @ message

        return int((self.etc_C @ x_temp)[0,0]*16)


class DetcMacInt8(DetcMac):
    def __init__(self, n = 4, m = 2*4, l = 1): # assuming the message is 2 float32s
        super().__init__()
        self.name = "EtcMacInt8"
        self.n = n # number of bytes of the states
        self.m = m # number of bytes of the message
        self.l = l # number of bytes of the tag
        self.tag_size: int = l

        # internal state update matrices
        self.x2x  = GF.Random((self.n,self.n))
        self.x_hat2x = GF.Random((self.n,self.n))
        self.m2x = GF.Random((self.n,self.m))

        # nu uprate matrices
        self.nu2nu = GF.Random((self.n,self.n))
        self.x2nu = GF.Random((self.n,self.n))

        # output matrix
        self.C = GF.Random((self.l,self.n))

        # states
        self.x = GF.Random(self.n)
        self.hat_x = self.x.copy()
        self.nu = GF.Random(self.n)
        self.updated_x_hat = True

    def __call__(self, message: npt.NDArray[np.float32]) -> int:
        x_new, x_hat_new, nu_new, new_tag = self.compute_updates(message)
        self.x = x_new
        self.nu = nu_new
        self.hat_x = x_hat_new
        return int(new_tag)
    
    def call_without_update(self, message: npt.NDArray[np.float32]) -> int:
        _, _, _, new_tag = self.compute_updates(message)
        return int(new_tag)

    @property
    def tag(self):
        return int((self.C @ self.x)[0,0])
    
    def compute_updates(self, message):
        # Update the internal states based on the received message
        m = GF(np.array([message], dtype=np.float32).view(np.uint8)).reshape(-1, 1)

        # nu update
        nu_new = (
            self.nu2nu @ self.nu
            + self.x2nu @ self.x
        )

        # x_hat update -> control trigger
        distance = np.sum(
            np.array(self.x + self.hat_x, dtype=np.uint8) # xor operation
        )

        threshold = np.sum(
            np.array(nu_new, dtype=np.uint8)
        )

        if distance > threshold:
            x_hat_new = self.x.copy()
            self.updated_x_hat = True
        else:
            x_hat_new = self.hat_x.copy()
            self.updated_x_hat = False


        # x update
        x_new = (
            self.x2x @ self.x
            + self.x_hat2x @ x_hat_new
            + self.m2x @ m
        )
        
        new_tag = int((self.C @ x_new)[0,0])

        return x_new, x_hat_new, nu_new, new_tag


      
if __name__ == "__main__":
    from OTSystem.support_functions import plot_step_signal
    print("Testing detc mac")
    authenticator = DetcMacFloat()
    N = 128
    
    measurement_signal = np.concatenate([np.sin(np.linspace(0, N * 0.01 * np.pi, N)).reshape(-1, 1), np.cos(np.linspace(0, N * 0.01 * np.pi, N)).reshape(-1, 1)], axis=1)
    signatures = np.zeros((N, 1), dtype=float)
    triggered = np.zeros((N, 1), dtype = int)

    for i in range(N):
        dist = measurement_signal[i].reshape(-1, 1)
        signature = authenticator(dist)
        signatures[i] = signature
        triggered[i] = int(authenticator.updated_x_hat)


    plot_step_signal(signatures.flatten(), triggered.flatten())