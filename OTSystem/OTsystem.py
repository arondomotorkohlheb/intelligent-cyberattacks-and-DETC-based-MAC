from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict

import pickle
import sys
import os
sys.path.append("..")

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import cvxpy as cp
from pathlib import Path


from OTSystem.detector import Detector, StaticBoundDetector, ResidualBasedDetector, CumulativeResidualDetector
from OTSystem.state_estimator import Estimator, KalmanEstimator, Observer
from OTSystem.plant import Plant, Qube
from OTSystem.communication import CommunicationChannel, Packet
from OTSystem.controller import Controller, LQR, MPC
from OTSystem.support_functions import *
from OTSystem.authenticator import DetcMac, DetcMacFloat, DetcMacInt8

class DetectorDescription(TypedDict):
    detector_type: str
    bound: float
    weight_vector: np.ndarray
    mu: float


class space:
    def __init__(self, P, flipped = False, dimensions = (0,1,2,3)):
        self.P: np.ndarray = P
        self.flipped = flipped
        self.dimensions = dimensions
        self.missing_dimensions = tuple(sorted(set(range(4)) - set(self.dimensions)))
        self.P_full = np.zeros((4, 4))
        self.P_full[np.ix_(list(self.dimensions), list(self.dimensions))] = self.P

    def contains(self, x: np.ndarray) -> bool:
        if not self.flipped:
            return bool(x.T @ self.P_full @ x <= 1)
        else:
            return bool(x.T @ self.P_full @ x >= 1)

    def sample_point(self, probability = 0.99):
        scale = 1.0 / chi2.ppf(probability, len(self.dimensions)) 
        covariance = scale * np.linalg.inv(self.P)

        x_nd = np.random.multivariate_normal(
            mean=np.zeros(len(self.dimensions)),
            cov=covariance
        )
        # add random uniform value fo the missing dimensions
        x = np.zeros(4)
        x[list(self.dimensions)] = x_nd
        x[list(self.missing_dimensions)] = np.random.uniform(-np.pi, np.pi, len(self.missing_dimensions))

        return x

    def sample_set_from_boundary(self, resolution = 8):
        # sample a random point on the surface of the ellipsoid defined by P
        # sample a random point on the unit sphere in n dimensions
        x_nd = ellipsoid_spherical_sample(self.P, resolution)
        x_nd_shape = x_nd.shape
        

        # add random uniform value fo the missing dimensions
        x = np.zeros((x_nd_shape[0], 4))
        x[:, list(self.dimensions)] = x_nd 
        x[:, list(self.missing_dimensions)] = np.random.uniform(-np.pi, np.pi, (x_nd_shape[0], len(self.missing_dimensions)))

        return x
    

class OTSystem:
    def __init__(self, plant: Qube, estimator: Estimator, controller: Controller, detectors: list[Detector], 
                 plant2controller: CommunicationChannel, detector_tuning: bool = False, delay: float = 0, DetcMac1: DetcMac | None = None, DetcMac2: DetcMac | None = None):
        self.plant = plant
        self.plant2controller = plant2controller
        self.estimator = estimator
        self.controller = controller
        self.controller.estimator = estimator
        self.detectors = detectors
        self.detector_tuning = detector_tuning
        for detector in self.detectors:
            detector.ot_system = self
        self.number_of_detections = 0
        self.delay = delay

        self.detc_mac_at_plant = DetcMac1
        self.detc_mac_at_controller = DetcMac2
        
        if self.delay >= self.plant.ts:
            raise ValueError(f"Delay {self.delay} is greater than or equal to the plant sampling time {self.plant.ts}. Please set delay < plant.ts.")

        #init plant and send one packet

        self.plant.update()
        measurement_message = np.array(self.plant.y, dtype=np.float32)
        if self.detc_mac_at_plant is not None:
            tag = self.detc_mac_at_plant(measurement_message)
        else:
            tag = None
        
        new_packet = Packet(message = measurement_message, tag = tag)
        self.plant2controller.send_packet(new_packet)

        if self.detc_mac_at_plant is None:
            self.name = f"{self.controller.name}_{self.estimator.name}"
        else:
            self.name = f"{self.controller.name}_{self.estimator.name}_{self.detc_mac_at_plant.name}"
        

        # create folder ot_systems if it does not exist
        if not os.path.exists(f"../figures/{self.name}"):
            os.makedirs(f"../figures/{self.name}")

        self.fname = f"{Path("ot_systems")}/{self.name}.pkl"
        self.figure_directory = f"../figures/{self.name}"
        self.object_directory = f"../ot_systems/{self.name}"

        print("finding nominal state space...")
        self.nominal_state_space = self.define_nominal_space(T = 200, resolution = 12)
        print("verifying nominal state space and measuring detection metrics...")
        self.verify_nominal_space_and_detection_metrics(repetitions = 10, T = 20)
        print("finding safe state space...")
        self.safe_state_space = self.define_safe_space_4d(T = 1, resolution = 10)
        print("finding unsafe state space...")
        self.unsafe_state_space = self.define_unsafe_space_3d(T = 1, resolution = 10)
        print("verifying safe state space...")
        self.safe_space_verification(repetitions = 100, T =1)
        print("verifying unsafe state space...")
        self.unsafe_space_verification(repetitions = 100, T =1)
        print("nominal state space sampling verification...")
        self.nominal_space_sampling_verification(repetitions = 100000)

        self.reset()        

        self.save()

    @property
    def info(self):
        return {
            "plant" : self.plant.info,
            "estimator" : self.estimator.info,
            "controller" : self.controller.info,
            "detectors" : [detector.info for detector in self.detectors],
            "plant2controller" : self.plant2controller.info,
            "delay" : self.delay,
        }
    
    def save(self):
        with open(self.fname, "wb") as f:
            pickle.dump(self, f)

    @property
    def plant_in_unsafe_state_space(self) -> bool:
        return self.unsafe_state_space.contains(self.plant.x)

    def reset(self, x0 = None, reset_detector_highest_value: bool = False):
        if x0 is None:
            x = self.sample_state_from_nominal_space()
        else:
            x = x0
        self.plant.t = 0
        self.plant.reset(x)
        self.estimator.reset(x_hat0 = x)
        self.controller.reset()
        for detector in self.detectors:
            detector.reset(reset_highest_value=reset_detector_highest_value)
        self.number_of_detections = 0
        self.plant2controller.reset()

        #init plant and send one packet
        self.plant.update()
        measurement_message = np.array(self.plant.y, dtype=np.float32)
        if self.detc_mac_at_plant is not None:
            tag = self.detc_mac_at_plant(measurement_message)
        else:
            tag = None
        
        new_packet = Packet(message = measurement_message, tag = tag)

        self.plant2controller.send_packet(new_packet)

    def step(self):
        #0 account for delay by updating the plant
        self.plant.update(ts = self.delay)

        #1 receive new packet from plant2controller channel
        self.measurement_packet = self.plant2controller.receive_packet()

        self.measurement_data = self.measurement_packet.message.copy()

        if self.detc_mac_at_controller is not None:
            tag = self.measurement_packet.tag
            if tag is None:
                raise ValueError("Tag is None, but DetcMac is used at controller. Please check the packet sent from plant to controller.")
            elif not self.detc_mac_at_controller.authentication_check(tag = tag, message = self.measurement_data):
                    self.number_of_detections += 1
                    self.measurement_packet.detected = True
        
        for detector in self.detectors:
            if detector.detect(self.measurement_data):
                self.number_of_detections += 1
                self.measurement_packet.detected = True
        if self.measurement_packet.detected:
            self.estimator.update(self.controller.last_control_input, self.plant.y)
        else:
            self.estimator.update(self.controller.last_control_input, self.measurement_data)

        self.plant.set_control_input(self.controller.control())

        self.plant.update(ts = self.plant.ts - self.delay)

        #3 create new packet
        measurement_message = np.array(self.plant.y, dtype=np.float32)
        if self.detc_mac_at_plant is not None:
            tag = self.detc_mac_at_plant(measurement_message)
        else:
            tag = None

        new_packet = Packet(message = measurement_message, tag = tag)
        
        #3 send new measurement packet to plant2controller channel
        self.plant2controller.send_packet(new_packet)

    def define_nominal_space(self, T: float = 0.1, resolution: int = 4) -> space:
        time_steps = int(T/self.plant.ts)
        states = np.zeros((time_steps, 4))

        angle_grids = [np.linspace(0, np.pi, int(resolution/2)), np.linspace(0, np.pi, int(resolution/2)), np.linspace(0, 2*np.pi, resolution)]
        angle_sorted_max_radius = {}
        

        for _ in range(time_steps):
            self.step()
            states[_] = [self.plant.x[0], self.plant.x[1], self.plant.x[2], self.plant.x[3]]
            radius, angles = cartesian_convert_to_nd_spherical(states[_])
            # find the angle grid corresponding to the angles
            angle1, angle2, angle3 = [np.digitize(angle, grid) - 1 for angle, grid in zip(angles, angle_grids)]

            if (angle1, angle2, angle3) not in angle_sorted_max_radius:
                angle_sorted_max_radius[(angle1, angle2, angle3)] = (radius, states[_])
            else:
                if radius > angle_sorted_max_radius[(angle1, angle2, angle3)][0]:
                    angle_sorted_max_radius[(angle1, angle2, angle3)] = (radius, states[_])
            

        # get only the 4d cartesian grid points in angle_sorted_max_radius as grid points
        X = np.array([point for _, point in angle_sorted_max_radius.values()])

        if X.size == 0:
            X = states

        N, n = X.shape

        P = cp.Variable((n, n), PSD=True)

        constraints = [
            cp.quad_form(X[i], P) <= 1
            for i in range(N)
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(P)),
            constraints # type: ignore
        )

        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        P_opt = P.value  #type: ignore

        plot_4d_ellipsoid_and_points(P_opt, points = states, highlighted_points = X, path = f"{self.figure_directory}/nominal_state_space_construction.png", ellipsoid_projection_name = r"$ x^\top H^{n} x = 1$", highlighted_points_name = r"$\mathcal{X}^{n}$", points1_name = r"$\mathcal{X}^{0}$", transparency = 0.06)

        # Sigma = np.cov(states, rowvar=False, bias=True)

        # print("extrapolated: ", Sigma)

        # probability = 0.99

        # scale = 1.0 / chi2.ppf(probability, 4)
        # covariance = scale * np.linalg.inv(P_opt) #type: ignore
        # print("computed: ", covariance)

        # heatmap_covariances(extrapolated=Sigma, computed=covariance, path=f"{self.figure_directory}/nominal_state_space_covariance_comparison.png")

        return space(P_opt)

    def verify_nominal_space_and_detection_metrics(self, repetitions: int = 100, T: float = 1):
        state_history = np.zeros((repetitions, int(T/self.plant.ts), self.plant.n))
        detection_history = np.zeros((repetitions * int(T/self.plant.ts), len(self.detectors)))
        x0_history = np.zeros((repetitions, self.plant.n))

        i4 = 0
        for i1 in range(repetitions):
            self.reset()
            x0_history[i1, :] = self.plant.x
            for i2 in range(int(T/self.plant.ts)):
                self.step()
                state_history[i1, i2, :] = self.plant.x

        self.detector_tuning = True
        if self.detector_tuning:
            for detector in self.detectors:
                detector.bound = detector.highest_value_encountered * 0.6

        if isinstance(self.detectors[-1], CumulativeResidualDetector) and isinstance(self.detectors[-2], CumulativeResidualDetector) and isinstance(self.detectors[-3], ResidualBasedDetector) and isinstance(self.detectors[-4], ResidualBasedDetector):
            self.detectors[-1].mu = self.detectors[-3].bound * 0.9
            self.detectors[-2].mu = self.detectors[-4].bound * 0.9
        else:
            raise ValueError("Detectors are not in the expected order. Please check the order of detectors in the OTSystem.")

        self.reset(x0=x0_history[0, :], reset_detector_highest_value = True)
        i4 = 0
        for i1 in range(repetitions):
            self.reset(x0=x0_history[i1, :])
            for i2 in range(int(T/self.plant.ts)):
                self.step()
                for i3, detector in enumerate(self.detectors):
                    detection_history[i4, i3] = detector.detection_metric_value
                i4 += 1

        self.detector_tuning = True
        if self.detector_tuning:
            for detector in self.detectors:
                detector.bound = detector.highest_value_encountered * 0.6
                

        plot_4d_ellipsoid_and_points(self.nominal_state_space.P, state_history.reshape(-1, self.plant.n), f"{self.figure_directory}/nominal_state_space_verification.png", ellipsoid_projection_name = r"$ x^\top H^{n} x = 1$", points1_name = r"$\mathcal N (0, H_{\mathbb{X}^n})$")
        plot_distribution({detector.notation: detection_history[:, i].flatten() for i, detector in enumerate(self.detectors)}, f"{self.figure_directory}/detection_metrics_distribution.png", bins=100, bounds={detector.notation: detector.bound for detector in self.detectors})

        #print the highest values
        for i, detector in enumerate(self.detectors):
            print(f"Highest {detector.name} metric value: {detector.highest_value_encountered}, bound: {detector.bound}")

    def sample_state_from_nominal_space(self, probability = 0.999):
        return self.nominal_state_space.sample_point(probability = probability)
    
    def sample_state_from_safe_space(self, probability = 0.99):
        return self.safe_state_space.sample_point(probability = probability)

    def sample_state_from_unsafe_space(self, probability = 0.5):
        while True:
            x =  self.unsafe_state_space.sample_point(probability = probability)
            if self.unsafe_state_space.contains(x):
                return x

    def check_stability_of_point(self, T:float, x: np.ndarray, stability_assumption: bool) -> tuple[bool, bool]:
        self.reset(x0=x)
        unstable_alpha = 120*np.pi/180

        for _ in range(int(T/self.plant.ts)):
            self.step()
            if np.abs(self.plant.x[3]) > unstable_alpha:
                return False, False
            if self.nominal_state_space.contains(self.plant.x):
                return True, False
            
        return stability_assumption, True

    def bisection_search_step(self, T:float, inner_point: np.ndarray, outer_point: np.ndarray, stability_assumption: bool) -> tuple[np.ndarray, bool, bool]:
        new_point = (inner_point.copy() + outer_point.copy())/2
        new_point_stable, stability_assumed = self.check_stability_of_point(T, new_point, stability_assumption = stability_assumption)
        return new_point, new_point_stable, stability_assumed

    def define_unsafe_space_2d(self, T: float = 0.1, resolution: int = 120) -> space:

        P1 = self.nominal_state_space.P.copy()

        P11 = np.zeros((2, 2))

        P11[0, 0] = P1[1, 1]
        P11[0, 1] = P1[1, 3]
        P11[1, 0] = P1[3, 1]
        P11[1, 1] = P1[3, 3]

        alpha_alpha_dot_boundary_points = ellipsoid_spherical_sample(P11, resolution=resolution)

        initial_points = np.vstack((np.zeros((resolution)), alpha_alpha_dot_boundary_points[:, 0], np.zeros((resolution)), alpha_alpha_dot_boundary_points[:, 1])).T

        stable_points_safe_assumption = []
        unstable_points_safe_assumption = []

        boundary_points_safe_assumption= []

        
        for inner_point_1 in initial_points:

            inner_point = inner_point_1.copy()

            outer_point = inner_point.copy()
            
            outer_point_stable = True
           
            while outer_point_stable:
                outer_point *= 2
                outer_point_stable, _ = self.check_stability_of_point(T, outer_point, stability_assumption = True)
                if outer_point_stable:
                    stable_points_safe_assumption.append(outer_point.copy())
                else:
                    unstable_points_safe_assumption.append(outer_point.copy())

            
            while np.sum(np.abs(inner_point.copy() - outer_point.copy())) > 0.1:
                new_point = (inner_point.copy() + outer_point.copy())/2
                new_point, new_point_stable, _ = self.bisection_search_step(T, inner_point, outer_point, stability_assumption = True)

               
                if new_point_stable:
                    inner_point = new_point.copy()
                    stable_points_safe_assumption.append(new_point.copy())
                else:
                    outer_point = new_point.copy()
                    unstable_points_safe_assumption.append(new_point.copy())

            boundary_points_safe_assumption.append(inner_point.copy())

            
        
        boundary_points_safe_assumption = np.array(boundary_points_safe_assumption)

        stable_points_safe_assumption = np.array(stable_points_safe_assumption)
        unstable_points_safe_assumption = np.array(unstable_points_safe_assumption)

        boundary_points_safe_assumption_2d = np.hstack((boundary_points_safe_assumption[:, 1].reshape(-1, 1), boundary_points_safe_assumption[:, 3].reshape(-1, 1)))

        Pus = cp.Variable((2, 2), PSD=True)

        constraints = [
            cp.quad_form(boundary_points_safe_assumption_2d[i], Pus) <= 1
            for i in range(len(boundary_points_safe_assumption_2d))
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(Pus)),
            constraints # type: ignore
        )
        
        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        Pus_value = Pus.value * 0.8 ** 2 #type: ignore

        # plot the points and the ellipsis


        plt.figure(figsize=(12, 12))

        #plot all the points: blue joint stable, red joint unstable, pink stable safe assumption, orange unstable safe assumption, purple stable unsafe assumption, brown unstable unsafe assumption
        plt.scatter(stable_points_safe_assumption[:, 1]* 180/np.pi, stable_points_safe_assumption[:, 3]* 180/np.pi, color='blue', label='Joint stable points', s=10, alpha=0.45)
        plt.scatter(unstable_points_safe_assumption[:, 1]* 180/np.pi, unstable_points_safe_assumption[:, 3]* 180/np.pi, color='red', label='Joint unstable points', s=10, alpha=0.45)
        
        #green boundary points
        plt.scatter(boundary_points_safe_assumption[:, 1]* 180/np.pi, boundary_points_safe_assumption[:, 3]* 180/np.pi, color='green', marker='o', label='Boundary points Safe Space', s=20)
        
        #plotting the ellipse defined by Pus_value
        ellipse_unsafe = ellipsoid_projection(Pus_value, dims=(0, 1), n_points=300)
        #plot all the points: green joint stable, red joint unstable, blue stable safe assumption, orange unstable safe assumption, purple stable unsafe assumption, brown unstable unsafe assumption
        if ellipse_unsafe is not None:
            plt.plot(ellipse_unsafe[0, :]* 180/np.pi, ellipse_unsafe[1, :]* 180/np.pi, color="green", linewidth=2, label=r"$\mathbb{X}^{us}$")

        # adding nominal state space ellipse in blue
        ellipse_nominal = ellipsoid_projection(self.nominal_state_space.P, dims=(1, 3), n_points=300)
        if ellipse_nominal is not None:
            plt.plot(ellipse_nominal[0, :]* 180/np.pi, ellipse_nominal[1, :]* 180/np.pi, color="blue", linewidth=2, label=r"$\mathbb{X}^n$")        


        plt.xlabel(r"$\dot \alpha [\degree/s]$")
        plt.ylabel(r"$ \alpha [\degree]$")
        plt.title("Finding Unsafe Space")
        plt.legend()        

        plt.savefig(f"{self.figure_directory}/2d_unsafe_space_definition.png")

        self.reset(reset_detector_highest_value = True)
        
        return space(Pus_value, dimensions=(1, 3), flipped=True)

    def define_safe_and_unsafe_spaces_4d(self, T: float = 0.1, resolution: int = 8) -> tuple[space, space]:


        initial_points = ellipsoid_spherical_sample(self.nominal_state_space.P, resolution=resolution)
        
        joint_inner_points = []
        joint_outer_points = []

        stable_points_safe_assumption = []
        unstable_points_safe_assumption = []

        stable_points_unsafe_assumption = []
        unstable_points_unsafe_assumption = []

        boundary_points_safe_assumption= []
        boundary_points_unsafe_assumption= []


        
        for inner_point_1 in initial_points:

            inner_point = inner_point_1.copy()

            outer_point = inner_point.copy()
            
            outer_point_stable = True
           
            while outer_point_stable:
                outer_point *= 2
                outer_point_stable, stability_assumed = self.check_stability_of_point(T, outer_point, stability_assumption = True)
                if outer_point_stable:
                    joint_inner_points.append(outer_point.copy())
                else:
                    joint_outer_points.append(outer_point.copy())

            
            
            processes_joint = True
            inner_point_safe_assumption = inner_point.copy()
            inner_point_unsafe_assumption = inner_point.copy()

            outer_point_safe_assumption = outer_point.copy()
            outer_point_unsafe_assumption = outer_point.copy()

            number_of_steps = 0
            while processes_joint and np.sum(np.abs(inner_point.copy() - outer_point.copy())) > 0.1 and number_of_steps < 30:
                new_point = (inner_point.copy() + outer_point.copy())/2
                new_point, new_point_stable, stability_assumed = self.bisection_search_step(T, inner_point, outer_point, stability_assumption = True)

                if stability_assumed:
                    processes_joint = False
                    if new_point_stable:
                        inner_point_safe_assumption = new_point.copy()
                        stable_points_safe_assumption.append(inner_point_safe_assumption.copy())
                    else:
                        outer_point_safe_assumption = new_point.copy()
                        unstable_points_safe_assumption.append(outer_point_safe_assumption.copy())
                else:
                    if new_point_stable:
                        inner_point = new_point.copy()
                        joint_inner_points.append(new_point.copy())
                    else:
                        outer_point = new_point.copy()
                        joint_outer_points.append(new_point.copy())

                number_of_steps += 1

            # at this point either processes_joint is False or the distance between inner and outer point is less than 0.1

            if processes_joint:
                boundary_points_safe_assumption.append(inner_point.copy())
                boundary_points_unsafe_assumption.append(outer_point.copy())
            else:
                number_of_steps = 0
                while np.sum(np.abs(inner_point_safe_assumption.copy() - outer_point_safe_assumption.copy())) > 0.1 and number_of_steps < 30: # constructing the safe space
                    new_point, new_point_stable, stability_assumed = self.bisection_search_step(T, inner_point_safe_assumption, outer_point_safe_assumption, stability_assumption = True)
                    if new_point_stable:
                        inner_point_safe_assumption = new_point.copy()
                        stable_points_safe_assumption.append(inner_point_safe_assumption.copy())
                    else:
                        outer_point_safe_assumption = new_point.copy()
                        unstable_points_safe_assumption.append(outer_point_safe_assumption.copy())
                    number_of_steps += 1

                boundary_points_safe_assumption.append(inner_point_safe_assumption.copy())

                inner_point_unsafe_assumption = inner_point.copy()
                outer_point_unsafe_assumption = outer_point.copy()

                number_of_steps = 0
                while np.sum(np.abs(inner_point_unsafe_assumption.copy() - outer_point_unsafe_assumption.copy())) > 0.1 and number_of_steps < 30: # constructing the unsafe space
                    new_point, new_point_stable, stability_assumed = self.bisection_search_step(T, inner_point_unsafe_assumption, outer_point_unsafe_assumption, stability_assumption = False)
                    if new_point_stable:
                        inner_point_unsafe_assumption = new_point.copy()
                        stable_points_unsafe_assumption.append(inner_point_unsafe_assumption.copy())
                    else:
                        outer_point_unsafe_assumption = new_point.copy()
                        unstable_points_unsafe_assumption.append(outer_point_unsafe_assumption.copy())
                    number_of_steps += 1

                boundary_points_unsafe_assumption.append(outer_point_unsafe_assumption.copy())


        
        boundary_points_safe_assumption = self.plant.state_contain(np.array(boundary_points_safe_assumption))
        boundary_points_unsafe_assumption = self.plant.state_contain(np.array(boundary_points_unsafe_assumption))

        joint_stable_points = self.plant.state_contain(np.array(joint_inner_points))
        joint_unstable_points = self.plant.state_contain(np.array(joint_outer_points))

        stable_points_safe_assumption = self.plant.state_contain(np.array(stable_points_safe_assumption))
        unstable_points_safe_assumption =self.plant.state_contain(np.array(unstable_points_safe_assumption))

        stable_points_unsafe_assumption = self.plant.state_contain(np.array(stable_points_unsafe_assumption))
        unstable_points_unsafe_assumption = self.plant.state_contain(np.array(unstable_points_unsafe_assumption))

        Ps = cp.Variable((4, 4), PSD=True)

        constraints = [
            cp.quad_form(boundary_points_safe_assumption[i], Ps) <= 1
            for i in range(len(boundary_points_safe_assumption))
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(Ps)),
            constraints # type: ignore
        )
        
        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        Ps_value = Ps.value * 2 ** 4  #type: ignore

        # plot the points and the ellipsis

        Pus = cp.Variable((4, 4), PSD=True)

        constraints = [
            cp.quad_form(boundary_points_unsafe_assumption[i], Pus) <= 1
            for i in range(len(boundary_points_unsafe_assumption))
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(Pus)),
            constraints # type: ignore
        )
        
        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        Pus_value = Pus.value

        plot_4d_ellipsoid_and_points(Ps_value, points = joint_stable_points, points2 = joint_unstable_points, highlighted_points = boundary_points_safe_assumption, path = f"{self.figure_directory}/safe_state_space_construction.png",transparency = 0.8)
        plot_4d_ellipsoid_and_points(Pus_value, points = joint_stable_points, points2 = joint_unstable_points, highlighted_points = boundary_points_unsafe_assumption, path = f"{self.figure_directory}/unsafe_state_space_construction.png", transparency = 0.8)
        
        return space(Ps_value), space(Pus_value, flipped=True)
   
    def define_safe_space_4d(self, T: float = 0.1, resolution: int = 8) -> space:

        initial_points = ellipsoid_spherical_sample(self.nominal_state_space.P, resolution=resolution)*0.6

        stable_points_unsafe_assumption = []
        unstable_points_unsafe_assumption = []

        boundary_points_unsafe_assumption= []
        
        for inner_point_1 in initial_points:

            inner_point = inner_point_1.copy()

            outer_point = inner_point.copy()

            stable_points_unsafe_assumption.append(outer_point.copy())
            
            outer_point_stable = True
           
            while outer_point_stable:
                outer_point *= 2
                outer_point_stable, _ = self.check_stability_of_point(T, outer_point, stability_assumption = False)
                if outer_point_stable:
                    stable_points_unsafe_assumption.append(outer_point.copy())
                else:
                    unstable_points_unsafe_assumption.append(outer_point.copy())
          

            number_of_steps = 0
            while np.sum(np.abs(inner_point.copy() - outer_point.copy())) > 0.1 and number_of_steps < 20:
                new_point = (inner_point.copy() + outer_point.copy())/2
                new_point_stable, _ =  self.check_stability_of_point(T, new_point, stability_assumption = False)

                if new_point_stable:
                    inner_point = new_point.copy()
                    stable_points_unsafe_assumption.append(new_point.copy())
                else:
                    outer_point = new_point.copy()
                    unstable_points_unsafe_assumption.append(new_point.copy())

                number_of_steps += 1
            
            boundary_points_unsafe_assumption.append(outer_point.copy())


        boundary_points_unsafe_assumption = self.plant.state_contain(np.array(boundary_points_unsafe_assumption))
        stable_points_unsafe_assumption = self.plant.state_contain(np.array(stable_points_unsafe_assumption))
        unstable_points_unsafe_assumption = self.plant.state_contain(np.array(unstable_points_unsafe_assumption))     

        Ps = cp.Variable((4, 4), PSD=True)

        constraints = [
            cp.quad_form(boundary_points_unsafe_assumption[i], Ps) <= 1
            for i in range(len(boundary_points_unsafe_assumption))
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(Ps)),
            constraints # type: ignore
        )
        
        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        Ps_value = Ps.value * 2.5 ** 4  #type: ignore

        plot_4d_ellipsoid_and_points(Ps_value, points = stable_points_unsafe_assumption, points2 = unstable_points_unsafe_assumption, ellipsoid_projection_name = r"$ x^\top H^{s} x = 1$", highlighted_points = boundary_points_unsafe_assumption, path = f"{self.figure_directory}/safe_state_space_construction4d.png",transparency = 0.8)
 
        return space(Ps_value)

    def define_unsafe_space_3d(self, T: float = 0.1, resolution: int = 8) -> space:

        reduced_P = np.zeros((3, 3))
        reduced_P[0, 0] = self.nominal_state_space.P[0, 0]
        reduced_P[0, 1] = self.nominal_state_space.P[0, 1]
        reduced_P[0, 2] = self.nominal_state_space.P[0, 3]
        reduced_P[1, 0] = self.nominal_state_space.P[1, 0]
        reduced_P[1, 1] = self.nominal_state_space.P[1, 1]
        reduced_P[1, 2] = self.nominal_state_space.P[1, 3]
        reduced_P[2, 0] = self.nominal_state_space.P[3, 0]
        reduced_P[2, 1] = self.nominal_state_space.P[3, 1]
        reduced_P[2, 2] = self.nominal_state_space.P[3, 3]

        initial_points3d = ellipsoid_spherical_sample(reduced_P, resolution=resolution)

        initial_points = np.vstack([initial_points3d[:, 0], initial_points3d[:, 1], np.zeros(initial_points3d.shape[0]), initial_points3d[:, 2]]).T

        stable_points_unsafe_assumption = []
        unstable_points_unsafe_assumption = []

        boundary_points_unsafe_assumption= []

        
        for inner_point_1 in initial_points:

            inner_point = inner_point_1.copy()

            outer_point = inner_point.copy()
            
            outer_point_stable = True
           
            while outer_point_stable:
                outer_point *= 2
                outer_point_stable, _ = self.check_stability_of_point(T, outer_point, stability_assumption = True)
                if outer_point_stable:
                    stable_points_unsafe_assumption.append(outer_point.copy())
                else:
                    unstable_points_unsafe_assumption.append(outer_point.copy())
          

            number_of_steps = 0
            while np.sum(np.abs(inner_point.copy() - outer_point.copy())) > 0.01 and number_of_steps < 20:
                new_point = (inner_point.copy() + outer_point.copy())/2
                new_point[2] = np.pi * 0.99
                new_point_stable1, _ =  self.check_stability_of_point(T, new_point, stability_assumption = True)
                new_point[2] = -np.pi * 0.99
                new_point_stable2, _ =  self.check_stability_of_point(T, new_point, stability_assumption = True)
                new_point[2] = 0
                new_point_stable3, _ =  self.check_stability_of_point(T, new_point, stability_assumption = True)

                new_point_stable = new_point_stable1 or new_point_stable2 or new_point_stable3


                if new_point_stable:
                    inner_point = new_point.copy()
                    stable_points_unsafe_assumption.append(new_point.copy())
                else:
                    outer_point = new_point.copy()
                    unstable_points_unsafe_assumption.append(new_point.copy())

                number_of_steps += 1
            
            boundary_points_unsafe_assumption.append(outer_point.copy())
        
        #convert all to arrays
        boundary_points_unsafe_assumption = np.array(boundary_points_unsafe_assumption)[:, [0, 1, 3]]

        stable_points_unsafe_assumption = np.array(stable_points_unsafe_assumption)[:, [0, 1, 3]]
        unstable_points_unsafe_assumption = np.array(unstable_points_unsafe_assumption)[:, [0, 1, 3]]

        Ps = cp.Variable((3, 3), PSD=True)

        constraints = [
            cp.quad_form(boundary_points_unsafe_assumption[i], Ps) <= 1
            for i in range(len(boundary_points_unsafe_assumption))
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(Ps)),
            constraints # type: ignore
        )

        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        Ps_value = Ps.value * 0.5 ** 3  #type: ignore
        
        # plot the points and the ellipsis

        plot_3d_ellipsoid_and_points(Ps_value, points = stable_points_unsafe_assumption, ellipsoid_projection_name = r"$ x^\top H^{us} x = 1$", points2 = unstable_points_unsafe_assumption, highlighted_points = boundary_points_unsafe_assumption, path = f"{self.figure_directory}/unsafe_state_space_construction_3d.png",transparency = 0.8)
        
        return space(Ps_value, dimensions = (0, 1, 3), flipped = True)
   
    def define_safe_space_3d(self, T: float = 0.1, resolution: int = 8) -> space:
       
        reduced_P = np.zeros((3, 3))
        reduced_P[0, 0] = self.nominal_state_space.P[0, 0]
        reduced_P[0, 1] = self.nominal_state_space.P[0, 1]
        reduced_P[0, 2] = self.nominal_state_space.P[0, 3]
        reduced_P[1, 0] = self.nominal_state_space.P[1, 0]
        reduced_P[1, 1] = self.nominal_state_space.P[1, 1]
        reduced_P[1, 2] = self.nominal_state_space.P[1, 3]
        reduced_P[2, 0] = self.nominal_state_space.P[3, 0]
        reduced_P[2, 1] = self.nominal_state_space.P[3, 1]
        reduced_P[2, 2] = self.nominal_state_space.P[3, 3]

        initial_points3d = ellipsoid_spherical_sample(reduced_P, resolution=resolution)

        initial_points = np.vstack([initial_points3d[:, 0], initial_points3d[:, 1], np.zeros(initial_points3d.shape[0]), initial_points3d[:, 2]]).T

        stable_points_unsafe_assumption = []
        unstable_points_unsafe_assumption = []

        boundary_points_unsafe_assumption= []

        
        for inner_point_1 in initial_points:

            inner_point = inner_point_1.copy()

            outer_point = inner_point.copy()
            
            outer_point_stable = True
           
            while outer_point_stable:
                outer_point *= 2
                outer_point_stable, _ = self.check_stability_of_point(T, outer_point, stability_assumption = False)
                if outer_point_stable:
                    stable_points_unsafe_assumption.append(outer_point.copy())
                else:
                    unstable_points_unsafe_assumption.append(outer_point.copy())
          

            number_of_steps = 0
            while np.sum(np.abs(inner_point.copy() - outer_point.copy())) > 0.1 and number_of_steps < 20:
                new_point = (inner_point.copy() + outer_point.copy())/2
                new_point[2] = np.pi
                new_point_stable1, _ =  self.check_stability_of_point(T, new_point, stability_assumption = False)
                new_point[2] = -np.pi
                new_point_stable2, _ =  self.check_stability_of_point(T, new_point, stability_assumption = False)
                new_point[2] = 0
                new_point_stable3, _ =  self.check_stability_of_point(T, new_point, stability_assumption = False)

                new_point_stable = new_point_stable1 and new_point_stable2 and new_point_stable3

                if new_point_stable:
                    inner_point = new_point.copy()
                    stable_points_unsafe_assumption.append(new_point.copy())
                else:
                    outer_point = new_point.copy()
                    unstable_points_unsafe_assumption.append(new_point.copy())

                number_of_steps += 1
            
            boundary_points_unsafe_assumption.append(outer_point.copy())
        
        #convert all to arrays
        boundary_points_unsafe_assumption = np.array(boundary_points_unsafe_assumption)[:, [0, 1, 3]]
        stable_points_unsafe_assumption = np.array(stable_points_unsafe_assumption)[:, [0, 1, 3]]
        unstable_points_unsafe_assumption = np.array(unstable_points_unsafe_assumption)[:, [0, 1, 3]]

        Ps = cp.Variable((3, 3), PSD=True)

        constraints = [
            cp.quad_form(boundary_points_unsafe_assumption[i], Ps) <= 1
            for i in range(len(boundary_points_unsafe_assumption))
        ]

        problem = cp.Problem(
            cp.Minimize(-cp.log_det(Ps)),
            constraints # type: ignore
        )

        while True:
            try:
                problem.solve(solver=cp.CLARABEL)  # or another solver supporting log_det
                break
            except cp.SolverError as e:
                print(f"SolverError: {e}. Retrying...")
                continue

        Ps_value = Ps.value * 4 ** 3  #type: ignore
        
        # plot the points and the ellipsis

        plot_3d_ellipsoid_and_points(Ps_value, points = stable_points_unsafe_assumption, ellipsoid_projection_name = r"$ x^\top H^{s} x = 1$", points2 = unstable_points_unsafe_assumption, highlighted_points = boundary_points_unsafe_assumption, path = f"{self.figure_directory}/safe_state_space_construction.png",transparency = 0.8)
        
        return space(Ps_value, dimensions = (0, 1, 3))

    def safe_space_verification(self, repetitions: int = 100, T: float = 1):
        state_history = [] # np.zeros((repetitions, int(T/self.plant.ts), self.plant.n))
        boundary_points = self.safe_state_space.sample_set_from_boundary(10)

        for x0 in boundary_points:
            self.reset(x0)
            for i2 in range(int(T/self.plant.ts)):
                self.step()
                state_history.append(self.plant.x)

        plot_4d_ellipsoid_and_points(self.safe_state_space.P_full, np.array(state_history), f"{self.figure_directory}/safe_state_space_verification.png", ellipsoid_projection_name = r"$ x^\top H^{s} x = 1$", highlighted_points = boundary_points)

        self.reset(reset_detector_highest_value = True)

    def unsafe_space_verification(self, repetitions: int = 100, T: float = 1):
        resolution = int(repetitions ** (1/len(self.unsafe_state_space.dimensions)))
        state_history = [] # np.zeros((repetitions, int(T/self.plant.ts), self.plant.n))
        boundary_points = self.unsafe_state_space.sample_set_from_boundary(10)

        for boundary_point in boundary_points:
            self.reset(boundary_point)
            for i2 in range(int(T/self.plant.ts)):
                self.step()
                if not np.any(np.isinf(self.plant.x )) and not np.any(np.isnan(self.plant.x)):
                    state_history.append(self.plant.x)
                else:
                    break

        plot_4d_ellipsoid_and_points(self.unsafe_state_space.P_full, np.array(state_history), f"{self.figure_directory}/unsafe_state_space_verification.png", highlighted_points = boundary_points, ellipsoid_projection_name = r"$ x^\top H^{us} x = 1$")

        self.reset(reset_detector_highest_value = True)
    
    def nominal_space_sampling_verification(self, repetitions: int = 10000):
        state_history = [] # np.zeros((repetitions, int(T/self.plant.ts), self.plant.n))

        for i1 in range(repetitions):
            x = self.sample_state_from_nominal_space()
            state_history.append(x)

        plot_4d_ellipsoid_and_points(self.nominal_state_space.P_full, np.array(state_history), f"{self.figure_directory}/nominal_state_space_sampling_verification.png", ellipsoid_projection_name = r"$ x^\top H^{n} x = 1$")

        self.reset(reset_detector_highest_value = True)

    def __str__(self):
        return str(self.info)

    def collect_ot_traffic_data(self, repetitions: int = 1000, T: float = 0.1):
        print(f"Collecting OT traffic data for {repetitions} repetitions and T={T} seconds ...")
        y_history = np.zeros((repetitions, int(T/self.plant.ts), self.plant.l))
        expected_y_history = np.zeros((repetitions, int(T/self.plant.ts), self.plant.l))
        u_history = np.zeros((repetitions, int(T/self.plant.ts), self.plant.m))

        for i1 in range(repetitions):
            self.reset()
            for i2 in range(int(T/self.plant.ts)):
                y_history[i1, i2, :] = self.plant.y
                u_history[i1, i2, :] = self.controller.last_control_input
                self.step()
                expected_y_history[i1, i2, :] = self.detectors[2].predicted_output #type: ignore
        
        
        filename = f"ot_systems/LQR_KalmanEstimator/ot_traffic_data{repetitions}.npz"

        np.savez(filename, y=y_history, expected_y=expected_y_history, u=u_history)

def create_OTsystem(controller_type: str, estimator_type: str, detectors_params: list[DetectorDescription] | str, delay = 0, DetcMac_type = None, load: bool = False) -> OTSystem:

    # check if OT system is already constructed and saved as a pickle file
    if DetcMac_type is None:
        name = Path(f"OTSystem/ot_systems/{controller_type}_{estimator_type}.pkl")
    else:
        name = Path(f"OTSystem/ot_systems/{controller_type}_{estimator_type}_{DetcMac_type}.pkl")
    
    if load:
        if DetcMac_type is None:
            name = Path(f"OTSystem/ot_systems/{controller_type}_{estimator_type}.pkl")
        else:
            name = Path(f"OTSystem/ot_systems/{controller_type}_{estimator_type}_{DetcMac_type}.pkl")

    # print(name)
    # check if file exists:
    if os.path.exists(name):
        with open(name, "rb") as f:
            print(f"Loading OT system from {name}")
            return pickle.load(f)
    else:
        print("OT system is being constructed ...")

        plant = Qube()
        plant2controller = CommunicationChannel(1/plant.ts)
        
        if estimator_type == "KalmanEstimator":
            estimator = KalmanEstimator(plant.get_LTI())
        elif estimator_type == "Observer":
            estimator = Observer(plant.get_LTI())
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")

        if controller_type == "LQR":
            controller = LQR(plant.get_LTI(), estimator)
        elif controller_type == "MPC":
            controller = MPC(plant.get_LTI(), estimator)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        if DetcMac_type is None:
            DetcMac1 = None
            DetcMac2 = None
        elif DetcMac_type == "DetcMacFloat":
            DetcMac1 = DetcMacFloat()
            DetcMac2 = DetcMac1.copy()
        elif DetcMac_type == "DetcMacInt8":
            DetcMac1 = DetcMacInt8()
            DetcMac2 = DetcMac1.copy()
        else:
            raise ValueError(f"Unknown DetcMac type: {DetcMac_type}")

        detectors = []

        if type(detectors_params) == str:
            detector_tuning = True
            if detectors_params == "full_auto":
                detectors_params = [
                    DetectorDescription(detector_type="StaticBoundDetector", bound=np.inf, weight_vector=np.array([1, 0]), mu=0),
                    DetectorDescription(detector_type="StaticBoundDetector", bound=np.inf, weight_vector=np.array([0, 1]), mu=0),
                    DetectorDescription(detector_type="ResidualBasedDetector", bound=np.inf, weight_vector=np.array([1, 0]), mu=0),
                    DetectorDescription(detector_type="ResidualBasedDetector", bound=np.inf, weight_vector=np.array([0, 1]), mu=0),
                    DetectorDescription(detector_type="CumulativeResidualDetector", bound=np.inf, weight_vector=np.array([1, 0]), mu=0.09),
                    DetectorDescription(detector_type="CumulativeResidualDetector", bound=np.inf, weight_vector=np.array([0, 1]), mu=0.06)
                    ]
            else:
                raise ValueError(f"not implemented detector generation method: {detectors_params}")
        else:
            detector_tuning = False

        for detector_description in detectors_params:
            if detector_description["detector_type"] == "StaticBoundDetector":# type: ignore
                detectors.append(StaticBoundDetector(detector_description["bound"], detector_description["weight_vector"]))# type: ignore
            elif detector_description["detector_type"] == "ResidualBasedDetector":# type: ignore
                detectors.append(ResidualBasedDetector(plant.get_LTI(), detector_description["bound"], detector_description["weight_vector"])) # type: ignore
            elif detector_description["detector_type"] == "CumulativeResidualDetector":# type: ignore
                detectors.append(CumulativeResidualDetector(plant.get_LTI(), detector_description["bound"], detector_description["weight_vector"], detector_description["mu"]))# type: ignore
            else:
                raise ValueError(f"Unknown detector type: {detector_description['detector_type']}")# type: ignore
        ot_system = OTSystem(plant=plant, controller=controller, estimator=estimator, detectors=detectors, plant2controller=plant2controller, detector_tuning=detector_tuning, DetcMac1=DetcMac1, DetcMac2=DetcMac2)# type: ignore
        return ot_system

def load_ots_make_mac_copies():

    detectors = "full_auto"
    load = True
    ot_system1 = create_OTsystem(controller_type = "LQR", estimator_type = "KalmanEstimator", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")
    ot_system2 = create_OTsystem(controller_type = "LQR", estimator_type = "Observer", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")
    ot_system3 = create_OTsystem(controller_type = "MPC", estimator_type = "KalmanEstimator", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")
    ot_system4 = create_OTsystem(controller_type = "MPC", estimator_type = "Observer", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")

    ot_systems = [ot_system1, ot_system2, ot_system3, ot_system4]
    

    DetcMacint8_1 = DetcMacInt8()
    DetcMacint8_2 = DetcMacint8_1.copy()

    Detcfloat_1 = DetcMacFloat()
    Detcfloat_2 = Detcfloat_1.copy()

    for ot_system in ot_systems:
        ot_system.detc_mac_at_controller = DetcMacint8_1.copy()
        ot_system.detc_mac_at_plant = DetcMacint8_2.copy()
        ot_system.name = f"{ot_system.controller.__class__.__name__}_{ot_system.estimator.__class__.__name__}_DetcMacInt8"
        ot_system.fname = f"{Path("ot_systems")}/{ot_system.name}.pkl"
        ot_system.figure_directory = f"../figures"
        ot_system.object_directory = f"../ot_systems/{ot_system.name}"

        ot_system.save()
    
    for ot_system in ot_systems:
        ot_system.detc_mac_at_controller = Detcfloat_1.copy()
        ot_system.detc_mac_at_plant = Detcfloat_2.copy()
        ot_system.name = f"{ot_system.controller.__class__.__name__}_{ot_system.estimator.__class__.__name__}_DetcMacFloat"

        ot_system.fname = f"{Path("ot_systems")}/{ot_system.name}.pkl"
        ot_system.figure_directory = f"../figures"
        ot_system.object_directory = f"../ot_systems/{ot_system.name}"
        ot_system.save()

def load_ot1_int8detc_mac():
    detectors = "full_auto"
    load = True
    ot_system1 = create_OTsystem(controller_type = "LQR", estimator_type = "KalmanEstimator", detectors_params = detectors, load=load, DetcMac_type = "DetcMacInt8")

    return ot_system1

def load_ot1_float32detc_mac():
    detectors = "full_auto"
    load = True
    ot_system1 = create_OTsystem(controller_type = "LQR", estimator_type = "KalmanEstimator", detectors_params = detectors, load=load, DetcMac_type = "DetcMacFloat")
    return ot_system1  

if __name__ == "__main__":
    # Example usage
    # load ot

    detectors = "full_auto"

    load = True
    ot_system1 = create_OTsystem(controller_type = "LQR", estimator_type = "KalmanEstimator", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")
    ot_system2 = create_OTsystem(controller_type = "LQR", estimator_type = "Observer", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")
    ot_system3 = create_OTsystem(controller_type = "MPC", estimator_type = "KalmanEstimator", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")
    ot_system4 = create_OTsystem(controller_type = "MPC", estimator_type = "Observer", detectors_params = detectors, load=load) #, DetcMac_type = "DetcMacFloat")

    ot_systems = [ot_system1, ot_system2, ot_system3, ot_system4]

    nominal_Plist = [ot_system.nominal_state_space.P_full for ot_system in ot_systems]
    safe_Plist = [ot_system.safe_state_space.P_full for ot_system in ot_systems]
    unsafe_Plist = [ot_system.unsafe_state_space.P for ot_system in ot_systems]

    plot_n_ellipsoids(nominal_Plist, path = f"../figures/nominal_state_space_comparison.png")
    plot_n_ellipsoids(safe_Plist, path = f"../figures/safe_state_space_comparison.png")
    plot_n_ellipsoids_3d(unsafe_Plist, path = f"../figures/unsafe_state_space_comparison.png")
    
    