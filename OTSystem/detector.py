from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod
import numpy as np

if TYPE_CHECKING:
    from OTSystem.plant import LTImodel
    from OTSystem.OTsystem import OTSystem


class Detector:
    def __init__(self):
        self.name: str
        self.ot_system: OTSystem
        self.bound: float
        self.highest_value_encountered: float
        self.weight_vector: np.ndarray
        self.notation: str

    @abstractmethod
    def detect(self, *args, **kwargs) -> bool:
        pass

    def reset(self, reset_highest_value: bool = True):
        if reset_highest_value:
            self.highest_value_encountered = 0

    @property
    @abstractmethod
    def info(self) -> dict:
        pass

    def __str__(self):
        return str(self.info)

    @property
    @abstractmethod
    def detection_metric_value(self) -> float:
        pass

class StaticBoundDetector(Detector):
    def __init__(self,bound, weight_vector):
        super().__init__()
        self.upper_bound = bound
        self.lower_bound = -bound
        self.bound = bound
        self.weight_vector = weight_vector
        self.name = "StaticBound" + ("1" if self.weight_vector[0] == 1 else "2")
        self.notation = r"$D^{I}_1$" if self.weight_vector[0] == 1 else r"$D^{I}_2$"
        self.highest_value_encountered = -np.inf
        
    def detect(self, state_measurement):
        self.measurement = state_measurement @ self.weight_vector

        self.highest_value_encountered = float(np.maximum(self.highest_value_encountered, self.measurement))
        if self.measurement > self.upper_bound or self.measurement < self.lower_bound:
            return True
        else:
            return False
    
    @property
    def detection_metric_value(self):
        return float(np.abs(self.measurement))
    
    @property
    def info(self) -> dict:
        return {
            "type": "static bound",
            "bound": self.bound,
            "weight_vector": self.weight_vector
        }


class ResidualBasedDetector(Detector):
    def __init__(self, model: LTImodel, bound, weight_vector):
        super().__init__()
        self.model = model
        self.bound = bound
        self.residual_weights = weight_vector
        self.name = "ResidualBased" + ("1" if self.residual_weights[0] == 1 else "2")
        self.notation = r"$ D^{II}_1$" if self.residual_weights[0] == 1 else r"$ D^{II}_2$"
        self.highest_value_encountered = 0
        self.residual = 0

    def detect(self, measurement):
        # predict next state based on current measurement and model
        predicted_state = self.model.predict(self.ot_system.estimator.state_estimate, self.ot_system.controller.last_control_input, self.ot_system.estimator.elapsed_time)
        predicted_output = self.model.C @ predicted_state
        self.predicted_output = predicted_output
        # calculate residual
        self.residual = float(np.linalg.norm(self.residual_weights @ (predicted_output - measurement)))
        self.highest_value_encountered = float(np.maximum(self.highest_value_encountered, self.residual))
        # detect if residual exceeds threshold
        if self.residual > self.bound:
            return True
        else:
            return False

    def reset(self, reset_highest_value: bool = True):
        super().reset(reset_highest_value)
        self.residual = 0
        self.predicted_output = np.zeros((2))

    @property
    def detection_metric_value(self):
        return self.residual

    @property
    def info(self) -> dict:
        return {
            "type": "residual based",
            "bound": self.bound,
            "weight_vector": self.residual_weights
        }

class CumulativeResidualDetector(Detector):
    def __init__(self, model: LTImodel, bound, weight_vector, mu):
        super().__init__()
        self.model = model
        self.bound = bound
        self.residual_weights = weight_vector
        self.mu = mu
        self.name = "CumulativeResidual" + ("1" if self.residual_weights[0] == 1 else "2")
        self.notation = r"$ D^{III}_1$" if self.residual_weights[0] == 1 else r"$ D^{III}_2$"
        self.highest_value_encountered = 0
        self.cumulative_residual = 0

    def reset(self, reset_highest_value: bool = True):
        super().reset(reset_highest_value)
        self.cumulative_residual = 0

    def detect(self, measurement):
        predicted_state = self.model.predict(self.ot_system.estimator.state_estimate, self.ot_system.controller.last_control_input, self.ot_system.estimator.elapsed_time)
        predicted_output = self.model.C @ predicted_state
        residual = np.abs(predicted_output - measurement)
        self.cumulative_residual = max(0, self.cumulative_residual + float(self.residual_weights @ residual) - self.mu)
        self.highest_value_encountered = float(np.maximum(self.highest_value_encountered, self.cumulative_residual))
        if self.cumulative_residual > self.bound:
            return True
        else:
            return False

    @property
    def detection_metric_value(self):
        return self.cumulative_residual
    
    @property
    def info(self) -> dict:
        return {
            "type": "cumulative residual based",
            "bound": self.bound,
            "weight_vector": self.residual_weights,
            "mu": self.mu
        }
    

    
if __name__ == "__main__":
    pass