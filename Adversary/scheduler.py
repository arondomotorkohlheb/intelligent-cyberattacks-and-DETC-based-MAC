from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod, ABC
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    # from Adversary.adversary_observer import Observation
    from policy import Policy
    from simulator import Simulator
    from Adversary.adversary_observer import AdversaryInformation

class Scheduler(ABC):
    def __init__(self):
        super().__init__()
        self.name: str
        self.previous_decision = False

    @abstractmethod
    def schedule(self, adversary_information: AdversaryInformation)-> bool:
        pass

    @property
    @abstractmethod
    def info(self) -> dict:
        pass

    def __str__(self):
        return str(self.info)

class AlwaysInject(Scheduler):
    def __init__(self):
        super().__init__()
        self.name: str = "AlwaysInject"

    def schedule(self, adversary_information: AdversaryInformation)-> bool:
        self.previous_decision = True
        return True
    
    @property
    def info(self) -> dict:
        return {
            "scheduler_type": self.name
        }

class RandomStep(Scheduler):
    def __init__(self, probability: float):
        super().__init__()
        self.name: str = "RandomStep"
        self.previous_decision = False
        self.probability = probability

    def schedule(self, adversary_information: AdversaryInformation)-> bool:
        if not self.previous_decision:
            decision = np.random.rand() < self.probability
            self.previous_decision = decision
            return decision
        else:
            return True
    
    @property
    def info(self) -> dict:
        return {
            "scheduler_type": self.name,
            "probability": self.probability
        }
        
class IntelligentScheduler(Scheduler):
    def __init__(self, observation_space: spaces.Space):
        super().__init__()
        self.name: str = "none"
        self.policy: Policy
        self.parameter_ranges = self.policy.parameter_ranges

    def schedule(self, adversary_information: AdversaryInformation)-> bool:
        if not self.previous_decision:
            self.previous_decision = bool(self.policy(adversary_information.previous_w_yr))
            return self.previous_decision
        else:
            return True
        
    def train(self, simulator: Simulator):
        self.policy.train(simulator)
        
    @property
    def info(self) -> dict:
        return {
            "scheduler_type": self.name,
            "policy": self.policy.info
        }

class ErrorBoundScheduler(Scheduler):
    def __init__(self, observation_space: spaces.Space, error_bound: float):
        super().__init__()
        self.name: str = "ErrorBoundScheduler"
        self.error_bound = error_bound

    def schedule(self, adversary_information: AdversaryInformation)-> bool:
        if not self.previous_decision:
            self.previous_decision = bool(self.rule(adversary_information))
            return self.previous_decision
        else:
            return True

    def rule(self, adversary_information: AdversaryInformation) -> bool:
        # Check if the last packet generation error is within the error bound
        last_error = np.linalg.norm(adversary_information.packet_generation_error, axis = 0)
        if np.all(last_error < self.error_bound):
            return True  # Inject a packet if the error exceeds the bound
        else:
            return False  # Do not inject a packet if the error is within the bound
        
    @property
    def info(self) -> dict:
        return {
            "scheduler_type": self.name,
            "error_bound": self.error_bound
        }
