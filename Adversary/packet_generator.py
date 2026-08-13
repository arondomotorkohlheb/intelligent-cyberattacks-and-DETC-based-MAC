from __future__ import annotations
from typing import TYPE_CHECKING

from gymnasium import spaces
from abc import abstractmethod, ABC
import numpy as np
from OTSystem.communication import Packet
from Adversary.policy import Policy, SupervisedNnPolicy, SupervisedRnnPolicy

if TYPE_CHECKING:
    # from Adversary.policy import Policy, SupervisedNnPolicy
    from Adversary.adversary_observer import AdversaryInformation, AdversaryObserver
    from simulator import Simulator


class PacketGenerator(ABC):
    def __init__(self, packet_data_size: int):
        super().__init__()
        self.packet_data_size = packet_data_size
        self.name: str
        self.observer: AdversaryObserver
        self.last_packet_data = np.zeros((self.packet_data_size))
        self.parameter_ranges: dict[str, tuple[float, float]] | dict[str, tuple[int, int]]  = {}

    def reset(self):
        self.last_packet_data = np.zeros((self.packet_data_size))

    def copy(self):
        return self.__class__(self.packet_data_size, **{param: getattr(self, param) for param in self.parameter_ranges.keys()})

    @abstractmethod
    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        pass

    @property
    @abstractmethod
    def info(self) -> dict:
        pass

    def __str__(self):
        return str(self.info)

class NoisePacketGenerator(PacketGenerator):
    def __init__(self, packet_data_size: int, bias: float, std: float):
        super().__init__(packet_data_size)
        self.bias = bias
        self.std = std
        self.name: str = "NoisePacketGenerator"
        self.parameter_ranges = {
            "bias": (-0.1, 0.1),
            "std": (0.001, 1.0)
        }

    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        self.last_packet_data = np.array(np.random.normal(loc=self.bias, scale=self.std, size=self.packet_data_size), dtype=np.float32)
        return Packet(self.last_packet_data, malicious=True)
    
    @property
    def info(self) -> dict:
        return {
            "packet_generator_type": self.name,
            "bias": self.bias,
            "std": self.std,
            "parameter_ranges": self.parameter_ranges
        }


class AdditiveNoisePacketGenerator(PacketGenerator):
    def __init__(self, packet_data_size: int, bias: float, std: float):
        super().__init__(packet_data_size)
        self.bias = bias
        self.std = std
        self.name: str = "AdditiveNoisePacketGenerator"
        self.parameter_ranges = {
            "bias": (-0.1, 0.1),
            "std": (0.001, 1.0)
        }

    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        self.last_packet_data = np.array(np.random.normal(loc=self.bias, scale=self.std, size=self.packet_data_size) + adversary_information.last_y, dtype=np.float32)
        return Packet(self.last_packet_data, malicious=True)

    @property
    def info(self) -> dict:
        return {
            "packet_generator_type": self.name,
            "bias": self.bias,
            "std": self.std,
            "parameter_ranges": self.parameter_ranges
        }

class AsymptoticPacketGenerator(PacketGenerator):
    def __init__(self, packet_data_size: int, factor = 0.9, std = 0.0):
        super().__init__(packet_data_size)
        self.factor = factor
        self.std = std
        self.name: str = "AsymptoticPacketGenerator"
        self.parameter_ranges = {
            "factor": (0.1, 0.999),
            "std": (0.0001, 0.01)
        }

    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        self.last_packet_data = np.array(adversary_information.last_yr * self.factor + np.random.normal(loc=0.0, scale=self.std, size=self.packet_data_size), dtype=np.float32)
        return Packet(self.last_packet_data, malicious=True)

    @property
    def info(self) -> dict:
        return {
            "packet_generator_type": self.name,
            "factor": self.factor,
            "std": self.std,
            "parameter_ranges": self.parameter_ranges
        }

class IntelligentPacketGenerator(PacketGenerator):
    def __init__(self, packet_data_size: int, adversary_observer: AdversaryObserver, number_of_neurons: int, number_of_layers: int):
        super().__init__(packet_data_size)
        self.adversary_observer = adversary_observer
        self.observation_space = adversary_observer.observation_space
        self.policy: Policy
        self.name: str
        self.number_of_neurons = number_of_neurons
        self.number_of_layers = number_of_layers

    @abstractmethod
    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        pass
    
    def train(self, simulator: Simulator):
        self.policy.train(simulator)
    
    def reset(self):
        super().reset()
        self.policy.reset()

    @property
    def info(self) -> dict:
        return {
            "packet_generator_type": self.name,
            "policy": self.policy.info
        }
    

class NNIntelligentPacketGenerator(IntelligentPacketGenerator):
    def __init__(self, packet_data_size: int, adversary_observer: AdversaryObserver, number_of_neurons: int, number_of_layers: int):
        super().__init__(packet_data_size, adversary_observer, number_of_neurons, number_of_layers)
        self.adversary_observer = adversary_observer
        self.observation_space = adversary_observer.observation_space
        self.policy = SupervisedNnPolicy(observation_space = adversary_observer.supervised_observation_space, 
                            action_space = spaces.Box(low = -1, high = 1, shape = (packet_data_size,)),
                            observation_space_description = "_pg_" + adversary_observer.observation_space_description,
                            action_space_description = f"({packet_data_size}c)",
                            number_of_neurons = number_of_neurons,
                            number_of_layers = number_of_layers)
        self.name: str = "NNIntelligentPacketGenerator"
        self.parameter_ranges = self.policy.parameter_ranges

    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        self.last_packet_data = self.policy(adversary_information.previous_w_ot_info)
        return Packet(self.last_packet_data, malicious=True)


class RnnIntelligentPacketGenerator(IntelligentPacketGenerator):
    def __init__(self, packet_data_size: int, adversary_observer: AdversaryObserver, number_of_neurons: int, number_of_layers: int):
        super().__init__(packet_data_size, adversary_observer, number_of_neurons, number_of_layers)
        self.observation_space = adversary_observer.observation_space
        self.policy = SupervisedRnnPolicy(observation_space = adversary_observer.supervised_observation_space, 
                             action_space = spaces.Box(low = -1, high = 1, shape = (packet_data_size,)),
                             observation_space_description = "_pg_" + adversary_observer.observation_space_description,
                             action_space_description = f"({packet_data_size}c)",
                             number_of_neurons = number_of_neurons,
                             number_of_layers = number_of_layers)
        self.name: str = "RnnIntelligentPacketGenerator"
        self.parameter_ranges = self.policy.parameter_ranges

    def generate_packet(self, adversary_information: AdversaryInformation)-> Packet:
        self.last_packet_data = self.policy(adversary_information.previous_ot_info.reshape(1, 1, -1))
        return Packet(self.last_packet_data, malicious=True)

    

# class RnnMacApproximator():