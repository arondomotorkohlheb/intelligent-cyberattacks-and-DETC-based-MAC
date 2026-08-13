from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces
from typing import TypedDict
import numpy as np

if TYPE_CHECKING:
    from OTSystem.communication import CommunicationChannel
    from Adversary.scheduler import Scheduler
    from Adversary.adversary import Adversary
    from simulator import Simulator


class AdversaryInformation:
    def __init__(self, packet_data_size: int, observations: np.ndarray):
        self.last_y = np.zeros((packet_data_size)) 
        self.last_yr = np.zeros((packet_data_size))
        self.last_yi = np.zeros((packet_data_size))
        self.last_action = False
        self.observations = observations
        self.previous_w_yr = np.zeros((observations.shape[0], packet_data_size))
        self.packet_generation_error = np.ones((observations.shape[0], packet_data_size))
        self.last_control_input = np.zeros((1))
        self.previous_w_u = np.zeros((observations.shape[0], 1))
        self.y_only = True

    @property
    def previous_w_ot_info(self):
        if self.y_only:
            return self.previous_w_yr
        else:
            return np.concatenate([self.previous_w_yr, self.previous_w_u], axis=1)

    @property
    def previous_ot_info(self):
        if self.y_only:
            return self.last_yr
        else:
            return np.concatenate([self.last_yr, self.last_control_input], axis=0)
    
    def __str__(self):
        return str(self.observations)

class AdversaryObserver(ABC):
    def __init__(self, communication_channel: CommunicationChannel, packet_data_size: int, window_size: int, delay: int):
        super().__init__()
        self.observation_space_description: str =  str(window_size) + "x(1b_4c)"
        self.communication_channel = communication_channel
        self.packet_data_size = packet_data_size
        self.window_size = window_size
        self.delay = delay
        self.observation_space_description: str

        self.adversary_information = AdversaryInformation(packet_data_size=packet_data_size, observations=self.init_observations())
        self.name: str

        self.observation_space: spaces.Box
        self.supervised_observation_space: spaces.Box
        self.adversary: Adversary

        self.y_delay_buffer = np.zeros((self.delay+1, self.packet_data_size))
        self.non_inject_counter = 0

    def reset(self):
        self.adversary_information = AdversaryInformation(packet_data_size=self.packet_data_size, observations=self.init_observations())
        self.y_delay_buffer = np.zeros((self.delay+1, self.packet_data_size))
        self.non_inject_counter = 0
    
    @property
    def info(self) -> dict:
        return {
            "adversary_observer_type": self.name,
            "packet_data_size": self.packet_data_size,
            "window_size": self.window_size,
            "delay": self.delay,
            "observation_space": str(self.observation_space)
        }

    def __str__(self):
        return str(self.info)

    @abstractmethod
    def observe(self, simulator: Simulator) -> None:
        pass

    @abstractmethod
    def init_observations(self) -> np.ndarray:
        pass

class InjectAndListen(AdversaryObserver):
    def __init__(self, communication_channel: CommunicationChannel, packet_data_size: int, window_size: int, delay: int):
        super().__init__(communication_channel=communication_channel, packet_data_size=packet_data_size, window_size=window_size, delay=delay)
        self.name: str = "InjectAndListen"
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, packet_data_size*2 + 1))
        self.supervised_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, packet_data_size))
        self.observation_space_description: str =  str(window_size) + "x(5c)"
    
    def init_observations(self) -> np.ndarray:
        return np.zeros((self.window_size, self.packet_data_size*2 + 1))

    def observe(self, simulator: Simulator) -> None:
        self.adversary_information.last_action = self.adversary.scheduler.previous_decision
        # print(f"last_action: {self.last_action}")  
        self.adversary_information.last_yi = self.adversary.packet_generator.last_packet_data #type: ignore
        
        new_packet = self.communication_channel.listen2traffic()

        if new_packet is not None:
            self.y_delay_buffer = np.roll(self.y_delay_buffer, shift=1, axis=0)
            self.y_delay_buffer[0] = new_packet.message

        self.adversary_information.last_y = self.y_delay_buffer[-1]

        if self.adversary_information.last_action:
            self.non_inject_counter = 0
        else:
            self.non_inject_counter += 1

        self.adversary_information.last_yr = self.adversary_information.last_yi if self.adversary_information.last_action else self.adversary_information.last_y #type: ignore
        # self.adversary_information.last_yr = self.adversary_information.last_yi if (self.adversary_information.last_action or self.non_inject_counter < self.delay) else self.adversary_information.last_y #type: ignore

        # print(f"last_y: {self.adversary_information.last_y}, last_yi: {self.adversary_information.last_yi}, last_yr: {self.adversary_information.last_yr}")

        self.adversary_information.observations = np.roll(self.adversary_information.observations, shift=1, axis=0) #type: ignore
        self.adversary_information.observations[0] = np.concatenate([[self.adversary_information.last_action], self.adversary_information.last_y, self.adversary_information.last_yi]) #type: ignore

        self.adversary_information.previous_w_yr = np.roll(self.adversary_information.previous_w_yr, shift=1, axis=0)
        self.adversary_information.previous_w_yr[0] = self.adversary_information.last_yr

        # if not self.adversary_information.last_action:
        self.adversary_information.packet_generation_error = np.roll(self.adversary_information.packet_generation_error, shift=1, axis=0)
        self.adversary_information.packet_generation_error[0] = np.abs(self.adversary_information.last_yi - self.adversary_information.last_y)
        # print(f"last_y: {self.adversary_information.last_y}, last_yi: {self.adversary_information.last_yi}, last_yr: {self.adversary_information.last_yr}, packet_generation_error: {self.adversary_information.packet_generation_error[0]}")

class InjectAndListen2Channels(AdversaryObserver):
    def __init__(self, communication_channel: CommunicationChannel, packet_data_size: int, window_size: int, delay: int):
        super().__init__(communication_channel=communication_channel, packet_data_size=packet_data_size, window_size=window_size, delay=delay)
        self.name: str = "InjectAndListen2Channels"
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, packet_data_size*2 + 1 + 1))
        self.supervised_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, packet_data_size + 1))
        self.observation_space_description: str =  str(window_size) + "x(6c)"
        self.adversary_information.y_only = False
    
    def init_observations(self) -> np.ndarray:
        return np.zeros((self.window_size, self.packet_data_size*2 + 1 + 1))

    def reset(self):
        super().reset()
        self.adversary_information.y_only = False

    def observe(self, simulator: Simulator) -> None:
        self.adversary_information.last_action = self.adversary.scheduler.previous_decision
        # print(f"last_action: {self.last_action}")  
        self.adversary_information.last_yi = self.adversary.packet_generator.last_packet_data #type: ignore
        
        new_packet = self.communication_channel.listen2traffic()
        if new_packet is not None:
            self.y_delay_buffer = np.roll(self.y_delay_buffer, shift=1, axis=0)
            self.y_delay_buffer[0] = new_packet.message

        self.adversary_information.last_y = self.y_delay_buffer[-1]

        if self.adversary_information.last_action:
            self.non_inject_counter = 0
        else:
            self.non_inject_counter += 1

        self.adversary_information.last_control_input = simulator.ot_system.controller.last_control_input
        self.adversary_information.last_yr = self.adversary_information.last_yi if self.adversary_information.last_action else self.adversary_information.last_y #type: ignore
        # self.adversary_information.last_yr = self.adversary_information.last_yi if (self.adversary_information.last_action or self.non_inject_counter < self.delay) else self.adversary_information.last_y #type: ignore
        # print(f"last_y: {self.adversary_information.last_y}, last_yi: {self.adversary_information.last_yi}, last_yr: {self.adversary_information.last_yr}")
        self.adversary_information.observations = np.roll(self.adversary_information.observations, shift=1, axis=0) #type: ignore
        self.adversary_information.observations[0] = np.concatenate([[self.adversary_information.last_action], self.adversary_information.last_y, self.adversary_information.last_yi, self.adversary_information.last_control_input]) #type: ignore

        self.adversary_information.previous_w_yr = np.roll(self.adversary_information.previous_w_yr, shift=1, axis=0)
        self.adversary_information.previous_w_yr[0] = self.adversary_information.last_yr
        # if not self.adversary_information.last_action:
        self.adversary_information.packet_generation_error = np.roll(self.adversary_information.packet_generation_error, shift=1, axis=0)
        self.adversary_information.packet_generation_error[0] = np.abs(self.adversary_information.last_yi - self.adversary_information.last_y)
        # print(f"last_y: {self.adversary_information.last_y}, last_yi: {self.adversary_information.last_yi}, last_yr: {self.adversary_information.last_yr}, packet_generation_error: {self.adversary_information.packet_generation_error[0]}")
        self.adversary_information.previous_w_u = np.roll(self.adversary_information.previous_w_u, shift=1, axis=0)
        self.adversary_information.previous_w_u[0] = self.adversary_information.last_control_input


class InjectOrListen(AdversaryObserver):
    def __init__(self, communication_channel: CommunicationChannel, packet_data_size: int, window_size: int, delay: int):
        super().__init__(communication_channel=communication_channel, packet_data_size=packet_data_size, window_size=window_size, delay=delay)
        self.name: str = "InjectOrListen"
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, packet_data_size+1))
        self.supervised_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, packet_data_size))

    def observe(self, simulator: Simulator) -> None:
        self.adversary_information.last_action = self.adversary.scheduler.previous_decision
        self.y_delay_buffer = np.roll(self.y_delay_buffer, shift=1, axis=0)

        if self.adversary_information.last_action:
            self.adversary_information.last_yi = self.adversary.packet_generator.last_packet_data #type: ignore
            self.y_delay_buffer[0] = np.zeros((self.packet_data_size))
        else:
            new_packet = self.communication_channel.listen2traffic()
            if new_packet is not None:
                self.y_delay_buffer[0] = new_packet.message
        
        self.adversary_information.last_y = self.y_delay_buffer[-1]

        if self.adversary_information.last_action:
            self.non_inject_counter = 0
        else:
            self.non_inject_counter += 1

        self.adversary_information.last_yr = self.adversary_information.last_yi if (self.adversary_information.last_action and self.non_inject_counter < self.delay) else self.adversary_information.last_y #type: ignore
        
        self.adversary_information.observations["action"] = np.roll(self.adversary_information.observations["action"], shift=1)
        self.adversary_information.observations["action"][0] = self.adversary_information.last_action

        self.adversary_information.observations["packet_data"] = np.roll(self.adversary_information.observations["packet_data"], shift=1, axis=0)
        self.adversary_information.observations["packet_data"][0] = self.adversary_information.last_yr

    def init_observations(self) -> np.ndarray:
        return np.zeros((self.window_size, self.packet_data_size))