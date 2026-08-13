import sys
import pickle
import numpy as np
from stable_baselines3 import PPO

from OTSystem.OTsystem import OTSystem, create_OTsystem, DetectorDescription
from OTSystem.OTsystem import *
from OTSystem.plant import *
from OTSystem.controller import *
from OTSystem.state_estimator import *
from OTSystem.detector import *
from OTSystem.communication import *

from Adversary.adversary import Adversary
from Adversary.adversary_observer import *
from Adversary.packet_generator import *
from Adversary.scheduler import AlwaysInject, IntelligentScheduler, RandomStep, ErrorBoundScheduler
from Adversary.policy import SupervisedRnnPolicy

class Simulator:
    def __init__(self, ot_system: OTSystem, adversary: Adversary):
        self.ot_system: OTSystem = ot_system
        self.adversary: Adversary = adversary
        self.object_directory = f"ot+adversary_systems/{self.ot_system.name}/{self.adversary.path}"
        self.figure_directory = f"figures/{self.ot_system.name}/{self.adversary.path}"
        self.training_data_directory = f"ot+adversary_systems/{self.ot_system.name}/{self.adversary.observer.__class__.__name__}"
        self.packet_generator_directory = f"ot+adversary_systems/{self.ot_system.name}/{self.adversary.packet_generator_path}"
        self.t = 0

        for directory in [self.object_directory, self.figure_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def save(self):
        # remove existing policies from the simulator object before saving
        if isinstance(self.adversary.packet_generator, IntelligentPacketGenerator):
            self.adversary.packet_generator.policy.model = None
        if isinstance(self.adversary.scheduler, IntelligentScheduler):
            self.adversary.scheduler.policy.model = None

        object_path = self.object_directory + "/simulator.pkl"

        with open(object_path, "wb") as f:
            pickle.dump(self, f)

    def load_policies(self):
        # reload policies from the saved files
        if isinstance(self.adversary.packet_generator, IntelligentPacketGenerator):
            self.adversary.packet_generator.policy.load_model(self)
        if isinstance(self.adversary.scheduler, IntelligentScheduler):
            self.adversary.scheduler.policy.load_model(self)

    def generate_training_data(self, number_of_steps: int, number_of_repetitions: int):

        y = np.zeros((number_of_repetitions, number_of_steps, self.ot_system.plant.l)) #type: ignore
        y_expected = np.zeros((number_of_repetitions, number_of_steps, self.ot_system.plant.l)) #type: ignore
        packet_data = np.zeros((number_of_repetitions, number_of_steps, self.adversary.packet_generator.packet_data_size)) #type: ignore
        previous_w_ot_info = np.zeros((number_of_repetitions, number_of_steps, *self.adversary.observer.adversary_information.previous_w_ot_info.shape)) #type: ignore
        previous_ot_info = np.zeros((number_of_repetitions, number_of_steps, *self.adversary.observer.adversary_information.previous_ot_info.shape)) #type: ignore

        for i1 in range(number_of_repetitions):
            self.reset()
            for i2 in range(number_of_steps):
                self.adversary.act(enable_attack = False) # adversary observer.observe()
                y[i1, i2] = self.ot_system.plant.y # type: ignore
                packet_data[i1, i2] = self.ot_system.plant2controller.listen2traffic().message # type: ignore
                previous_w_ot_info[i1, i2] = self.adversary.observer.adversary_information.previous_w_ot_info
                previous_ot_info[i1, i2] = self.adversary.observer.adversary_information.previous_ot_info
                self.ot_system.step() #processing incoming packets and therefore updating the expected output of the plant
                self.t += self.ot_system.plant.ts
        
        training_data = {
            "y": y,
            "previous_w_ot_info": previous_w_ot_info,
            "previous_ot_info": previous_ot_info,
            "packet_data" : packet_data,
        }

        #save training data to file at training data directory with np save
        training_data_path = self.training_data_directory + f"/training_data_{number_of_repetitions}x{number_of_steps}x{self.adversary.observer.adversary_information.previous_w_ot_info.shape}.npy"
        np.save(training_data_path, training_data) #type: ignore
        self.training_data = training_data
        print(f"Training data saved at {training_data_path}")

    def load_training_data(self, number_of_steps: int, number_of_repetitions: int):
        # try loading training data, if doesn't exist, generate it
        training_data_path = self.training_data_directory + f"/training_data_{number_of_repetitions}x{number_of_steps}x{self.adversary.observer.adversary_information.previous_w_ot_info.shape}.npy"
        try:
            self.training_data = np.load(training_data_path, allow_pickle=True).item() #type: ignore
            print(f"Training data loaded from {training_data_path}")
        except FileNotFoundError:
            print(f"Training data not found at {training_data_path}. Generating new data...")
            self.generate_training_data(number_of_steps, number_of_repetitions)

    @property
    def true_plant_state(self):
        return self.ot_system.plant.x
    
    @property
    def state_estimate(self):
        return self.ot_system.estimator.state_estimate

    def step(self, enable_attack: bool = True):
        self.adversary.act(enable_attack)
        self.ot_system.step()
        self.t += self.ot_system.plant.ts

    def reset(self, x0: np.ndarray | None = None):
        if x0 is None:
            self.ot_system.reset()
        else:
            self.ot_system.reset(x0)
        
        self.adversary.reset()
        self.t = 0

        if isinstance(self.adversary.packet_generator, IntelligentPacketGenerator) or isinstance(self.adversary.scheduler, IntelligentScheduler): 
            self.adversary.packet_generator.policy.reset() # type: ignore
            for _ in range(self.adversary.observer.window_size+self.adversary.observer.delay): # populating the observations without injecting packets
                self.step(enable_attack=False)

            if isinstance(self.adversary.packet_generator.policy, SupervisedRnnPolicy): # type: ignore
                self.adversary.packet_generator.policy.reset() # type: ignore
                self.adversary.packet_generator.policy((np.expand_dims(np.array(self.adversary.observer.adversary_information.previous_w_ot_info), axis=0))) # type: ignore
                # print(f"Initial hidden state of the RNN: {self.adversary.packet_generator.policy.h}") # type: ignore
                # exit()
                # print(f"number of detections after RNN memory reset: {self.ot_system.number_of_detections}")

    @property
    def info(self) -> dict:
        return {
            "ot_system": self.ot_system.info,
            "adversary": self.adversary.info,
        }
        
    def __str__(self):
        return str(self.info)

def create_simulator(controller_type: str, estimator_type: str, detectors_params: list[DetectorDescription] | str, adversary_observer_type: str, packet_generator_type: str, scheduler_type: str, window_size: int, delay: int, packet_generator_params: dict, scheduler_params: dict, authenticator_type: str | None) -> Simulator:

    #read simulator, if not present create it
    
    ot_system = create_OTsystem(controller_type, estimator_type, detectors_params, DetcMac_type = authenticator_type)
    packet_data_size = ot_system.plant2controller.listen2traffic().message.shape[0] # type: ignore

    if adversary_observer_type == "InjectAndListen":
        adversary_observer = InjectAndListen(communication_channel=ot_system.plant2controller, packet_data_size=packet_data_size, window_size=window_size, delay=delay)
    elif adversary_observer_type == "InjectAndListen2Channels":
        adversary_observer = InjectAndListen2Channels(communication_channel=ot_system.plant2controller, packet_data_size=packet_data_size, window_size=window_size, delay=delay)
    elif adversary_observer_type == "InjectOrListen":
        adversary_observer = InjectOrListen(communication_channel=ot_system.plant2controller, packet_data_size=packet_data_size, window_size=window_size, delay=delay)
    else:
        raise ValueError(f"Unknown adversary observer type: {adversary_observer_type}")

    if packet_generator_type == "NoisePacketGenerator":
        packet_generator = NoisePacketGenerator(packet_data_size=packet_data_size, bias=packet_generator_params["bias"], std=packet_generator_params["std"])
    elif packet_generator_type == "AdditiveNoisePacketGenerator":
        packet_generator = AdditiveNoisePacketGenerator(packet_data_size=packet_data_size, bias=packet_generator_params["bias"], std=packet_generator_params["std"])
    elif packet_generator_type == "RnnIntelligentPacketGenerator":
        packet_generator = RnnIntelligentPacketGenerator(packet_data_size=packet_data_size, adversary_observer=adversary_observer, number_of_neurons=packet_generator_params["number_of_neurons"], number_of_layers=packet_generator_params["number_of_layers"])   
    elif packet_generator_type == "NnIntelligentPacketGenerator":
        packet_generator = NNIntelligentPacketGenerator(packet_data_size=packet_data_size, adversary_observer=adversary_observer, number_of_neurons=packet_generator_params["number_of_neurons"], number_of_layers=packet_generator_params["number_of_layers"])
    elif packet_generator_type == "AsymptoticPacketGenerator":
        packet_generator = AsymptoticPacketGenerator(packet_data_size=packet_data_size, factor=packet_generator_params["factor"])
    else:
        raise ValueError(f"Unknown packet generator type: {packet_generator_type}")

    if scheduler_type == "RandomStep":
        scheduler = RandomStep(probability=scheduler_params["probability"])
    elif scheduler_type == "AlwaysInject":
        scheduler = AlwaysInject()
    elif scheduler_type == "ErrorBoundScheduler":
        scheduler = ErrorBoundScheduler(observation_space = adversary_observer.observation_space, error_bound=scheduler_params["error_bound"])
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    adversary = Adversary(observer=adversary_observer, packet_generator=packet_generator, scheduler=scheduler)

    simulator = Simulator(ot_system=ot_system, adversary=adversary)

    adversary.simulator = simulator

    return simulator

if __name__ == "__main__":
    # add OTSystem and Adversary to path
    
    sys.path.append("OTSystem")
    sys.path.append("Adversary")
    simulator = create_simulator("LQR", "KalmanEstimator", "full_auto", "InjectAndListen", "NoisePacketGenerator", "RandomStep", 3, 1, {"bias": 0.0, "std": 0.1}, {"probability": 0.5}, "EmptyAuthenticator")

    for _ in range(10):
        simulator.step()
        print("-"*80)
        print(f"t: {simulator.t:.3f}, true state: {simulator.true_plant_state}, estimated state: {simulator.state_estimate}")
        print(f"number of detected attacks: {simulator.ot_system.number_of_detections}")
        print(f"adversary observation: {simulator.adversary.observer.adversary_information.observations}")
        print(f"adversary action: {simulator.adversary.packet_generator.last_packet_data}")
        print(f"adversary scheduler decision: {simulator.adversary.scheduler.previous_decision}")
        print("-"*80)
