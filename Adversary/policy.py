from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod, ABC
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import torch
from OTSystem.detector import CumulativeResidualDetector, ResidualBasedDetector
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from torch.utils.data import TensorDataset, DataLoader

from torch import nn

if TYPE_CHECKING:
    from simulator import Simulator
    from OTSystem.detector import CumulativeResidualDetector, ResidualBasedDetector

class Policy(ABC):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box,  number_of_neurons: int, number_of_layers: int, observation_space_description: str = "", action_space_description: str = ""):
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_space_description = observation_space_description
        self.action_space_description = action_space_description
        self.set_action = None
        self.training_hyperparameters: dict
        self.model = None

        #initial values to find hyperparameters
        self.number_of_layers = number_of_layers
        self.number_of_neurons = number_of_neurons

        self.model_class: type

        self.policy_type: str

        self.parameter_ranges: dict[str, tuple[int, int]] = {
            "number_of_layers" : (1, 4),
            "number_of_neurons" : (2, 64)
        }
    
    def reset(self):
        self.set_action = None

    @property
    def name(self) -> str:
        return str(self.__class__.__name__) + str(self.model_class.__name__) + str(self.number_of_layers)+ "x" + str(self.number_of_neurons) + self.observation_space_description + "x" + self.action_space_description
    
    @property
    def info(self) -> dict:
        return {
            "policy_type": self.__class__.__name__,
            "policy_model_class": self.model_class.__name__,
            "policy_number_of_layers": self.number_of_layers,
            "policy_number_of_neurons": self.number_of_neurons,
            "policy_name": self.name,
            "observation_space": str(self.observation_space),
            "action_space": str(self.action_space),
            "number_of_layers": self.number_of_layers,
            "number_of_neurons": self.number_of_neurons,
            "training_hyperparameters": self.training_hyperparameters if hasattr(self, "training_hyperparameters") else "None"
        }

    @abstractmethod
    def train(self,
        simulator: Simulator,
        number_of_steps: int = 64,
        number_of_repetitions: int = 256*8,
        number_of_epochs: int = 128*16):

        if self.model is None:
            self.model = self.create_empty_model()

        simulator.load_training_data(number_of_steps=number_of_steps, number_of_repetitions=number_of_repetitions)

    @abstractmethod
    def create_empty_model(self):
        pass

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def load_model(self, simulator: Simulator) -> bool:
        pass


class SupervisedNnPolicy(Policy):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box,  number_of_neurons: int, number_of_layers: int, observation_space_description: str = "", action_space_description: str = ""):
        super().__init__(observation_space, action_space,  number_of_neurons, number_of_layers, observation_space_description, action_space_description)
        self.model: nn.Sequential | None = None

        self.input_dim = int(np.prod(observation_space.shape))
        self.output_dim = int(np.prod(action_space.shape))

        self.single_input_dim = observation_space.shape[1]

        self.bias_enabled = False
   
        self.model_class = nn.Sequential
        self.model = self.create_empty_model()

        self.optimizer =torch.optim.Adam(
                                    self.model.parameters(),  #type: ignore
                                    lr=2e-2,
                                )
        self.loss_fn = torch.nn.MSELoss()
        self.policy_type = "Nn"

    def create_empty_model(self):
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, self.number_of_neurons, bias = self.bias_enabled))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(self.number_of_layers - 1):
            layers.append(nn.Linear(self.number_of_neurons, self.number_of_neurons, bias = self.bias_enabled))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(self.number_of_neurons, self.output_dim, bias = self.bias_enabled))

        return self.model_class(*layers)
    
    def train(self,
        simulator: Simulator,
        number_of_steps: int = 64,
        number_of_repetitions: int = 256*8,
        number_of_epochs: int = 128*16):

        super().train(simulator, number_of_steps, number_of_repetitions, number_of_epochs)

        training_data_input = simulator.training_data["previous_w_ot_info"]
        training_data_target = simulator.training_data["packet_data"]

        # print(training_data_input[0, 0])
        # print("->")
        # print(training_data_target[0, 0])
        # exit()
        
        training_data_input = training_data_input.reshape(-1, self.input_dim) #type: ignore
        training_data_target = training_data_target.reshape(-1, self.output_dim)

        # print(training_data_input[0])
        # print("->")
        # print(training_data_target[0])

        # exit()

        x = torch.as_tensor(training_data_input, dtype=torch.float32)
        y = torch.as_tensor(training_data_target, dtype=torch.float32)

        self.model.train() # type: ignore
        previous_loss = float('inf')
        counter = 0
        for epoch in range(number_of_epochs+1):
            prediction = self.model(x) # type: ignore

            loss = self.loss_fn(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 64 == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")
            
            if loss.item() < 1e-6 or abs(previous_loss - loss.item()) < 1e-6:
                counter += 1
            else:
                counter = 0
                previous_loss = loss.item()

            if counter >= 200:
                print(f"Early stopping at epoch {epoch} due to convergence.")
                break
                    

        torch.save(self.model.state_dict(), f"{simulator.object_directory}/{self.name}.pt") #type: ignore

    def eval_prediction_horizon_1(self, simulator: Simulator, total_timesteps: int = int(1e4)):
        simulator.reset()
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet. Please load or train the model first.")

        print(f"Evaluating one step prediction error for policy {self.name} for {total_timesteps} timesteps...")

        # do one more step in order to obtain the target y with memory of the previous step
        # simulator.step(enable_attack=False)

        predictions = np.zeros((int(total_timesteps), self.output_dim))
        targets = np.zeros((int(total_timesteps), self.output_dim))
        
        # input = simulator.adversary.observer.adversary_information.observations.flatten()
        # predictions[0, :] = self.model(torch.tensor(input, dtype=torch.float32)).detach().numpy() #type: ignore

        for i in range(int(total_timesteps)):
            simulator.step(enable_attack=False)

            input = simulator.adversary.observer.adversary_information.previous_w_yr.flatten()

            target = simulator.ot_system.plant.y #simulator.adversary.observer.adversary_information.last_y.flatten()

            prediction = self.model(torch.tensor(input, dtype=torch.float32)).detach().numpy() #type: ignore

            predictions[i, :] = prediction
            targets[i, :] = target
        
        simulator.step(enable_attack=False)
        # targets[int(total_timesteps)-1, :] = simulator.adversary.observer.adversary_information.last_y.flatten()

        mse = np.mean(np.square(predictions - targets))
        print(f"Mean Squared Error (MSE) for policy {self.name}: {mse}")
     
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet. Please load or train the model first.")
        else:
            output = self.model(torch.tensor(observation.flatten(), dtype=torch.float32)).detach().numpy()
            return output
   
    def load_model(self, simulator: Simulator) -> bool:
        if self.model is None:
            self.model = self.create_empty_model()
        try:
            self.model.load_state_dict(torch.load(f"{simulator.object_directory}/{self.name}.pt"))  #type: ignore
            print(f"Loaded existing policy model from {simulator.object_directory}/{self.name}.pt")
            return True
        except:
            print(f"No existing policy model found at {simulator.object_directory}/{self.name}.pt. Please train the model first.")
            return False


class SupervisedRnnPolicy(Policy):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, number_of_neurons: int, number_of_layers: int, observation_space_description: str = "", action_space_description: str = ""):
        super().__init__(observation_space, action_space, number_of_neurons, number_of_layers, observation_space_description, action_space_description)
        self.model: nn.GRU | nn.RNN | None = None

        self.input_dim = int(observation_space.shape[1])
        self.output_dim = int(np.prod(action_space.shape))

        self.model_class = nn.RNN
        self.model = self.create_empty_model()
        
        self.fc = nn.Linear(self.number_of_neurons, self.output_dim, bias = False)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()), # type: ignore
            lr=9e-3,
        )

        self.loss_fn = torch.nn.MSELoss()
        self.policy_type = "Rnn"

    def create_empty_model(self): #  -> nn.GRU | nn.RNN
        self.h = None #torch.zeros(self.number_of_layers, 1, self.number_of_neurons)  # Initial hidden state
        return self.model_class(
            input_size=self.input_dim,
            hidden_size=self.number_of_neurons,
            num_layers=self.number_of_layers,
            batch_first=True,
            bias = False,
            nonlinearity="tanh"
        )
    
    def reset(self):
        self.set_action = None
        self.h = None # torch.zeros(self.number_of_layers, 1, self.number_of_neurons)  # Reset hidden state

    def forward(self, x):
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet. Please load or train the model first.")
        else:
            gru_out, self.h = self.model(x, self.h)
            return self.fc(gru_out[:, -1, :]) # gru + linear pass
        
    def training_forward(self, x):
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet. Please load or train the model first.")
        else:
            gru_out, _ = self.model(x)

            # print("gru_out:")
            # print("  requires_grad:", gru_out.requires_grad)
            # print("  mean:", gru_out.mean().item())
            # print("  grad_fn:", gru_out.grad_fn)

            prediction = self.fc(gru_out)

            # print("prediction:")
            # print("  requires_grad:", prediction.requires_grad)
            # print("  mean:", prediction.mean().item())
            # print("  grad_fn:", prediction.grad_fn)
            # exit()
            return prediction
    
    def train(
        self,
        simulator: Simulator,
        number_of_steps: int = 64,
        number_of_repetitions: int = 256*8,
        number_of_epochs: int = 128*16):

        super().train(simulator, number_of_steps, number_of_repetitions, number_of_epochs)

        training_data_input = simulator.training_data["previous_ot_info"] #.reshape(number_of_repetitions, number_of_steps, self.input_dim) #type: ignore
        training_data_target = simulator.training_data["packet_data"]

        x = torch.as_tensor(training_data_input, dtype=torch.float32)
        y = torch.tensor(training_data_target, dtype=torch.float32)

        self.model.train() # type: ignore
        previous_loss = float('inf')
        for epoch in range(number_of_epochs+1):

            # RNN forward pass
            rnn_output, _ = self.model(x) # type: ignore

            # Map hidden states -> actions
            prediction = self.fc(rnn_output)

            # Compare every timestep
            loss = self.loss_fn(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 64 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.6f}")

            if loss.item() < 1e-6 or abs(previous_loss - loss.item()) < 1e-6:
                print(f"Early stopping at epoch {epoch} due to convergence.")
                break
            else:
                previous_loss = loss.item()

        torch.save(
            {
                "gru": self.model.state_dict(),# type: ignore
                "fc": self.fc.state_dict(),
            },
            f"{simulator.object_directory}/{self.name}.pt",
        )

        print("Saved trained policy model to", f"{simulator.object_directory}/{self.name}.pt")
        self.model = self.create_empty_model()  # Reset the model after training
        self.load_model(simulator)  # Load the trained model for evaluation
        exit()
        
    def __call__(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet. Please load or train the model first.")
        elif observation.ndim != 3:
            raise ValueError(f"Expected observation to have 3 dimensions (batch size x sequence length x input dimensions), but got {observation.ndim} dimensions.")
        else:
            # print("batch size x sequence length x input dimensions:", observation.shape)
            if isinstance(observation, np.ndarray):
                observation_torch = torch.tensor(observation, dtype=torch.float32) #type: ignore
            # print(self.forward(observation).detach().numpy())
                # print(observation_torch)
            # exit()
                return self.forward(observation_torch).detach().numpy().flatten()
            else:
                return self.forward(observation).detach().numpy().flatten()
                
    def load_model(self, simulator: Simulator) -> bool:
        if self.model is None:
            self.model = self.create_empty_model()

        path = f"{simulator.object_directory}/{self.name}.pt"

        try:
            checkpoint = torch.load(path, map_location="cpu")

            self.model.load_state_dict(checkpoint["gru"]) # type: ignore
            self.fc.load_state_dict(checkpoint["fc"])

            print(f"Loaded existing policy model from {path}")
            return True

        except Exception as e:
            print(f"Failed to load policy model from {path}: {e}")
            return False

# class RLPolicy(Policy):
#     def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, observation_space_description: str = "", action_space_description: str = ""):
#         super().__init__(observation_space, action_space, observation_space_description, action_space_description)

#         self.training_hyperparameters = {'learning_rate': 3e-4, 'gamma': 0.99, 'batch_size': 128, 'n_steps': 512, 'n_epochs': 2000, 'ent_coef': 0.005}

#         #initial values to find hyperparameters
#         self.number_of_layers = 1
#         self.number_of_neurons = 256

#         self.model_class = PPO

#         self.model: PPO | RecurrentPPO | None = None

#     def __call__(self, observation: np.ndarray) -> np.ndarray:
#         if self.set_action is not None:
#             return self.set_action
#         else:
#             if self.model is not None:
#                 action, _ = self.model.predict(observation, deterministic=True)
#                 return action
#             else:
#                 raise ValueError("Model is not loaded or trained yet. Please load or train the model first.")

#     def load_model(self, simulator: Simulator):
#         try:
#             self.model = self.model_class.load(f"{simulator.object_directory}/{self.name}.zip")
#             print(f"Loaded existing policy model from {simulator.object_directory}/{self.name}.zip")
#         except:
#             print(f"No existing policy model found at {simulator.object_directory}/{self.name}.zip. Please train the model first.")
    
#     def train(self, simulator: Simulator, total_timesteps: int = 10000, reset_model: bool = False):
        
#         env = RLPolicyEnvironment(policy=self, simulator=simulator, T = 2)
        
#         if self.model is None:
#             try:
#                 self.model = self.model_class.load(f"{simulator.object_directory}/{self.name}.zip")
#                 print(f"Loaded existing policy model from {simulator.object_directory}/{self.name}.zip")
#                 print(self.model)
#             except:
#                 print(f"No existing policy model found at {simulator.object_directory}/{self.name}.zip. Creating a new model.")
#                 policy_kwargs = {
#                                     "net_arch": {
#                                         "pi": [self.number_of_neurons] * self.number_of_layers,
#                                         "vf": [self.number_of_neurons] * self.number_of_layers,
#                                     }
#                                 }
                
#                 self.model = self.model_class(
#                     "MlpLstmPolicy" if self.model_class == RecurrentPPO else "MlpPolicy",
#                     env,
#                     verbose=1,
#                     policy_kwargs=policy_kwargs,
#                     **self.training_hyperparameters,
#                 )

#         if self.model is not None:
#             self.model.set_env(env) #type: ignore
            
#             self.model.learn(total_timesteps=total_timesteps)

#             self.set_action = None
#             env.close()

#             # save the model to a file
#             self.model.save(f"{simulator.object_directory}/{self.name}.zip")
#             print(f"Policy model saved to {simulator.object_directory}/{self.name}.zip")
#         else:
#             raise ValueError("Model is not initialized. Please initialize the model first.")


# class RewardCallback(BaseCallback):
#     def __init__(self):
#         super().__init__()
#         self.episode_rewards = []

#     def _on_step(self) -> bool:

#         for info in self.locals["infos"]:
#             if "episode" in info:
#                 self.episode_rewards.append(info["episode"]["r"])

#         return True


# #training environment for the reinforcement learning policy
# class RLPolicyEnvironment(gym.Env):
#     def __init__(self, policy: Policy, simulator: Simulator, T:float):
#         self.simulator: Simulator = simulator
#         self.policy = policy
#         self.observation_space = self.policy.observation_space
#         self.action_space = self.policy.action_space
#         self.T = T
#         self.reward_weights = np.array([0,1])

#     def reset(self, seed: int | None = None, options: dict[str, Any] | None = None): # type: ignore
#         # print information on the system in it's terminal state
        
#         self.simulator.reset()

#         info = {"number of detected attacks": self.simulator.ot_system.number_of_detections, "elapsed time": self.simulator.t, "alpha (deg)": self.simulator.ot_system.plant.x[3]*180/np.pi}

#         observation = self.simulator.adversary.observer.adversary_information.observations
#         return observation, info
    
#     def step(self, action):

#         # print("="*100)
#         # print(f"elapsed time: {self.simulator.t}, alpha (deg): {self.simulator.ot_system.plant.x[3]*180/np.pi}, alpha hat(deg): {self.simulator.ot_system.estimator.state_estimate[3]*180/np.pi}, detected: {self.simulator.ot_system.number_of_detections > 0}")
#         # [print(detector.name, detector.detection_metric_value, detector.bound) for detector in self.simulator.ot_system.detectors]
#         # print("-"*100)
#         # print(f"action: {action}")
        
    
#         # self.policy.set_action = action

#         transition_matrix = np.array(action)
        
#         # generated_packet_data_difference = action * 0.002

#         true_output = self.simulator.ot_system.plant.y

#         previous_yr = self.simulator.adversary.observer.adversary_information.last_yr

#         generated_packet_data = transition_matrix @ previous_yr # + generated_packet_data_difference

#         self.policy.set_action = generated_packet_data # type: ignore

#         self.simulator.step(enable_attack=True)
        
#         observation = self.simulator.adversary.observer.adversary_information.observations

#         # print(f"expected_measurement_data: {self.simulator.ot_system.detectors[2].predicted_output}")# type: ignore
#         # print(f"last_packet_data: {generated_packet_data}")

#         # print(f"observation: {observation}")

#         undetected_reward = ((self.simulator.ot_system.plant.unstable_state - self.simulator.ot_system.estimator.state_estimate[self.simulator.ot_system.plant.unstable_state_index])*180/np.pi)**2
        
        
#         # print("-"*100)
#         # print(f"elapsed time: {self.simulator.t}, alpha (deg): {self.simulator.ot_system.plant.x[3]*180/np.pi}, alpha hat(deg): {self.simulator.ot_system.estimator.state_estimate[3]*180/np.pi}, detected: {self.simulator.ot_system.number_of_detections > 0}")
#         # [print(detector.name, detector.detection_metric_value, detector.bound) for detector in self.simulator.ot_system.detectors]
#         # print("="*100)

#         detected_reward = 0
#         reward_I = -np.sum(np.square(np.array((true_output - generated_packet_data)))) # type: ignore
        
#         detected = self.simulator.ot_system.number_of_detections > 0
#         if detected:
#             detected_reward = reward_I # - np.sum(np.square(np.array([detector.detection_metric_value-detector.bound for detector in self.simulator.ot_system.detectors if  detector.detection_metric_value > detector.bound]))) # type: ignore
        
#         reward = detected_reward if detected else undetected_reward

#         truncated = self.simulator.t >= self.T

#         terminated = self.simulator.ot_system.number_of_detections > 0

#         info = {"number of detected attacks": self.simulator.ot_system.number_of_detections, "elapsed time": self.simulator.t, "alpha (deg)": self.simulator.ot_system.plant.x[3]*180/np.pi}

#         return (
#             observation,
#             reward_I,
#             terminated,
#             truncated,
#             info
#         )
