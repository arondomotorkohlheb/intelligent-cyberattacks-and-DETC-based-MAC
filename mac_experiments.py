from OTSystem.OTsystem import *
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

def generate_mac_data(type: str, number_of_steps = int(1e5)):
    if type == "float32":
        ot1 = load_ot1_float32detc_mac()
    elif type == "int8":
        ot1 = load_ot1_int8detc_mac()
    else:
        raise ValueError(f"Unsupported data type: {type}")
    ot1.reset()
    input_data = np.zeros((number_of_steps, 2), dtype=np.float32)
    target_data = np.zeros((number_of_steps, 1), dtype=np.float32)
    for i in range(number_of_steps):
        packet = ot1.plant2controller.listen2traffic()
        if packet is not None:
            input_data[i] = packet.message
            target_data[i] = packet.tag
        else:
            raise ValueError("No packet transmitted from plant to controller")
        ot1.step()

    np.save(f"mac_experiments/mac{type}input_data_{number_of_steps}.npy", input_data)
    np.save(f"mac_experiments/mac{type}target_data_{number_of_steps}.npy", target_data)

    return input_data, target_data

class RnnMacApproximatorfloat32:
    def __init__(self, number_of_neurons: int, number_of_layers: int, model_class: str, detc_dtype: str):
        self.input_dim = 2
        self.output_dim = 1
        self.number_of_neurons = number_of_neurons
        self.number_of_layers = number_of_layers
        self.model_class_name = model_class
        self.detc_dtype = detc_dtype

        if model_class == "nn.RNN":
            self.model_class = nn.RNN
            rnn_kwargs = {"nonlinearity": "tanh"}
        elif model_class == "nn.LSTM":
            self.model_class = nn.LSTM
            rnn_kwargs = {}
        elif model_class == "nn.GRU":
            self.model_class = nn.GRU
            rnn_kwargs = {}
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        self.model = self.model_class(
            input_size=self.input_dim,
            hidden_size=self.number_of_neurons,
            num_layers=self.number_of_layers,
            batch_first=True,
                **rnn_kwargs,
        )

        self.fc = nn.Linear(
            self.number_of_neurons,
            self.output_dim,
            bias=False,
        )

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr=9e-3,
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, hidden=None):
        """
        x:
            (batch, sequence_length, input_dim)

        hidden:
            (num_layers, batch, hidden_size)

        Returns:
            prediction:
                (batch, sequence_length, output_dim)

            hidden:
                (num_layers, batch, hidden_size)
        """

        rnn_output, hidden = self.model(x, hidden)

        # Apply the linear layer to every timestep
        prediction = self.fc(rnn_output)

        return prediction, hidden

    def train(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        sequence_length: int = 32,
    ):

        # -------------------------------------------------
        # Convert data to tensors
        # -------------------------------------------------

        x = torch.as_tensor(
            input_data,
            dtype=torch.float32,
        )

        y = torch.as_tensor(
            target_data,
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Make sure dimensions are correct
        # -------------------------------------------------

        # x should be:
        # (time, input_dim)
        #
        # y should be:
        # (time, output_dim)

        if x.ndim != 2:
            raise ValueError(
                f"input_data must have shape "
                f"(time, {self.input_dim}), got {x.shape}"
            )

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if y.ndim != 2:
            raise ValueError(
                f"target_data must have shape "
                f"(time, {self.output_dim}), got {y.shape}"
            )

        if len(x) != len(y):
            raise ValueError(
                "input_data and target_data must have "
                "the same number of timesteps."
            )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        self.model.train()
        self.fc.train()

        hidden = None
        epoch_loss = 0.0
        number_of_updates = 0

        # IMPORTANT:
        # Do NOT shuffle the sequence.
        #
        # Each window follows directly after the previous
        # window in time.
        for start in range(
            0,
            len(x) - sequence_length + 1,
            sequence_length,
        ):

            end = start + sequence_length

            # -----------------------------------------
            # Get TBPTT window
            # -----------------------------------------

            x_window = x[start:end]
            y_window = y[start:end]

            # Add batch dimension
            #
            # (sequence_length, input_dim)
            #       ↓
            # (1, sequence_length, input_dim)

            x_window = x_window.unsqueeze(0)
            y_window = y_window.unsqueeze(0)

            # -----------------------------------------
            # Forward pass
            # -----------------------------------------

            prediction, hidden = self.forward(
                x_window,
                hidden,
            )

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = self.loss_fn(
                prediction,
                y_window,
            )

            # -----------------------------------------
            # Backpropagation
            # -----------------------------------------

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            # -----------------------------------------
            # Detach hidden state
            # -----------------------------------------
            #
            # Keep the numerical hidden state, but
            # remove its connection to the previous
            # computation graph.
            #
            # This is the key part of TBPTT.

            if isinstance(hidden, tuple):
                hidden = tuple(state.detach() for state in hidden)
            else:
                hidden = hidden.detach()

            # -----------------------------------------
            # Logging
            # -----------------------------------------

            epoch_loss += loss.item()
            number_of_updates += 1

        epoch_loss /= number_of_updates

        print(
            f"Loss: {epoch_loss:.6f}"
        )

        # save the model
        torch.save(self.model.state_dict(), f"mac_experiments\\trained_models\\rnn_mac_approximator_model_{self.detc_dtype}_{self.model_class_name}_{self.number_of_neurons}n_{self.number_of_layers}l.pth")

    def load_model(self) -> bool:
        fname = f"mac_experiments\\trained_models\\rnn_mac_approximator_model_{self.detc_dtype}_{self.model_class_name}_{self.number_of_neurons}n_{self.number_of_layers}l.pth"
        try:
            self.model.load_state_dict(torch.load(fname))
            print(f"Loaded model from {fname}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {fname}")
            return False

    def evaluate(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
    ) -> tuple[np.ndarray, float]:

        # -------------------------------------------------
        # Convert data to tensors
        # -------------------------------------------------

        x = torch.as_tensor(
            input_data,
            dtype=torch.float32,
        )

        y = torch.as_tensor(
            target_data,
            dtype=torch.float32,
        )

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # -------------------------------------------------
        # Evaluation mode
        # -------------------------------------------------

        self.model.eval()
        self.fc.eval()

        hidden = None

        predictions = []

        # -------------------------------------------------
        # Process the unseen data sequentially
        # -------------------------------------------------

        with torch.no_grad():

            # One timestep at a time.
            #
            # This mimics online operation:
            #
            # observation -> prediction -> next observation

            for t in range(len(x)):

                # Shape:
                # (input_dim,)
                #
                # ->
                #
                # (1, 1, input_dim)
                x_t = x[t].unsqueeze(0).unsqueeze(0)

                prediction, hidden = self.forward(
                    x_t,
                    hidden,
                )

                # prediction:
                # (1, 1, output_dim)
                prediction = prediction.squeeze(0).squeeze(0)

                predictions.append(
                    prediction.cpu()
                )

        # -------------------------------------------------
        # Stack predictions
        # -------------------------------------------------

        predictions = torch.stack(predictions)

        # -------------------------------------------------
        # Calculate test loss
        # -------------------------------------------------

        loss = self.loss_fn(
            predictions,
            y,
        )

        print(
            f"Evaluation loss: {loss.item():.6f}"
        )

        return (
            predictions.numpy(),
            loss.item(),
        )

def experiment(datatype: str):
    try:
        input_data = np.load(f"mac_experiments\\mac{datatype}_input_data_100000.npy")
        target_data = np.load(f"mac_experiments\\mac{datatype}_target_data_100000.npy")
    except FileNotFoundError:
        input_data, target_data = generate_mac_data(datatype, number_of_steps=int(1e5))

    training_input_data = input_data[:80000]
    training_target_data = target_data[:80000]

    test_input_data = input_data[80000:]
    test_target_data = target_data[80000:]

    numbers_of_neurons = [1,2, 4, 8, 16, 32, 64, 128, 256]
    numbers_of_layers = [1, 2, 4, 8, 16]

    model_classes = ["nn.RNN", "nn.LSTM", "nn.GRU"]

    success_ratio_results = np.zeros((len(model_classes), len(numbers_of_neurons), len(numbers_of_layers)), dtype=np.float32)
    average_undetected_sequence_length_results = np.zeros((len(model_classes), len(numbers_of_neurons), len(numbers_of_layers)), dtype=np.float32)

    for model_class in model_classes:
        for number_of_neurons in numbers_of_neurons:
            for number_of_layers in numbers_of_layers:
                print(f"Model {model_class} with {number_of_neurons} neurons and {number_of_layers} layers")
                
                rnn_mac_approximator = RnnMacApproximatorfloat32(
                    number_of_neurons=number_of_neurons,
                    number_of_layers=number_of_layers,
                    model_class=model_class,
                    detc_dtype=datatype
                )

                if not rnn_mac_approximator.load_model():
                    print("Training model...")

                    rnn_mac_approximator.train(
                        training_input_data,
                        training_target_data,
                        sequence_length=32,
                    )
                else:
                    print("Model loaded")

                predictions, loss = rnn_mac_approximator.evaluate(
                    test_input_data,
                    test_target_data,
                )

                correctly_predicted = np.zeros(test_target_data.shape[0], dtype=bool)
                for i in range(test_target_data.shape[0]):
                    correctly_predicted[i] = int(predictions[i,0]) == int(test_target_data[i,0])

                changes = np.diff(
                    np.concatenate(([False], correctly_predicted, [False])).astype(int)
                )

                lengths = (
                    np.where(changes == -1)[0]
                    - np.where(changes == 1)[0]
                )

                average_length = np.mean(lengths) if len(lengths) > 0 else 0
                
                print(f"Correctly predicted {np.sum(correctly_predicted)} out of {len(test_target_data)} meaning {np.sum(correctly_predicted)/len(test_target_data)*100:.2f}% accuracy")
                print(f"Average undetected sequence length: {average_length:.2f} timesteps")
                success_ratio_results[model_classes.index(model_class), numbers_of_neurons.index(number_of_neurons), numbers_of_layers.index(number_of_layers)] = np.sum(correctly_predicted)/len(test_target_data)
                average_undetected_sequence_length_results[model_classes.index(model_class), numbers_of_neurons.index(number_of_neurons), numbers_of_layers.index(number_of_layers)] = average_length
        
    np.savez(
        f"mac_experiments/rnn_mac_approximator_results_{datatype}_{model_classes}x{numbers_of_neurons}x{numbers_of_layers}.npz",
        success_ratio_results=success_ratio_results,
        average_undetected_sequence_length_results=average_undetected_sequence_length_results,
        numbers_of_neurons=numbers_of_neurons,
        numbers_of_layers=numbers_of_layers,
        model_classes=model_classes,
    )

if __name__ == "__main__":
    experiment("float32")
    experiment("int8")