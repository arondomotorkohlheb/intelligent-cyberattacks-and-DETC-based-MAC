from __future__ import annotations
from typing import TYPE_CHECKING

import sys
sys.path.append("..")

from matplotlib import scale
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd

from Adversary.packet_generator import *
from Adversary.scheduler import *
from Adversary.adversary_observer import *
from Adversary.policy import Policy
from OTSystem.communication import CommunicationChannel, Packet

if TYPE_CHECKING:
    from simulator import Simulator


from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy.spatial import Delaunay


def plot_heatmap(
    parameter_values: dict[str, list[float]],
    evaluation_results: list[float],
    repetitions: int,
    save_path: str,
    scale = (0, 10),
) -> None:
    """
    Plot an interpolated heatmap from sampled two-parameter evaluation results.

    If mask_outside_hull is True, only the region inside the convex hull of
    sampled points is displayed.
    """

    parameter_names = list(parameter_values.keys())

    parameter_1_name, parameter_2_name = parameter_names

    x = np.asarray(parameter_values[parameter_1_name], dtype=float)
    y = np.asarray(parameter_values[parameter_2_name], dtype=float)
    z = np.asarray(evaluation_results, dtype=float)*180/np.pi # plot is in degrees
    
    points = np.column_stack((x, y))

    resolution = 200

    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)

    grid_z = griddata(
        points=points,
        values=z,
        xi=(grid_x, grid_y),
        method="linear",
    )

    # Determine which grid points lie inside the sampled region
    hull = Delaunay(points)

    grid_points = np.column_stack(
        (
            grid_x.ravel(),
            grid_y.ravel(),
        )
    )

    inside = hull.find_simplex(grid_points) >= 0
    inside = inside.reshape(grid_x.shape)

    # Hide unsupported regions
    grid_z[~inside] = np.nan
    

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3 , 4))

    plt.imshow(
        grid_z,
        extent=[
            x.min(), #type: ignore
            x.max(), #type: ignore
            y.min(), #type: ignore
            y.max(), #type: ignore
        ],
        origin="lower",
        aspect="auto",
        cmap = "Blues",
    )

    colorbar = plt.colorbar(label=r"$\bar{\Delta \alpha}$ (degrees)", orientation="vertical")
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+"))
    )
    colorbar.update_ticks()

    plt.scatter(
        x,
        y,
        s=5,
        color="white",
    )

    plt.xlabel(parameter_1_name)
    plt.ylabel(parameter_2_name)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

class Adversary:
    def __init__(self, observer: AdversaryObserver, packet_generator: PacketGenerator, scheduler: Scheduler):
        self.observer: AdversaryObserver = observer
        self.packet_generator: PacketGenerator = packet_generator
        self.scheduler: Scheduler = scheduler
        self.observer.adversary = self
        self.simulator: Simulator
        self.evaluation: np.ndarray

    def reset(self):
        self.observer.reset()
        self.packet_generator.reset()
        self.scheduler.previous_decision = False
        

    def act(self, enable_attack: bool = True):
        self.observer.observe(self.simulator)
        if enable_attack and self.scheduler.schedule(self.observer.adversary_information):
            self.observer.communication_channel.replace_packet(self.packet_generator.generate_packet(self.observer.adversary_information))
        else:
            self.packet_generator.generate_packet(self.observer.adversary_information)
        
        # if enable_attack:
            # print(":"*100)
            # print("adversary information:")
            # print(self.observer.adversary_information.previous_ot_info)
            # print(self.observer.adversary_information.previous_w_ot_info)

            # print("last packet data:")
            # print(self.packet_generator.last_packet_data)
            # print("simulator plant output:")
            # print(self.simulator.ot_system.plant.y)
            # print("detector 2 predicted output:")
            # print(self.simulator.ot_system.detectors[2].predicted_output) #type: ignore
            # print(":"*100)
        

    @property
    def name(self):
        return self.observer.name + "_" + self.packet_generator.name + "_" + self.scheduler.name
    
    @property
    def path(self):
        return self.observer.name + "/" + self.packet_generator.name + "/" + self.scheduler.name

    @property
    def packet_generator_path(self):
        return self.observer.name + "/" + self.packet_generator.name

    @property
    def info(self) -> dict:
        return {
            "adversary_name": self.name,
            "observer": self.observer.info,
            "packet_generator": self.packet_generator.info,
            "scheduler": self.scheduler.info
        }
    
    def tune_packet_generator(self, n_trials: int = 100, repetitions: int = 100):
        scheduler_backup: Scheduler = self.scheduler
        self.scheduler = AlwaysInject()
        self.tune_component(self.packet_generator, n_trials=n_trials, repetitions=repetitions)
        self.scheduler = scheduler_backup

    def tune_scheduler(self, n_trials: int = 100, repetitions: int = 100):
        if isinstance(self.scheduler, IntelligentScheduler):
            self.tune_component(self.scheduler, n_trials=n_trials, repetitions=repetitions)
        else:
            print("not tuneable Scheduler")

    def tune_component(self, component : PacketGenerator | IntelligentScheduler, n_trials: int = 100, repetitions: int = 100):
        print(f"Tuning {component.name} for {n_trials} trials and {repetitions} repetitions...")
        def objective(trial: optuna.Trial):
            for parameter_name, parameter_range in component.parameter_ranges.items():
                if isinstance(parameter_range[0], float):
                    value = trial.suggest_float(parameter_name, parameter_range[0], parameter_range[1])
                elif isinstance(parameter_range[0], int):
                    value = trial.suggest_int(parameter_name, parameter_range[0], parameter_range[1]) # type: ignore
                else:
                    raise ValueError(f"Unsupported parameter type for {parameter_name}: {type(parameter_range[0])}")
                
                setattr(component, parameter_name, value)
            
            self.simulator.reset()

            if isinstance(component, (IntelligentPacketGenerator, IntelligentScheduler)):
                component.train(self.simulator)

            evaluation_results = self.evaluate(repetitions=repetitions)
            return float(np.mean(evaluation_results["estimation_errors"]))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, )
        # create a heat map with the checked parameters and the corresponding evaluation results


        # Set the best parameters found
        for parameter_name in component.parameter_ranges.keys():
            setattr(component, parameter_name, study.best_params[parameter_name])
        
        #based on the intermediate results, create a heatmap of the parameter space and the corresponding evaluation results

        parameter_history = {parameter_name: [] for parameter_name in component.parameter_ranges.keys()}
        [parameter_history[parameter_name].append(trial.params[parameter_name]) for trial in study.trials for parameter_name in component.parameter_ranges.keys()]

        plot_heatmap(parameter_history, [trial.value for trial in study.trials], repetitions, self.simulator.figure_directory+ "/" + component.name + "_tuning_heatmap.png") #type: ignore

    def evaluate(self, repetitions: int = 100) -> dict[str, np.ndarray]:
        print(f"Evaluating adversary policy {self.name} for {repetitions} repetitions...")
        results_x0 = []
        results_estimation_error = []
        detection_occurrancies = 0
        undetected_timesteps = []
        terminal_states = []
        success_occurrancies = 0
        injection_start_steps = []

        for _ in range(repetitions):
            self.simulator.reset()
            x0 = self.simulator.ot_system.plant.x.copy()
            results_x0.append(x0)
            i = 0
            attack_enabled = True
            injection_start_index = -1
            while i < int(1e5):                
                self.simulator.step(attack_enabled)
                if self.scheduler.previous_decision and injection_start_index == -1:
                    injection_start_index = i
                    injection_start_steps.append(injection_start_index)

                if self.simulator.ot_system.unsafe_state_space.contains(self.simulator.ot_system.plant.x):
                    terminal_states.append(self.simulator.ot_system.plant.x.copy())
                    undetected_timesteps.append(i-injection_start_index)
                    success_occurrancies += 1
                    break

                elif self.simulator.ot_system.number_of_detections > 0 and attack_enabled:
                    undetected_timesteps.append(i-injection_start_index)
                    results_estimation_error.append(np.abs((self.simulator.ot_system.estimator.state_estimate - self.simulator.ot_system.plant.x)[self.simulator.ot_system.plant.unstable_state_index]))
                    attack_enabled = False
                    detection_occurrancies += 1

                elif (self.simulator.ot_system.safe_state_space.contains(self.simulator.ot_system.plant.x) and not attack_enabled) or i >= int(1e5)-1:
                    if self.simulator.ot_system.number_of_detections == 0:
                        undetected_timesteps.append(i-injection_start_index)
                        results_estimation_error.append(np.abs((self.simulator.ot_system.estimator.state_estimate - self.simulator.ot_system.plant.x)[self.simulator.ot_system.plant.unstable_state_index]))
                    terminal_states.append(self.simulator.ot_system.plant.x.copy())
                    break
                i += 1

        evaluation_results = {
            "initial_states": np.array(results_x0),
            "estimation_errors": np.array(results_estimation_error),
            "undetected_timesteps": np.array(undetected_timesteps),
            "terminal_states": np.array(terminal_states),
            "success_rate": success_occurrancies/repetitions,
            "injection_start_steps": np.array(injection_start_steps),
            "detection_rate": detection_occurrancies/repetitions,
        }

        return evaluation_results

    def __str__(self):
        return str(self.info)

if __name__ == "__main__":
    pass