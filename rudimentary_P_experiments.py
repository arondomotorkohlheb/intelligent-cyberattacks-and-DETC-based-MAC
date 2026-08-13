from pprint import pprint

from OTSystem import authenticator
from simulator import *

from OTSystem.OTsystem import *
from OTSystem.plant import *
from OTSystem.controller import *
from OTSystem.state_estimator import *
from OTSystem.detector import *
from OTSystem.communication import *

from Adversary.adversary import *
from Adversary.adversary_observer import *
from Adversary.packet_generator import *
from Adversary.scheduler import *
from Adversary.policy import *

from plotting_support import *

from factory import generate_simulators
from itertools import product as cartesian_product


adversary_delay = 1

adversary_observer = "InjectAndListen" # 2Channels


packet_generators: list[tuple[str, dict]] = [
    ("NoisePacketGenerator", {"bias": 0.0, "std": 0.01}),
    ("AdditiveNoisePacketGenerator", {"bias": 0.0, "std": 0.01}),
    ("AsymptoticPacketGenerator", {"factor": 0.9, "std": 0.01}),
]

scheduler: tuple[str, dict] = ("AlwaysInject", {})

controllers: list[str] = ["MPC", "LQR"]
estimators: list[str] = ["KalmanEstimator", "Observer"]
detector_params = "full_auto"

combinations = cartesian_product(
    packet_generators,
    controllers,
    estimators)

detection_rate = np.zeros((len(packet_generators), len(controllers), len(estimators)))
success_rate = np.zeros((len(packet_generators), len(controllers), len(estimators)))
estimated_alpha_differnce = np.zeros((len(packet_generators), len(controllers), len(estimators)))
detection_time = np.zeros((len(packet_generators), len(controllers), len(estimators)))

estimated_alpha_variance = np.zeros((len(packet_generators), len(controllers), len(estimators)))
detection_time_variance = np.zeros((len(packet_generators), len(controllers), len(estimators)))

ot_with_adversary_simulators = []
for combination in combinations:
        pg_name, pg_params = combination[0][0], combination[0][1]
        controller = combination[1]
        estimator = combination[2]

        simulator = create_simulator(controller, estimator, detector_params, adversary_observer, pg_name, scheduler[0], 2, adversary_delay, pg_params, scheduler[1], authenticator_type=None)
        simulator.adversary.tune_packet_generator(n_trials = 10, repetitions = 10)

        eval_dict = simulator.adversary.evaluate(repetitions=1000)

        detection_rate[packet_generators.index(combination[0]), controllers.index(controller), estimators.index(estimator)] = eval_dict["detection_rate"]
        success_rate[packet_generators.index(combination[0]), controllers.index(controller), estimators.index(estimator)] = eval_dict["success_rate"]
        estimated_alpha_differnce[packet_generators.index(combination[0]), controllers.index(controller), estimators.index(estimator)] = np.mean(eval_dict["estimation_errors"])
        detection_time[packet_generators.index(combination[0]), controllers.index(controller), estimators.index(estimator)] = np.mean(eval_dict["undetected_timesteps"])

        estimated_alpha_variance[packet_generators.index(combination[0]), controllers.index(controller), estimators.index(estimator)] = np.var(eval_dict["estimation_errors"])
        detection_time_variance[packet_generators.index(combination[0]), controllers.index(controller), estimators.index(estimator)] = np.var(eval_dict["undetected_timesteps"])




saving_dict = {
    "detection_rate": detection_rate,
    "success_rate": success_rate,
    "estimated_alpha_difference": estimated_alpha_differnce,
    "detection_time": detection_time,
    "estimated_alpha_variance": estimated_alpha_variance,
    "detection_time_variance": detection_time_variance,
    "packet_generators": packet_generators,
    "controllers": controllers,
    "estimators": estimators}

np.savez("ot+adversary_systems/rudimentary_P_experiments_results.npz", **saving_dict)

import numpy as np
import matplotlib.pyplot as plt
evaluation_metrics = [
    "success_rate",
    "detection_rate",
    "detection_time",
    "estimated_alpha_difference",
]

evaluation_metric_names = [
    r"$\bar{s}$",
    r"$\bar{\delta}$",
    r"$\bar{T}_{\delta}$",
    r"$\overline{\Delta\alpha}$",
]

packet_generator_names = [
    r"$\mathcal{P}^{g}$",
    r"$\mathcal{P}^{+g}$",
    r"$\mathcal{P}^{as}$",
]

controller_names = [
    r"$\mathcal{C}^{MPC}$",
    r"$\mathcal{C}^{LQR}$",
]

estimator_names = [
    r"$\mathcal{E}^{\mathcal{K}}$",
    r"$\mathcal{E}^{\mathcal{L}}$",
]


# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------
fig, axes = plt.subplots(
    4,
    3,
    figsize=(6, 6),
    sharex="col",
    sharey="row",
    gridspec_kw={
        "wspace": 0,
        "hspace": 0,
    },
)

images = np.empty((4, 3), dtype=object)

cmap = plt.cm.Blues


# ---------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------

for i, metric in enumerate(evaluation_metrics):
    for j, pg in enumerate(packet_generator_names):

        ax = axes[i, j]

        data = saving_dict[metric][j]

        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            vmin=0,
        )

        images[i, j] = im

        ax.set_xticks([])
        ax.set_yticks([])

        # -------------------------------------------------
        # Metric name on LEFT
        # -------------------------------------------------

        # Only add it to the first column so it doesn't
        # appear three times per row.
        if j == 0:
            ax.set_ylabel(
                evaluation_metric_names[i],
                fontsize=13,
                labelpad=12,
            )

# i want x ticks only in the lowest row and y ticks only in the leftmost column
# first: setting last row ticks
for j in range(3):
    ax = axes[3, j]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(controller_names, fontsize=12)

# second: setting first column ticks
for i in range(4):
    ax = axes[i, 0]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(estimator_names, fontsize=12)
    # should point to the left, not to the right
    ax.yaxis.tick_left()

# ---------------------------------------------------------
# Column names: packet generators
# ---------------------------------------------------------

for j, name in enumerate(packet_generator_names):
    axes[0, j].set_title(
        name,
        fontsize=14,
        pad=12,
    )


# ---------------------------------------------------------
# Shared row colorbars
# ---------------------------------------------------------

for i in range(len(evaluation_metrics)):

    # Shared colour scale
    vmin = 0

    vmax = max(
        images[i, j].get_array().max()
        for j in range(3)
    )

    vmax = max(vmax, 0.001)

    # If the row is all zeros, keep the normalization strictly positive so
    # matplotlib does not expand the colorbar to a symmetric +/- epsilon range.
    if vmax <= vmin:
        vmax = vmin + np.finfo(float).eps

    for j in range(3):
        images[i, j].set_clim(vmin, vmax)

    # Hide y ticks on the shared axes except the leftmost column.
    for j in range(1, 3):
        axes[i, j].tick_params(axis="y", which="both", left=False, labelleft=False)

    cbar = fig.colorbar(
        images[i, 0],
        ax=axes[i, :],
        location="right",
        fraction=0.2,
        pad=0.03,
        shrink=0.8
    )

plt.savefig(
    "figures/rudimentary_P_experiments_results.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()