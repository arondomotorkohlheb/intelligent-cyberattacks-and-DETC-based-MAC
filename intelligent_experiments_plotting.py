import numpy as np
import matplotlib.pyplot as plt

window_sizes =  [2, 8, 32, 64]
adversary_delays = [1]
numbers_of_layers = [1,2,4]

numbers_of_neurons = [2, 8, 32, 64]
adversary_observers = ["InjectAndListen", "InjectAndListen2Channels"]

controllers = ["MPC", "LQR"]
estimators = ["KalmanEstimator", "Observer"]

supervised_policy_types = ["Nn"]
success_rate = np.load("ot+adversary_systems\\_success_rate_of_attackers2_[2, 8, 32, 64]x[2, 8, 32, 64]x[1, 2, 4].npz")["success_rate_of_attackers"]
# detection_rate = np.load("ot+adversary_systems\_detection_rate_of_attackers_[2, 8, 32, 64]x[2, 8, 32, 64]x[1, 2, 4].npz")["detection_rate_of_attackers"]
# detection_time = np.load("ot+adversary_systems\_undetected_timesteps_of_attackers_[2, 8, 32, 64]x[2, 8, 32, 64]x[1, 2, 4].npz")["undetected_timesteps_of_attackers"]
# estimated_alpha_difference = np.load("ot+adversary_systems\_estimation_error_of_attackers_[2, 8, 32, 64]x[2, 8, 32, 64]x[1, 2, 4].npz")["estimation_error_of_attackers"]

adversary_observer_names = [
    r"$\mathcal{O}^{\mathcal{I}}$",
    r"$\mathcal{O}^{\mathcal{II}}$",
]

controller_names = [
    r"$\mathcal{C}^{MPC}$",
    r"$\mathcal{C}^{LQR}$",
]

estimator_names = [
    r"$\mathcal{E}^{\mathcal{K}}$",
    r"$\mathcal{E}^{\mathcal{L}}$",
]
primary_y_axis = window_sizes
primary_y_axis_names = [r"$w$" + f"={window_sizes[i]}" for i in range(len(window_sizes))]

primary_x_axis = adversary_observers
primary_x_axis_names = adversary_observer_names

secondary_y_axis = numbers_of_layers
secondary_y_axis_names = [r"$n_l$" + f"={numbers_of_layers[i]}" for i in range(len(numbers_of_layers))]

secondary_x_axis = numbers_of_neurons
secondary_x_axis_names = [r"$n_n$" + f"={numbers_of_neurons[i]}" for i in range(len(numbers_of_neurons))]

for controller in controllers:
    for estimator in estimators:
        controller_index = controllers.index(controller)
        estimator_index = estimators.index(estimator)

        # ---------------------------------------------------------
        # Figure
        # ---------------------------------------------------------
        fig, axes = plt.subplots(
            len(primary_y_axis),
            len(primary_x_axis),
            figsize=(3,5),
            sharex="col",
            sharey="row",
            gridspec_kw={
                "wspace": 0,
                "hspace": 0,
            },
        )

        images = np.empty((len(primary_y_axis), len(primary_x_axis)), dtype=object)

        cmap = plt.cm.Blues


        for i, primary_y_ax in enumerate(primary_y_axis):
            for j, primary_x_ax in enumerate(primary_x_axis):

                ax = axes[i, j]

                data = success_rate[controller_index, estimator_index, j, i, 0, :, :].T

                im = ax.imshow(
                    data,
                    cmap=cmap,
                    aspect="auto", 
                    vmax=1,
                )

                images[i, j] = im

                ax.set_xticks([])
                ax.set_yticks([])

                if j == 0:
                    ax.set_ylabel(
                        primary_y_axis_names[i],
                        fontsize=12,
                        labelpad=12,
                    )
                    ax.yaxis.set_label_position("left")
                    ax.yaxis.tick_right()

                if i == 0:
                    ax.set_xlabel(
                        primary_x_axis_names[j],
                        fontsize=12,
                        labelpad=12,
                    )
                    ax.xaxis.set_label_position("top")
                
                if i == len(primary_y_axis) - 1:
                    ax.set_xticks(range(len(secondary_x_axis)))
                    ax.set_xticklabels(secondary_x_axis_names, fontsize=10, rotation=-90)

                if j == len(primary_x_axis) - 1:
                    ax.set_yticks(range(len(secondary_y_axis)))
                    ax.set_yticklabels(secondary_y_axis_names, fontsize=10)
                    ax.tick_params(axis="y", labelleft=False, labelright=True, left=False, right=True)

        plt.savefig(
            f"figures/{controller}_{estimator}_intelligent_P_experiments_results.png",
            dpi=300,
            bbox_inches="tight",
        )