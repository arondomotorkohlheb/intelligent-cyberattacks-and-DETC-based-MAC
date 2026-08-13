from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy as np

def plt_interpolated_grid_heatmap(
    value_matrix,
    axis1_parameters,
    axis2_parameters,
    axis1_label,
    axis2_label,
    title,
    filename,
    max_value=1,
):

    # Original parameter values
    x = np.array(axis2_parameters)
    y = np.array(axis1_parameters)

    Z = value_matrix

    # Integer grid
    x_full = np.arange(x.min(), x.max() + 1)
    y_full = np.arange(y.min(), y.max() + 1)

    # Linear interpolation
    interp = RegularGridInterpolator(
        (y, x),
        Z,
        method="linear",
        bounds_error=False,
        fill_value=None, #type: ignore
    )

    # Evaluate on integer grid
    Y, X = np.meshgrid(y_full, x_full, indexing="ij")
    points = np.column_stack((Y.ravel(), X.ravel()))
    Z_full = interp(points).reshape(len(y_full), len(x_full))

    cmap = LinearSegmentedColormap.from_list(
        "white_darkred",
        ["white", "#8B0000"]
    )

    plt.figure(figsize=(6, 6))

    img = plt.imshow(
        Z_full,
        origin="lower",
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=max_value),
        interpolation="nearest",
        extent=[x_full[0] - 0.5,x_full[-1] + 0.5,y_full[0] - 0.5,y_full[-1] + 0.5], #type: ignore
        aspect="equal",
    )

    # Only show actual integer parameter values
    plt.xticks(x_full)
    plt.yticks(y_full)

    plt.xlabel(axis2_label)
    plt.ylabel(axis1_label)
    plt.title(title)

    # Colorbar
    plt.colorbar(img, label="Value")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()

def plt_grid_heatmap(
    value_matrix,
    axis1_parameters,
    axis2_parameters,
    axis1_label,
    axis2_label,
    title,
    filename,
    max_value=1,
    colour = "red",
    metric_name = r"$\bar{s}$",
):

    # Original parameter values
    x = np.array(axis2_parameters)
    y = np.array(axis1_parameters)


    Z = value_matrix
    if colour == "red":
        cmap = LinearSegmentedColormap.from_list(
            "white_darkred",
            ["white", "#8B0000"]
        )
    elif colour == "blue":
        cmap = LinearSegmentedColormap.from_list(
            "white_darkblue",
            ["white", "#00008B"]
        )
    else:
        raise ValueError("Invalid colour specified. Choose 'red' or 'blue'.")

    plt.figure(figsize=(6, 6))

    img = plt.imshow(
        Z,
        origin="lower",
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=max_value),
        aspect="equal",
    )

    # Only show actual integer parameter values
    plt.xticks(range(len(axis2_parameters)), labels=axis2_parameters)
    plt.yticks(range(len(axis1_parameters)), labels=axis1_parameters)
    # i want axis labels to not be a number line, but just the list of values in the list put for each grid
    plt.xlabel(axis2_label)
    plt.ylabel(axis1_label)
    plt.title(title)

    # Colorbar
    plt.colorbar(img, label="Value")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()

def plotting_composite_heatmaps(
        data4d,
        primary_y_axis,
        primary_y_axis_names,
        primary_x_axis,
        primary_x_axis_names,
        secondary_y_axis,
        secondary_y_axis_names,
        secondary_x_axis,
        secondary_x_axis_names,
        fname,
        metric_name = r"$\bar{s}$",):
            
    fig, axes = plt.subplots(
            len(primary_y_axis),
            len(primary_x_axis),
            figsize=(8,8),
            sharex="col",
            sharey="row",
            gridspec_kw={
                "wspace": 0,
                "hspace": 0,
            },
        )

    images = np.empty((len(primary_y_axis), len(primary_x_axis)), dtype=object)

    cmap = plt.cm.Blues

    vmax = max(np.max(data4d), 1)

    for i, primary_y_ax in enumerate(primary_y_axis):
        for j, primary_x_ax in enumerate(primary_x_axis):

            ax = axes[i, j]

            data = data4d[j, i, :, :].T

            im = ax.imshow(
                data,
                cmap=cmap,
                aspect="auto", 
                vmax=vmax,
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

    # adjust the rendered canvas to include the colorbar at the bottom, I want to stretch the rendered canvas downwards, but keep the same figure size 

    # Create space at the bottom of the figure
    fig.subplots_adjust(bottom=0.2)

    # cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  #type: ignore I want this to be more downwards, so that it is not overlapping with the x-axis labels, and I want it to be horizontal
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) #type: ignore
    cbar = fig.colorbar(images[0, 0], cax=cbar_ax, orientation='horizontal')

    plt.savefig(
        fname,
        dpi=300,
        bbox_inches="tight",
    )


    plt.show()

if __name__ == "__main__":
    # load data files
    window_sizes = [2,4,8,16,32,64]
    adversary_delays = [1]
    numbers_of_neurons = [2, 4, 8, 16, 32, 64]
    supervised_policy_type = "Nn"
    training_data_repetition_count = int(1e3)
    fname1 = f"high-level-analysis/{supervised_policy_type}_training_size_{training_data_repetition_count}_success_rate_of_attackers_{window_sizes}x{numbers_of_neurons}.npz"
    fname2 = f"high-level-analysis/{supervised_policy_type}_training_size_{training_data_repetition_count}_undetected_timesteps_of_attackers_{window_sizes}x{numbers_of_neurons}.npz"

    data1 = np.load(fname1)
    data2 = np.load(fname2)

    # i want to make a heatmap of the success rate of attackers and undetected timesteps of attackers for different window sizes and numbers of neurons but this time 
    # i want the heatmap to not interpolate, only use the actual values window sizes and number of neurons for thee axis
    plt_grid_heatmap(
        value_matrix=data1["success_rate_of_attackers"],
        axis1_parameters=data1["window_sizes"],
        axis2_parameters=data1["numbers_of_neurons"],
        axis1_label="Window Size",
        axis2_label="Number of Neurons",
        title="Success Rate of Attackers",
        filename=f"high-level-analysis/{supervised_policy_type}_success_rate_of_attackers_{window_sizes}x{numbers_of_neurons}.png",
        max_value=1
    )