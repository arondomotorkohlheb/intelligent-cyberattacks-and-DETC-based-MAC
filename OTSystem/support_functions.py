import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def cartesian_convert_to_nd_spherical(x: np.ndarray):
    """
    Convert Cartesian coordinates to an n dimensional spherical coordinate system. The first element of the returned tuple is the radius, and the subsequent elements are the angles in radians.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    r = np.linalg.norm(x)
    angles = np.empty(n - 1)

    for i in range(n - 1):
        if i == n - 2:
            angles[i] = np.arctan2(x[i + 1], x[i])
        else:
            angles[i] = np.arccos(x[i] / np.linalg.norm(x[i:]))

    return r, angles

def ellipsoid_spherical_sample(P, resolution=10):
    """
    Sample points on boundary x^T P x = 1 using hyperspherical coordinates.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    # --- build correct angle vectors ---
    angle_vectors = [
        np.linspace(0, np.pi, int(resolution/2), endpoint=True)
        for _ in range(n - 1)
    ]

    angle_vectors[-1] = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # --- meshgrid ---
    meshes = np.meshgrid(*angle_vectors, indexing="ij")

    angles = np.stack([m.ravel() for m in meshes], axis=1)

    # --- hyperspherical coordinates ---
    sines = np.sin(angles)
    sin_prod = np.concatenate(
        [np.ones((angles.shape[0], 1)),
         np.cumprod(sines, axis=1)],
        axis=1
    )

    unit = np.empty((angles.shape[0], n))

    unit[:, :-1] = sin_prod[:, :-1] * np.cos(angles)
    unit[:, -1] = sin_prod[:, -1]
   
    # --- map to ellipsoid ---
    A = np.linalg.inv(P)
    L = np.linalg.cholesky(A)

    points = unit @ L.T

    return points

def ellipsoid_spherical_sample_2(P, resolution=10):
    """
    Sample points on boundary x^T P x = 1 using hyperspherical coordinates.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    # --- build correct angle vectors ---
    angle_vectors = [
        np.linspace(0, np.pi, int(resolution/2), endpoint=True)
        for _ in range(n - 1)
    ]

    angle_vectors[-1] = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # --- meshgrid ---
    meshes = np.meshgrid(*angle_vectors, indexing="ij")

    angles = np.stack([m.ravel() for m in meshes], axis=1)

    # --- hyperspherical coordinates ---
    sines = np.sin(angles)
    sin_prod = np.concatenate(
        [np.ones((angles.shape[0], 1)),
         np.cumprod(sines, axis=1)],
        axis=1
    )

    unit = np.empty((angles.shape[0], n))

    unit[:, :-1] = sin_prod[:, :-1] * np.cos(angles)
    unit[:, -1] = sin_prod[:, -1]
   
    # scale with ellispoid to much the volumne
    points = unit * np.linalg.det(P) * 0.06

    return points

def ellipsoid_projection(P, dims=(0, 1), ax = None, n_points=300, **plot_kwargs):
    """
    Plot the coordinate projection of
        (x - c)^T P (x - c) <= 1

    Parameters
    ----------
    P : (n, n) ndarray
        Positive-definite shape matrix.
    c : (n,) ndarray or None
        Ellipsoid center. Defaults to zero.
    dims : tuple[int, int]
        Coordinates to retain, e.g. (0, 2).
    ax : matplotlib.axes.Axes or None
        Existing axis; created if omitted.
    """

    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    c = np.zeros(n)

    if np.linalg.matrix_rank(P[np.ix_(dims, dims)]) < 2:
        return None
    else:
        if np.linalg.matrix_rank(P) < n:
            P_full_rank = P.copy()
            dims2 = list(dims).copy()
            # reduce P to a matrix with full rank: 
            # find the index for which the column and row are zeros
            for i in range(n):
                if np.all(P[i, :] == 0) and np.all(P[:, i] == 0):
                    P_full_rank = np.delete(P_full_rank, i, axis=0)
                    P_full_rank = np.delete(P_full_rank, i, axis=1)
                    dims2 = [d - 1 if d > i else d for d in dims2]

            dims2 = tuple(dims2)
            if np.linalg.matrix_rank(P_full_rank) < P_full_rank.shape[0]:
                A_2d = np.linalg.inv(P[np.ix_(dims, dims)])
            else:
                A = np.linalg.inv(P_full_rank)
                A_2d = A[np.ix_(dims2, dims2)]
        else:
            A = np.linalg.inv(P)
            A_2d = A[np.ix_(dims, dims)]

    # Coordinate projection: select the relevant covariance block
    c_2d = c[list(dims)]
    # check if A_2d is full rank

    if np.linalg.matrix_rank(A_2d) < A_2d.shape[0]:
        return None

    # A_2d = L L^T, maps the unit circle to the projected ellipse
    L = np.linalg.cholesky(A_2d)

    theta = np.linspace(0, 2 * np.pi, n_points)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))

    ellipse = c_2d[:, None] + L @ unit_circle

    return ellipse

def plot_4d_ellipsoid_and_points(P, points, path, ellipsoid_projection_name = None, highlighted_points = None, point_color = "tab:blue", points2 = None, point_color2 = "tab:orange", highlighted_points_name = r"$\mathcal{X}^{s}$", points1_name = "Sampled states", points2_name = "Unstable states", transparency = 0.06):
    state_labels = [r"$\dot  \theta [\degree/s]$", r"$\dot \alpha [\degree/s]$", r"$\theta [\degree]$", r"$\alpha [\degree]$"]
    state_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    plt.figure(figsize=(12, 9))
    for subplot_index, (first_index, second_index) in enumerate(state_pairs, start=1):
        plt.subplot(2, 3, subplot_index)

        plt.scatter(
            points[:, first_index]*180/np.pi,
            points[:, second_index]*180/np.pi,
            s=6,
            alpha=transparency,
            color=point_color,
            label=points1_name,
        )
        if highlighted_points is not None:
            plt.scatter(
                    highlighted_points[:, first_index]*180/np.pi,
                    highlighted_points[:, second_index]*180/np.pi,
                    s=28,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                    marker="o",
                    label=highlighted_points_name,
                )
        if points2 is not None:
            plt.scatter(
                points2[:, first_index]*180/np.pi,
                points2[:, second_index]*180/np.pi,
                s=6,
                alpha=transparency,
                color=point_color2,
                label=points2_name,
            )

        ellipse = ellipsoid_projection(P, dims=(first_index, second_index), n_points=300)
        if ellipse is not None:
            ellipse = ellipse * 180 / np.pi
            plt.plot(ellipse[0, :], ellipse[1, :], color="black", linewidth=2, label=ellipsoid_projection_name)

        plt.xlabel(state_labels[first_index])
        plt.ylabel(state_labels[second_index])
        plt.title(f"{state_labels[first_index]} vs {state_labels[second_index]}")
    
    legend_handle = Line2D(
                            [0], [0],
                            marker="o",
                            linestyle="",
                            markerfacecolor="blue",
                            markeredgecolor="blue",
                            markersize=2,
                            alpha=1
                            )

    plt.legend([legend_handle], ["Samples"])

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_3d_ellipsoid_and_points(P, points, path, ellipsoid_projection_name = None, highlighted_points = None, P2 = None, n_points=300, values=None, scale = (0,1), point_color = "tab:blue", points2 = None, point_color2 = "tab:orange", highlighted_points_name = r"$\mathcal{X}^{s}$", points1_name = "Sampled states", points2_name = "Unstable states", transparency = 0.06):
    state_labels = [r"$\dot  \theta [\degree/s]$", r"$\dot \alpha [\degree/s]$", r"$\alpha [\degree]$"]
    state_pairs = [(0, 1), (0, 2), (1, 2)]

    plt.figure(figsize=(12, 4))
    for subplot_index, (first_index, second_index) in enumerate(state_pairs, start=1):
        plt.subplot(1, 3, subplot_index)
        if values is None:
            plt.scatter(
                points[:, first_index]*180/np.pi,
                points[:, second_index]*180/np.pi,
                s=6,
                alpha=transparency,
                color=point_color,
                label=points1_name,
            )
            if highlighted_points is not None:
                plt.scatter(
                        highlighted_points[:, first_index]*180/np.pi,
                        highlighted_points[:, second_index]*180/np.pi,
                        s=28,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.2,
                        marker="o",
                        label=highlighted_points_name,
                    )
            if points2 is not None:
                plt.scatter(
                    points2[:, first_index]*180/np.pi,
                    points2[:, second_index]*180/np.pi,
                    s=6,
                    alpha=transparency,
                    color=point_color2,
                    label=points2_name,
                )
        else:
            green_red = LinearSegmentedColormap.from_list(
                    "green_red",
                    ["green", "red"]
                )
            plt.scatter(
                points[:, first_index]*180/np.pi,
                points[:, second_index]*180/np.pi,
                c=values,
                s=6,
                alpha=transparency,
                cmap=green_red,
                vmin=scale[0],
                vmax=scale[1],
                label=points1_name,
            )
            if highlighted_points is not None:
                norm = Normalize(vmin=scale[0], vmax=scale[1])
                plt.scatter(
                    highlighted_points[:, first_index] * 180 / np.pi,
                    highlighted_points[:, second_index] * 180 / np.pi,
                    s=28,
                    facecolors="none",
                    edgecolors=green_red(norm(values)),
                    linewidths=1.2,
                    marker="o",
                    label=highlighted_points_name,
                )
        ellipse = ellipsoid_projection(P, dims=(first_index, second_index), n_points=300)
        if P2 is not None:
            ellipse2 = ellipsoid_projection(P2, dims=(first_index, second_index), n_points=300)
            if ellipse2 is not None:
                ellipse2 = ellipse2 * 180 / np.pi
                plt.plot(ellipse2[0, :], ellipse2[1, :], color="orange", linewidth=2, label="Ellipsoid boundary projection")

        if ellipse is not None:
            ellipse = ellipse * 180 / np.pi
            plt.plot(ellipse[0, :], ellipse[1, :], color="black", linewidth=2, label="Ellipsoid boundary projection")
        

        plt.xlabel(state_labels[first_index])
        plt.ylabel(state_labels[second_index])
        plt.title(f"{state_labels[first_index]} vs {state_labels[second_index]}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_distribution(values : dict, path, bins: int = 100, bounds: None | dict = None):
    i = 0
    number_of_data_sets = len(values.keys())
    plt.figure(figsize=(12, 12))
    for key in values.keys():
        plt.subplot(number_of_data_sets, 1, i+1)
        plt.hist(values[key], bins=bins, density=True, label=key)
        # add the detector bound as a vertical red line for each
        if bounds is not None and key in bounds:
            plt.axvline(bounds[key], color='red', linewidth=2, label='Detector bound')
        plt.ylabel(f"{key}", rotation=80)
        i += 1
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def heatmap_covariances(
    extrapolated=None,
    computed=None,
    path="../figures/LQR_KalmanEstimator/heatmap_covariances.png",
):
    if extrapolated is None:
        extrapolated = np.array([
            [4.90170178e-01, 3.19023431e-02, -3.96036426e-04, 2.09740511e-03],
            [3.19023431e-02, 5.54364550e-03, -2.15997055e-03, -5.53956851e-06],
            [-3.96036426e-04, -2.15997055e-03, 4.57891804e-02, 9.31359807e-04],
            [2.09740511e-03, -5.53956851e-06, 9.31359807e-04, 4.23411459e-05],
        ])

    if computed is None:
        computed = np.array([
            [7.37698155e-01, 5.63569247e-02, -1.74520755e-02, 2.02382511e-03],
            [5.63569247e-02, 1.27471523e-02, -4.58968859e-03, -1.34021457e-04],
            [-1.74520755e-02, -4.58968859e-03, 4.42892151e-02, 9.13722548e-04],
            [2.02382511e-03, -1.34021457e-04, 9.13722548e-04, 4.35507233e-05],
        ])
    

    cmap = LinearSegmentedColormap.from_list(
        "red_white_blue",
        ["red", "white", "blue"]
    )    

    difference = np.abs(computed - extrapolated)

    cov_min = min(computed.min(), extrapolated.min())
    cov_max = max(computed.max(), extrapolated.max())


    fig, axs = plt.subplots(
        1, 3, figsize=(16, 5), constrained_layout=True
    )

    titles = [
        "Computed covariance",
        "Extrapolated covariance",
        "Absolute difference",
    ]
    data = [computed, extrapolated, difference]

    # Shared scale for the covariance matrices
    axs[0].imshow(data[0], vmin=cov_min, vmax=cov_max)
    axs[1].imshow(data[1], vmin=cov_min, vmax=cov_max)
    im_diff = axs[2].imshow(data[2], vmin=cov_min, vmax=cov_max)

    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    # colorbar that goes from vmin to vmax at ax=axs[2]

    # Colorbar for difference
    fig.colorbar(
        im_diff,
        ax=axs[2],
        fraction=0.046,
        pad=0.04,
    )

    plt.savefig(path, dpi=300)
    plt.close(fig)

def plot_n_ellipsoids(Plist, path, colourlist = ["orange", "green", "purple", "brown"], label_list = [r"$\Sigma^1$", r"$\Sigma^2$", r"$\Sigma^3$", r"$\Sigma^4$"]):
    state_labels = [r"$\dot  \theta [\degree/s]$", r"$\dot \alpha [\degree/s]$", r"$\theta [\degree]$", r"$\alpha [\degree]$"]
    state_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    plt.figure(figsize=(12, 9))
    for subplot_index, (first_index, second_index) in enumerate(state_pairs, start=1):
        plt.subplot(2, 3, subplot_index)
        for i, P in enumerate(Plist):
            ellipse = ellipsoid_projection(P, dims=(first_index, second_index), n_points=300)
            if ellipse is not None:
                ellipse = ellipse * 180 / np.pi
                plt.plot(ellipse[0, :], ellipse[1, :], color=colourlist[i], linewidth=2, label=label_list[i])

        plt.xlabel(state_labels[first_index])
        plt.ylabel(state_labels[second_index])
        plt.title(f"{state_labels[first_index]} vs {state_labels[second_index]}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_n_ellipsoids_3d(Plist, path, colourlist = ["orange", "green", "purple", "brown"], label_list = [r"$\Sigma^1$", r"$\Sigma^2$", r"$\Sigma^3$", r"$\Sigma^4$"]):
    state_labels = [r"$\dot  \theta [\degree/s]$", r"$\dot \alpha [\degree/s]$", r"$\alpha [\degree]$"]
    state_pairs = [(0, 1), (0, 2), (1, 2)]

    plt.figure(figsize=(12, 9))
    for subplot_index, (first_index, second_index) in enumerate(state_pairs, start=1):
        plt.subplot(2, 3, subplot_index)
        for i, P in enumerate(Plist):
            ellipse = ellipsoid_projection(P, dims=(first_index, second_index), n_points=300)
            if ellipse is not None:
                ellipse = ellipse * 180 / np.pi
                plt.plot(ellipse[0, :], ellipse[1, :], color=colourlist[i], linewidth=2, label=label_list[i])

        plt.xlabel(state_labels[first_index])
        plt.ylabel(state_labels[second_index])
        plt.title(f"{state_labels[first_index]} vs {state_labels[second_index]}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_step_signal(values, binary_color, colors=("tab:blue", "tab:red")):
    """
    Plot a blocky signal with arbitrary y-values.
    
    values: array of signal levels
    binary_color: binary array selecting the color of each segment
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    values = np.asarray(values)
    binary_color = np.asarray(binary_color)

    x = np.arange(len(values))

    for i in range(len(values) - 1):
        # Horizontal segment
        ax.hlines(
            values[i],
            x[i],
            x[i+1],
            color=colors[binary_color[i]],
            linewidth=2
        )

        # Vertical transition
        if values[i] != values[i+1]:
            ax.vlines(
                x[i+1],
                values[i],
                values[i+1],
                color=colors[binary_color[i]],
                linewidth=2
            )

    ax.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    heatmap_covariances()
