from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

sine_signal = np.sin(2 * np.pi * 0.1 * np.arange(10))

plt.plot(sine_signal, label=r"$\mathcal{X}^{s}$", color="tab:blue", alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Amplitude")

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

plt.show()