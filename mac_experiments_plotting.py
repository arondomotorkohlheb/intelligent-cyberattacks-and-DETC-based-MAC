from plotting_support import *

numbers_of_neurons = [1,2, 4, 8, 16, 32, 64, 128, 256]
numbers_of_neurons_names = [r"$n_n$" + f"={numbers_of_neurons[i]}" for i in range(len(numbers_of_neurons))]

numbers_of_layers = [1, 2, 4, 8, 16]
numbers_of_layers_names = [r"$n_l$" + f"={numbers_of_layers[i]}" for i in range(len(numbers_of_layers))]

model_classes = ["nn.RNN", "nn.LSTM", "nn.GRU"]
model_class_names = ["RNN", "LSTM", "GRU"]

datatype_names = [r"$\mathcal{M}^{float32}$", r"$\mathcal{M}^{int8}$"]

int8_results_success_rate = np.load("mac_experiments\\rnn_mac_approximator_results_int8_['nn.RNN', 'nn.LSTM', 'nn.GRU']x[1, 2, 4, 8, 16, 32, 64, 128, 256]x[1, 2, 4, 8, 16].npz")["success_ratio_results"]
int8_results_success_undetected_steps = np.load("mac_experiments\\rnn_mac_approximator_results_int8_['nn.RNN', 'nn.LSTM', 'nn.GRU']x[1, 2, 4, 8, 16, 32, 64, 128, 256]x[1, 2, 4, 8, 16].npz")["average_undetected_sequence_length_results"]

float32_results_success_rate = np.load("mac_experiments\\rnn_mac_approximator_results_float32_['nn.RNN', 'nn.LSTM', 'nn.GRU']x[1, 2, 4, 8, 16, 32, 64, 128, 256]x[1, 2, 4, 8, 16].npz")["success_ratio_results"]
float32_results_success_undetected_steps = np.load("mac_experiments\\rnn_mac_approximator_results_float32_['nn.RNN', 'nn.LSTM', 'nn.GRU']x[1, 2, 4, 8, 16, 32, 64, 128, 256]x[1, 2, 4, 8, 16].npz")["average_undetected_sequence_length_results"]

data4d_success_rate = np.stack([
    float32_results_success_rate,
    int8_results_success_rate,
], axis=0)

data4d_success_undetected_steps = np.stack([
    float32_results_success_undetected_steps,
    int8_results_success_undetected_steps,
], axis=0)

plotting_composite_heatmaps(
    data4d_success_rate,
    primary_y_axis = model_classes,
    primary_y_axis_names = model_class_names,
    primary_x_axis = datatype_names,
    primary_x_axis_names = datatype_names,
    secondary_y_axis = numbers_of_layers,
    secondary_y_axis_names = numbers_of_layers_names,
    secondary_x_axis = numbers_of_neurons,
    secondary_x_axis_names = numbers_of_neurons_names,
    fname = "figures\\rnn_mac_approximator_results_success_rate_heatmap.png",
    metric_name = r"$\bar{s}$"
)

plotting_composite_heatmaps(
    data4d_success_undetected_steps,
    primary_y_axis = model_classes,
    primary_y_axis_names = model_class_names,
    primary_x_axis = datatype_names,
    primary_x_axis_names = datatype_names,
    secondary_y_axis = numbers_of_layers,
    secondary_y_axis_names = numbers_of_layers_names,
    secondary_x_axis = numbers_of_neurons,
    secondary_x_axis_names = numbers_of_neurons_names,
    fname = "figures\\rnn_mac_approximator_results_success_undetected_steps_heatmap.png",
    metric_name = r"$\bar{T_{\delta}}$"
)