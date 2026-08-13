from pprint import pprint

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

from itertools import product as cartesian_product

from factory import generate_simulators
 
window_sizes =  [2, 8, 32, 64]
adversary_delays = [1]
packet_generation_nn_layers = [1,2,4]

numbers_of_neurons = [2, 8, 32, 64]
adversary_observers = ["InjectAndListen", "InjectAndListen2Channels"]

# window_sizes =  [32, 64]
# adversary_observers = ["InjectAndListen2Channels"]
# packet_generation_nn_layers = [2,4]
# numbers_of_neurons = [32, 64]

controllers = ["MPC", "LQR"]
estimators = ["KalmanEstimator", "Observer"]

supervised_policy_types = ["Nn"] #, "Rnn"]

packet_generators = [
    (f"{supervised_policy_type}IntelligentPacketGenerator", {
        "number_of_neurons": n,
        "number_of_layers": layers
    }) for n in numbers_of_neurons for layers in packet_generation_nn_layers for supervised_policy_type in supervised_policy_types
]


schedulers = [
    ("AlwaysInject", {})
]

success_rate_of_attackers = np.zeros((len(controllers), len(estimators), len(adversary_observers), len(window_sizes), len(supervised_policy_types), len(numbers_of_neurons), len(packet_generation_nn_layers)))
undetected_timesteps_of_attackers = np.zeros((len(controllers), len(estimators), len(adversary_observers), len(window_sizes), len(supervised_policy_types),len(numbers_of_neurons), len(packet_generation_nn_layers)))
detection_rate_of_attackers = np.zeros((len(controllers), len(estimators), len(adversary_observers), len(window_sizes), len(supervised_policy_types), len(numbers_of_neurons), len(packet_generation_nn_layers)))
estimation_error_of_attackers = np.zeros((len(controllers), len(estimators), len(adversary_observers), len(window_sizes), len(supervised_policy_types), len(numbers_of_neurons), len(packet_generation_nn_layers)))

detector_params = "full_auto"

ot_with_adversary_simulators: list[Simulator] = []
    # Cartesian product of all configuration choices

combinations = cartesian_product(
    window_sizes,
    adversary_delays,
    controllers,
    estimators,
    adversary_observers,
    packet_generators,
    schedulers,
)

for combination in combinations:
    window_size = combination[0]
    adversary_delay = combination[1]
    controller = combination[2]
    estimator = combination[3]
    observer = combination[4]

    pg_name, pg_params = combination[5][0], combination[5][1]
    sch_name, sch_params = combination[6][0], combination[6][1]

    simulator = create_simulator(controller, estimator, detector_params, observer, pg_name, sch_name, window_size, adversary_delay, pg_params, sch_params, authenticator_type = None)
    # just training first

    if isinstance(simulator.adversary.packet_generator, IntelligentPacketGenerator):
        if not simulator.adversary.packet_generator.policy.load_model(simulator):
            simulator.adversary.packet_generator.policy.train(simulator)
        
        eval_dict = simulator.adversary.evaluate(repetitions=100)
        print(":"*100)
        print(f"Policy type: {simulator.adversary.packet_generator.policy.__class__.__name__}, observer_type: {simulator.adversary.observer.name}, window size: {simulator.adversary.observer.window_size}, number of neurons: {simulator.adversary.packet_generator.policy.number_of_neurons}, number of layers: {simulator.adversary.packet_generator.policy.number_of_layers}:")
        print(f"average undetected timesteps: {np.mean(eval_dict['undetected_timesteps'])}")
        print(f"average alpha estimation error: {np.mean(np.abs(eval_dict['estimation_errors']))*180/np.pi} degrees")
        print(f"average terminal alpha: {np.mean(np.abs(eval_dict['terminal_states'][:, 3]))*180/np.pi} degrees")
        print(f"success rate: {eval_dict['success_rate']}")
        print(f"detection rate: {eval_dict['detection_rate']}")
        print(":"*100)

        success_rate_of_attackers[
            controllers.index(controller),
            estimators.index(estimator),
            adversary_observers.index(observer),
            window_sizes.index(window_size),
            supervised_policy_types.index(simulator.adversary.packet_generator.policy.policy_type),
            numbers_of_neurons.index(pg_params["number_of_neurons"]),
            packet_generation_nn_layers.index(pg_params["number_of_layers"])
        ] = eval_dict["success_rate"]

        undetected_timesteps_of_attackers[
            controllers.index(controller),
            estimators.index(estimator),
            adversary_observers.index(observer),
            window_sizes.index(window_size),
            supervised_policy_types.index(simulator.adversary.packet_generator.policy.policy_type),
            numbers_of_neurons.index(pg_params["number_of_neurons"]),
            packet_generation_nn_layers.index(pg_params["number_of_layers"])
        ] = np.mean(eval_dict["undetected_timesteps"])

        detection_rate_of_attackers[
            controllers.index(controller),
            estimators.index(estimator),
            adversary_observers.index(observer),
            window_sizes.index(window_size),
            supervised_policy_types.index(simulator.adversary.packet_generator.policy.policy_type),
            numbers_of_neurons.index(pg_params["number_of_neurons"]),
            packet_generation_nn_layers.index(pg_params["number_of_layers"])
        ] = eval_dict["detection_rate"]

        estimation_error_of_attackers[
            controllers.index(controller),
            estimators.index(estimator),
            adversary_observers.index(observer),
            window_sizes.index(window_size),
            supervised_policy_types.index(simulator.adversary.packet_generator.policy.policy_type),
            numbers_of_neurons.index(pg_params["number_of_neurons"]),
            packet_generation_nn_layers.index(pg_params["number_of_layers"])
        ] = np.mean(eval_dict["estimation_errors"])*180/np.pi

    else:
        raise ValueError(f"Not intelligent packet generator type: {type(simulator.adversary.packet_generator)}")
    
    del simulator

root = "ot+adversary_systems"

np.savez(f"{root}/_success_rate_of_attackers2_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.npz", success_rate_of_attackers=success_rate_of_attackers, window_sizes=window_sizes, numbers_of_neurons=numbers_of_neurons)
np.savez(f"{root}/_undetected_timesteps_of_attackers2_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.npz", undetected_timesteps_of_attackers=undetected_timesteps_of_attackers, window_sizes=window_sizes, numbers_of_neurons=numbers_of_neurons)
np.savez(f"{root}/_detection_rate_of_attackers2_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.npz", detection_rate_of_attackers=detection_rate_of_attackers, window_sizes=window_sizes, numbers_of_neurons=numbers_of_neurons)
np.savez(f"{root}/_estimation_error_of_attackers2_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.npz", estimation_error_of_attackers=estimation_error_of_attackers, window_sizes=window_sizes, numbers_of_neurons=numbers_of_neurons)

# plot_filename = simulator.figure_directory + f"/{supervised_policy_type}_success_rate_of_attackers_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.png"
# plt_grid_heatmap(success_rate_of_attackers, window_sizes, numbers_of_neurons, "Window Size", "Number of Neurons", "Success Rate of Attackers", plot_filename, max_value=1) #type: ignore

# plot_filename = simulator.figure_directory + f"/{supervised_policy_type}_undetected_timesteps_of_attackers_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.png"
# plt_grid_heatmap(undetected_timesteps_of_attackers, window_sizes, numbers_of_neurons, "Window Size", "Number of Neurons", "Average Undetected Timesteps of Attackers", plot_filename, max_value=np.max(np.array(undetected_timesteps_of_attackers))) #type: ignore

# plot_filename = simulator.figure_directory + f"/{supervised_policy_type}_detection_rate_of_attackers_{window_sizes}x{numbers_of_neurons}x{packet_generation_nn_layers}.png"
# plt_grid_heatmap(detection_rate_of_attackers, window_sizes, numbers_of_neurons, "Window Size", "Number of Neurons", "Detection Rate of Attackers", plot_filename, max_value=1) #type: ignore