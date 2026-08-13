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


def generate_simulators(
        window_sizes: list[int],
        adversary_delays: list[int],
        adversary_observers: list[str],
        packet_generators: list[tuple[str, dict]],
        schedulers: list[tuple[str, dict]],
        authenticator:str | None,
        controllers: list[str],
        estimators: list[str],
):
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

        simulator = create_simulator(controller, estimator, detector_params, observer, pg_name, sch_name, window_size, adversary_delay, pg_params, sch_params, authenticator)
        ot_with_adversary_simulators.append(simulator)
    
    return ot_with_adversary_simulators