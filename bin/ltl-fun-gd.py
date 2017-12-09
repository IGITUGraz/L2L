import logging.config
import os

import sys
sys.path.append('.')
import numpy as np
from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizees.functions import tools as function_tools
from ltl.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
# from ltl.optimizers.gradientdescent.optimizer import ClassicGDParameters
# from ltl.optimizers.gradientdescent.optimizer import StochasticGDParameters
# from ltl.optimizers.gradientdescent.optimizer import AdamParameters
from ltl.optimizers.gradientdescent.optimizer import RMSPropParameters
from ltl.paths import Paths
from ltl.recorder import Recorder

from ltl.logging_tools import create_shared_logger_data, configure_loggers

logger = logging.getLogger('bin.ltl-fun-gradientdescent')


def main():
    name = 'LTL-FUN-GD'
    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      freeze_input=False,
                      multiproc=True,
                      use_scoop=False,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      )
    create_shared_logger_data(logger_names=['bin', 'optimizers'],
                              log_levels=['INFO', 'INFO'],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    ## Benchmark function
    function_id = 7
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=False)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    # parameters = ClassicGDParameters(learning_rate=0.01, exploration_step_size=0.01,
    #                                  n_random_steps=5, n_iteration=100,
    #                                  stop_criterion=np.Inf)
    # parameters = AdamParameters(learning_rate=0.01, exploration_step_size=0.01, n_random_steps=5, first_order_decay=0.8,
    #                             second_order_decay=0.8, n_iteration=100, stop_criterion=np.Inf)
    # parameters = StochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
    #                                     exploration_step_size=0.01, n_random_steps=5, n_iteration=100,
    #                                     stop_criterion=np.Inf)
    parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.5,
                                   n_random_steps=20, momentum_decay=0.3,
                                   n_iteration=200, stop_criterion=np.Inf, seed=np.random.randint(1, 2**32))

    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(-0.1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Add Recorder
    recorder = Recorder(trajectory=traj,
                        optimizee_name=benchmark_name, optimizee_parameters=benchmark_parameters,
                        optimizer_name=optimizer.__class__.__name__,
                        optimizer_parameters=optimizer.get_params())
    recorder.start()

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    ## Outerloop optimizer end
    optimizer.end(traj)
    recorder.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
