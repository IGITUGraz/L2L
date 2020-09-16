import logging.config
import os

import numpy as np
from l2l.utils.environment import Environment

from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules
from l2l.paths import Paths

from l2l.logging_tools import create_shared_logger_data, configure_loggers

from l2l.utils import JUBE_runner as jube

logger = logging.getLogger('bin.l2l-fun-sa')


def main():
    name = 'L2L-FunctionGenerator-SA'
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
    # This initializes an environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      # freeze_input=True,
                      # multiproc=True,
                      # use_scoop=True,
                      # wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
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

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")
    # Execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec", "python " +
                                  os.path.join(paths.simulation_path, "run_files/run_optimizee.py"))
    # Paths
    traj.f_add_parameter_to_group("JUBE_params", "paths", paths)


    ## Benchmark function
    function_id = 14
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    #Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, paths.simulation_path)

    ## Outerloop optimizer initialization
    parameters = SimulatedAnnealingParameters(n_parallel_runs=50, noisy_step=.03, temp_decay=.99, n_iteration=100,
                                              stop_criterion=np.Inf, seed=np.random.randint(1e5), cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)

    optimizer = SimulatedAnnealingOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                            optimizee_fitness_weights=(-1,),
                                            parameters=parameters,
                                            optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    ## Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
