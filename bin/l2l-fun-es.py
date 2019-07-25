import logging.config

import numpy as np
from l2l.utils.environment import Environment

from l2l import dict_to_list
import l2l.utils.JUBE_runner as jube
from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from l2l.paths import Paths
import os

logger = logging.getLogger('bin.l2l-fun-es')


def run_experiment():
    name = 'L2L-FUN-ES'
    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("You have not set the root path to store your results."
                                " Write the path to a path.conf text file in the bin directory"
                                " before running the simulation")

    trajectory_name = 'mirroring-and-fitness-shaping'

    paths = Paths(name, dict(run_num='test'), root_dir_path=root_dir_path, suffix="-" + trajectory_name)

    print("All output logs can be found in directory ", paths.logs_path)

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(
        trajectory=trajectory_name,
        filename=paths.output_dir_path,
        file_title='{} data'.format(name),
        comment='{} data'.format(name),
        add_time=True,
        automatic_storing=True,
        log_stdout=False,  # Sends stdout to logs
    )
    create_shared_logger_data(
        logger_names=['bin', 'optimizers'],
        log_levels=['INFO', 'INFO'],
        log_to_consoles=[True, True],
        sim_name=name,
        log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory
    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")
    traj.f_add_parameter_to_group("JUBE_params", "exec", "python3 " +
                                  os.path.join(paths.simulation_path, "run_files/run_optimizee.py"))
    # Paths
    traj.f_add_parameter_to_group("JUBE_params", "paths", paths)

    ## Benchmark function
    function_id = 14
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 200
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    # Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, paths.simulation_path)

    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    parameters = EvolutionStrategiesParameters(
        learning_rate=0.1,
        noise_std=1.0,
        mirrored_sampling_enabled=True,
        fitness_shaping_enabled=True,
        pop_size=20,
        n_iteration=1000,
        stop_criterion=np.Inf,
        seed=optimizer_seed)

    optimizer = EvolutionStrategiesOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
        optimizee_fitness_weights=(-1.,),
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

    return traj.v_storage_service.filename, traj.v_name, paths


def main():
    filename, trajname, paths = run_experiment()
    logger.info("Plotting now")


if __name__ == '__main__':
    main()
