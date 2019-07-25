import logging.config
import os

import numpy as np
from l2l.utils.environment import Environment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.mnist.optimizee import MNISTOptimizeeParameters, MNISTOptimizee
from l2l.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.paths import Paths

from l2l.utils import JUBE_runner as jube

logger = logging.getLogger('bin.l2l-mnist-es')


def run_experiment():
    name = 'L2L-MNIST-CE'
    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("You have not set the root path to store your results."
                                " Write the path to a path.conf text file in the bin directory"
                                " before running the simulation")

    trajectory_name = 'small-mnist-full-monty'

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
    # Execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec", "python3 " +
                                  os.path.join(paths.simulation_path, "run_files/run_optimizee.py"))
    # Paths
    traj.f_add_parameter_to_group("JUBE_params", "paths", paths)


    optimizee_seed = 200

    optimizee_parameters = MNISTOptimizeeParameters(n_hidden=10, seed=optimizee_seed, use_small_mnist=True)
    ## Innerloop simulator
    optimizee = MNISTOptimizee(traj, optimizee_parameters)

    # Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, paths.simulation_path)

    logger.info("Optimizee parameters: %s", optimizee_parameters)

    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = CrossEntropyParameters(pop_size=40, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=5000,
                                                  distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                                  stop_criterion=np.inf, seed=optimizer_seed)

    logger.info("Optimizer parameters: %s", optimizer_parameters)

    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(1.,),
                                      parameters=optimizer_parameters,
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
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
