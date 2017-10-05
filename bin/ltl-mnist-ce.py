import logging.config

import numpy as np
from pypet import Environment

from ltl.dataprocessing import get_skeleton_traj, get_var_from_generations
from ltl.logging_tools import create_shared_logger_data, configure_loggers
from ltl.optimizees.mnist.optimizee import MNISTOptimizeeParameters, MNISTOptimizee
from ltl.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
from ltl.optimizers.crossentropy.distribution import NoisyGaussian
from ltl.paths import Paths
from ltl.recorder import Recorder

logger = logging.getLogger('bin.ltl-mnist-es')


def run_experiment():
    name = 'LTL-MNIST-CE'
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
    # This initializes a PyPet environment
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

    optimizee_seed = 200

    optimizee_parameters = MNISTOptimizeeParameters(n_hidden=10, seed=optimizee_seed, use_small_mnist=True)
    ## Innerloop simulator
    optimizee = MNISTOptimizee(traj, optimizee_parameters)

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

    # Add Recorder
    recorder = Recorder(
        trajectory=traj,
        optimizee_name=optimizee.__class__.__name__,
        optimizee_parameters=optimizee_parameters,
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

    return traj.v_storage_service.filename, traj.v_name, paths


def process_results(filename, trajname, paths):
    from ltl.matplotlib_ import plt

    traj = get_skeleton_traj(filename, trajname)

    # fitness_list, run_id_list = get_var_from_runs(traj, 'results.fitness', with_ids=True, status_interval=200)
    algorithm_params_list = get_var_from_generations(traj, 'algorithm_params')

    best_fitness_list = [x['best_fitness_in_run'] for x in algorithm_params_list]
    average_fitness_list = [x['average_fitness_in_run'] for x in algorithm_params_list]
    generation_list = [x['generation'] for x in algorithm_params_list]
    # current_individual_fitness = [x['current_individual_fitness'] for x in algorithm_params_list]

    # pop_size_list = [params_dict['pop_size'] for params_dict in algorithm_params_list]

    # pop_size_list_cumsum = np.cumsum(pop_size_list)
    # gen_no_list = np.zeros_like(run_id_list)  # gen_no_list[i] = gen no of ith run
    # gen_no_list[pop_size_list_cumsum[:-1]] = 1
    # gen_no_list = np.cumsum(gen_no_list)
    #
    # fitness_list = np.array(fitness_list)

    fig, ax = plt.subplots(figsize=(15, 7))
    ngen = 5000
    ax.plot(generation_list[:ngen], best_fitness_list[:ngen], label='best fitness')
    ax.plot(generation_list[:ngen], average_fitness_list[:ngen], label='average fitness')
    ax.annotate('%.2f%%' % (best_fitness_list[1000] * 100), xy=(1000, best_fitness_list[1000]), xytext=(1000, 1.05),
                arrowprops=dict(facecolor='black', shrink=0.05), xycoords='data',
                )
    ax.annotate('%.2f%%' % (best_fitness_list[2000] * 100), xy=(2000, best_fitness_list[2000]), xytext=(2000, 1.05),
                arrowprops=dict(facecolor='black', shrink=0.05), xycoords='data',
                )
    ax.annotate('%.2f%%' % (best_fitness_list[3000] * 100), xy=(3000, best_fitness_list[3000]), xytext=(3000, 1.05),
                arrowprops=dict(facecolor='black', shrink=0.05), xycoords='data',
                )
    ax.annotate('%.2f%%' % (best_fitness_list[4000] * 100), xy=(4000, best_fitness_list[4000]), xytext=(4000, 1.05),
                arrowprops=dict(facecolor='black', shrink=0.05), xycoords='data',
                )
    ax.annotate('%.2f%%' % (best_fitness_list[5000] * 100), xy=(5000, best_fitness_list[5000]), xytext=(5000, 1.05),
                arrowprops=dict(facecolor='black', shrink=0.05), xycoords='data',
                )
    ax.set(ylim=[0, 1.1], xlabel='Generation', ylabel="Performance")
    ax.legend()
    ax.set_title("Testing", fontsize='small')
    fig.savefig(paths.get_fpath('mnist-small-ce-performance', '.png'))


def main():
    filename, trajname, paths = run_experiment()
    logger.info("Plotting now")
    process_results(filename, trajname, paths)


if __name__ == '__main__':
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
