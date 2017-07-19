import os

import numpy as np
import logging

from pypet import Environment, pypetconstants
from ltl.optimizees.functions import FunctionGeneratorOptimizee, BenchmarkedFunctions
from ltl.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from ltl.optimizers.crossentropy.distribution import NoisyGaussian
from ltl.dataprocessing import get_skeleton_traj, get_var_from_generations, get_var_from_runs
from ltl.paths import Paths

from ltl.matplotlib_ import plt

from ltl.logging_tools import create_shared_logger_data

def get_root_dir():
    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    return root_dir_path


def run_experiment():
    name = 'LTL-FUN-CE-dataproc'
    root_dir = get_root_dir()
    paths = Paths(root_dir_name=name, root_dir_path=root_dir, param_dict={'run_no': 'test'})

    env = Environment(trajectory=name, filename=paths.results_path, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      freeze_input=True,
                      multiproc=True,
                      use_scoop=True,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      automatic_storing=True,
                      logger_names=('bin.ltl-ce', 'ltl.dataprocessing'),
                      log_levels=(logging.INFO, logging.INFO, logging.INFO, logging.INFO),
                      log_stdout=False,
                      )
    create_shared_logger_data(logger_names=['bin', 'optimizers'],
                              log_levels=['INFO', 'INFO'],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)


    traj = env.trajectory

    function_name = 'Rosenbrock2d'
    (benchmark_name, benchmark_function), benchmark_parameters = \
        BenchmarkedFunctions().get_function_by_name(function_name, noise=True)

    # NOTE: Innerloop Simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=106)

    # NOTE: Outerloop Simulator
    parameters = CrossEntropyParameters(pop_size=50, rho=0.2, smoothing=0.0, temp_decay=0, n_iteration=180,
                                        distribution=NoisyGaussian(additive_noise=1,
                                                                   noise_decay=0.95),
                                        stop_criterion=np.inf, seed=102)

    optimizer = CrossEntropyOptimizer(traj=traj,
                                      optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(-0.1,),
                                      optimizee_bounding_func=optimizee.bounding_func,
                                      parameters=parameters)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Outerloop optimizer end
    optimizer.end(traj)

    env.disable_logging()
    return traj.v_storage_service.filename, traj.v_name


def process_results(filename, trajname):
    traj = get_skeleton_traj(filename, trajname)

    fitness_list, run_id_list = get_var_from_runs(traj, 'results.fitness', with_ids=True, status_interval=200)
    algorithm_params_list = get_var_from_generations(traj, 'algorithm_params')

    pop_size_list = [params_dict['pop_size'] for params_dict in algorithm_params_list]

    fitness_list = np.array(fitness_list)
    run_id_list = np.array(run_id_list)
    pop_size_list = np.array(pop_size_list)

    pop_size_list_cumsum = np.cumsum(pop_size_list)
    gen_no_list = np.zeros_like(run_id_list)  # gen_no_list[i] = gen no of ith run
    gen_no_list[pop_size_list_cumsum[:-1]] = 1
    gen_no_list = np.cumsum(gen_no_list)

    fitness_by_generation_list = np.split(fitness_list, pop_size_list_cumsum)[:-1]
    worst_fitness_in_generation_list = np.array([np.max(fitness_array) for fitness_array in fitness_by_generation_list])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(gen_no_list, fitness_list, c='b', s=4, marker='.', label='run fitnesses')
    ax.scatter(np.arange(len(pop_size_list)), worst_fitness_in_generation_list,
               c='r', s=4, marker='.', label='worst fitnesses')

    ax.set_xlabel('Generation Number')
    ax.set_ylabel('Fitness Value')
    ax.legend()
    fig.savefig('ce-processing-results.png', dpi=300)
    plt.show()


def main():
    filename, trajname = run_experiment()
    process_results(filename, trajname)


if __name__ == '__main__':
    main()
