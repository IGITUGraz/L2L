import logging.config

import numpy as np
from utils.environment import Environment

from l2l import dict_to_list
from l2l.dataprocessing import get_skeleton_traj, get_var_from_runs, get_var_from_generations
from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from l2l.paths import Paths

import utils.JUBE_runner as jube

logger = logging.getLogger('bin.l2l-fun-es')


def run_experiment():
    name = 'LTL-FUN-ES'
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
    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # Name of the scheduler
    # traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")
    # Command to submit jobs to the schedulers
    traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")
    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "1")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "00:01:00")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "1")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "1")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "1")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address", "s.diaz@fz-juelich.de")
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    # JUBE parameters for multiprocessing. Relevant even without scheduler.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "1")
    # The execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec", "mpirun python3 " + root_dir_path +
                                  "/run_files/run_optimizee.py")
    # Ready file for a generation
    traj.f_add_parameter_to_group("JUBE_params", "ready_file", root_dir_path + "/readyfiles/ready_w_")
    # Path where the job will be executed
    traj.f_add_parameter_to_group("JUBE_params", "work_path", root_dir_path)


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
    jube.prepare_optimizee(optimizee, root_dir_path)

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


def process_results(filename, trajname, paths):
    # NOTE: This is written specifically for benchmark function number 14

    from l2l.matplotlib_ import plt

    traj = get_skeleton_traj(filename, trajname)

    fitness_list, run_id_list = get_var_from_runs(traj, 'results.fitness', with_ids=True, status_interval=200)
    algorithm_params_list = get_var_from_generations(traj, 'algorithm_params')

    best_fitness_list = [x['best_fitness_in_run'] for x in algorithm_params_list]
    average_fitness_list = [x['average_fitness_in_run'] for x in algorithm_params_list]
    generation_list = [x['generation'] for x in algorithm_params_list]
    current_individual_fitness = [x['current_individual_fitness'] for x in algorithm_params_list]

    pop_size_list = [params_dict['pop_size'] for params_dict in algorithm_params_list]

    pop_size_list_cumsum = np.cumsum(pop_size_list)
    gen_no_list = np.zeros_like(run_id_list)  # gen_no_list[i] = gen no of ith run
    gen_no_list[pop_size_list_cumsum[:-1]] = 1
    gen_no_list = np.cumsum(gen_no_list)

    # NOTE: This is because the value of the fitness weight is negative!!
    fitness_list = -1 * np.array(fitness_list)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(gen_no_list, fitness_list, '.', label='fitness distribution')
    ax.plot(generation_list, current_individual_fitness, label='current individual fitness')
    ax.plot(generation_list, best_fitness_list, label='best fitness')
    ax.plot(generation_list, average_fitness_list, label='average fitness')
    ax.legend()
    ax.set_title("Testing", fontsize='small')
    fig.savefig(paths.get_fpath('fitness-v-generation', 'png'))

    individual_list, run_id_list = get_var_from_runs(traj, 'results.individual', with_ids=True, status_interval=200)

    individual_list_arr = [dict_to_list(ind) for ind in individual_list]

    xs = [p[0] for p in individual_list_arr]
    ys = [p[1] for p in individual_list_arr]

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(xs, ys)
    # NOTE: -5, 5 'coz this is the range for benchmark function number 14
    ax.set(xlim=(-5, 5), ylim=(-5, 5))

    fig.savefig(paths.get_fpath('es-explored-points', 'png'))

    logger.info("Plots are in %s", paths.results_path)


def main():
    filename, trajname, paths = run_experiment()
    logger.info("Plotting now")
    process_results(filename, trajname, paths)


if __name__ == '__main__':
    main()
