import logging.config
import os

import yaml
from utils.environment import  Environment
import numpy as np

from ltl.optimizees.functions import tools as function_tools
from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from ltl.paths import Paths
from ltl.recorder import Recorder

import utils.JUBE_runner as jube

logger = logging.getLogger('ltl-fun-ga')


def main():
    name = 'LTL-FUN-GA'
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

    with open("bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    print("All output can be found in file ", log_output_file)
    print("Change the values in logging.yaml to control log level and destination")
    print("e.g. change the handler to console for the loggers you're interesting in to get output to stdout")

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )

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
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    # Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, root_dir_path)

    ## Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5,
                                            MUTPB=0.3, NGEN=100, indpb=0.02,
                                            tournsize=15, matepar=0.5,
                                            mutpar=1
                                            )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters)

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
    optimizer.end()
    recorder.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
