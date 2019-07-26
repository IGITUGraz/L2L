"""
This file is a typical example of a script used to run a L2L experiment. Read the comments in the file for more
explanations
"""

import logging.config
import os

from l2l.utils.environment import Environment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.optimizee import Optimizee
from l2l.optimizers.optimizer import Optimizer, OptimizerParameters
from l2l.paths import Paths

from l2l.utils import JUBE_runner as jube

# We first setup the logger and read the logging config which controls the verbosity and destination of the logs from
# various parts of the code.
logger = logging.getLogger('bin.l2l-optimizee-optimizer')


def main():
    # TODO when using the template: Give some *meaningful* name here
    name = 'L2L'

    # TODO when using the template: make a path.conf file and write the root path there
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

    # Load the logging config which tells us where and what to log (loglevel, destination)

    print("All output logs can be found in directory ", paths.logs_path)

    # Create an environment that handles running our simulation
    # This initializes an environment. This environment is based on the Pypet implementation.
    # Uncomment 'freeze_input', 'multipproc', 'use_scoop' and 'wrap_mode' lines to disable running the experiment
    # across cores and nodes.
    env = Environment(trajectory=name, filename=paths.output_dir_path, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      freeze_input=False,
                      multiproc=True,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      )
    create_shared_logger_data(logger_names=['bin', 'optimizers'],
                              log_levels=['INFO', 'INFO'],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment.
    traj = env.trajectory

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")
    # Execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec", "python " +
                                  os.path.join(paths.simulation_path, "run_files/run_optimizee.py"))
    # Paths
    traj.f_add_parameter_to_group("JUBE_params", "paths", paths)

    # Scheduler parameters
    # Name of the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")
    # Command to submit jobs to the schedulers
    traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")
    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "1")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "00:01:00")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "1")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address", "me@myemail.com")
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    # JUBE parameters for multiprocessing.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "1")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "1")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "1")

    ## Innerloop simulator
    # TODO when using the template: Change the optimizee to the appropriate Optimizee class
    optimizee = Optimizee(traj)

    # Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, paths.simulation_path)

    ## Outerloop optimizer initialization
    # TODO when using the template: Change the optimizer to the appropriate Optimizer class
    # and use the right value for optimizee_fitness_weights. Length is the number of dimensions of fitness, and
    # negative value implies minimization and vice versa
    optimizer_parameters = OptimizerParameters()
    optimizer = Optimizer(traj, optimizee.create_individual, (1.0,), optimizer_parameters)

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
