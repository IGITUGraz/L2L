import logging.config
import os

from l2l.utils.environment import Environment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.paths import Paths
import l2l.utils.JUBE_runner as jube


class Experiment(object):
    def __init__(self, root_dir_path):
        """
        Prepares and starts the l2l simulation.

        For an example see `L2L/bin/l2l-template.py`

        :param root_dir_path: str, Path to the results folder. Accepts relative
        paths. Will check if the folder exists and create if not.
        """
        self.root_dir_path = os.path.abspath(root_dir_path)
        self.logger = logging.getLogger('bin.l2l')
        self.paths = None
        self.env = None
        self.traj = None
        self.optimizee = None
        self.optimizer = None

    def prepare_experiment(self, **kwargs):
        """
        Prepare the experiment by creating the enviroment and
        :param kwargs: optional dictionary, contains
            - name: str, name of the run, Default: L2L-run
            - trajectory_name: str, name of the trajectory, Default: trajectory
            - log_stdout: bool, if stdout should be sent to logs, Default:False
            - jube_parameter: dict, User specified parameter for jube.
                See notes section for default jube parameter
            - multiprocessing, bool, enable multiprocessing, Default: False
        :return traj, trajectory object
        :return all_jube_params, dict, a dictionary with all parameters for jube
            given by the user and default ones

        :notes
           Default JUBE parameters are:
            - scheduler: None,
            - submit_cmd: sbatch,
            - job_file: job.run,
            - nodes: 1,
            - walltime: 01:00:00,
            - ppn: 1,
            - cpu_pp: 1,
            - threads_pp: 4,
            - mail_mode: ALL,
            - err_file: stderr,
            - out_file: stdout,
            - tasks_per_job: 1,
            - exec: python3 + self.paths.simulation_path +
                "run_files/run_optimizee.py"
            - ready_file: self.paths.root_dir_path + "ready_files/ready_w_"
            - work_path: self.paths.root_dir_path,
            - paths_obj: self.paths
        """
        name = kwargs.get('name', 'L2L-run')
        if not os.path.isdir(self.root_dir_path):
            os.mkdir(os.path.abspath(self.root_dir_path))
            print('Created a folder at {}'.format(self.root_dir_path))

        trajectory_name = kwargs.get('trajectory_name', 'trajectory')

        self.paths = Paths(name, {},
                           root_dir_path=self.root_dir_path,
                           suffix="-" + trajectory_name)

        print("All output logs can be found in directory ",
              self.paths.logs_path)

        # Create an environment that handles running our simulation
        # This initializes an environment
        self.env = Environment(
            trajectory=trajectory_name,
            filename=self.paths.output_dir_path,
            file_title='{} data'.format(name),
            comment='{} data'.format(name),
            add_time=True,
            automatic_storing=True,
            log_stdout=kwargs.get('log_stdout', False),  # Sends stdout to logs
            multiprocessing=kwargs.get('multiprocessing', True)
        )

        create_shared_logger_data(
            logger_names=['bin', 'optimizers'],
            log_levels=['INFO', 'INFO'],
            log_to_consoles=[True, True],
            sim_name=name,
            log_directory=self.paths.logs_path)
        configure_loggers()

        # Get the trajectory from the environment
        self.traj = self.env.trajectory

        # Set JUBE params
        default_jube_params = {
            # "scheduler": "None",
            "submit_cmd": "sbatch",
            "job_file": "job.run",
            "nodes": "1",
            "walltime": "01:00:00",
            "ppn": "1",
            "cpu_pp": "1",
            "threads_pp": "4",
            "mail_mode": "ALL",
            "err_file": "stderr",
            "out_file": "stdout",
            "tasks_per_job": "1",
            "exec": "python " + os.path.join(self.paths.simulation_path,
                                              "run_files/run_optimizee.py"),
            "ready_file": os.path.join(self.paths.root_dir_path,
                                       "ready_files/ready_w_"),
            "work_path": self.paths.root_dir_path,
            "paths_obj": self.paths,
        }
        # Will contain all jube parameters
        all_jube_params = {}
        self.traj.f_add_parameter_group("JUBE_params",
                                        "Contains JUBE parameters")
        # Go through the parameter dictionary and add to the trajectory
        if kwargs.get('jube_parameter'):
            for k, v in kwargs['jube_parameter'].items():
                if k == "exec":
                    val = v + " " + os.path.join(self.paths.simulation_path,
                                                 "run_files/run_optimizee.py")
                    self.traj.f_add_parameter_to_group("JUBE_params", k, val)
                    all_jube_params[k] = val
                else:
                    self.traj.f_add_parameter_to_group("JUBE_params", k, v)
                    all_jube_params[k] = v
        # Default parameter are added if they are not already set by the user
        for k, v in default_jube_params.items():
            if k not in kwargs.get('jube_parameter').keys():
                self.traj.f_add_parameter_to_group("JUBE_params", k, v)
                all_jube_params[k] = v
        print('JUBE parameters used: {}'.format(all_jube_params))
        return self.traj, all_jube_params

    def run_experiment(self, optimizee, optimizee_parameters, optimizer,
                       optimizer_parameters):
        """
        Runs the simulation with all parameter combinations

        Optimizee and optimizer object are required as well as their parameters
        as namedtuples.

        :param optimizee: optimizee object
        :param optimizee_parameters: Namedtuple, parameters of the optimizee
        :param optimizer: optimizer object
        :param optimizer_parameters: Namedtuple, parameters of the optimizer
        """
        self.optimizee = optimizee
        self.optimizer = optimizer
        self.optimizer = optimizer
        self.logger.info("Optimizee parameters: %s", optimizee_parameters)
        self.logger.info("Optimizer parameters: %s", optimizer_parameters)
        jube.prepare_optimizee(optimizee, self.paths.simulation_path)
        # Add post processing
        self.env.add_postprocessing(optimizer.post_process)
        # Run the simulation
        self.env.run(optimizee.simulate)

    def end_experiment(self, optimizer):
        """
        Ends the experiment and disables the logging

        :param optimizer: optimizer object
        :return traj, trajectory object
        :return path, Path object
        """
        # Outer-loop optimizer end
        optimizer.end(self.traj)
        # Finally disable logging and close all log-files
        self.env.disable_logging()
        return self.traj, self.paths
