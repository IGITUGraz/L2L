from git import Repo
import datetime
from pypet import Environment
import numpy as np
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import argparse



class Recorder:
    """
    Implements a recorder that records simulation of one run of some optimization process.
    After creating instance use start() method checks if the repository is dirty and commits
    need to be made. end() ends the recording session and produces an .md table and a plot of
    the fitness progress.

    :param description:
      One-line description of the run
    :param ~pypet.environment.Environment:
      Environment that was used for recording data during the simulation
    :param optimizee_name:
      Optimizee name
    :param optimizee_parameters:
      Optimizee parameters as named tuple. Set None, if no parameters are required
    :param optimizer_name:
      Optimizer name
    :param optimizer_parameters:
      Optimizer parameters as named tuple.
    """
    def __init__(self,
                 environment, optimizee_name, optimizee_parameters, optimizer_name, optimizer_parameters):
        self.record_flag, self.username, self.description = self.__process_args__()
        self.environment = environment
        self.optimizee_name = optimizee_name
        if optimizee_parameters is None:
            self.optimizee_parameters = {}
        else:
            self.optimizee_parameters = optimizee_parameters._asdict()
        self.optimizer_name = optimizer_name
        self.optimizer_parameters = optimizer_parameters._asdict()
        self.n_iteration = None
        self.optima_found = None
        self.actual_optima = None
        self.runtime = None
        self.git_commit_id = None
        self.start_time = None
        self.end_time = None

    def start(self):
        """
        Starts the recording session by checking that repository is not dirty.
        """
        repo = Repo()
        if repo.bare:
            raise Exception("Not a git repository (or any of the parent directories): .git")
        if repo.is_dirty():
            raise Exception('Commit your changes first.(use "git add" and then "git commit")')
        self.start_time = datetime.datetime.now()
        self.git_commit_id = repo.head.commit.hexsha

    def end(self):
        """
        Ends the recording session by creating .md table with simulation details.
        Table is then saved to the current directory.
        """
        traj = self.environment.trajectory
        self.optima_found = traj.res.final_fitness
        self.n_iteration = traj.res.n_iteration

        self.end_time = datetime.datetime.now()
        self.runtime = self.end_time - self.start_time
        self.__parse_md__()

    def __parse_md__(self):
        fname = "result_details.md"
        env = Environment(
             loader=FileSystemLoader('postproc/templates'))

        context = {'cur_date_': self.end_time,
                   'username_': self.username,
                   'description_': self.description,
                   'optimizee_name_': self.optimizee_name,
                   'optimizee_params_': self.optimizee_parameters,
                   'optimizer_name_': self.optimizer_name,
                   'optimizer_params_': self.optimizer_parameters,
                   'n_iteration_': self.n_iteration,
                   'optima_found_': self.optima_found,
                   'actual_optima_': self.actual_optima,
                   'runtime_': self.runtime,
                   'git_commit_id': self.git_commit_id}
        template = env.get_template("md-template.jinja")
        with open(fname, 'w') as f:
            rendered_data = template.render(context)
            f.write(rendered_data)
    def __process_args__(self):
        record_flag = False
        name = False
        description = ""
        parser = argparse.ArgumentParser(description="Main parser.")
        parser.add_argument('--record_experiment', dest='record_flag', action='store_true')
        parser.add_argument('--username', dest="username", type=str, required=False)
        parser.add_argument('--description', dest="description", type=str, required=False)
        args = parser.parse_args()

        if args.record_flag and (args.username is None or args.description is None):
            raise Exception("--record_experiment requires --name and --description")
        name = args.username
        description = args.description
        return record_flag, name, description