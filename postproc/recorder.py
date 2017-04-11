from git import Repo
import datetime
from jinja2 import Environment, FileSystemLoader
import argparse
import os


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
    :param optimizee_description:
      Optimizee description
    :param optimizer_name:
      Optimizer name
    :param optimizer_parameters:
      Optimizer parameters as named tuple.
    """
    def __init__(self, trajectory,
                 optimizee_id, optimizee_name, optimizee_parameters, optimizer_name, optimizer_parameters):
        self.record_flag, self.username, self.description = self._process_args()
        self.trajectory = trajectory
        self.optimizee_id = optimizee_id
        self.optimizee_name = optimizee_name
        self.optimizee_parameters = optimizee_parameters
        self.optimizer_name = optimizer_name
        self.optimizer_parameters = optimizer_parameters
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
        if not self.record_flag:
            return
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
        if not self.record_flag:
            return

        traj = self.trajectory
        self.optima_found = traj.res.final_fitness
        self.n_iteration = traj.res.n_iteration

        self.end_time = datetime.datetime.now()
        self.runtime = self.end_time - self.start_time
        self._parse_md()

    def _parse_md(self):
        fname = "result_details.md"
        env = Environment(loader=FileSystemLoader('postproc/templates'))

        context = {'cur_date_': self.end_time,
                   'username_': self.username,
                   'description_': self.description,
                   'optimizee_name_': self.optimizee_name,
                   'optimizee_id_': self.optimizee_id,
                   'optimizee_parameters_': self.optimizee_parameters,
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
            print("Recorder details have been written to " + os.curdir + "/" + f.name)

    def _process_args(self):
        parser = argparse.ArgumentParser(description="Main parser.")
        parser.add_argument('--record_experiment', dest='record_flag', action='store_true')
        parser.add_argument('--username', dest="username", type=str, required=False)
        parser.add_argument('--description', dest="description", type=str, required=False)
        args = parser.parse_args()

        if args.record_flag and (args.username is None or args.description is None):
            raise Exception("--record_experiment requires --name and --description")
        name = args.username
        description = args.description
        record_flag = args.record_flag
        return record_flag, name, description
