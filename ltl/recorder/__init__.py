import argparse
import datetime
import os
import yaml
import warnings
import ltl

from git import Repo
from jinja2 import Environment, FileSystemLoader


class Recorder:
    """
    Implements a recorder that records simulation of one run of some optimization process.
    After creating instance use start() method checks if the repository is dirty and commits
    need to be made. end() ends the recording session and produces an .md table and a plot of
    the fitness progress.

    :param  ~pypet.trajectory.Trajectory trajectory: PyPet trajectory
    :param optimizee_id:
      One of the ids given to the Optimizee. Currently applies only to the benchmark functions
    :param str optimizee_name:
      Name of the optimizee
    :param optimizee_parameters:
      Optimizee parameters (:obj:`dict` or :obj:`namedtuple`)
    :param optimizer_name:
      Name of the Optimizer 
    :param ~collections.namedtuple optimizer_parameters:
      Optimizer parameters as named tuple.
    """
    def __init__(self, trajectory,
                 optimizee_name, optimizee_parameters, optimizer_name, optimizer_parameters):
        if optimizee_parameters is None:
            warnings.warn("optimizee_parameters is set to None.")
        if optimizer_parameters is None:
            warnings.warn("optimizer_parameters is set to None.")
        self.record_flag, self.username, self.description = self._process_args()
        self.trajectory = trajectory
        self.optimizee_name = optimizee_name
        self.optimizee_parameters = optimizee_parameters
        self.optimizer_name = optimizer_name
        self.optimizer_parameters = optimizer_parameters
        self.n_iteration = None
        self.optima_found = None
        self.actual_optima = None
        self.runtime = None
        self.git_commit_id = None
        self.git_commit_url = None
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
        self.git_commit_url = (repo.remotes.origin.url).replace(".git", "/commit/") + repo.head.commit.hexsha

    def end(self):
        """
        Ends the recording session by creating .md table with simulation details.
        Table is then saved to the current directory.
        """
        if not self.record_flag:
            return

        traj = self.trajectory
        self.optima_found = traj.res.final_fitness
        self.individual_found = traj.res.final_individual
        self.n_iteration = traj.res.n_iteration

        self.end_time = datetime.datetime.now()
        self.runtime = self.end_time - self.start_time
        self._parse_md()

    def _parse_md(self):
        fname = "result_details.md"
        abs_ltl_path = os.path.abspath(ltl.__file__).replace("/__init__.py","")
        env = Environment(loader=FileSystemLoader(abs_ltl_path + '/recorder/templates'))
        dir_name = "results/"
        dir_name += self.optimizer_name + "-"
        dir_name += self.optimizee_name + "-"
        end_time_parsed = str(self.end_time.strftime("%Y-%m-%d %H:%M:%S")).replace(":","-")
        end_time_parsed = end_time_parsed.replace(" ","--")
        dir_name += end_time_parsed + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(dir_name + 'optimizee_parameters.yml', 'w') as ofile:
            yaml.dump(dict(self.optimizee_parameters), ofile, default_flow_style=None)
        self.optimizee_parameters = 'optimizee_parameters.yml'

        with open(dir_name + 'optimizer_parameters.yml', 'w') as ofile:
            yaml.dump(dict(self.optimizer_parameters), ofile, default_flow_style=None)
        self.optimizer_parameters = 'optimizer_parameters.yml'
        with open(dir_name + 'optima_coordinates.yml', 'w') as ofile:
            print(self.individual_found)
            yaml.dump(self.individual_found, ofile, default_flow_style=None)
        self.individual_found = 'optima_coordinates.yml'

        context = {'cur_date_': self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                   'username_': self.username,
                   'description_': self.description,
                   'optimizee_name_': self.optimizee_name,
                   'optimizee_parameters_': self.optimizee_parameters,
                   'optimizer_name_': self.optimizer_name,
                   'optimizer_parameters_': self.optimizer_parameters,
                   'n_iteration_': self.n_iteration,
                   'optima_found_': self.optima_found,
                   'individual_found_': self.individual_found,
                   'actual_optima_': self.actual_optima,
                   'runtime_': self.runtime,
                   'git_commit_id': self.git_commit_id,
                   'git_commit_url_': self.git_commit_url,
                   'hasattr': hasattr,
                   'isinstance': isinstance,
                   'str': str}
        template = env.get_template("md-template.jinja")
        with open(dir_name + fname, 'w') as f:
            rendered_data = template.render(context)
            f.write(rendered_data)
        print("Recorder details have been written to " + f.name)

    def _process_args(self):
        parser = argparse.ArgumentParser(description="Main parser.")
        parser.add_argument('--record-experiment', dest='record_flag', action='store_true')
        parser.add_argument('--username', dest="username", type=str, required=False)
        parser.add_argument('--description', dest="description", type=str, required=False)
        args = parser.parse_args()

        if args.record_flag and (args.username is None or args.description is None):
            raise Exception("--record-experiment requires --name and --description")
        name = args.username
        description = args.description
        record_flag = args.record_flag
        return record_flag, name, description
