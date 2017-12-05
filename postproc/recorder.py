from __future__ import with_statement
from __future__ import absolute_import
from git import Repo
import datetime
from jinja2 import Environment, FileSystemLoader
import argparse
import os
from io import open


class Recorder(object):
    u"""
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
        u"""
        Starts the recording session by checking that repository is not dirty.
        """
        if not self.record_flag:
            return
        repo = Repo()
        if repo.bare:
            raise Exception(u"Not a git repository (or any of the parent directories): .git")
        if repo.is_dirty():
            raise Exception(u'Commit your changes first.(use "git add" and then "git commit")')
        self.start_time = datetime.datetime.now()
        self.git_commit_id = repo.head.commit.hexsha

    def end(self):
        u"""
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
        fname = u"result_details.md"
        env = Environment(loader=FileSystemLoader(u'postproc/templates'))

        context = {u'cur_date_': self.end_time,
                   u'username_': self.username,
                   u'description_': self.description,
                   u'optimizee_name_': self.optimizee_name,
                   u'optimizee_id_': self.optimizee_id,
                   u'optimizee_parameters_': self.optimizee_parameters,
                   u'optimizer_name_': self.optimizer_name,
                   u'optimizer_params_': self.optimizer_parameters,
                   u'n_iteration_': self.n_iteration,
                   u'optima_found_': self.optima_found,
                   u'actual_optima_': self.actual_optima,
                   u'runtime_': self.runtime,
                   u'git_commit_id': self.git_commit_id,
                   u'hasattr': hasattr,
                   u'str': unicode}
        template = env.get_template(u"md-template.jinja")
        with open(fname, u'w') as f:
            rendered_data = template.render(context)
            f.write(rendered_data)
        print u"Recorder details have been written to " + os.curdir + u"/" + f.name

    def _process_args(self):
        parser = argparse.ArgumentParser(description=u"Main parser.")
        parser.add_argument(u'--record_experiment', dest=u'record_flag', action=u'store_true')
        parser.add_argument(u'--username', dest=u"username", type=unicode, required=False)
        parser.add_argument(u'--description', dest=u"description", type=unicode, required=False)
        args = parser.parse_args()

        if args.record_flag and (args.username is None or args.description is None):
            raise Exception(u"--record_experiment requires --name and --description")
        name = args.username
        description = args.description
        record_flag = args.record_flag
        return record_flag, name, description
