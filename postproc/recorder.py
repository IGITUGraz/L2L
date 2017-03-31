from git import Repo
import datetime

class Recorder:
    def __init__(self, optimizee_name, optimizee_parameters, optimizer_name, optimizer_parameters):
        self.optimizee_name = optimizee_name
        self.optimizee_parameters = optimizee_parameters
        self.optimizer_name = optimizer_name
        self.optimizer_parameters = optimizer_parameters
        self.number_of_iterations = 0
        self.optima_found = 0
        self.actual_optima = None
        self.runtime = 0
        self.git_commit_id = 0
        self.start_time = "Not given"
        self.end_time = "Not given"

    def start(self):
        repo = Repo()
        if repo.bare:
            raise Exception("Not a git repository (or any of the parent directories): .git")
        if repo.is_dirty():
            raise Exception('Commit your changes first.(use "git add" and then "git commit")')
        self.start_time = datetime.datetime.now()
        self.git_commit_id = repo.head.commit.hexsha


    def end(self):
        pass