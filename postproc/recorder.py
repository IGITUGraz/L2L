from git import Repo


class Recorder:
    def __init__(self):
        self.optimizee_name = "Not given"
        self.optimizee_params = None
        self.optimizer_name = "Not given"
        self.optimizer_params = None
        self.number_of_iterations = 0
        self.optima_found = 0
        self.actual_optima = 0
        self.runtime = 0
        self.git_commit_id = 0
        self.date = "Not given"
    def start(self):
        repo = Repo()
        if repo.bare:
            raise Exception("Not a git repository (or any of the parent directories): .git")
        if repo.is_dirty():
            raise Exception('Commit your changes first.(use "git add" and then "git commit")')
        self.git_commit_id = repo.head.commit.hexsha
        print(self.git_commit_id)

    def end(self):
        pass