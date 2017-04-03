from git import Repo
import datetime
from pypet import Environment
import numpy as np
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self, description,
                 environment, optimizee_name, optimizee_parameters, optimizer_name, optimizer_parameters):
        self.environment = environment
        self.description = description
        # self.trajectory_name = environment._trajectory_name
        # self.traj_file = traj_file
        self.optimizee_name = optimizee_name
        if optimizee_parameters is None:
            self.optimizee_parameters = {}
        else:
            self.optimizee_parameters = optimizee_parameters._asdict()
        self.optimizer_name = optimizer_name
        self.optimizer_parameters = optimizer_parameters._asdict()
        self.n_iteration = optimizer_parameters.n_iteration
        self.optima_found = None
        self.actual_optima = 'Not known'
        self.runtime = None
        self.git_commit_id = None
        self.start_time = "Not given"
        self.end_time = "Not given"
        self.fitness_list = [-np.inf]

    def start(self):
        repo = Repo()
        # if repo.bare:
        #     raise Exception("Not a git repository (or any of the parent directories): .git")
        # if repo.is_dirty():
        #     raise Exception('Commit your changes first.(use "git add" and then "git commit")')
        self.start_time = datetime.datetime.now()
        self.git_commit_id = repo.head.commit.hexsha


    def end(self):
        # self.environment = Environment()
        example_run = 0
        # traj = self.environment.trajectory
        # traj = Trajectory(self.trajectory_name, add_time=False)

        # Let's load the trajectory from the file
        # Only load the parameters, we will load the results on the fly as we need them
        # traj.f_load(filename=self.traj_file, load_parameters=2,
        #     load_results=2, load_derived_parameters=0)
        # traj.v_idx = example_run

        # We'll simply use auto loading so all data will be loaded when needed.
        # traj.v_auto_load = True


        traj = self.environment.trajectory
        run_set = traj.res.run_set_00000
        for i in reversed(range(self.n_iteration)):
            individual = run_set[i]
            if individual['individual'].accepted and self.optima_found is None:
                self.optima_found = individual['fitness']
            if individual['individual'].accepted:
                self.fitness_list.append(individual['fitness'])
            else:
                self.fitness_list.append(self.fitness_list[-1])
        self.fitness_list = self.fitness_list[1:]

        self.end_time = datetime.datetime.now()
        self.runtime = self.end_time - self.start_time
        self.parse_md()
        self.plot_fitness()

        # fitnesses = []
        # for res in traj.res:
        #     result_individual = res.fitness
        #     print(res.fitness)
        #     fitnesses.append(result_individual)
        # max_fitness = np.asarray(fitnesses).max()
        # # for res in results_individuals
        # print("Result max: ", traj.res[0].fitness)

    def parse_md(self):
        fname = "result_details.md"
        env = Environment(
             loader=FileSystemLoader('postproc/templates'))

        context = {'cur_date_' : self.end_time,
                   'description_' : self.description,
                   'optimizee_name_' : self.optimizee_name,
                   'optimizee_params_' : self.optimizee_parameters,
                   'optimizer_name_' : self.optimizer_name,
                   'optimizer_params_' : self.optimizer_parameters,
                   'n_iteration_' : self.n_iteration,
                   'optima_found_' : self.optima_found,
                   'actual_optima_' : self.actual_optima,
                   'runtime_' : self.runtime,
                   'git_commit_id' : self.git_commit_id}
        template = env.get_template("md-template.jinja")
        with open(fname, 'w') as f:
            rendered_data = template.render(context)
            f.write(rendered_data)

    def plot_fitness(self):
        plt.plot(self.fitness_list)
        plt.ylabel('Fitness')
        plt.xlabel('Iteration')
        plt.savefig('fitness_progress.png', format='png')