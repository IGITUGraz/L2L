import numpy as np
import sklearn.gaussian_process as gp

from ltl.optimizees.optimizee import Optimizee
from ltl import dict_to_list


class SurrogateOptimizee(Optimizee):
    """Surrogate Objective for optimizing computational hard objectives. Currently works with gaussian processes from
    scikit learn.
    """

    def __init__(self, traj, optimizee, noise_level=0.0):
        self.optimizee = optimizee
        self.individuals = []
        self.results = []
        self.n_samples = 0
        self.min_samples = 20
        self.kernel = gp.kernels.RBF(length_scale=0.1) + gp.kernels.WhiteKernel(noise_level=noise_level)
        self.noise_level = noise_level
        self.fitted = False
        self.gaussian_processes = []

        individual = optimizee.create_individual()
        _, self.individual_dict_spec = dict_to_list(individual, get_dict_spec=True)

    def create_individual(self):
        """
        Delegates to optimize
        """
        return self.optimizee.create_individual()

    def bounding_func(self, individual):
        """
        Delegates to optimizee
        """
        return self.optimizee.bounding_func(individual)

    def _traj_individual_to_dict(self, traj):
        """
        transforms current traj individual to corresponding dict
        :param traj: Trajectory containing the individual
        :return: A dict representing the individual
        """
        individual = dict()
        for key in self.individual_dict_spec:
            key = key[0]
            individual[key] = traj.individual[key]
        return individual

    def simulate(self, traj):
        """
        Tries to return an approximation, else runs original simulation

        :param ~pypet.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """
        individual = np.array(dict_to_list(self._traj_individual_to_dict(traj)))
        if self.n_samples > self.min_samples:
            if self.fitted:
                approximation = []
                for gaussian_process in self.gaussian_processes:
                    y, std = gaussian_process.predict(individual, return_std=True)
                    approximation.append([y, std])
                approximation = np.array(approximation)
                if np.max(approximation[:, 1]) < 0.2 + self.noise_level:
                    print('approximation hit')
                    traj.f_add_result('$set.$.approximated', True)
                    return tuple(x for x in approximation[:, 0])

        simulation_result = self.optimizee.simulate(traj)
        # fit model
        x = np.array(simulation_result)
        self.individuals.append(individual)
        self.results.append(x)
        for i in range(len(x)):
            if len(self.gaussian_processes) < len(x):
                gpr = gp.GaussianProcessRegressor(kernel=self.kernel)
                self.gaussian_processes.append(gpr)
            else:
                gpr = self.gaussian_processes[i]
            gpr.fit(self.individuals, self.results)
            self.fitted = True
        self.n_samples += 1
        return simulation_result
