import numpy as np


class Distribution():
    """Generic base for a distribution. Needs to implement the functions fit and sample.
    """
    def fit(self, individuals):
        """This function fits the distributions parameters to the given samples
        in maximum likelihood fashion.

        :param individuals: A list or array of individuals to fit to.
        """
        pass
 
    def sample(self, n_individuals):
        """Samples n_individuals from the current parametrized distribution.

        :param n_individuals: An integer specifiyng the amount of individuals to sample

        :returns: A list or array of n_individuals
        """
        pass

    def log(self, logger):
        """Additional callback for logging

        :param logger: The logger
        """
        pass

    def add_results(self, generation_name, traj):
        """Callback to add the current parametrization to given PyPET trajectory.

        :param generation_name: Current generation name
        :param traj: PyPET trajectory
        """
        pass


class Gaussian(Distribution):
    """Gaussian distribution.
    """
    def __init__(self, initial_individuals):
        """Initializes the parameters with given individuals

        :param initial_individuals: The initial individuals to fit to.
        """
        self.mean = np.zeros(initial_individuals[0].shape)
        self.cov = np.zeros((initial_individuals[0].shape[0], initial_individuals[0].shape[0]))
        self.fit(initial_individuals)

    def fit(self, data_list, smooth_update=1):
        """Fit a gaussian distribution to the given data

        :param data_list: list or numpy array with individuals as rows
        :param smooth_update: determines to which extent the new samples
        account for the new distribution.
        default is 1 -> old parameters are fully discarded
        """
        mean = np.mean(data_list, axis=0)
        # lookup np.cov
        cov_mat = np.cov(data_list, rowvar=False)

        self.mean = smooth_update * mean + (1 - smooth_update) * self.mean
        self.cov = smooth_update * cov_mat + (1 - smooth_update) * self.cov

    def sample(self, n_individuals):
        """Sample n_individuals individuals under the current parametrization

        :param n_individuals: number of individuals to sample.

        :returns numpy array with n_individual rows of individuals
        """
        return np.random.multivariate_normal(self.mean, self.cov, n_individuals)

    def log(self, logger):
        """Logs current parametrization

        :param logger: The logger
        """
        logger.info('  Inferred gaussian center: {}'.format(self.mean))
        logger.info('  Inferred gaussian cov   : {}'.format(self.cov))

    def add_results(self, generation_name, traj):
        """Adds the current parametrization to the trajectory

        :param generation_name: Current generation_name
        :param traj: PyPET trajectory
        """
        traj.results.generation_params.f_add_result(generation_name + '.gaussian_center', self.mean,
                                                    comment='center of gaussian distribution estimated from the '
                                                            'evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.gaussian_covariance_matrix', self.cov,
                                                    comment='covariance matrix from the evaluated generation')
