import numpy as np
import logging
logger = logging.getLogger("ltl-distribution")

class Distribution():
    """Generic base for a distribution. Needs to implement the functions fit and sample.
    """
    def fit(self, individuals):
        """This function fits the distributions parameters to the given samples
        in maximum likelihood fashion.

        :param individuals: A list or array of individuals to fit to.
        :returns tuple containing the inferred distribution parameters and a short description
        """
        pass
 
    def sample(self, n_individuals):
        """Samples n_individuals from the current parametrized distribution.

        :param n_individuals: An integer specifiyng the amount of individuals to sample

        :returns: A list or array of n_individuals
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
        
        :returns tuple containing the inferred mean and covariance of the gaussian
        """
        mean = np.mean(data_list, axis=0)
        cov_mat = np.cov(data_list, rowvar=False)

        self.mean = smooth_update * mean + (1 - smooth_update) * self.mean
        self.cov = smooth_update * cov_mat + (1 - smooth_update) * self.cov
        
        logger.debug('  Inferred gaussian center: {}'.format(self.mean))
        logger.debug('  Inferred gaussian cov   : {}'.format(self.cov))
        
        return [('.gaussian_center', self.mean, 'center of gaussian distribution estimated from the '
                                                'evaluated generation'), 
                ('.gaussian_covariance_matrix', self.cov, 'covariance matrix from the evaluated generation')]
        

    def sample(self, n_individuals):
        """Sample n_individuals individuals under the current parametrization

        :param n_individuals: number of individuals to sample.

        :returns numpy array with n_individual rows of individuals
        """
        return np.random.multivariate_normal(self.mean, self.cov, n_individuals)
