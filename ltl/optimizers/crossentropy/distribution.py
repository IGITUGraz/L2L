import logging
import abc
from abc import ABCMeta
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

logger = logging.getLogger('ltl-distribution')


class Distribution(metaclass=ABCMeta):
    """Generic base for a distribution. Needs to implement the functions fit and sample.
    """
    
    @abc.abstractmethod
    def fit(self, individuals):
        """This function fits the distributions parameters to the given samples
        in maximum likelihood fashion.

        :param individuals: A list or array of individuals to fit to.
        :returns a dict describing the current parametrization
        """
        pass

    @abc.abstractmethod
    def sample(self, n_individuals):
        """Samples n_individuals from the current parametrized distribution.

        :param n_individuals: An integer specifiyng the amount of individuals to sample

        :returns: A list or array of n_individuals
        """
        pass


class Gaussian(Distribution):
    """Gaussian distribution.
    """
    
    def __init__(self):
        """Initializes the distributions members
        """
        self.mean = None
        self.cov = None

    def fit(self, data_list, smooth_update=0):
        """Fit a gaussian distribution to the given data

        :param data_list: list or numpy array with individuals as rows
        :param smooth_update: determines to which extent the new samples
        account for the new distribution.
        default is 0 -> old parameters are fully discarded
        
        :returns dict specifying current parametrization
        """
        mean = np.mean(data_list, axis=0)
        cov_mat = np.cov(data_list, rowvar=False)

        if self.mean is None:
            self.mean = mean
            self.cov = cov_mat

        self.mean = smooth_update * self.mean + (1 - smooth_update) * mean
        self.cov = smooth_update * self.cov + (1 - smooth_update) * cov_mat

        logger.debug('Gaussian center\n%s', self.mean)
        logger.debug('Gaussian cov\n%s', self.cov)
        
        return {'mean': self.mean, 'covariance_matrix': self.cov}
        
    def sample(self, n_individuals):
        """Sample n_individuals individuals under the current parametrization
 
        :param n_individuals: number of individuals to sample.
 
        :returns numpy array with n_individual rows of individuals
        """
        return np.random.multivariate_normal(self.mean, self.cov, n_individuals)


class GaussianMixture():
    """Gaussian Mixture Model
    """
    def __init__(self, n_components=2):
        """
        
        :param n_components: components of the mixture model
        """
        self.bayesian_mixture = BayesianGaussianMixture(n_components)
        self.fitted = False
        self.parametrization = ('covariance_prior_', 'covariances_', 'mean_precision_prior_', 'mean_prior_',
                                'means_', 'weight_concentration_prior_', 'weight_concentration_', 'weights_')

    def fit(self, data_list, smooth_update=0):
        """Fits data_list on the parametrized model
        
        :param data_list: 
        :param smooth_update: 
        :return: 
        """
        old = self.bayesian_mixture
        self.bayesian_mixture.fit(data_list)
        for p in self.parametrization:
            orig = old.__getattribute__(p)
            new = self.bayesian_mixture.__getattribute__(p)
            if isinstance(orig, tuple):
                mix = tuple(smooth_update * a + (1 - smooth_update) * b for a, b in zip(orig, new))
            else:
                mix = smooth_update * orig + (1 - smooth_update) * new
            self.bayesian_mixture.__setattr__(p, mix)
        return {'mean': self.bayesian_mixture.means_, 'covariance_matrix': self.bayesian_mixture.covariances_,
                'weights': self.bayesian_mixture.weights_}

    def sample(self, n_individuals):
        """
        
        :param n_individuals: 
        :return: 
        """
        individuals, _ = self.bayesian_mixture.sample(n_individuals)
        return individuals


class NoisyGaussian(Gaussian):
    """Additive Noisy Gaussian distribution. The initialization of its noise components
    happens during the first fit where the magnitude of the noise in each
    diagonalized component is estimated.
    """
    def __init__(self, noise_decay=0.99, noise_bias=0.05):
        """Initializes the noisy distribution

        :param noise_decay: Multiplicative decay of the noise components
        :param noise_bias: Factor to the variance of the first fit. This is then used as
                           additive noise.

                noise_var = noise_bias * var(first_fit)
        """
        Gaussian.__init__(self)
        self.noise_decay = noise_decay
        self.noise_bias = noise_bias
        self.noisy_cov = None
        self.noise = None

    def fit(self, data_list, smooth_update=0):
        """Fits the parameters to the given data (see .Gaussian) and additionally
        adds noise in form of variance to the covariance matrix. Also, the noise
        is decayed after each step

        :param data_list: Data to be fitted to
        :param smooth_update: Smooth the parameter update with regard to the
            previous configuration

        :returns dict describing parameter configuration
        """
        Gaussian.fit(self, data_list, smooth_update)
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)

        # determine noise variance
        if self.noise is None:
            self.noise = self.noise_bias * eigenvalues
            self.noise /= self.noise_decay

        self.noise *= self.noise_decay
        diagonalized_covariance = np.diag(eigenvalues + self.noise)
        self.noisy_cov = eigenvectors.dot(diagonalized_covariance.dot(eigenvectors.T))

        logger.debug('Noisy cov\n%s', self.noisy_cov)
        return {'mean': self.mean, 'covariance_matrix': self.noisy_cov}

    def sample(self, n_individuals):
        """Samples from current parametrization

        :returns n_individuals Individuals
        """
        return np.random.multivariate_normal(self.mean, self.noisy_cov, n_individuals)
