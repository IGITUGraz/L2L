import logging
import abc
from abc import ABCMeta
import numpy as np
import sklearn.mixture

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


class BayesianGaussianMixture():
    """BayesianGaussianMixture from sklearn
    Unlike normal Gaussian mixture, the algorithm has tendency to set the weights of non present modes close to zero.
    Meaning that it effectively inferences the number of active modes present in the given data.
    """
    def __init__(self, n_components=2, **kwargs):
        """ Initialize the distribution

        :param n_components: components of the mixture model
        """
        self.bayesian_mixture = sklearn.mixture.BayesianGaussianMixture(
            n_components, weight_concentration_prior_type='dirichlet_distribution', **kwargs)
        # taken from check_fitted function of BaysianGaussianMixture in the sklearn repository
        self.parametrization = ('covariances_', 'means_', 'weight_concentration_', 'weights_',
                                'mean_precision_', 'degrees_of_freedom_', 'precisions_', 'precisions_cholesky_')

    def _postprocess_fitted(self, model):
        """postprocesses the fitted model, adding the possibility to add noise or something
        """
        pass

    def _append_additional_parameters(self, distribution_parameters):
        """
        appends additional parametrization
        
        :param distribution_parameters: the dictionary that contains the distributions parametrization
        """
        pass

    def fit(self, data_list, smooth_update=0):
        """Fits data_list on the parametrized model
        
        :param data_list: list or numpy array with individuals as rows
        :param smooth_update: determines to which extent the new samples
        account for the new distribution.
        :return: dict specifiying current parametrization
        """
        old = self.bayesian_mixture
        self.bayesian_mixture.fit(data_list)
        self._postprocess_fitted(self.bayesian_mixture)
        distribution_parameters = dict()

        # smooth update and fill out distribution parameters dict to return
        # distribution parameters can also be tuples of ndarray
        for p in self.parametrization:
            orig = old.__getattribute__(p)
            new = self.bayesian_mixture.__getattribute__(p)
            if isinstance(orig, tuple):
                mix = tuple(smooth_update * a + (1 - smooth_update) * b for a, b in zip(orig, new))
                for index in range(len(mix)):
                    distribution_parameters[p + '_' + str(index)] = mix[index]
            else:
                mix = smooth_update * orig + (1 - smooth_update) * new
                distribution_parameters[p] = mix
            self.bayesian_mixture.__setattr__(p, mix)
            if p == 'covariances_':
                logger.info('New covariances:\n' + str(mix))
            elif p == 'means_':
                logger.info('New means:\n' + str(mix))
        self._append_additional_parameters(distribution_parameters)
        return distribution_parameters

    def sample(self, n_individuals):
        """Sample n_individuals individuals under the current parametrization
        
        :param n_individuals: number of individuals to sample
        :return: numpy array with n_individuals rows of individuals
        """
        individuals, _ = self.bayesian_mixture.sample(n_individuals)
        return individuals


class NoisyBayesianGaussianMixture(BayesianGaussianMixture):
    """NoisyBayesianGaussianMixture is basically the same as BayesianGaussianMixture but superimposed with noise
    """
    def __init__(self, n_components, additive_noise=None, noise_decay=0.95, **kwargs):
        """
        Initializes the Noisy Bayesian Gaussian Mixture Model with noise components
        :param n_components: number of components in the mixture model
        :param additive_noise: vector representing the diagonal covariances that get added to the 
        diagonalized covariance matrix. If None it will get to np.ones
        :param noise_decay: factor that will decay the additive noise
        :param kwargs: additional arguments that get passed to the underlying scikit learn model
        """
        BayesianGaussianMixture.__init__(self, n_components, **kwargs)
        self.noise_decay = noise_decay
        if additive_noise is not None:
            self.additive_noise = np.array(additive_noise, dtype=np.float)

    def _postprocess_fitted(self, model):
        """
        adds noise to the diagonalized components
        
        :param model: the considered model
        """
        if hasattr(model, 'covariances_'):
            for cov in model.covariances_:
                _, eigenvectors = np.linalg.eig(cov)
                if self.additive_noise is None:
                    self.additive_noise = np.ones(eigenvectors.shape[-1])
                noise = np.diag(self.additive_noise)
                self.additive_noise *= self.noise_decay
                cov += eigenvectors.dot(noise.dot(eigenvectors.T))

    def _append_additional_parameters(self, distribution_parameters):
        """
        appends noise parameters
        
        :param distribution_parameters: the dictionary that contains the distributions parametrization
        """
        distribution_parameters['additive_noise'] = self.additive_noise


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

        self.noise_bias *= self.noise_decay
        diagonalized_covariance = np.diag(eigenvalues + self.noise)
        self.noisy_cov = eigenvectors.dot(diagonalized_covariance.dot(eigenvectors.T))

        logger.debug('Noisy cov\n%s', self.noisy_cov)
        return {'mean': self.mean, 'covariance_matrix': self.noisy_cov, 'noise_bias': self.noise_bias}

    def sample(self, n_individuals):
        """Samples from current parametrization

        :returns n_individuals Individuals
        """
        return np.random.multivariate_normal(self.mean, self.noisy_cov, n_individuals)
