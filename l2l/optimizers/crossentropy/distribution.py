import abc
import logging
from abc import ABCMeta

import numpy as np
import sklearn.mixture

logger = logging.getLogger('optimizers.crossentropy.distribution')


class Distribution(metaclass=ABCMeta):
    """
    Generic base for a distribution. Needs to implement the functions fit and sample.
    """

    @abc.abstractmethod
    def init_random_state(self, random_state):
        """
        Used to initialize the random number generator which is used to fit/sample data. Note
        that if the random_state is already set, this raises an `AssertionError`. The reason
        this is not a part of the constructor is that the distribution random state must be
        initialized only by the optimizer and not in the main function (where it is
        constructed). It is essential to call this function before using the distribution

        :param random_state: An instance of class:`numpy.random.RandomState`
        """
        pass

    @abc.abstractmethod
    def fit(self, data_list):
        """This function fits the distributions parameters to the given samples
        in maximum likelihood fashion.

        :param data_list: A list or array of individuals to fit to.
        :return dict: a dict describing the current parametrization
        """
        pass

    @abc.abstractmethod
    def sample(self, n_individuals):
        """Samples n_individuals from the current parametrized distribution.

        :param n_individuals: An integer specifying the amount of individuals to sample

        :return: A list or array of n_individuals
        """
        pass

    @abc.abstractmethod
    def get_params(self):
        """
        :return: the parametrization of the distribution as a dict
        """
        pass


class Gaussian(Distribution):
    """
    Gaussian distribution.
    """

    def __init__(self):
        self.random_state = None
        self.mean = None
        self.cov = None

    def init_random_state(self, random_state):
        assert self.random_state is None, "The random_state has already been set for the distribution"
        assert isinstance(random_state, np.random.RandomState)
        self.random_state = random_state

    def get_params(self):
        params_dict_items = [("distribution_name", self.__class__.__name__)]
        return dict(params_dict_items)

    def fit(self, data_list, smooth_update=0):
        """
        Fit a gaussian distribution to the given data

        :param data_list: list or numpy array with individuals as rows
        :param smooth_update: determines to which extent the new samples account for the new distribution.
          default is 0 -> old parameters are fully discarded
        
        :return dict: specifying current parametrization
        """
        assert self.random_state is not None, \
            "The random_state for the distribution has not been set, call the" \
            " 'init_random_state' member function to set it"

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
 
        :return: numpy array with n_individual rows of individuals
        """
        assert self.random_state is not None, \
            "The random_state for the distribution has not been set, call the" \
            " 'init_random_state' member function to set it"
        return self.random_state.multivariate_normal(self.mean, self.cov, n_individuals)


class BayesianGaussianMixture(Distribution):
    """
    BayesianGaussianMixture from sklearn

    Unlike normal Gaussian mixture, the algorithm has tendency to set the weights of
    non present modes close to zero. Meaning that it effectively inferences the
    number of active modes present in the given data.

    :param n_components: components of the mixture model
    :param kwargs: Additional arguments that get passed on to :class:`sklearn.mixture.BayesianGaussianMixture`
    """

    def __init__(self, n_components=2, **kwargs):
        self.random_state = None
        self.bayesian_mixture = sklearn.mixture.BayesianGaussianMixture(
            n_components,
            weight_concentration_prior_type='dirichlet_distribution',
            random_state=self.random_state, **kwargs)
        # taken from check_fitted function of BaysianGaussianMixture in the sklearn repository
        self.parametrization = ('covariances_', 'means_', 'weight_concentration_', 'weights_',
                                'mean_precision_', 'degrees_of_freedom_', 'precisions_', 'precisions_cholesky_')
        self.n_components = n_components

    def init_random_state(self, random_state):
        assert self.random_state is None, "The random_state has already been set for the distribution"
        assert isinstance(random_state, np.random.RandomState)
        self.random_state = random_state
        self.bayesian_mixture.random_state = self.random_state

    def _postprocess_fitted(self, model):
        """
        postprocesses the fitted model, adding the possibility to add noise or something
        """
        pass

    def _append_additional_parameters(self, distribution_parameters):
        """
        appends additional parametrization
        
        :param distribution_parameters: the dictionary that contains the distributions
            parametrization
        """
        pass

    def get_params(self):
        params_dict_items = [("distribution_name", self.__class__.__name__),
                             ("n_components", self.n_components)]
        return dict(params_dict_items)

    def fit(self, data_list, smooth_update=0):
        """
        Fits data_list on the parametrized model
        
        :param data_list: list or numpy array with individuals as rows
        :param smooth_update: determines to which extent the new samples account for the
            new distribution.
        :return: dict specifiying current parametrization
        """
        assert self.random_state is not None, \
            "The random_state for the distribution has not been set, call the" \
            " 'init_random_state' member function to set it"

        old = self.bayesian_mixture
        self.bayesian_mixture.fit(data_list)
        self._postprocess_fitted(self.bayesian_mixture)
        distribution_parameters = dict()

        # smooth update and fill out distribution parameters dict to return
        # distribution parameters can also be tuples of ndarray
        for p in self.parametrization:
            hdf_name = p.rstrip('_')  # remove sklearn trailing underscore
            orig = getattr(old, p)
            new = getattr(self.bayesian_mixture, p)
            if isinstance(orig, tuple):
                mix = tuple(smooth_update * a + (1 - smooth_update) * b for a, b in zip(orig, new))
                for index in range(len(mix)):
                    distribution_parameters[hdf_name + '_' + str(index)] = mix[index]
            else:
                mix = smooth_update * orig + (1 - smooth_update) * new
                distribution_parameters[hdf_name] = mix
            setattr(self.bayesian_mixture, p, mix)
            if p == 'covariances_':
                logger.debug('New covariances:\n%s', str(mix))
            elif p == 'means_':
                logger.debug('New means:\n%s', str(mix))
        self._append_additional_parameters(distribution_parameters)
        return distribution_parameters

    def sample(self, n_individuals):
        """
        Sample n_individuals individuals under the current parametrization
        
        :param n_individuals: number of individuals to sample
        :return: numpy array with n_individuals rows of individuals
        """
        assert self.random_state is not None, \
            "The random_state for the distribution has not been set, call the" \
            " 'init_random_state' member function to set it"
        individuals, _ = self.bayesian_mixture.sample(n_individuals)
        return individuals


class NoisyBayesianGaussianMixture(BayesianGaussianMixture):
    """
    NoisyBayesianGaussianMixture is basically the same as BayesianGaussianMixture
    but superimposed with noise

    :param n_components: number of components in the mixture model
    :param noise_magnitude: scalar factor that affects the magnitude of noise
        applied on the fitted distribution parameters+

    :param coordinate_scale: This should be a vector representing the scaling of
        the coordinates. The noise applied to each coordinate `i` is

          noise_magnitude * coordinate_scale[i]

        Defaults to 1 for each coordinate.

    :param noise_decay: factor that will decay the additive noise
    :param kwargs: additional arguments that get passed on to
        :class:`.BayesianGaussianMixture` distribution
    """

    def __init__(self, n_components, noise_magnitude=1.0, coordinate_scale=None, noise_decay=0.95, **kwargs):
        BayesianGaussianMixture.__init__(self, n_components=n_components, **kwargs)
        self.noise_decay = noise_decay
        self.noise_magnitude = np.float64(noise_magnitude)
        if coordinate_scale is None:
            self.coordinate_scale = np.float64(1)
        else:
            self.coordinate_scale = np.array(coordinate_scale).astype(np.float64)
        self.current_noise_magnitude = self.noise_magnitude
        self.noise_value = None  # list containing the additive noise values for each component

    def _postprocess_fitted(self, model):
        """
        adds noise to the diagonalized components
        
        :param model: the considered model
        """
        if hasattr(model, 'covariances_'):
            self.noise_value = []
            n_dims = model.covariances_[0].shape[0]
            for i, cov in enumerate(model.covariances_):
                current_noise_value = \
                    np.abs(self.random_state.normal(loc=0.0,
                                                    scale=self.current_noise_magnitude * self.coordinate_scale,
                                                    size=n_dims))
                self.noise_value.append(current_noise_value)
                model.covariances_[i] += np.diag(current_noise_value)
            self.current_noise_magnitude *= self.noise_decay

    def _append_additional_parameters(self, distribution_parameters):
        """
        appends noise parameters
        
        :param distribution_parameters: the dictionary that contains the distributions parametrization
        """
        distribution_parameters['noise_value'] = np.array(self.noise_value)

    def get_params(self):
        params_dict = super().get_params()
        params_dict.update(dict(distribution_name=self.__class__.__name__,
                                noise_magnitude=self.noise_magnitude,
                                coordinate_scale=self.coordinate_scale,
                                noise_decay=self.noise_decay))
        return dict(params_dict)


class NoisyGaussian(Gaussian):
    """
    Additive Noisy Gaussian distribution. The initialization of its noise components
    happens during the first fit where the magnitude of the noise in each
    diagonalized component is estimated.

    :param noise_magnitude: scalar factor that affects the magnitude of noise
        applied on the distribution parameters
    :param coordinate_scale: This should be a vector representing the scaling of
        the coordinates. The noise applied to each coordinate `i` is
        `noise_magnitude*coordinate_scale[i]`
    :param noise_decay: Multiplicative decay of the noise components
    """

    def __init__(self, noise_magnitude=1.0, coordinate_scale=None, noise_decay=0.95):
        Gaussian.__init__(self)
        self.noise_decay = noise_decay
        self.noise_magnitude = np.float64(noise_magnitude)
        if coordinate_scale is None:
            self.coordinate_scale = np.float64(1)
        else:
            self.coordinate_scale = np.array(coordinate_scale).astype(np.float64)
        self.current_noise_magnitude = self.noise_magnitude
        self.noise_value = None  # vector containing the 
        self.noisy_cov = None

    def get_params(self):
        params_dict = super().get_params()
        params_dict.update(dict(distribution_name=self.__class__.__name__,
                                noise_magnitude=self.noise_magnitude,
                                coordinate_scale=self.coordinate_scale,
                                noise_decay=self.noise_decay))
        return params_dict

    def fit(self, data_list, smooth_update=0):
        """
        Fits the parameters to the given data (see :class:`.Gaussian`) and additionally
        adds noise in form of variance to the covariance matrix. Also, the noise
        is decayed after each step

        :param data_list: Data to be fitted to
        :param smooth_update: Smooth the parameter update with regard to the
            previous configuration

        :return dict: describing parameter configuration
        """
        assert self.random_state is not None, \
            "The random_state for the distribution has not been set, call the" \
            " 'init_random_state' member function to set it"

        Gaussian.fit(self, data_list, smooth_update)
        n_dims = self.cov.shape[0]
        self.noise_value = np.abs(
            self.random_state.normal(loc=0.0, scale=self.current_noise_magnitude * self.coordinate_scale,
                                     size=n_dims))
        self.noisy_cov = self.cov + np.diag(self.noise_value)
        self.current_noise_magnitude *= self.noise_decay

        logger.debug('Noisy cov\n%s', self.noisy_cov)
        return {'mean': self.mean, 'covariance_matrix': self.noisy_cov, 'noise_value': self.noise_value}

    def sample(self, n_individuals):
        """
        Samples from current parametrization

        :return: n_individuals Individuals
        """
        assert self.random_state is not None, \
            "The random_state for the distribution has not been set, call the" \
            " 'init_random_state' member function to set it"

        return self.random_state.multivariate_normal(self.mean, self.noisy_cov, n_individuals)
