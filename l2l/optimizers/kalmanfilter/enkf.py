import numpy as np
import abc

from .kalman_utils import _get_shapes, _encode_targets, _get_batches
from abc import ABCMeta

try:
    from numba import jit
except ModuleNotFoundError:
    def jit(**kwargs):
        def decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
        return decorator

class KalmanFilter(metaclass=ABCMeta):

    @abc.abstractmethod
    def fit(self, ensemble, ensemble_size,
            observations, model_output, gamma, noise, p):
        pass


class EnsembleKalmanFilter(KalmanFilter):
    def __init__(self, maxit=1, n_batches=1, online=False):
        """
        Ensemble Kalman Filter (EnKF)

        EnKF following the formulation found in Iglesias et al. (2013),
        The Ensemble Kalman Filter for Inverse Problems.

        :param maxit: int, maximum number of iterations
        :param n_batches, int,  number of batches to used in mini-batch. If set
            to `1` uses the whole given dataset. Default is `1`.
        :param online, bool, True if one random data point is requested,
            between [0, dims], otherwise do mini-batch. `dims` is the number of
            observations. Default is False
        """
        self.Cpp = None
        self.Cup = None
        self.ensemble = None
        self.observations = None

        self.maxit = maxit
        self.online = online
        self.n_batches = n_batches
        self.gamma = 0.
        self.gamma_s = 0
        self.dims = 0

    def fit(self, ensemble, ensemble_size,
            observations, model_output, gamma, noise=0., p=None, u_exact=None):
        """
        Prediction and update step of the EnKF

        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param u_exact: nd numpy array, exact control, e.g. if weight sof the model
                   are known. Default is `None`
        :param observations: nd numpy array, observation or targets
        :param model_output: nd numpy array, output of the model
            In terms of the Kalman Filter the model maps the ensembles (dim n)
            into the observed data `y` (dim k). E.g. network output activity
        :param noise: nd numpy array, Noise can be added to the model (for `gamma`)
            and is used in the misfit calculation for convergence.
            E.g. multivariate normal distribution. Default is `0.0`
        :param  gamma: nd numpy array, Normalizes the model-data distance in the
            update step, :`noise * I` (I is identity matrix) or
            :math:`\\gamma=I` if `noise` is zero
        :param p: nd numpy array
            Exact solution given by :math:`G * u_exact`, where `G` is inverse
            of a linear elliptic function `L`, it maps the control into the
            observed data, Default is `None`
        :return self, Possible outputs are:
            ensembles: nd numpy array, optimized `ensembles`
            Cpp: nd numpy array, covariance matrix of the model output
            Cup: nd numpy array, covariance matrix of the model output and the
                ensembles
        """
        model_output = model_output.copy()
        # get shapes
        self.gamma_s, self.dims = _get_shapes(observations, model_output)

        if isinstance(gamma, (int, float)):
            if float(gamma) == 0.:
                self.gamma = np.eye(self.gamma_s)
        else:
            self.gamma = gamma

        # copy the data so we do not overwrite the original arguments
        self.ensemble = ensemble.copy()
        self.observations = observations.copy()
        self.observations = _encode_targets(observations, self.gamma_s)

        for i in range(self.maxit):
            if (i % 100) == 0:
                print('Iteration {}/{}'.format(i, self.maxit))

            # now get mini_batches
            if self.n_batches > self.dims:
                num_batches = 1
            else:
                num_batches = self.n_batches
            mini_batches = _get_batches(
                num_batches, shape=self.dims, online=self.online)

            for idx in mini_batches:
                # in case of online learning idx should be an int
                # put it into a list to loop over it
                if not isinstance(idx, np.ndarray):
                    idx = [idx]

                for d in idx:
                    # now get only the individuals output according to idx
                    g_tmp = model_output[:, :, d]
                    # Calculate the covariances
                    Cpp = _cov_mat(g_tmp, g_tmp, ensemble_size)
                    Cup = _cov_mat(self.ensemble, g_tmp, ensemble_size)
                    self.ensemble = _update_step(self.ensemble,
                                                 self.observations[d],
                                                 g_tmp, self.gamma,
                                                 Cpp, Cup)
        return self


@jit(nopython=True, parallel=True)
def _update_step(ensemble, observations, g, gamma, Cpp, Cup):
    """
    Update step of the kalman filter

    Calculates the covariances and returns new ensembles
    """
    return ensemble + (
                Cup @ np.linalg.lstsq(Cpp + gamma, (observations - g).T)[0]).T


@jit(parallel=True)
def _cov_mat(x, y, ensemble_size):
    """
    Covariance matrix
    """
    return np.tensordot((x - np.mean(x, axis=0)), (y - np.mean(y, axis=0)),
                        axes=[0, 0]) / ensemble_size


