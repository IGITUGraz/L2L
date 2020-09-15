import numpy as np
from l2l.optimizers.kalmanfilter.kalman_utils import _get_shapes, \
    _encode_targets, _get_batches

try:
    from numba import jit
except ModuleNotFoundError:
    # create a mock decorator if `numba` is not found
    def jit(**kwargs):
        def decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
        return decorator


class EnsembleKalmanFilter:
    def __init__(self, maxit=1, n_batches=1, online=False):
        """
        Ensemble Kalman Filter (EnKF)

        EnKF following the formulation found in Iglesias et al. (2013),
        The Ensemble Kalman Filter for Inverse Problems.
        doi:10.1088/0266-5611/29/4/045001

        :param maxit: int, maximum number of iterations
        :param n_batches, int,  number of batches to used in mini-batch. If set
            to `1` uses the whole given dataset. Default is `1`.
        :param online, bool, True if one random data point is requested,
            between [0, dims], otherwise do mini-batch. `dims` is the number of
            observations. Default is False
        """
        self.ensemble = None
        self.observations = None

        self.maxit = maxit
        self.online = online
        self.n_batches = n_batches
        self.gamma = 0.
        self.gamma_s = 0
        self.dims = 0

    def fit(self, ensemble, ensemble_size,
            observations, model_output, gamma):
        """
        Prediction and update step of the EnKF

        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param observations: nd numpy array, observation or targets
        :param model_output: nd numpy array, output of the model
            In terms of the Kalman Filter the model maps the ensembles (dim n)
            into the observed data `y` (dim k). E.g. network output activity
        :return self, Possible outputs are:
            ensembles: nd numpy array, optimized `ensembles`
            c_pp: nd numpy array, covariance matrix of the model output
            c_up: nd numpy array, covariance matrix of the model output and the
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
        self.observations = _encode_targets(observations, self.gamma_s)

        for i in range(self.maxit):
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
                    c_pp = _cov_mat(g_tmp, g_tmp, ensemble_size)
                    c_up = _cov_mat(self.ensemble, g_tmp, ensemble_size)
                    self.ensemble = _update_step(self.ensemble,
                                                 self.observations[d],
                                                 g_tmp, self.gamma,
                                                 c_pp, c_up)
        return self


@jit(nopython=True, parallel=True)
def _update_step(ensemble, observations, g, gamma, cpp, cup):
    """
    Update step of the kalman filter

    Calculates the covariances and returns new ensembles
    """
    return ensemble + (
                cup @ np.linalg.lstsq(cpp + gamma, (observations - g).T)[0]).T


@jit(parallel=True)
def _cov_mat(x, y, ensemble_size):
    """
    Covariance matrix
    """
    return np.tensordot((x - np.mean(x, axis=0)), (y - np.mean(y, axis=0)),
                        axes=[0, 0]) / ensemble_size


