from ltl.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, \
    MichalewiczParameters, ShekelParameters, \
    RastriginParameters, ChasmParameters


class BenchmarkedFunctions:
    """
    Implements benchmarked functions class for an easier call of the benchmarked functions

    :param noise: Boolean value indicating if the Gaussian noise will be applied on the resulting function.
    :param mu: Scalar indicating the mean of the Gaussian noise.
    :param sigma: Scalar indicating the standard deviation of the Gaussian noise.
    """
    def __init__(self, noise=False, mu=0., sigma=0.01):
        self.function_name_map = [("Rastrigin2d", self._create_rastrigin2d(noise, mu, sigma)),
                                  ("Chasm", self._create_chasm(noise, mu, sigma)),
                                  ("Shekel2d", self._create_shekel2d(noise, mu, sigma)),
                                  ("Michalewicz2d", self._create_michalewicz2d(noise, mu, sigma)),
                                  ("3Gaussians2d", self._create_3gaussians2d(noise, mu, sigma))]

    def get_function_by_index(self, id):
        """
        Get the benchmarked function with given id
        :param id: Function id
        """
        return self.function_name_map[id]

    def _create_rastrigin2d(self, noise, mu, sigma):
        return FunctionGenerator([RastriginParameters()], dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_chasm(self, noise, mu, sigma):
        return FunctionGenerator([ChasmParameters()], dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_shekel2d(self, noise, mu, sigma):
        return FunctionGenerator([ShekelParameters(A='default', c='default')], dims=2, noise=noise,
                                 mu=mu, sigma=sigma)

    def _create_michalewicz2d(self, noise, mu, sigma):
        return FunctionGenerator([MichalewiczParameters(m='default')], dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_3gaussians2d(self, noise, mu, sigma):
        fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                     GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                     GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
        return FunctionGenerator(fg_params, dims=2, noise=noise, mu=mu, sigma=sigma)
