from l2l.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, \
    MichalewiczParameters, ShekelParameters, EasomParameters, LangermannParameters, \
    RastriginParameters, ChasmParameters, RosenbrockParameters, AckleyParameters, PermutationParameters
from collections import OrderedDict


class BenchmarkedFunctions:
    """
    Implements benchmarked functions class for an easier call of the benchmarked functions

    """
    def __init__(self):
        self.function_name_map = [("Rastrigin2d", self._create_rastrigin2d),
                                  ("Rastrigin10d", self._create_rastrigin10d),
                                  ("Rosenbrock2d", self._create_rosenbrock2d),
                                  ("Rosenbrock10d", self._create_rosenbrock10d),
                                  ("Ackley2d", self._create_ackley2d),
                                  ("Ackley10d", self._create_ackley10d),
                                  ("Chasm", self._create_chasm),
                                  ("Gauss2d", self._create_gauss2d),
                                  ("Shekel2d", self._create_shekel2d),
                                  ("Langermann2d", self._create_langermann),
                                  ("Michalewicz2d", self._create_michalewicz2d),
                                  ("Permutation2d", self._create_permutation2d),
                                  ("Easom2d", self._create_easom2d),
                                  ("Easom10d", self._create_easom10d),
                                  ("3Gaussians2d", self._create_3gaussians2d)]
        self.function_name_index_map = OrderedDict([(name, index)
                                                    for index, (name, _) in enumerate(self.function_name_map)])

    def get_function_by_index(self, id_, noise=False, mu=0., sigma=0.01):
        """
        Get the benchmarked function with given id
        :param id_: Function id
        :param noise: Indicates whether the function should provide noisy values
        :param mu: mean of the noise
        :param sigma: convariance of the noise

        :return function_name_map entry for the given id and the parameters for the benchmark
        """

        #first update the noise for the given function
        return_function_name, return_function_creator = self.function_name_map[id_]
        return_function = return_function_creator(noise, mu, sigma)

        return (return_function_name, return_function), self.get_params(return_function, id_)

    def get_function_by_name(self, name, noise=False, mu=0., sigma=0.01):
        """
        Get the benchmarked function with given id
        :param name_: Function name in self.function_name_map
        :param noise: Indicates whether the function should provide noisy values
        :param mu: mean of the noise
        :param sigma: convariance of the noise

        :return function_name_map entry for the given id and the parameters for the benchmark
        """
        try:
            id_ = self.function_name_index_map[name]
        except KeyError:
            raise ValueError('There exists no function by name {}'.format(name))
        return_function_name, return_function_creator = self.function_name_map[id_]
        return_function = return_function_creator(noise, mu, sigma)

        return (return_function_name, return_function), self.get_params(return_function, id_)

    def get_params(self, fg_object, id):
        params_dict_items = [("benchmark_id", id)]
        function_params_items = fg_object.get_params().items()
        params_dict_items += function_params_items
        return OrderedDict(params_dict_items)

    def _create_rastrigin2d(self, noise, mu, sigma):
        return FunctionGenerator([RastriginParameters()],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_rastrigin10d(self, noise, mu, sigma):
        return FunctionGenerator([RastriginParameters()],
                                 dims=10, noise=noise, mu=mu, sigma=sigma)

    def _create_rosenbrock2d(self, noise, mu, sigma):
        return FunctionGenerator([RosenbrockParameters()],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_rosenbrock10d(self, noise, mu, sigma):
        return FunctionGenerator([RosenbrockParameters()],
                                 dims=10, noise=noise, mu=mu, sigma=sigma)

    def _create_ackley2d(self, noise, mu, sigma):
        return FunctionGenerator([AckleyParameters()],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_ackley10d(self, noise, mu, sigma):
        return FunctionGenerator([AckleyParameters()],
                                 dims=10, noise=noise, mu=mu, sigma=sigma)

    def _create_chasm(self, noise, mu, sigma):
        return FunctionGenerator([ChasmParameters()],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_gauss2d(self, noise, mu, sigma):
        return FunctionGenerator([GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.])],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_shekel2d(self, noise, mu, sigma):
        return FunctionGenerator([ShekelParameters(A='default', c='default')],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_michalewicz2d(self, noise, mu, sigma):
        return FunctionGenerator([MichalewiczParameters(m='default')],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_permutation2d(self, noise, mu, sigma):
        return FunctionGenerator([PermutationParameters(beta=0.005)],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_easom2d(self, noise, mu, sigma):
        return FunctionGenerator([EasomParameters()],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_easom10d(self, noise, mu, sigma):
        return FunctionGenerator([EasomParameters()],
                                 dims=10, noise=noise, mu=mu, sigma=sigma)

    def _create_langermann(self, noise, mu, sigma):
        return FunctionGenerator([LangermannParameters(A='default', c='default')],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_3gaussians2d(self, noise, mu, sigma):
        fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                     GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                     GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
        return FunctionGenerator(fg_params, dims=2, noise=noise, mu=mu, sigma=sigma)
