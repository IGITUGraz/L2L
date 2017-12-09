from __future__ import absolute_import
from ltl.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, \
    MichalewiczParameters, ShekelParameters, EasomParameters, LangermannParameters, \
    RastriginParameters, ChasmParameters, RosenbrockParameters, AckleyParameters, PermutationParameters
from collections import OrderedDict


class BenchmarkedFunctions(object):
    u"""
    Implements benchmarked functions class for an easier call of the benchmarked functions

    """
    def __init__(self):
        self.function_name_map = [(u"Rastrigin2d", self._create_rastrigin2d),
                                  (u"Rastrigin10d", self._create_rastrigin10d),
                                  (u"Rosenbrock2d", self._create_rosenbrock2d),
                                  (u"Rosenbrock10d", self._create_rosenbrock10d),
                                  (u"Ackley2d", self._create_ackley2d),
                                  (u"Ackley10d", self._create_ackley10d),
                                  (u"Chasm", self._create_chasm),
                                  (u"Gauss2d", self._create_gauss2d),
                                  (u"Shekel2d", self._create_shekel2d),
                                  (u"Langermann2d", self._create_langermann),
                                  (u"Michalewicz2d", self._create_michalewicz2d),
                                  (u"Permutation2d", self._create_permutation2d),
                                  (u"Easom2d", self._create_easom2d),
                                  (u"Easom10d", self._create_easom10d),
                                  (u"3Gaussians2d", self._create_3gaussians2d)]
        self.function_name_index_map = OrderedDict([(name, index)
                                                    for index, (name, _) in enumerate(self.function_name_map)])

    def get_function_by_index(self, id_, noise=False, mu=0., sigma=0.01):
        u"""
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
        u"""
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
            raise ValueError(u'There exists no function by name {}'.format(name))
        return_function_name, return_function_creator = self.function_name_map[id_]
        return_function = return_function_creator(noise, mu, sigma)

        return (return_function_name, return_function), self.get_params(return_function, id_)

    def get_params(self, fg_object, id):
        params_dict_items = [(u"benchmark_id", id)]
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
        return FunctionGenerator([ShekelParameters(A=u'default', c=u'default')],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_michalewicz2d(self, noise, mu, sigma):
        return FunctionGenerator([MichalewiczParameters(m=u'default')],
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
        return FunctionGenerator([LangermannParameters(A=u'default', c=u'default')],
                                 dims=2, noise=noise, mu=mu, sigma=sigma)

    def _create_3gaussians2d(self, noise, mu, sigma):
        fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                     GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                     GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
        return FunctionGenerator(fg_params, dims=2, noise=noise, mu=mu, sigma=sigma)
