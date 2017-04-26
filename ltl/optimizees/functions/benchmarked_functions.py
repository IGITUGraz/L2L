from ltl.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, \
    MichalewiczParameters, ShekelParameters, EasomParameters, LangermannParameters, \
    RastriginParameters, ChasmParameters, RosenbrockParameters, AckleyParameters, PermutationParameters


class BenchmarkedFunctions:
    """
    Implements benchmarked functions class for an easier call of the benchmarked functions

    :param noise: Boolean value indicating if the Gaussian noise will be applied on the resulting function.
    :param mu: Scalar indicating the mean of the Gaussian noise.
    :param sigma: Scalar indicating the standard deviation of the Gaussian noise.
    """
    def __init__(self):
        self.function_name_map = [("Rastrigin2d", self._create_rastrigin2d()),
                                  ("Rastrigin10d", self._create_rastrigin10d()),
                                  ("Rosenbrock2d", self._create_rosenbrock2d()),
                                  ("Rosenbrock10d", self._create_rosenbrock10d()),
                                  ("Ackley2d", self._create_ackley2d()),
                                  ("Ackley10d", self._create_ackley10d()),
                                  ("Chasm", self._create_chasm()),
                                  ("Gauss2d", self._create_gauss2d()),
                                  ("Shekel2d", self._create_shekel2d()),
                                  ("Langermann2d", self._create_langermann()),
                                  ("Michalewicz2d", self._create_michalewicz2d()),
                                  ("Permutation2d", self._create_permutation2d()),
                                  ("Easom2d", self._create_easom2d()),
                                  ("Easom10d", self._create_easom10d()),
                                  ("3Gaussians2d", self._create_3gaussians2d())]

    def get_function_by_index(self, id, noise=False, mu=0., sigma=0.01):
        """
        Get the benchmarked function with given id
        :param id: Function id
        :param noise: Indicates whether the function should provide noisy values
        :param mu: mean of the noise
        :param sigma: convariance of the noise

        :return function_name_map entry for the given id and the parameters for the benchmark
        """

        #first update the noise for the given function
        self.function_name_map[id][1].noise = noise
        self.function_name_map[id][1].mu = mu
        self.function_name_map[id][1].sigma = sigma

        return self.function_name_map[id], self._parse_parameters(id, noise, mu, sigma)

    def _parse_parameters(self, id, noise, mu, sigma):
        if noise:
            params_dict = {"benchmark_id": id,
                           "mu": mu,
                           "sigma": sigma}
        else:
            params_dict = {"id": id}
        return params_dict

    def _create_rastrigin2d(self):
        return FunctionGenerator([RastriginParameters()], dims=2)

    def _create_rastrigin10d(self):
        return FunctionGenerator([RastriginParameters()], dims=10)

    def _create_rosenbrock2d(self):
        return FunctionGenerator([RosenbrockParameters()], dims=2)

    def _create_rosenbrock10d(self):
        return FunctionGenerator([RosenbrockParameters()], dims=10)

    def _create_ackley2d(self):
        return FunctionGenerator([AckleyParameters()], dims=2)

    def _create_ackley10d(self):
        return FunctionGenerator([AckleyParameters()], dims=10)

    def _create_chasm(self):
        return FunctionGenerator([ChasmParameters()], dims=2)

    def _create_gauss2d(self):
        return FunctionGenerator([GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.])], dims=2)

    def _create_shekel2d(self):
        return FunctionGenerator([ShekelParameters(A='default', c='default')], dims=2)

    def _create_michalewicz2d(self):
        return FunctionGenerator([MichalewiczParameters(m='default')], dims=2)

    def _create_permutation2d(self):
        return FunctionGenerator([PermutationParameters(beta=0.005)], dims=2)

    def _create_easom2d(self):
        return FunctionGenerator([EasomParameters()], dims=2)

    def _create_easom10d(self):
        return FunctionGenerator([EasomParameters()], dims=10)

    def _create_langermann(self):
        return FunctionGenerator([LangermannParameters(A='default', c='default')], dims=2)

    def _create_3gaussians2d(self):
        fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                     GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                     GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
        return FunctionGenerator(fg_params, dims=2)
