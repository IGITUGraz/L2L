from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict

import numpy as np


class FunctionGenerator:
    """
    Implements a function generator that generates parametrized test functions from the given set of parameters.
    Given the list of function descriptions in the constructor, the cost_function represents the sum of all those
    functions.

    :param fg_params: List of function parameters. Each parameter is a namedtuple that describes a single test function.
    :param dims: Dimensionality of the functions.
    :param bound: Two element list containing minimum and maximum value of function input. If None, the bound is
                  computed from default bounds of given functions.
    :param noise: Boolean value indicating if the Gaussian noise will be applied on the resulting function.
    :param mu: Scalar indicating the mean of the Gaussian noise.
    :param sigma: Scalar indicating the standard deviation of the Gaussian noise.
    """

    def __init__(self, fg_params, dims=2, bound=None, noise=False, mu=0., sigma=0.01):
        self.dims = dims
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.actual_optima = None
        cost_functions = dict(GaussianParameters=Gaussian,
                              PermutationParameters=Permutation,
                              EasomParameters=Easom,
                              LangermannParameters=Langermann,
                              MichalewiczParameters=Michalewicz,
                              ShekelParameters=Shekel,
                              RastriginParameters=Rastrigin,
                              RosenbrockParameters=Rosenbrock,
                              AckleyParameters=Ackley,
                              ChasmParameters=Chasm)

        self.gen_functions = []
        self.function_parameters = fg_params
        # The class name of the parameter named tuple indexes the actual function class,
        # which is initialized using the given param and dims
        for param in fg_params:
            function_class = cost_functions[param.__class__.__name__](param, dims)
            self.gen_functions.append(function_class)

        if bound is not None:
            self.bound = bound
        else:
            bounds_min = [function_class.bound[0] for function_class in self.gen_functions]
            bounds_max = [function_class.bound[1] for function_class in self.gen_functions]
            bound_min = np.min(bounds_min)
            bound_max = np.max(bounds_max)
            self.bound = [bound_min, bound_max]

    def cost_function(self, x, random_state=None):
        """It gets the value of the function. If the function includes noise, the `random_state`
        parameter must be specified

        :param ~numpy.random.RandomState random_state: The random generator used to generate the
            noise for the function.
        """
        res = 0.
        for f in self.gen_functions:
            res += f(x)

        if self.noise:
            assert isinstance(random_state, np.random.RandomState)
            res += random_state.normal(self.mu, self.sigma)

        return res

    def get_params(self):
        fg_params = []
        for param in self.function_parameters:
            fg_params.append({type(param).__name__: dict(param._asdict())})

        if self.noise:
            params_dict_items = [("dims", self.dims),
                                 ("mu", self.mu),
                                 ("sigma", self.sigma)]
        else:
            params_dict_items = [("dims", self.dims)]
        params_dict_items += [("functions", fg_params)]
        return OrderedDict(params_dict_items)


class Function(ABC):
    """
    Base class for all test functions.
    """

    @abstractmethod
    def __call__(self, x):
        """
        :param x: input data vector with length equal to the function dimensionality
        :return: the resulting scalar output of the function
        """
        pass


ShekelParameters = namedtuple('ShekelParameters', ['A', 'c'])
ShekelParameters.__doc__ = """
:param A: matrix m*n of coordinates of all minima (m equals length of c, and n equals dims)
:param c: list of inverse intensities of minima
"""


class Shekel(Function):
    """
    The Shekels (foxholes) function has variable number of local minima (length of c or A).
    It can be customized by defining the coordinates of minima in matrix A,
    and inverse of their intensities in c.
    reference: https://www.sfu.ca/~ssurjano/shekel.html

    :param params: Instance of :func:`~collections.namedtuple` :class:`ShekelParameters`
    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        if params.A == 'default' and params.c == 'default' and dims == 2:
            self.c = (1. / 10.) * np.array([1, 2, 5, 2, 3, 1, 1])
            self.A = np.array([[3, 5],
                               [5, 2],
                               [2, 1],
                               [3, 3],
                               [2, 7],
                               [1, 4],
                               [7, 9]])
        else:
            self.c = np.array(params.c)
            self.A = np.array(params.A)
            if self.c.size != self.A.shape[0]:
                raise Exception("Parameters A and c do not match.")
            if self.A.shape[1] != dims:
                raise Exception("Shape of parameter A does not match the dimensionality.")

        self.dims = dims
        self.bound = [0, 10]

    def __call__(self, x):
        x = np.array(x)
        value = 0
        for i in range(self.A.shape[0]):
            sum_diff_sq = np.sum((x - self.A[i]) ** 2 + self.c[i]) ** -1
            value += sum_diff_sq
        return -value


MichalewiczParameters = namedtuple('MichalewiczParameters', ['m'])
MichalewiczParameters.__doc__ = """
:param m: steepness factor
"""


class Michalewicz(Function):
    """
    The Michalewicz function is multimodal with number of local minima equal to factoriel of the number of dimensions.
    It accepts only a parameter m which defines the steepness of the valleys and ridges. Larger m leads to a more
    difficult function to minimize. The recommended value for m is 10 which is defined if no parameters are given.
    reference: https://www.sfu.ca/~ssurjano/michal.html

    :param params: Instance of :func:`~collections.namedtuple` :class:`MichalewiczParameters`
    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        if params.m == 'default':
            self.m = 10
        else:
            self.m = params.m

        self.dims = dims
        self.bound = [0, np.pi]

    def __call__(self, x):
        x = np.array(x)
        i = np.arange(1, self.dims + 1)
        a = (i * x ** 2) / np.pi
        b = np.sin(a) ** (2 * self.m)
        value = -np.sum(np.sin(x) * b)
        return value


LangermannParameters = namedtuple('LangermannParameters', ['A', 'c'])
LangermannParameters.__doc__ = """
:param A: matrix m*n of coordinates of all minima, (m equals length of c, and n equals dims)
:param c: list of intensities of minima
"""


class Langermann(Function):
    """
    The Langermann function is multimodal, with many unevenly distributed local minima.
    It can be customized by defining the coordinates of centers of sub-functions in matrix A,
    and their intensities in c.
    reference: https://www.sfu.ca/~ssurjano/langer.html

    :param params: Instance of :func:`~collections.namedtuple` :class:`LangermannParameters`
    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        if params.A == 'default' and params.c == 'default' and dims == 2:
            self.c = np.array([1, 2, 5, 2, 3])
            self.A = np.array([[3, 5],
                               [5, 2],
                               [2, 1],
                               [1, 4],
                               [7, 9]])
        else:
            self.c = np.array(params.c)
            self.A = np.array(params.A)
            if self.c.size != self.A.shape[0]:
                raise Exception("Parameters A and c do not match.")
            if self.A.shape[1] != dims:
                raise Exception("Shape of parameter A does not match the dimensionality.")

        self.dims = dims
        self.bound = [0, 10]

    def __call__(self, x):
        x = np.array(x)
        value = 0
        for i in range(self.A.shape[0]):
            sum_diff_sq = np.sum((x - self.A[i]) ** 2)
            value += self.c[i] * np.exp((-1 / np.pi) * sum_diff_sq) * np.cos(np.pi * sum_diff_sq)
        return value


EasomParameters = namedtuple('EasomParameters', [])


class Easom(Function):
    """
    The Easom function has several local minima. It is unimodal,
    and the global minimum has a small area relative to the search space.
    reference: https://www.sfu.ca/~ssurjano/easom.html

    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        self.dims = dims
        self.bound = [-10, 10]

    def __call__(self, x):
        x = np.array(x)
        cos_x = np.cos(x)
        x_min_pi = (x - np.pi) ** 2
        value = -cos_x.prod() * np.exp(-np.sum(x_min_pi))
        return value


PermutationParameters = namedtuple('PermutationParameters', ['beta'])
PermutationParameters.__doc__ = """
:param beta: non-negative, difference between global and local minima (smaller means harder)
"""


class Permutation(Function):
    """
    beta is a non-negative parameter. The smaller beta, the more difficult problem becomes since the global minimum
    is difficult to distinguish from local minima near permuted solutions. For beta=0, every permuted solution is a
    global minimum, too.
    This problem therefore appear useful to test the ability of a global minimization algorithm to reach the global
    minimum successfully and to discriminate it from other local minima.
    reference: http://solon.cma.univie.ac.at/glopt/my_problems.html

    :param params: Instance of :func:`~collections.namedtuple` :class:`PermutationParameters`
    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        if not len(params) == 1:
            raise Exception("Number of parameters does not equal 1.")
        beta = np.array(params.beta)

        if np.isscalar(beta):
            raise Exception("Beta paramater must always be a scalar value.")
        self.dims = dims
        self.beta = beta
        self.bound = [-dims, dims]

    def __call__(self, x):
        x = np.array(x)
        ks = np.array(range(1, self.dims + 1))
        i = np.array(range(1, self.dims + 1))
        value = np.array([np.sum((i ** k + self.beta) * ((x / i) ** k - 1), axis=0) for k in ks])
        value = np.sum(value ** 2)
        return value


GaussianParameters = namedtuple('GaussianParameters', ['sigma', 'mean'])
GaussianParameters.__doc__ = """
:param sigma: covariance matrix
:param mean: list containing coordinates of the peak (mean, median, mode)
"""


class Gaussian(Function):
    """
    The multi-dimensional Gaussian (normal) distribution function.

    :param params: Instance of :func:`~collections.namedtuple` :class:`GaussianParameters`
    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        if not len(params) == 2:
            raise Exception("Number of parameters does not equal 2.")
        sigma = np.array(params.sigma)
        mean = np.array(params.mean)

        if (dims > 1 and (not sigma.shape[0] == sigma.shape[1] == mean.shape[0] == dims)) or \
                (dims == 1 and (not sigma.shape == mean.shape == tuple())):
            raise Exception("Shapes do not match the given dimensionality.")
        self.dims = dims
        self.sigma = sigma
        self.mean = mean
        self.bound = [-5, 5]

    def __call__(self, x):
        x = np.array(x)
        value = 1 / np.sqrt((2 * np.pi) ** self.dims * np.linalg.det(self.sigma))
        value = value * np.exp(-0.5 * (np.transpose(x - self.mean).dot(np.linalg.inv(self.sigma))).dot((x - self.mean)))
        return -value


RastriginParameters = namedtuple('RastriginParameters', [])


class Rastrigin(Function):
    """
    Rastrigin function is a multimodal function with a large number of local minima.
    reference: https://www.sfu.ca/%7Essurjano/rastr.html

    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        self.dims = dims
        self.bound = [-5, 5]

    def __call__(self, x):
        x = np.array(x)
        return np.sum(x ** 2 + 10 - 10 * np.cos(2 * np.pi * x))


RosenbrockParameters = namedtuple('RosenbrockParameters', [])


class Rosenbrock(Function):
    """
    Rosenbrock function is a unimodal valley-shaped function.
    reference: https://www.sfu.ca/~ssurjano/rosen.html

    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        self.dims = dims
        self.bound = [-2, 2]

    def __call__(self, x):
        x = np.array(x)
        x_1 = x[1:self.dims]
        x_0 = x[0:self.dims - 1]
        value = 100 * (x_1 - x_0 ** 2) ** 2 + (1 - x_0) ** 2
        value = sum(value)
        return value


AckleyParameters = namedtuple('AckleyParameters', [])


class Ackley(Function):
    """
    Ackley function has a large hole in at the centre surrounded by small hill like regions. Algorithms can get
    trapped in one of its many local minima.
    reference: https://www.sfu.ca/~ssurjano/ackley.html

    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        self.dims = dims
        self.bound = [-2, 2]

    def __call__(self, x):
        x = np.array(x)
        return np.exp(1) + 20 - 20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.dims)) \
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / self.dims)


ChasmParameters = namedtuple('ChasmParameters', [])


class Chasm(Function):
    """
    Chasm is characterized by a large flat area with a very large slope that halves the two parts of the
    function.

    :param dims: dimensionality of the function
    """

    def __init__(self, params, dims):
        if dims != 2:
            raise Exception("Dimensionality of the function must equal 2.")

        self.dims = dims
        self.bound = [-2, 2]

    def __call__(self, x):
        x = np.array(x)
        return 1e3 * np.abs(x[0]) / (1e3 * np.abs(x[0]) + 1) + 1e-2 * np.abs(x[1])
