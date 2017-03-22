from abc import ABC, abstractmethod
import numpy as np


class FunctionGenerator:
    """
    Implements a function generator that generates parametrized test functions from the given set of parameters.
    Given the list of function descriptions in the constructor, the cost_function represents the sum of all those
    functions.

    :param fg_params: List of parameters. Each parameter is a dictionary that describes a single test function.
                       Each dictionary contains two keys ("name" and "params"), first expecting the name of the
                       function, and the second the list of function specific parameters.
    :param dims: Dimensionality of the functions.
    :param bound: Two element list containing minimum and maximum value of function input. If None, the bound is
                  computed from default bounds of given functions.
    :param noise: Boolean value indicating if the Gaussian noise will be applied on the resulting function.
    :param mu: Scalar indicating the mean of the Gaussian noise.
    :param sigma: Scalar indicating the standard deviation of the Gaussian noise.
    """
    def __init__(self, fg_params, dims=2, bound=None, noise=False, mu=0., sigma=0.01):
        cost_functions = {}
        self.dims = dims
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        name_list = ['gaussian', 'permutation', 'easom', 'langermann', 'michalewicz', 'shekel']
        function_list = [Gaussian, Permutation, Easom, Langermann, Michalewicz, Shekel]

        # Create a dictionary which associate the function and state bound to a cost name
        for n, f in zip(name_list, function_list):
            cost_functions[n] = f

        self.gen_functions = []
        gen_functions_classes = []
        for param in fg_params:
            f_name = param["name"]
            function_class = cost_functions[f_name](param["params"], dims)
            self.gen_functions.append(function_class.call)
            gen_functions_classes.append(function_class)

        if bound is not None:
            self.bound = bound
        else:
            bounds_min = [function_class.bound[0] for function_class in gen_functions_classes]
            bounds_max = [function_class.bound[1] for function_class in gen_functions_classes]
            bound_min = np.min(bounds_min)
            bound_max = np.max(bounds_max)
            self.bound = [bound_min, bound_max]

    def cost_function(self, x):
        res = 0.
        for f in self.gen_functions:
            res += f(x)

        if self.noise:
            res += np.random.normal(self.mu, self.sigma)

        return res

    def plot(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(self.bound[0], self.bound[1], 0.05)
        Y = np.arange(self.bound[0], self.bound[1], 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = [self.cost_function([x, y]) for x, y in zip(X.ravel(), Y.ravel())]
        # Z = []
        # i = 0
        # # print(self.cost_function([1., 2., 3.]))
        # print("computing cost function [", sep='', end='')
        # for x, y in zip(X.ravel(), Y.ravel()):
        #     Z.append(self.cost_function([x, y]))
        #     i += 1
        #     if i % (X.size / 100) == 0:
        #         perc = (i / X.size) * 100
        #         # print("%d%%" % (perc,))
        #         print(".", sep='', end='', flush=True)
        # print("]")
        Z = np.array(Z).reshape(X.shape)
        # print(Z)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


class TestFunction(ABC):
    """
    Base class for all test functions.
    """
    @abstractmethod
    def call(self, x):
        """
        :param x: input data vector with length equal to the function dimensionality
        :return: the resulting scalar output of the function
        """
        pass


class Shekel(TestFunction):
    """
    The Shekels (foxholes) function has variable number of local minima (length of c or A).
    It can be customized by defining the coordinates of minima in matrix A,
    and inverse of their intensities in c.
    reference: https://www.sfu.ca/~ssurjano/shekel.html

    :param params: a dictionary containing:
        :key "A": matrix m*n of coordinates of all minima (m equals length of c, and n equals dims)
        :key "c": list of inverse intensities of minima
    :param dims: dimensionality of the function
    """
    def __init__(self, params, dims):
        if params is None and dims == 2:
            self.c = (1. / 10.) * np.array([1, 2, 5, 2, 3, 1, 1])
            self.A = np.array([[3, 5],
                               [5, 2],
                               [2, 1],
                               [3, 3],
                               [2, 7],
                               [1, 4],
                               [7, 9]])
        elif params is not None:
            self.c = np.array(params["c"])
            self.A = np.array(params["A"])
            if self.c.size != self.A.shape[0]:
                raise Exception("Parameters A and c do not match.")
            if self.A.shape[1] != dims:
                raise Exception("Shape of parameter A does not match the dimensionality.")
        else:
            raise Exception("Parameters not defined.")

        self.dims = dims
        self.bound = [0, 10]

    def call(self, x):
        x = np.array(x)
        value = 0
        for i in range(self.A.shape[0]):
            sum_diff_sq = np.sum((x - self.A[i])**2 + self.c[i]) ** -1
            value += sum_diff_sq
        return -value


class Michalewicz(TestFunction):
    """
    The Michalewicz function is multimodal with number of local minima equal to factoriel of the number of dimensions.
    It accepts only a parameter m which defines the steepness of the valleys and ridges. Larger m leads to a more
    difficult function to minimize. The recommended value for m is 10 which is defined if no parameters are given.
    reference: https://www.sfu.ca/~ssurjano/michal.html

    :param params: a dictionary containing:
        :key "m": steepness factor
    :param dims: dimensionality of the function
    """
    def __init__(self, params, dims):
        if params is None:
            self.m = 10
        else:
            self.m = params["m"]

        self.dims = dims
        self.bound = [0, np.pi]

    def call(self, x):
        x = np.array(x)
        i = np.arange(1, self.dims + 1)
        a = (i * x**2) / np.pi
        b = np.sin(a)**(2 * self.m)
        value = -np.sum(np.sin(x) * b)
        return value


class Langermann(TestFunction):
    """
    The Langermann function is multimodal, with many unevenly distributed local minima.
    It can be customized by defining the coordinates of centers of sub-functions in matrix A,
    and their intensities in c.
    reference: https://www.sfu.ca/~ssurjano/langer.html

    :param params: a dictionary containing:
        :key "A": matrix m*n of coordinates of all minima, (m equals length of c, and n equals dims)
        :key "c": list of intensities of minima
    :param dims: dimensionality of the function
    """
    def __init__(self, params, dims):
        if params is None and dims == 2:
            self.c = np.array([1, 2, 5, 2, 3])
            self.A = np.array([[3, 5],
                               [5, 2],
                               [2, 1],
                               [1, 4],
                               [7, 9]])
        elif params is not None:
            self.c = np.array(params["c"])
            self.A = np.array(params["A"])
            if self.c.size != self.A.shape[0]:
                raise Exception("Parameters A and c do not match.")
            if self.A.shape[1] != dims:
                raise Exception("Shape of parameter A does not match the dimensionality.")
        else:
            raise Exception("Parameters not defined.")

        self.dims = dims
        self.bound = [0, 10]

    def call(self, x):
        x = np.array(x)
        value = 0
        for i in range(self.A.shape[0]):
            sum_diff_sq = np.sum((x - self.A[i])**2)
            value += self.c[i] * np.exp((-1 / np.pi) * sum_diff_sq) * np.cos(np.pi * sum_diff_sq)
        return value


class Easom(TestFunction):
    """
    The Easom function has several local minima. It is unimodal,
    and the global minimum has a small area relative to the search space.
    reference: https://www.sfu.ca/~ssurjano/easom.html

    :param dims: dimensionality of the function
    """
    def __init__(self, params, dims):
        if params is not None:
            raise Exception("Function does not take parameters.")

        self.dims = dims
        self.bound = [-10, 10]

    def call(self, x):
        x = np.array(x)
        cos_x = np.cos(x)
        x_min_pi = (x - np.pi)**2
        value = -cos_x.prod() * np.exp(-np.sum(x_min_pi))
        return value


class Permutation(TestFunction):
    """
    beta is a non-negative parameter. The smaller beta, the more difficult problem becomes since the global minimum
    is difficult to distinguish from local minima near permuted solutions. For beta=0, every permuted solution is a
    global minimum, too.
    This problem therefore appear useful to test the ability of a global minimization algorithm to reach the global
    minimum successfully and to discriminate it from other local minima.
    reference: http://solon.cma.univie.ac.at/glopt/my_problems.html

    :param params: a dictionary containing:
        :key "beta": non-negative, difference between global and local minima (smaller means harder)
    :param dims: dimensionality of the function
    """
    def __init__(self, params, dims):
        if not len(params) == 1:
            raise Exception("Number of parameters does not equal 1.")
        beta = np.array(params["beta"])

        if np.isscalar(beta):
            raise Exception("Beta paramater must always be a scalar value.")
        self.dims = dims
        self.beta = beta
        self.bound = [-dims, dims]

    def call(self, x):
        x = np.array(x)
        ks = np.array(range(1, self.dims + 1))
        i = np.array(range(1, self.dims + 1))
        value = np.array([np.sum((i**k + self.beta) * ((x / i)**k - 1), axis=0) for k in ks])
        value = np.sum(value**2)
        return value


class Gaussian(TestFunction):
    """
    The multi-dimensional Gaussian (normal) distribution function.

    :param params: a dictionary containing:
        :key "sigma": variance matrix
        :key "mean": list containing coordinates of the peak (mean, median, mode)
    :param dims: dimensionality of the function
    """
    def __init__(self, params, dims):
        if not len(params) == 2:
            raise Exception("Number of parameters does not equal 2.")
        sigma = np.array(params["sigma"])
        mean = np.array(params["mean"])

        if (dims > 1 and (not sigma.shape[0] == sigma.shape[1] == mean.shape[0] == dims)) or \
                (dims == 1 and (not sigma.shape == mean.shape == tuple())):
            raise Exception("Shapes do not match the given dimensionality.")
        self.dims = dims
        self.sigma = sigma
        self.mean = mean
        self.bound = [-5, 5]

    def call(self, x):
        x = np.array(x)
        value = 1 / np.sqrt((2 * np.pi)**self.dims * np.linalg.det(self.sigma))
        value = value * np.exp(-0.5 * (np.transpose(x - self.mean).dot(np.linalg.inv(self.sigma))).dot((x - self.mean)))
        return -value
