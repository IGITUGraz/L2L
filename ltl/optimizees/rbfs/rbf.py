import numpy as np


class RBF:
    def __init__(self, rbf_params, dims=2, bound_min=0., bound_max=10., noise=False, mu=0., sigma=0.01):
        cost_functions = {}
        self.bound = [bound_min, bound_max]
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        name_list = ['gaussian', 'permutation', 'easom', 'langermann', 'michalewicz', 'shekel']
        function_list = [Gaussian, Permutation, Easom, Langermann, Michalewicz, Shekel]

        # Create a dictionary which associate the function and state bound to a cost name
        for n, f in zip(name_list, function_list):
            cost_functions[n] = f

        self.gen_functions = []
        for param in rbf_params:
            f_name = param["name"]
            function_class = cost_functions[f_name](param["params"], dims)
            self.gen_functions.append(function_class.call)

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


class Shekel:
    def __init__(self, params, dims):
        if params is None and dims == 2:
            self.c = (1./10.) * np.array([1, 2, 5, 2, 3, 1, 1])
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

    def call(self, x):
        x = np.array(x)
        value = 0
        for i in range(self.A.shape[0]):
            sum_diff_sq = np.sum((x - self.A[i])**2 + self.c[i]) ** -1
            value += sum_diff_sq
        return -value


class Michalewicz:
    def __init__(self, params, dims):
        if params is None:
            self.m = 10
        else:
            self.m = params["m"]

        self.dims = dims

    def call(self, x):
        x = np.array(x)
        i = np.arange(1, self.dims+1)
        a = (i * x**2)/np.pi
        b = np.sin(a)**(2*self.m)
        value = -np.sum(np.sin(x) * b)
        return value


class Langermann:
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

    def call(self, x):
        x = np.array(x)
        value = 0
        for i in range(self.A.shape[0]):
            sum_diff_sq = np.sum((x - self.A[i])**2)
            value += self.c[i] * np.exp((-1/np.pi) * sum_diff_sq) * np.cos(np.pi * sum_diff_sq)
        return value


class Easom:
    def __init__(self, params, dims):
        if params is not None:
            raise Exception("Function does not take parameters.")

        self.dims = dims

    def call(self, x):
        x = np.array(x)
        cos_x = np.cos(x)
        x_min_pi = (x - np.pi)**2
        value = -cos_x.prod() * np.exp(-np.sum(x_min_pi))
        return value


class Permutation:
    def __init__(self, params, dims):
        if not len(params) == 1:
            raise Exception("Number of parameters does not equal 1.")
        beta = np.array(params["beta"])

        if np.isscalar(beta):
            raise Exception("Beta paramater must always be a scalar value.")
        self.dims = dims
        self.beta = beta

    def call(self, x):
        # sum_{k=1}^n (sum_{i=1}^n [i^k+beta][(x_i/i)^k-1])^2
        x = np.array(x)
        ks = np.array(range(1, self.dims+1))
        i = np.array(range(1, self.dims+1))
        value = np.array([np.sum((i**k + self.beta) * ((x / i)**k - 1), axis=0) for k in ks])
        value = np.sum(value**2)
        return value


class Gaussian:
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

    def call(self, x):
        x = np.array(x)
        value = 1 / np.sqrt((2*np.pi)**self.dims * np.linalg.det(self.sigma))
        value = value * np.exp(-0.5 * (np.transpose(x - self.mean).dot(np.linalg.inv(self.sigma))).dot((x - self.mean)))
        return value
