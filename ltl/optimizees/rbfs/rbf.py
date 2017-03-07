import numpy as np


class RBF:
    def __init__(self, rbf_params, dims):
        cost_functions = {}
        self.bound = [-5, 5]
        name_list = ['gaussian']
        function_list = [Gaussian]

        # Create a dictionnary which associate the function and state bound to a cost name
        for n, f in zip(name_list, function_list):
            cost_functions[n] = f

        self.gen_function = []
        for param in rbf_params:
            f_name = param["name"]
            function_class = cost_functions[f_name](param["params"], dims)
            self.gen_function.append(function_class.call)

    def cost_function(self, x):
        res = 0.
        for f in self.gen_function:
            res += f(x)

        return res


class Gaussian:
    def __init__(self, params, dims):
        if not len(params) == 2:
            raise Exception("Number of paramaters does not equal 2.")
        sigma = np.array(params[0])
        mean = np.array(params[1])

        if (dims > 1 and (not sigma.shape[0] == sigma.shape[1] == mean.shape[0] == dims)) or \
                (dims == 1 and (not sigma.shape == mean.shape == tuple() )):
            raise Exception("Shapes do not match the given dimensionality.")
        self.dims = dims
        self.sigma = sigma
        self.mean = mean

    def call(self, x):
        x = np.array(x)
        value = 1 / np.sqrt((2*np.pi)**self.dims * np.linalg.det(self.sigma))
        value = value * np.exp(-0.5 * (np.transpose(x - self.mean).dot(np.linalg.inv(self.sigma))).dot((x - self.mean)))
        print("rbfcall")
        return value
#
# rbf_params = [{'name' : 'gaussian', 'params' : [ [[1.,0.],[0.,1.]], [1.,1.]]}]
# inst = RBF(rbf_params, 2)
# func = RBF.cost_function
# print(func(inst,x=[0,0]))
# print(func(inst,x=[1,1]))
# print(func(inst,x=[2,2]))
