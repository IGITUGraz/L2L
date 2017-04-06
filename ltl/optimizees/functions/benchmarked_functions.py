from ltl.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, PermutationParameters, \
    EasomParameters, LangermannParameters, MichalewiczParameters, ShekelParameters, RastriginParameters, \
    RosenbrockParameters, ChasmParameters, AckleyParameters


class BenchmarkedFunctions:
    def __init__(self):
        self.function_names = ["Rastrigin2d", "Chasm", "Shekel2d", "Michalewicz2d", "3Gaussians2d"]
        self.functions = [self.__create_rastrigin__, self.__create_chasm__, self.__create_shekel__,
                          self.__create_michalewicz__, self.__create_3gaussians__]

    def get_function_by_index(self, ind):
        function = self.functions[ind]()
        function_name = self.function_names[ind]
        return function, function_name


    def __create_rastrigin__(self):
        return FunctionGenerator([RosenbrockParameters()], dims=2)

    def __create_chasm__(self):
        return FunctionGenerator([ChasmParameters()], dims=2)

    def __create_shekel__(self):
        return FunctionGenerator([ShekelParameters(A='default', c='default')], dims=2)

    def __create_michalewicz__(self):
        return FunctionGenerator([MichalewiczParameters(m='default')], dims=2)

    def __create_3gaussians__(self):
        fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                     GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                     GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
        return FunctionGenerator(fg_params, dims=2, noise=True)

