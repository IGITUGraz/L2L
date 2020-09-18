import unittest

from l2l.utils.experiment import Experiment

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from collections import namedtuple


class OptimizerTestCase(unittest.TestCase):

    def setUp(self):
        # Test function
        function_id = 14
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)


        self.experiment = Experiment(root_dir_path='/home/sandra/Documents/L2L/results')
        jube_params = {}
        self.trajectory, all_jube_params = self.experiment.prepare_experiment(name='L2L',
                                                              log_stdout=True,
                                                              jube_parameter = jube_params)
        self.optimizee_parameters = namedtuple('OptimizeeParameters', [])
        self.optimizee = FunctionGeneratorOptimizee(self.trajectory,benchmark_function, seed=1)
