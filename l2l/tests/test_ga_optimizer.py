import unittest

import numpy as np
from l2l.utils.environment import Environment
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee

class GAOptimizerTestCase(unittest.TestCase):

    def setUp(self):
        name = "test_trajectory"
        self.env = Environment(
            trajectory=name,
            filename=".",
            file_title='{} data'.format(name),
            comment='{} data'.format(name),
            add_time=True,
            automatic_storing=True,
            log_stdout=False,  # Sends stdout to logs
        )
        self.trajectory = self.env.trajectory
        ## Benchmark function
        function_id = 14
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee_seed = 1
        self.optimizee = FunctionGeneratorOptimizee(self.trajectory, benchmark_function, seed=optimizee_seed)

    def test_setup(self):

        parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5,
                                                MUTPB=0.3, NGEN=100, indpb=0.02,
                                                tournsize=15, matepar=0.5,
                                                mutpar=1
                                                )

        optimizer = GeneticAlgorithmOptimizer(self.trajectory, optimizee_create_individual=self.optimizee.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=parameters)

        self.assertIsNotNone(optimizer.parameters)
        try:
            optimizer.post_process()
        except Exception:
            self.fail()


def suite():
    suite = unittest.makeSuite(GAOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()