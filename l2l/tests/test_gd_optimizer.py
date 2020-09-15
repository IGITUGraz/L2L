import unittest

import numpy as np
from l2l.utils.environment import Environment
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee

class CEOptimizerTestCase(unittest.TestCase):

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
        self.traj = self.env.trajectory
        ## Benchmark function
        function_id = 14
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee_seed = 1
        self.optimizee = FunctionGeneratorOptimizee(self.traj, benchmark_function, seed=optimizee_seed)

    def test_setup(self):

        parameters = EvolutionStrategiesParameters(
        learning_rate=0.1,
        noise_std=1.0,
        mirrored_sampling_enabled=True,
        fitness_shaping_enabled=True,
        pop_size=1,
        n_iteration=1,
        stop_criterion=np.Inf,
        seed=1)

        optimizer = EvolutionStrategiesOptimizer(
        self.traj,
        optimizee_create_individual=self.optimizee.create_individual,
        optimizee_fitness_weights=(-1.,),
        parameters=parameters,
        optimizee_bounding_func=self.optimizee.bounding_func)

        self.assertIsNotNone(optimizer.parameters)
        try:
            optimizer.post_process()
        except Exception:
            self.fail()


def suite():
    suite = unittest.makeSuite(CEOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()