import unittest

import numpy as np
from l2l.utils.environment import Environment
from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee


class SAOptimizerTestCase(unittest.TestCase):

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

        parameters = SimulatedAnnealingParameters(n_parallel_runs=1, noisy_step=.03, temp_decay=.99, n_iteration=1,
                                                  stop_criterion=np.Inf, seed=np.random.randint(1e5),
                                                  cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)

        optimizer = SimulatedAnnealingOptimizer(self.trajectory, optimizee_create_individual=self.optimizee.create_individual,
                                                optimizee_fitness_weights=(-1,),
                                                parameters=parameters,
                                                optimizee_bounding_func=self.optimizee.bounding_func)

        self.assertIsNotNone(optimizer.parameters)
        self.env.add_postprocessing(optimizer.post_process)

        try:
            self.env.run(self.optimizee.simulate)
        except Exception:
            self.fail()
        optimizer.end(self.trajectory)


def suite():
    suite = unittest.makeSuite(SAOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()