import unittest

from l2l.tests.test_optimizer import OptimizerTestCase
import numpy as np
from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules


class SAOptimizerTestCase(OptimizerTestCase):

    def test_sa(self):
        optimizer_parameters = SimulatedAnnealingParameters(n_parallel_runs=1, noisy_step=.03, temp_decay=.99, n_iteration=1,
                                                  stop_criterion=np.Inf, seed=np.random.randint(1e5),
                                                  cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)

        optimizer = SimulatedAnnealingOptimizer(self.trajectory, optimizee_create_individual=self.optimizee.create_individual,
                                                optimizee_fitness_weights=(-1,),
                                                parameters=optimizer_parameters,
                                                optimizee_bounding_func=self.optimizee.bounding_func)
        self.assertIsNotNone(optimizer.parameters)
        self.assertIsNotNone(self.experiment)

        try:

            self.experiment.run_experiment(optimizee=self.optimizee,
                                  optimizee_parameters=self.optimizee_parameters,
                                  optimizer=optimizer,
                                  optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(e.__name__)

def suite():
    suite = unittest.makeSuite(SAOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()