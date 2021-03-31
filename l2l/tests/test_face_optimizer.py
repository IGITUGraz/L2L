import unittest

import numpy as np
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.crossentropy.distribution import Gaussian
from l2l.optimizers.face import FACEOptimizer, FACEParameters


class FACEOptimizerTestCase(OptimizerTestCase):

    def test_setup(self):

        optimizer_parameters = FACEParameters(min_pop_size=2, max_pop_size=3, n_elite=1, smoothing=0.2, temp_decay=0,
                                    n_iteration=1,
                                    distribution=Gaussian(), n_expand=5, stop_criterion=np.inf, seed=1)
        optimizer = FACEOptimizer(self.trajectory, optimizee_create_individual=self.optimizee.create_individual,
                                  optimizee_fitness_weights=(-0.1,),
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
        best = self.experiment.optimizer.best_individual['coords']
        self.assertEqual(best[0], -4.998856251826551)
        self.assertEqual(best[1], -1.9766742736816023)
        self.experiment.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(FACEOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()