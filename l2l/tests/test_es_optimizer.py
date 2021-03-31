import unittest

import numpy as np
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer


class ESOptimizerTestCase(OptimizerTestCase):

    def test_setup(self):

        optimizer_parameters = EvolutionStrategiesParameters(
            learning_rate=0.1,
            noise_std=1.0,
            mirrored_sampling_enabled=True,
            fitness_shaping_enabled=True,
            pop_size=1,
            n_iteration=1,
            stop_criterion=np.Inf,
            seed=1)

        optimizer = EvolutionStrategiesOptimizer(
            self.trajectory,
            optimizee_create_individual=self.optimizee.create_individual,
            optimizee_fitness_weights=(-1.,),
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
        self.assertEqual(best[0], 0.7945654106889819)
        self.assertEqual(best[1], 1.5914885207715055)
        self.experiment.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(ESOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
