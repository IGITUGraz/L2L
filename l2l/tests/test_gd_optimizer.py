import unittest

import numpy as np
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.utils.experiment import Experiment

from l2l import list_to_dict


class GDOptimizerTestCase(OptimizerTestCase):

    def test_gd(self):
        optimizer_parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                       n_random_steps=1, momentum_decay=0.5,
                                       n_iteration=1, stop_criterion=np.Inf, seed=99)

        optimizer = GradientDescentOptimizer(self.trajectory,
                                             optimizee_create_individual=self.optimizee.create_individual,
                                             optimizee_fitness_weights=(0.1,),
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
        print(self.experiment.optimizer)
        best = list_to_dict(self.experiment.optimizer.current_individual.tolist(),
                             self.experiment.optimizer.optimizee_individual_dict_spec)['coords']
        self.assertEqual(best[0],-4.998856251826551)
        self.assertEqual(best[1],-1.9766742736816023)
        self.experiment.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(GDOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()