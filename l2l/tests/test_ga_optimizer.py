import unittest

from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters


class GAOptimizerTestCase(OptimizerTestCase):

    def test_setup(self):

        optimizer_parameters = GeneticAlgorithmParameters(seed=0, pop_size=1, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=1, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1
                                                          )

        optimizer = GeneticAlgorithmOptimizer(self.trajectory, optimizee_create_individual=self.optimizee.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters)

        self.assertIsNotNone(optimizer.parameters)
        self.assertIsNotNone(self.experiment)

        try:

            self.experiment.run_experiment(optimizee=self.optimizee,
                                           optimizee_parameters=self.optimizee_parameters,
                                           optimizer=optimizer,
                                           optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(Exception.__name__)
        best = self.experiment.optimizer.best_individual['coords']
        self.assertEqual(best[0], -4.998856251826551)
        self.assertEqual(best[1], -1.9766742736816023)
        self.experiment.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(GAOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
