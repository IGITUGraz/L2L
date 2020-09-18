import unittest

from l2l.tests.test_optimizer import OptimizerTestCase

from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters

from l2l import list_to_dict


class GSOptimizerTestCase(OptimizerTestCase):

    def test_gd(self):
        n_grid_divs_per_axis = 2
        optimizer_parameters = GridSearchParameters(param_grid={
            'coords': (self.optimizee.bound[0], self.optimizee.bound[1], n_grid_divs_per_axis)
        })
        optimizer = GridSearchOptimizer(self.trajectory, optimizee_create_individual=self.optimizee.create_individual,
                                        optimizee_fitness_weights=(-0.1,),
                                        parameters=optimizer_parameters)
        self.assertIsNotNone(optimizer.parameters)

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
    suite = unittest.makeSuite(GSOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()