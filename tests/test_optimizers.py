import logging
import unittest
from unittest.mock import MagicMock

import numpy as np

from ltl import DummyTrajectory, dict_to_list
from ltl.optimizees.functions.function_generator import FunctionGenerator, QuadraticParameters
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('test-optimizers')


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.fn = FunctionGenerator([QuadraticParameters(a=1, b=0, c=0)], dims=1, mu=0., sigma=0.)
        self.optimizee_traj = MagicMock()
        self.optimizee = FunctionGeneratorOptimizee(self.optimizee_traj, self.fn, seed=100)
        logger.debug("All values of quadratic function are %s",
                     [(x, self.fn.cost_function(x)) for x in np.linspace(-1., 1., 11).tolist()])

    def _optimize(self, traj, optimizer):
        for i in range(traj.parameters.n_iteration):
            logger.debug("==== Evaluated population is %s", optimizer.eval_pop)
            results = []
            for j, pop in enumerate(optimizer.eval_pop):
                results.append((i, self.fn.cost_function(dict_to_list(pop))))
            optimizer.post_process(traj, results)
        return optimizer.eval_pop

    def test_simulated_annealing(self):
        from ltl.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters
        from ltl.optimizers.simulatedannealing.optimizer import AvailableCoolingSchedules
        from ltl.optimizers.simulatedannealing.optimizer import SimulatedAnnealingOptimizer

        traj = DummyTrajectory()

        parameters_default = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.01, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.DEFAULT)

        parameters_logarithmic = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.01, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.LOGARITHMIC)

        parameters_exponential = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.005, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.EXPONENTIAL)

        parameters_linear_multiplicative = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.01, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.LINEAR_MULTIPLICATIVE)

        parameters_quadratic_multiplicative = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.01, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.QUADRATIC_MULTIPLICATIVE)

        parameters_linear_adaptive = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.001, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.LINEAR_ADDAPTIVE)

        parameters_quadratic_adaptive = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.001, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)

        parameters_exponential_adaptive = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.001, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE)

        parameters_trignometric_adaptive = SimulatedAnnealingParameters(
            n_parallel_runs=10, noisy_step=.001, temp_decay=.99, n_iteration=1000, stop_criterion=np.Inf, seed=1234,
            cooling_schedule=AvailableCoolingSchedules.TRIGONOMETRIC_ADDAPTIVE)

        # for parameters in [parameters_default, parameters_logarithmic, parameters_exponential,
        #                    parameters_linear_multiplicative, parameters_quadratic_multiplicative,
        #                    parameters_linear_adaptive, parameters_quadratic_adaptive, parameters_exponential_adaptive,
        #                    parameters_trignometric_adaptive]:
        for parameters in [parameters_default, parameters_logarithmic, parameters_exponential,
                           parameters_linear_multiplicative, parameters_quadratic_multiplicative,
                           parameters_linear_adaptive, parameters_quadratic_adaptive, parameters_exponential_adaptive,
                           parameters_trignometric_adaptive]:
            optimizer = SimulatedAnnealingOptimizer(traj, optimizee_create_individual=self.optimizee.create_individual,
                                                    optimizee_fitness_weights=(-1000.,),
                                                    parameters=parameters,
                                                    optimizee_bounding_func=self.optimizee.bounding_func)

            final_values = self._optimize(traj, optimizer)
            logger.debug("Cooling schedule: %s, min value %f", parameters.cooling_schedule.name,
                         np.min(np.abs([dict_to_list(fv) for fv in final_values])))
            self.assertLess(np.min(np.abs([dict_to_list(fv) for fv in final_values])), 0.01,
                            "Fitness goal not reached for schedule {}".format(parameters.cooling_schedule.name))
            # for fitness in [self.fn.cost_function(dict_to_list(fv)) for fv in final_vales]:
            #     self.assertLess(fitness, 0.01,
            #                     "Fitness goal not reached for schedule {}".format(parameters.cooling_schedule.name))

    def test_gradient_descent(self):
        traj = DummyTrajectory()

        from ltl.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
        from ltl.optimizers.gradientdescent.optimizer import ClassicGDParameters
        from ltl.optimizers.gradientdescent.optimizer import AdamParameters
        from ltl.optimizers.gradientdescent.optimizer import StochasticGDParameters
        from ltl.optimizers.gradientdescent.optimizer import RMSPropParameters

        classic_gd_parameters = ClassicGDParameters(learning_rate=0.001, exploration_rate=0.001, n_random_steps=5,
                                                    n_iteration=100, stop_criterion=np.Inf, seed=42)

        adam_parameters = AdamParameters(learning_rate=0.01, exploration_rate=0.001, n_random_steps=5,
                                         first_order_decay=0.8, second_order_decay=0.8, n_iteration=100,
                                         stop_criterion=np.Inf,
                                         seed=42)

        sgd_parameters = StochasticGDParameters(learning_rate=0.001, stochastic_deviation=1., stochastic_decay=0.99,
                                                exploration_rate=0.001, n_random_steps=5, n_iteration=100,
                                                stop_criterion=np.Inf, seed=42)

        rms_prop_parameters = RMSPropParameters(learning_rate=0.001, exploration_rate=0.001, n_random_steps=5,
                                                momentum_decay=0.9,
                                                n_iteration=1000, stop_criterion=np.Inf, seed=42)

        for parameters in [classic_gd_parameters, adam_parameters, sgd_parameters, rms_prop_parameters]:
            variant = parameters.__class__.__name__.replace("Parameters", "")
            logger.info("Running for variant %s", variant)
            optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=self.optimizee.create_individual,
                                                 optimizee_fitness_weights=(10.,),
                                                 parameters=parameters,
                                                 optimizee_bounding_func=self.optimizee.bounding_func)

            final_values = self._optimize(traj, optimizer)
            logger.debug("variant: %s, min value %f", variant, np.min(np.abs([dict_to_list(fv) for fv in final_values])))
            self.assertLess(np.min(np.abs([dict_to_list(fv) for fv in final_values])), 0.01,
                            "Fitness goal not reached for schedule {}".format(parameters.cooling_schedule.name))

    def test_cross_entropy(self):
        traj = DummyTrajectory()

        from ltl.optimizers.crossentropy.optimizer import CrossEntropyParameters
        from ltl.optimizers.crossentropy.optimizer import CrossEntropyOptimizer
        from ltl.optimizers.crossentropy.distribution import NoisyGaussian

        parameters = CrossEntropyParameters(pop_size=50, rho=0.2, smoothing=0.0, temp_decay=0, n_iteration=5,
                                            distribution=NoisyGaussian(additive_noise=1,
                                                                       noise_decay=0.95),
                                            stop_criterion=np.inf, seed=102)

        optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=self.optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters,
                                          optimizee_bounding_func=self.optimizee.bounding_func)

        final_values = self._optimize(traj, optimizer)
        print(final_values)
        self.assertLess(np.min(np.abs([dict_to_list(fv) for fv in final_values])), 0.01,
                        "Fitness goal not reached")
        # for fitness in [self.fn.cost_function(dict_to_list(fv)) for fv in final_vales]:
        #     self.assertLess(fitness, 0.01, "Fitness goal not reached")
        # print("Points", [dict_to_list(fv) for fv in final_vales])
        # print("Fitnesses", [self.fn.cost_function(dict_to_list(fv)) for fv in final_vales])


if __name__ == "__main__":
    unittest.main()
