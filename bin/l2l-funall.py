import logging.config
import os
import itertools

import numpy as np
from l2l.utils.environment import Environment

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.crossentropy.distribution import NoisyGaussian, Gaussian
from l2l.optimizers.crossentropy.optimizer import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.face.optimizer import FACEOptimizer, FACEParameters
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer, RMSPropParameters, ClassicGDParameters, \
    AdamParameters, StochasticGDParameters
from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
from l2l.utils.experiment import Experiment


def main():
    experiment = Experiment(root_dir_path='../results')
    name = 'L2L-FUNALL'
    jube_params = {}
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True,
                                            jube_parameter=jube_params)
    n_iterations = 100
    seed = 1

    # NOTE: Need to use lambdas here since we want the distributions within CE, FACE etc. optimizers to be reinitialized
    #  afresh each time since it seems like they are stateful.
    optimizers = [
        (CrossEntropyOptimizer,
         lambda: CrossEntropyParameters(pop_size=50, rho=0.2, smoothing=0.0, temp_decay=0,
                                        n_iteration=n_iterations,
                                        distribution=NoisyGaussian(noise_decay=0.95),
                                        stop_criterion=np.inf, seed=seed)),
        (FACEOptimizer,
         lambda: FACEParameters(min_pop_size=20, max_pop_size=50, n_elite=10, smoothing=0.2, temp_decay=0,
                                n_iteration=n_iterations, distribution=Gaussian(), n_expand=5,
                                seed=seed, stop_criterion=np.inf)),
        (GradientDescentOptimizer,
         lambda: RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01, n_random_steps=5, momentum_decay=0.5,
                                   n_iteration=n_iterations, stop_criterion=np.Inf, seed=seed)),
        (GradientDescentOptimizer,
         lambda: ClassicGDParameters(learning_rate=0.01, exploration_step_size=0.01, n_random_steps=5,
                                     n_iteration=n_iterations, stop_criterion=np.Inf, seed=seed)),
        (GradientDescentOptimizer,
         lambda: AdamParameters(learning_rate=0.01, exploration_step_size=0.01, n_random_steps=5, first_order_decay=0.8,
                                second_order_decay=0.8, n_iteration=n_iterations, stop_criterion=np.Inf, seed=seed)),
        (GradientDescentOptimizer,
         lambda: StochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
                                        exploration_step_size=0.01, n_random_steps=5, n_iteration=n_iterations,
                                        stop_criterion=np.Inf, seed=seed))
    ]

    # NOTE: Benchmark functions
    bench_functs = BenchmarkedFunctions()
    function_ids = range(len(bench_functs.function_name_map))

    for function_id, (optimizer_class, optimizer_parameters_fn) in itertools.product(function_ids, optimizers):
        optimizer_parameters = optimizer_parameters_fn()

        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=100)

        optimizee_fitness_weights = -1.
        # Gradient descent does descent!
        if optimizer_class == GradientDescentOptimizer:
            optimizee_fitness_weights = +1.
        # Grid search optimizer input depends on optimizee!
        elif optimizer_class == GridSearchOptimizer:
            optimizer_parameters = GridSearchParameters(param_grid={
                'coords': (optimizee.bound[0], optimizee.bound[1], 30)
            })

        optimizer = optimizer_class(traj, optimizee_create_individual=optimizee.create_individual,
                                    optimizee_fitness_weights=(optimizee_fitness_weights,),
                                    parameters=optimizer_parameters,
                                    optimizee_bounding_func=optimizee.bounding_func)

        experiment.run_experiment(optimizee=optimizee,
                                  optimizee_parameters=None,
                                  optimizer=optimizer,
                                  optimizer_parameters=optimizer_parameters)

        experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
