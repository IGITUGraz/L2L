import logging.config
import os
import itertools

import numpy as np
from pypet import Environment

from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizers.crossentropy.distribution import NoisyGaussian, Gaussian
from ltl.optimizers.crossentropy.optimizer import CrossEntropyOptimizer, CrossEntropyParameters
from ltl.optimizers.face.optimizer import FACEOptimizer, FACEParameters
from ltl.optimizers.gradientdescent.optimizer import GradientDescentOptimizer, RMSPropParameters, ClassicGDParameters, \
    AdamParameters, StochasticGDParameters
from ltl.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
from ltl.paths import Paths
from ltl.recorder import Recorder

from ltl.logging_tools import create_shared_logger_data, configure_loggers

logger = logging.getLogger('bin.ltl-fun-all')


def main():
    name = 'LTL-FUNALL'
    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    n_iterations = 100

    # NOTE: Need to use lambdas here since we want the distributions within CE, FACE etc. optimizers to be reinitialized
    #  afresh each time since it seems like they are stateful.
    optimizers = [
        (CrossEntropyOptimizer,
         lambda: CrossEntropyParameters(pop_size=50, rho=0.2, smoothing=0.0, temp_decay=0,
                                        n_iteration=n_iterations,
                                        distribution=NoisyGaussian(noise_decay=0.95, noise_bias=0.05))),
        (FACEOptimizer,
         lambda: FACEParameters(min_pop_size=20, max_pop_size=50, n_elite=10, smoothing=0.2, temp_decay=0,
                                n_iteration=n_iterations, distribution=Gaussian(), n_expand=5)),
        (GradientDescentOptimizer,
         lambda: RMSPropParameters(learning_rate=0.01, exploration_rate=0.01, n_random_steps=5, momentum_decay=0.5,
                                   n_iteration=n_iterations, stop_criterion=np.Inf)),
        (GradientDescentOptimizer,
         lambda: ClassicGDParameters(learning_rate=0.01, exploration_rate=0.01, n_random_steps=5,
                                     n_iteration=n_iterations, stop_criterion=np.Inf)),
        (GradientDescentOptimizer,
         lambda: AdamParameters(learning_rate=0.01, exploration_rate=0.01, n_random_steps=5, first_order_decay=0.8,
                                second_order_decay=0.8, n_iteration=n_iterations, stop_criterion=np.Inf)),
        (GradientDescentOptimizer,
         lambda: StochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
                                        exploration_rate=0.01, n_random_steps=5, n_iteration=n_iterations,
                                        stop_criterion=np.Inf))
    ]

    # NOTE: Benchmark functions
    bench_functs = BenchmarkedFunctions()
    function_ids = range(len(bench_functs.function_name_map))

    for function_id, (optimizer_class, optimizer_parameters_fn) in itertools.product(function_ids, optimizers):
        logger.info("Running benchmark for %s optimizer and function id %d", optimizer_class, function_id)
        optimizer_parameters = optimizer_parameters_fn()

        # Create an environment that handles running our simulation
        # This initializes a PyPet environment
        env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                          comment='{} data'.format(name),
                          # freeze_input=True,
                          # multiproc=True,
                          # use_scoop=True,
                          # wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                          add_time=True,
                          automatic_storing=True,
                          log_stdout=False,  # Sends stdout to logs
                          )
        create_shared_logger_data(logger_names=['bin', 'optimizers'],
                                  log_levels=['INFO', 'INFO'],
                                  log_to_consoles=[True, True],
                                  sim_name=name,
                                  log_directory=paths.logs_path)
        configure_loggers()

        # Get the trajectory from the environment
        traj = env.trajectory

        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee = FunctionGeneratorOptimizee(traj, benchmark_function)

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

        # Add post processing
        env.add_postprocessing(optimizer.post_process)

        # Add Recorder
        recorder = Recorder(trajectory=traj,
                            optimizee_name=benchmark_name, optimizee_parameters=benchmark_parameters,
                            optimizer_name=optimizer.__class__.__name__,
                            optimizer_parameters=optimizer.get_params())
        recorder.start()

        # Run the simulation with all parameter combinations
        env.run(optimizee.simulate)

        # NOTE: Outerloop optimizer end
        optimizer.end(traj)
        recorder.end()

        # Finally disable logging and close all log-files
        env.disable_logging()


if __name__ == '__main__':
    main()
