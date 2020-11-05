from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.utils.experiment import Experiment

import numpy as np


def main():
    experiment = Experiment(root_dir_path='../results')
    name = 'L2L-FUN-CE'

    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True)

    ## Benchmark function
    function_id = 14
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=101)

    ## Outerloop optimizer initialization
    parameters = CrossEntropyParameters(pop_size=10, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=1000,
                                        distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                        stop_criterion=np.inf, seed=102)
    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(-0.1,),
                                      parameters=parameters,
                                      optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizee=optimizee, optimizer=optimizer)
    # End experiment
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
