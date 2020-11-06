import numpy as np

from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizers.crossentropy.optimizer import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.crossentropy.distribution import NoisyBayesianGaussianMixture
from l2l.utils.experiment import Experiment


def main():
    name = 'L2L-FUN-CE'
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True)


    function_id = 14
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    parameters = CrossEntropyParameters(pop_size=50, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=160,
                                        distribution=NoisyBayesianGaussianMixture(n_components=3,
                                                                                  noise_magnitude=1.,
                                                                                  noise_decay=0.9,
                                                                                  weight_concentration_prior=1.5),
                                        stop_criterion=np.inf, seed=103)
    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                            optimizee_fitness_weights=(-0.1,),
                                            parameters=parameters,
                                            optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters)
    # End experiment
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
