import os

import yaml
import numpy as np

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters

from l2l.utils.experiment import Experiment


def main():
    experiment = Experiment(root_dir_path='../results')
    name = 'L2L-FUN-GA'
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True)

    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5,
                                            MUTPB=0.3, NGEN=100, indpb=0.02,
                                            tournsize=15, matepar=0.5,
                                            mutpar=1
                                            )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters)
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=parameters)
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
