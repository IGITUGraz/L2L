import numpy as np

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.crossentropy.distribution import Gaussian
from l2l.optimizers.face.optimizer import FACEOptimizer, FACEParameters
from l2l.utils.experiment import Experiment


def main():
    name = 'L2L-FUN-FACE'
    experiment = Experiment("../results/")
    trajectory_name = name
    traj, all_jube_params, = experiment.prepare_experiment(name=name,
                                                           trajectory_name=trajectory_name,
                                                           log_stdout=True)

    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    parameters = FACEParameters(min_pop_size=20, max_pop_size=50, n_elite=10, smoothing=0.2, temp_decay=0,
                                n_iteration=1,
                                distribution=Gaussian(), n_expand=5, stop_criterion=np.inf, seed=109)
    optimizer = FACEOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                              optimizee_fitness_weights=(-0.1,),
                              parameters=parameters,
                              optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters,
                              optimizee_parameters=None)
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
