from l2l.utils.experiment import Experiment
import numpy as np

from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters


def main():
    name = 'L2L-FUN-GS'
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True)

    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    n_grid_divs_per_axis = 30
    parameters = GridSearchParameters(param_grid={
        'coords': (optimizee.bound[0], optimizee.bound[1], n_grid_divs_per_axis)
    })
    optimizer = GridSearchOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                    optimizee_fitness_weights=(-0.1,),
                                    parameters=parameters)
    # Experiment run
    experiment.run_experiment(optimizee=optimizee, optimizer=optimizer,
                              optimizee_parameters=parameters)
    # End experiment
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
