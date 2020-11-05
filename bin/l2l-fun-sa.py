import numpy as np
from l2l.utils.experiment import Experiment
from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules


def main():
    name = 'L2L-FunctionGenerator-SA'
    experiment = Experiment("../results/")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          log_stdout=True)

    ## Benchmark function
    function_id = 14
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    parameters = SimulatedAnnealingParameters(n_parallel_runs=50, noisy_step=.03, temp_decay=.99, n_iteration=100,
                                              stop_criterion=np.Inf, seed=np.random.randint(1e5), cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)

    optimizer = SimulatedAnnealingOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                            optimizee_fitness_weights=(-1,),
                                            parameters=parameters,
                                            optimizee_bounding_func=optimizee.bounding_func)
    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters)
    # End experiment
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
