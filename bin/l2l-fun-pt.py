import numpy as np
from l2l.utils.experiment import Experiment

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.paralleltempering.optimizer import ParallelTemperingParameters, ParallelTemperingOptimizer, AvailableCoolingSchedules


def main():
    name = 'L2L-FunctionGenerator-PT'
    experiment = Experiment("../results/")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          log_stdout=True)

    ## Benchmark function
    function_id = 14
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    #--------------------------------------------------------------------------
    # configure settings for parallel tempering:
    # for each of the parallel runs chose
    # a cooling schedule
    # an upper and lower temperature bound
    # a decay parameter
    #--------------------------------------------------------------------------
    
    # specify the number of parallel running schedules. Each following container
    # has to have an entry for each parallel run 
    n_parallel_runs = 5

    # for detailed information on the cooling schedules see either the wiki or
    # the documentaition in l2l.optimizers.paralleltempering.optimizer
    cooling_schedules = [AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE,
                         AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE,
                         AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE,
                         AvailableCoolingSchedules.LINEAR_ADDAPTIVE,
                         AvailableCoolingSchedules.LINEAR_ADDAPTIVE]

    # has to be from 1 to 0, first entry hast to be larger than second
    # represents the starting temperature and the ending temperature
    temperature_bounds = np.array([
        [0.8, 0],
        [0.7, 0],
        [0.6, 0],
        [1, 0.1],
        [0.9, 0.2]])

    # decay parameter for each schedule. If needed can be different for each
    # schedule
    decay_parameters = np.full(n_parallel_runs, 0.99)
    #--------------------------------------------------------------------------
    # end of configuration
    #--------------------------------------------------------------------------

    # Check, if the temperature bounds and decay parameters are reasonable.
    assert (((temperature_bounds.all() <= 1) and (temperature_bounds.all() >= 0)) and (temperature_bounds[:, 0].all(
    ) > temperature_bounds[:, 1].all())), print("Warning: Temperature bounds are not within specifications.")
    assert ((decay_parameters.all() <= 1) and (decay_parameters.all() >= 0)), print(
        "Warning: Decay parameter not within specifications.")

    ## Outerloop optimizer initialization
    parameters = ParallelTemperingParameters(n_parallel_runs=n_parallel_runs, noisy_step=.03, n_iteration=1000, stop_criterion=np.Inf,
                                             seed=np.random.randint(1e5), cooling_schedules=cooling_schedules,
                                             temperature_bounds=temperature_bounds, decay_parameters=decay_parameters)
    optimizer = ParallelTemperingOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
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
