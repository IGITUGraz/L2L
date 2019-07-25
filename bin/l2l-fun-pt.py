import logging.config
import os

import numpy as np
from l2l.utils.environment import Environment

from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.paralleltempering.optimizer import ParallelTemperingParameters, ParallelTemperingOptimizer, AvailableCoolingSchedules
from l2l.paths import Paths
from l2l.utils import JUBE_runner as jube

from l2l.logging_tools import create_shared_logger_data, configure_loggers

logger = logging.getLogger('bin.l2l-fun-pt')


def main():
    name = 'L2L-FunctionGenerator-PT'
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

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      # freeze_input=True,
                      # multiproc=True,
                      # use_scoop=True,
                      # wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
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

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")
    traj.f_add_parameter_to_group("JUBE_params", "exec", "python3 " +
                                  os.path.join(paths.simulation_path, "run_files/run_optimizee.py"))
    # Paths
    traj.f_add_parameter_to_group("JUBE_params", "paths", paths)

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

    # Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, paths.simulation_path)

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

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    ## Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
