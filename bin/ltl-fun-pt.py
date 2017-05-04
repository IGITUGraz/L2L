import logging.config
import os

import numpy as np
import yaml
from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.functions import tools as function_tools
from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizers.paralleltempering.optimizer import ParallelTemperingParameters, ParallelTemperingOptimizer, AvailableCoolingSchedules
from ltl.paths import Paths
from ltl.recorder import Recorder

logger = logging.getLogger('ltl-lsm-pt')


def main():
    name = 'LTL-FunctionGenerator-PT'
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

    with open("bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    print("All output can be found in file ", log_output_file)
    print("Change the values in logging.yaml to control log level and destination")
    print("e.g. change the handler to console for the loggers you're interesting in to get output to stdout")

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      freeze_input=True,
                      multiproc=True,
                      use_scoop=True,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      automatic_storing=True,
                      log_stdout=True,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )
    
    # Get the trajectory from the environment
    traj = env.trajectory

    # NOTE: Benchmark function
    function_id = 1
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    function_tools.plot(benchmark_function)

    # NOTE: Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function)

    #--------------------------------------------------------------------------
    # configure settings for parallel tempering:
    # for each of the parallel runs chose
    # a cooling schedule
    # an upper and lower temperature bound
    # a decay parameter
    #--------------------------------------------------------------------------
    n_parallel_runs = 9

    cooling_schedules = [AvailableCoolingSchedules for _ in range(n_parallel_runs)]
    cooling_schedules[0] = AvailableCoolingSchedules.DEFAULT
    cooling_schedules[1] = AvailableCoolingSchedules.LOGARITHMIC
    cooling_schedules[2] = AvailableCoolingSchedules.EXPONENTIAL
    cooling_schedules[3] = AvailableCoolingSchedules.LINEAR_MULTIPLICATIVE
    cooling_schedules[4] = AvailableCoolingSchedules.QUADRATIC_MULTIPLICATIVE
    cooling_schedules[5] = AvailableCoolingSchedules.LINEAR_ADDAPTIVE
    cooling_schedules[6] = AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE
    cooling_schedules[7] = AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE
    cooling_schedules[8] = AvailableCoolingSchedules.TRIGONOMETRIC_ADDAPTIVE

    #has to be from 1 to 0, first entry hast to be larger than second
    temperature_bounds = np.zeros((n_parallel_runs,2))
    temperature_bounds[0] = [1,0]
    temperature_bounds[1] = [0.9,0]
    temperature_bounds[2] = [0.8,0]
    temperature_bounds[3] = [0.7,0]
    temperature_bounds[4] = [0.6,0]
    temperature_bounds[5] = [1,0.1]
    temperature_bounds[6] = [1,0.2]
    temperature_bounds[7] = [1,0.3]
    temperature_bounds[8] = [1,0.4]

    # decay parameter for each schedule seperately
    decay_parameters = np.zeros((n_parallel_runs))
    for i in range(0,n_parallel_runs):
        decay_parameters[i] = 0.98 #all the same for now
    #--------------------------------------------------------------------------
    # end of configuration
    #--------------------------------------------------------------------------
    
    # Check, if the temperature bounds and decay parameters are reasonable. 
    assert (((temperature_bounds.all() <= 1) and (temperature_bounds.all() >= 0)) and (temperature_bounds[:,0].all() > temperature_bounds[:,1].all())), print("Warning: Temperature bounds are not within specifications.")
    assert ((decay_parameters.all() <= 1) and (decay_parameters.all() >= 0)), print("Warning: Decay parameter not within specifications.")
    
    # NOTE: Outerloop optimizer initialization
    parameters = ParallelTemperingParameters(n_parallel_runs, noisy_step=.3, n_iteration=100, stop_criterion=np.Inf,
                                              seed=np.random.randint(1e5), cooling_schedules=cooling_schedules, 
                                              temperature_bounds=temperature_bounds, decay_parameters=decay_parameters)
    optimizer = ParallelTemperingOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                                  optimizee_fitness_weights=(-0.1,),
                                                  parameters=parameters,
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

    # NOTE: Innerloop optimizee end
    optimizee.end()
    # NOTE: Outerloop optimizer end
    optimizer.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
