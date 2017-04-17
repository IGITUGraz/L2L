import os
import warnings

import logging.config

import numpy as np
import yaml

from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.functions.optimizee import FunctionOptimizee
from ltl.optimizers.paralleltempering.optimizer import ParallelTemperingParameters, ParallelTemperingOptimizer, AvailableCoolingSchedules
from ltl.paths import Paths

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-lsm-ga')


def main():
    name = 'LTL-FUN-PT'
    root_dir_path = None  # CHANGE THIS to the directory where your simulation results are contained
    assert root_dir_path is not None, \
           "You have not set the root path to store your results." \
           " Set it manually in the code (by setting the variable 'root_dir_path')" \
           " before running the simulation"
    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    with open("/home/daniel/Schreibtisch/LTL-master/bin/logging.yaml") as f:
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

    # NOTE: Innerloop simulator
    optimizee = FunctionOptimizee(traj, 'rastrigin')

    # configure settings for parallel tempering
    n_parallel_runs = 9
    #available_cooling_schedules = Enum('Schedule', 'DEF LOG EXP LINMULT QUADMULT LINADD QUADADD EXPADD TRIGADD')
    #cooling_schedules= np.zeros((n_parallel_runs,1))
    cooling_schedules = [AvailableCoolingSchedules for _ in range(9)]
    cooling_schedules[0] = AvailableCoolingSchedules.DEF
    cooling_schedules[1] = AvailableCoolingSchedules.LOG
    cooling_schedules[2] = AvailableCoolingSchedules.EXP
    cooling_schedules[3] = AvailableCoolingSchedules.LINMULT
    cooling_schedules[4] = AvailableCoolingSchedules.QUADMULT
    cooling_schedules[5] = AvailableCoolingSchedules.LINADD
    cooling_schedules[6] = AvailableCoolingSchedules.QUADADD
    cooling_schedules[7] = AvailableCoolingSchedules.EXPADD
    cooling_schedules[8] = AvailableCoolingSchedules.TRIGADD
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
