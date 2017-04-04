import os
import warnings

import logging.config

import numpy as np
import yaml

from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.functions.optimizee import FunctionOptimizee
from ltl.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer
from ltl.paths import Paths
from ltl.optimizees.optimizee import Optimizee
from ltl.optimizers.optimizer import Optimizer
from postproc.recorder import Recorder

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-lsm-ga')


def main():
    name = 'LTL-RECFUN-SA'
    root_dir_path = "/home/sinisa/Uni/Project_CI/results"  # CHANGE THIS to the directory where your simulation results are contained
    assert root_dir_path is not None, \
           "You have not set the root path to store your results." \
           " Set it manually in the code (by setting the variable 'root_dir_path')" \
           " before running the simulation"
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
    print(traj_file)

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      # freeze_input=True,
                      # multiproc=True,
                      # use_scoop=True,
                      # wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      automatic_storing=True,
                      log_stdout=True,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )

    # Get the trajectory from the environment
    traj = env.trajectory

    # NOTE: Innerloop simulator
    optimizee_name = "rastrigin"
    optimizee = FunctionOptimizee(traj, optimizee_name)

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class
    optimizer_name = "SimulatedAnnealing"
    parameters = SimulatedAnnealingParameters(noisy_step=.3, temp_decay=.998, n_iteration=10, stop_criterion=np.Inf,
                                              seed=np.random.randint(1e5))
    optimizer = SimulatedAnnealingOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                                  optimizee_fitness_weights=(-0.1,),
                                                  parameters=parameters,
                                                  optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    optimizee_params = None
    recorder = Recorder(description='description', environment=env,
                        optimizee_name=optimizee_name, optimizee_parameters=optimizee_params,
                        optimizer_name=optimizer_name, optimizer_parameters=parameters)
    # recorder.parse_md()
    # exit()
    recorder.start()
    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)
    # NOTE: Innerloop optimizee end
    optimizee.end()
    # NOTE: Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()
    recorder.end()


if __name__ == '__main__':
    main()
