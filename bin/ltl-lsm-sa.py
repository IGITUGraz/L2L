import os
import warnings

import logging.config

import numpy as np
import yaml

from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.lsm.optimizee import LSMOptimizee
from ltl.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer
from ltl.paths import Paths

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-lsm-sa')


def main():
    name = 'LSM-SA'
    paths = Paths(name, dict(run_no='test'), root_dir_path='/home/anand/output')
    print("All output can be found in file ", paths.output_dir_path)
    print("Change the values in logging.yaml to control log level and destination")
    print("e.g. change the handler to console for the loggers you're interesting in to get output to stdout")

    with open("bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        l_dict['handlers']['file']['filename'] = os.path.join(paths.output_dir_path,
                                                              l_dict['handlers']['file']['filename'])
        logging.config.dictConfig(l_dict)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation

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
    lsm = LSMOptimizee()

    # NOTE: Outerloop optimizer initialization
    parameters = SimulatedAnnealingParameters(noisy_step=.3, temp_decay=.99, n_iteration=1000, stop_criterion=-np.Inf,
                                              bound=[-np.Inf, np.Inf], seed=42)
    sa = SimulatedAnnealingOptimizer(traj, optimizee_create_individual=lsm.create_individual,
                                     optimizee_fitness_weights=(-1.0,), parameters=parameters)

    # Add post processing
    env.add_postprocessing(sa.post_process)

    # Run the simulation with all parameter combinations
    env.run(lsm.simulate)

    # NOTE: Innerloop optimizee end
    lsm.end()
    # NOTE: Outerloop optimizer end
    sa.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
