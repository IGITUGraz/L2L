import os
import warnings

import logging.config
import yaml

from pypet import Environment
from pypet import pypetconstants

from ltl.paths import Paths
from ltl.optimizees.optimizee import Optimizee
from ltl.optimizers.optimizer import Optimizer

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-lsm-ga')

with open("bin/logging.yaml") as f:
    logging.config.dictConfig(yaml.load(f))


def main():
    # TODO: Give some *meaningful* name here
    name = 'LTL'

    # TODO: Change the `root_dir_path` here
    paths = Paths(name, dict(run_no='test'), root_dir_path='/home/anand/output')

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
    # TODO: Change the optimizee to the appropriate Optimizee class
    optimizee = Optimizee()

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class
    optimizer = Optimizer(traj, ...)

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
