"""
This file is a typical example of a script used to run a LTL experiment. Read the comments in the file for more
explanations
"""

import logging.config
import os

import yaml
from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.optimizee import Optimizee
from ltl.optimizers.optimizer import Optimizer, OptimizerParameters
from ltl.paths import Paths
# We first setup the logger and read the logging config which controls the verbosity and destination of the logs from
# various parts of the code.
from ltl.recorder import Recorder

logger = logging.getLogger('ltl-optimizee-optimizer')


def main():
    # TODO when using the template: Give some *meaningful* name here
    name = 'LTL'

    # TODO when using the template: make a path.conf file and write the root path there
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

    # Load the logging config which tells us where and what to log (loglevel, destination)
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
    # This initializes a PyPet environment. See Pypet documentation for more details on environment and trajectory.
    # Uncomment 'freeze_input', 'multipproc', 'use_scoop' and 'wrap_mode' lines to disable running the experiment
    # across cores and nodes.
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

    # Get the trajectory from the environment.
    traj = env.trajectory

    # NOTE: Innerloop simulator
    # TODO when using the template: Change the optimizee to the appropriate Optimizee class
    optimizee = Optimizee()

    # NOTE: Outerloop optimizer initialization
    # TODO when using the template: Change the optimizer to the appropriate Optimizer class
    # and use the right value for optimizee_fitness_weights. Length is the number of dimensions of fitness, and
    # negative value implies minimization and vice versa
    optimizer_parameters = OptimizerParameters()
    optimizer = Optimizer(traj, optimizee.create_individual, (1.0,), optimizer_parameters)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Add Recorder
    # TODO: Change the names, ids and parameters passed in
    recorder = Recorder(trajectory=traj, optimizee_id='optimizee_id',
                        optimizee_name='optimizee', optimizee_parameters=dict(),
                        optimizer_name=optimizer.__class__.__name__, optimizer_parameters=dict())
    recorder.start()

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Innerloop optimizee end
    optimizee.end()
    # NOTE: Outerloop optimizer end
    optimizer.end()

    recorder.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
