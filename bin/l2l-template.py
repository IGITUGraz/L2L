"""
This file is a typical example of a script used to run a L2L experiment. Read the comments in the file for more
explanations
"""

import logging.config
import os

from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.optimizees.optimizee import Optimizee, OptimizeeParameters
from l2l.optimizers.optimizer import Optimizer, OptimizerParameters
from l2l.paths import Paths

from l2l.utils import JUBE_runner as jube

# We first setup the logger and read the logging config which controls the verbosity and destination of the logs from
# various parts of the code.
logger = logging.getLogger('bin.l2l-optimizee-optimizer')


def main():
    # TODO: use  the experiment module to prepare and run later the simulation
    # define a directory to store the results
    experiment = Experiment(root_dir_path='~/home/user/L2L/results')
    # TODO when using the template: use keywords to prepare the experiment and
    #  create a dictionary for jube parameters
    # prepare_experiment returns the trajectory and all jube parameters
    jube_params = {"nodes": "2",
                   "walltime": "10:00:00",
                   "ppn": "1",
                   "cpu_pp": "1"}
    traj, all_jube_params = experiment.prepare_experiment(name='L2L',
                                                          log_stdout=True,
                                                          **jube_params)

    ## Innerloop simulator
    # TODO when using the template: Change the optimizee to the appropriate
    #  Optimizee class
    optimizee = Optimizee(traj)
    # TODO Create optimizee parameters
    optimizee_parameters = OptimizeeParameters()

    ## Outerloop optimizer initialization
    # TODO when using the template: Change the optimizer to the appropriate
    #  Optimizer class and use the right value for optimizee_fitness_weights.
    #  Length is the number of dimensions of fitness, and negative value
    #  implies minimization and vice versa
    optimizer_parameters = OptimizerParameters()
    optimizer = Optimizer(traj, optimizee.create_individual, (1.0,),
                          optimizer_parameters)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)
    experiment.end(optimizer)


if __name__ == '__main__':
    main()
