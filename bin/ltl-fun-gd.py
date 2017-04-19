import logging.config
import os
import warnings

import numpy as np
import yaml
from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.functions.function_generator import GaussianParameters, FunctionGenerator
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
# from ltl.optimizers.gradientdescent.optimizer import ClassicGDParameters
# from ltl.optimizers.gradientdescent.optimizer import StochasticGDParameters
# from ltl.optimizers.gradientdescent.optimizer import AdamParameters
from ltl.optimizers.gradientdescent.optimizer import RMSPropParameters
from ltl.paths import Paths

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-lsm-gradientdescent')


def main():
    name = 'LTL-FUN-GD'
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

    # NOTE: Innerloop simulator
    fg_instance = FunctionGenerator([GaussianParameters(sigma=[[1., 0.], [0., 1.]], mean=[1., 1.])],
                                    dims=2, bound=[0, 2])
    optimizee = FunctionGeneratorOptimizee(traj, fg_instance)

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class

    # parameters = ClassicGDParameters(learning_rate=0.01, exploration_rate=0.01, n_random_steps=5, n_iteration=100,
    #                                 stop_criterion=np.Inf)
    # parameters = AdamParameters(learning_rate=0.01, exploration_rate=0.01, n_random_steps=5, first_order_decay=0.8,
    #                            second_order_decay=0.8, n_iteration=100, stop_criterion=np.Inf)
    # parameters = StochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
    #                                    exploration_rate=0.01, n_random_steps=5, n_iteration=100,
    #                                    stop_criterion=np.Inf)
    parameters = RMSPropParameters(learning_rate=0.01, exploration_rate=0.01, n_random_steps=5, momentum_decay=0.5,
                                   n_iteration=100, stop_criterion=np.Inf)

    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(0.1,),
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
