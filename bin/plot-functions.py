import os
import warnings

import logging.config

import yaml

from ltl.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, PermutationParameters, \
    EasomParameters, LangermannParameters, MichalewiczParameters, ShekelParameters, RastriginParameters, \
    RosenbrockParameters, ChasmParameters
from ltl.paths import Paths

warnings.filterwarnings("ignore")

logger = logging.getLogger('plot-function-generator')


def main():
    name = 'plot-function-generator'
    root_dir_path = None  # CHANGE THIS to the directory where your simulation results are contained
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

    fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                 GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                 GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
    FunctionGenerator(fg_params, dims=2, noise=True).plot()

    FunctionGenerator([PermutationParameters(beta=0.005)], dims=2).plot()

    FunctionGenerator([EasomParameters()], dims=3).plot()

    FunctionGenerator([LangermannParameters(A='default', c='default')], dims=2).plot()

    FunctionGenerator([MichalewiczParameters(m='default')], dims=2).plot()

    FunctionGenerator([ShekelParameters(A='default', c='default')], dims=2).plot()

    fg_params = [ShekelParameters(A=[[8, 5]], c=[0.08]),
                 LangermannParameters(A='default', c='default')]
    FunctionGenerator(fg_params, dims=2).plot()

    FunctionGenerator([RastriginParameters()], dims=2).plot()

    FunctionGenerator([RosenbrockParameters()], dims=2).plot()

    FunctionGenerator([ChasmParameters()], dims=2).plot()


if __name__ == '__main__':
    main()
