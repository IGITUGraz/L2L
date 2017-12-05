from __future__ import with_statement
from __future__ import absolute_import
import logging.config
import os
import warnings

import yaml

from ltl.optimizees.functions.function_generator import FunctionGenerator, GaussianParameters, PermutationParameters, \
    EasomParameters, LangermannParameters, MichalewiczParameters, ShekelParameters, RastriginParameters, \
    RosenbrockParameters, ChasmParameters, AckleyParameters
from ltl.paths import Paths
from io import open

warnings.filterwarnings(u"ignore")

logger = logging.getLogger(u'bin.plot-function-generator')


def main():
    name = u'plot-function-generator'
    try:
        with open(u'bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            u"You have not set the root path to store your results."
            u" Write the path to a path.conf text file in the bin directory"
            u" before running the simulation"
        )
    paths = Paths(name, dict(run_no=u'test'), root_dir_path=root_dir_path)

    with open(u"bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict[u'handlers'][u'file'][u'filename'])
        l_dict[u'handlers'][u'file'][u'filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    print u"All output can be found in file ", log_output_file
    print u"Change the values in logging.yaml to control log level and destination"
    print u"e.g. change the handler to console for the loggers you're interesting in to get output to stdout"

    fg_params = [GaussianParameters(sigma=[[1.5, .1], [.1, .3]], mean=[-1., -1.]),
                 GaussianParameters(sigma=[[.25, .3], [.3, 1.]], mean=[1., 1.]),
                 GaussianParameters(sigma=[[.5, .25], [.25, 1.3]], mean=[2., -2.])]
    FunctionGenerator(fg_params, dims=2, noise=True).plot()

    FunctionGenerator([PermutationParameters(beta=0.005)], dims=2).plot()

    FunctionGenerator([EasomParameters()], dims=3).plot()

    FunctionGenerator([LangermannParameters(A=u'default', c=u'default')], dims=2).plot()

    FunctionGenerator([MichalewiczParameters(m=u'default')], dims=2).plot()

    FunctionGenerator([ShekelParameters(A=u'default', c=u'default')], dims=2).plot()

    fg_params = [ShekelParameters(A=[[8, 5]], c=[0.08]),
                 LangermannParameters(A=u'default', c=u'default')]
    FunctionGenerator(fg_params, dims=2).plot()

    FunctionGenerator([RastriginParameters()], dims=2).plot()

    FunctionGenerator([RosenbrockParameters()], dims=2).plot()

    FunctionGenerator([ChasmParameters()], dims=2).plot()

    FunctionGenerator([AckleyParameters()], dims=2).plot()


if __name__ == u'__main__':
    main()
