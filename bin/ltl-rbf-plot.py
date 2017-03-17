import os
import warnings

import logging.config

import yaml

from ltl.optimizees.rbfs.rbf import RBF
from ltl.paths import Paths

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-rbf-plot')


def main():
    name = 'LTL-RBF-PLOT'
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

    rbf_params = [{'name': 'gaussian', 'params': [[[1.5, .1], [.1, .3]], [-1., -1.]]},
                  {'name': 'gaussian', 'params': [[[.25, .3], [.3, 1.]], [1., 1.]]},
                  {'name': 'gaussian', 'params': [[[.5, .25], [.25, 1.3]], [2., -2.]]}]
    rbf_instance = RBF(rbf_params, 2)
    rbf_instance.plot()

if __name__ == '__main__':
    main()
