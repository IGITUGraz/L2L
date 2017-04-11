import os
import warnings
import logging.config
import yaml
from pypet import Environment
from pypet import pypetconstants
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions import tools as function_tools
from ltl.optimizers.crossentropy.optimizer import CrossEntropyOptimizer, CrossEntropyParameters
from ltl.paths import Paths
from postproc.recorder import Recorder
from ltl.optimizers.crossentropy.distribution import NoisyGaussian

warnings.filterwarnings("ignore")

logger = logging.getLogger('ltl-fun-ce')


def main():
    name = 'LTL-FUN-CE'
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
                      log_stdout=False,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )

    # Get the trajectory from the environment
    traj = env.trajectory

    function_id = 4
    bench_functs = BenchmarkedFunctions(noise=True)
    fg_name, fg_params = bench_functs.get_function_by_index(4)

    function_tools.plot(fg_params)

    # NOTE: Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, fg_params)

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class
    parameters = CrossEntropyParameters(pop_size=50, rho=0.2, smoothing=0.0, temp_decay=0, n_iteration=5,
                                        distribution=NoisyGaussian(noise_decay=0.95, noise_bias=0.05))
    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                            optimizee_fitness_weights=(-0.1,),
                                            parameters=parameters,
                                            optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Add Recorder
    recorder = Recorder(trajectory=traj, optimizee_id=function_id,
                        optimizee_name=fg_name, optimizee_parameters=fg_params,
                        optimizer_name=optimizer.__class__.__name__, optimizer_parameters=parameters)
    recorder.start()

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Innerloop optimizee end
    optimizee.end()
    # NOTE: Outerloop optimizer end
    optimizer.end(traj)
    recorder.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main()
