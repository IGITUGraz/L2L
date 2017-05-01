import logging.config
import os
import yaml
from pypet import Environment
from pypet import pypetconstants

from ltl.optimizees.functions import tools as function_tools
from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizers.crossentropy.distribution import Gaussian
from ltl.optimizers.face.optimizer import FACEOptimizer, FACEParameters
from ltl.paths import Paths
from ltl.recorder import Recorder

logger = logging.getLogger('ltl-fun-face')


def main():
    name = 'LTL-FUN-FACE'
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

    bench_functs = BenchmarkedFunctions()
    for function_id in range(len(bench_functs.function_name_map)):
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

        # NOTE: Benchmark function
        function_id = 4
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        # NOTE: Innerloop simulator
        optimizee = FunctionGeneratorOptimizee(traj, benchmark_function)

        # NOTE: Outerloop optimizer initialization
        # TODO: Change the optimizer to the appropriate Optimizer class
        parameters = FACEParameters(min_pop_size=20, max_pop_size=50, n_elite=10, smoothing=0.2, temp_decay=0,
                                    n_iteration=1000,
                                    distribution=Gaussian(), n_expand=5)
        optimizer = FACEOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                  optimizee_fitness_weights=(-0.1,),
                                  parameters=parameters,
                                  optimizee_bounding_func=optimizee.bounding_func)

        # Add post processing
        env.add_postprocessing(optimizer.post_process)

        # Add Recorder
        recorder = Recorder(trajectory=traj,
                            optimizee_name=benchmark_name, optimizee_parameters=benchmark_parameters,
                            optimizer_name=optimizer.__class__.__name__,
                            optimizer_parameters=optimizer.get_params())
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
    main()
