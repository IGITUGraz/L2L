import unittest

from l2l.utils.environment import Environment

import l2l.utils.JUBE_runner as jube
from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.paths import Paths
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.utils.experiment import Experiment

import os


class SetupTestCase(unittest.TestCase):

    def setUp(self):
        self.experiment = Experiment(root_dir_path='../../results')
        jube_params = {}
        try:
            self.trajectory, _ = self.experiment.prepare_experiment(
                name='test_trajectory',
                log_stdout=True,
                add_time=True,
                automatic_storing=True,
                jube_parameter=jube_params)
        except FileNotFoundError as fe:
            self.fail(
                "{} \n L2L is not well configured. Missing path file.".format(
                    fe))
        self.paths = self.experiment.paths

    def test_paths(self):
        self.assertIsNotNone(self.paths)
        self.assertIsNotNone(Paths.simulation_path)

    def test_environment_trajectory_setup(self):
        self.assertIsNotNone(self.trajectory.individual)

    def test_trajectory_parms_setup(self):
        self.trajectory.f_add_parameter_group("Test_params", "Contains Test parameters")
        self.trajectory.f_add_parameter_to_group("Test_params", "param1", "value1")
        self.assertEqual("value1", self.trajectory.Test_params.params["param1"])

    def test_juberunner_setup(self):
        self.experiment = Experiment(root_dir_path='../../results')
        self.trajectory, _ = self.experiment.prepare_experiment(
            name='test_trajectory',
            trajectory='test_trajectory',
            filename=".",
            file_title='{} data'.format('test_trajectory'),
            comment='{} data'.format('test_trajectory'),
            add_time=True,
            automatic_storing=True,
            log_stdout=False,
            jube_parameter={}
        )
        self.trajectory.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")
        self.trajectory.f_add_parameter_to_group("JUBE_params", "exec", "python " +
                                      os.path.join(self.paths.simulation_path,
                                                   "run_files/run_optimizee.py"))
        self.trajectory.f_add_parameter_to_group("JUBE_params", "paths", self.paths)

        ## Benchmark function
        function_id = 14
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee_seed = 1
        optimizee = FunctionGeneratorOptimizee(self.trajectory, benchmark_function,
                                               seed=optimizee_seed)

        jube.prepare_optimizee(optimizee, self.paths.root_dir_path)

        fname = os.path.join(self.paths.root_dir_path, "optimizee.bin")

        try:
            f = open(fname, "r")
            f.close()
        except Exception:
            self.fail()


def suite():
    suite = unittest.makeSuite(SetupTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
