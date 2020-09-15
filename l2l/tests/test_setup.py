import unittest

from l2l.utils.environment import Environment

import l2l.utils.JUBE_runner as jube
from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.paths import Paths
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee

import os

class SetupTestCase(unittest.TestCase):

    def setUp(self):
        self.name = "test_trajectory"
        try:
            with open('../../bin/path.conf') as f:
                root_dir_path = f.read().strip()
        except FileNotFoundError:
            self.fail("L2L is not well configured. Missing path file.")
        self.paths = Paths(self.name, dict(run_num='test'), root_dir_path=root_dir_path, suffix="-" + self.name)
        self.assertIsNotNone(self.paths)
        self.assertIsNotNone(Paths.simulation_path)

    def test_environment_trajectory_setup(self):

        env = Environment(
            trajectory=self.name,
            filename=".",
            file_title='{} data'.format(self.name),
            comment='{} data'.format(self.name),
            add_time=True,
            automatic_storing=True,
            log_stdout=False,
        )
        traj = env.trajectory
        self.assertIsNotNone(traj.individual)


    def test_trajectory_parms_setup(self):
        env = Environment(
            trajectory=self.name,
            filename=".",
            file_title='{} data'.format(self.name),
            comment='{} data'.format(self.name),
            add_time=True,
            automatic_storing=True,
            log_stdout=False,
        )
        traj = env.trajectory
        traj.f_add_parameter_group("Test_params", "Contains Test parameters")
        traj.f_add_parameter_to_group("Test_params", "param1", "value1")
        self.assertEqual("value1", traj.Test_params.params["param1"])

    def test_logger_setup(self):
        create_shared_logger_data(
            logger_names=['bin', 'optimizers'],
            log_levels=['INFO', 'INFO'],
            log_to_consoles=[True, True],
            sim_name="test_sim",
            log_directory=".")
        configure_loggers()

    def test_data_structures_setup(self):
        pass

    def test_juberunner_setup(self):
        name = "test_trajectory"
        self.env = Environment(
            trajectory=name,
            filename=".",
            file_title='{} data'.format(name),
            comment='{} data'.format(name),
            add_time=True,
            automatic_storing=True,
            log_stdout=False,  # Sends stdout to logs
        )
        self.traj = self.env.trajectory
        ## Benchmark function
        function_id = 14
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee_seed = 1
        self.optimizee = FunctionGeneratorOptimizee(self.traj, benchmark_function, seed=optimizee_seed)

        jube.prepare_optimizee(self.optimizee, self.paths.root_dir_path)

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