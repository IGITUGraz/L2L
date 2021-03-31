import os
import unittest
import shutil


from l2l.tests import test_ce_optimizer
from l2l.tests import test_ga_optimizer
from l2l.tests import test_sa_optimizer
from l2l.tests import test_gd_optimizer
from l2l.tests import test_innerloop
from l2l.tests import test_outerloop
from l2l.tests import test_setup


def test_suite():

    suite = unittest.TestSuite()
    suite.addTest(test_setup.suite())
    suite.addTest(test_outerloop.suite())
    suite.addTest(test_innerloop.suite())
    suite.addTest(test_ce_optimizer.suite())
    suite.addTest(test_sa_optimizer.suite())
    suite.addTest(test_gd_optimizer.suite())
    suite.addTest(test_ga_optimizer.suite())

    return suite


if __name__ == "__main__":

    runner = unittest.TextTestRunner(verbosity=2)
    root_dir_path = '../../results'
    runner.run(test_suite())
    if os.path.exists(root_dir_path):
        shutil.rmtree(root_dir_path)
