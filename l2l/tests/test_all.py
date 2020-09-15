import unittest


from . import test_ce_optimizer
from . import test_ga_optimizer
from . import test_sa_optimizer
from . import test_gd_optimizer
from . import test_innerloop
from . import test_outerloop
from . import test_setup


def suite():

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
    runner.run(suite())