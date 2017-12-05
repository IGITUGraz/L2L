u"""
This file installs the ltl package.
Note that it does not perform any installation of the documentation. For this, follow the specified procedure in the
 README
"""

from __future__ import with_statement
from __future__ import absolute_import
from setuptools import setup
from io import open


def get_requirements(filename):
    u"""
    Helper function to read the list of requirements from a file
    """
    with open(filename) as requirements_file:
        reqs = requirements_file.read().strip(u'\n').splitlines()
    return reqs


setup(
    name=u"Learning to Learn",
    version=u"0.1.0",
    packages=[u'ltl'],
    author=u"Anand Subramoney, Arjun Rao",
    author_email=u"anand@igi.tugraz.at, arjun@igi.tugraz.at",
    description=u"This module provides the infrastructure create optimizers and "
                u"optimizees in order to implement learning-to-learn",
    install_requires=get_requirements(u'requirements.txt'),
    provides=[u'ltl'],
)
