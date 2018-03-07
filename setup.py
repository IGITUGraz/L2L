from setuptools import setup
from setuptools import find_packages
import re

from ltl.version import FULL_VERSION

"""
This file installs the ltl package.
Note that it does not perform any installation of the documentation. For this, follow the specified procedure in the
 README. For updating the version, update MAJOR_VERSION and FULL_VERSION in ltl/version.py
"""


def get_requirements(filename):
    """
    Helper function to read the list of requirements from a file
    """
    dependency_links = []
    with open(filename) as requirements_file:
        requirements = requirements_file.read().strip('\n').splitlines()
    for i, req in enumerate(requirements):
        if ':' in req:
            match_obj = re.match(r"git\+(?:https|ssh|http):.*#egg=(\w+)-(.*)", req)
            assert match_obj, "Cannot make sense of url {}".format(req)
            requirements[i] = "{req}=={ver}".format(req=match_obj.group(1), ver=match_obj.group(2))
            dependency_links.append(req)
    return requirements, dependency_links


requirements, dependency_links = get_requirements('requirements.txt')
setup(
    name="Learning to Learn",
    version=FULL_VERSION,
    packages=find_packages("."),
    author="Anand Subramoney, Arjun Rao",
    author_email="anand@igi.tugraz.at, arjun@igi.tugraz.at",
    description="This module provides the infrastructure create optimizers and "
                "optimizees in order to implement learning-to-learn",
    install_requires=requirements,
    provides=['ltl'],
    dependency_links=dependency_links,
)
