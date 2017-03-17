# This file installs the ltl package. Note that it does not perform any installation of the documentation. For this, follow the specified procedure in the README
from setuptools import setup

def get_requirements(filename):
    """
    Helper function to read the list of requirements from a file
    """
    with open(filename) as requirements_file:
        reqs = requirements_file.read().strip('\n').split('\n')
    return reqs

setup(
    name="learntolearn",
    version="0.1.0",
    packages=['ltl'],
    author="Arjun Rao",
    author_email="arjun@igi.tugraz.at",
    description="This module provides the infrastructure create optimizers and "
                "optimizees in order to implement learning-to-learn",
    install_requires=get_requirements('requirements.txt'),
    provides=['ltl'],
)
