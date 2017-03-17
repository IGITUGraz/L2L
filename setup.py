# This file installs the ltl package. Note that it does not perform any installation of the documentation. For this, follow the specified procedure in the README
from setuptools import setup
setup(
    name="learntolearn",
    version="0.1.0",
    packages=['ltl'],
    author="Arjun Rao",
    author_email="arjun@igi.tugraz.at",
    description="This module provides the infrastructure create optimizers and "
                "optimizees in order to implement learning-to-learn",
    license="MIT",
    keywords="Generic Builder builder generic",
    install_requires=['numpy',
                      'scipy',
                      'pyyaml',
                      'matplotlib',
                      'pypet==0.4.0',
                      'scoop==0.7.1.1',
                      'sphinx',
                      'deap'],
    provides=['ltl'],
)
