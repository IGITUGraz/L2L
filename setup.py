u"""
This file installs the ltl package.
Note that it does not perform any installation of the documentation. For this, follow the specified procedure in the
 README
"""

from __future__ import with_statement
from __future__ import absolute_import
from setuptools import setup
from io import open
import re


def get_requirements(filename):
    """
    Helper function to read the list of requirements from a file
    """
    dependency_links = []
    with open(filename) as requirements_file:
        requirements = requirements_file.read().strip('\n').splitlines()
    for i, req in enumerate(requirements):
        if ':' in req:
            match_obj = re.match(ur"git\+(?:https|ssh|http):.*#egg=(\w+)-(.*)", req)
            assert match_obj, u"Cannot make sence of url {}".format(req)
            requirements[i] = u"{req}=={ver}".format(req=match_obj.group(1), ver=match_obj.group(2))
            dependency_links.append(req)
    return requirements, dependency_links

requirements, dependency_links = get_requirements(u'requirements.txt')
setup(
    name=u"Learning to Learn",
    version=u"0.1.0",
    packages=[u'ltl'],
    author=u"Anand Subramoney, Arjun Rao",
    author_email=u"anand@igi.tugraz.at, arjun@igi.tugraz.at",
    description=u"This module provides the infrastructure create optimizers and "
                u"optimizees in order to implement learning-to-learn",
    install_requires=requirements,
    provides=[u'ltl'],
    dependency_links=dependency_links,
)
