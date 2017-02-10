.. LTL documentation master file, created by
   sphinx-quickstart on Fri Feb 10 18:25:53 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LTL's documentation!
===============================

Contents:

.. toctree::
   :maxdepth: 2

Introduction
============
Blah Blah Blah

Other packages used
-------------------
* PyPet: This is a parameter exploration toolkit that managers exploration of parameter space *and* storing the results in a standard format (HDF5). Documentation can be found `here <https://pypet.readthedocs.io/en/latest/>`_.
* SCOOP: This is optionally used for distributing individual Optimizee simulations across multiple hosts in a cluster. Documentation can be found `here <https://scoop.readthedocs.io/en/0.7/>`_.

Running a LTL simulation
========================
To run a LTL simulation, copy the file :file:`bin/learn-to-learn-template.py` to
:file:`bin/learn-to-learn-{optimizeeabbr}-{optimizerabbr}.py`. Then fill in all the **TODOs** . Especially
the parts with the initialization of the appropriate `Optimizers` and `Optimizees`. The rest of the code should
be left in place for logging and PyPet.

Coding Guidelines
=================
* Always use the `logger` object obtained from `logger = logging.getLogger('logger-name')` to output messages to a
  console/file. You can modify the :file:`bin/logging.yaml` file to choose the output level and to redirect messages to
  console or file.


Adding new algorithms
=====================

Writing a new inner-loop Optimizee
----------------------------------
.. autoclass:: ltl.optimizees.optimizee.Optimizee
   :members:

Writing a new outer-loop Optimizer
----------------------------------
.. autoclass:: ltl.optimizers.optimizer.Optimizer
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

