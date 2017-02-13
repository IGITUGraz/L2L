.. LTL documentation master file, created by
   sphinx-quickstart on Fri Feb 10 18:25:53 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================
Welcome to LTL's documentation!
===============================

LTL
============

Introduction
------------
This is Learning to Learn framework for experimenting with many different algorithm. The basic idea behind "Learning to
Learn" is to have an "outer loop" optimizer optimizing the parameters an "inner loop" optimizee. This particular
framwork is written for the case where the cycle goes as follows:

1. The outer-loop optimizer generates an instance of a set of parameters and provides it to the
   inner-loop optimizee
2. The inner-loop optimizee evaluates how well this set of parameters performs and returns a "fitness" vector for each
   parameter in the set of parameters
3. The outer-loop optimizer generates a new set of parameters using the fitness vector it got back from the inner-loop
   optimizee


On the whole, what this means is that the outer-loop Optimizer works only with parameters and fitness values and doesn't
have access to the actual underlying model of the optimizee. And the only thing the optimizee does is to evaluate the
fitness of the given parameter. This fitness can be anything -- the reward achieved in a domain, the mean squared error
on a dataset etc.

The interface
-------------

One single instance of a parameter is called an "individual" borrowing terminology from evolutionary algorithms. The key
method of communication between the optimizer and the optimizee is the :obj:`traj` object passed in both to :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` and to :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` methods. 

This :obj:`traj` object is an instance of class :class:`~pypet.trajectory.Trajectory` from `PyPet <https://pypet.readthedocs.io/en/latest/>`_. We use the `PyPet <https://pypet.readthedocs.io/en/latest/>`_ package to not only record the parameters and results,
but also communicate between the Optimizer and optimizee using this :obj:`traj` object which contains the fitness values for each parameter in the previous iteration. See the `PyPet documentation <https://pypet.readthedocs.io/en/latest/manual/introduction.html#what-to-do-with-pypet>`_ 
for more documentation to understand how PyPet works.


Writing new algorithms
+++++++++++++++++++++++++

* For a new **Optimizee**: Create a copy of the class :class:`~ltl.optimizees.optimizee.Optimizee` with an appropriate name and fill in the functions.
* For a new **Optimizer**: Create a copy of the class :class:`~ltl.optimizers.optimizer.Optimizer` with an appropriate name and fill in the functions.
* For a new **experiment**: Create a copy of the file :mod:`ltl-template` with an appropriate name and fill in the *TODOs*.

Examples
+++++++++++++

* See :class:`~ltl.optimizees.functions.optimizee.FunctionOptimizee` for an example of optimizee (based on simple function minimization).
* See :class:`~ltl.optimizers.simulatedannealing.optimizer.SimulatedAnnealingOptimizer` for an example of an
  implementation of simulated annealing


Optimizee
+++++++++++++
The optimizee subclasses :class:`~ltl.optimizees.optimizee.Optimizee` with a class that contains three mandatory methods
(Documentation linked below): 

1. :meth:`~ltl.optimizees.optimizee.Optimizee.create_individual` : Called to initialize parameters
2. :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` : Runs the actual simulation and returns a fitness vector
3. :meth:`~ltl.optimizees.optimizee.Optimizee.end` : Tertiary method to do cleanup, printing results etc.

See the class documentation for more details: :class:`~ltl.optimizees.optimizee.Optimizee`

Some notes:

* :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` should always return a tuple!

Optimizer
+++++++++++++
The optimizer subclasses :class:`~ltl.optimizers.optimizer.Optimizer` with a class that contains two mandatory methods:

1. :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` : knowing the fitness for the current parameters, it generates a new set of parameters and runs the next batch of simulations.
2. :meth:`~ltl.optimizers.optimizer.Optimizer.end` : Tertiary method to do cleanup, printing results etc.

See the class documentation for more details: :class:`~ltl.optimizers.optimizer.Optimizer`

Some notes:

* It always maximizes. Set the `optimizee_fitness_weights` to a tuple containing a negative value to make it minimize
* New runs of the optimizer are trigerred by calls to :meth:`~ltl.optimizers.optimizer.Optimizer._expand_trajectory`
  after setting :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop` to the new list of parameters that need to be
  evaluated in the next cycle

Running an LTL simulation
--------------------------

To run a LTL simulation, copy the file :file:`bin/ltl-template.py` to
:file:`bin/ltl-{optimizeeabbr}-{optimizerabbr}.py`. Then fill in all the **TODOs** . Especially the parts with the
initialization of the appropriate `Optimizers` and `Optimizees`. The rest of the code should be left in place for
logging and PyPet. See the source of :file:`bin/ltl-template.py` for more details.


Coding Guidelines
=================
* Always use the `logger` object obtained from `logger = logging.getLogger('logger-name')` to output messages to a
  console/file. You can modify the :file:`bin/logging.yaml` file to choose the output level and to redirect messages to
  console or file.


Other packages used
-------------------
* `PyPet <https://pypet.readthedocs.io/en/latest/>`_: This is a parameter exploration toolkit that managers exploration of parameter space *and* storing the results in a standard format (HDF5).
* `SCOOP <https://scoop.readthedocs.io/en/0.7/>`_: This is optionally used for distributing individual Optimizee simulations across multiple hosts in a cluster.


Parallelization
-----------------

PyPet also supports running different instances of the experiments on different cores and hosts (using the `SCOOP <https://scoop.readthedocs.io/en/0.7/>`_ library). This is enabled by default in the scripts in `bin/`. 
To run experiments with scoop, you shoud start your instance of python with `python3 -m scoop script.py`. See the `scoop documentation <https://scoop.readthedocs.io/en/0.7/usage.html#how-to-launch-scoop-programs>`_ for more details.

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
