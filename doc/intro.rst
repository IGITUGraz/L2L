===============================
Welcome to LTL's documentation!
===============================

LTL
===

Introduction
------------
This is Learning to Learn framework for experimenting with many different algorithm. The basic idea behind "Learning to
Learn" is to have an "outer loop" optimizer optimizing the parameters an "inner loop" optimizee. This particular
framwork is written for the case where the cycle goes as follows:

1. The outer-loop optimizer generates an instance of a set of parameters and provides it to the inner-loop optimizee
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

One single instance of a parameter is called an **"individual"** borrowing terminology from evolutionary algorithms.
Similarly, the set of parameters evaluated at each **"generation"** is called a **"population"**.

`PyPet <https://pypet.readthedocs.io/en/latest/>`_'s interface is used extensively to:

1. Run the :ref:`iteration-loop` that:

   a) Runs `Optimizees` (potentially in parallel) to evaluate the fitness of each *individual* from  *population* of
      *individuals* (i.e. parameters) 
   b) Feeds back the results of the fitness evaluation back to the `Optimizer` and generate a new *population* of
      parameters
   c) Performs a) for the new set of parameters.
2. Manage the :ref:`communication` between `Optimizer` and `Optimizee`.  This is done using the the :obj:`traj` object of type 
   :class:`~pypet.trajectory.Trajectory`.
3. Store the results of all the runs, both the parameters and their fitnesses along with any other arbitrary data
   included by the user, into a single hdf5 file.

.. _iteration-loop:

Iteration loop
++++++++++++++

The progress of execution in the script shown in :doc:`ltl-bin` goes as follows:

1. At the beginning, a *population* of *individuals*  is created by the `Optimizer` by calling the `Optimizee`'s
   :meth:`~.Optimizee.create_individual` method.
2. The `Optimizer` then puts these *individuals* in its member variable :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop` 
   and calls its :meth:`~ltl.optimizers.optimizer.Optimizer._expand_trajectory` method. This is the \*key\* step to
   starting and continuing the loop and should be done in all new `Optimizer` s added.

   .. _third-step:

3. `PyPet <https://pypet.readthedocs.io/en/latest/>`_  creates one `Optimizee` run for each *individual* (parameter) 
   in :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop` by calling the `Optimizee`'s :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` 
   method.  Each `Optimizee` run can happen in parallel across cores and even across nodes if enabled as described in
   :ref:`parallelization`. 
4. the `Optimizee`'s :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` method runs whatever simulation it has to run
   with the given *individual* (parameter) and returns a Python :obj:`tuple` with one or more fitness values [#]_.
5. Once the runs are done, `PyPet <https://pypet.readthedocs.io/en/latest/>`_ calls the `Optimizer`'s 
   :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` method [#]_ with the list of *individuals* and their fitness
   values as returned by the `Optimizee`'s :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` method.
   The `Optimizer` can choose to do whatever it wants with the fitnesses, and use
   it to create a new set of *individuals* which it puts into its :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop`
   attribute (after clearing it of the old *population*). 
6. The loop continues from :ref:`3. <third-step>`

   
.. [#] **NOTE:** Even if there is only one fitness value, this function should still return a :obj:`tuple`
.. [#] This is done using `PyPet <https://pypet.readthedocs.io/en/latest/>`_'s postprocessing facility and its
   :meth:`~pypet.trajectory.Trajectory.f_expand()` function as documented `here
   <http://pypet.readthedocs.io/en/latest/cookbook/environment.html#expanding-your-trajectory-via-post-processing>`_. 
   
.. _communication:

Communication
+++++++++++++

The key method of communication between the optimizer and the optimizee is the :obj:`traj` object passed in both to
:meth:`~ltl.optimizees.optimizee.Optimizee.simulate` and to :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
methods.

This :obj:`traj` object is an instance of class :class:`~pypet.trajectory.Trajectory` from `PyPet <https://pypet.readthedocs.io/en/latest/>`_. 
To quote from the PyPet website:

    The whole project evolves around a novel container object called trajectory. A trajectory is a container for parameters
    and results of numerical simulations in python. In fact a trajectory instantiates a tree and the tree structure will be
    mapped one to one in the HDF5 file when you store data to disk. ...

    ... a trajectory contains parameters, the basic building blocks that completely define the initial
    conditions of your numerical simulations. Usually, these are very basic data types, like integers, floats or maybe a
    bit more complex numpy arrays.

In the :meth:`.Optimizee.simulate`, the *individual* (i.e. parameter) to be simulated can be accessed using
`traj.individual`. 

All the other parameters that we are not exploring over that are required for the simulation can be set using
:meth:`~.f_add_parameter` and accessed with dot notation e.g. `param1` can be accessed as `traj.param1`
. It is recommended to add all such parameters to the trajectory in the constructors of the `Optimizers` and
`Optimizees` and access it from the :obj:`traj` in other functions. This will allow PyPet to store the parameters too.


See the `PyPet documentation <https://pypet.readthedocs.io/en/latest/manual/introduction.html#what-to-do-with-pypet>`_ for more
documentation to understand how PyPet works.


Writing new algorithms
----------------------

* For a new **Optimizee**: Create a copy of the class :class:`~ltl.optimizees.optimizee.Optimizee` into a new python
  module with an appropriate name and fill in the functions. E.g. for a DMS task optimizee, you would create
  a module (i.e. directory with a `__init__.py` file) as `ltl/optimizees/dms/` and copy the above class there.
* For a new **Optimizer**: Create a copy of the class :class:`~ltl.optimizers.optimizer.Optimizer` into a new python
  module with an appropriate name and fill in the functions. (same as above)
* For a new **experiment**: Create a copy of the file :file:`bin/ltl-template.py` with an appropriate name and fill in
  the *TODOs*.

Examples
++++++++

* See :class:`~ltl.optimizees.functions.optimizee.FunctionOptimizee` for an example of an `Optimizee` (based on simple
  function minimization).
* See :class:`~ltl.optimizers.simulatedannealing.optimizer.SimulatedAnnealingOptimizer` for an example of an
  implementation of simulated annealing `Optimizer`.


Optimizee
+++++++++
The optimizee subclasses :class:`~ltl.optimizees.optimizee.Optimizee` with a class that contains three mandatory methods
(Documentation linked below):

1. :meth:`~ltl.optimizees.optimizee.Optimizee.create_individual` : Called to initialize parameters
2. :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` : Runs the actual simulation and returns a fitness vector
3. :meth:`~ltl.optimizees.optimizee.Optimizee.end` : Tertiary method to do cleanup, printing results etc.

See the class documentation for more details: :class:`~ltl.optimizees.optimizee.Optimizee`

Some notes:

* :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` should always return a tuple!

Optimizer
+++++++++
The optimizer subclasses :class:`~ltl.optimizers.optimizer.Optimizer` with a class that contains two mandatory methods:

1. :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` : knowing the fitness for the current parameters, it generates a new set of parameters and runs the next batch of simulations.
2. :meth:`~ltl.optimizers.optimizer.Optimizer.end` : Tertiary method to do cleanup, printing results etc.

See the class documentation for more details: :class:`~ltl.optimizers.optimizer.Optimizer`

Some notes:

* The `Optimizer` should be written (and the existing ones are written) to always maximize the fitness. Set the
  `optimizee_fitness_weights` to a tuple containing a negative value to make it minimize that fitness dimension.
* New runs of the optimizer are trigerred by calls to :meth:`~ltl.optimizers.optimizer.Optimizer._expand_trajectory`
  after setting :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop` to the new list of parameters that need to be
  evaluated in the next cycle
* All the (non-exploring) paramters to the `Optimizer` is passed in to its constructor through a
  :func:`~collections.namedtuple` to keep the paramters documented. For examples see
  :class:`.GeneticAlgorithmParameters` or :class:`.SimulatedAnnealingParameters`

Running an LTL simulation
-------------------------

To run a LTL simulation, copy the file :file:`bin/ltl-template.py` (see :doc:`ltl-bin`) to
:file:`bin/ltl-{optimizeeabbr}-{optimizerabbr}.py`. Then fill in all the **TODOs** . Especially the parts with the
initialization of the appropriate `Optimizers` and `Optimizees`. The rest of the code should be left in place for
logging and PyPet. See the source of :file:`bin/ltl-template.py` for more details.


Coding Guidelines
=================
* Always use the `logger` object obtained from::

    logger = logging.getLogger('logger-name')

  to output messages to a
  console/file. You can modify the :file:`bin/logging.yaml` file to choose the output level and to redirect messages to
  console or file.


Other packages used
-------------------
* `PyPet <https://pypet.readthedocs.io/en/latest/>`_: This is a parameter exploration toolkit that managers exploration
  of parameter space *and* storing the results in a standard format (HDF5).
* `SCOOP <https://scoop.readthedocs.io/en/0.7/>`_: This is optionally used for distributing individual Optimizee
  simulations across multiple hosts in a cluster.

.. _parallelization:

Parallelization
---------------

PyPet also supports running different instances of the experiments on different cores and hosts (using the 
`SCOOP <https://scoop.readthedocs.io/en/0.7/>`_ library). This is enabled by default in the scripts in `bin/`.
To run experiments with scoop, you shoud start your instance of python with `python3 -m scoop script.py`. See the 
`scoop documentation <https://scoop.readthedocs.io/en/0.7/usage.html#how-to-launch-scoop-programs>`_ for more details.


Code documentation
==================
.. toctree::

    ltl
    ltl-bin

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
