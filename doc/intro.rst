Overview
========

Introduction
************
This is the Learning to Learn framework for experimenting with many different algorithms. The basic idea behind "Learning to
Learn" is to have an "outer loop" optimizer optimizing the parameters of an "inner loop" optimizee. This particular
framework is written for the case where the cycle goes as follows:

1. The outer-loop optimizer generates an instance of a set of parameters and provides it to the inner-loop optimizee
2. The inner-loop optimizee evaluates how well this set of parameters performs and returns a "fitness" vector for each
   parameter in the set of parameters
3. The outer-loop optimizer generates a new set of parameters using the fitness vector it got back from the inner-loop
   optimizee


On the whole, what this means is that the outer-loop Optimizer works only with parameters and fitness values and doesn't
have access to the actual underlying model of the optimizee. And the only thing the optimizee does is to evaluate the
fitness of the given parameter. This fitness can be anything -- the reward achieved in a domain, the mean squared error
on a dataset etc.

The Interface
*************

Terminology
~~~~~~~~~~~

.. _individual:
.. _individuals:

*Individual*:
  An Individual refers to an instance of hyper-parameters used in the  `Optimizee`. This means that the `Optimizer`
  tests multiple individuals of an optimizee to perform an optimization. This terminology is borrowed from evolutionary
  algorithms. It is equivalent to the hyper-parameters of the `Optimizee`.

.. _generation:

*Generation*:
  This term, again borrowed from evolutionary algorithms, refers to a single iteration of the outer-loop. Refer to
  the Top-Level iteration Description in the introduction to see exactly what is done each generation

.. _population:

*Population*:
  Used to denote a set of individuals_ that are evaluated in the same generation_.

Other terms such as Trajectory, individual-dict, are defined as they are introduced below.


.. _iteration-loop:

Iteration loop
~~~~~~~~~~~~~~


The progress of execution in the script shown in :doc:`ltl-bin` goes as follows:

1. At the beginning, a *population* of *individuals*  is created by the `Optimizer` by calling the `Optimizee`'s
   :meth:`~.Optimizee.create_individual` method.
2. The `Optimizer` then puts these *individuals* in its member variable :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop`
   and calls its :meth:`~ltl.optimizers.optimizer.Optimizer._expand_trajectory` method. This is the \*key\* step to
   starting and continuing the loop and should be done in all new `Optimizer` s added.

   .. _third-step:

3. :ref:`Pypet <Pypet-Section>`  creates one `Optimizee` run for each *individual* (parameter)
   in :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop` by calling the `Optimizee`'s :meth:`~ltl.optimizees.optimizee.Optimizee.simulate`
   method.  Each `Optimizee` run can happen in parallel across cores and even across nodes if enabled as described in
   :ref:`parallelization`.
4. the `Optimizee`'s :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` method runs whatever simulation it has to run
   with the given *individual* (parameter) and returns a Python :obj:`tuple` with one or more fitness values [#]_.
5. Once the runs are done, :ref:`Pypet <Pypet-Section>` calls the `Optimizer`'s
   :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` method [#]_ with the list of *individuals* and their fitness
   values as returned by the `Optimizee`'s :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` method.
   The `Optimizer` can choose to do whatever it wants with the fitnesses, and use
   it to create a new set of *individuals* which it puts into its :attr:`~ltl.optimizers.optimizer.Optimizer.eval_pop`
   attribute (after clearing it of the old *population*).
6. The loop continues from :ref:`3. <third-step>`


.. [#] **NOTE:** Even if there is only one fitness value, this function should still return a :obj:`tuple`
.. [#] This is done using `PyPet <https://pythonhosted.org/pypet/>`_'s postprocessing facility and its
   :meth:`~pypet.trajectory.Trajectory.f_expand()` function as documented `here
   <http://pythonhosted.org/pypet/cookbook/environment.html#expanding-your-trajectory-via-post-processing>`_.

.. _Pypet-Section:

Usage of PyPet
~~~~~~~~~~~~~~

`PyPet <https://pythonhosted.org/pypet/>`_'s interface is used extensively to:

1. Run the :ref:`iteration-loop` that:

   a) Runs `Optimizees` (potentially in parallel) to evaluate the fitness of each individual_ in a population_.
   b) Feeds back the results of the fitness evaluation back to the `Optimizer` which generates a new population
      of individuals to evaluate.
   c) Loop back to a) with the new set of parameters.
2. Manage the :ref:`communication` between `Optimizer` and `Optimizee`.  This is done using the the :obj:`traj` object of type
   :class:`~pypet.trajectory.Trajectory`.
3. Store the results of all the runs, both the parameters and their fitnesses along with any other arbitrary data
   included by the user, into a single hdf5 file.

Note that most of the pypet functionality, especially those regarding the usage of trajectories is NOT abstracted
out. The user is therefore engouraged to familiarize himself with the working of pypet trajectories.


Writing new algorithms
**********************

* For a new **Optimizee**: Create a copy of the class :class:`~ltl.optimizees.optimizee.Optimizee` into a new python
  module with an appropriate name and fill in the functions. E.g. for a DMS task optimizee, you would create
  a module (i.e. directory with a `__init__.py` file) as `ltl/optimizees/dms/` and copy the above class there.
* For a new **Optimizer**: Create a copy of the class :class:`~ltl.optimizers.optimizer.Optimizer` into a new python
  module with an appropriate name and fill in the functions. (same as above)
* For a new **experiment**: Create a copy of the file :file:`bin/ltl-template.py` with an appropriate name and fill in
  the *TODOs*.
* Add an entry in :file:`bin/logging.yaml` for the new class/file you created. See logging_.

Details for implementing the `Optimizer`, `Optimizee` and experiment follow.


.. _communication:

Important Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~

Trajectory
----------

The Trajectory is a container that is central to the pypet simulation library. To quote from the PyPet website:

    The whole project evolves around a novel container object called trajectory. A trajectory is a container for parameters
    and results of numerical simulations in python. In fact a trajectory instantiates a tree and the tree structure will be
    mapped one to one in the HDF5 file when you store data to disk. ...

    ... a trajectory contains parameters, the basic building blocks that completely define the initial
    conditions of your numerical simulations. Usually, these are very basic data types, like integers, floats or maybe a
    bit more complex numpy arrays.

In the simulations using the LTL Framework, there is a single :class:`~pypet.trajectory.Trajectory` object (called
:obj:`traj`). This object forms the backbone of communication between the optimizer, optimizee, and the
PyPet framework. In short, it is used to acheive the following:

1.  Storage of the parameters of the optimizer, optimizee, and individuals_ of the optimizee
2.  Storage of the results of our simulation
3.  Adaptive exploration of parameters via trajectory expansion.

:obj:`traj` object is passed as a mandatory argument to the constructors of both the Optimizee and Optimizer.
Additionally, PyPet automatically passes this object as an argument to the functions
:meth:`~ltl.optimizees.optimizee.Optimizee.simulate` and :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`

.. _Individual-Dict:
.. _Individual-Dicts:

Individual-Dict
---------------

This is the data structure used to represent individuals_. This is basically a :class:`dict` that has the parameter
names as keys, and parameter values as values. The following need to be noted about the parameters stored in an
*Individual-Dict*.

1.  The parameter names must be the dot-separated full-name (e.g. ``'sim_control.seed'``) of the parameter.
    This name must be the name by which it is stored in the `individual` parameter group of the :obj:`traj`.
    To understand this, look at :ref:`constructor of Optimizee<optimizee-constructor>`.

2.  The dictionary must contain **exactly** those parameters that are going to be explored by the `Optimizer`.
    This is because this dictionary is used to expand the trajectory in the
    :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` function. See the note about expanding trajectories in
    ref:`optimizer-constructor`

In the documentation above, whenever the term individual_ is used, it is assumed that the object referred to is
an `Individual-Dict`. Also note that the Individual-Dict is not a separate class but merely a specification for
specifying individuals of an optimizee via a dict.

In other places in the documentation, the Individual-Dict may also be referred to as a parameter dict, due to the
fact that its keys represent parameter names.

.. _traj-interaction:

Optimizee
~~~~~~~~~

The optimizee subclasses :class:`~ltl.optimizees.optimizee.Optimizee` with a class that contains four mandatory methods
(Documentation linked below):

1. :meth:`~ltl.optimizees.optimizee.Optimizee.create_individual` : Called to return a random individual_ (returns an Individual-Dict_)
2. :meth:`~ltl.optimizees.optimizee.Optimizee.bounding_func` : Called to return a clipped version of individual_ (returns an Individual-Dict_)
3. :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` : Runs the actual simulation and returns a fitness vector

In order to maintain a consistent framework for communication between the optimizer, optimizee, and :ref:`PyPet <Pypet-Section>`
it is required to enforce certain requirements on the behaviour of the above functions. The details of these requirements for the
`Optimizee` functions are given below

.. _optimizee-constructor:

Constructor of Optimizee
------------------------

This function may perform any one-time initialization operations that are required for the particular optimizee.
In addition to this, It *must perform* the job of initializing parameters in the trajectory. These parameters must
be created in the parameter subgroup named `individual` (i.e. using ``traj.parameter.individual.f_add_parameter()``). The
following is a contract that must be obeyed by this constructor.

  All parameters that are explored for the optimizee must be created in the trajectory under the `individual`
  parameter group. Moreover, the names by which they are stored (excluding the `individual`) must be equal to the
  key of the Individual-Dict entry representing that parameter.

As an example, if one wanted a parameter named ``sim_control.seed`` to be a part of the trajectory, one would do
the following.

    traj.individual.f_add_parameter('sim_control.seed', 1010)

If one intends ``sim_control.seed`` to be a parameter over which to explore, the Individual-Dict_ describing an
individual of the optimizee must contain a key named ``'sim_control.seed'``

*NOTE* that the parameter group named `individual` itself is created in the constructor of the base `Optimizee` class.
Thus, the derived class need only implement the addition of parameters as shown above.

The :meth:`~ltl.optimizees.optimizee.Optimizee.create_individual` function:
---------------------------------------------------------------------------

This must return an individual_ of the optimizee, i.e. it must return an Individual-Dict_ representing
a valid random individual of the optimizee.

The :meth:`~ltl.optimizees.optimizee.Optimizee.simulate` function:
------------------------------------------------------------------

This function only receives as argument the trajectory :obj:`traj` set to a particular run. Thus it must source all
required parameters from the :obj:`traj` and the member variables of the `Optimizee` class. It must run the inner loop
with these parameters and always return a tuple (*even for 1-D fitness!!*) representing the fitness to be used for
optimizing.
_
See the class documentation for more details: :class:`~ltl.optimizees.optimizee.Optimizee`

Optimizer
~~~~~~~~~

The optimizer subclasses :class:`~ltl.optimizers.optimizer.Optimizer` with a class that contains two mandatory methods:

1. :meth:`~ltl.optimizers.optimizer.Optimizer.__init__`: This is the constructor which performs the duties of
   initializing the trajectory and the initial generation_ of the simulation.
2. :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` : knowing the fitness for the current parameters, it
   generates a new set of parameters and runs the next batch of simulations.

And one optional method:

1. :meth:`~ltl.optimizers.optimizer.Optimizer.end` : Tertiary method to do cleanup, printing results etc.

Note that in order to maintain a consistent framework for communication between the optimizer, optimizee, and
:ref:`PyPet <Pypet-Section>`, we enforce a certain protocol for the above function. The details of this protocol
are outlined below

.. _optimizer-constructor:

Constructor of Optimizer
------------------------

Perform any one-time initialization tasks including creating optimizer parameters in the trajectory. Note that
optimizer parameters are created in the root parameter group of :obj:`traj` (i.e. `traj.par.f_add_parameter(...)`).

Create a list of individuals_ by using the `optimizee_create_individual` function. These are the individuals that
will be simulated in the first generation (i.e. generation index 0). Assign `self.eval_pop` to the list of above
individuals, and call `self._expand_trajectory()` to expand the trajectory to include the parameters corresponding
to these individuals.

`self._expand_trajectory()` relies on the fact that the individual_ objects are Individual-Dicts_ and uses the keys
to access and assign the relevant optimizee parameters in the parameter group :obj:`traj.individual`. This is
the reason for the contract enforced on the Optimizee constructor

Note that all the (non-exploring) paramters to the `Optimizer` is passed in to its constructor through a
:func:`~collections.namedtuple` to keep the paramters documented. For examples see :class:`.GeneticAlgorithmParameters`
or :class:`.SimulatedAnnealingParameters`

The :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` function:
----------------------------------------------------------------------

This function receives, along with the trajectory :obj:`traj`, a list of tuples. Each tuple has the structure
`(run_index, run_fitness_tuple)`. The :meth:`~ltl.optimizers.optimizer.Optimizer.post_process` function also has
access to the individuals whose fitness was calculated (via the member `self.eval_pop`), and the generation index
(`self.g`), along with any other user defined member variable.

Using the above, the function must calculate a new population_ of individuals_ to explore (remember individuals
are always in Individual-Dict_ form). It must then store this list of individuals in `self.eval_pop` and call
`self._expand_trajectory()`. [See :ref:`optimizer-constructor` for trajectory expansion details]. Also, `self.g`
must be incremented

In case one wishes to terminate the simulation after the current generation, one must simply **not call**
`self._expand_trajectory()`. Do not call `self._expand_trajectory()` with an empty `self.eval_pop` as it will
raise an error

Some points to remember are the following:

1.  The call to `self._expand_trajectory` not only causes the trajectory to store more parameter values to explore,
    bu, due to the mechanism underlying :meth:`~pypet.trajectory.Trajectory.f_expand()`, also causes the
    :ref:`Pypet <Pypet-Section>` framework to run the optimizee :meth:`~ltl.optimizees.optimizee.Optimizee.simulate`
    function on these parameters. Look at the documentation referenced in the footnote of iteration-loop_ for more
    details on this

2.  **Always** build the optimizer to maximize fitness. The weights that are passed in to the optimizer constructor
    can be made negative if one wishes to perform minimization

.. See the `PyPet documentation <https://pythonhosted.org/pypet/manual/introduction.html#what-to-do-with-pypet>`_ for more
.. documentation to understand how PyPet works.

See the class documentation for more details: :class:`~ltl.optimizers.optimizer.Optimizer`


Running an LTL simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Before running a simulation for the first time, you need to specify the output directory for your results. To do so,
create a new file :file:`bin/path.conf` with a single entry containing an absolute path or a path relative to the top-
level LTL directory, e.g. :file:`./output_results/`, and create an empty folder at the path you specified. You also need
to commit any staged files to your local repo. Failing to follow these instructions raises an error when trying to run
any of the test simulations.

To run a LTL simulation, copy the file :file:`bin/ltl-template.py` (see :doc:`ltl-bin`) to
:file:`bin/ltl-{optimizeeabbr}-{optimizerabbr}.py`. Then fill in all the **TODOs** . Especially the parts with the
initialization of the appropriate `Optimizers` and `Optimizees`. The rest of the code should be left in place for
logging, recording and PyPet. See the source of :file:`bin/ltl-template.py` for more details.


Parameter Bounding
~~~~~~~~~~~~~~~~~~

Most optimizees impose bounds on their parameters in some form. For example the
:class:`~.FunctionGeneratorOptimizee` imposes a rectangular bound on the set of valid coordinates. Most Optimizers
on the other hand do not have direct access to these bounds. Hence, If the optimizer wishes to support bounding, it must
accept a bounding-function_ as an argument.

.. _bounding-function:

Bounding Function:

  This is a function that takes as an argument an individual_ of the Optimizee (an Individual-Dict_) and returns an
  individual_ that is a 'bounded' version of the said individual. This bounding may for instance be implemented by means
  of clipping or normalization. Both the :class:`~.FunctionGeneratorOptimizee` and the
  :class:`~.MNISTOptimizee` implement bounding functions in their classes which may be used in case a
  function is required for bounding.
  NOTE: Remember to un-bound the value in the `Optimizee`'s `simulate` function before using it in your simulation.

Examples
********

* See :class:`~.FunctionGeneratorOptimizee` for an example of an `Optimizee` (based on simple function minimization).
* See :class:`~.SimulatedAnnealingOptimizer` for an example of an implementation of simulated annealing `Optimizer`.
* See :ref:`ltl-experiments` for an example implementation of an LTL experiment with an arbitrary `Optimizee` and `Optimizer`.


.. _data-postprocessing:

Data postprocessing
*******************

Having run the simulation, the next superpower required is the ability to make sense of all the data that we've dumped
into the trajectory and (consequently) the HDF5 file. Of course you could use the functions that pypet provides for this
purpose but the complexity of the interface is rather discouraging. Therefore to cover the most common cases (In fact, I
really haven't YET come across any other cases), We have created the :mod:`~ltl.dataprocessing` with the relevant
functions. Look up the documentation of the module for further details.

.. _parallelization:

Parallelization
***************

PyPet also supports running different instances of the experiments on different cores and hosts (using the
`SCOOP <https://scoop.readthedocs.io/en/0.7/>`_ library). The parameters passed to PyPet determine which Parallelization
mode to use.

* For both single-node-multi-core parallelization (SNMCP) and multi-node-multi-core parallelization (MNMCP) pass in the following arguments
  when initializing the :class:`~pypet.environment.Environment` in the `bin` script.

.. code:: python

    multiproc=True,
    use_scoop=True,
    wrap_mode=pypetconstants.WRAP_MODE_LOCAL,

* For SNMCP, run the script with ``python -m scoop <script name>``

* For MNMCP, run the script with ``python -m scoop --hostfile hosts.txt <script name>``, where `hosts.txt` contains a list
  of hosts and the number of cores.

See the `scoop documentation <https://scoop.readthedocs.io/en/0.7/usage.html#how-to-launch-scoop-programs>`_ for more details.

.. _logging:

Logging
*******

1.  Always use the `logger` object obtained from::

      logger = logging.getLogger('heirarchical.logger.name')

    to output messages to a console/file.

2.  Setting up logging in a multiprocessing environment is a mind-numbingly painful process. Therefore, to keep users
    sane, we have provided the module :mod:`~ltl.logging_tools` with 2 functions which can be used to conveniently setup
    logging. See the module documentation for more details.

3.  As far as using loggers is concerned, the convention is one logger per file. The name of the logger should reflect
    module hierarchy. For example, the logger used in the file `optimizers/crossentropy/optimizer.py` is named
    ``'optimizers.crossentropy'``

4.  A logger is uniquely identified by its name throughout the python process (i.e. it's kinda like a global variable).
    Thus if two different files use ``'optimizer.crossentropy'`` then their logs will be redirect to the same logger.

You can modify the :file:`bin/logging.yaml` file to choose the output level and to redirect messages to console or
file.

See the `Python logging tutorial <https://docs.python.org/3/howto/logging.html>`_ for more details.

Additional Utilities and Protocols
**********************************

While the essential interface between Optimizers, Optimizers, and :ref:`PyPet <Pypet-Section>` is completely defined
above, The practical implementation of Optimizers and Optimizees demands certain frequently used data structures and
functions. These are detailed here

dict-to-list-to-dict Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benefit of treating individuals_ as Individual-Dicts_ is that it allows properly named parameters in the optimizee,
however this comes at the cost of the optimizer being unable to generalize across different Optimizee classes with
different Individual-Dicts_ representing the individual. One solution for this is that most Optimizers prefer to behave
like they are optimizing a vector (in the case of python, a list). Thus, the Optimizer requires the ability to convert
back and forth between a list and a dictionary. For this purpose, we have the following functions

1.  :meth:`~ltl.dict_to_list`
2.  :meth:`~ltl.list_to_dict`

Check their documentation for more details.



Other packages used
*******************
* `PyPet <https://pythonhosted.org/pypet/>`_: This is a parameter exploration toolkit that managers exploration
  of parameter space *and* storing the results in a standard format (HDF5).
* `SCOOP <https://scoop.readthedocs.io/en/0.7/>`_: This is optionally used for distributing individual Optimizee
  simulations across multiple hosts in a cluster.
