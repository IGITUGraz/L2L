Overview
========

Introduction
************
This is the Learning to Learn gradient-free optimization framework for experimenting with many different algorithms. The
basic idea behind "Learning to Learn" is to have an "outer loop" optimizer optimizing the parameters of an "inner loop"
optimizee. This particular framework is written for the case where the cycle goes as follows:

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


The progress of execution in the script shown in :doc:`l2l-bin` goes as follows:

1. At the beginning, a *population* of *individuals*  is created by the `Optimizer` by calling the `Optimizee`'s
   :meth:`~.Optimizee.create_individual` method.
2. The `Optimizer` then puts these *individuals* in its member variable :attr:`~l2l.optimizers.optimizer.Optimizer.eval_pop`
   and calls its :meth:`~l2l.optimizers.optimizer.Optimizer._expand_trajectory` method. This is the \*key\* step to
   starting and continuing the loop and should be done in all new `Optimizer` s added.

   .. _third-step:

3. One `Optimizee` run for each *individual* (parameter) is created in
   :attr:`~l2l.optimizers.optimizer.Optimizer.eval_pop` by calling the `Optimizee`'s
   :meth:`~l2l.optimizees.optimizee.Optimizee.simulate` method.  Each `Optimizee` run can happen in parallel across
   cores and even across nodes if enabled as described in :ref:`parallelization`.
4. the `Optimizee`'s :meth:`~l2l.optimizees.optimizee.Optimizee.simulate` method runs whatever simulation it has to run
   with the given *individual* (parameter) and returns a Python :obj:`tuple` with one or more fitness values [#]_.
5. Once the runs are done, `Optimizer`'s :meth:`~l2l.optimizers.optimizer.Optimizer.post_process` method is called 
   with the list of *individuals* and their fitness values as returned by the `Optimizee`'s
   :meth:`~l2l.optimizees.optimizee.Optimizee.simulate` method.  The `Optimizer` can choose to do whatever it wants with
   the fitnesses, and use it to create a new set of *individuals* which it puts into its
   :attr:`~l2l.optimizers.optimizer.Optimizer.eval_pop` attribute (after clearing it of the old *population*).
6. The loop continues from :ref:`3. <third-step>`


.. [#] **NOTE:** Even if there is only one fitness value, this function should still return a :obj:`tuple`



Writing new algorithms
**********************

* For a new **Optimizee**: Create a copy of the class :class:`~l2l.optimizees.optimizee.Optimizee` into a new python
  module with an appropriate name and fill in the functions. E.g. for a DMS task optimizee, you would create
  a module (i.e. directory with a `__init__.py` file) as `l2l/optimizees/dms/` and copy the above class there.
* For a new **Optimizer**: Create a copy of the class :class:`~l2l.optimizers.optimizer.Optimizer` into a new python
  module with an appropriate name and fill in the functions. (same as above)
* For a new **experiment**: Create a copy of the file :file:`bin/l2l-template.py` with an appropriate name and fill in
  the *TODOs*.
* Add an entry in :file:`bin/logging.yaml` for the new class/file you created. See logging_.

Details for implementing the `Optimizer`, `Optimizee` and experiment follow.


.. _communication:

Important Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~

Trajectory
----------

The Trajectory is a container that is central to the simulation library. This concept is borrowed from `PyPet
<https://pythonhosted.org/pypet/>`_. To quote from the PyPet website:

    The whole project evolves around a novel container object called trajectory. A trajectory is a container for parameters
    and results of numerical simulations in python. In fact a trajectory instantiates a tree and the tree structure will be
    mapped one to one in the HDF5 file when you store data to disk. ...

    ... a trajectory contains parameters, the basic building blocks that completely define the initial
    conditions of your numerical simulations. Usually, these are very basic data types, like integers, floats or maybe a
    bit more complex numpy arrays.

In the simulations using the L2L Framework, there is a single :class:`~l2l.utils.trajectory.Trajectory` object (called
:obj:`traj`). This object forms the backbone of communication between the optimizer and the optimizee. In short, it is
used to acheive the following:

1.  Storage of the parameters of the optimizer, optimizee, and individuals_ of the optimizee
2.  Storage of the results of our simulation
3.  Adaptive exploration of parameters via trajectory expansion.

:obj:`traj` object is passed as a mandatory argument to the constructors of both the Optimizee and Optimizer.
Additionally, we automatically passes this object as an argument to the functions
:meth:`~l2l.optimizees.optimizee.Optimizee.simulate` and :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`

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
    :meth:`~l2l.optimizers.optimizer.Optimizer.post_process` function. See the note about expanding trajectories in
    ref:`optimizer-constructor`

In the documentation above, whenever the term individual_ is used, it is assumed that the object referred to is
an `Individual-Dict`. Also note that the Individual-Dict is not a separate class but merely a specification for
specifying individuals of an optimizee via a dict.

In other places in the documentation, the Individual-Dict may also be referred to as a parameter dict, due to the
fact that its keys represent parameter names.

.. _traj-interaction:

Optimizee
~~~~~~~~~

The optimizee subclasses :class:`~l2l.optimizees.optimizee.Optimizee` with a class that contains four mandatory methods
(Documentation linked below):

1. :meth:`~l2l.optimizees.optimizee.Optimizee.create_individual` : Called to return a random individual_ (returns an Individual-Dict_)
2. :meth:`~l2l.optimizees.optimizee.Optimizee.simulate` : Runs the actual simulation and returns a fitness vector

In order to maintain a consistent framework for communication between the optimizer and optimizee it is required to
enforce certain requirements on the behaviour of the above functions. The details of these requirements for the
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

.. code:: python

    traj.individual.f_add_parameter('sim_control.seed', 1010)

If one intends ``sim_control.seed`` to be a parameter over which to explore, the Individual-Dict_ describing an
individual of the optimizee must contain a key named ``'sim_control.seed'``

*NOTE* that the parameter group named `individual` itself is created in the constructor of the base `Optimizee` class.
Thus, the derived class need only implement the addition of parameters as shown above.

The :meth:`~l2l.optimizees.optimizee.Optimizee.create_individual` function:
---------------------------------------------------------------------------

This must return an individual_ of the optimizee, i.e. it must return an Individual-Dict_ representing
a valid random individual of the optimizee.

The :meth:`~l2l.optimizees.optimizee.Optimizee.simulate` function:
------------------------------------------------------------------

This function only receives as argument the trajectory :obj:`traj` set to a particular run. Thus it must source all
required parameters from the :obj:`traj` and the member variables of the `Optimizee` class. It must run the inner loop
with these parameters and always return a tuple (*even for 1-D fitness!!*) representing the fitness to be used for
optimizing.
_
See the class documentation for more details: :class:`~l2l.optimizees.optimizee.Optimizee`

Optimizer
~~~~~~~~~

The optimizer subclasses :class:`~l2l.optimizers.optimizer.Optimizer` with a class that contains two mandatory methods:

1. :meth:`~l2l.optimizers.optimizer.Optimizer.__init__`: This is the constructor which performs the duties of
   initializing the trajectory and the initial generation_ of the simulation.
2. :meth:`~l2l.optimizers.optimizer.Optimizer.post_process` : knowing the fitness for the current parameters, it
   generates a new set of parameters and runs the next batch of simulations.

And one optional method:

1. :meth:`~l2l.optimizers.optimizer.Optimizer.end` : Tertiary method to do cleanup, printing results etc.

Note that in order to maintain a consistent framework for communication between the optimizer and optimizee, we enforce
a certain protocol for the above function. The details of this protocol are outlined below

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

The :meth:`~l2l.optimizers.optimizer.Optimizer.post_process` function:
----------------------------------------------------------------------

This function receives, along with the trajectory :obj:`traj`, a list of tuples. Each tuple has the structure
`(run_index, run_fitness_tuple)`. The :meth:`~l2l.optimizers.optimizer.Optimizer.post_process` function also has
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

1.  The call to `self._expand_trajectory` not only causes the trajectory to store more parameter values to explore, but,
    due to the mechanism underlying :meth:`~l2l.utils.trajectory.Trajectory.f_expand()`, also causes the framework to run
    the optimizee :meth:`~l2l.optimizees.optimizee.Optimizee.simulate` function on these parameters. Look at the
    documentation referenced in the footnote of iteration-loop_ for more details on this

2.  **Always** build the optimizer to maximize fitness. The weights that are passed in to the optimizer constructor
    can be made negative if one wishes to perform minimization

See the class documentation for more details: :class:`~l2l.optimizers.optimizer.Optimizer`


Running an L2L simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Before running a simulation for the first time, you need to specify the output directory for your results. To do so,
create a new file :file:`bin/path.conf` with a single entry containing an absolute path or a path relative to the top-
level L2L directory, e.g. :file:`./output_results/`, and create an empty folder at the path you specified. You also need
to commit any staged files to your local repo. Failing to follow these instructions raises an error when trying to run
any of the test simulations.

To run a L2L simulation, copy the file :file:`bin/l2l-template.py` (see :doc:`l2l-bin`) to
:file:`bin/l2l-{optimizeeabbr}-{optimizerabbr}.py`. Then fill in all the **TODOs** . Especially the parts with the
initialization of the appropriate `Optimizers` and `Optimizees`. The rest of the code should be left in place for
logging and recording. See the source of :file:`bin/l2l-template.py` for more details.

Execution setup
~~~~~~~~~~~~~~~
The L2L framework works with JUBE in order to deploy the execution of the different instances of the optimizee on
the available computational resources. This requires that the trajectory contains a parameter group called JUBE_params
which contains details for the right execution of the program.

**Mandatory** steps to define the execution of the optimizees:
1. Add a parameter group to the :obj: traj called JUBE_params using its :meth: f_add_parameter_group.
2. Setup the execution command :attr: exec by using the trajectory :meth: f_add_parameter_to_group.
Add parameter to group receives three parameters, which in this case should be specified as:
group_name=JUBE_params, key="exec", val=<execution command string>
This <execution command string> will be used to launch individual optimizees. An example of a simple call without using MPI calls
is: "python " + os.path.join(paths.simulation_path, "run_files/run_optimizee.py"
3. Setup the ready and working paths :attr: exec by using the trajectory :meth: f_add_parameter_to_group.
Add parameter to group receives three parameters, which in this case should be specified as:
group_name=JUBE_params, key="paths", val=<path object>
<path object> should contain the root working path. An example of this path is:
paths = Paths(name, dict(run_num='test'), root_dir_path=<root_dir_path>, suffix="-example")

In order to launch simulations on a laptop or a local cluster without a scheduler, only the mandatory parameters must
be specified. These parameters are part of the template.

To launch the simulations on a cluster with a scheduler, the following optional parameters must be defined. They currently match
slurm but this can also be adjusted to other schedulers.
1. Name of the scheduler, :atr: "scheduler", e.g. "Slurm"
2. Command to submit jobs to the schedulers, :atr: "submit_cmd", e.g. "sbatch"
3. Template file for the particular scheduler, :atr: "job_file", e.g. "job.run"
4. Number of nodes to request for each run, :atr: "nodes", e.g. "1"
5. Requested time for the compute resources, :atr: "walltime", e.g. "00:01:00"
6. MPI Processes per node, :atr: "ppn", e.g. "1"
7. CPU cores per MPI process, :atr: "cpu_pp", e.g. "1"
8. Threads per process, :atr: "threads_pp", e.g. "1"
9. Type of emails to be sent from the scheduler, :atr: "mail_mode", e.g. "ALL"
10. Email to notify events from the scheduler, :atr: "mail_address", e.g. "me@mymail.com"
11. Error file for the job, :atr: "err_file", e.g. "stderr"
12. Output file for the job, :atr: "out_file", e.g. "stdout"
13. MPI Processes per job, :atr: "tasks_per_job", e.g. "1"

See the :file: 'l2l-template-scheduler.py' for a base file with all these parameters.

Examples
********

* See :class:`~.FunctionGeneratorOptimizee` for an example of an `Optimizee` (based on simple function minimization).
* See :class:`~.SimulatedAnnealingOptimizer` for an example of an implementation of simulated annealing `Optimizer`.
* See :ref:`l2l-experiments` for an example implementation of an L2L experiment with an arbitrary `Optimizee` and `Optimizer`.


.. _data-postprocessing:

Data postprocessing
*******************

Todo...

.. _parallelization:

Parallelization
***************

We also support running different instances of the experiments on different cores and hosts using Jube.


.. _logging:

Logging
*******

1.  Always use the `logger` object obtained from::

      logger = logging.getLogger('heirarchical.logger.name')

    to output messages to a console/file.

2.  Setting up logging in a multiprocessing environment is a mind-numbingly painful process. Therefore, to keep users
    sane, we have provided the module :mod:`~l2l.logging_tools` with 2 functions which can be used to conveniently setup
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

While the essential interface between Optimizers and Optimizers is completely defined above, The practical
implementation of Optimizers and Optimizees demands certain frequently used data structures and functions. These are
detailed here

dict-to-list-to-dict Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benefit of treating individuals_ as Individual-Dicts_ is that it allows properly named parameters in the optimizee,
however this comes at the cost of the optimizer being unable to generalize across different Optimizee classes with
different Individual-Dicts_ representing the individual. One solution for this is that most Optimizers prefer to behave
like they are optimizing a vector (in the case of python, a list). Thus, the Optimizer requires the ability to convert
back and forth between a list and a dictionary. For this purpose, we have the following functions

1.  :meth:`~l2l.dict_to_list`
2.  :meth:`~l2l.list_to_dict`

Check their documentation for more details.
