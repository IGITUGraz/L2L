Quickstart
==========

Running an experiment
+++++++++++++++++++++

With existing implementations of optimizees and optimizers: 

* See :ref:`l2l-experiments` for an example implementation of an L2L experiment with an arbitrary `Optimizee` and
  `Optimizer`. The source code also contains `many examples <https://github.com/IGITUGraz/L2L/tree/master/bin>`_ of
  scripts for various combinations of Optimizees and Optimizers.
* See :ref:`data-postprocessing` for details on how to use the generated data for plots and analysis.

Writing Optimizees and Optimizers
+++++++++++++++++++++++++++++++++

* See :class:`~l2l.optimizees.functions.optimizee.FunctionGeneratorOptimizee` for an example of an `Optimizee` (based on simple
  function minimization).
* See :class:`~l2l.optimizers.simulatedannealing.optimizer.SimulatedAnnealingOptimizer` for an example of an
  implementation of simulated annealing `Optimizer`.
