Quickstart
==========

Running an experiment
+++++++++++++++++++++

With existing implementations of optimizees and optimizers: 

* See :ref:`ltl-experiments` for an example implementation of an LTL experiment with an arbitrary `Optimizee` and
  `Optimizer`. The source code also contains `many examples <https://github.com/IGITUGraz/LTL/tree/master/bin>`_ of
  scripts for various combinations of Optimizees and Optimizers.
* See :ref:`data-postprocessing` for details on how to use the generated data for plots and analysis.

Writing Optimizees and Optimizers
+++++++++++++++++++++++++++++++++

* See :class:`~ltl.optimizees.functions.optimizee.FunctionGeneratorOptimizee` for an example of an `Optimizee` (based on simple
  function minimization).
* See :class:`~ltl.optimizers.simulatedannealing.optimizer.SimulatedAnnealingOptimizer` for an example of an
  implementation of simulated annealing `Optimizer`.
