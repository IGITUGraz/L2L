Data Processing Utility Functions
=================================

The following Utility functions make the following assumptions regarding the storage
of data in the trajectory.

1.  All the results for each run data is stored under ``'$set.$'``. To clarify, it
    should not be stored under simply ``'$'``.

2.  All generation wise parameters (i.e. parameters that correspond to a particular
    generation rather than) are stored under generation groups whose name is given
    by "generation_<Number>" (Note that Number does not have trailing 0's). See
    :meth:`~.CrossEntropyOptimizer.post_process` of class
    :class:`~.CrossEntropyOptimizer` to see how generation
    parameters are stored.

Module Functions
----------------

.. automodule:: l2l.dataprocessing
    :members:
    :undoc-members:
    :show-inheritance:
