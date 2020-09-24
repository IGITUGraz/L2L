L2L Gradient-free Optimization Framework
++++++++++++++++++++++++++++++++++++++++

.. image:: https://travis-ci.org/Meta-optimization/L2L.svg?branch=master
    :target: https://travis-ci.org/Meta-optimization/L2L
    
.. image:: https://coveralls.io/repos/github/Meta-optimization/L2L/badge.svg?branch=master
    :target: https://coveralls.io/github/Meta-optimization/L2L?branch=master


About
*****

The L2L (Learning-to-learn) gradient-free optimization framework contains well documented and tested implementations of various gradient free optimization algorithms. It also defines an API that makes it easy to optimize (hyper-)parameters for any task (optimizee). All the implementations in this package are parallel and can run across different cores and nodes (but equally well on a single core).

NOTE: The L2L framework is currently in **BETA**

Getting Started
***************


If you are developing a new Optimizee or want to try out a new Optimizee with the Optimizers in the L2L package, install
L2L as a python package. See section `Installing the L2L Package`_ for details on how to install the package (this
automatically installs all requirements). 

Documentation is available at `<https://igitugraz.github.io/L2L/>`_.


Installing the L2L Package
**************************

From the Top-Level directory of the directory, run the following command:

    pip3 install --editable . --process-dependency-links [--user]

*The `--user` flag is to be used if you wish to install in the user path as opposed
to the root path (e.g. when one does not have sudo access)*

The above will install the package by creating symlinks to the code files in the 
relevant directory containing python modules. This means that you can change any
of the code files and see the changes reflected in the package immediately (i.e.
without requiring a reinstall). In order to uninstall one may run the following:

    pip3 uninstall Learning-to-Learn

*Note that if the setup was done using sudo access, then the uninstall must also
be done using sudo access*

Having installed this package, we now have access to the top level `l2l` module
which contains all the relevant modules relevant for using the l2l package.

This should also install the `sphinx` package which should now enable you to build
the documentation as specified below.


Building Documentation
**********************
Run the following command from the `doc` directory

    make html 

And open the documentation with 

   firefox _build/html/index.html

All further (and extensive) documentation is in the html documentation!
