.. image:: https://circleci.com/gh/IGITUGraz/LTL.svg?style=svg&circle-token=227d26445f67e74ecc1c8904688859b1c49c292f
    :target: https://circleci.com/gh/IGITUGraz/LTL

LTL Gradient-free Optimization Framework
++++++++++++++++++++++++++++++++++++++++

About
*****

The LTL (Learning-to-learn) gradient-free optimization framework contains well documented and tested implementations of various gradient free optimization algorithms. It also defines an API that makes it easy to optimize (hyper-)parameters for any task (optimizee). All the implementations in this package are parallel and can run across different cores and nodes (but equally well on a single core). 

NOTE: The LTL framework is currently in **BETA**

Getting Started
***************


If you are developing a new Optimizee or want to try out a new Optimizee with the Optimizers in the LTL package, install
LTL as a python package. See section `Installing the LTL Package`_ for details on how to install the package (this
automatically installs all requirements). See the `wiki <https://github.com/IGITUGraz/LTL/wiki/Writing-new-
Optimizees>`_ for more details on how to write a new optimizee.


Documentation is available at `<https://igitugraz.github.io/LTL/>`_.


Installing the LTL Package
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

Having installed this package, we now have access to the top level `ltl` module
which contains all the relevant modules relevant for using the ltl package.

This should also install the `sphinx` package which should now enable you to build
the documentation as specified below.


Building Documentation
**********************
Run the following command from the `doc` directory

    make html 

And open the documentation with 

   firefox _build/html/index.html

All further (and extensive) documentation is in the html documentation!
