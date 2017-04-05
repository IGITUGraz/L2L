.. image:: https://circleci.com/gh/IGITUGraz/LTL.svg?style=svg&circle-token=227d26445f67e74ecc1c8904688859b1c49c292f
    :target: https://circleci.com/gh/IGITUGraz/LTL

Learning To Learn (LTL)
+++++++++++++++++++++++

Getting Started
---------------

Choose one of the following options to get started:

* If you are developing a new Optimizee or want to try out a new Optimizee with the Optimizers in the LTL package, install LTL as a python package. See section `Installing the LTL Package`_ for more details (this automatically installs all requirements). See the `wiki <https://github.com/IGITUGraz/LTL/wiki/Writing-new-Optimizees>`_ for more details.
* If you want to add a new Optimizer or want to modify the function Optimizees in the LTL package, install the requirements according to `Installing Requirements`_.

In both cases, you should build the documentation according to `Building Documentation`_ and read it.
   

Installing the LTL Package
--------------------------

From the Top-Level directory of the directory, run the following command:

    python3 setup.py develop [--user]

*The `--user` flag is to be used if you wish to install in the user path as opposed
to the root path (e.g. when one does not have sudo access)*

The above will install the package by creating symlinks to the code files in the 
relevant directory containing python modules. This means that you can change any
of the code files and see the changes reflected in the package immediately (i.e.
without requiring a reinstall). In order to uninstall one may run the following:

    python3 setup.py develop --uninstall

*Note that if the setup was done using sudo access, then the uninstall must also
be done using sudo access*

Having installed this package, we now have access to the top level `ltl` module
which contains all the relevant modules relevant for using the ltl package.

This should also install the `sphinx` package which should now enable you to build
the documentation as specified below.

Installing Requirements
-----------------------

If you wish to install LTL as a python package, then you may ignore this section and continue from the section `Installing the LTL Package`_. If you however, wish to use it as-is, without installing, then the relevant requirements must be installed. This can be done by running the following from the root directory of the repository:

    pip3 install --user -r requirements.text

Each optimizees and optimizers may have their own dependencies specified in the requirements.txt file within their
respective package.

Building Documentation
----------------------
Run the following command from the `doc` directory

    make html 

And open the documentation with 

   firefox _build/html/index.html

All further (and extensive) documentation is in the html documentation!
Go read it!
