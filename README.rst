.. image:: https://circleci.com/gh/IGITUGraz/LTL.svg?style=svg&circle-token=227d26445f67e74ecc1c8904688859b1c49c292f
    :target: https://circleci.com/gh/IGITUGraz/LTL
   
Installing Requirements
-----------------------

If one wishes to install LTL as a python package, then one may ignore this section and continue from the section `To Install The LTL Package`_. If you however, wish to use it as-is, without installing, then the relevant requirements must be installed. This can be done by running the following from the root directory of the repository:

    pip3 install --user -r requirements.text

Each optimizees and optimizers may have their own dependencies specified in the requirements.txt file within their
respective package.

To Install The LTL Package
-------------------------

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

To build documentation:
-----------------------
Run the following command from the `doc` directory

    make html 

And open the documentation with 

   firefox _build/html/index.html

All further (and extensive) documentation is in the html documentation!
Go read it!
