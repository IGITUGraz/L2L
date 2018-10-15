The code in the L2L package will potentially be used by other people. So these guidelines are necessary to make the code useful for others:

General Guidelines
==================

* Since we have so many people working with the same repository, it would help a lot if you are familiar with git. `Try Git <https://try.github.io/levels/1/challenges/1>`_ and `The Git Book <https://git-scm.com/book/en/v2>`_ are two very good resources for learning git.
* Try to follow the existing code style as much as possible, stick to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_, don’t overdo OOP (really), and keep your code well commented. 
* For each pull request and commit, a Python style check is done by running `run-style-check.sh` in the root directory. This has to return with 0 errors for the pull request to be merged in. Before every pull request, you should run this script to make sure there are no errors.
* Write documentation for all your changes, both in the code itself in the form of docstrings, and updates to `intro.rst <https://github.com/IGITUGraz/L2L/blob/master/doc/intro.rst>`_.

Working with the repository
===========================

* Create a separate git branch for yourself and don't work off master. You can either create one branch for yourself or different feature branches for specific changes. Please make sure that the code remains within the private repository – so no public forks. 
* Use `pull requests <https://github.com/IGITUGraz/L2L/pulls>`_ for merging code into the master branch. (See below for details about using pull requests)
* Use `Github issues <https://github.com/IGITUGraz/L2L/issues>`_ for tracking tasks and reporting bugs in the framework.  When you start implementing something specific, create and assign appropriate issues to yourself. You can use this for tracking progress and notes about implementation.

Pull Requests
=============
All project related code has to be merged into master, so it's available to other people.

* For each *logical unit of change* (not the entire project!), open a pull request on GitHub. 
* Either @maharjun or @anandtrex will look at the code and give comments on code style, implementation etc. and once approved, merge it into master.  
* **IMPORTANT:** It’s easier for everyone if you try to merge in changes a small part at a time with pull requests instead of doing a full merge in the end.
