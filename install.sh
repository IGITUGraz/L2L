#!/bin/bash
if [ $# -gt 1 ]
then
    if [ -r $1 ]
    then
        source $1
    fi
fi
pip install http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=latest
pip install git+https://github.com/IGITUGraz/sdict.git@master#egg=sdict
python setup.py develop
