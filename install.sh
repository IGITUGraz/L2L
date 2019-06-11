#!/bin/bash
if [ $# -gt 1 ]
then
    if [ -r $1 ]
    then
        source $1
    fi
fi
pip install http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=latest
python setup.py develop
