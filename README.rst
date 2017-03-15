.. image:: https://circleci.com/gh/IGITUGraz/LTL.svg?style=svg&circle-token=227d26445f67e74ecc1c8904688859b1c49c292f
    :target: https://circleci.com/gh/IGITUGraz/LTL
    
To Install dependencies:
------------------------

Run this from the LTL directory

    pip3 install --user -r requirements.text

Each optimizees and optimizers may have their own dependencies specified in the requirements.txt file within their
respective package.

To build documentation:
-----------------------
Run the following command from the `doc` directory

    make html 

And open the documentation with 

   firefox _build/html/index.html

All further (and extensive) documentation is in the html documentation!
Go read it!
