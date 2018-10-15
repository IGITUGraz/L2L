# ***************************************************************************************
# *    Title: pypet/cartesian_product
# *    Author: Robert Meyer
# *    Date: 2018
# *    Code version: 0.4.3
# *    Availability: https://github.com/SmokinCaterpillar/pypet
# LICENCE:
#
# Copyright (c) 2013-2018, Robert Meyer
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
#   Neither the name of the author nor the names of other contributors
#   may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
# ***************************************************************************************/

import itertools as itools


def cartesian_product(parameter_dict, combined_parameters=()):
    """ Generates a Cartesian product of the input parameter dictionary.

    For example:

    >>> print cartesian_product({'param1':[1,2,3], 'param2':[42.0, 52.5]})
    {'param1':[1,1,2,2,3,3],'param2': [42.0,52.5,42.0,52.5,42.0,52.5]}

    :param parameter_dict:

        Dictionary containing parameter names as keys and iterables of data to explore.

    :param combined_parameters:

        Tuple of tuples. Defines the order of the parameters and parameters that are
        linked together.
        If an inner tuple contains only a single item, you can spare the
        inner tuple brackets.


        For example:

        >>> print cartesian_product( {'param1': [42.0, 52.5], 'param2':['a', 'b'], 'param3' : [1,2,3]}, ('param3',('param1', 'param2')))
        {param3':[1,1,2,2,3,3],'param1' : [42.0,52.5,42.0,52.5,42.0,52.5], 'param2':['a','b','a','b','a','b']}

    :returns: Dictionary with cartesian product lists.

    """
    if not combined_parameters:
        combined_parameters = list(parameter_dict)
    else:
        combined_parameters = list(combined_parameters)

    for idx, item in enumerate(combined_parameters):
        if isinstance(item, str):
            combined_parameters[idx] = (item,)

    iterator_list = []
    for item_tuple in combined_parameters:
        inner_iterator_list = [parameter_dict[key] for key in item_tuple]
        zipped_iterator = zip(*inner_iterator_list)
        iterator_list.append(zipped_iterator)

    result_dict = {}
    for key in parameter_dict:
        result_dict[key] = []

    cartesian_iterator = itools.product(*iterator_list)

    for cartesian_tuple in cartesian_iterator:
        for idx, item_tuple in enumerate(combined_parameters):
            for inneridx, key in enumerate(item_tuple):
                result_dict[key].append(cartesian_tuple[idx][inneridx])

    return result_dict
