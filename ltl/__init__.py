import functools

import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from warnings import warn
from enum import Enum
from collections import Iterable, Mapping
from numbers import Integral, Real
from timeit import default_timer as timer

import numpy as np

__author__ = 'anand'


def static_vars(**kwargs):
    """
    Provides functionality similar to static in C/C++ for functions that use this decorator
    :param kwargs:
    :return:
    """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


class sdictm(object):
    """
    A dictionary which allows accessing it's values using a dot notation. i.e. `d['a']` can be accessed as `d.a`
    Mutable version
    """
    _INSTANCE_VAR_LIST = ['_data']

    def __init__(self, obj):
        self._data = OrderedDict()
        assert obj is not None
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, dict):
                    self._data[key] = self.__class__(val)
                elif isinstance(val, list):
                    self._data[key] = []
                    for v in val:
                        if isinstance(v, dict):
                            self._data[key].append(self.__class__(v))
                        else:
                            self._data[key] = val
                else:
                    self._data[key] = val
        else:
            raise RuntimeError("should be initialized with a dictionary only")
        assert isinstance(self._data, dict)

    def __getattr__(self, attr):
        if attr == '__getstate__':
            raise AttributeError()
        if attr in self._INSTANCE_VAR_LIST:
            return object.__getattribute__(self, attr)
        ret = self._data.get(attr)
        if ret is None:
            warn("Returning None value for {}".format(attr), stacklevel=2)
        return ret

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __set__(self, key, value):
        self._data[key] = value

    def __setitem__(self, key, value):
        self.__set__(key, value)

    def __setattr__(self, attr, value):
        if attr in self._INSTANCE_VAR_LIST:
            object.__setattr__(self, attr, value)
        else:
            self._data[attr] = value

    def get(self, key, default_value):
        value = self[key]
        if value is None:
            return default_value
        else:
            return value

    def keys(self):
        return self._data.keys()

    def todict(self):
        dic_data = OrderedDict()
        for key, value in self._data.items():
            if isinstance(value, sdictm):
                dic_data[key] = value.todict()
            elif isinstance(value, list):
                dic_data[key] = []
                for v in value:
                    if isinstance(v, sdictm):
                        dic_data[key].append(v.todict())
                    else:
                        dic_data[key].append(v)
            else:
                dic_data[key] = value
        return dic_data

    def copy(self):
        """
        Return a copy of the class. The copy is deep.
        :return:
        """
        return self.__class__(self.todict())

    def update(self, quiet=False, **kwargs):
        """
        Update the dictionary with the values given in the function (only goes one level down)
        :param kwargs:
        :return:
        """

        print = functools.partial(printq, quiet=quiet)

        for key, value in kwargs.items():
            if key in self._data:
                print("Replacing {} with {} for key {}".format(self._data[key], value, key))
            else:
                print("Adding new key {} with value {}".format(key, value))
            self._data[key] = value

        return self

    def apply(self, fn):
        """
        Recursively apply fn on all leaf key, value pairs
        :param fn:
        :return:
        """
        for key, value in self._data.copy().items():
            if isinstance(value, sdictm):
                value.apply(fn)
            elif isinstance(value, list):
                contains_sdictm = False
                for i, v in enumerate(value):
                    if isinstance(v, sdictm):
                        v.apply(fn)
                        contains_sdictm = True
                if not contains_sdictm:
                    fn(self._data, key, value)
            else:
                fn(self._data, key, value)

    def frozen(self):
        return sdict(self.todict())


class sdict(sdictm):
    """
    Immutable version of :class:`~ltl.sdictm`
    """

    def __set__(self, attr, value):
        raise RuntimeError("Immutable dictionary")

    def __setattr__(self, attr, value):
        if attr in self._INSTANCE_VAR_LIST:
            object.__setattr__(self, attr, value)
        else:
            raise RuntimeError("Immutable dictionary")

    def update(self, **kwargs):
        raise RuntimeError("Immutable dictionary")

    def apply(self, fn):
        raise RuntimeError("Immutable dictionary")


class DictEntryType(Enum):
    Sequence = 0
    Scalar = 1


def dict_to_list(input_dict, get_dict_spec=False):
    """
    This function converts the given dictionary into a list. 

    :param dict input_dict: The dictionary to be converted into a list. It is assumed that each key
      in the dicionary is a string and each value is either a scalar numerical value or an
      iterable.

    :param bool get_dict_spec: This is a flag that indicates whether one wants the dictionary
      specification tuple to be returned. It is disabled by default


      Dict-Specification:
        The Dict-Specification tuple is required to convert back from the list to dictionary. 

        It is a tuple of tuples, one tuple for each key of the dictionary. Each tuple is of the
        following form:

            *(key_name, value_type, value_length)*

        key_type may be one of :attr:`.DictEntryType.Sequence` or :attr:`.DictEntryType.Scalar`. In case the
        object is a scalar, value_length is 1.

    :returns: The list representing the contents of the dict. If get_dict_spec is True, the dict
      specification is also returned. Note that the keys are always sorted ascendingly by name
    """
    return_list = []
    dict_items = sorted(input_dict.items())
    dict_spec = []
    for key, value in dict_items:

        if isinstance(value, Iterable):
            value_list = list(input_dict[key])
            return_list.extend(value_list)
            dict_spec.append((key, DictEntryType.Sequence, len(value_list)))
        else:
            return_list.append(input_dict[key])
            dict_spec.append((key, DictEntryType.Scalar, 1))

    if get_dict_spec:
        return return_list, dict_spec
    else:
        return return_list


def list_to_dict(input_list, dict_spec):
    """
    This function converts a list back into a dict.

    :param list input_list: This is the list to be converted into a dict.

    :param tuple dict_spec: This is the dict specification that conveys information regarding the dict
      to be converted to. See :func:`.dict_to_list` for information about the dict specification.

    :returns: A Dictionary representing the list. Note that for valid behaviour, the list should have been generated by
        :func:`.dict_to_list`. Moreover, This operation is not a perfect inverse of :func:`.list_to_dict`. While the types
        of individual elements are preserved, the types of the iterables inside which they reside are not.
        :func:`.list_to_dict` creates all iterables as python lists
    """
    cursor = 0
    return_dict = {}
    for dict_entry in dict_spec:
        key = dict_entry[0]
        value_type = dict_entry[1]
        value_len = dict_entry[2]
        if value_type == DictEntryType.Sequence:
            return_dict[key] = input_list[cursor:cursor + value_len]
        elif value_type == DictEntryType.Scalar:
            return_dict[key] = input_list[cursor]
        cursor += value_len
    assert cursor == len(input_list), "Incorrect Parameter List length, Somethings not right"
    return return_dict


def get_grouped_dict(dict_iter):
    """
    This function takes an iterator of :class:`dict` objects and returns a grouped dict. It
    assumes that each dict has the same keys. It then returns a dict with the same keys as
    these dicts but with the values as being the list of values across the dicts in dict_iter

    :param dict_iter: An iterable containing dictionaries

    :returns: a grouped dictionary `return_dict` such that `return_dict[key]` is a list with
      `return_dict[key][i] = dict_iter[i][key]`. If the dict_iter is empty, an empty dictionary is returned
    """
    paramdict_tuple = tuple(dict_iter)

    return_dict = {}
    if paramdict_tuple:
        param_names = list(paramdict_tuple[0].keys())
        for par_name in param_names:
            return_dict[par_name] = []

        for param_dict in paramdict_tuple:
            for par_name in param_names:
                return_dict[par_name].append(param_dict[par_name])

    return return_dict


def convert_dict_to_numpy(input_dict):
    """
    This function takes a dictionary as input and converts the arguments to
    numpy data types. This is useful when importing parameters from config
    files.

    NOTE: Only the following data types are converted:

    1.  Scalars integer or float. They are converted to `np.int64` and
        `numpy.float64` respectively

    2.  Iterables. All elements of the iterable are converted to a numpy array
        via ``np.array(...)``

    3.  Dictionaries. Dictionaries are recursively converted

    :param input_dict: This is the dictionary to convert

    :returns: The converted output dictionary
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Iterable) and not isinstance(value, str):
            new_value = np.array(value)
        elif isinstance(value, Integral) and not isinstance(value, bool):
            new_value = np.int64(value)
        elif isinstance(value, Real) and not isinstance(value, bool):
            new_value = np.float64(value)
        elif isinstance(value, Mapping):
            new_value = convert_dict_to_numpy(value)
        else:
            new_value = value
        output_dict[key] = new_value

    return output_dict


def printq(s, quiet):
    if not quiet:
        print(s)


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


def get(obj, key, default_value):
    try:
        return obj[key]
    except Exception:
        return default_value


@contextmanager
def stdout_redirected(filename):
    """
    This context manager causes all writes to stdout (whether within python or its
    subprocesses) to be redirected to the filename specified. For usage, look at
    example below::

        import os

        with stdout_redirected(filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")

    inspired from the article
    http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    
    :param filename: The filename (NOT file stream object, this is to ensure that
        the stream is always a valid file object) to which the stdout is to be
        redirected
    """

    os_stdout_fd = sys.stdout.fileno()
    assert os_stdout_fd == 1, "Doesn't work if stdout is not the actual __stdout__"

    sys.stdout.flush()
    old_stdout = sys.stdout
    old_stdout_fd_dup = os.dup(sys.__stdout__.fileno())

    # # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == os_stdout_fd == 1

    def _redirect_stdout(filestream):
        os.dup2(filestream.fileno(), os_stdout_fd)  # os_stdout_fd writes to 'to' file
        sys.stdout = os.fdopen(os_stdout_fd, 'w')  # Python writes to os_stdout_fd

    def _revert_stdout():
        sys.stdout.close()
        os.dup2(old_stdout_fd_dup, os_stdout_fd)
        sys.stdout = old_stdout

    with open(filename, 'w') as file:
        _redirect_stdout(filestream=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _revert_stdout()


@contextmanager
def stdout_discarded():
    """
    This context manager causes all writes to stdout (whether within python or its
    subprocesses) to be redirected to `os.devnull`, thereby effectively discarding the
    output. For usage look at example below::

        import os

        with stdout_discarded():
            print("from Python")
            os.system("echo non-Python applications are also supported")
    """
    with stdout_redirected(os.devnull):
        yield


class DummyTrajectory:
    def __init__(self):
        self.individual = lambda: None

        def f_add_parameter(*args, **kwargs):
            pass

        self.individual.f_add_parameter = f_add_parameter

    def f_add_parameter_group(self, *args, **kwargs):
        pass


@contextmanager
def timed(logger, section_name='Run'):
    start = timer()
    yield
    end = timer()
    logger.info("{} took {:.3f} seconds".format(section_name, end - start))
