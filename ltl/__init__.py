from collections import OrderedDict
from warnings import warn

import functools

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
    Immutable version
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
    except:
        return default_value


class DummyTrajectory:
    pass
