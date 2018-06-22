import logging

from collections import OrderedDict


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
        if '.' in attr:
            # This is triggered exclusively in the case where __getattr__ is called from __getitem__
            attrs = attr.split('.')
            ret = self._data.get(attrs[0])
            for at in attrs[1:]:
                ret = ret[at]
        else:
            ret = self._data.get(attr)
        if ret is None:
            new_val = self.__class__({})
            self._data[attr] = new_val
            ret = new_val
        return ret

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __set__(self, key, value):
        if '.' in key:
            attrs = key.split('.')
            base = self.__getattr__(attrs[0])
            for at in attrs[1:-1]:
                base = base[at]
            base[attrs[-1]] = value
        else:
            if key in self._data:
                logging.warning("Key %s already in dictionary" % key)
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

    def update(self, **kwargs):
        """
        Update the dictionary with the values given in the function (only goes one level down)
        :param kwargs:
        :return:
        """

        for key, value in kwargs.items():
            if key in self._data:
                logging.debug("Replacing {} with {} for key {}".format(self._data[key], value, key))
            else:
                logging.debug("Adding new key {} with value {}".format(key, value))
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

    def __str__(self):
        return str(self._data)


class sdict(sdictm):
    """
    Immutable version of :class:`~ltl.sdictm`
    """

    def __set__(self, attr, value):
        raise NotImplementedError("Immutable dictionary")

    def __setattr__(self, attr, value):
        if attr in self._INSTANCE_VAR_LIST:
            object.__setattr__(self, attr, value)
        else:
            raise NotImplementedError("Immutable dictionary")

    def __getattr__(self, attr):
        """
        Same as sdictm, except that trying to access an unknown key returns None. (So this doesn't work recursively
        for unknown keys)
        :param attr:
        :return:
        """
        if attr == '__getstate__':
            raise AttributeError()
        if attr in self._INSTANCE_VAR_LIST:
            return object.__getattribute__(self, attr)
        if '.' in attr:
            attrs = attr.split('.')
            ret = self._data.get(attrs[0])
            for at in attrs[1:]:
                ret = ret[at]
        else:
            ret = self._data.get(attr)
        return ret

    def update(self, **kwargs):
        raise NotImplementedError("Immutable dictionary")