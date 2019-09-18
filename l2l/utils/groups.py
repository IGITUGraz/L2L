from l2l import sdictm
import logging

logging = logging.getLogger("util.groups")


class ParameterGroup:
    """
    This class is a Dictionary which can be used to store parameters. It is used to fit the pypet already existing
    interface with the trajectory
    """

    def __init__(self):
        self.params = {}

    def f_add_parameter(self, key, val, comment=""):
        """
        Adds parameter with name key and value val. The comment is ignored for the moment but kept for
        compatibility with the pypet groups
        :param key: Name of the parameter
        :param val: Value of the parameter
        :param comment: Ignores for the moment
        """
        self.params[key] = val

    def __str__(self):
        return str(self.params)

    def __repr__(self):
        return self.params.__repr__()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class ResultGroup(sdictm):
    """
    ResultGroup is a class derived from sdictm, which is a dictorary with parameters accessible using . (dot)
    Used to keep the interface with pypet trajectory result groups
    """

    def __init__(self):
        super(ResultGroup, self).__init__({})
        self._data = {}

    def f_add_result_group(self, name, comment=""):
        """
        Adds a new results group to this dictionary
        :param name: Name of the new result group
        :param comment: Ignored for the moment
        """
        self._data[name] = ResultGroup()

    def f_add_result(self,key, val, comment=""):
        """
        Adds a result in a result group. The name of the result group precedes the name of the result name and
        they are split by a . (dot)
        In case this result is not to be part of a result group, it is added to the root level of the dictionary.
        :param key: the name of the result to add. Preceded by a result group name if it is to be added to an existing
        group. Produces an error if the value is to be added to a non existent result group.
        :param val: Value of the result to be added
        :exception: Produces an exception if the value is to be added to a non existent result group.
        """
        if '.' in str(key):
            subkey = key.split('.')
            if subkey[0] in self._data.keys():
                self._data[subkey[0]].f_add_result(subkey[1], val)
            else:
                logging.exception("Key not found when adding to result group")
                raise Exception("Group name not found when adding value to result group")
        else:
            self._data[key] = val

    def f_add_result_to_group(self, group_name, key, val, comment=""):
        """
        Adds a result in a group.
        :param group_name: name of the group
        :param key: the name of the result to add.
        :param val:
        :exception Produces an exception if the value is to be added to an inexistent result group.
        """
        if group_name in self._data.keys():
            self._data[group_name].f_add_result(key, val)
        else:
            logging.exception("Key not found when adding to result group")
            raise Exception("Group name not found when adding value to result group")

    def __str__(self):
        return str(self.results)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class ParameterDict(sdictm):
    """
    ParameterDict is a class derived from sdictm which takes care of holding parameters in the trajectory
    The interface was kept to match the one from pypet parameters.
    """

    def __init__(self, traj):
        super(ParameterDict, self).__init__({})
        self.trajectory = traj

    def __getattr__(self, attr):
        """
        This function has been overwritten in order to allow a particular access to values in the dictionary.
        If attr is ind_idx, it returns the id from the current result with index trajectory.v_idx
        :param attr: Contains the attribute name to be accessed
        :return: the value of the attribute name indicated by attr
        """
        if attr == '__getstate__':
            raise AttributeError()
        if attr == 'ind_idx':
            return [i[0] for i in self.trajectory.current_results].index(self.trajectory.v_idx)
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

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
