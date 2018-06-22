from utils.sdict import sdictm

class ParameterGroup():
    def __init__(self):
        self.params = {}

    def f_add_parameter(self, key, val, comment=""):
        self.params[key] = val
        print(self.params)

    def __str__(self):
        return str(self.params)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class ResultGroup(sdictm):
    def __init__(self):
        self._data = {}

    def f_add_result_group(self, name, comment = ""):
        self._data[name] = ResultGroup()
        self.comment = comment

    def f_add_result(self,key, val):
        if '.' in str(key):
            subkey = key.split('.')
            if subkey[0] in self._data.keys():
                self._data[subkey[0]].f_add_result(subkey[1], val)
        else:
            self._data[key] = val

    def f_add_result_to_group(self, group_name, key, val):
        self._data[group_name].f_add_result(key, val)

    def __str__(self):
        return str(self.results)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class ParameterDict(sdictm):

    def __init__(self, traj):
        super(ParameterDict, self).__init__({})
        self.trajectory = traj

    def __getattr__(self, attr):
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