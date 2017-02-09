import os
from collections import OrderedDict

import re

import itertools

__author__ = 'anand'


class Paths:
    def __init__(self, root_dir_name, param_dict, suffix="", root_dir_path='./results'):
        self._root_dir_name = root_dir_name
        self._root_dir_path = root_dir_path
        self._suffix = suffix
        self._param_combo = order_dict_alphabetically(param_dict)

    @property
    def root_dir_path(self):
        return os.path.join(self._root_dir_path, self._root_dir_name)

    @property
    def output_dir_path(self):
        return os.path.join(self.root_dir_path, make_param_string(**self._param_combo))

    # The functions that should actually be used are below
    @property
    def results_path(self):
        path = os.path.join(self.output_dir_path, "results")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def simulation_path(self):
        path = os.path.join(self.output_dir_path, "simulation")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def data_path(self):
        path = os.path.join(self.output_dir_path, "data")
        os.makedirs(path, exist_ok=True)
        return path

    # General function to generate paths
    def get_fpath(self, name, ext, **kwargs):
        d = self._param_combo.copy()
        d.update(kwargs)
        return os.path.join(self.results_path, "{}-{}{}.{}".format(name, make_param_string(**d), self._suffix, ext))


def make_param_string(delimiter='-', **kwargs):
    param_string = ""
    for key in sorted(kwargs):
        param_string += delimiter
        param_string += key.replace('_', delimiter)
        val = kwargs[key]
        if isinstance(val, float):
            param_string += "{}{:.2f}".format(delimiter, val)
        else:
            param_string += "{}{}".format(delimiter, val)
    param_string = re.sub("^-", "", param_string)
    return param_string


def order_dict_alphabetically(d):
    od = OrderedDict()
    for key in sorted(list(d.keys())):
        assert key not in od
        od[key] = d[key]
    return od


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


class PathsMap:
    def __init__(self, param_lists, args_name, n_networks, suffix):
        self._root_dir_name = args_name
        self._suffix = suffix

        param_lists.update(dict(network_num=range(n_networks)))
        self.param_lists = param_lists

        list_dict = dict_product(param_lists)
        self.paths_map = {}
        for param_combo in list_dict:
            key = tuple(order_dict_alphabetically(param_combo).items())
            assert key not in self.paths_map
            self.paths_map[key] = Paths(args_name, param_combo, suffix)

    @property
    def paths_list(self):
        return list(self.paths_map.values())

    def get(self, **kwargs):
        param_combo = kwargs
        key = tuple(order_dict_alphabetically(param_combo).items())
        return self.paths_map[key]

    def filter(self, **kwargs):
        filtered_list = []
        for key, paths in self.paths_map.items():
            params_combo = OrderedDict(key)
            for param_name, param_value in kwargs.items():
                if params_combo[param_name] != param_value:
                    break
            else:
                filtered_list.append(paths)

        return filtered_list

    # Aggregate reults paths
    @property
    def root_dir_path(self):
        return os.path.join(Paths._HOME_DIR_PATH, self._root_dir_name)

    @property
    def agg_results_path(self):
        path = os.path.join(self.root_dir_path, "results")
        os.makedirs(path, exist_ok=True)
        return path

    def get_agg_fpath(self, name, param_combo, ext, **kwargs):
        d = param_combo.copy()
        d.update(kwargs)
        return os.path.join(self.agg_results_path, "{}-{}{}.{}"
                            .format(name, make_param_string(**d), self._suffix, ext))
