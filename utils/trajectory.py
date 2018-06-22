import time
from utils.groups import ParameterGroup, ResultGroup, ParameterDict
from utils.individual import Individual


class Trajectory:

    def __init__(self, **keyword_args):

        if 'name' in keyword_args:
            self._name = keyword_args['name']
        init_time = time.time()
        self._timestamp = init_time

        self._parameters = ParameterDict(self)  # Contains all parameters
        self._derived_parameters = {}  # Contains all derived parameters
        self._results = {}  # Contains all results
        self._explored_parameters = {}  # Contains all explored parameters, even when they are not
        self._config = {}  # Contains all config parameters
        self._run_information = {}
        self._id = 0
        self._run_started = False  # For manually using a trajectory
        self.individual = Individual()
        self.results = ResultGroup()
        self.results.f_add_result_group('all_results', "Contains all the results")
        self.current_results = {}
        self._parameters.parameter_group= {}
        self._parameters.parameter= {}
        self.individuals = {}
        self.v_idx = 0

    def f_add_parameter_group(self, name, comment):
        self._parameters[name] = ParameterGroup()

    def f_add_parameter_to_group(self, group_name, key, val):
        self._parameters[group_name].f_add_parameter(key, val)

    def f_add_result(self,key, val):
        if key == 'generation_params':
            self.results[key] = ResultGroup()
        else:
            self._results[key] = val

    def f_add_parameter(self, key, val, comment=""):
        self._parameters[key] = val
        print(self._parameters)

    def f_expand(self, build_dict, fail_safe=True):
        params = {}
        gen = []
        ind_idx = []
        for key in build_dict.keys():
            if key == 'generation':
                gen = build_dict['generation']
            elif key == 'ind_idx':
                ind_idx = build_dict['ind_idx']
            else:
                params[key] = build_dict[key]
        generation = gen[0]
        self.individuals[generation] = []

        for i in ind_idx:
            ind = Individual(generation,i,[])
            for j in params:
                ind.f_add_parameter(j, params[j][i])
            self.individuals[generation].append(ind)

    def __str__(self):
        return str(self._parameters)

    def __getattr__(self, attr):
        if '.' in attr:
            # This is triggered exclusively in the case where __getattr__ is called from __getitem__
            attrs = attr.split('.')
            ret = self._parameters.get(attrs[0])
            for at in attrs[1:]:
                ret = ret[at]
        elif attr == 'par':
            ret = self._parameters
        else:
            ret = self._parameters.get(attr,default_value=None)
        return ret

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __getstate__(self):
        print (self.__dict__)
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)