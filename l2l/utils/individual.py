from l2l.utils.groups import ParameterGroup


class Individual(ParameterGroup):
    """
    This class represents individuals in the parameter search. It derives from a Parameter group.
    The main elements which make an individual are the ID of its generation, its individual ID and the
    params specific for its run.
    """

    def __init__(self, generation=0, ind_idx=0, params=[]):
        """
        Initialization of the individual
        :param generation: ID of the generation to which this individual belongs to
        :param ind_idx: global ID of the individual
        :param params: individual parameters which are used to execute the optimizee simulate function
        """
        self.params = {}
        for i in params:
            self.f_add_parameter(i.keys()[0], i[i.keys()[0]])
        self.generation = generation
        self.ind_idx = ind_idx

    def __getattr__(self, attr):
        if attr == 'keys':
            return self.params.keys()
        ret = self.params.get('individual.' + attr)
        return ret

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
