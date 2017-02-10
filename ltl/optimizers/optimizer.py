from pypet import cartesian_product


class Optimizer:
    def __init__(self):
        self.g = None
        self.eval_pop = None

    def _expand_trajectory(self, traj):
        # Add as many explored runs as individuals that need to be evaluated.
        # Furthermore, add the individuals as explored parameters.
        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`:
        # This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``,
        # so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product({'generation': [self.g],
                                         'ind_idx': range(len(self.eval_pop)),
                                         'individual': self.eval_pop},
                                        [('ind_idx', 'individual'), 'generation']))
