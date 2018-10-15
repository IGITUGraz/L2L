import numpy as np

from l2l.optimizees.optimizee import Optimizee


class FunctionGeneratorOptimizee(Optimizee):
    """
    Implements a simple function optimizee. Functions are generated using the FunctionGenerator.
    NOTE: Make sure the optimizee_fitness_weights is set to (-1,) to minimize the value of the function

    :param traj: The trajectory used to conduct the optimization.
    :param fg_instance: Instance of the FunctionGenerator class.
    :param seed: The random seed used for generation of optimizee individuals. It uses a copy of
        the fg_instance and overrides the random generator using one seeded by `seed`. Note that this
        random generator is also the one used by the :class:`.FunctionGeneratorOptimizee` itself.
        NOTE that this seed is converted to an np.uint32.
    """

    def __init__(self, traj, fg_instance, seed):
        super().__init__(traj)

        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)

        self.fg_instance = fg_instance
        self.dims = self.fg_instance.dims
        self.cost_fn = self.fg_instance.cost_function
        self.bound = self.fg_instance.bound

        # create_individual can be called because __init__ is complete except for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        traj.individual.f_add_parameter('seed', seed)

        self.random_state = np.random.RandomState(seed=seed)

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        return {'coords': self.random_state.rand(self.dims) * (self.bound[1] - self.bound[0]) + self.bound[0]}

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'coords': np.clip(individual['coords'], a_min=self.bound[0], a_max=self.bound[1])}

    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~l2l.utils.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """
        # configure_loggers(exactly_once=True)    # logger configuration is here since this function is paralellised
        # logging is now taken care by jube for each individual

        individual = np.array(traj.individual.coords)
        return (self.cost_fn(individual, random_state=self.random_state), )
