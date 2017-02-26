import numpy as np

from ltl.optimizees.functions.tools import get_cost_function
from ltl.optimizees.optimizee import Optimizee

class FunctionOptimizee(Optimizee):
    """
    Implements a simple function optimizer. (For ALS assignment 1 question 1)
    There are four possible functions we might want to optimize: 'rastrigin', 'rosenbrock', 'ackley', 'chasm'
    Given the name of the function we want to optimize in the constructor, the simulate returns the value of the function.
    NOTE: Make sure the optimizee_fitness_weights is set to (-1,) to minimize the value of the function

    :param str cost_fn_name: one of 'rastrigin', 'rosenbrock', 'ackley', 'chasm'
    """

    def __init__(self, cost_fn_name):
        self.indiv_param_spec = [('coords', 'seq', 2)]
        super().__init__()
        self.cost_fn, self.bound = get_cost_function(cost_fn_name)

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate ramdomly
        return {'coords':np.random.rand(2) * (self.bound[1] - self.bound[0]) + self.bound[0]}

    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~pypet.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """
        individual = np.array(traj.individual.coords)
        return (self.cost_fn(individual),)
