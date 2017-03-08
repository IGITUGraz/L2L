
import logging
from collections import namedtuple

import numpy as np

from ltl.optimizers.optimizer import Optimizer
from ltl.optimizers.crossentropy.distribution import Distribution
from ltl import dict_to_list, list_to_dict
logger = logging.getLogger("ltl-ce")

CrossEntropyParameters = namedtuple('CrossEntropyParameters',
                                    ['n_sample', 'rho', 'distribution', 'smooth_update',
                                     'n_iteration', 'stop_criterion', 'seed'])
CrossEntropyParameters.__doc__ = """
:param n_sample: Number of individuals to sample for distribution update
:param rho: Rho quantile determining fitness threshold for distribution update
:param Distribution distribution: Distribution to sample the individuals from
:param smooth_update: Weight of sampled rho-quantile distribution parametrization in next parametrization
:param n_iteration: Number of iterations to carry out
:param stop_criterion: Stop if change in fitness is below this value
:param seed: Random seed
"""


class CrossEntropyOptimizer(Optimizer):
    """
    Class for a generic cross entropy optimizer.
    In the pseudo code the algorithm does:

    For n iterations do:
        - Sample n_sample individuals and evaluate fitness
        - Sort and determine rho quantile of fitnesses
        - Update distribution parametrization (max lh) with regard to sampled rho quantile individuals

    NOTE: This expects all parameters of the system to be of floating point

    :param  ~pypet.trajectory.Trajectory traj:
      Use this pypet trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`
    
    :param optimizee_create_individual:
      Function that creates a new individual
    
    :param optimizee_fitness_weights: 
      Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each
      element of the fitness vector)
    
    :param parameters: 
      Instance of :func:`~collections.namedtuple` :class:`SimulatedAnnealingParameters` containing the
      parameters needed by the Optimizer
    
    :param optimizee_bounding_func:
      This is a function that takes an individual as argument and returns another individual that is
      within bounds (The bounds are defined by the function itself). If not provided, the individuals
      are not bounded.
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters)
        self.optimizee_bounding_func = optimizee_bounding_func
        
        # The following parameters are recorded
        traj.f_add_parameter('n_sample', parameters.n_sample, comment='Sampling size')
        traj.f_add_parameter('rho', parameters.rho, comment='Rho quantile')
        traj.f_add_parameter('distribution', parameters.distribution.name, comment='Distribution function')
        traj.f_add_parameter('smooth_update', parameters.smooth_update, comment='Smooth update weight')

        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')

        _, self.optimizee_individual_dict_spec = \
            dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)

        # Determine initial distribution
        self.eval_pop = []
        for i in range(parameters.n_sample):
            individual = self.optimizee_create_individual()
            if self.optimizee_bounding_func is not None:
                individual = self.optimizee_bounding_func(individual)
            self.eval_pop.append(dict_to_list(individual))

        # Max Likelihood
        self.current_distribution = parameters.distribution.fit(self.eval_pop)

        traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        n_iteration, stop_criterion = traj.n_iteration, traj.stop_criterion

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))

        # Add results
        # traj.v_idx = run_index # <- result[0]
        # ind_index = traj.par.ind_idx
        # traj.f_add_result (see sa)

        # check abort criterions

        # sort according to weighted fitness

        # max likelihood distribution parametrization update (distribution.fit())

        # sample n_sample individuals and clear+extend self.eval_pop

        traj.v_idx = -1  # set the trajectory back to default

        logger.info("-- End of generation {} --".format(self.g))

        # ------- Create the next generation by crossover and mutation -------- #
        # not necessary for the last generation
        self.eval_pop.clear()
        if self.g < n_iteration - 1 and stop_criterion > self.current_fitness_value:
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        logger.info("The last individual was %s with fitness %s", self.current_individual, self.current_fitness_value)
        logger.info("-- End of (successful) annealing --")
