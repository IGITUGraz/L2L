
import logging
from collections import namedtuple

import numpy as np

from ltl.optimizers.optimizer import Optimizer
from ltl import dict_to_list, list_to_dict, get_grouped_dict
logger = logging.getLogger("ltl-sa")

CrossEntropyParameters = namedtuple('CrossEntropyParameters',
                                    ['pop_size', 'rho', 'n_iteration'])
CrossEntropyParameters.__doc__ = """
:param pop_size: Number of individuals per simulation / Number of parallel Simulated Annealing runs
:param rho: fraction of solutions to be considered elite in each iteration.
:param n_iteration: number of iterations to perform
"""

class CrossEntropyOptimizer(Optimizer):
    """
    Class for a generic cross entropy optimizer.
    In the pseudo code the algorithm does:

    For n iterations do:
        - Sample individuals from distribution
        - evaluate individuals and get fitnesss
        - pick rho * pop_size number of elite individuals
        - Fit the distribution family to the new elite individuals by minimizing cross entropy
    return final distribution parameters.
    (These contain information regarding the location of the maxima)

    NOTE: This expects all parameters of the system to be of numpy.float64. Note that this irritating
    restriction on the kind of floating point type rewuired is put in place due to PyPet's crankiness
    regarding types.

    :param  ~pypet.trajectory.Trajectory traj:
      Use this pypet trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`
    
    :param optimizee_create_individual:
      Function that creates a new individual. All parameters of the Individual-Dict returned should be
      of numpy.float64 type
    
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
        traj.f_add_parameter('pop_size', parameters.pop_size,
                             comment='Number of individuals simulated in each run')
        traj.f_add_parameter('rho', parameters.rho,
                             comment='Fraction of individuals considered elite in each generation')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')

        temp_indiv, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(),
                                                                       get_dict_spec=True)

        traj.f_add_derived_parameter('n_elite', int(parameters.rho*parameters.pop_size + 0.5),
                                     comment='Actual number of elite individuals per generation')
        traj.f_add_derived_parameter('dimension', len(temp_indiv),
                                     comment='The dimension of the parameter space of the optimizee')

        # Added a generation-wise parameter logging
        traj.results.f_add_result_group('generation_params',
                                        comment='This contains the optimizer parameters that are'
                                                ' common across a generation')

        # The following parameters are recorded as generation parameters i.e. once per generation
        self.g = 0  # the current generation
        self.gamma = -np.inf  # This is the value above which the samples are considered elite in the
                              # current generation
        self.best_fitness = -np.inf # The best fitness acheived in this run

        # Distribution parameters
        self.gaussian_center = np.zeros((traj.dimension,), dtype=np.float64)
        self.gaussian_std = np.inf

        # The first iteration does not pick the values out of the gaussian distribution. It picks randomly
        # (or at-least as randomly as optimizee_create_individual creates individuals)
        
        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the cross entropy algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        self.eval_pop = current_eval_pop
        self.eval_pop_asarray = np.array([dict_to_list(x) for x in self.eval_pop])
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """

        pop_size, n_elite, n_iteration, dimension = \
            traj.pop_size, traj.n_elite, traj.n_iteration, traj.dimension

        old_eval_pop = self.eval_pop.copy()
        old_eval_pop_asarray = self.eval_pop_asarray

        self.eval_pop.clear()

        #**************************************************************************************************************
        # Storing run-information in the trajectory
        #**************************************************************************************************************
        for run_index, fitness in fitnesses_results:
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx
            
            traj.f_add_result('$set.$.individual', old_eval_pop[ind_index])
            traj.f_add_result('$set.$.fitness', fitness)
        traj.v_idx = -1

        #**************************************************************************************************************
        # Reading fitnesses and performing distribution update
        #**************************************************************************************************************
        
        dot_product = lambda x, y: sum(f * w for f, w in zip(x, y))
        weighted_fitness_list = np.array([dot_product(fitness, self.optimizee_fitness_weights)
                                          for _, fitness in fitnesses_results])

        # Performs descending arg-sort of weights
        weight_sorted_indices = np.argsort(-weighted_fitness_list)

        generation_name = 'generation_{}'.format(self.g)

        # Filtering, keeping only elite samples, note that this performs sorting as
        # well due to the indirection array used
        old_eval_pop = [old_eval_pop[i] for i in weight_sorted_indices[:n_elite]]
        old_eval_pop_asarray = old_eval_pop_asarray[weight_sorted_indices[:n_elite]]
        weighted_fitness_list = weighted_fitness_list[weight_sorted_indices[:n_elite]]

        # Fitting New distribution parameters.
        self.gamma = weighted_fitness_list[-1]
        self.gaussian_center = np.mean(old_eval_pop_asarray, axis=0)
        self.gaussian_std = np.std(old_eval_pop_asarray, axis=0)
        self.best_fitness = weighted_fitness_list[0]

        traj.v_idx = -1  # set the trajectory back to default
        logger.info("-- End of generation {} --".format(self.g))
        logger.info("  Evaluated %i individuals" % len(fitnesses_results))
        logger.info('  Best Fitness Individual: {}'.format(old_eval_pop[0]))
        logger.info('  Maximum fitness value: {}'.format(self.best_fitness))
        logger.info('  Calculated gamma: {}'.format(self.gamma))
        logger.info('  Inferred gaussian center: {}'.format(self.gaussian_center))
        logger.info('  Inferred gaussian std   : {}'.format(self.gaussian_std))
        
        #**************************************************************************************************************
        # Storing Generation Parameters / Results in the trajectory
        #**************************************************************************************************************
        # These entries correspond to the generation that has been simulated prior to this post-processing run

        traj.results.generation_params.f_add_result(generation_name + '.g', self.g,
                                                    comment='The index of the evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.gamma', self.gamma,
                                                    comment='The fitness threshold inferred from the evaluated '
                                                            'generation (This is used in sampling the next generation')
        traj.results.generation_params.f_add_result(generation_name + '.gaussian_center', self.gaussian_center,
                                                    comment='center of gaussian distribution estimated from the '
                                                            'evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.gaussian_std', self.gaussian_std,
                                                    comment='standard deviation of the gaussian distribution inferred'
                                                            ' from the evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.best_fitness', self.best_fitness,
                                                    comment='The highest fitness among the individuals in the '
                                                            'evaluated generation')

        #**************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        #**************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run

        fitnesses_results.clear()
        # Not necessary for the last generation
        if self.g < n_iteration - 1:
            self.eval_pop_asarray = np.random.normal(loc=self.gaussian_center, scale=self.gaussian_std,
                                                 size=(pop_size, dimension))
            self.eval_pop = [list_to_dict(ind_asarray, self.optimizee_individual_dict_spec)
                             for ind_asarray in self.eval_pop_asarray]
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) CE optimization --")

