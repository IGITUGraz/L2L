import logging
from collections import namedtuple

import numpy as np

from ltl.optimizers.optimizer import Optimizer
from ltl import dict_to_list, list_to_dict
logger = logging.getLogger("ltl-ce")

CrossEntropyParameters = namedtuple('CrossEntropyParameters',
                                    ['pop_size', 'rho', 'smoothing', 'temp_decay', 'n_iteration', 'stop_criterion', 'distribution'])
CrossEntropyParameters.__doc__ = """
:param pop_size: Number of individuals per simulation

:param rho: fraction of solutions to be considered elite in each iteration.

:param smoothing: This is a factor between 0 and 1 that determines the weight assigned to the previous distribution
  parameters while calculating the new distribution parameters. The smoothing is done as a linear combination of the 
  optimal parameters for the current data, and the previous distribution as follows:
    
    new_params = smoothing*old_params + (1-smoothing)*optimal_new_params

:param temp_decay: This parameter is the factor (necessarily between 0 and 1) by which the temperature decays each
  generation. To see the use of temperature, look at the documentation of :class:`CrossEntropyOptimizer`

:param n_iteration: Number of iterations to perform

:param stop_criterion: Stop if this fitness is reached.

:param distribution: Distribution class to use. Has to implement a fit and sample function.
"""


class CrossEntropyOptimizer(Optimizer):
    """
    Class for a generic cross entropy optimizer.
    In the pseudo code the algorithm does:

    For n iterations do:
      - Sample individuals from distribution
      - evaluate individuals and get fitness
      - pick rho * pop_size number of elite individuals
      - Out of the remaining non-elite individuals, select them using a simulated-annealing style
        selection based on the difference between their fitness and the `1-rho` quantile (*gamma*)
        fitness, and the current temperature
      - Fit the distribution family to the new elite individuals by minimizing cross entropy.
        The distribution fitting is smoothed to prevent premature convergence to local minima.
        A weight equal to the `smoothing` parameter is assigned to the previous parameters when
        smoothing.     

    return final distribution parameters.
    (The final distribution parameters contain information regarding the location of the maxima)
    
    NOTE: This expects all parameters of the system to be of numpy.float64. Note that this irritating
    restriction on the kind of floating point type rewired is put in place due to PyPet's crankiness
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
      Instance of :func:`~collections.namedtuple` :class:`CrossEntropyParameters` containing the
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
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion,
                             comment='Stop if best individual reaches this fitness')
        # cannot keep track of class type
        #traj.f_add_parameter('distribution', parameters.distribution,
        #                     comment='Distribution function')
        traj.f_add_parameter('smoothing', parameters.smoothing,
                             comment='Weight of old parameters in smoothing')
        traj.f_add_parameter('temp_decay', parameters.temp_decay,
                             comment='Decay factor for temperature')        

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
        self.best_fitness = -np.inf # The best fitness achieved in this run
        self.T = 1  # This is the temperature used to filter evaluated samples in this run

        # The first iteration does not pick the values out of the Gaussian distribution. It picks randomly
        # (or at-least as randomly as optimizee_create_individual creates individuals)
        
        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the cross entropy algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        self.eval_pop = current_eval_pop
        self.eval_pop_asarray = np.array([dict_to_list(x) for x in self.eval_pop])
        
        # Max Likelihood
        self.current_distribution = parameters.distribution(self.eval_pop_asarray)
        
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """

        pop_size, n_elite, n_iteration, dimension, smoothing, temp_decay = \
            traj.pop_size, traj.n_elite, traj.n_iteration, traj.dimension, traj.smoothing, traj.temp_decay

        weighted_fitness_list = []
        #**************************************************************************************************************
        # Storing run-information in the trajectory
        # Reading fitnesses and performing distribution update
        #**************************************************************************************************************
        for run_index, fitness in fitnesses_results:
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx
            
            traj.f_add_result('$set.$.individual', self.eval_pop[ind_index])
            traj.f_add_result('$set.$.fitness', fitness)

            weighted_fitness_list.append(np.dot(fitness, self.optimizee_fitness_weights))
        traj.v_idx = -1 # set trajectory back to default

        # Performs descending arg-sort of weighted fitness
        fitness_sorted_indices = np.argsort(-weighted_fitness_list)

        generation_name = 'generation_{}'.format(self.g)

        # Sorting the data according to fitness
        sorted_eval_pop_asarray = self.eval_pop_asarray[fitness_sorted_indices]
        sorted_weighted_fitness_list = np.asarray(weighted_fitness_list)[fitness_sorted_indices]

        # Filtering, keeping all elite samples, note that this performs sorting as
        # well due to the indirection array used
        elite_eval_pop_asarray = sorted_eval_pop_asarray[:n_elite]

        self.gamma = sorted_weighted_fitness_list[n_elite-1]

        # Keeping non-elite samples with certain probability dependent on temperature (like Simulated Annealing)
        non_elite_selection_probs = np.exp((weighted_fitness_list[n_elite:] - self.gamma)/self.T)
        non_elite_selected_indices = np.random.random(non_elite_selection_probs.size) < non_elite_selection_probs

        non_elite_eval_pop_asarray = sorted_eval_pop_asarray[n_elite:][non_elite_selected_indices]

        final_eval_pop_asarray = np.concatenate((elite_eval_pop_asarray, non_elite_eval_pop_asarray))

        # Fitting New distribution parameters.
        self.current_distribution.fit(final_eval_pop_asarray, smoothing)
        self.best_fitness = sorted_weighted_fitness_list[0]

        logger.info("-- End of generation {} --".format(self.g))
        logger.info("  Evaluated %i individuals" % len(fitnesses_results))
        logger.info('  Best Fitness Individual: {}'.format(self.eval_pop[fitness_sorted_indices[0]]))
        logger.info('  Maximum fitness value: {}'.format(self.best_fitness))
        logger.info('  Calculated gamma: {}'.format(self.gamma))
        self.current_distribution.log(logger)
        
        #**************************************************************************************************************
        # Storing Generation Parameters / Results in the trajectory
        #**************************************************************************************************************
        # These entries correspond to the generation that has been simulated prior to this post-processing run

        traj.results.generation_params.f_add_result(generation_name + '.g', self.g,
                                                    comment='The index of the evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.gamma', self.gamma,
                                                    comment='The fitness threshold inferred from the evaluated '
                                                            'generation (This is used in sampling the next generation')
        traj.results.generation_params.f_add_result(generation_name + '.T', self.T,
                                                    comment='Temperature used to select non-elite elements among the'
                                                            'individuals of the evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.best_fitness', self.best_fitness,
                                                    comment='The highest fitness among the individuals in the '
                                                            'evaluated generation')
        self.current_distribution.add_results(generation_name, traj)

        #**************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        #**************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run

        fitnesses_results.clear()
        self.eval_pop.clear()
        # Not necessary for the last generation
        if self.g < n_iteration - 1 and self.best_fitness < traj.stop_criterion:
            #Sample from the constructed distribution
            self.eval_pop_asarray = self.current_distribution.sample(pop_size)
            self.eval_pop = [list_to_dict(ind_asarray, self.optimizee_individual_dict_spec)
                             for ind_asarray in self.eval_pop_asarray]
            self.g += 1  # Update generation counter
            self.T *= temp_decay
            self._expand_trajectory(traj)

    def end(self):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) CE optimization --")

