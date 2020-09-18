import logging
from collections import namedtuple

import numpy as np

from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.face")

FACEParameters = namedtuple('FACEParameters',
                            ['min_pop_size', 'max_pop_size', 'n_elite', 'smoothing', 'temp_decay', 'n_iteration',
                             'distribution', 'stop_criterion', 'n_expand', 'seed'])

FACEParameters.__doc__ = """
:param min_pop_size: Minimal number of individuals per simulation.
:param max_pop_size: This is the minimum amount of samples taken into account for the FACE algorithm
:param n_elite: Number of individuals to be considered as elite
:param smoothing: This is a factor between 0 and 1 that determines the weight assigned to the previous distribution
  parameters while calculating the new distribution parameters. The smoothing is done as a linear combination of the 
  optimal parameters for the current data, and the previous distribution as follows:
    
    new_params = smoothing*old_params + (1-smoothing)*optimal_new_params
    
:param temp_decay: This parameter is the factor (necessarily between 0 and 1) by which the temperature decays each
  generation. To see the use of temperature, look at the documentation of :class:`.FACEOptimizer`
:param n_iteration: Number of iterations to perform
:param distribution: Distribution class to use. Has to implement a fit and sample function.
:param stop_criterion: (Optional) Stop if this fitness is reached.
:param n_expand: (Optional) This is the amount by which the sample size is increased if FACE becomes active
"""


class FACEOptimizer(Optimizer):
    """
    Class for Fully Adaptive Crossentropy Optimizer (adaptive sample size)
    In the pseudo code the algorithm does:

    For n iterations do:
    
      1. Sample individuals from distribution
      2. evaluate individuals and get fitness
      3. check if gamma or best individuals fitness increased
      4. if not increase population size by n_expand (if not yet max_pop_size else stop) and sample again (1)
         else set pop_size = min_pop_size and proceed
      5. pick n_elite individuals with highest fitness
      6. Out of the remaining non-elite individuals, select them using a simulated-annealing style
         selection based on the difference between their fitness and the `1-rho` quantile (*gamma*)
         fitness, and the current temperature
      7. Fit the distribution family to the new elite individuals by minimizing cross entropy.
         The distribution fitting is smoothed to prevent premature convergence to local minima.
         A weight equal to the `smoothing` parameter is assigned to the previous parameters when
         smoothing.

    return final distribution parameters.
    (The final distribution parameters contain information regarding the location of the maxima)
    
    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store the parameters of the specific runs.
      The parameters should be initialized based on the values in `parameters`
    :param optimizee_create_individual: Function that creates a new individual. All parameters of the Individual-Dict
      returned should be of numpy.float64 type
    :param optimizee_fitness_weights: Fitness weights. The fitness returned by the Optimizee is multiplied by these
      values (one for each element of the fitness vector)
    :param parameters: Instance of :func:`~collections.namedtuple` :class:`.FACEParameters` containing the
      parameters needed by the Optimizer
    """

    def __init__(self, traj, optimizee_create_individual, optimizee_fitness_weights, parameters,
                 optimizee_bounding_func=None):

        super().__init__(traj, optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights, parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func

        if parameters.min_pop_size < 1:
            raise Exception("min_pop_size needs to be greater than 0")
        if parameters.max_pop_size < parameters.min_pop_size:
            raise Exception("max_pop_size needs to be greater or equal to min_pop_size")
        if parameters.n_elite > parameters.min_pop_size:
            raise Exception("n_elite exceeds min_pop_size")
        if parameters.temp_decay < 0 or parameters.temp_decay > 1:
            raise Exception("temp_decay not in range")
        if parameters.smoothing >= 1 or parameters.smoothing < 0:
            raise Exception("smoothing has to be in interval [0, 1)")
        if parameters.seed is None:
            raise Exception("The 'seed' must be set")

        # The following parameters are recorded
        traj.f_add_parameter('min_pop_size', parameters.min_pop_size,
                             comment='Number of minimal individuals simulated in each run')
        traj.f_add_parameter('max_pop_size', parameters.max_pop_size,
                             comment='Maximal individuals in population')
        traj.f_add_parameter('n_elite', parameters.n_elite,
                             comment='Number of individuals to be considered as elite')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('n_expand', parameters.n_expand,
                             comment='Expanding of population size in case of FACE')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion,
                             comment='Stop if best individual reaches this fitness')
        traj.f_add_parameter('smoothing', parameters.smoothing,
                             comment='Weight of old parameters in smoothing')
        traj.f_add_parameter('temp_decay', parameters.temp_decay,
                             comment='Decay factor for temperature')
        traj.f_add_parameter('seed', np.uint32(parameters.seed),
                             comment='Random seed used by optimizer')

        self.random_state = np.random.RandomState(seed=traj.par.seed)
        temp_indiv, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(),
                                                                       get_dict_spec=True)
        traj.f_add_derived_parameter('dimension', len(temp_indiv),
                                     comment='The dimension of the parameter space of the optimizee')

        # Added a generation-wise parameter logging
        traj.results.f_add_result_group('generation_params',
                                        comment='This contains the optimizer parameters that are'
                                                ' common across a generation')

        # The following parameters are recorded as generation parameters i.e. once per generation
        self.g = 0  # the current generation
        # This is the value above which the samples are considered elite in the
        # current generation
        self.gamma = -np.inf
        self.T = 1  # This is the temperature used to filter evaluated samples in this run
        self.pop_size = parameters.min_pop_size  # Population size is dynamic in FACE
        self.best_fitness_in_run = -np.inf
        self.best_individual = None

        # The first iteration does not pick the values out of the Gaussian distribution. It picks randomly
        # (or at-least as randomly as optimizee_create_individual creates individuals)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the cross entropy algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.min_pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        self.eval_pop = current_eval_pop
        self.eval_pop_asarray = np.array([dict_to_list(x) for x in self.eval_pop])

        # Max Likelihood
        self.current_distribution = parameters.distribution
        self.current_distribution.init_random_state(self.random_state)
        self.current_distribution.fit(self.eval_pop_asarray)

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`
        """

        n_elite, n_iteration, smoothing, temp_decay, min_pop_size, max_pop_size = \
            traj.n_elite, traj.n_iteration, traj.smoothing, traj.temp_decay, traj.min_pop_size, traj.max_pop_size
        stop_criterion, n_expand = traj.stop_criterion, traj.n_expand

        weighted_fitness_list = []
        # **************************************************************************************************************
        # Storing run-information in the trajectory
        # Reading fitnesses and performing distribution update
        # **************************************************************************************************************
        for run_index, fitness in fitnesses_results:
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx

            traj.f_add_result('$set.$.individual', self.eval_pop[ind_index])
            traj.f_add_result('$set.$.fitness', fitness)

            weighted_fitness_list.append(np.dot(fitness, self.optimizee_fitness_weights))
        traj.v_idx = -1  # set trajectory back to default

        # Performs descending arg-sort of weighted fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))

        generation_name = 'generation_{}'.format(self.g)

        # Sorting the data according to fitness
        sorted_population = self.eval_pop_asarray[fitness_sorting_indices]
        sorted_fitess = np.asarray(weighted_fitness_list)[fitness_sorting_indices]

        # Elite individuals are with performance better than or equal to the (1-rho) quantile.
        # See original describtion of cross entropy for optimization
        elite_individuals = sorted_population[:n_elite]

        previous_best_fitness = self.best_fitness_in_run
        self.best_individual = list_to_dict(sorted_population[0],
                                            self.optimizee_individual_dict_spec)
        self.best_fitness_in_run = sorted_fitess[0]
        previous_gamma = self.gamma
        self.gamma = sorted_fitess[n_elite - 1]

        logger.info("-- End of generation %d --", self.g)
        logger.info("  Evaluated %d individuals", len(fitnesses_results))
        logger.info('  Best Fitness: %.4f', self.best_fitness_in_run)
        logger.debug('  Calculated gamma: %.4f', self.gamma)

        # **************************************************************************************************************
        # Storing Generation Parameters / Results in the trajectory
        # **************************************************************************************************************
        # These entries correspond to the generation that has been simulated prior to this post-processing run
        traj.results.generation_params.f_add_result_group(generation_name, 'New generation added to results')
        traj.results.generation_params.f_add_result(generation_name + '.g', self.g,
                                                    comment='The index of the evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.gamma', self.gamma,
                                                    comment='The fitness threshold inferred from the evaluated '
                                                            'generation (This is used in sampling the next generation')
        traj.results.generation_params.f_add_result(generation_name + '.T', self.T,
                                                    comment='Temperature used to select non-elite elements among the'
                                                            'individuals of the evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.best_fitness_in_run', self.best_fitness_in_run,
                                                    comment='The highest fitness among the individuals in the '
                                                            'evaluated generation')
        traj.results.generation_params.f_add_result(generation_name + '.pop_size', self.pop_size,
                                                    comment='Population size')

        # Check stopping
        if self.g >= n_iteration or self.best_fitness_in_run >= stop_criterion:
            return

        expand = True
        if self.best_fitness_in_run > previous_best_fitness or self.gamma > previous_gamma:
            # shrink population size
            self.pop_size = (self.pop_size + min_pop_size) // 2
            # new distribution fit
            individuals_to_be_fitted = elite_individuals

            # Temperature dependent sampling of non elite individuals
            if temp_decay > 0:
                # Keeping non-elite samples with certain probability dependent on temperature (like Simulated Annealing)
                non_elite_selection_probs = np.clip(np.exp((weighted_fitness_list[n_elite:] - self.gamma) / self.T),
                                                    amin=0.0, a_max=1.0)
                non_elite_selected_indices = self.random_state.binomial(1, p=non_elite_selection_probs)
                non_elite_eval_pop_asarray = sorted_population[n_elite:][non_elite_selected_indices]
                individuals_to_be_fitted = np.concatenate((elite_individuals, non_elite_eval_pop_asarray))

            # Fitting New distribution parameters.
            self.distribution_results = self.current_distribution.fit(individuals_to_be_fitted, smoothing)
        elif self.pop_size + n_expand <= max_pop_size:
            # Increase pop size by one, resample, FACE part
            logger.info('  FACE increase population size by %d', n_expand)
            self.pop_size += n_expand
        else:
            # Stop algorithm
            expand = False
            logger.warning('  Possibly diverged')

        # Add the results of the distribution fitting to the trajectory
        for parameter_key, parameter_value in self.distribution_results.items():
            traj.results.generation_params.f_add_result(generation_name + '.' + parameter_key, parameter_value)

        # **************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        # **************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run
        fitnesses_results.clear()
        self.eval_pop.clear()
        if expand:
            # Sample from the constructed distribution
            self.eval_pop_asarray = self.current_distribution.sample(self.pop_size)
            self.eval_pop = [list_to_dict(ind_asarray, self.optimizee_individual_dict_spec)
                             for ind_asarray in self.eval_pop_asarray]
            # Clip to boundaries
            if self.optimizee_bounding_func is not None:
                self.eval_pop = [self.optimizee_bounding_func(individual) for individual in self.eval_pop]
                self.eval_pop_asarray = np.array([dict_to_list(x) for x in self.eval_pop])
            self.g += 1  # Update generation counter
            self.T *= temp_decay
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.end`
        """
        best_last_indiv_dict = self.best_individual

        traj.f_add_result('final_individual', best_last_indiv_dict)
        traj.f_add_result('final_fitness', self.best_fitness_in_run)
        traj.f_add_result('n_iteration', self.g)

        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) FACE optimization --")
        logger.info("-- Final distribution parameters --")
        for parameter_key, parameter_value in sorted(self.distribution_results.items()):
            logger.info('  %s: %s', parameter_key, parameter_value)
