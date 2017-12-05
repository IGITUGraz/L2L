from __future__ import division
from __future__ import absolute_import
import logging
from collections import namedtuple

import numpy as np

from ltl import dict_to_list, list_to_dict
from ltl.optimizers.optimizer import Optimizer

logger = logging.getLogger(u"optimizers.naturalevolutionstrategies")

NaturalEvolutionStrategiesParameters = namedtuple(u'NaturalEvolutionStrategiesParameters', [
    u'learning_rate_mu',
    u'learning_rate_sigma',
    u'mu',
    u'sigma',
    u'mirrored_sampling_enabled',
    u'fitness_shaping_enabled',
    u'pop_size',
    u'n_iteration',
    u'stop_criterion',
    u'seed',
])

NaturalEvolutionStrategiesParameters.__doc__ = u"""
:param learning_rate_mu: Learning rate for mean of distribution
:param learning_rate_sigma: Learning rate for standard deviation of distribution
:param mu: Initial mean of search distribution
:param sigma: Initial standard deviation of search distribution
:param mirrored_sampling_enabled: Should we turn on mirrored sampling i.e. sampling both e and -e
:param fitness_shaping_enabled: Should we turn on fitness shaping i.e. using only top `fitness_shaping_ratio` to update
       current individual?
:param pop_size: Number of individuals per simulation.
:param n_iteration: Number of iterations to perform
:param stop_criterion: (Optional) Stop if this fitness is reached.
:param seed: The random seed used for generating new individuals
"""


class NaturalEvolutionStrategiesOptimizer(Optimizer):
    u"""
    Class Implementing the separable natural evolution strategies optimizer in natural coordinates as in:

    Wierstra, D., Schaul, T., Peters, J., & Schmidhuber, J. (2008). Natural evolution strategies.
    In Evolutionary Computation, 2008. CEC 2008.(IEEE World Congress on Computational Intelligence) (pp. 3381-3387).

    Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J. (2010). Exponential natural evolution strategies.
    In Proceedings of the 12th annual conference on Genetic and evolutionary computation (pp. 393-400).

    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., & Schmidhuber, J. (2014). Natural evolution strategies.
    In Journal of Machine Learning Research, 15(1) (pp. 949-980).

    In the pseudo code the algorithm does:

    For n iterations do:
      - Sample individuals z from multinormal search distribution with parameters mu, sigma

        s <- sample from N(0,1)
        z <- mu + sigma * s

      - If mirrored sampling is enabled, also sample individuals with opposite perturbations s

        z <- [mu + sigma * s, mu - sigma * s]

      - evaluate individuals z and get fitnesses F_i(z)
      - Update the parameters of the search distribution as

            mu_{t+1} <- mu_{t+1} + eta_mu * sigma * sum(F_i * s_i)
            sigma_{t+1} <- sigma_t * exp(eta_sigma / 2 * sum(F_i * (s_i ** 2 - 1))

      - If fitness shaping is enabled, F_i is replaced with the utility u_i in the previous step, which is calculated as:

            u_i = max(0, log(n/2 + 1) - log(k)) / sum_{k=1}^{n}{max(0, log(n/2 + 1) - log(k))} - 1 / n

        where k and i are the indices of the individuals in descending order of fitness F_i


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
      Instance of :func:`~collections.namedtuple` :class:`.NaturalEvolutionstrategiesParameters` containing the
      parameters needed by the Optimizer

    :param optimizee_bounding_func:
      This is a function that takes an individual as argument and returns another individual that is
      within bounds (The bounds are defined by the function itself). If not provided, the individuals
      are not bounded.

    """

    def __init__(self,
                 traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):

        super(NaturalEvolutionStrategiesOptimizer, self).__init__(
            traj,
            optimizee_create_individual=optimizee_create_individual,
            optimizee_fitness_weights=optimizee_fitness_weights,
            parameters=parameters,
            optimizee_bounding_func=optimizee_bounding_func)

        self.recorder_parameters = parameters
        self.optimizee_bounding_func = optimizee_bounding_func

        # If a parameter is set to `None`, use default value as described in Wierstra et al. (2014)
        if parameters.learning_rate_mu is None:
            learning_rate_mu = 1.
        else:
            learning_rate_mu = parameters.learning_rate_mu

        if parameters.learning_rate_sigma is None:
            learning_rate_sigma = (3 + np.log(len(parameters.mu))) / (5. * np.sqrt(len(parameters.mu)))
        else:
            learning_rate_sigma = parameters.learning_rate_sigma

        if parameters.pop_size is None:
            pop_size = 4 + int(np.floor(3 * np.log(len(parameters.mu))))
        else:
            pop_size = parameters.pop_size

        if pop_size < 1:
            raise ValueError(u"pop_size needs to be greater than 0")

        # The following parameters are recorded
        traj.f_add_parameter(u'learning_rate_mu', learning_rate_mu, comment=u'Learning rate mu')
        traj.f_add_parameter(u'learning_rate_sigma', learning_rate_sigma, comment=u'Learning rate mu')
        traj.f_add_parameter(u'mu', parameters.mu, comment=u'Initial mean of search distribution')
        traj.f_add_parameter(u'sigma', parameters.sigma, comment=u'Initial standard deviation of search distribution')
        traj.f_add_parameter(
            u'mirrored_sampling_enabled',
            parameters.mirrored_sampling_enabled,
            comment=u'Flag to enable mirrored sampling')
        traj.f_add_parameter(
            u'fitness_shaping_enabled', parameters.fitness_shaping_enabled, comment=u'Flag to enable fitness shaping')
        traj.f_add_parameter(
            u'pop_size', pop_size, comment=u'Number of minimal individuals simulated in each run')
        traj.f_add_parameter(u'n_iteration', parameters.n_iteration, comment=u'Number of iterations to run')
        traj.f_add_parameter(
            u'stop_criterion', parameters.stop_criterion, comment=u'Stop if best individual reaches this fitness')
        traj.f_add_parameter(
            u'seed', np.uint32(parameters.seed), comment=u'Seed used for random number generation in optimizer')

        self.random_state = np.random.RandomState(traj.parameters.seed)

        self.current_individual_arr, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(), get_dict_spec=True)

        traj.f_add_derived_parameter(
            u'dimension',
            self.current_individual_arr.shape,
            comment=u'The dimension of the parameter space of the optimizee')

        # Added a generation-wise parameter logging
        traj.results.f_add_result_group(
            u'generation_params',
            comment=u'This contains the optimizer parameters that are'
                    u' common across a generation')

        # The following parameters are recorded as generation parameters i.e. once per generation
        self.g = 0  # the current generation
        self.pop_size = pop_size  # Population size is dynamic in FACE
        self.best_fitness_in_run = -np.inf
        self.best_individual_in_run = None

        # Set initial parameters of search distribution
        self.mu = traj.mu
        self.sigma = traj.sigma

        # Generate initial distribution
        self.current_perturbations = self._get_perturbations(traj)
        current_eval_pop_arr = (self.mu + self.sigma * self.current_perturbations).tolist()

        self.eval_pop = [list_to_dict(ind, self.optimizee_individual_dict_spec) for ind in current_eval_pop_arr]

        # Bounding function has to be applied AFTER the individual has been converted to a dict
        if optimizee_bounding_func is not None:
            self.eval_pop = [self.optimizee_bounding_func(ind) for ind in self.eval_pop]

        self.eval_pop_arr = np.array([dict_to_list(ind) for ind in self.eval_pop])

        self._expand_trajectory(traj)

    def _get_perturbations(self, traj):
        perturbations = self.random_state.randn(traj.pop_size, *traj.dimension)

        if traj.mirrored_sampling_enabled:
            return np.vstack([perturbations, -perturbations])

        return perturbations

    def get_params(self):
        u"""
        Get parameters used for recorder
        :return: Dictionary containing recorder parameters
        """

        param_dict = self.recorder_parameters._asdict()
        return param_dict

    def post_process(self, traj, fitnesses_results):
        u"""
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """

        n_iteration, stop_criterion, fitness_shaping_enabled = \
            traj.n_iteration, traj.stop_criterion, traj.fitness_shaping_enabled

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

            traj.f_add_result(u'$set.$.individual', self.eval_pop[ind_index])
            traj.f_add_result(u'$set.$.fitness', fitness)

            weighted_fitness_list.append(np.dot(fitness, self.optimizee_fitness_weights))
        traj.v_idx = -1  # set trajectory back to default

        weighted_fitness_list = np.array(weighted_fitness_list).ravel()
        # NOTE: It is necessary to clear the finesses_results to clear the data in the reference, and del
        # is used to make sure it's not used in the rest of this function
        fitnesses_results.clear()
        del fitnesses_results

        # Last fitness is for the previous `current_individual_arr`
        weighted_fitness_list = weighted_fitness_list[:-1]
        current_individual_fitness = weighted_fitness_list[-1]

        # Performs descending arg-sort of weighted fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))

        # Sorting the data according to fitness
        sorted_population = self.eval_pop_arr[fitness_sorting_indices]
        sorted_fitness = np.asarray(weighted_fitness_list)[fitness_sorting_indices]
        sorted_perturbations = self.current_perturbations[fitness_sorting_indices]

        self.best_individual_in_run = sorted_population[0]
        self.best_fitness_in_run = sorted_fitness[0]

        logger.info(u"-- End of generation %d --", self.g)
        logger.info(u"  Evaluated %d individuals", len(weighted_fitness_list) + 1)
        logger.info(u'  Best Fitness: %.4f', self.best_fitness_in_run)
        logger.info(u'  Average Fitness: %.4f', np.mean(sorted_fitness))

        # **************************************************************************************************************
        # Storing Generation Parameters / Results in the trajectory
        # **************************************************************************************************************
        # These entries correspond to the generation that has been simulated prior to this post-processing run

        # Documentation of algorithm parameters for the current generation
        #
        # generation          - The index of the evaluated generation
        # best_fitness_in_run - The highest fitness among the individuals in the
        #                       evaluated generation
        # pop_size            - Population size
        generation_result_dict = {
            u'generation': self.g,
            u'best_fitness_in_run': self.best_fitness_in_run,
            u'current_individual_fitness': current_individual_fitness,
            u'average_fitness_in_run': np.mean(sorted_fitness),
            u'pop_size': self.pop_size
        }

        generation_name = u'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)
        traj.results.generation_params.f_add_result(
            generation_name + u'.algorithm_params',
            generation_result_dict,
            comment=u"These are the parameters that correspond to the algorithm. "
                    u"Look at the source code for `EvolutionStrategiesOptimizer::post_process()` "
                    u"for comments documenting these parameters"
        )

        traj.results.generation_params.f_add_result(
            generation_name + u'.distribution_params', {u'mu': self.mu.copy(), u'sigma': self.sigma.copy()},
            comment=u"These are the parameters of the distribution that underlies the"
                    u" currently evaluated generation")

        if fitness_shaping_enabled:
            fitnesses_to_fit = self._compute_utility(sorted_fitness)
        else:
            fitnesses_to_fit = sorted_fitness

        assert len(fitnesses_to_fit) == len(sorted_perturbations)

        # **************************************************************************************************************
        # Update the parameters of the search distribution using the natural gradient in natural coordinates
        # **************************************************************************************************************
        self.mu += traj.learning_rate_mu * traj.sigma * np.dot(fitnesses_to_fit, sorted_perturbations)
        self.sigma *= np.exp(traj.learning_rate_sigma / 2. * np.dot(fitnesses_to_fit, sorted_perturbations ** 2 - 1.))

        # **************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        # **************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run

        self.eval_pop.clear()

        # check if to stop
        if self.g < n_iteration - 1 and self.best_fitness_in_run < stop_criterion:
            self.current_perturbations = self._get_perturbations(traj)
            current_eval_pop_arr = (self.mu + self.sigma * self.current_perturbations).tolist()

            self.eval_pop = [list_to_dict(ind, self.optimizee_individual_dict_spec) for ind in current_eval_pop_arr]

            # Bounding function has to be applied AFTER the individual has been converted to a dict
            if self.optimizee_bounding_func is not None:
                self.eval_pop = [self.optimizee_bounding_func(ind) for ind in self.eval_pop]

            self.eval_pop_arr = np.array([dict_to_list(ind) for ind in self.eval_pop])

            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def _compute_utility(self, sorted_fitness):
        n_individuals = len(sorted_fitness)
        sorted_utilities = np.array([max(0., np.log((n_individuals / 2) + 1) - np.log(i + 1)) for i in xrange(n_individuals)])
        sorted_utilities /= np.sum(sorted_utilities)
        sorted_utilities -= (1. / n_individuals)
        return sorted_utilities

    def end(self, traj):
        u"""
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        best_last_indiv_dict = list_to_dict(self.best_individual_in_run.tolist(), self.optimizee_individual_dict_spec)

        traj.f_add_result(u'final_individual', best_last_indiv_dict)
        traj.f_add_result(u'final_fitness', self.best_fitness_in_run)
        traj.f_add_result(u'n_iteration', self.g + 1)

        # ------------ Finished all runs and print result --------------- #
        logger.info(u"-- End of (successful) ES optimization --")
