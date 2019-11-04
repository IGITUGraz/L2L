import logging
from collections import namedtuple

import numpy as np

from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.evolutionstrategies")

EvolutionStrategiesParameters = namedtuple('EvolutionStrategiesParameters', [
    'learning_rate',
    'noise_std',
    'mirrored_sampling_enabled',
    'fitness_shaping_enabled',
    'pop_size',
    'n_iteration',
    'stop_criterion',
    'seed',
])

EvolutionStrategiesParameters.__doc__ = """
:param learning_rate: Learning rate
:param noise_std: Standard deviation of the step size (The step has 0 mean)
:param mirrored_sampling_enabled: Should we turn on mirrored sampling i.e. sampling both e and -e
:param fitness_shaping_enabled: Should we turn on fitness shaping i.e. using only top `fitness_shaping_ratio` to update
       current individual?
:param pop_size: Number of individuals per simulation.
:param n_iteration: Number of iterations to perform
:param stop_criterion: (Optional) Stop if this fitness is reached.
:param seed: The random seed used for generating new individuals
"""


class EvolutionStrategiesOptimizer(Optimizer):
    """
    Class Implementing the evolution strategies optimizer

    as in: Salimans, T., Ho, J., Chen, X. & Sutskever, I. Evolution Strategies as a Scalable Alternative to
            Reinforcement   Learning. arXiv:1703.03864 [cs, stat] (2017).

    In the pseudo code the algorithm does:

    For n iterations do:
      - Perturb the current individual by adding a value with 0 mean and `noise_std` standard deviation
      - If mirrored sampling is enabled, also perturb the current individual by subtracting the same values that were
        added in the previous step
      - evaluate individuals and get fitness
      - Update the fitness as

            theta_{t+1} <- theta_t + alpha  * sum{F_i * e_i} / (n * sigma^2)

        where F_i is the fitness and e_i is the perturbation
      - If fitness shaping is enabled, F_i is replaced with the utility u_i in the previous step, which is calculated as:

            u_i = max(0, log(n/2 + 1) - log(k)) / sum_{k=1}^{n}{max(0, log(n/2 + 1) - log(k))} - 1 / n

        As in the paper: Wierstra, D. et al. Natural Evolution Strategies. Journal of Machine Learning Research 15,
         949–980 (2014).

        where k and i are the indices of the individuals in descending order of fitness F_i



    NOTE: This is not the most efficient implementation in terms of communication, since the new parameters are
    communicated to the individuals rather than the seed as in the paper.
    NOTE: Doesn't yet contain fitness shaping and mirrored sampling

    :param  ~l2l.utils.trajectory.Trajectory traj:
      Use this trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`

    :param optimizee_create_individual:
      Function that creates a new individual. All parameters of the Individual-Dict returned should be
      of numpy.float64 type

    :param optimizee_fitness_weights:
      Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each
      element of the fitness vector)

    :param parameters:
      Instance of :func:`~collections.namedtuple` :class:`.CrossEntropyParameters` containing the
      parameters needed by the Optimizer

    """

    def __init__(self,
                 traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):

        super().__init__(
            traj,
            optimizee_create_individual=optimizee_create_individual,
            optimizee_fitness_weights=optimizee_fitness_weights,
            parameters=parameters,
            optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func

        if parameters.pop_size < 1:
            raise Exception("pop_size needs to be greater than 0")

        # The following parameters are recorded
        traj.f_add_parameter('learning_rate', parameters.learning_rate, comment='Learning rate')
        traj.f_add_parameter('noise_std', parameters.noise_std, comment='Standard deviation of noise')
        traj.f_add_parameter(
            'mirrored_sampling_enabled',
            parameters.mirrored_sampling_enabled,
            comment='Flag to enable mirrored sampling')
        traj.f_add_parameter(
            'fitness_shaping_enabled', parameters.fitness_shaping_enabled, comment='Flag to enable fitness shaping')
        traj.f_add_parameter(
            'pop_size', parameters.pop_size, comment='Number of minimal individuals simulated in each run')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iterations to run')
        traj.f_add_parameter(
            'stop_criterion', parameters.stop_criterion, comment='Stop if best individual reaches this fitness')
        traj.f_add_parameter(
            'seed', np.uint32(parameters.seed), comment='Seed used for random number generation in optimizer')

        self.random_state = np.random.RandomState(traj.parameters.seed)

        self.current_individual_arr, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(), get_dict_spec=True)

        noise_std_shape = np.array(parameters.noise_std).shape
        assert noise_std_shape == () or noise_std_shape == self.current_individual_arr.shape

        traj.f_add_derived_parameter(
            'dimension',
            self.current_individual_arr.shape,
            comment='The dimension of the parameter space of the optimizee')

        # Added a generation-wise parameter logging
        traj.results.f_add_result_group(
            'generation_params',
            comment='This contains the optimizer parameters that are'
                    ' common across a generation')

        # The following parameters are recorded as generation parameters i.e. once per generation
        self.g = 0  # the current generation
        self.pop_size = parameters.pop_size  # Population size is dynamic in FACE
        self.best_fitness_in_run = -np.inf
        self.best_individual_in_run = None

        # The first iteration does not pick the values out of the Gaussian distribution. It picks randomly
        # (or at-least as randomly as optimizee_create_individual creates individuals)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the cross entropy algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        self.current_perturbations = self._get_perturbations(traj)
        current_eval_pop_arr = (self.current_individual_arr + self.current_perturbations).tolist()

        self.eval_pop = [list_to_dict(ind, self.optimizee_individual_dict_spec) for ind in current_eval_pop_arr]
        self.eval_pop.append(list_to_dict(self.current_individual_arr, self.optimizee_individual_dict_spec))

        # Bounding function has to be applied AFTER the individual has been converted to a dict
        if optimizee_bounding_func is not None:
            self.eval_pop[:] = [self.optimizee_bounding_func(ind) for ind in self.eval_pop]

        self.eval_pop_arr = np.array([dict_to_list(ind) for ind in self.eval_pop])

        self._expand_trajectory(traj)

    def _get_perturbations(self, traj):
        pop_size, noise_std, mirrored_sampling_enabled = traj.pop_size, traj.noise_std, traj.mirrored_sampling_enabled
        perturbations = noise_std * self.random_state.randn(pop_size, *self.current_individual_arr.shape)
        if mirrored_sampling_enabled:
            return np.vstack((perturbations, -perturbations))
        return perturbations

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`
        """

        n_iteration, stop_criterion, learning_rate, noise_std, fitness_shaping_enabled = \
            traj.n_iteration, traj.stop_criterion, traj.learning_rate, traj.noise_std, traj.fitness_shaping_enabled

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

        weighted_fitness_list = np.array(weighted_fitness_list).ravel()
        # NOTE: It is necessary to clear the finesses_results to clear the data in the reference, and del
        # ^ is used to make sure it's not used in the rest of this function
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

        logger.info("-- End of generation %d --", self.g)
        logger.info("  Evaluated %d individuals", len(weighted_fitness_list) + 1)
        logger.info('  Best Fitness: %.4f', self.best_fitness_in_run)
        logger.info('  Average Fitness: %.4f', np.mean(sorted_fitness))

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
            'generation': self.g,
            'best_fitness_in_run': self.best_fitness_in_run,
            'current_individual_fitness': current_individual_fitness,
            'average_fitness_in_run': np.mean(sorted_fitness),
            'pop_size': self.pop_size
        }

        generation_name = 'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params',
            generation_result_dict,
            comment="These are the parameters that correspond to the algorithm. "
                    "Look at the source code for `EvolutionStrategiesOptimizer::post_process()` "
                    "for comments documenting these parameters"
        )

        if fitness_shaping_enabled:
            sorted_utilities = []
            n_individuals = len(sorted_fitness)
            for i in range(n_individuals):
                u = max(0., np.log((n_individuals / 2) + 1) - np.log(i + 1))
                sorted_utilities.append(u)
            sorted_utilities = np.array(sorted_utilities)
            sorted_utilities /= np.sum(sorted_utilities)
            sorted_utilities -= (1. / n_individuals)
            # assert np.sum(sorted_utilities) == 0., "Sum of utilities is not 0, but %.4f" % np.sum(sorted_utilities)
            fitnesses_to_fit = sorted_utilities
        else:
            fitnesses_to_fit = sorted_fitness

        assert len(fitnesses_to_fit) == len(sorted_perturbations)

        self.current_individual_arr += learning_rate \
                                       * np.sum([f * e for f, e in zip(fitnesses_to_fit, sorted_perturbations)], axis=0) \
                                       / (len(fitnesses_to_fit) * np.asarray(noise_std) ** 2)

        # **************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        # **************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run
        self.eval_pop.clear()

        # check if to stop
        if self.g < n_iteration - 1 and self.best_fitness_in_run < stop_criterion:
            self.current_perturbations = self._get_perturbations(traj)
            current_eval_pop_arr = (self.current_individual_arr + self.current_perturbations).tolist()

            self.eval_pop[:] = [list_to_dict(ind, self.optimizee_individual_dict_spec) \
                                for ind in current_eval_pop_arr]
            self.eval_pop.append(
                list_to_dict(self.current_individual_arr, self.optimizee_individual_dict_spec))

            # Bounding function has to be applied AFTER the individual has been converted to a dict
            if self.optimizee_bounding_func is not None:
                self.eval_pop[:] = [self.optimizee_bounding_func(ind) for ind in self.eval_pop]

            self.eval_pop_arr[:] = np.asarray([dict_to_list(ind) for ind in self.eval_pop])

            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.end`
        """
        best_last_indiv_dict = list_to_dict(self.best_individual_in_run.tolist(),
                                            self.optimizee_individual_dict_spec)

        traj.f_add_result('final_individual', best_last_indiv_dict)
        traj.f_add_result('final_fitness', self.best_fitness_in_run)
        traj.f_add_result('n_iteration', self.g + 1)

        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) ES optimization --")
