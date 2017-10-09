import logging
from collections import namedtuple

import numpy as np
import randomstate.prng.xoroshiro128plus as rnd

from ltl import dict_to_list, list_to_dict
from ltl.optimizers.optimizer import Optimizer

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
      Instance of :func:`~collections.namedtuple` :class:`.CrossEntropyParameters` containing the
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

        super().__init__(
            traj,
            optimizee_create_individual=optimizee_create_individual,
            optimizee_fitness_weights=optimizee_fitness_weights,
            parameters=parameters,
            optimizee_bounding_func=optimizee_bounding_func)

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

        self.dimension = np.array(self.current_individual_arr).shape

        traj.f_add_derived_parameter(
            'dimension',
            self.dimension,
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

        self.eval_pop = [list_to_dict(self.current_individual_arr, self.optimizee_individual_dict_spec) for _ in
                         range(len(self.current_individual_arr))]

        self._expand_trajectory(traj)

    @staticmethod
    def _get_all_agents_perturbations(current_generation, parameter_size, es_parameters):
        population_size, initial_random_seed, noise_std, mirrored_sampling_enabled = \
            es_parameters.pop_size, es_parameters.seed, es_parameters.noise_std, es_parameters.mirrored_sampling_enabled
        # In this case we don't care about indices
        r = rnd.RandomState(seed=(initial_random_seed + current_generation))
        # r.advance(current_generation * population_size * parameter_size)
        all_perturbations = noise_std * r.standard_normal(size=(population_size, parameter_size))

        if mirrored_sampling_enabled:
            return np.vstack((all_perturbations, -all_perturbations))

    @staticmethod
    def _get_single_individual_perturbation(individual_idx, current_generation, parameter_size,
                                            es_parameters):
        all_perturbations = EvolutionStrategiesOptimizer._get_all_agents_perturbations(current_generation,
                                                                                       parameter_size,
                                                                                       es_parameters)
        return all_perturbations[individual_idx, :]

    @staticmethod
    def update_current_individual(current_individual, weighted_fitness_list, previous_generation, parameter_size,
                                  es_parameters):
        if previous_generation == -1:
            assert sum(weighted_fitness_list) == 0
            return current_individual

        learning_rate, noise_std, fitness_shaping_enabled = \
            es_parameters.learning_rate, es_parameters.noise_std, es_parameters.fitness_shaping_enabled

        # NOTE: We need to get all perturbations for the previous generation. But this is handled in the optimizee!
        all_perturbations = EvolutionStrategiesOptimizer._get_all_agents_perturbations(previous_generation,
                                                                                       parameter_size,
                                                                                       es_parameters)

        # NOTE: It is important that both the fitnesses and the perturbations are sorted by the run indices
        #^ Here sorting is by fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))
        sorted_fitness = np.asarray(weighted_fitness_list)[fitness_sorting_indices]
        sorted_perturbations = all_perturbations[fitness_sorting_indices]

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

        perturbations_to_fit = sorted_perturbations

        assert len(fitnesses_to_fit) == len(sorted_perturbations)

        current_individual += learning_rate \
                              * np.sum([f * e for f, e in zip(fitnesses_to_fit, perturbations_to_fit)], axis=0) \
                              / (len(fitnesses_to_fit) * noise_std ** 2)
        return current_individual

    @staticmethod
    def get_new_individual(individual_idx, current_individual, current_generation, parameter_size, es_parameters):
        perturbation = EvolutionStrategiesOptimizer._get_single_individual_perturbation(individual_idx,
                                                                                        current_generation,
                                                                                        parameter_size, es_parameters)
        new_individual = current_individual + perturbation
        return new_individual

    def get_params(self):
        """
        Get parameters used for recorder
        :return: Dictionary containing recorder parameters
        """

        param_dict = self.parameters._asdict()
        return param_dict

    # @profile
    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """

        n_iteration, stop_criterion, learning_rate, noise_std, fitness_shaping_enabled = \
            traj.n_iteration, traj.stop_criterion, traj.learning_rate, traj.noise_std, traj.fitness_shaping_enabled

        #**************************************************************************************************************
        # Storing run-information in the trajectory
        # Reading fitnesses and performing distribution update
        #**************************************************************************************************************
        fitness_idx_pairs = []
        for run_index, fitness in fitnesses_results:
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx

            traj.f_add_result('$set.$.individual', self.eval_pop[ind_index])
            traj.f_add_result('$set.$.fitness', fitness)

            fitness_idx_pairs.append((ind_index, fitness))

        traj.v_idx = -1  # set trajectory back to default

        # Sort fitnesses by individual index. This is very important
        fitness_idx_pairs_sorted = sorted(fitness_idx_pairs, key=lambda x: x[0])
        weighted_fitness_list = [np.dot(f[1], self.optimizee_fitness_weights) for f in fitness_idx_pairs_sorted]

        weighted_fitness_list = np.array(weighted_fitness_list).ravel()
        # NOTE: It is necessary to clear the finesses_results to clear the data in the reference, and del
        #^ is used to make sure it's not used in the rest of this function
        fitnesses_results.clear()
        del fitnesses_results

        # Last fitness is for the previous `current_individual_arr`
        # weighted_fitness_list = weighted_fitness_list[:-1]
        # current_individual_fitness = weighted_fitness_list[-1]

        # Performs descending arg-sort of weighted fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))
        # Sorting the data according to fitness
        sorted_fitness = np.asarray(weighted_fitness_list)[fitness_sorting_indices]

        self.best_fitness_in_run = sorted_fitness[0]

        logger.info("-- End of generation %d --", self.g)
        logger.info("  Evaluated %d individuals", len(weighted_fitness_list) + 1)
        logger.info('  Best Fitness: %.4f', self.best_fitness_in_run)
        logger.info('  Average Fitness: %.4f', np.mean(sorted_fitness))

        #**************************************************************************************************************
        # Storing Generation Parameters / Results in the trajectory
        #**************************************************************************************************************
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

        #**************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        #**************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run
        self.eval_pop.clear()

        # check if to stop
        if self.g < n_iteration - 1 and self.best_fitness_in_run < stop_criterion:
            self.eval_pop = [list_to_dict(weighted_fitness_list, self.optimizee_individual_dict_spec) for _ in
                             range(len(self.current_individual_arr))]

            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """

        traj.f_add_result('final_fitness', self.best_fitness_in_run)
        traj.f_add_result('n_iteration', self.g + 1)

        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) ES optimization --")
