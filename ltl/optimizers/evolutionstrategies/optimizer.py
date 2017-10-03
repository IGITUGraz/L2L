import logging
from collections import namedtuple

import numpy as np

from ltl import dict_to_list, list_to_dict
from ltl.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.evolutionstrategies")

EvolutionStrategiesParameters = namedtuple('EvolutionStrategiesParameters', [
    'learning_rate', 'noise_std', 'use_mirrored_sampling', 'pop_size', 'n_iteration', 'stop_criterion', 'seed'
])

EvolutionStrategiesParameters.__doc__ = """
:param learning_rate: Learning rate
:param noise_std: Standard deviation of the step size (The step has 0 mean)
:param use_mirrored_sampling: Should we turn on mirrored sampling i.e. sampling both e and -e

:param pop_size: Number of individuals per simulation.
:param n_iteration: Number of iterations to perform
:param stop_criterion: (Optional) Stop if this fitness is reached.
:param seed: The random seed used for generating new individuals
"""


class EvolutionStrategiesOptimizer(Optimizer):
    """
    Class Implementing the evolution strategies optimizer

    as in: Salimans, T., Ho, J., Chen, X. & Sutskever, I. Evolution Strategies as a Scalable Alternative to Reinforcement   Learning. arXiv:1703.03864 [cs, stat] (2017).

    In the pseudo code the algorithm does:

    For n iterations do:
      - Perturb the current individual with 0 mean and `noise_std` standard deviation
      - evaluate individuals and get fitness
      - Update the fitness as

            theta_{t+1} <- theta_t + alpha  * sum{F_i * e_i} / (n * sigma^2)

        where F_i is the fitness and e_i is the perturbation

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

        self.recorder_parameters = parameters
        self.optimizee_bounding_func = optimizee_bounding_func

        if parameters.pop_size < 1:
            raise Exception("pop_size needs to be greater than 0")

        # The following parameters are recorded
        traj.f_add_parameter('learning_rate', parameters.learning_rate, comment='Learning rate')
        traj.f_add_parameter('noise_std', parameters.noise_std, comment='Standard deviation of noise')
        traj.f_add_parameter(
            'use_mirrored_sampling', parameters.use_mirrored_sampling, comment='Flag to enable mirrored sampling')
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
        self.g = 0    # the current generation
        self.pop_size = parameters.pop_size    # Population size is dynamic in FACE
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

        # Bounding function has to be applied AFTER the individual has been converted to a dict
        if optimizee_bounding_func is not None:
            self.eval_pop = [self.optimizee_bounding_func(ind) for ind in self.eval_pop]

        self.eval_pop_arr = np.array([dict_to_list(ind) for ind in self.eval_pop])

        self._expand_trajectory(traj)

    def _get_perturbations(self, traj):
        pop_size, noise_std, use_mirrored_sampling = traj.pop_size, traj.noise_std, traj.use_mirrored_sampling
        perturbations = noise_std * self.random_state.randn(pop_size, *self.current_individual_arr.shape)
        if use_mirrored_sampling:
            return np.vstack((perturbations, -perturbations))
        return perturbations

    def get_params(self):
        """
        Get parameters used for recorder
        :return: Dictionary containing recorder parameters
        """

        param_dict = self.recorder_parameters._asdict()
        return param_dict

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """

        n_iteration, stop_criterion, learning_rate, noise_std = \
            traj.n_iteration, traj.stop_criterion, traj.learning_rate, traj.noise_std

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
        traj.v_idx = -1    # set trajectory back to default

        # Performs descending arg-sort of weighted fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))

        # Sorting the data according to fitness
        sorted_population = self.eval_pop_arr[fitness_sorting_indices]
        sorted_fitness = np.asarray(weighted_fitness_list)[fitness_sorting_indices]

        self.best_individual_in_run = sorted_population[0]
        self.best_fitness_in_run = sorted_fitness[0]

        logger.info("-- End of generation %d --", self.g)
        logger.info("  Evaluated %d individuals", len(fitnesses_results))
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
            comment="These are the parameters that correspond to the algorithm, look at the source code"
            " for `EvolutionStrategiesOptimizer::post_process()` for comments documenting these"
            " parameters")

        self.current_individual_arr += \
            learning_rate \
            * np.sum([f * e for f, e in zip(weighted_fitness_list, self.current_perturbations)], axis=0) \
            / (len(weighted_fitness_list) * noise_std ** 2)

        #**************************************************************************************************************
        # Create the next generation by sampling the inferred distribution
        #**************************************************************************************************************
        # Note that this is only done in case the evaluated run is not the last run
        fitnesses_results.clear()
        self.eval_pop.clear()

        # check if to stop
        if self.g < n_iteration - 1 and self.best_fitness_in_run < stop_criterion:
            self.current_perturbations = self._get_perturbations(traj)
            current_eval_pop_arr = (self.current_individual_arr + self.current_perturbations).tolist()

            self.eval_pop = [list_to_dict(ind, self.optimizee_individual_dict_spec) for ind in current_eval_pop_arr]

            # Bounding function has to be applied AFTER the individual has been converted to a dict
            if self.optimizee_bounding_func is not None:
                self.eval_pop = [self.optimizee_bounding_func(ind) for ind in self.eval_pop]

            self.eval_pop_arr = np.array([dict_to_list(ind) for ind in self.eval_pop])

            self.g += 1    # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        best_last_indiv_dict = list_to_dict(self.best_individual_in_run.tolist(), self.optimizee_individual_dict_spec)

        traj.f_add_result('final_individual', best_last_indiv_dict)
        traj.f_add_result('final_fitness', self.best_fitness_in_run)
        traj.f_add_result('n_iteration', self.g + 1)

        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) ES optimization --")
