import logging
from collections import namedtuple

import numpy as np

from ltl.optimizers.optimizer import Optimizer

logger = logging.getLogger("ltl-sa")

SimulatedAnnealingParameters = namedtuple('SimulatedAnnealingParameters',
                                          ['noisy_step', 'temp_decay', 'n_iteration', 'stop_criterion', 'bound',
                                           'seed'])
SimulatedAnnealingParameters.__doc__ = """
:param noisy_step: Size of the random step
:param temp_decay: A function of the form f(t) = temperature at time t
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param bound: bound of the function
:param seed: Random seed
"""


class SimulatedAnnealingOptimizer(Optimizer):
    """
    Class for a generic simulate annealing solver.
    In the pseudo code the algorithm does:

    For n iterations do:
        - Take a step of size noisy step in a random direction
        - If it reduces the cost, keep the solution
        - Otherwise keep with probability exp(- (f_new - f) / T)

    :param  ~pypet.trajectory.Trajectory traj: Use this pypet trajectory to store the parameters of the specific runs. The parameters should be initialized based on the values in `parameters`
    :param optimizee_create_individual: Function that creates a new individual
    :param optimizee_fitness_weights: Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each element of the fitness vector)
    :param parameters: Instance of :func:`~collections.namedtuple` :class:`SimulatedAnnealingParameters` containing the parameters needed by the Optimizer
    """
    def __init__(self, traj, optimizee_create_individual, optimizee_fitness_weights, parameters):
        # The following parameters are recorded
        super().__init__(traj, optimizee_create_individual, optimizee_fitness_weights, parameters)
        traj.f_add_parameter('noisy_step', parameters.noisy_step, comment='Size of the random step')
        traj.f_add_parameter('temp_decay', parameters.temp_decay,
                             comment='A temperature decay parameter (multiplicative)')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('bound', parameters.bound,
                             comment='Bound [x_min,x_max] of the solution space assuming that all coordinates must '
                                     'stay within [x_min,x_max]')
        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')

        self.current_individual = np.array(optimizee_create_individual())
        # Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter('generation', 0, comment='Current generation')
        traj.f_add_parameter('ind_idx', 0, comment='Index of individual')
        traj.f_add_derived_parameter('individual', self.current_individual, 'An individual of the population')

        traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')

        # The following parameters are NOT recorded
        self.optimizee_fitness_weights = optimizee_fitness_weights
        self.T = 1.  # Initialize temperature
        self.g = 0  # the current generation

        # Keep track of current fitness value to decide whether we want the next individual to be accepted or not
        self.current_fitness_value = -np.Inf

        new_individual = np.clip(
            self.current_individual + np.random.randn(self.current_individual.size) * parameters.noisy_step,
            parameters.bound[0], parameters.bound[1])
        self.eval_pop = [new_individual]
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        noisy_step, temp_decay, n_iteration, stop_criterion, bound = \
            traj.noisy_step, traj.temp_decay, traj.n_iteration, traj.stop_criterion, traj.bound
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()
        self.T *= temp_decay

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))
        # NOTE: Currently works with only one individual at a time.
        # In principle, can be used with many different individuals evaluated in parallel
        assert len(fitnesses_results) == 1
        while fitnesses_results:
            result = fitnesses_results.pop()
            # Update fitnesses
            # NOTE: The fitness here is a tuple! For now, we'll only support fitnesses with one element
            run_index, fitness = result  # The environment returns tuples: [(run_idx, run), ...]
            weighted_fitness = tuple(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
            assert len(weighted_fitness) == 1
            weighted_fitness = weighted_fitness[0]

            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx
            individual = old_eval_pop[ind_index]

            # Accept or reject the new solution
            r = np.random.rand()
            p = np.exp(-(weighted_fitness - self.current_fitness_value) / self.T)

            traj.f_add_result('$set.$.individual', list(individual))
            # Watchout! if weighted fitness is a tuple/np array it should be converted to a list first here
            traj.f_add_result('$set.$.fitness', weighted_fitness)

            if r < p or weighted_fitness >= self.current_fitness_value:
                self.current_fitness_value = weighted_fitness
                self.current_individual = individual

            new_individual = np.clip(
                self.current_individual + np.random.randn(self.current_individual.size) * noisy_step,
                bound[0], bound[1])
            logger.debug("Current best fitness is %.2f. New individual is %s", self.current_fitness_value,
                         new_individual)
            self.eval_pop.append(new_individual)
        traj.v_idx = -1  # set the trajectory back to default

        logger.info("-- End of generation {} --".format(self.g))

        # ------- Create the next generation by crossover and mutation -------- #
        # not necessary for the last generation
        if self.g < n_iteration - 1 and stop_criterion < self.current_fitness_value:
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        # TODO: Maype print some individuals here?
        logger.info("-- End of (successful) annealing --")
