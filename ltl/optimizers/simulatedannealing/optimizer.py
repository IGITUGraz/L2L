
import logging
from collections import namedtuple

import numpy as np

from ltl.optimizers.optimizer import Optimizer
from ltl import dict_to_list
from ltl import list_to_dict
logger = logging.getLogger("ltl-sa")

SimulatedAnnealingParameters = namedtuple('SimulatedAnnealingParameters',
                                          ['n_parallel_runs', 'noisy_step', 'temp_decay', 'n_iteration', 'stop_criterion', 'seed'])
SimulatedAnnealingParameters.__doc__ = """
:param n_parallel_runs: Number of individuals per simulation / Number of parallel Simulated Annealing runs
:param noisy_step: Size of the random step
:param temp_decay: A function of the form f(t) = temperature at time t
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
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
        traj.f_add_parameter('n_parallel_runs', parameters.n_parallel_runs,
                             comment='Number of parallel simulated annealing runs / Size of Population')
        traj.f_add_parameter('noisy_step', parameters.noisy_step, comment='Size of the random step')
        traj.f_add_parameter('temp_decay', parameters.temp_decay,
                             comment='A temperature decay parameter (multiplicative)')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')

        _, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the simulated annealing algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        self.current_individual_list = [np.array(dict_to_list(self.optimizee_create_individual()))
                                        for _ in range(parameters.n_parallel_runs)]


        # The following parameters are NOT recorded
        self.T = 1.  # Initialize temperature
        self.g = 0  # the current generation

        # Keep track of current fitness value to decide whether we want the next individual to be accepted or not
        self.current_fitness_value_list = [-np.Inf] * parameters.n_parallel_runs

        new_individual_list = [
            list_to_dict(ind_as_list + np.random.normal(0.0, parameters.noisy_step, ind_as_list.size) * traj.noisy_step * self.T,
                         self.optimizee_individual_dict_spec)
            for ind_as_list in self.current_individual_list
        ]
        if optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]

        self.eval_pop = new_individual_list
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        noisy_step, temp_decay, n_iteration, stop_criterion = \
            traj.noisy_step, traj.temp_decay, traj.n_iteration, traj.stop_criterion
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()
        self.T *= temp_decay

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))
        # NOTE: Currently works with only one individual at a time.
        # In principle, can be used with many different individuals evaluated in parallel
        assert len(fitnesses_results) == traj.n_parallel_runs
        weighted_fitness_list = []
        for i, (run_index, fitness) in enumerate(fitnesses_results):
            
            # Update fitnesses
            # NOTE: The fitness here is a tuple! For now, we'll only support fitnesses with one element
            weighted_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
            weighted_fitness_list.append(weighted_fitness)

            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx
            individual = old_eval_pop[ind_index]

            # Accept or reject the new solution
            current_fitness_value_i = self.current_fitness_value_list[i]
            r = np.random.rand()
            p = np.exp((weighted_fitness - current_fitness_value_i) / self.T)

            # Accept
            accepted_flag = False
            if r < p or weighted_fitness >= current_fitness_value_i:
                accepted_flag = True
                self.current_fitness_value_list[i] = weighted_fitness
                self.current_individual_list[i] = np.array(dict_to_list(individual))

            traj.f_add_result('$set.$.individual', individual)
            # Watchout! if weighted fitness is a tuple/np array it should be converted to a list first here
            traj.f_add_result('$set.$.fitness', weighted_fitness)

            current_individual = self.current_individual_list[i]
            new_individual = list_to_dict(current_individual + np.random.randn(current_individual.size) * noisy_step * self.T,
                                          self.optimizee_individual_dict_spec)
            if self.optimizee_bounding_func is not None:
                new_individual = self.optimizee_bounding_func(new_individual)

            logger.debug("Current best fitness for individual %d is %.2f. New individual is %s", 
                         i, self.current_fitness_value_list[i], new_individual)
            self.eval_pop.append(new_individual)

        logger.debug("Current best fitness within population is %.2f", max(self.current_fitness_value_list))

        traj.v_idx = -1  # set the trajectory back to default
        logger.info("-- End of generation {} --".format(self.g))

        # ------- Create the next generation by crossover and mutation -------- #
        # not necessary for the last generation
        if self.g < n_iteration - 1 and stop_criterion > max(self.current_fitness_value_list):
            fitnesses_results.clear()
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        best_last_indiv_index = np.argmax(self.current_fitness_value_list)
        best_last_indiv = self.current_individual_list[best_last_indiv_index]
        best_last_fitness = self.current_fitness_value_list[best_last_indiv_index]

        traj.f_add_result('final_individual', self.current_fitness_value_list[-1])
        traj.f_add_result('final_fitness', self.current_fitness_value_list[-1])
        traj.f_add_result('n_iteration', self.g + 1)

        logger.info("The best last individual was %s with fitness %s", best_last_indiv, best_last_fitness)
        logger.info("-- End of (successful) annealing --")
