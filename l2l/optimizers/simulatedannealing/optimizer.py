import logging
from collections import namedtuple

import numpy as np
from enum import Enum

from l2l import dict_to_list
from l2l import list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.simulatedannealing")

SimulatedAnnealingParameters = namedtuple('SimulatedAnnealingParameters',
                                          ['n_parallel_runs', 'noisy_step', 'temp_decay', 'n_iteration', 'stop_criterion', 'seed', 'cooling_schedule'])
SimulatedAnnealingParameters.__doc__ = """
:param n_parallel_runs: Number of individuals per simulation / Number of parallel Simulated Annealing runs
:param noisy_step: Size of the random step
:param temp_decay: A function of the form f(t) = temperature at time t
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param seed: Random seed
:param cooling_schedule: Which of the available schedules to use

"""

AvailableCoolingSchedules = Enum('Schedule', 'DEFAULT LOGARITHMIC EXPONENTIAL LINEAR_MULTIPLICATIVE QUADRATIC_MULTIPLICATIVE LINEAR_ADDAPTIVE QUADRATIC_ADDAPTIVE EXPONENTIAL_ADDAPTIVE TRIGONOMETRIC_ADDAPTIVE')

"""
Multiplicative Monotonic Cooling
This schedule type multiplies the starting temperature by a factor that 
decreases over time (number k of the performed iteration steps). It requires a 
decay parameter (alpha) but not an ending temperature, as the prgression of the 
temperature is well definded by the decay parameter only. The Multiplicative 
Monotonic Cooling schedules are: Exponential multiplicative cooling, 
Logarithmical multiplicative cooling, Linear multiplicative cooling and 
Quadratic multiplicative cooling.
Source: Kirkpatrick, Gelatt and Vecchi (1983)

- Exponential multiplicative cooling
Default cooling schedule for typical applications of simulated annealing. Each 
step, the temperature T_k is multiplied by the factor alpha (which has to be 
between 0 and 1) or in other words it is the starting temperature T_0 
multiplied by the factor alpha by the power of k: T_k = T_0 * alpha^k

- Logarithmical multiplicative cooling
The factor by which the temperature decreases, is indirectly proportional to 
the log of k.  Therefore it slows down the cooling, the further progressed 
the schedule is. Alpha has to be largert than one. 
T_k = T_0 / ( 1 + alpha* log (1 + k) )

- Linear multiplicative cooling
Behaves similar to Logarithmical multiplicative cooling in that the decrease 
gets lower over time, but not as pronounced. The decrease is indirectly 
proportional to alpha times k and alpha has to be larger than zero:
T_k = T_0 / ( 1 + alpha*k)

- Quadratic multiplicative cooling 
This schedule stays at high temperatures longer, than the other schedules and 
has a steeper cooling later in the process. Alpha has to be larger than zero.
T_k = T_0 / ( 1 + alpha*k^2)

Additive Monotonic Cooling
The differences to Multiplicative Monotonic Cooling are, that the final 
temperature T_n and the number of iterations n are needed also. So this 
cannot be used as intended, if the stop criterion is something different, 
than a certain number of iteration steps. A decay parameter is not needed. 
Each temperature is computed, by adding a term to the final temperature. The 
Additive Monotonic Cooling schedules are: Linear additive cooling, Quadratic 
additive cooling, Exponential additive cooling and Trigonometric additive 
cooling.
Source. Additive monotonic cooling B. T. Luke (2005) 

- Linear additive cooling 
This schedule adds a term to the final temperature, which decreases linearily 
with the progression of the schedule.
T_k = T_n + (T_0 -T_n)*((n-k)/n)

- Quadratic additive cooling 
This schedule adds a term to the final temperature, which decreases q
uadratically with the progression of the schedule.
T_k = T_n + (T_0 -T_n)*((n-k)/n)^2

- Exponential additive
Uses a complicated formula, to come up with a schedule, that has a slow start, 
a steep decrease in temperature in the middle and a slow decrease at the end 
of the process.
T_k = T_n + (T_0 - T_n) * (1/(1+exp( 2*ln(T_0 - T_n)/n * (k- n/2) ) ) )

- Trigonometric additive cooling
This schedule has a similar behavior as Exponential additive, but less pronounced. 
T_k = T_n + (T_0 - T_n)/2 * (1+cos(k*pi/n))

"""


class SimulatedAnnealingOptimizer(Optimizer):
    """
    Class for a generic simulate annealing solver.
    In the pseudo code the algorithm does:

    For n iterations do:
        1. Take a step of size noisy step in a random direction
        2. If it reduces the cost, keep the solution
        3. Otherwise keep with probability exp(- (f_new - f) / T)

    NOTE: This expects all parameters of the system to be of floating point

    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`
    :param optimizee_create_individual: Function that creates a new individual
    :param optimizee_fitness_weights: Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each
      element of the fitness vector)
    :param parameters: Instance of :func:`~collections.namedtuple` :class:`.SimulatedAnnealingParameters` containing the
      parameters needed by the Optimizer
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters, optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func

        # The following parameters are recorded
        traj.f_add_parameter('n_parallel_runs', parameters.n_parallel_runs,
                             comment='Number of parallel simulated annealing runs / Size of Population')
        traj.f_add_parameter('noisy_step', parameters.noisy_step, comment='Size of the random step')
        traj.f_add_parameter('temp_decay', parameters.temp_decay,
                             comment='A temperature decay parameter (multiplicative)')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('seed', np.uint32(parameters.seed), comment='Seed for RNG')

        _, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the simulated annealing algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        self.current_individual_list = [np.array(dict_to_list(self.optimizee_create_individual()))
                                        for _ in range(parameters.n_parallel_runs)]
        self.random_state = np.random.RandomState(parameters.seed)

        # The following parameters are NOT recorded
        self.T = 1.  # Initialize temperature
        self.g = 0  # the current generation

        # Keep track of current fitness value to decide whether we want the next individual to be accepted or not
        self.current_fitness_value_list = [-np.Inf] * parameters.n_parallel_runs

        new_individual_list = [
            list_to_dict(
                ind_as_list + self.random_state.normal(0.0, parameters.noisy_step, ind_as_list.size) * traj.noisy_step * self.T,
                self.optimizee_individual_dict_spec)
            for ind_as_list in self.current_individual_list
        ]
        if optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]

        self.eval_pop = new_individual_list
        self._expand_trajectory(traj)
        
        self.cooling_schedule = parameters.cooling_schedule

    def cooling(self,temperature, cooling_schedule, temperature_decay, temperature_end, steps_total):        
        # assumes, that the temperature always starts at 1
        T0 = 1
        k = self.g + 1    
      
        if cooling_schedule == AvailableCoolingSchedules.DEFAULT:
            return temperature * temperature_decay
          
        # Simulated Annealing and Boltzmann Machines: 
        # A Stochastic Approach to Combinatorial Optimization and Neural Computing (1989)
        elif cooling_schedule == AvailableCoolingSchedules.LOGARITHMIC:
            return T0 / (1 + np.log(1 + k))
            
        # Kirkpatrick, Gelatt and Vecchi (1983)
        elif cooling_schedule == AvailableCoolingSchedules.EXPONENTIAL:
            alpha = 0.85 
            return T0 * (alpha ** (k))
        elif cooling_schedule == AvailableCoolingSchedules.LINEAR_MULTIPLICATIVE:
            alpha = 1
            return T0 / (1 + alpha * k)
        elif cooling_schedule == AvailableCoolingSchedules.QUADRATIC_MULTIPLICATIVE:
            alpha = 1
            return T0 / (1 + alpha * np.square(k))
            
        # Additive monotonic cooling B. T. Luke (2005) 
        elif cooling_schedule == AvailableCoolingSchedules.LINEAR_ADDAPTIVE:
            return temperature_end + (T0 - temperature) * ((steps_total - k) / steps_total)
        elif cooling_schedule == AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE:
            return temperature_end + (T0 - temperature) * np.square((steps_total - k) / steps_total)
        elif cooling_schedule == AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE:            
            return temperature_end + (T0 - temperature) * (1 / (1 + np.exp((2 * np.log(T0 - temperature_end) / steps_total) * (k - steps_total / 2))))
        elif cooling_schedule == AvailableCoolingSchedules.TRIGONOMETRIC_ADDAPTIVE:
            return temperature_end + (T0 - temperature_end) * (1 + np.cos(k * 3.1415 / steps_total)) / 2

        return -1

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`
        """
        noisy_step, temp_decay, n_iteration, stop_criterion = \
            traj.noisy_step, traj.temp_decay, traj.n_iteration, traj.stop_criterion
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()
        temperature = self.T
        temperature_end = 0
        self.T = self.cooling(temperature, self.cooling_schedule, temp_decay, temperature_end, n_iteration)
        logger.info("  Evaluating %i individuals" % len(fitnesses_results))

        assert len(fitnesses_results) == traj.n_parallel_runs
        weighted_fitness_list = []
        for i, (run_index, fitness) in enumerate(fitnesses_results):

            weighted_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
            weighted_fitness_list.append(weighted_fitness)

            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx
            individual = old_eval_pop[ind_index]

            # Accept or reject the new solution
            current_fitness_value_i = self.current_fitness_value_list[i]
            r = self.random_state.rand()
            p = np.exp((weighted_fitness - current_fitness_value_i) / self.T)
            
            # Accept
            if r < p or weighted_fitness >= current_fitness_value_i:
                self.current_fitness_value_list[i] = weighted_fitness
                self.current_individual_list[i] = np.array(dict_to_list(individual))

            traj.f_add_result('$set.$.individual', individual)
            # Watchout! if weighted fitness is a tuple/np array it should be converted to a list first here
            traj.f_add_result('$set.$.fitness', weighted_fitness)

            current_individual = self.current_individual_list[i]
            new_individual = list_to_dict(
                current_individual + self.random_state.randn(current_individual.size) * noisy_step * self.T,
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
        See :meth:`~l2l.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        best_last_indiv_index = np.argmax(self.current_fitness_value_list)
        best_last_indiv = self.current_individual_list[best_last_indiv_index]
        best_last_fitness = self.current_fitness_value_list[best_last_indiv_index]

        best_last_indiv_dict = list_to_dict(best_last_indiv.tolist(), self.optimizee_individual_dict_spec)
        traj.f_add_result('final_individual', best_last_indiv_dict)
        traj.f_add_result('final_fitness', best_last_fitness)
        traj.f_add_result('n_iteration', self.g + 1)

        logger.info("The best last individual was %s with fitness %s", best_last_indiv, best_last_fitness)
        logger.info("-- End of (successful) annealing --")
