
import logging
import random
from collections import namedtuple

import numpy as np
from enum import Enum

from l2l.optimizers.optimizer import Optimizer
from l2l import dict_to_list
from l2l import list_to_dict

logger = logging.getLogger("optimizers.paralleltempering")

ParallelTemperingParameters = namedtuple('ParallelTemperingParameters',
                                          ['n_parallel_runs', 'noisy_step', 'n_iteration', 'stop_criterion', 'seed', 'cooling_schedules', 'temperature_bounds', 'decay_parameters'])
ParallelTemperingParameters.__doc__ = """
:param n_parallel_runs: Number of parallel Simulated Annealing runs
:param noisy_step: Size of the random step
:param decay_parameters: List of decay parameter for each cooling schedule
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param seed: Random seed
:param cooling_schedules: List with cooling schedules to use.
:param temperature_bounds: List of upper and lower bound of the temperature for
    each schedule. The first entry is the upper bound (starting temperature) 
    and the second entry is the ending temperature
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


class ParallelTemperingOptimizer(Optimizer):
    """
    Class for a parallel tempering solver.
    
    Parallel Tempering is a search algorithm, that uses multiple simulated 
    annealing algorithms at the same time and has a certain chance of two 
    annealing algorithms switching temperatures. Each of the annealing 
    algorithms can have different cooling schedules and respective decay 
    parameters or staring/ ending temperatures. This effectively has a similar 
    functional effect, as a single simulated annealing with multiple coolings 
    and reheatings, but needs fewer parameters (like when to reheat and how 
    often). For details on simulated annealing, please read the documentation 
    on it. 
    
    Note: For simplicity sake, not the positions, but the temperature and
    the schedule are swapped, which ammounts to the exact same. The 
    temperature and the schedules are each stored in lists, which are both 
    indexed by 'compare_indices'. If the swap criterion between two schedules
    are met, the respective entries for 'compare_indices' are swapped.
    To get the parallel runs, 'n_parallel_runs" is used - each individual 
    is one of the parallel runs..
    
    The algorithm does:

    For n iterations and each cooling schedule do:
        - Take a step of size noisy step in a random direction
        - If it reduces the cost, keep the solution
        - Otherwise keep with probability exp(- (f_new - f) / T)
        - Swap positions between two randomly chosen schedules 
          with probability exp(-((f_1 - f_2) * (1 / (k * T_1) - 1 / (k * T_2)))) with k being a constant
        
    NOTE: This expects all parameters of the system to be of floating point

    :param  ~l2l.utils.trajectory.Trajectory traj:
      Use this trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`
    
    :param optimizee_create_individual:
      Function that creates a new individual
    
    :param optimizee_fitness_weights: 
      Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each
      element of the fitness vector)
    
    :param parameters: 
      Instance of :func:`~collections.namedtuple` :class:`SimulatedAnnealingParameters` containing the
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
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')
        
        cooling_schedules_string = ''
        bounds_list = []
        decay_list = []
        schedules_list = []
        for i in range(0,traj.n_parallel_runs):
            bounds_list.append(str(parameters.temperature_bounds[i,:]))
            bounds_list.append(' ')
            decay_list.append(str(parameters.decay_parameters[i]))
            decay_list.append(' ')
            schedules_list.append(str(parameters.cooling_schedules[i]))
            schedules_list.append(' ')
        temperature_bounds_string = ''.join(bounds_list)
        decay_parameters_string = ''.join(decay_list)
        cooling_schedules_string = ''.join(schedules_list)
        
        traj.f_add_parameter('temperature_bounds', temperature_bounds_string,
                             comment='The max and min temperature of the respective schedule')
        traj.f_add_parameter('decay_parameters', decay_parameters_string,
                             comment='The one parameter, most schedules need')
        traj.f_add_parameter('cooling_schedules', cooling_schedules_string,
                             comment='The used cooling schedule')

        _, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the simulated annealing algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        self.current_individual_list = [np.array(dict_to_list(self.optimizee_create_individual()))
                                        for _ in range(parameters.n_parallel_runs)]

        traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')

        self.T_all = parameters.temperature_bounds[:,0]  # Initialize temperature
        self.T = 1
        self.g = 0  # the current generation
        self.cooling_schedules = parameters.cooling_schedules
        self.decay_parameters = parameters.decay_parameters
        self.temperature_bounds = parameters.temperature_bounds
        # Keep track of current fitness value to decide whether we want the next individual to be accepted or not
        self.current_fitness_value_list = [-np.Inf] * parameters.n_parallel_runs

        new_individual_list = [
            list_to_dict(ind_as_list + np.random.normal(0.0, parameters.noisy_step, ind_as_list.size) * traj.noisy_step,
                         self.optimizee_individual_dict_spec)
            for ind_as_list in self.current_individual_list
        ]
        if optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]

        self.eval_pop = new_individual_list
        self._expand_trajectory(traj)
        
        #initialize container for the indices of the parallel runs
        self.parallel_indices = []
        for i in range(0,traj.n_parallel_runs):
            self.parallel_indices.append(i)
        
        self.available_cooling_schedules = AvailableCoolingSchedules
        
        # assert if all cooling schedules are among the known cooling schedules
        schedule_known = True  # start off as True - if any schdule is unknown gets False
        for i in range(np.size(self.cooling_schedules)):
            schedule_known = schedule_known and self.cooling_schedules[i] in AvailableCoolingSchedules
        
        assert schedule_known, print("Warning: Unknown cooling schedule")
        
    def cooling(self,temperature, cooling_schedule, decay_parameter, temperature_bounds, steps_total):        
        
        T0, temperature_end = temperature_bounds
        
        k = self.g + 1        
        if cooling_schedule == AvailableCoolingSchedules.DEFAULT:
            return temperature * decay_parameter
      
        # Simulated Annealing and Boltzmann Machines: 
        # A Stochastic Approach to Combinatorial Optimization and Neural Computing (1989)
        elif cooling_schedule == AvailableCoolingSchedules.LOGARITHMIC:
            return T0 / (1 + np.log(1 + k))
            
        # Kirkpatrick, Gelatt and Vecchi (1983)
        elif cooling_schedule == AvailableCoolingSchedules.EXPONENTIAL:
            return T0 * (decay_parameter ** (k))
        elif cooling_schedule == AvailableCoolingSchedules.LINEAR_MULTIPLICATIVE:
            return T0 / (1 + decay_parameter * k)
        elif cooling_schedule == AvailableCoolingSchedules.QUADRATIC_MULTIPLICATIVE:
            return T0 / (1 + decay_parameter * np.square(k))
            
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

    # get tthe transistion probability between two simulated annealing systems with
    # tempereatures T and energies E
    def metropolis_hasting(self,E1,E2,T1,T2):
        # k = 1.387 * (10 ** -23)  # boltzmann konstant
        # Note: do not use real Blotzmann kosntant, because both energies and temperatures are divorced from any real physical representation
        k = 5
        p = np.exp(-np.abs((E1 - E2) * (1 / (k * T1) - 1 / (k * T2))))
        if p < 1:
            return p
            
        return 1
        
    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`
        """
        noisy_step, n_iteration, stop_criterion = \
            traj.noisy_step, traj.n_iteration, traj.stop_criterion
        cooling_schedules = self.cooling_schedules
        decay_parameters = self.decay_parameters
        temperature_bounds = self.temperature_bounds
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()
        temperature = self.T_all
        for i in range(0,traj.n_parallel_runs):
            self.T_all[self.parallel_indices[i]] = self.cooling(temperature[self.parallel_indices[i]], cooling_schedules[self.parallel_indices[i]], decay_parameters[self.parallel_indices[i]], temperature_bounds[self.parallel_indices[i],:], n_iteration)
        logger.info("  Evaluating %i individuals" % len(fitnesses_results))
  
        assert len(fitnesses_results) == traj.n_parallel_runs
        weighted_fitness_list = []
        for i, (run_index, fitness) in enumerate(fitnesses_results):
            
            self.T = self.T_all[self.parallel_indices[i]]

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
            if r < p or weighted_fitness >= current_fitness_value_i:
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
            
        # the parallel tempering swapping starts here
        for i in range(0,traj.n_parallel_runs):
            
            #make a random choice from all the other parallel runs
            compare_indices = []
            for j in range(0,traj.n_parallel_runs):
                compare_indices.append(i)
            compare_indices.remove(i)
            random_choice = random.choice(compare_indices)
            
            #random variable with unit distribution betwwen 0 and 1
            random_variable = np.random.rand() 
            
            #swap if criterion is met
            if (self.metropolis_hasting(self.current_fitness_value_list[i], self.current_fitness_value_list[random_choice], self.T_all[i], self.T_all[random_choice]) > random_variable):
                temp = self.parallel_indices[i]
                self.parallel_indices[i] = self.parallel_indices[random_choice]
                self.parallel_indices[random_choice] = temp
                
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
        logger.info("-- End of (successful) parallel tempering --")
