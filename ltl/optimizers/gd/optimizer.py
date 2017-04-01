
import logging
from collections import namedtuple


import numpy as np

from ltl.optimizers.optimizer import Optimizer
from ltl import dict_to_list
from ltl import list_to_dict
logger = logging.getLogger("ltl-gd")

ClassicGDParameters = namedtuple('ClassicGDParameters',
    ['learning_rate', 'exploration_rate', 'n_random_steps', 'n_iteration', 'stop_criterion'])
ClassicGDParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_rate: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
"""

StochGDParameters = namedtuple('StochGDParameters',
    ['learning_rate', 'stoch_deviation', 'stoch_decay', 'exploration_rate', 'n_random_steps', 'n_iteration',
    'stop_criterion'])
StochGDParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param stoch_deviation: The standard deviation of the random vector used to perturbate the gradient
:param stoch_decay: The decay of the influence of the random vector that is added to the gradient 
    (set to 0 to disable stochastic perturbation)
:param exploration_rate: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
"""

AdamParameters = namedtuple('AdamParameters',
    ['learning_rate', 'exploration_rate', 'n_random_steps', 'first_order_decay', 'second_order_decay', 'n_iteration',
    'stop_criterion'])
AdamParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_rate: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param first_order_decay: Specifies the amount of decay of the historic first order momentum per gradient descent step
:param second_order_decay: Specifies the amount of decay of the historic second order momentum per gradient descent step
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value

"""

RMSPropParameters = namedtuple('RMSPropParameters',
    ['learning_rate', 'exploration_rate', 'n_random_steps', 'momentum_decay', 'n_iteration', 'stop_criterion'])
RMSPropParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_rate: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param momentum_decay: Specifies the decay of the historic momentum at each gradient descent step
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
"""

class GradientDescentOptimizer(Optimizer):
    """
    Class for a generic gradient descent solver.
    In the pseudo code the algorithm does:

    For n iterations do:
        - Explore the fitness of individuals in the close vicinity of the current one
        - Calculate the gradient based on these fitnesses.
        - Create the new 'current individual' by taking a step in the parameters space along the direction
            of the largest ascent of the plane

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
      Instance of :func:`~collections.namedtuple` :class:`ClassicGDParameters`,
      :func:`~collections.namedtuple` :class:`StochGDParameters`, 
      :func:`~collections.namedtuple` :class:`RMSPropParameters` or 
      :func:`~collections.namedtuple` :class:`AdamParameters` containing the
      parameters needed by the Optimizer. The type of this parameter is used to select one of the GD variants.
    
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
        
        traj.f_add_parameter('learning_rate', parameters.learning_rate, comment='Value of learning rate')
        traj.f_add_parameter('exploration_rate', parameters.exploration_rate, 
            comment='Standard deviation of the random steps')
        traj.f_add_parameter('n_random_steps', parameters.n_random_steps, 
            comment='Amount of random steps taken for calculating the gradient')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')

        _, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the gradient descent algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        self.current_individual = np.array(dict_to_list(self.optimizee_create_individual()))

         # Depending on the algorithm used, initialize the necessary variables
        self.updateFunction = None
        if type(parameters) is ClassicGDParameters:
            self.initClassicGD(parameters, traj)
        elif type(parameters) is StochGDParameters:
            self.initStochGD(parameters, traj)
        elif type(parameters) is AdamParameters:
            self.initAdam(parameters, traj)
        elif type(parameters) is RMSPropParameters:
            self.initRMSProp(parameters, traj)


        traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')

        # Explore the neighbourhood in the parameter space of current individual
        new_individual_list = [
            list_to_dict(self.current_individual + 
                np.random.normal(0.0, parameters.exploration_rate, self.current_individual.size),
                self.optimizee_individual_dict_spec)
            for i in range(parameters.n_random_steps)
        ]

        # Also add the current individual to determine it's fitness
        new_individual_list.append(list_to_dict(self.current_individual, self.optimizee_individual_dict_spec))
            
        if optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]

        # Storing the fitness of the current individual
        self.current_fitness = -np.Inf
        self.g = 0
        
        self.eval_pop = new_individual_list
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))
        
        assert len(fitnesses_results) - 1 == traj.n_random_steps

        # We need to collect the directions of the random steps along with the fitness evaluated there
        fitnesses = np.zeros((traj.n_random_steps))
        dx = np.zeros((traj.n_random_steps, len(self.current_individual)))

        for i, (run_index, fitness) in enumerate(fitnesses_results):
            ind_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))

            # The last element of the list is the evaluation of the individual obtained via gradient descent
            if i == len(fitnesses_results) - 1:
                self.current_fitness = ind_fitness
            else:
                fitnesses[i] = ind_fitness

                # We need to convert the current run index into an ind_idx
                # (index of individual within one generation
                traj.v_idx = run_index
                ind_index = traj.par.ind_idx
                individual = old_eval_pop[ind_index]
                
                dx[i] = np.array(dict_to_list(individual)) - self.current_individual
            
        logger.debug("Current fitness is %.2f", self.current_fitness) 

        traj.v_idx = -1  # set the trajectory back to default
        logger.info("-- End of iteration {}, current fitness is {} --".format(self.g, self.current_fitness))

        if self.g < traj.n_iteration - 1 and traj.stop_criterion > self.current_fitness:
            # Create new individual using the appropriate gradient descent
            self.updateFunction(traj, np.dot(np.linalg.pinv(dx), fitnesses - self.current_fitness))

        
            # Explore the neighbourhood in the parameter space of the current individual
            new_individual_list = [
                list_to_dict(self.current_individual + 
                    np.random.normal(0.0, traj.exploration_rate, self.current_individual.size),
                    self.optimizee_individual_dict_spec)
                for i in range(traj.n_random_steps)
            ]
            new_individual_list.append(list_to_dict(self.current_individual, self.optimizee_individual_dict_spec))
            if self.optimizee_bounding_func is not None:
                new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]
            fitnesses_results.clear()
            self.eval_pop = new_individual_list
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        logger.info("The last individual was %s with fitness %s", self.current_individual, self.current_fitness)
        logger.info("-- End of (successful) gradient descent --")


    def initClassicGD(self, parameters, traj):
        """
        Classic Gradient Descent specific initializiation.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.

        :return:
        """
        self.updateFunction = self.classicGDUpdate
    
    def initRMSProp(self, parameters, traj):
        """
        RMSProp specific initializiation.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.

        :return:
        """

        self.updateFunction = self.rmspropUpdate

        traj.f_add_parameter('momentum_decay', parameters.momentum_decay, 
            comment='Decay of the historic momentum at each gradient descent step')

        self.delta = 10**(-6)
        self.r = np.zeros(len(self.current_individual))

    def initAdam(self, parameters, traj):
        """
        ADAM specific initializiation.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.

        :return:
        """

        self.updateFunction = self.adamUpdate

        traj.f_add_parameter('first_order_decay', parameters.first_order_decay, 
            comment='Decay of the first order momentum')
        traj.f_add_parameter('second_order_decay', parameters.second_order_decay, 
            comment='Decay of the second order momentum')

        self.delta = 10**(-8)
        self.s = np.zeros(len(self.current_individual))
        self.r = np.zeros(len(self.current_individual)) 

    def initStochGD(self, parameters, traj):
        """
        Stochastic Gradient Descent specific initializiation.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.

        :return:
        """

        self.updateFunction = self.stochGDUpdate

        traj.f_add_parameter('stoch_deviation', parameters.stoch_deviation, 
            comment='Standard deviation of the random vector added to the gradient')
        traj.f_add_parameter('stoch_decay', parameters.stoch_decay, comment='Decay of the random vector')


    def classicGDUpdate(self, traj, gradient):
        """
        Updates the current individual using the classic Gradient Descent algorithm.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm

        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual

        :return:
        """
        self.current_individual += traj.learning_rate * gradient

    def rmspropUpdate(self, traj, gradient):
        """
        Updates the current individual using the RMSProp algorithm.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm

        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual

        :return:
        """

        self.r = traj.momentum_decay * self.r + (1 - traj.momentum_decay) * np.multiply(gradient, gradient)
        self.current_individual += np.multiply(traj.learning_rate / (np.sqrt(self.r + self.delta)), gradient)

    def adamUpdate(self, traj, gradient):
        """
        Updates the current individual using the ADAM algorithm.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm

        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual

        :return:
        """

        self.s = traj.first_order_decay * self.s + (1 - traj.first_order_decay) * gradient
        self.r = traj.second_order_decay * self.r + (1 - traj.second_order_decay) * np.multiply(gradient, gradient)
        s_corrected = self.s / (1 - traj.first_order_decay ** (self.g + 1))
        r_corrected = self.r/ (1 - traj.second_order_decay ** (self.g + 1))

        self.current_individual += np.multiply(traj.learning_rate * s_corrected / (np.sqrt(r_corrected) + self.delta), 
            gradient)

    def stochGDUpdate(self, traj, gradient):
        """
        Updates the current individual using a stochastic version of the gradient descent algorithm.

        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm

        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual

        :return:
        """

        gradient += (np.random.normal(0.0, traj.stoch_deviation, self.current_individual.size) * 
            traj.stoch_decay**(self.g + 1))
        self.current_individual += traj.learning_rate * gradient
