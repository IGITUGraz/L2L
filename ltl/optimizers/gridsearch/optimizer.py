import logging
from collections import namedtuple

import numpy as np
from pypet.utils.explore import cartesian_product

from ltl.optimizers.optimizer import Optimizer
from ltl import dict_to_list
from ltl import DictEntryType

logger = logging.getLogger("ltl-gs")

GridSearchParameters = namedtuple('GridSearchParameters', [])


class GridSearchOptimizer(Optimizer):
    """
    This class implements a basic grid search optimizer. It runs the optimizee on a given grid of parameter values and
    returns the best fitness found. moreover, this can also simply be used to run a grid search and process the results
    stored in the traj in any manner desired.

    Notes regarding what it does -

    1.  This algorithm does not do any kind of adaptive searching and thus the concept of generations does not apply
        per se. That said, it is currently implemented as a series of runs in a single generation. All of these runs
        are declared in the constructor itself. The :func:`.Optimizer.post_process()` function simply prints the
        individual with the maximal fitness.
    
    2.  This algorithm doesnt make use of self.eval_pop and :func:`.Optimizer::_expand_trajectory()` simply because the
        cartesian product can be used more efficiently directly. (Imagine having to split a dict of 10000 parameter
        combinations into 10000 small `Individual-Dict`s and storing into eval_pop only to join them and call
        `traj.f_expand()` in :func:`.Optimizer::_expand_trajectory()`)

    :param  ~pypet.trajectory.Trajectory traj: Use this pypet trajectory to store the parameters of the specific runs.
        The parameters should be initialized based on the values in :param parameters:

    :param optimizee_create_individual: A function which when called returns one instance of parameter (or "individual")

    :param optimizee_fitness_weights: The weights which should be multiplied with the fitness returned from the
        :class:`~ltl.optimizees.optimizee.Optimizee` -- one for each element of the fitness (fitness can be
        multi-dimensional). If some element is negative, the Optimizer minimizes that element of fitness instead of
        maximizing. By default, the `Optimizer` maximizes all fitness dimensions.
    
    :param optimizee_param_grid: This is the data structure specifying the grid over which to search. This should be a
        dictionary as follows::

            (lower_bound, higher_bound, n_steps) = optimizee_param_grid['param_name']
        
        Where the interval `[lower_bound, upper_bound]` is divided into `n_steps` intervals thereby providing
        `n_steps+1` points for the grid.

        Note that there must be as many keys as there are in the `Individual-Dict` returned by the function
        `optimizee_create_individual`. Also, if any of the parameters of the idividuals is an array, then the above
        grid specification applies to each element of the array.

    :param parameters: A named tuple containing the parameters for the Optimizer class
    
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_param_grid):
        super().__init__(traj, optimizee_create_individual, optimizee_fitness_weights,
                         parameters)
        
        # Initializing basic variables
        self.optimizee_create_individual = optimizee_create_individual
        self.optimizee_fitness_weights = optimizee_fitness_weights
        
        sample_individual = self.optimizee_create_individual()

        # Assert validity of optimizee_param_grid
        assert set(sample_individual.keys()) == set(optimizee_param_grid.keys()), \
            "The Parameters of optimizee_param_grid don't match those of the optimizee individual"
        
        # Generate parameter dictionary based on optimizee_param_grid
        self.param_list = {}
        _, optimizee_individual_param_spec = dict_to_list(sample_individual, get_dict_spec=True)

        for param_name, param_type, param_length in optimizee_individual_param_spec:
            param_lower_bound = optimizee_param_grid[param_name][0]
            param_upper_bound = optimizee_param_grid[param_name][1]
            param_n_steps = optimizee_param_grid[param_name][2]
            if param_type == DictEntryType.Scalar:
                self.param_list[param_name] = np.linspace(param_lower_bound, param_upper_bound, param_n_steps + 1)
            elif param_type == DictEntryType.Sequence:
                curr_param_list = np.linspace(param_lower_bound, param_upper_bound)
                curr_param_list = np.meshgrid(*([curr_param_list] * param_length), indexing='ij')
                curr_param_list = [x.ravel() for x in curr_param_list]
                curr_param_list = np.stack(curr_param_list, axis=-1)
                self.param_list[param_name] = curr_param_list

        self.param_list = cartesian_product(self.param_list, tuple(sorted(optimizee_param_grid.keys())))
        
        # Adding the bounds information to the trajectory
        traj.par.f_add_parameter_group('grid_spec')
        for param_name, param_grid_spec in optimizee_param_grid.items():
            traj.par.grid_spec.f_add_parameter(param_name + '.lower_bound', )
        
        # Expanding the trajectory
        self.param_list = {('individual.' + key):value for key, value in self.param_list.items()}
        traj.f_expand(self.param_list)
        #: The current generation number
        self.g = None
        #: The population (i.e. list of individuals) to be evaluated at the next iteration
        self.eval_pop = None

    def post_process(self, traj, fitnesses_results):
        """
        In this optimizer, the post_proces function merely returns the best individual out of the grid and
        does not expand the trajectory. It also stores any relevant results
        """
        logger.info('Finished Simulation')
        logger.info('-------------------')
        logger.info('')

        run_idx_array = np.array([x[0] for x in fitnesses_results])
        fitness_array = np.array([x[1] for x in fitnesses_results])
        optimizee_fitness_weights = np.reshape(np.array(self.optimizee_fitness_weights), (-1, 1))

        weighted_fitness_array = np.matmul(fitness_array, optimizee_fitness_weights).ravel()
        max_fitness_indiv_index = np.argmax(weighted_fitness_array)

        logger.info('Storing Results')
        logger.info('---------------')

        for run_idx, run_fitness, run_weighted_fitness in zip(run_idx_array, fitness_array, weighted_fitness_array):
            traj.v_idx = run_idx
            traj.results.f_add_result('$set.$.fitness', np.array(run_fitness))
            traj.results.f_add_result('$set.$.weighted_fitness', run_weighted_fitness)

        logger.info('Best Individual is:')
        logger.info('')

        traj.v_idx = run_idx_array[max_fitness_indiv_index]
        individual = traj.par.individual
        for param_node in individual.f_iter_leaves():
            logger.info('  %s: %s', param_node.v_name, param_node.f_get())
        
        logger.info('  with fitness: %s', fitness_array[max_fitness_indiv_index])
        logger.info('  with weighted fitness: %s', weighted_fitness_array[max_fitness_indiv_index])

        traj.v_idx = -1

    def end(self):
        """
        Run any code required to clean-up, print final individuals etc.
        """

        logger.info('x -------------------------------- x')
        logger.info('  Completed SUCCESSFUL Grid Search  ')
        logger.info('x -------------------------------- x')
