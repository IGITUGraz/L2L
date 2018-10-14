import logging
from collections import namedtuple

import numpy as np
from pypet.utils.explore import cartesian_product

from ltl import DictEntryType
from ltl import dict_to_list
from ltl.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.gridsearch")

GridSearchParameters = namedtuple('GridSearchParameters', ['param_grid'])
GridSearchParameters.__doc__ = """
:param param_grid dict: This is the data structure specifying the grid over which to search. This should be a
    dictionary as follows::
    
        optimizee_param_grid['param_name'] = (lower_bound, higher_bound, n_steps)
    
    Where the interval `[lower_bound, upper_bound]` is divided into `n_steps` intervals thereby providing
    `n_steps + 1` points for the grid.
    
    Note that there must be as many keys as there are in the `Individual-Dict` returned by the function
    :meth:`.Optimizee.create_individual`. Also, if any of the parameters of the individuals is an array, then the above
    grid specification applies to each element of the array.
"""


class GridSearchOptimizer(Optimizer):
    """
    This class implements a basic grid search optimizer. It runs the optimizee on a given grid of parameter values and
    returns the best fitness found. moreover, this can also simply be used to run a grid search and process the results
    stored in the traj in any manner desired.

    Notes regarding what it does -

    1.  This algorithm does not do any kind of adaptive searching and thus the concept of generations does not apply
        per se. That said, it is currently implemented as a series of runs in a single generation. All of these runs
        are declared in the constructor itself. The :meth:`.Optimizer.post_process()` function simply prints the
        individual with the maximal fitness.
    
    2.  This algorithm doesnt make use of self.eval_pop and :meth:`.Optimizer._expand_trajectory()` simply because the
        cartesian product can be used more efficiently directly. (Imagine having to split a dict of 10000 parameter
        combinations into 10000 small `Individual-Dict`s and storing into eval_pop only to join them and call
        `traj.f_expand()` in :meth:`.Optimizer._expand_trajectory()`)

    :param  ~utils.trajectory.Trajectory traj: Use this trajectory to store the parameters of the specific runs.
        The parameters should be initialized based on the values in `parameters`

    :param optimizee_create_individual: A function which when called returns one instance of parameter (or "individual")

    :param optimizee_fitness_weights: The weights which should be multiplied with the fitness returned from the
        :class:`~ltl.optimizees.optimizee.Optimizee` -- one for each element of the fitness (fitness can be
        multi-dimensional). If some element is negative, the Optimizer minimizes that element of fitness instead of
        maximizing. By default, the `Optimizer` maximizes all fitness dimensions.
    
    :param parameters: An instance of :class:`.GridSearchParameters`

    :param optimizee_bounding_func: A function that returns the bound (between some limits) value of an individual
    
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj, optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights, parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.best_individual = None
        self.best_fitness = None

        sample_individual = self.optimizee_create_individual()

        # Generate parameter dictionary based on optimizee_param_grid
        self.param_list = {}
        _, optimizee_individual_param_spec = dict_to_list(sample_individual, get_dict_spec=True)
        self.optimizee_individual_dict_spec = optimizee_individual_param_spec

        optimizee_param_grid = parameters.param_grid
        # Assert validity of optimizee_param_grid
        assert set(sample_individual.keys()) == set(optimizee_param_grid.keys()), \
            "The Parameters of optimizee_param_grid don't match those of the optimizee individual"

        for param_name, param_type, param_length in optimizee_individual_param_spec:
            param_lower_bound, param_upper_bound, param_n_steps = optimizee_param_grid[param_name]
            if param_type == DictEntryType.Scalar:
                self.param_list[param_name] = np.linspace(param_lower_bound, param_upper_bound, param_n_steps + 1)
            elif param_type == DictEntryType.Sequence:
                curr_param_list = np.linspace(param_lower_bound, param_upper_bound, param_n_steps + 1)
                curr_param_list = np.meshgrid(*([curr_param_list] * param_length), indexing='ij')
                curr_param_list = [x.ravel() for x in curr_param_list]
                curr_param_list = np.stack(curr_param_list, axis=-1)
                self.param_list[param_name] = curr_param_list

        self.param_list = cartesian_product(self.param_list, tuple(sorted(optimizee_param_grid.keys())))

        # Adding the bounds information to the trajectory
        traj.parameters.f_add_parameter_group('grid_spec')
        for param_name, param_grid_spec in optimizee_param_grid.items():
            traj.parameters.grid_spec.f_add_parameter(param_name + '.lower_bound', )

        # Expanding the trajectory
        self.param_list = {('individual.' + key): value for key, value in self.param_list.items()}
        traj.f_expand(self.param_list)
        #: The current generation number
        self.g = 0
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
        self.best_individual = {}
        for param_name, _, _ in self.optimizee_individual_dict_spec:
            logger.info('  %s: %s', param_name, individual[param_name])
            self.best_individual[param_name] = individual[param_name]

        self.best_fitness = fitness_array[max_fitness_indiv_index]
        logger.info('  with fitness: %s', fitness_array[max_fitness_indiv_index])
        logger.info('  with weighted fitness: %s', weighted_fitness_array[max_fitness_indiv_index])

        self.g += 1
        traj.v_idx = -1

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.
        """
        traj.f_add_result('final_individual', self.best_individual)
        traj.f_add_result('final_fitness', self.best_fitness)
        traj.f_add_result('n_iteration', self.g)

        logger.info('x -------------------------------- x')
        logger.info('  Completed SUCCESSFUL Grid Search  ')
        logger.info('x -------------------------------- x')
