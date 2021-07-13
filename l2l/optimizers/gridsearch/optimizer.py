import logging
from collections import namedtuple

import numpy as np
from l2l.utils.tools import cartesian_product

from l2l import DictEntryType
from l2l import dict_to_list
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.gridsearch")

GridSearchParameters = namedtuple('GridSearchParameters', ['param_grid'])
GridSearchParameters.__doc__ = """
:param dict param_grid: This is the data structure specifying the grid over which to search. This should be a
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

    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store the parameters of the specific runs.
        The parameters should be initialized based on the values in `parameters`

    :param optimizee_create_individual: A function which when called returns one instance of parameter (or "individual")

    :param optimizee_fitness_weights: The weights which should be multiplied with the fitness returned from the
        :class:`~l2l.optimizees.optimizee.Optimizee` -- one for each element of the fitness (fitness can be
        multi-dimensional). If some element is negative, the Optimizer minimizes that element of fitness instead of
        maximizing. By default, the `Optimizer` maximizes all fitness dimensions.
    
    :param parameters: An instance of :class:`.GridSearchParameters`

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
        self.size = len(self.param_list[list(self.param_list.keys())[0]])

        # Adding the bounds information to the trajectory
        traj.f_add_parameter_group('grid_spec')
        for param_name, param_grid_spec in optimizee_param_grid.items():
            traj.grid_spec.f_add_parameter(param_name + '.lower_bound', param_grid_spec[0])
            traj.grid_spec.f_add_parameter(param_name + '.uper_bound', param_grid_spec[1])
        traj.f_add_parameter('n_iteration', 1, comment='Grid search does only 1 iteration')
        #: The current generation number
        self.g = 0
        # Expanding the trajectory
        grouped_params_dict = {'individual.' + key: value for key, value in self.param_list.items()}
        final_params_dict = {'generation': [self.g],
                             'ind_idx': range(self.size)}
        final_params_dict.update(grouped_params_dict)
        traj.f_expand(cartesian_product(final_params_dict,
                                        [('ind_idx',) + tuple(grouped_params_dict.keys()), 'generation']))

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

        weighted_fitness_array = np.multiply(fitness_array, optimizee_fitness_weights).ravel()
        max_fitness_indiv_index = np.argmax(weighted_fitness_array)

        logger.info('Storing Results')
        logger.info('---------------')

        for run_idx, run_fitness, run_weighted_fitness in zip(run_idx_array, fitness_array, weighted_fitness_array):
            traj.v_idx = run_idx
            traj.f_add_result('$set.$.fitness', np.array(run_fitness))
            traj.f_add_result('$set.$.weighted_fitness', run_weighted_fitness)

        logger.info('Best Individual is:')
        logger.info('')

        traj.v_idx = run_idx_array[max_fitness_indiv_index]
        individual = traj.individual
        self.best_individual = {}
        for param_name, _, _ in self.optimizee_individual_dict_spec:
            param_value = self.param_list[param_name][max_fitness_indiv_index]
            logger.info('  %s: %s', param_name, param_value)
            self.best_individual[param_name] = param_value

        self.best_fitness = fitness_array[max_fitness_indiv_index]
        logger.info('  with fitness: %s', fitness_array[max_fitness_indiv_index])
        logger.info('  with weighted fitness: %s', weighted_fitness_array[max_fitness_indiv_index])

        self.g += 1
        traj.v_idx = -1

    def _expand_trajectory(self, traj):
        """
        Add as many explored runs as individuals that need to be evaluated. Furthermore, add the individuals as explored
        parameters.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return:
        """

        grouped_params_dict = get_grouped_dict(self.eval_pop)
        grouped_params_dict = {'individual.' + key: val for key, val in grouped_params_dict.items()}

        final_params_dict = {'generation': [self.g],
                             'ind_idx': range(len(self.eval_pop))}
        final_params_dict.update(grouped_params_dict)

        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`: This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product(final_params_dict,
                                        [('ind_idx',) + tuple(grouped_params_dict.keys()), 'generation']))

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
