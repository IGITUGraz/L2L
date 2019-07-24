import logging
import numpy as np

from collections import namedtuple
from l2l import get_grouped_dict
from l2l.utils.tools import cartesian_product
from .enkf import EnsembleKalmanFilter as EnKF
from l2l import dict_to_list
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['noise', 'gamma', 'maxit', 'n_iteration',
                             'pop_size', 'n_batches', 'online', 'epsilon',
                             'decay_rate', 'seed', 'data', 'observations']
)

EnsembleKalmanFilterParameters.__doc__ = """
:param noise: float, Noise level
:param gamma
:param maxit: int, Epochs to run inside the Kalman Filter
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
:param n_batches: int, Number of mini-batches to use in the Kalman Filter
:param online: bool, Indicates if only one data point will used, 
               Default: False
:param epsilon: float, A value which is used when sampling from the best individual. 
                The value is multiplied to the covariance matrix as follows:
                :math:`\\epsilon * I` where I is the identity matrix with the 
                same size as the covariance matrix. The value is 
                exponentially decaying and should be in [0,1] and used in 
                combination with `decay_rate`. 
:param decay_rate: float, Decay rate for the sampling. 
                For the exponential decay as follows:
                .. math::
                    \\epsilon = \\epsilon_0 e^{-decay_rate * epoch}

                Where :math:`\\epsilon` is the value from `epsilon`. The
:param seed: The random seed used to sample and fit the distribution. 
             Uses a random generator seeded with this seed.
:param data: nd numpy array, numpy array containing data in format 
             (batch_size, data)
:param observations: nd numpy array, observation or targets, should be e.g. 
                     array of integers
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 optimizee_create_new_individuals,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func
        self.optimizee_create_individual = optimizee_create_individual
        self.optimizee_fitness_weights = optimizee_fitness_weights
        self.optimizee_create_new_individuals = optimizee_create_new_individuals
        self.parameters = parameters

        traj.f_add_parameter('gamma', parameters.gamma, comment='Noise level')
        traj.f_add_parameter('noise', parameters.noise,
                             comment='Multivariate noise distribution')
        traj.f_add_parameter('maxit', parameters.maxit,
                             comment='Maximum iterations')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('shuffle', parameters.shuffle)
        traj.f_add_parameter('n_batches', parameters.n_batches)
        traj.f_add_parameter('online', parameters.online)
        traj.f_add_parameter('epsilon', parameters.epsilon)
        traj.f_add_parameter('decay_rate', parameters.decay_rate)
        traj.f_add_parameter('seed', np.uint32(parameters.seed),
                             comment='Seed used for random number generation '
                                     'in optimizer')
        traj.f_add_parameter('pop_size', parameters.pop_size)
        traj.f_add_parameter('data', parameters.data)
        traj.f_add_parameter('observations', parameters.observations)

        _, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(), get_dict_spec=True)

        traj.results.f_add_result_group('generation_params')

        # Set the random state seed for distribution
        self.random_state = np.random.RandomState(traj.parameters.seed)

        # for the sampling procedure
        # `epsilon` value given by the user
        self.epsilon = parameters.epsilon
        # decay rate
        self.decay_rate = parameters.decay_rate

        #: The population (i.e. list of individuals) to be evaluated at the
        # next iteration
        current_eval_pop = [self.optimizee_create_individual() for _ in
                            range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in
                                current_eval_pop]

        self.eval_pop = current_eval_pop
        self.best_fitness = 0.
        self.best_individual = None

        self.inputs, self.targets = parameters.data, parameters.observations

        for e in self.eval_pop:
            e["inputs"] = self.inputs
            e["targets"] = self.targets

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        This is the key function of this class. Given a set of :obj:`fitnesses_results`,
        and the :obj:`traj`, it uses the fitness to decide on the next set of
        parameters to be evaluated. Then it fills the :attr:`.Optimizer.eval_pop`
        with the list of parameters it wants evaluated at the next simulation
        cycle, increments :attr:`.Optimizer.g` and calls :meth:`._expand_trajectory`

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :param list fitnesses_results: This is a list of fitness results that contain tuples run index and the fitness.
            It is of the form `[(run_idx, run), ...]`

        """

        # old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()

        individuals = traj.individuals[traj.generation]
        gamma = traj.gamma
        ens_res = []
        ens_fitnesses = []

        # go over all individuals
        for i in individuals:
            # optimization
            ens = np.array(i.ens)
            ensemble_size = ens.shape[0]
            # get the score/fitness of the individual
            fitness_per_individual = traj.current_results[i.ind_idx][1][
                'loss']
            ens_fitnesses.append(fitness_per_individual)
            model_output = traj.current_results[i.ind_idx][1]['out']
            enkf = EnKF(maxit=traj.maxit,
                        online=traj.online,
                        n_batches=traj.n_batches)
            enkf.fit(ensemble=ens,
                     ensemble_size=ensemble_size,
                     u_exact=None,
                     observations=self.targets,
                     model_output=model_output,
                     noise=traj.noise, p=None, gamma=gamma)
            ens_res.append(enkf.ensemble)

        generation_name = 'generation_{}'.format(traj.generation)
        traj.results.generation_params.f_add_result_group(generation_name)
        ens_fitnesses = np.array(ens_fitnesses)

        generation_result_dict = {
            'generation': traj.generation,
            'ensemble_fitnesses': ens_fitnesses,
        }
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        if traj.generation > 1 and traj.generation % 1000 == 0:
            params, self.best_fitness, self.best_individual = self._new_individuals(
                traj, ens_fitnesses, individuals)
            self.eval_pop = [dict(ens=params[i],
                                  inputs=self.inputs,
                                  targets=self.targets)
                             for i in range(traj.pop_size)]
        else:
            self.eval_pop = [dict(ens=ens_res[i],
                                  inputs=self.inputs,
                                  targets=self.targets
                                  )
                             for i in range(traj.pop_size)]
        traj.generation += 1
        self._expand_trajectory(traj)

    def _new_individuals(self, traj, fitnesses, individuals):
        """
        Sample new individuals by first ranking and then sampling from a
        gaussian distribution. The
        """
        ranking_idx = list(reversed(np.argsort(fitnesses)))
        best_fitness = fitnesses[ranking_idx][0]
        best_ranking_idx = ranking_idx[0]
        best_individual = individuals[best_ranking_idx]
        # do the decay
        eps = self.epsilon * np.exp(-self.decay_rate * traj.generation)
        # now do the sampling
        params = [
            self.optimizee_create_new_individuals(self.random_state,
                                                  individuals[
                                                      best_ranking_idx].params,
                                                  eps)
            for _ in range(traj.pop_size)]
        return params, best_fitness, best_individual

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        """
        traj.f_add_result('final_individual', self.best_individual)

        logger.info(
            "The last individual {} was with fitness {}".format(
                self.best_individual, self.best_fitness))
        logger.info("-- End of (successful) EnKF optimization --")

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
        grouped_params_dict = {'individual.' + key: val for key, val in
                               grouped_params_dict.items()}

        final_params_dict = {'generation': [traj.generation],
                             'ind_idx': range(len(self.eval_pop))}
        final_params_dict.update(grouped_params_dict)

        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`: This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product(final_params_dict,
                                        [('ind_idx',) + tuple(
                                            grouped_params_dict.keys()),
                                         'generation']))
