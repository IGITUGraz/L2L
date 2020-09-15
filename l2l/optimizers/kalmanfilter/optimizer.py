import logging
import numpy as np

from collections import namedtuple
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter as EnKF
from l2l import dict_to_list
from l2l.optimizers.optimizer import Optimizer
from l2l.optimizers.crossentropy.distribution import Gaussian

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['noise', 'gamma', 'max_iterations', 'n_iteration',
                             'pop_size', 'online', 'seed',
                             'data', 'observations', 'sampling_generation']
)

EnsembleKalmanFilterParameters.__doc__ = """
:param noise: float, Noise level
:param gamma
:param max_iteration: int, Epochs/Repititions to run inside the Kalman Filter, 
                      Default: 1
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
:param online: bool, Indicates if only one data point will used, 
               Default: False
:param sampling_generation: After `sampling_gerneration` steps a gaussian sampling 
        on the parameters of the best individual is done, ranked by the fitness
        value 
:param seed: The random seed used to sample and fit the distribution. 
             Uses a random generator seeded with this seed.
:param data: nd numpy array, numpy array containing data in format 
             (batch_size, data)
:param observations: nd numpy array, observation or targets, should be an
                     array of integers
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer

    Note: Requires targets or observations and model output (`model_output`).
    Targets have to come from the dataset, model outputs have to come from the
    model itself and should be called `model_output` returned as a dictionary
    from the optmizee. `model_output` should have a 3 dimensional form  as
    `(n_ensembles, ensemble, batch_size)`. `batch_size` corresponds to the
    number of observations, `n_ensembles` to `pop_size` and ensemble are the
    parameters to optimize.
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
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
        self.parameters = parameters

        traj.f_add_parameter('gamma', parameters.gamma, comment='Noise level')
        traj.f_add_parameter('noise', parameters.noise,
                             comment='Multivariate noise distribution')
        traj.f_add_parameter('max_iterations', parameters.max_iterations,
                             comment='Maximum iterations')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('shuffle', parameters.shuffle)
        traj.f_add_parameter('n_batches', parameters.n_batches)
        traj.f_add_parameter('online', parameters.online)
        traj.f_add_parameter('sampling_generation', parameters.sampling_generation)
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
        self.eval_pop.clear()

        individuals = traj.individuals[traj.generation]
        gamma = traj.gamma
        ens_fitnesses = []
        ensembles = []
        model_output = []
        ensemble_size = traj.pop_size

        # go over all individuals
        for i in individuals:
            # optimization
            ens = np.array(i.ensemble)
            ensembles.append(ens)
            # get the score/fitness of the individual
            fitness_per_individual = traj.current_results[i.ind_idx][1][
                'loss']
            ens_fitnesses.append(fitness_per_individual)
            # model output
            mo = np.array(traj.current_results[i.ind_idx][1]['model_output'])
            model_output.append(mo)
        enkf = EnKF(maxit=traj.max_iterations,
                    online=traj.online)
        enkf.fit(ensemble=ensembles,
                 ensemble_size=ensemble_size,
                 observations=self.targets,
                 model_output=model_output,
                 gamma=gamma)
        # results
        ens_res = enkf.ensemble

        generation_name = 'generation_{}'.format(traj.generation)
        traj.results.generation_params.f_add_result_group(generation_name)
        ens_fitnesses = np.array(ens_fitnesses)

        generation_result_dict = {
            'generation': traj.generation,
            'ensemble_fitnesses': ens_fitnesses,
        }
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        if traj.generation > 1 and traj.generation % traj.sampling_generation == 0:
            params, self.best_fitness, self.best_individual = self._new_individuals(
                traj, ens_fitnesses, individuals, ensemble_size)
            self.eval_pop = [dict(ensemble=params[i],
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

    @staticmethod
    def _create_individual_distribution(random_state, weights,
                                        ensemble_size):
        dist = Gaussian()
        dist.init_random_state(random_state)
        dist.fit(weights)
        new_individuals = dist.sample(ensemble_size)
        return new_individuals

    def _new_individuals(self, traj, fitnesses, individuals, ensemble_size):
        """
        Sample new individuals by first ranking and then sampling from a
        gaussian distribution. The
        """
        ranking_idx = list(reversed(np.argsort(fitnesses)))
        best_fitness = fitnesses[ranking_idx][0]
        best_ranking_idx = ranking_idx[0]
        best_individual = individuals[best_ranking_idx]
        # now do the sampling
        params = [
            self._create_individual_distribution(self.random_state,
                                                 individuals[
                                                     best_ranking_idx].params,
                                                 ensemble_size)
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
