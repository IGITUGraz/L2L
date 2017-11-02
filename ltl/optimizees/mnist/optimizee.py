import logging
from collections import namedtuple

import numpy as np
from sklearn.datasets import load_digits, fetch_mldata

from ltl.logging_tools import configure_loggers
from ltl.optimizees.optimizee import Optimizee
from ltl.optimizers.evolutionstrategies import EvolutionStrategiesOptimizer
from .nn import NeuralNetworkClassifier

logger = logging.getLogger("optimizees.mnist")

MNISTOptimizeeParameters = namedtuple('MNISTOptimizeeParameters',
                                      ['n_hidden', 'seed', 'use_small_mnist', 'activation_function', 'batch_size',
                                       'use_weight_decay', 'weight_decay_parameter'])


class MNISTOptimizee(Optimizee):
    """
    Implements a simple function optimizee. Functions are generated using the FunctionGenerator.
    NOTE: Make sure the optimizee_fitness_weights is set to (-1,) to minimize the value of the function

    :param traj: The trajectory used to conduct the optimization.
    :param .MNISTOptimizeeParameters parameters:
    """

    def __init__(self, traj, optimizee_parameters, es_parameters, per_worker_storage):
        super().__init__(traj)
        self.per_worker_storage = per_worker_storage

        seed = optimizee_parameters.seed
        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)

        if optimizee_parameters.use_small_mnist:
            # 8 x 8 images
            mnist_digits = load_digits()
            n_input = np.prod(mnist_digits.images.shape[1:])
            n_images = len(mnist_digits.images)  # 1797
            data_images = mnist_digits.images.reshape(n_images, -1) / 16.  # -> 1797 x 64
            data_targets = mnist_digits.target
        else:
            # 28 x 28 images
            mnist_digits = fetch_mldata('MNIST original')
            n_input = np.prod(mnist_digits.data.shape[1:])
            data_images = mnist_digits.data / 255.  # -> 70000 x 284
            n_images = len(data_images)
            data_targets = mnist_digits.target

        n_training = int(n_images * 0.85)
        train_data_images, train_data_targets = data_images[n_training:, ...], data_targets[n_training:, ...]
        self.test_data_images, self.test_data_targets = data_images[:n_training, ...], data_targets[:n_training, ...]

        self.pop_size = es_parameters.pop_size
        self.es_parameters = es_parameters

        n_optimizer_iterations = es_parameters.n_iteration
        batch_size = optimizee_parameters.batch_size
        self.training_batches = []
        for i in range(n_optimizer_iterations):
            train_batch = _next_batch(batch_size, train_data_images, train_data_targets, self.random_state)
            self.training_batches.append(train_batch)

        self.recorder_parameters = optimizee_parameters._asdict()

        n_hidden = optimizee_parameters.n_hidden
        n_output = 10  # This is always true for mnist
        activation_function = optimizee_parameters.activation_function
        self.nn = NeuralNetworkClassifier(n_input, n_hidden, n_output, activation_function)

        # global per_worker_storage
        flattened_weights = self.get_weights()
        self.per_worker_storage.store_current_individual(generation=-1, current_individual=flattened_weights)

        self.parameter_size = len(flattened_weights)
        self.use_weight_decay = optimizee_parameters.use_weight_decay
        self.weight_decay_parameter = optimizee_parameters.weight_decay_parameter

        ## Store things in trajectories

        # create_individual can be called because __init__ is complete except for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)

        traj.f_add_parameter_group('individual.network', 'Contains parameters of the optimizee')

        for key, val in optimizee_parameters._asdict().items():
            if key == 'activation_function':
                val = str(val)
            traj.individual.network.f_add_parameter(key, val)

        traj.individual.f_add_parameter('seed', seed)

        traj.individual.f_add_parameter('n_training', n_training)

    def get_params(self):
        """
        Get the important parameters of the optimizee. This is used by :class:`ltl.recorder`
        for recording the optimizee parameters.

        :return: a :class:`dict`
        """
        return self.recorder_parameters

    def create_individual(self):
        return dict(all_fitnesses=np.zeros(self.pop_size))

    def get_weights(self):
        """
        Creates a random value of parameter within given bounds
        """

        weight_shapes = self.nn.get_weights_shapes()
        cumulative_num_weights_per_layer = np.cumsum([np.prod(weight_shape) for weight_shape in weight_shapes])

        flattened_weights = np.empty(cumulative_num_weights_per_layer[-1])
        for i, weight_shape in enumerate(weight_shapes):
            if i == 0:
                flattened_weights[:cumulative_num_weights_per_layer[i]] = \
                    self.random_state.randn(np.prod(weight_shape)) / np.sqrt(weight_shape[1])
            else:
                flattened_weights[cumulative_num_weights_per_layer[i - 1]:cumulative_num_weights_per_layer[i]] = \
                    self.random_state.randn(np.prod(weight_shape)) / np.sqrt(weight_shape[1])

        return flattened_weights

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return individual

    # @profile
    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~pypet.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """

        # logger configuration is here since this function is paralellised
        configure_loggers(exactly_once=True)

        g = traj.generation
        prev_g = g - 1
        ind_idx = traj.ind_idx

        weighted_fitness_list = traj.individual.all_fitnesses

        # NOTE: Current individual is always for the previous generation for which all fitnesses are known.
        #^ For the case of the first generation, the "current indiv" is the initial value
        current_individual = self.per_worker_storage.get_current_individual(prev_g)
        if current_individual is None:
            prev_indiv = self.per_worker_storage.get_previous_individual()
            current_individual = EvolutionStrategiesOptimizer.update_current_individual(prev_indiv,
                                                                                        weighted_fitness_list,
                                                                                        prev_g,
                                                                                        self.parameter_size,
                                                                                        self.es_parameters)
            self.per_worker_storage.store_current_individual(generation=prev_g, current_individual=current_individual)

        ## Then we calculate the new individual for this idx for the current generation
        flattened_weights = EvolutionStrategiesOptimizer.get_new_individual(ind_idx, current_individual, g,
                                                                            self.parameter_size, self.es_parameters)

        weight_shapes = self.nn.get_weights_shapes()

        cumulative_num_weights_per_layer = np.cumsum([np.prod(weight_shape) for weight_shape in weight_shapes])

        weights = []
        for i, weight_shape in enumerate(weight_shapes):
            if i == 0:
                w = flattened_weights[:cumulative_num_weights_per_layer[i]].reshape(weight_shape)
            else:
                w = flattened_weights[
                    cumulative_num_weights_per_layer[i - 1]:cumulative_num_weights_per_layer[i]].reshape(weight_shape)
            weights.append(w)

        self.nn.set_weights(*weights)

        test_score = self.nn.score(self.test_data_images, self.test_data_targets)

        train_score = self.nn.score(*self.training_batches[g])

        if self.use_weight_decay:
            train_score -= self.weight_decay_parameter * np.sum(flattened_weights ** 2)

        traj.f_add_result('$set.$.test_score', test_score)
        traj.f_add_result('$set.$.train_score', train_score)

        return train_score


def _next_batch(num, data, labels, random_state):
    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0, len(data))
    random_state.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
