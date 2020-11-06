import numpy as np
from l2l.utils.experiment import Experiment

from l2l.optimizees.mnist.optimizee import MNISTOptimizeeParameters, MNISTOptimizee
from l2l.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
from l2l.optimizers.crossentropy.distribution import NoisyGaussian


def run_experiment():
    name = 'L2L-MNIST-CE'
    experiment = Experiment("../results/")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          log_stdout=True)
    optimizee_seed = 200

    optimizee_parameters = MNISTOptimizeeParameters(n_hidden=10, seed=optimizee_seed, use_small_mnist=True)
    ## Innerloop simulator
    optimizee = MNISTOptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = CrossEntropyParameters(pop_size=40, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=5000,
                                                  distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                                  stop_criterion=np.inf, seed=optimizer_seed)


    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(1.,),
                                      parameters=optimizer_parameters,
                                      optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
