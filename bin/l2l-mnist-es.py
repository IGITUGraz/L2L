import numpy as np
from l2l.utils.experiment import Experiment
from l2l.optimizees.mnist.optimizee import MNISTOptimizeeParameters, MNISTOptimizee
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer

def run_experiment():
    name = 'L2L-MNIST-ES'
    trajectory_name = 'mirroring-and-fitness-shaping'
    experiment = Experiment("../results/")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=trajectory_name,
                                                          log_stdout=True)

    optimizee_seed = 200
    optimizee_parameters = MNISTOptimizeeParameters(n_hidden=10, seed=optimizee_seed, use_small_mnist=True)
    ## Innerloop simulator
    optimizee = MNISTOptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = EvolutionStrategiesParameters(
        learning_rate=0.1,
        noise_std=0.1,
        mirrored_sampling_enabled=True,
        fitness_shaping_enabled=True,
        pop_size=20,
        n_iteration=2000,
        stop_criterion=np.Inf,
        seed=optimizer_seed)

    optimizer = EvolutionStrategiesOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
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
