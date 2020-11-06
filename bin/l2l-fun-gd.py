import numpy as np
from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
# from l2l.optimizers.gradientdescent.optimizer import ClassicGDParameters
# from l2l.optimizers.gradientdescent.optimizer import StochasticGDParameters
# from l2l.optimizers.gradientdescent.optimizer import AdamParameters
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters


def main():
    name = 'L2L-FUN-GD'
    experiment = Experiment("../results")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name)

    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function,
                                           seed=optimizee_seed)

    ## Outerloop optimizer initialization
    # parameters = ClassicGDParameters(learning_rate=0.01, exploration_step_size=0.01,
    #                                  n_random_steps=5, n_iteration=100,
    #                                  stop_criterion=np.Inf)
    # parameters = AdamParameters(learning_rate=0.01, exploration_step_size=0.01, n_random_steps=5, first_order_decay=0.8,
    #                             second_order_decay=0.8, n_iteration=100, stop_criterion=np.Inf)
    # parameters = StochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
    #                                     exploration_step_size=0.01, n_random_steps=5, n_iteration=100,
    #                                     stop_criterion=np.Inf)
    parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=5, momentum_decay=0.5,
                                   n_iteration=100, stop_criterion=np.Inf, seed=99)

    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(0.1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters)

    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
