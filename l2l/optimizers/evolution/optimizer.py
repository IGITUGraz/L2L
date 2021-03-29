import logging
import random

from collections import namedtuple

from deap import base, creator, tools
from deap.tools import HallOfFame

from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("l2l-ga")

GeneticAlgorithmParameters = namedtuple('GeneticAlgorithmParameters',
                                        ['seed', 'pop_size', 'cx_prob', 'mut_prob', 'n_iteration', 'ind_prob', 'tourn_size', 'mate_par',
                                         'mut_par'])
GeneticAlgorithmParameters.__doc__ = """
:param seed: Random seed
:param pop_size: Size of the population
:param cx_prob: Crossover probability
:param mut_prob: Mutation probability
:param n_iteration: Number of generations simulation should run for
:param ind_prob: Probability of mutation of each element in individual
:param tourn_size: Size of the tournamaent used for fitness evaluation and selection
:param mate_par: Parameter used for blending two values during mating
:param mut_par: Standard deviation for the gaussian addition mutation.
"""


class GeneticAlgorithmOptimizer(Optimizer):
    """
    Implements evolutionary algorithm

    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store the parameters of the specific runs.
      The parameters should be initialized based on the values in `parameters`
    :param optimizee_create_individual: Function that creates a new individual
    :param optimizee_fitness_weights: Fitness weights. The fitness returned by the Optimizee is multiplied by these
      values (one for each element of the fitness vector)
    :param parameters: Instance of :func:`~collections.namedtuple` :class:`.GeneticAlgorithmOptimizer` containing the parameters
      needed by the Optimizer
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):

        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters, optimizee_bounding_func=optimizee_bounding_func)
        self.optimizee_bounding_func = optimizee_bounding_func
        __, self.optimizee_individual_dict_spec = dict_to_list(optimizee_create_individual(), get_dict_spec=True)

        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')
        traj.f_add_parameter('pop_size', parameters.pop_size, comment='Population size')  # 185
        traj.f_add_parameter('cx_prob', parameters.cx_prob, comment='Crossover term')
        traj.f_add_parameter('mut_prob', parameters.mut_prob, comment='Mutation probability')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of generations')

        traj.f_add_parameter('ind_prob', parameters.ind_prob, comment='Mutation parameter')
        traj.f_add_parameter('tourn_size', parameters.tourn_size, comment='Selection parameter')

        # ------- Create and register functions with DEAP ------- #
        # delay_rate, slope, std_err, max_fraction_active
        creator.create("FitnessMax", base.Fitness, weights=self.optimizee_fitness_weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # Structure initializers
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         lambda: dict_to_list(optimizee_create_individual()))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operator registering
        # This complex piece of code is only necessary because we're using the
        # DEAP framework and would like to decorate the DEAP mutation operator
        def bounding_decorator(func):
            def bounding_wrapper(*args, **kwargs):
                if self.optimizee_bounding_func is None:
                    return func(*args, **kwargs)
                else:
                    # Deap Functions modify individuals in-place, Hence we must do the same
                    result_individuals_deap = func(*args, **kwargs)
                    result_individuals = [list_to_dict(x, self.optimizee_individual_dict_spec)
                                          for x in result_individuals_deap]
                    bounded_individuals = [self.optimizee_bounding_func(x) for x in result_individuals]
                    for i, deap_indiv in enumerate(result_individuals_deap):
                        deap_indiv[:] = dict_to_list(bounded_individuals[i])
                    print("Bounded Individual: {}".format(bounded_individuals))
                    return result_individuals_deap

            return bounding_wrapper

        toolbox.register("mate", tools.cxBlend, alpha=parameters.mate_par)
        toolbox.decorate("mate", bounding_decorator)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=parameters.mut_par, indpb=traj.ind_prob)
        toolbox.decorate("mutate", bounding_decorator)
        toolbox.register("select", tools.selTournament, tournsize=traj.tourn_size)

        # ------- Initialize Population and Trajectory -------- #
        # NOTE: The Individual object implements the list interface.
        self.pop = toolbox.population(n=traj.pop_size)
        self.eval_pop_inds = [ind for ind in self.pop if not ind.fitness.valid]
        self.eval_pop = [list_to_dict(ind, self.optimizee_individual_dict_spec)
                         for ind in self.eval_pop_inds]

        self.g = 0  # the current generation
        self.toolbox = toolbox  # the DEAP toolbox
        self.hall_of_fame = HallOfFame(20)
        self.best_individual = None

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`
        """
        CXPB, MUTPB, NGEN = traj.cx_prob, traj.mut_prob, traj.n_iteration

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))

        #**************************************************************************************************************
        # Storing run-information in the trajectory
        # Reading fitnesses and performing distribution update
        #**************************************************************************************************************
        for run_index, fitness in fitnesses_results:
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx

            traj.f_add_result('$set.$.individual', self.eval_pop[ind_index])
            traj.f_add_result('$set.$.fitness', fitness)

            # Use the ind_idx to update the fitness
            individual = self.eval_pop_inds[ind_index]
            individual.fitness.values = fitness

        traj.v_idx = -1  # set the trajectory back to default

        logger.info("-- End of generation {} --".format(self.g))
        best_inds = tools.selBest(self.eval_pop_inds, 2)
        self.best_individual = list_to_dict(best_inds[0], self.optimizee_individual_dict_spec)
        for best_ind in best_inds:
            print("Best individual is %s, %s" % (list_to_dict(best_ind, self.optimizee_individual_dict_spec),
                                                 best_ind.fitness.values))

        self.hall_of_fame.update(self.eval_pop_inds)

        logger.info("-- Hall of fame --")
        for hof_ind in tools.selBest(self.hall_of_fame, 2):
            logger.info("HOF individual is %s, %s" % (list_to_dict(hof_ind, self.optimizee_individual_dict_spec),
                                                      hof_ind.fitness.values))

        # ------- Create the next generation by crossover and mutation -------- #
        if self.g < NGEN - 1:  # not necessary for the last generation
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            if len(set(map(tuple, offspring))) < len(offspring):
                logger.info("Mutating more")
                for i, o1 in enumerate(offspring[:-1]):
                    for o2 in offspring[i + 1:]:
                        if tuple(o1) == tuple(o2):
                            if random.random() < 0.8:
                                self.toolbox.mutate(o2)
                                del o2.fitness.values

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

            self.eval_pop_inds = [ind for ind in self.pop if not ind.fitness.valid]
            self.eval_pop = [list_to_dict(ind, self.optimizee_individual_dict_spec)
                             for ind in self.eval_pop_inds]

            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        logger.info("-- End of (successful) evolution --")
        best_inds = tools.selBest(self.pop, 10)
        for best_ind in best_inds:
            logger.info("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

        logger.info("-- Hall of fame --")
        for hof_ind in self.hall_of_fame:
            logger.info("HOF individual is %s, %s" % (hof_ind, hof_ind.fitness.values))
