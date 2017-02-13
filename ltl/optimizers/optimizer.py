from pypet import cartesian_product


class Optimizer:
    """
    This is the base class for the Optimizers i.e. the outer loop algorithms. These algorithms generate parameters, \
    give them to the inner loop to be evaluated, and with the resulting fitness modify the parameters in some way.

    """

    def __init__(self, traj, optimizee_create_individual, optimizee_fitness_weights, parameters):
        """
        Initialize the Optimizer (run one time code)

        :param  ~pypet.trajectory.Trajectory traj: Use this pypet trajectory to store the parameters of the specific runs.
        The parameters should be initialized based on the values in :param parameters:

        :param optimizee_create_individual: A function which when called returns one instance of parameter (or "individual")

        :param optimizee_fitness_weights: The weights which should be multiplied with the fitness returned from the
        :class:`~Optimizee`. If negative, the Optimizer minimizes instead of maximizing.

        :param parameters: A named tuple containing the parameters for the Optimizer class
        """
        self.g = None
        self.eval_pop = None

    def post_process(self, traj, fitnesses_results):
        """
        This is the key function of this class. Given a set of :obj:`fitnesses_results`,  and the :obj:`traj`, it uses \
        the fitness to decide on the next set of parameters to be evaluated. Then it fills the :attr:`eval_pop` with \
        the list of parameters it wants evaluated at the next simulation cycle, increments :attr:`g` and calls \
        :meth:`_expand_trajectory`

        :param  ~pypet.trajectory.Trajectory traj: The :mod:`pypet` trajectory that contains the parameters and the \
        individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g. \
        param1 is accessible using `traj.param1`

        :param list fitnesses_results: This is a list of fitness results that contain tuples run index and the fitness. \
        It is of the form `[(run_idx, run), ...]` as constructed by :mod:`pypet`

        """
        # NOTE: Always remember to keep the following two lines.
        # TODO: Set eval_pop to the values of parameters you want to evaluate in the next cycle
        # self.eval_pop = ...
        self.g += 1
        self._expand_trajectory(traj)

    def end(self):
        """
        Run any code required to clean-up, print final individuals etc.
        """
        pass

    def _expand_trajectory(self, traj):
        """
        Add as many explored runs as individuals that need to be evaluated. Furthermore, add the individuals as \
        explored parameters.

        :param  ~pypet.trajectory.Trajectory traj: The :mod:`pypet` trajectory that contains the parameters and the \
        individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g. \
        param1 is accessible using `traj.param1`

        :return:
        """

        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`: This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product({'generation': [self.g],
                                         'ind_idx': range(len(self.eval_pop)),
                                         'individual': self.eval_pop},
                                        [('ind_idx', 'individual'), 'generation']))
