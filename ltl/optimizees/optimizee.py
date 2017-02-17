class Optimizee:
    """
    This is the base class for the Optimizees, i.e. the inner loop algorithms. Often, these are the implementations
    that interact with the environment. Given a set of paramters, it runs the simulation and returns the fitness
    achieved with those parameters.
    """

    def __init__(self):
        pass

    def create_individual(self):
        """
        Create one individual i.e. one instance of parameters. This is used by the :class:`ltl.optimizers.*` to
        initialize the individual/parameters. After that, the change in parameters is model specific e.g. In simulated
        annealing, it is perturbed on specific criteria

        :return: a :class:`list`
        """
        pass

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)

        :param  ~pypet.trajectory.Trajectory traj: The :mod:`pypet` trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.

        """
        pass

    def end(self):
        """
        Run any code required to clean-up, print final individuals etc.
        """
        pass
