class Optimizee:
    """
    This is the base class for the Optimizees, i.e. the inner loop algorithms. Often, these are the implementations
    that interact with the environment. Given a set of parameters, it runs the simulation and returns the fitness
    achieved with those parameters.
    """

    def __init__(self, traj):
        """
        This is the base class init function. Any implementation must in this class add a
        parameter add its parameters to this trajectory under the parameter group 'individual'
        which is created here in the base class. It is especially necessary to add all explored
        parameters (i.e. parameters that are returned via create_individual) to the trajectory.
        """
        traj.f_add_parameter_group('individual', 'Contains parameters of the optimizee')

    def get_params(self):
        """
        Get the important parameters of the optimizee. This is used by :class:`ltl.recorder`
        for recording the optimizee parameters.

        :return: a :class:`dict`
        """

    def create_individual(self):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary
        with dot-separated parameter names as keys and parameter values as values. This is used 
        by the :class:`ltl.optimizers.*` via the function create_individual() to initialize the
        individual/parameters. After that, the change in parameters is model specific e.g. In
        simulated annealing, it is perturbed on specific criteria

        :return dict: A dictionary containing the names of the paramters and their values
        """

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping

        :return dict: A dictionary in the same format as :ref:`.create_individual`, with the values clipped as needed.
        """

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)

        :param  ~pypet.trajectory.Trajectory traj: The :mod:`pypet` trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.

        """
