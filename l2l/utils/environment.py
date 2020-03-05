import inspect
import logging
import os
import pickle

from l2l.utils.JUBE_runner import JUBERunner
from l2l.utils.trajectory import Trajectory

logger = logging.getLogger("utils.Environment")


class Environment:
    """
    The Environment class takes the place of the pypet Environment and provides
    the required functionality to execute the inner loop. This means it uses
    either JUBE or sequential calls in order to execute all individuals in a
    generation. Based on the pypet environment concept:
    https://github.com/SmokinCaterpillar/pypet
    """

    def __init__(self, *args, **keyword_args):
        """
        Initializes an Environment
        :param args: arguments passed to the environment initialization
        :param keyword_args: arguments by keyword. Relevant keywords are
                             trajectory and filename.
        The trajectory object holds individual parameters and history per
        generation of the exploration process.
        """
        if 'trajectory' in keyword_args:
            traj = keyword_args['trajectory']
            if isinstance(traj, Trajectory):
                self.trajectory = traj
            else:
                self.trajectory = Trajectory(name=traj)
        if 'filename' in keyword_args:
            self.filename = keyword_args['filename']
            self.path = os.path.abspath(os.path.dirname(self.filename))
        else:
            stack = inspect.stack()
            self.path = os.path.dirname(stack[1].filename)

        self.per_gen_path = os.path.abspath(
                                os.path.join(self.path, 'per_gen_trajectories'))
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.per_gen_path, exist_ok=True)

        self.automatic_storing = keyword_args.get('automatic_storing', True)

        self.postprocessing = None
        self.multiprocessing = True
        if 'multiprocessing' in keyword_args:
            self.multiprocessing = keyword_args['multiprocessing']
        self.run_id = 0

        self.logging = False
        self.enable_logging()



    def run(self, runfunc):
        """
        Runs the optimizees using either JUBE or sequential calls.
        :param runfunc: The function to be called from the optimizee
        :return: the results of running a whole generation. Dictionary
                 indexed by generation id.
        """
        result = {}
        gen = self.trajectory.par['generation']
        n_loops = self.trajectory.par['n_iteration']
        for it in range(gen, n_loops):
            result[it] = []
            if self.multiprocessing:
                # Multiprocessing is done through JUBE, either with or
                # without scheduler
                logging.info(
                    "Environment run starting JUBERunner for n iterations: " +
                    str(self.trajectory.par['n_iteration']))
                jube = JUBERunner(self.trajectory)
                # Initialize new JUBE run and execute it
                try:
                    jube.write_pop_for_jube(self.trajectory, it)
                    result[it][:] = jube.run(self.trajectory, it)
                except Exception as e:
                    if self.logging:
                        logger.exception(
                            "Error launching JUBE run: %s" % str(e.__cause__))
                    raise e

            else:
                # Sequential calls to the runfunc in the optimizee
                # Call runfunc on each individual from the trajectory
                try:
                    for ind in self.trajectory.individuals[it]:
                        self.trajectory.individual = ind
                        fitness = runfunc(self.trajectory)
                        result[it].append((ind.ind_idx, fitness))
                        self.run_id = self.run_id + 1

                        import gc
                        gc.collect()

                except Exception as e:
                    if self.logging:
                        logger.exception(
                            "Error during serial execution of individuals: %s" %\
                            str(e.__cause__))
                    raise e

            # Add results to the trajectory
            self.trajectory.results.f_add_result_to_group(
                                                "all_results", it, result[it])
            self.trajectory.current_results = result[it]
            self.trajectory.par['generation'] = it

            if self.automatic_storing:
                trajfname = "Trajectory_{}_{:020d}.bin".format('final', it)
                traj_path = os.path.join(self.per_gen_path, trajfname)
                with open(traj_path, "wb") as handle:
                    pickle.dump(
                        self.trajectory, handle, pickle.HIGHEST_PROTOCOL)

            # Perform the postprocessing step in order to generate the new
            # parameter set
            self.postprocessing(self.trajectory, result[it])

        return result

    def add_postprocessing(self, func):
        """
        Function to add a postprocessing step
        :param func: the function which performs the postprocessing.
                     Postprocessing is the step where the results are assessed
                     in order to produce a new set of parameters for the next
                     generation.
        """
        self.postprocessing = func

    def enable_logging(self):
        """
        Function to enable logging
        TODO think about removing this.
        """
        self.logging = True

    def disable_logging(self):
        """
        Function to enable logging
        """
        self.logging = False
