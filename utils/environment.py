from utils.trajectory import Trajectory
from utils.JUBE_runner import JUBERunner


class Environment:

    def __init__(self, *args, **keyword_args):
        if 'trajectory' in keyword_args:
            self.trajectory = Trajectory(name=keyword_args['trajectory'])
        if 'filename' in keyword_args:
            self.filename = keyword_args['filename']
        self.postprocessing = None
        self.multiprocessing = True
        self.run_id = 0

    def run(self, runfunc):
        #    args = JubeArgs()
        #    create_jube_xml_files(self.trajectory)
        #    main.run_new_benchmark(args_dictionary)

        result = {}
        for it in range(self.trajectory.par['n_iteration']):
            if self.multiprocessing:
                jube = JUBERunner(self.trajectory)
                result[it] = []
                #Initialize new JUBE run
                try:
                    jube.write_pop_for_jube(self.trajectory,it)
                    result[it] = jube.run(self.trajectory,it)

                except RuntimeError:
                    print("Error launching JUBE")
                self.trajectory.results.f_add_result_to_group("all_results", it, result[it])
                self.trajectory.current_results = result[it]
                self.postprocessing(self.trajectory, result[it])

            else:
                result[it] = []
                for ind in self.trajectory.individuals[it]:
                    self.trajectory.individual = ind
                    result[it].append((ind.ind_idx, runfunc(self.trajectory)))
                    self.run_id = self.run_id + 1
                self.trajectory.results.f_add_result_to_group("all_results", it, result[it])
                self.trajectory.current_results = result[it]
                self.postprocessing(self.trajectory, result[it])

        return result

    def add_postprocessing(self, func):
        self.postprocessing = func

    def enable_logging(self):
        self.logging = True

    def disable_logging(self):
        self.logging = False
