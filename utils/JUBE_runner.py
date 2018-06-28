from JUBE.jube2.main import main, continue_benchmarks
import os.path
import pickle
import time
import logging
import shutil

logger = logging.getLogger("JUBERunner")


class JUBERunner():
    """
    JUBERunner is a class that takes care of handling the interaction with JUBE and generating the
    right files in order to specify the JUBE runs with the parameters defined by the optimzier.
    This class consists of helper tools to generate JUBE configuration files as well as routines to
    interact with JUBE and gather the results to pass them back to the environment.
    """

    def __init__(self, trajectory):
        """
        Initializes the JUBERunner using the parameters found inside the trajectory in the
        param dictionary called JUBE_params.

        :param trajectory: A trajectory object holding the parameters to use in the initialization
        """
        self.trajectory = trajectory
        self.done = False
        if not 'JUBE_params' in self.trajectory.par.keys():
            print("exception")
        else:
            args = self.trajectory.par["JUBE_params"].params

        self._prefix = args.get('fileprefix',"")
        self.jube_config = {
            'submit_cmd': args.get('submit_cmd',""),
            'job_file': args.get('job_file',""),
            'nodes': args.get('nodes',""),
            'walltime': args.get('walltime',""),
            'ppn': args.get('ppn',""),
            'ready_file': args.get('ready_file',""),
            'mail_mode':args.get('mail_mode',""),
            'mail_address': args.get('mail_address',""),
            'err_file': args.get('err_file',"error.out"),
            'out_file': args.get('out_file',"jout.out"),
            'tasks_per_job': args.get('tasks_per_job', "1"),
            'cpu_pp': args.get('cpu_pp', "1"),
        }
        self.scheduler = "None"
        if 'scheduler' in args.keys():
            self.scheduler = args.get('scheduler'),
        self.exec = args['exec']
        self.filename = ""
        self.path = args['work_path']
        # Create directories for workspace
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.path + "jube_xml"):
            os.makedirs(self.path + "jube_xml")
        if not os.path.exists(self.path + "run_files"):
            os.makedirs(self.path + "run_files")
        if not os.path.exists(self.path + "ready_files"):
            os.makedirs(self.path + "ready_files")
        if not os.path.exists(self.path + "trajectories"):
            os.makedirs(self.path + "trajectories")
            if not os.path.exists(self.path + "results"):
                os.makedirs(self.path + "results")


    def write_pop_for_jube(self, trajectory, generation):
        """
        Writes an XML file which contains the parameters for JUBE
        :param trajectory: A trajectory object holding the parameters to generate the JUBE XML file for each generation
        :param generation: Id of the current generation
        """
        self.trajectory = trajectory
        eval_pop = trajectory.individuals[generation]
        self.generation = generation
        self.filename = self.path + "jube_xml/_jube_" + str(self.generation) + ".xml"

        f = open(self.filename, 'w')
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<jube>\n")
        f.write("  <benchmark name=\"ltl_inner_loop\" outpath=\"bench_run\">\n")

        # Write the parameters for this run
        f.write("    <parameterset name=\"ltl_parameters\">\n")
        f.write("      <parameter name=\"index\" type=\"int\">")
        indexes = ""
        for individual in eval_pop:
            indexes = indexes + str(individual.ind_idx) + ","
        indexes = indexes[:-1]
        f.write(indexes)
        f.write("</parameter>\n")
        f.write("    </parameterset>\n\n")

        f.write("    <!-- benchmark configuration -->\n")

        f.write("    <!-- Job configuration -->\n")
        f.write("    <parameterset name=\"execute_set\">\n")
        f.write("    <parameter name=\"exec\">" + self.exec + "</parameter>\n")
        f.write("    <parameter name=\"tasks_per_job\">" + self.jube_config['tasks_per_job'] + "</parameter>\n")
        if self.scheduler != 'None':
            f.write("    <parameter name=\"submit_cmd\">" + self.jube_config['submit_cmd'] + "</parameter>\n")
            f.write("    <parameter name=\"job_file\">" + self.jube_config['job_file'] + "$index " + str(
                generation) + "</parameter>\n")
            f.write("    <parameter name=\"nodes\" type=\"int\">" + self.jube_config['nodes'] + "</parameter>\n")
            f.write("    <parameter name=\"walltime\">" + self.jube_config['walltime'] + "</parameter>\n")
            f.write("    <parameter name=\"ppn\" type=\"int\">" + self.jube_config['ppn'] + "</parameter>\n")
            f.write("    <parameter name=\"ready_file_scheduler\" mode=\"python\" type=\"string\"> "
                    "/ready_files/ready_ + ${index} + </parameter>\n")
            f.write("    <parameter name=\"ready_file\">" + self.jube_config['ready_file'] + str(
                self.generation) + "</parameter>\n")
            f.write("    <parameter name=\"mail_mode\">" + self.jube_config['mail_mode'] + "</parameter>\n")
            f.write("    <parameter name=\"mail_address\">" + self.jube_config['mail_address'] + "</parameter>\n")
            f.write("    <parameter name=\"err_file\">" + self.jube_config['err_file'] + "</parameter>\n")
            f.write("    <parameter name=\"out_file\">" + self.jube_config['out_file'] + "</parameter>\n")

        f.write("    </parameterset>\n")

        # Write the specific scheduler file
        if self.scheduler != 'None':
            self.write_scheduler_file(f)

        f.write("    <!-- Operation -->\n")
        f.write("    <step name=\"submit\" work_dir=\"" + self.path +
                "/work/jobsystem_bench_${jube_benchmark_id}_${jube_wp_id}\" >\n")
        f.write("    <use>ltl_parameters</use>\n")
        f.write("    <use>execute_set</use>\n")

        if self.scheduler != 'None':
            f.write("    <use>files,sub_job</use>\n")
            f.write("    <do done_file=\""+self.path + "/ready_files/ready_w_" + str(
                self.generation) + "\">$submit_cmd $job_file </do> <!-- shell command -->\n")
        else:
            f.write("    <do done_file=\"" + self.path + "/ready_files/ready_w_" + str(
                self.generation) + "\">$exec $index " + str(self.generation) +
                    " -n $tasks_per_job </do> <!-- shell command -->\n")

        f.write("    </step>   \n")

        # Close
        f.write("  </benchmark>\n")
        f.write("</jube>\n")
        f.close()
        logger.info("Generated JUBE XML file for generation: " + str(self.generation))

    def write_scheduler_file(self, f):
        """
        Writes the scheduler specific part of the JUBE XML specification file
        :param f: the handle to the XML configuration file
        """
        f.write("    <!-- Load jobfile -->\n")
        f.write("    <fileset name=\"files\">\n")
        f.write("    <copy>${job_file}.in</copy>\n")
        f.write("    </fileset>\n")

        f.write("    <!-- Substitute jobfile -->\n")
        f.write("    <substituteset name=\"sub_job\">\n")
        f.write("    <iofile in=\"${job_file}.in\" out=\"$job_file\" />\n")
        f.write("    <sub source=\"#NODES#\" dest=\"$nodes\" />\n")
        f.write("    <sub source=\"#PROCS_PER_NODE#\" dest=\"$ppn\" />\n")
        f.write("    <sub source=\"#WALLTIME#\" dest=\"$walltime\" />\n")
        f.write("    <sub source=\"#ERROR_FILEPATH#\" dest=\"$err_file\" />\n")
        f.write("    <sub source=\"#OUT_FILEPATH#\" dest=\"$out_file\" />\n")
        f.write("    <sub source=\"#MAIL_ADDRESS#\" dest=\"$mail_address\" />\n")
        f.write("    <sub source=\"#MAIL_MODE#\" dest=\"$mail_mode\" />\n")
        f.write("    <sub source=\"#EXEC#\" dest=\"$exec $index " + str(self.generation) + " -n $tasks_per_job\"/>\n")
        f.write("    <sub source=\"#READY#\" dest=\"$ready_file" + str(self.generation) + "\" />\n")
        f.write("    </substituteset> \n")

    def collect_results_from_run(self, generation, individuals):
        """
        Collects the results generated by each individual in the generation. Results are, for the moment, stored
        in individual binary files.
        :param generation: generation id
        :param individuals: list of individuals which were executed in this generation
        :return results: a list containing objects produced as results of the execution of each individual
        """
        results = []
        for ind in individuals:
            handle = open(self.path + "/results/results_" + str(ind.ind_idx) + "_" + str(generation) + ".bin", "rb")
            results.append((ind.ind_idx,pickle.load(handle)))
            handle.close()

        return results

    def run(self, trajectory, generation):
        """
        Takes care of running the generation by preparing the JUBE configuration files and, waiting for the execution
        by JUBE and gathering the results.
        This is the main function of the JUBE_runner
        :param trajectory: trajectory object storing individual parameters for each generation
        :param generation: id of the generation
        :return results: a list containing objects produced as results of the execution of each individual
        """
        args = []
        args.append("run")
        args.append(self.filename)
        self.done = False
        ready_files = []
        path_ready = self.path + "ready_files/ready_"
        self.prepare_run_file(path_ready)

        # Dump all trajectories for each optimizee run in the generation
        for ind in self.trajectory.individuals[generation]:
            trajectory.individual = ind
            handle = open(self.path + "/trajectories/trajectory_" + str(ind.ind_idx) + "_"+str(generation) + ".bin",
                          "wb")
            pickle.dump(trajectory, handle, pickle.HIGHEST_PROTOCOL)
            handle.close()
            ready_files.append(path_ready+str(ind.ind_idx))

        # Call the main function from JUBE
        logger.info("JUBE running generation: " + str(self.generation))
        main(args)

        # Wait for ready files to be written
        while not self.is_done(ready_files):
            time.sleep(5)

        # Touch done generation
        logger.info("JUBE finished generation: " + str(self.generation))
        f = open(self.path + "/ready_files/ready_w_" + str(generation), "w")
        f.close()

        self.done = True
        results = self.collect_results_from_run(generation, self.trajectory.individuals[generation])
        return results

    def is_done(self,files):
        """
        Identifies if all files marking the end of the execution of individuals in a generation are present or not.
        :param files: list of ready files to check
        :return true if all files are present, false otherwise
        """
        done = True
        for f in files:
            if not os.path.isfile(f):  # self.scheduler_config['ready_file']
                done = False
        return done

    def prepare_run_file(self, path_ready):
        """
        Writes a python run file which takes care of loading the optimizee from a binary file, the trajectory object
        of each individual. Then executes the 'simulate' function of the optimizee using the trajectory and
        writes the results in a binary file.
        :param path_ready: path to store the ready files
        :return true if all files are present, false otherwise
        """
        f = open(self.path + "/run_files/run_optimizee.py", "w")
        f.write("import pickle\n" +
                "import sys\n" +
                "idx = sys.argv[1]\n" +
                "iteration = sys.argv[2]\n" +
                "handle_trajectory = open(\"" + self.path + "/trajectories/trajectory_\" + str(idx) + \"_\" + "
                                                            "str(iteration) + \".bin\", \"rb\")\n" +
                "trajectory = pickle.load(handle_trajectory)\n" +
                "handle_trajectory.close()\n" +
                "handle_optimizee = open(\"" + self.path + "/optimizee.bin\", \"rb\")\n" +
                "optimizee = pickle.load(handle_optimizee)\n" +
                "handle_optimizee.close()\n\n" +
                "res = optimizee.simulate(trajectory)\n\n" +
                "handle_res = open(\"" + self.path + "/results/results_\" + str(idx) + \"_\" + str(iteration) + "
                                                     "\".bin\", \"wb\")\n" +
                "pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)\n" +
                "handle_res.close()\n\n" +
                "handle_res = open(\"" + path_ready + "\" + str(idx), \"wb\")\n" +
                "handle_res.close()")
        f.close()


def prepare_optimizee(optimizee, path):
    """
    Helper function used to dump the optimizee it a binary file for later loading during run.
    :param optimizee: the optimizee to dump into a binary file
    :param path: The path to store the optimizee.
    """
    # Serialize optimizee object so each process can run simulate on it independently on the CNs
    f = open(path+"/optimizee.bin","wb")
    pickle.dump(optimizee, f)
    f.close()
    logger.info("Serialized optimizee writen to path: " + path + "/optimizee.bin")


