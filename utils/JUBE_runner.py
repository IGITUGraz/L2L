from JUBE.jube2.main import main
import os.path
import pickle
import time

class JUBERunner():

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.done = False
        if not 'JUBE_params' in self.trajectory.par.keys():
            print("exception")
        else:
            args = self.trajectory.par["JUBE_params"].params

        self._prefix = args.get('fileprefix',"")
        self.scheduler = "None"
        self.scheduler_config = {}
        print(args.keys())
        self.scheduler_config = {
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
        }
        if 'scheduler' in args.keys():
            self.scheduler = args.get('scheduler', "None"),
        self.exec =  args['exec']
        self.filename = ""
        self.path = args['work_path']


    def set_prefix(self, fprefix):
        self._prefix = fprefix

    def set_iteration(self,iteration):
        self._iteration = iteration

    def write_pop_for_jube(self, trajectory, generation):
        """
        Writes an XML file which contains the parameters for JUBE
        """
        self.trajectory = trajectory
        eval_pop = trajectory.individuals[generation]
        self.generation = generation
        self.filename = self._prefix + "_jube_" + str(self.generation) + ".xml"
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
        f.write("    <parameter name=\"submit_cmd\">" + self.scheduler_config['submit_cmd'] + "</parameter>\n")
        f.write("    <parameter name=\"job_file\">" + self.scheduler_config['job_file'] + "$index " + str(
                generation)+ "</parameter>\n")
        f.write("    <parameter name=\"nodes\" type=\"int\">" + self.scheduler_config['nodes'] + "</parameter>\n")
        f.write("    <parameter name=\"walltime\">" + self.scheduler_config['walltime'] + "</parameter>\n")
        f.write("    <parameter name=\"ppn\" type=\"int\">" + self.scheduler_config['ppn'] + "</parameter>\n")
        f.write("    <parameter name=\"ready_file_scheduler\" mode=\"python\" type=\"string\"> /ready_files/ready_ + ${index} + </parameter>\n")
        f.write("    <parameter name=\"ready_file\">" + self.scheduler_config['ready_file'] + str(
            self.generation) + "</parameter>\n")
        f.write("    <parameter name=\"mail_mode\">" + self.scheduler_config['mail_mode'] + "</parameter>\n")
        f.write("    <parameter name=\"mail_address\">" + self.scheduler_config['mail_address'] + "</parameter>\n")
        f.write("    <parameter name=\"err_file\">" + self.scheduler_config['err_file'] + "</parameter>\n")
        f.write("    <parameter name=\"out_file\">" + self.scheduler_config['out_file'] + "</parameter>\n")
        f.write("    <parameter name=\"exec\">" + self.exec + "</parameter>\n")
        f.write("    </parameterset>\n")

        # Write the scheduler attributes
        if self.scheduler != 'None':
            self.write_scheduler_file(f)

        f.write("    <!-- Operation -->\n")
        f.write("    <step name=\"submit\" work_dir=\""+ self.path +"/work/jobsystem_bench_${jube_benchmark_id}_${jube_wp_id}\" >\n")
        f.write("    <use>ltl_parameters</use>\n")
        f.write("    <use>execute_set</use>\n")

        if self.scheduler != 'None':
            f.write("    <use>files,sub_job</use>\n")
            f.write("    <do done_file=\""+self.path + "/ready_files/ready_w_" + str(self.generation) + "\">$submit_cmd $job_file </do> <!-- shell command -->\n")
        else:
            #f.write("    <do done_file=\""+self.path + "/ready_files/ready_w_" + str(self.generation) + "\">$exec $index " + str(
            #    generation) + " </do> <!-- shell command -->\n")
            f.write("    <do done_file=\"" + self.path + "/ready_files/ready_w_" + str(
                self.generation) + "\">$exec $index " + str(self.generation) + " </do> <!-- shell command -->\n")

        f.write("    </step>   \n")

        # Close
        f.write("  </benchmark>\n")
        f.write("</jube>\n")
        f.close()

    def write_scheduler_file(self, f):
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
        f.write("    <sub source=\"#EXEC#\" dest=\"$exec\" />\n")
        f.write("    <sub source=\"#READY#\" dest=\"$ready_file" + str(self.generation) + "\" />\n")
        f.write("    </substituteset> \n")

    def collect_results_from_run(self, iteration, individuals):
        results = []
        for ind in individuals:
            handle = open(self.path + "/results/results_" + str(ind.ind_idx) + "_" + str(iteration) + ".bin", "rb")
            results.append((ind.ind_idx,pickle.load(handle)))
            handle.close()

        return results

    def run(self, trajectory, iteration):
        args = []
        args.append("run")
        args.append(self.filename)
        self.done = False
        ready_files = []
        path_ready = self.path + "/ready_files/ready_"
        self.prepare_run_file(path_ready)
        # Dump all trajectories for each optimizee run in the generation
        for ind in self.trajectory.individuals[iteration]:
            trajectory.individual = ind
            handle = open(self.path + "/trajectories/trajectory_"+str(ind.ind_idx)+"_"+str(iteration)+".bin", "wb")
            pickle.dump(trajectory, handle, pickle.HIGHEST_PROTOCOL)
            handle.close()
            ready_files.append(path_ready+str(ind.ind_idx))
        main(args)
        while not self.is_done(ready_files):
            time.sleep(5)
        #Touch done generation
        f = open(self.path + "/ready_files/ready_w_" + str(iteration), "w")
        f.close()
        self.done = True
        results = self.collect_results_from_run(iteration, self.trajectory.individuals[iteration])
        return results

    def is_done(self,files):
        done = True
        for f in files:
            if not os.path.isfile(f):  # self.scheduler_config['ready_file']
                done = False
        return done

    def prepare_run_file(self, path_ready):
        f = open(self.path + "/run_files/run_optimizee.py", "w")
        f.write("import pickle\n" +
                "import sys\n" +
                "idx = sys.argv[1]\n" +
                "iteration = sys.argv[2]\n" +
                "handle_trajectory = open(\"" + self.path + "/trajectories/trajectory_\" + str(idx) + \"_\" + str(iteration) + \".bin\", \"rb\")\n" +
                "trajectory = pickle.load(handle_trajectory)\n" +
                "handle_trajectory.close()\n" +
                "handle_optimizee = open(\"" + self.path + "/optimizee.bin\", \"rb\")\n" +
                "optimizee = pickle.load(handle_optimizee)\n" +
                "handle_optimizee.close()\n\n" +
                "res = optimizee.simulate(trajectory)\n\n" +
                "handle_res = open(\"" + self.path + "/results/results_\" + str(idx) + \"_\" + str(iteration) + \".bin\", \"wb\")\n" +
                "pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)\n" +
                "handle_res.close()\n\n" +
                "#Touch ready file\n" +
                "handle_res = open(\"" + path_ready + "\" + str(idx), \"wb\")\n" +
                "handle_res.close()")
        f.close()
        return path_ready

def prepare_optimizee(optimizee, path):
    # Serialize optimizee object so each process can run simulate on it independently on the CNs
    f = open(path+"/optimizee.bin","wb")
    pickle.dump(optimizee, f)
    f.close()


