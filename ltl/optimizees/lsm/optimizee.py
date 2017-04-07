import logging
from enum import Enum

import numpy as np

import nest
import nest.raster_plot
from ltl import sdict

from ltl.optimizees.lsm.tools import get_spike_times, get_liquid_states, train_readout, \
    test_readout, divide_train_test, generate_stimuls_xor, plot_spiketrains, generate_stimuls_mem
from ltl.optimizees.optimizee import Optimizee

logger = logging.getLogger("ltl-lsm")


class Tasks(Enum):
    XOR = 1
    FADING_MEMORY = 2


class LSMOptimizee(Optimizee):
    def __init__(self, traj, task, n_NEST_threads=1):
        super().__init__(traj)

        assert task in Tasks
        self.task = task

        self.n_NEST_threads = n_NEST_threads
        self._initialize()

        # create_individual can be called because __init__ is complete except for traj initialization
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)

    def _initialize(self):
        # Set parameters of the NEST simulation kernel
        nest.SetKernelStatus({'print_time': False,
                              'local_num_threads': self.n_NEST_threads})

        # dynamic parameters
        f0 = 10.

        def get_u_0(U, D, F):
            return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))

        def get_x_0(U, D, F):
            return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))

        syn_param_EE = {"tau_psc": 2.0,
                        "tau_fac": 1.,  # facilitation time constant in ms
                        "tau_rec": 813.,  # recovery time constant in ms
                        "U": 0.59,  # utilization
                        "u": get_u_0(0.59, 813., 1.),
                        "x": get_x_0(0.59, 813., 1.),
                        }
        nest.CopyModel("tsodyks_synapse", "EE", syn_param_EE)  # synapse model for E->E connections

        syn_param_EI = {"tau_psc": 2.0,
                        "tau_fac": 1790.,  # facilitation time constant in ms
                        "tau_rec": 399.,  # recovery time constant in ms
                        "U": 0.049,  # utilization
                        "u": get_u_0(0.049, 399., 1790.),
                        "x": get_x_0(0.049, 399., 1790.),
                        }
        nest.CopyModel("tsodyks_synapse", "EI", syn_param_EI)  # synapse model for E->I connections

        syn_param_IE = {"tau_psc": 2.0,
                        "tau_fac": 376.,  # facilitation time constant in ms
                        "tau_rec": 45.,  # recovery time constant in ms
                        "U": 0.016,  # utilization
                        "u": get_u_0(0.016, 45., 376.),
                        "x": get_x_0(0.016, 45., 376.),
                        }
        nest.CopyModel("tsodyks_synapse", "IE", syn_param_IE)  # synapse model for I->E connections

        syn_param_II = {"tau_psc": 2.0,
                        "tau_fac": 21.,  # facilitation time constant in ms
                        "tau_rec": 706.,  # recovery time constant in ms
                        "U": 0.25,  # utilization
                        "u": get_u_0(0.25, 706., 21.),
                        "x": get_x_0(0.25, 706., 21.),
                        }
        nest.CopyModel("tsodyks_synapse", "II", syn_param_II)  # synapse model for I->I connections

        J_noise = 1.0  # strength of synapses from noise input [pA]
        nest.CopyModel('static_synapse_hom_w',
                       'excitatory_noise',
                       {'weight': J_noise})

    def create_individual(self):
        jee, jei, jie, jii = np.random.randint(1, 20, 4).astype(np.float64)
        return dict(jee=jee, jei=jei, jie=jie, jii=jii)

    def bounding_func(self, individual):
        individual = {key: np.float64(value if value > 0.01 else 0.01) for key, value in individual.items()}
        return individual

    def simulate(self, traj, should_plot=False, debug=False):

        jee = traj.individual.jee
        jei = traj.individual.jei
        jie = traj.individual.jie
        jii = traj.individual.jii

        if jee < 0 or jei < 0 or jie < 0 or jii < 0:
            return (np.inf,)

        logger.info("Running for %.2f, %.2f, %.2f, %.2f", jee, jei, jie, jii)

        simtime_ms = 200000.  # how long shall we simulate [ms]
        if debug:
            simtime_ms = 20000.

        N_rec = 500  # Number of neurons to record from

        # Network parameters.
        delay_dict = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)

        N_E = 1000  # 2000  # number of excitatory neurons
        N_I = 250  # 500  # number of inhibitory neurons
        N_neurons = N_E + N_I  # total number of neurons

        C_E = 2  # int(N_E / 20)  # number of excitatory synapses per neuron
        C_I = 1  # int(N_E / 20)  # number of inhibitory synapses per neuron
        C_inp = 100  # int(N_E / 20)  # number of outgoing input synapses per input neuron

        w_scale = 10.0
        J_EE = w_scale * jee  # strength of E->E synapses [pA]
        J_EI = w_scale * jei  # strength of E->I synapses [pA]
        J_IE = w_scale * -jie  # strength of inhibitory synapses [pA]
        J_II = w_scale * -jii  # strength of inhibitory synapses [pA]

        p_rate = 100.0  # this is used to simulate input from neurons around the populations

        # Create nodes -------------------------------------------------
        nest.ResetKernel()
        self._initialize()
        nest.SetDefaults('iaf_psc_exp',
                         {'C_m': 30.0,  # 1.0,
                          'tau_m': 30.0,
                          'E_L': 0.0,
                          'V_th': 15.0,
                          'tau_syn_ex': 3.0,
                          'tau_syn_in': 2.0,
                          'V_reset': 13.8})

        # Create excitatory and inhibitory populations
        nodes = nest.Create('iaf_psc_exp', N_neurons)

        nest.SetStatus(nodes,
                       [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])
        nodes_E = nodes[:N_E]
        nodes_I = nodes[N_E:]

        # create spike detectors from excitatory and inhibitory populations
        spikes = nest.Create('spike_detector', 2,
                             [{'label': 'ex_spd'},
                              {'label': 'in_spd'}])
        spikes_E = spikes[:1]
        spikes_I = spikes[1:]

        # create input generators
        dt_stim_ms = 300.  #[ms]
        stim_len_ms = 50.  #[ms]
        Rs = 200.  #[Hz]

        if self.task == Tasks.XOR:
            inp_spikes, targets = generate_stimuls_xor(dt_stim_ms, stim_len_ms, Rs, simtime_ms)
            readout_delay = 0.030  # [sec]
        elif self.task == Tasks.FADING_MEMORY:
            inp_spikes, targets = generate_stimuls_mem(dt_stim_ms, stim_len_ms, Rs, simtime_ms)
            readout_delay = (dt_stim_ms - stim_len_ms - 1) * 1e-3  # [sec]
        else:
            raise RuntimeError("Unknown task {}".format(self.task.name))

        # create two spike generators,
        # set their spike_times of i-th generator to inp_spikes[i]
        spike_generators = nest.Create("spike_generator", 2)
        for (sg, sp) in zip(spike_generators, inp_spikes):
            nest.SetStatus([sg], {'spike_times': sp})

        # Connect nodes ------------------------------------------------
        # connect E to E with excitatory synapse model and fixed indegree C_E
        nest.Connect(nodes_E, nodes_E,
                     {'rule': 'fixed_indegree',
                      'indegree': C_E},
                     {'model': 'EE',
                      'delay': delay_dict,
                      "weight": {"distribution": "normal", "mu": J_EE, "sigma": 0.7 * J_EE},
                      })

        # connect E to I with static synapse model and fixed indegree C_E
        # Set weights uniformly in [0.5*J_EI, 1.5*J_EI]
        nest.Connect(nodes_E, nodes_I,
                     {'rule': 'fixed_indegree',
                      'indegree': C_E},
                     {'model': 'EI',
                      'delay': delay_dict,
                      "weight": {"distribution": "normal", "mu": J_EI, "sigma": 0.7 * J_EI},
                      })

        # connect I to E with static synapse model and fixed indegree C_E
        nest.Connect(nodes_I, nodes_E,
                     {'rule': 'fixed_indegree',
                      'indegree': C_I},
                     {'model': 'IE',
                      'delay': delay_dict,
                      "weight": {"distribution": "normal", "mu": J_IE, "sigma": 0.7 * abs(J_IE)},
                      })

        # connect I to I with static synapse model and fixed indegree C_E
        nest.Connect(nodes_I, nodes_I,
                     {'rule': 'fixed_indegree',
                      'indegree': C_I},
                     {'model': 'II',
                      'delay': delay_dict,
                      "weight": {"distribution": "normal", "mu": J_II, "sigma": 0.7 * abs(J_II)},
                      })

        # connect one noise generator to all neurons
        # Create noise input
        noise = nest.Create('poisson_generator', 1, {'rate': p_rate})
        nest.Connect(noise, nodes, syn_spec={'model': 'excitatory_noise', 'delay': delay_dict})

        # connect input neurons to E-pool
        # Each input neuron makes C_input synapses
        # distribute weights uniformly in (2.5*J_EE, 7.5*J_EE)

        INPUT_ON = True
        if INPUT_ON:
            nest.Connect(spike_generators, nodes,
                         {'rule': 'fixed_outdegree',
                          'outdegree': C_inp},
                         {'model': 'static_synapse',
                          'delay': delay_dict,
                          'weight': {'distribution': 'uniform',
                                     'low': 2.5 * J_EE,
                                     'high': 7.5 * J_EE}})

        # connect all recorded E/I neurons to the respective detector
        nest.Connect(nodes_E[:N_rec], spikes_E)
        nest.Connect(nodes_I[:N_rec], spikes_I)

        # SIMULATE!! -----------------------------------------------------
        nest.Simulate(simtime_ms)

        #compute excitatory rate
        events = nest.GetStatus(spikes, 'n_events')
        rate_ex = events[0] / simtime_ms * 1000.0 / N_rec
        logger.debug('Excitatory rate   : %.2f Hz', rate_ex)

        #compute inhibitory rate
        rate_in = events[1] / simtime_ms * 1000.0 / N_rec
        logger.debug('Inhibitory rate   : %.2f Hz', rate_in)

        spike_times_s = spike_times_s_E = get_spike_times(spikes_E)  # returns spike times in seconds
        spike_times_s_I = get_spike_times(spikes_I)  # returns spike times in seconds
        if debug or should_plot:
            num_stims_to_plot = 2
            plot_spiketrains(spike_times_s_E, spike_times_s_I, [0, num_stims_to_plot * dt_stim_ms * 1e-3],
                             'raster-start.png')
            plot_spiketrains(spike_times_s_E, spike_times_s_I,
                             [(simtime_ms - num_stims_to_plot * dt_stim_ms) * 1e-3, simtime_ms * 1e-3],
                             'raster-end.png')

            if debug:
                return (0,)

        # train the readout on 20 randomly chosen training sets
        NUM_TRAIN = 30

        tau_lsm = 0.020  #[sec]

        rec_time_start = (dt_stim_ms / 1000 + stim_len_ms / 1000 + readout_delay)  # time of first liquid state [sec]
        # times when liquid states are extracted [sec]
        times = np.arange(rec_time_start, simtime_ms / 1000, dt_stim_ms / 1000)
        logger.debug("Extract Liquid States...")
        states = get_liquid_states(spike_times_s, times, tau_lsm)
        states = states[5:, :]  # disregard first 5 stimuli
        targets = targets[5:]
        Nstates = np.size(states, 0)
        # add constant component to states for bias
        states = np.hstack([states, np.ones((Nstates, 1))])
        train_frac = 0.8
        err_train = np.zeros(NUM_TRAIN)
        err_test = np.zeros(NUM_TRAIN)
        logger.debug("Computing Least Squares...")
        reg_const = 5.0
        for trial in range(NUM_TRAIN):
            states_train, states_test, targets_train, targets_test = divide_train_test(states, targets, train_frac)
            # compute least squares solution
            w = train_readout(states_train, targets_train, reg_fact=reg_const)
            err_train[trial] = test_readout(w, states_train, targets_train)
            err_test[trial] = test_readout(w, states_test, targets_test)
        training_error_mean, training_error_std = np.mean(err_train) * 100, np.std(err_train * 100)
        testing_error_mean, testing_error_std = np.mean(err_test) * 100, np.std(err_test * 100)

        logger.info("Done running. Training error was %f %% +- %f and Testing error was %f %% +- %f "
                    "for weights %.2f, %.2f, %.2f, %.2f",
                    training_error_mean, training_error_std, testing_error_mean, testing_error_std, jee, jei, jie, jii)
        return (testing_error_mean,)

    def end(self):
        logger.info("End of all experiments. Cleaning up...")
        # There's nothing to clean up though


def main():
    import yaml
    import os
    import logging.config

    from ltl import DummyTrajectory
    from ltl.paths import Paths
    from ltl import timed

    # TODO: Set root_dir_path here
    paths = Paths('ltl-lc-fading-memory', dict(run_num='test'), root_dir_path=None)
    with open("bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    fake_traj = DummyTrajectory()
    optimizee = LSMOptimizee(fake_traj, task=Tasks.XOR, n_NEST_threads=15)

    fake_traj.individual = sdict(optimizee.create_individual())

    with timed(logger):
        testing_error = optimizee.simulate(fake_traj)
    logger.info("Testing error is %s", testing_error)


if __name__ == "__main__":
    main()
