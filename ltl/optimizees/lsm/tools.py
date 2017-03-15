import numpy as np
import pylab
from pylab import find


def get_spike_times(spike_rec):
    """
   Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
   Each array has all spike times of one sender (neuron) in units of [sec]
    """
    import nest
    events = nest.GetStatus(spike_rec)[0]['events']
    min_idx = min(events['senders'])
    max_idx = max(events['senders'])
    spikes = []
    for i in range(min_idx, max_idx + 1):
        idx = find(events['senders'] == i)
        spikes.append(events['times'][idx] / 1000.0)  # convert times to [sec]
    return spikes


def cross_correlate_spikes(s1, s2, binsize, corr_range):
    """
    # Compute cross-correlation between two spike trains
    # The implementation is rather inefficient
    :param s1:
    :param s2:
    :param binsize:
    :param corr_range:
    :return:
    """

    cr_lo = corr_range[0]
    cr_hi = corr_range[1]
    ttt = corr_range[1] - corr_range[0]
    Nbins = np.ceil(ttt / binsize)
    Nbins_h = round(Nbins / 2)
    corr = np.zeros(Nbins + 1)
    s1a = np.append(s1, np.inf)
    for t in s2:
        idx = 0
        while s1a[idx] < t + cr_lo:
            idx += 1
        while s1a[idx] < t + cr_hi:
            idxc = round((t - s1a[idx]) / binsize) + Nbins_h
            corr[idxc] += 1
            idx += 1
    return corr


def avg_cross_correlate_spikes(spikes, num_pairs, binsize, corr_range):
    """
       computes average cross-crrelation between pairs of spike trains in spikes in the
       range defince by corr_range and with bin-size defined by binsize.
    """
    i = np.random.randint(len(spikes))
    j = np.random.randint(len(spikes))
    if i == j:
        j = (i + 1) % len(spikes)
    s1 = spikes[i]
    s2 = spikes[j]
    corr = cross_correlate_spikes(s1, s2, binsize, corr_range)
    for p in range(1, num_pairs):
        i = np.random.randint(len(spikes))
        j = np.random.randint(len(spikes))
        if i == j:
            j = (i + 1) % len(spikes)
        s1 = spikes[i]
        s2 = spikes[j]
        corr += cross_correlate_spikes(s1, s2, binsize, corr_range)
    return corr


def avg_cross_correlate_spikes_2sets(spikes1, spikes2, binsize, corr_range):
    s1 = spikes1[0]
    s2 = spikes2[0]
    corr = cross_correlate_spikes(s1, s2, binsize, corr_range)
    for i in range(1, len(spikes1)):
        for j in range(1, len(spikes2)):
            s1 = spikes1[i]
            s2 = spikes2[j]
            corr += cross_correlate_spikes(s1, s2, binsize, corr_range)
    return corr


def poisson_generator(rate, t_start=0.0, t_stop=1000.0, rng=None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    :param rate: the rate of the discharge (in Hz)
    :param t_start: the beginning of the SpikeTrain (in ms)
    :param t_stop: the end of the SpikeTrain (in ms)
    :param rng: A random number generator

    Examples:
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)
    """

    if rng is None:
        rng = np.random

    #number = int((t_stop-t_start)/1000.0*2.0*rate)

    # less wasteful than double length method above
    n = (t_stop - t_start) / 1000.0 * rate
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)

    if number > 0:
        isi = rng.exponential(1.0 / rate, int(number)) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun

        t_last = spikes[-1] + rng.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))

    else:
        spikes = np.resize(spikes, (i,))

    return spikes


def generate_stimuls_mem(dt_stim, stim_len, Rs, Tsim):
    """
    # Creates stimulus spikes for two input neurons
    # dt_stim...stimulus bursts come everey dt_stim ms
    # stim_len..length of stimulus burst in [ms]
    # Rs........rate of stimulus burst [Hz]
    # Tsim......simulation time [ms]
    # returns
    # spikes....s[i] spike times of i-th neuron [ms]
    """
    spikes = [np.array([]), np.array([])]
    Nstim = int(np.floor((Tsim - stim_len) / dt_stim))
    targets = np.random.randint(2, size=Nstim)
    t = dt_stim
    for rb in targets:
        spikes[rb] = np.append(spikes[rb], t + poisson_generator(Rs, t_stop=stim_len))
        t = t + dt_stim
    # round to simulation precision
    for i in range(len(spikes)):
        spikes[i] *= 10
        spikes[i] = spikes[i].round() + 1.0
        spikes[i] = spikes[i] / 10.0

    return spikes, targets


def generate_stimuls_xor(dt_stim, stim_len, Rs, Tsim):
    """
    # Creates stimulus spikes for two input neurons
    # dt_stim...stimulus bursts come everey dt_stim ms
    # stim_len..length of stimulus burst in [ms]
    # Rs........rate of stimulus burst [Hz]
    # Tsim......simulation time [ms]
    # returns
    # spikes....s[i] spike times of i-th neuron [ms]
    """
    spikes = [np.array([]), np.array([])]
    Nstim = int(np.floor((Tsim - stim_len) / dt_stim))
    in1 = np.random.randint(2, size=Nstim)
    in2 = np.random.randint(2, size=Nstim)
    t = dt_stim
    for rb in in1:
        if rb == 1:
            spikes[0] = np.append(spikes[0], t + poisson_generator(Rs, t_stop=stim_len))
        t = t + dt_stim
    t = dt_stim
    for rb in in2:
        if rb == 1:
            spikes[1] = np.append(spikes[1], t + poisson_generator(Rs, t_stop=stim_len))
        t = t + dt_stim
    # round to simulation precision
    for i in range(len(spikes)):
        spikes[i] *= 10
        spikes[i] = spikes[i].round() + 1.0
        spikes[i] = spikes[i] / 10.0
    targets_bin = np.logical_xor(in1, in2)
    targets = np.zeros(len(targets_bin))
    targets[targets_bin] = 1

    return spikes, targets


def get_liquid_states(spike_times, times, tau):
    """
    # returns the liquid states
    # spike_times[i]...numpy-array of spike-times of neuron i in [sec]
    # times............tunes when liquid states should be extracted [sec]
    # tau..............time constant for liquid state filter [sec]
    # returns:
    # states... numpy array with states[i,j] the state of neuron j in example i
    """
    N = np.size(spike_times, 0)
    T = np.size(times, 0)
    states = np.zeros((T, N))
    t_window = 3 * tau
    n = 0
    for spt in spike_times:
        spt2 = spt.__copy__()
        t_idx = T - 1
        for t in reversed(times):
            spt2 = spt2[spt2 < t]
            cur_times = spt2[spt2 >= t - t_window]
            states[t_idx, n] = sum(np.exp(-(t - cur_times) / tau))
            t_idx -= 1
        n += 1
    return states


def divide_train_test(states, targets, train_frac):
    """
    # divides liquid states and targets into
    # training set and test set
    # randomly chooses round(train_frac*len(targets)) exmaples for training, rest for testing
    # states... numpy array with states[i,j] the state of neuron j in example i
    # targets.. the targets for training/testing. targets[i] is target of example i
    # train fraction...fraction in (0,1) of training examples
    # returns:
    #    states_train..training states in same format as states
    #    states_test...test states in same format as states
    #    targets_train..training targets in same format as targets
    #    targets_test...test targets in same format as targets
    """
    Nstates = np.size(states, 0)
    Ntrain = round(Nstates * train_frac)
    Nstates - Ntrain
    idx_states = np.random.permutation(Nstates)
    idx_train = idx_states[:Ntrain]
    idx_test = idx_states[Ntrain:]
    states_train = states[idx_train, :]
    states_test = states[idx_test, :]
    targets_train = targets[idx_train]
    targets_test = targets[idx_test]
    return states_train, states_test, targets_train, targets_test


def train_readout(states, targets, reg_fact=0):
    """
    # train readout with linear regression
    # states... numpy array with states[i,j] the state of neuron j in example i
    # targets.. the targets for training/testing. targets[i] is target of example i
    # reg_fact..regularization factor. If set to 0, no regularization is performed
    # returns:
    #    w...weight vector
    """
    if reg_fact == 0:
        w = np.linalg.lstsq(states, targets)[0]
    else:
        w = np.dot(np.dot(pylab.inv(reg_fact * pylab.eye(np.size(states, 1)) + np.dot(states.T, states)), states.T),
                   targets)
    return w


def test_readout(w, states, targets):
    """
    # compute misclassification rate of linear readout with weights w
    # states... numpy array with states[i,j] the state of neuron j in example i
    # targets.. the targets for training/testing. targets[i] is target of example i
    # returns:
    #   err...the misclassification rate
    """
    yr = np.dot(states, w)  # compute prediction
    # compute error
    y = np.zeros(np.size(yr))
    y[yr >= 0.5] = 1
    err = (1. * sum(y != targets)) / len(targets)
    return err
