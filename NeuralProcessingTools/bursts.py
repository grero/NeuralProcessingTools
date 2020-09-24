import numpy as np
from DataProcessingTools.objects import DPObject
from .spiketrain import Spiketrain
import scipy.special as special
import os


class PercentileThreshold():
    def __init__(self, value):
        self._value = value

    def value(self, x):
        return np.percentile(x, self._value)


class AbsoluteThreshold():
    def __init__(self, value):
        self._value = value

    def value(self, x):
        return self._value


class LogNormalThreshold():
    def __init__(self, value):
        self._value = value

    def value(self, x):
        y = np.log(x)
        mu = y.mean()
        sigma = y.std()
        p = self._value
        # quantile function for a log-normal distribution
        qtl = np.exp(mu + np.sqrt(2)*sigma*special.erfinv(2*p-1))
        return qtl

def find_spikebursts(spiketrain, threshold=PercentileThreshold(10)):
    isi = np.diff(spiketrain)
    t = threshold.value(isi)
    burst_idx = []
    burst_length = []
    j = 0
    in_burst = False
    for i in range(len(spiketrain)-1):
        if spiketrain[i+1] - spiketrain[i] <= t:
            in_burst = True
            if len(burst_idx) <= j:
                burst_idx.append(i)
                burst_length.append(1)
            burst_length[j] += 1
        else:
            if in_burst:
                j += 1
                in_burst = False

    return burst_idx, burst_length


class SpikeBursts(DPObject):
    filename = "spikebursts.hkl"
    level = "cell"
    argsList = [("threshold", PercentileThreshold(10))]

    def create(self, *args, **kwargs):
        spiketrain = Spiketrain(*args, **kwargs)
        if spiketrain.dirs == []:
            self.burst_start = np.array((),dtype=np.float64)
            self.burst_length = np.array((),dtype=np.float64)
            self.burst_rate = 0.0
            self.spikes_in_burst = 0.0
            self.dirs = []
            self.setdidx = []
        else:
            spiketimes = spiketrain.timestamps.flatten()
            burst_idx, burst_length = find_spikebursts(spiketimes,
                                                       self.args["threshold"])
            self.burst_start = spiketimes[burst_idx]
            bidx = np.array(burst_idx, dtype=np.uint64)
            blen = np.array(burst_length, dtype=np.uint64)
            self.burst_length = (spiketimes[bidx + blen-1] -
                                 spiketimes[bidx])
            # burst rate in bursts per second
            self.burst_rate = len(burst_idx)/(spiketimes[-1] - spiketimes[0])
            self.burst_rate *= 1000
            self.spikes_in_burst = blen.sum()/len(spiketimes)
            self.dirs = [os.getcwd()]
            self.setidx = [0 for i in range(len(burst_length))]

    def append(self, bursts):
        DPObject.append(self, bursts)
        self.spikes_in_burst = np.append(np.atleast_1d(self.spikes_in_burst), 
                                         bursts.spikes_in_burst)
        self.burst_rate = np.append(np.atleast_1d(self.burst_rate),[bursts.burst_rate])
        self.burst_start = np.append(self.burst_start, bursts.burst_start)
        self.burst_length = np.append(self.burst_length, bursts.burst_length)
