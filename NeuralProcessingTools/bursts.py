import numpy as np
from DataProcessingTools.objects import DPObject
from .spiketrain import Spiketrain
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
    filename = "spikebursts.mat"
    level = "cell"
    argsList = [("threshold", PercentileThreshold(10))]

    def create(self, *args, **kwargs):
        spiketrain = Spiketrain(*args, **kwargs)
        burst_idx, burst_length = find_spikebursts(spiketrain, self.args["threshold"]) 
        self.burst_idx = burst_idx
        self.burst_length = burst_length
        self.dirs = [os.getcwd()]
