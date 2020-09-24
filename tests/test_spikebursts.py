import DataProcessingTools as DPT
import NeuralProcessingTools as NPT
import scipy.io as sio
import numpy as np
import os
import tempfile


def test_spikebursts():
    spiketrain = [0.1, 0.3, 0.4, 0.45, 0.46, 0.47, 0.7, 0.8, 0.81, 0.83, 0.85, 0.91]
    burst_idx, burst_length = NPT.bursts.find_spikebursts(spiketrain)
    assert burst_idx == [3, 7]
    assert burst_length == [3, 2]

def test_object():
    spiketrain = [0.1, 0.3, 0.4, 0.45, 0.46, 0.47, 0.7, 0.8, 0.81, 0.83, 0.85, 0.91]
    tempdir = tempfile.gettempdir()
    with DPT.misc.CWD(tempdir):
        cellpth = "Pancake/20130923/session01/array01/channel001/cell01"
        os.makedirs(cellpth)
        with DPT.misc.CWD(cellpth):
            sio.savemat("unit.mat",
                        {"timestamps": spiketrain,
                         "spikeForm": [0.0]})
            spikebursts = NPT.SpikeBursts(saveLevel=1)
            assert os.path.isfile(spikebursts.get_filename())
            assert np.allclose(spikebursts.burst_start, [0.45, 0.8])
            assert np.allclose(spikebursts.burst_length, [0.02, 0.01])
            assert spikebursts.burst_rate == 1000*2/0.81
            os.remove(spikebursts.get_filename())
            os.remove("unit.mat")

        os.removedirs(cellpth)

