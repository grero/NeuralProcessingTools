import NeuralProcessingTools as NPT
import DataProcessingTools as DPT 
import wget
import tempfile
import os
import numpy as np


def test_loading():
    tdir = tempfile.gettempdir()
    with DPT.misc.CWD(tdir):
        wget.download("http://cortex.nus.edu.sg/testdata/W2020106s02a01g001_lowpass.mat", "lowpass.mat")
        lfpdata = NPT.LFPData(loadFrom="lowpass.mat")
        assert len(lfpdata.data) == 8970762
        assert lfpdata.low_freq == 0.1
        assert lfpdata.high_freq == 300.0
        assert lfpdata.filter_name == "Butterworth"
        assert lfpdata.filter_order == 4

        # filter in the gamma band
        gamma_lfp = lfpdata.filter(20.0, 40.0)
        gamma_lfp.save()
        lfpdata2 = NPT.LFPData(loadFrom=gamma_lfp.get_filename())
        assert np.allclose(gamma_lfp.data, lfpdata2.data)
        os.unlink(gamma_lfp.get_filename())
        os.unlink("lowpass.mat")
        
