import DataProcessingTools as DPT
import NeuralProcessingTools as NPT
from NeuralProcessingTools.trialstructures import OldWorkingMemoryTrials
import numpy as np
import tempfile
import wget
import os


def test_oldtrials():
    tempdir = tempfile.gettempdir()
    with DPT.misc.CWD(tempdir):
        pth = "Pancake/20130923/session01"
        if not os.path.isdir(pth):
            os.makedirs(pth)
        
        with DPT.misc.CWD(pth):
            __dir__ = os.path.dirname(__file__)
            do_unlink = False
            if not os.path.isfile("event_data.mat"):
                wget.download("http://cortex.nus.edu.sg/testdata/J20140807_event_data.mat", "event_data.mat")
            trials = OldWorkingMemoryTrials()
            assert (trials.events == "trial_start").sum() == 934
            assert (trials.events == "reward_on").sum() == 369
            tidx = np.where(trials.events == "stimulus_on_1_(2, 2)")[0]
            assert tidx[0] == 2
            assert trials.ncols == 4
            assert trials.nrows == 4
            if do_unlink:
                os.unlink("event_data.mat")
