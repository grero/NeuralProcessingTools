import DataProcessingTools as DPT
import NeuralProcessingTools as NPT
from NeuralProcessingTools.trialstructures import OldWorkingMemoryTrials
import tempfile
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
                os.link(os.path.join(__dir__, "event_data.mat"), "event_data.mat")
                do_unlink = True
            trials = OldWorkingMemoryTrials()
            assert (trials.events == "trial_start").sum() == 1708
            assert (trials.events == "reward_on").sum() == 621
            if do_unlink:
                os.unlink("event_data.mat")
