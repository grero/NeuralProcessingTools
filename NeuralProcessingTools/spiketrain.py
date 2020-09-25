import DataProcessingTools
from DataProcessingTools.objects import DPObject
import numpy as np
import scipy.io as sio
import os


class Spiketrain(DPObject):
    """
    Spiketrain()
    """
    level = "cell"
    filename = "unit.mat"

    def __init__(self, *args, **kwargs):
        kwargs["redoLevel"] = 0
        DPObject.__init__(self, *args, **kwargs)
        # always load since we do not create spike trains here.
        if os.path.isfile(self.filename):
            self.load()
            self.dirs = [os.getcwd()]
        else:
            self.dirs = []
            self.timestamps = np.array((), dtype=np.float64)
            self.spikeshape = np.array((), dtype=np.float64)

    def load(self, fname=None):
        q = sio.loadmat(self.filename)
        self.timestamps = q["timestamps"]
        self.spikeshape = q["spikeForm"]

    def get_filename(self):
        return self.filename
