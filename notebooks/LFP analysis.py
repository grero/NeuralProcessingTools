# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: py38
# ---

# ## Local field potential analysis

# %matplotlib widget

import DataProcessingTools as DPT
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal

# We first load some already low-pass filtered data from one channel recorded from Whiskey

datadir = "/Volumes/FastData/data/workingMemory/Whiskey/20200106/session02/array01/channel001"
with DPT.misc.CWD(datadir):
    lfpdata = DPT.LFPData(loadFrom="lowpass.mat")

# +

lfpdata.plot()
# -

# This data contains frequencies from 0.1 to 300 Hz, as you can see from the parameters used to create the data

lfpdata.low_freq, lfpdata.high_freq

# If we want to look at e.g. the gamma band, which typically runs from 20 to 40 Hz, we can filter the data again

gamma_data = lfpdata.filter(20.0, 40.0)
gamma_data.plot()

# ### Aligning to trial events
#
# First, let us load the trial structure for the same recording session

with DPT.misc.CWD(datadir):
    trials = DPT.trialstructures.WorkingMemoryTrials()

# We can now align the LFPs to the onset of the first stimulus of each correct trial.

rewardOnset, cidx, stimIdx = trials.get_timestamps("reward_on")
alignto, stimidx, trialLabel = trials.get_stim(0, cidx)
alignto = 1000*np.array(alignto)
ntrials = len(alignto)
pre_window = 100
post_window = 1000
aligned_lfp = np.zeros((pre_window+post_window, ntrials))
for i in range(ntrials):
    idx_pre = int(alignto[i]-100)
    idx_post = int(alignto[i]+ 1000)
    aligned_lfp[:,i] = lfpdata.data[idx_pre:idx_post]                    

# Plot a trial aligned to the stimulus onset.

fig = plt.figure()
ax = fig.add_subplot(111)
# indicate stimulus onset, from 0 to 300ms
ax.axvspan(0, 300, color=(0.8, 0.8, 0.8, 0.3))
ax.plot(aligned_lfp[:,5])


# ### Spectral content
# We can also look at the spectral content of the signal

Sxx = np.zeros((65,9, aligned_lfp.shape[0]))
for i in range(aligned_lfp.shape[0]):
    f, t, Sxx[:,:,i] = signal.spectrogram(aligned_lfp[:,2], fs=lfpdata.sampling_rate, nperseg=128)


fig = plt.figure()
ax = fig.add_subplot(111)
# only look at frequencies below 100Hz
idx = f < 100
img = ax.pcolormesh(t-0.1, f[idx], np.log10(Sxx[idx,:,:].mean(2)))
plt.colorbar(img, label="Power [dB]")

Sxx.shape


# ## Test case
# Let's construct a simple test case to convince ourselves that the code does what it is supposed to do.
#
# We use the trial markers above to simulate LFP responses where the onset of the target induces a strong, transient modulation of the gamma band. That is, we let the power of the frequency band from 20 to 40 Hz follow a double exponential (fast rise, slow fall) after target onset.

# +
def window(t, t0, a,b):
    if t < t0:
        return 0.0
    return np.exp(-(t-t0)/a)*(1-np.exp(-(t-t0)/b))

window(1.0, 0.0, 0.3, 0.4)
# -

t = np.linspace(0.0, 9.0, 900)
y = [1.0 + 20*window(t, 0.0, 0.2, 0.05) for t in t]
tp = np.linspace(-100.0, 1000.0,1100)
yp = np.sin(2*np.pi*tp/(1.0/40.0))
yp[200:] = yp[200:]*y
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tp,yp)

# +
ts = np.arange(0.0, len(lfpdata.data)-1, 1.0)
sdata = np.sin(2*np.pi*ts/(1.0/40.0)/1000.0)
for i in range(len(alignto)):
    idx = int(alignto[i])
    dd = sdata[idx+100:idx+1000]
    sdata[idx+100:idx+1000] = dd*y

# add more spectral content
# draw frequencies from 1/f distribution
# maximum frequency is the Nyquist frequency, i.e. 500Hz
sfreqs = np.linspace(0.01, 500.0, 500)
cdf = 1.0/sfreqs
scdf = np.sort(cdf)
ffi = np.random.random(100)
freqs = np.zeros((100,))
for i in range(100):
    idx = np.searchsorted(scdf, ffi[i])
    freqs[i] = sfreqs[-idx]

# random phases
phase = 2*np.pi*np.random.random(100)

# add in the extra spectral content with a scaling factor to 
# make the 40Hz component added above stand out.
for i in range(100):
    q = np.sin(2*np.pi*ts/(1.0/freqs[i])/1000.0 + phase[i])/12.5
    sdata += q

# -

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sdata[30000:40000])

# We can now re-align to the stimulus onset events and look at the spectral content

aligned_lfp = np.zeros((1100,ntrials))
for i in range(ntrials):
    idx_pre = int(alignto[i]-100)
    idx_post = int(alignto[i]+ 1000)
    aligned_lfp[:,i] = sdata[idx_pre:idx_post]    

# check the gamma band of a single trial
b,a = signal.butter(4, [30.0/1000, 50.0/1000], btype='bandpass')
yy = signal.filtfilt(b,a, aligned_lfp[:,5])

Sxx = np.zeros((65,9, aligned_lfp.shape[0]))
for i in range(aligned_lfp.shape[0]):
    f, t, Sxx[:,:,i] = signal.spectrogram(aligned_lfp[:,2], fs=lfpdata.sampling_rate, nperseg=128)

fig = plt.figure()
ax1 = fig.add_subplot(211)
tq = np.arange(-0.1, 1.0, 0.001)
# indicate stimulus onset, from 0 to 300ms
ax1.axvspan(0, 0.3, color=(0.8, 0.8, 0.8, 0.3))
ax1.plot(tq, aligned_lfp[:,5])
ax3 = ax1.twinx()
ax3.plot(tq, yy, color="red")
ax2 = fig.add_subplot(212, sharex=ax1)
# only look at frequencies below 100Hz
idx = f < 100
img = ax2.pcolormesh(t-0.1, f[idx], np.log10(Sxx[idx,:,:].mean(2)))
plt.colorbar(img, label="Power [dB]", ax=ax2, orientation='horizontal')

len(alignto)

len(lfpdata.data)

lfpdata.filter_name

gamma_data.filter_coefs

len(1)


