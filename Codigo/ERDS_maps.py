# -*- coding: utf-8 -*-
"""
In this code the ERDS maps will be visualized for determining a suitable frequency and time for segmenting 
and filtering the data.

@author: Ali Abdul Ameer Abbas (with the help of the MNE documentation) 
"""


import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap
import pymatreader

#%% load and preprocess data ####################################################
path = '../Data/CLA/CLASubjectA1601083StLRHand.mat' 
path = '../Data/CLA/CLASubjectC1512233StLRHand.mat' 
#path = '../Data/CLA/CLASubjectC1512163StLRHand.mat' 
#path = '../Data/CLA/CLASubjectC1511263StLRHand.mat'
#path = '../Data/CLA/CLASubjectE1512253StLRHand.mat'
#path = '../Data/CLA/CLASubjectF1509163StLRHand.mat'

struct = pymatreader.read_mat(path, ignore_fields=['previous'], variable_names = 'o')
struct = struct['o']


chann = struct['chnames']
chann = chann[:-1] # pop th X3 channel, which is only used for the data adquisition. 
info = mne.create_info(ch_names = chann, 
                               sfreq = float(struct['sampFreq']),ch_types='eeg', montage='standard_1020', verbose=None)

data = np.array(struct['data'])
data = data[:,:-1]
data_V = data*1e-6 # mne reads units are in Volts

raw = mne.io.RawArray(data_V.transpose(),info, copy = "both")

raw = raw.set_eeg_reference('average', projection= False)

def type_pos(markers, sfreq = 200.0 ):
    """
        Gets the marker position. 
        Just left and right positions, NO PASS marker
        Only takes into account the right and left class, not the passive class.
        
        INPUTS:
            markers --> Markers.
            sfreq --> Sampling frequency.
        OUTPUTS: 
            mark --> markers.
            pos --> position of the marker.
            time --> time of the marker.        
    """
    mark = []
    pos = []
    time = []
    desc = ['left', 'right'] # for assigning the movements--> left =1, right = 2, pass = 3
    for i in range(len(markers)-1):
        if markers[i]==0 and markers[i+1] != 0 and (markers[i+1] in [1,2]):
            
            mark.append(desc[markers[i+1]-1])
            pos.append((i+2))
            time.append((i+2)/sfreq)
        else:
           continue
    return mark, pos, time

markers =  np.array(struct['marker']).transpose()
[mark, pos, time] = type_pos(markers)

annotations = mne.Annotations(time, 3.0, mark) # this is the annotation of every class
raw.set_annotations(annotations)

events, _ = mne.events_from_annotations(raw)

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])

#%% Get the epoch data ##################################################################
tmin, tmax = -1, 4  # define epochs around events (in s)
event_ids = dict(left=1, right=2)  # map event IDs to tasks

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)

#%% compute ERDS maps ##################################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.jet, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None)  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)

tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")

for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
            
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()