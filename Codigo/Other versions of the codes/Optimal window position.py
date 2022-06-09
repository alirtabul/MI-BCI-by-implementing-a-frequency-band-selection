# -*- coding: utf-8 -*-
"""
Find the optimal time window for each session

"""
# import the libraries
import mne
import matplotlib # IMPORTANT--> It must be version 3.2.2
import matplotlib.pyplot as plt
import pylab
import pymatreader 
import pandas as pd
import numpy as np
import scipy.stats

path1 = '../Data/CLA/CLASubjectC1512233StLRHand.mat' # best time 2.1  3.9
path2 = '../Data/CLA/CLASubjectC1512163StLRHand.mat' # best time 2.1  3.9
path3 = '../Data/CLA/CLASubjectC1511263StLRHand.mat'
fs= 200


path1_B = '../Data/CLA/CLASubjectB1510193StLRHand.mat' # best time 2.1  3.9
path2_B = '../Data/CLA/CLASubjectB1510203StLRHand.mat' # best time 2.1  3.9
path3_B = '../Data/CLA/CLASubjectB1512153StLRHand.mat'


path1_E = '../Data/CLA/CLASubjectE1512253StLRHand.mat' # best time 2.1  3.9
path2_E = '../Data/CLA/CLASubjectE1601193StLRHand.mat' # best time 2.1  3.9
path3_E = '../Data/CLA/CLASubjectE1601223StLRHand.mat'

path1_F = '../Data/CLA/CLASubjectF1509163StLRHand.mat' # best time 2.1  3.9
path2_F = '../Data/CLA/CLASubjectF1509173StLRHand.mat' # best time 2.1  3.9
path3_F = '../Data/CLA/CLASubjectF1509283StLRHand.mat'

# Now we simply pass the matlab files into a python compatible file.
struct = pymatreader.read_mat(path3_F, ignore_fields=['previous'], variable_names = 'o')
struct = struct['o']

chann = struct['chnames'] # let's save the channel names
chann = chann[:-1] # pop th X3 channel, which is only used for the data adquisition. 
#Now we save important information from the dataset.
info = mne.create_info(ch_names = chann, 
                               sfreq = float(struct['sampFreq']),ch_types='eeg', montage='standard_1020', verbose=None)
# we create a data variable. 
data = np.array(struct['data'])
data = data[:,:-1]
data_V = data*1e-6 # mne reads units are in Volts

# Lets plot the raw data!
# To plot it in a separate window use %matplotlib qt, to plot it in the same window %matplotlib inline
raw = mne.io.RawArray(data_V.transpose(),info, copy = "both")

# We filter the signal for motor imagery anlysis
raw = raw.set_eeg_reference('average', projection= False)
raw.filter(18., 26., fir_design='firwin', skip_by_annotation='edge') # normally 7-30 Hz

#%% lets create annotations with markers:


def type_pos_pas(markers, sfreq = 200.0 ):
    """
    Pass mark is gathered too.
    
    """
    
    mark = []
    pos = []
    time = []
    desc = ['left', 'right', 'pass'] # for assigning the movements--> left =1, right = 2, pass = 3
    for i in range(len(markers)-1):
        if markers[i]==0 and markers[i+1] != 0 and (markers[i+1] in [1,2,3]):
            
            mark.append(desc[markers[i+1]-1])
            pos.append((i+2))
            time.append((i+2)/sfreq)
        else:
           continue
    return mark, pos, time

def type_pos(markers, sfreq = 200.0 ):
    """
    Just left and right positions, NO PASS marker
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
[mark, pos, time] = type_pos_pas(markers)

# We create a function which will separate left/right marks from pass,
# left/pass marks  from right, and right/pass marks from left.

def multi_annotations(marks, time, window_time = 1.0):
    mark_left_right = []
    time_left_right = []
    for mar in range(len(mark)):
       if mark[mar] != 'pass':
           mark_left_right.append(mark[mar]) 
           time_left_right.append(time[mar]) 
           
    mark_left_pass = []
    time_left_pass = []
    for mar in range(len(mark)):
       if mark[mar] != 'right':
           mark_left_pass.append(mark[mar]) 
           time_left_pass.append(time[mar]) 
           
    mark_right_pass = []
    time_right_pass = []
    for mar in range(len(mark)):
       if mark[mar] != 'left':
           mark_right_pass.append(mark[mar]) 
           time_right_pass.append(time[mar]) 
        
    annotations_left_right = mne.Annotations(time_left_right, window_time, mark_left_right)
    annotations_left_pass  = mne.Annotations(time_left_pass, window_time,  mark_left_pass)
    annotations_right_pass = mne.Annotations(time_right_pass, window_time, mark_right_pass)     
    
    return annotations_left_right, annotations_left_pass, annotations_right_pass

[annotations_l_r, annotations_l_p, annotations_r_p] = multi_annotations(mark, time, window_time = 1.0)

annotations = mne.Annotations(time, 1.0, mark) # this is the annotation of every class
raw.set_annotations(annotations_l_r)

#%%
# ICA for blink removal 

ica = mne.preprocessing.ICA(n_components = 10, random_state = 0,  method='fastica')

filt_raw = raw.copy()
filt_raw.filter(l_freq=1., h_freq=None) # it's recommended to HP filter at 1 Hz 
ica.fit(filt_raw)

# Once the bad components are detected, we procide to remove them 

ica.exclude = [] # We must wisely choose the bad components based on the ICA sources plot

raw_ica = ica.apply(raw.copy(), exclude = ica.exclude)

#%% This part will create an optimal window for each subject
t_total = 4
win_len = 0.9
tim = {'t_min':0, 't_max':win_len}
ite = np.arange(tim['t_max'],t_total,0.1)

total_acc = []
acc  = 0.0
best = [0,0]


for i in range(len(ite)):
    tim['t_min']+=0.1
    tim['t_max']+=0.1
    
    events = mne.events_from_annotations(raw)
    events
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                            exclude='bads')  
  
    epochs = mne.Epochs(raw_ica, events[0], event_id = events[1], preload = True, tmin=tim['t_min'], tmax=tim['t_max'], baseline=None, picks = picks)
    
   

    #COMMON SPATIAL PATTERN (CSP)
    from mne.decoding import CSP

    labels = epochs.events[:, -1] # In documentation they put a -2 after the []
    epochs_train = epochs.copy()
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    
    #IMPORTANT, set_baseline, must be 0
    # reg may be 'auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
    csp = CSP(n_components=10, reg='oas', rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur  
    csp.fit_transform(epochs_data, labels)
    
        
    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    import matplotlib.pyplot as plt

    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=None)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()

  
    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    
    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))
    total_acc.append(np.mean(scores))
    actual_acc = np.mean(scores)
    
    if actual_acc > acc:
        acc = actual_acc
        best = [tim['t_min'], tim['t_max']]


plt.plot(np.arange(0,round(t_total-win_len,1),0.1), total_acc)
print(best)