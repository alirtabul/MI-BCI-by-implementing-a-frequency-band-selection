#%%
"""
In this script the EEG is going to be ploted and epoched into its relative motor imagery event, using 
the MNE library.  
"""
# import the libraries
import mne
import matplotlib # IMPORTANT--> It must be version 3.2.2
import matplotlib.pyplot as plt
import pylab
import pymatreader 
import pandas as pd
import numpy as np
import os

# %matplotlib inline # displaying in the same code
# %matplotlib qt # displaying in a separate windows
#%%
# Firstly, we must locate the files. 
path = '../Data/CLA/CLASubjectA1601083StLRHand.mat' # best time 0.6  1.4, 
path = '../Data/CLA/CLASubjectC1512233StLRHand.mat' # best time 2.1  3.9
#path = '../Data/CLA/CLASubjectC1512163StLRHand.mat' # best time 2.1  3.9
# Now we simply pass the matlab files into a python compatible file.
struct = pymatreader.read_mat(path, ignore_fields=['previous'], variable_names = 'o')
struct = struct['o']
# Le'ts set the time in which the epochs usually start
tmin = 2.1
tmax = 3


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
raw.plot(title='RAW EEG') 

# We filter the signal for motor imagery anlysis
raw = raw.set_eeg_reference('average', projection= False)
raw.filter(15., 26., fir_design='firwin', skip_by_annotation='edge') # normally 7-30 Hz


mne.compute_rank(raw, rank='info')
raw.plot()


plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rc('axes', titlesize=30, labelsize=25)


fig = raw.plot_psd(dB=False, xscale='linear', estimate='power')  # show the power spectrum density 
fig.suptitle('Power spectral density (PSD)',  fontsize = 30)
plt.show()

fig = raw.plot_psd(dB=True, xscale='linear', average = False )  # show the power spectrum density 
fig.suptitle('Power spectral density (PSD) (dB)', fontsize = 30)
plt.show()




fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d')
raw.plot_sensors(show_names = False, axes = ax2d)
raw.plot_sensors(show_names = False, axes = ax3d, kind = '3d')
# Make a sphere to project the 3d electrodes
u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, np.pi, 200)
x = 0.1 * np.outer(np.cos(u), np.sin(v))
y = 0.1 * np.outer(np.sin(u), np.sin(v))
z = 0.15 * np.outer(np.ones(np.size(u)), np.cos(v))
ax3d.plot_surface(x, y, z,alpha = 0.8, color='navajowhite')
plt.show()

raw_lap = mne.preprocessing.compute_current_source_density(raw)  # Apply the surface Laplacian (CSD)
raw_lap.plot()


#%%
# lets create annotations with markers:


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

# We plot (also topographically) the ICA components to exclude the blinkings.  
ica.plot_components(outlines = 'skirt', colorbar=True, contours = 0)
ica.plot_sources(raw) 
# Once the bad components are detected, we procide to remove them 

ica.exclude = [] # We must wisely choose the bad components based on the ICA sources plot

raw_ica = ica.apply(raw.copy(), exclude = ica.exclude)

raw.plot()
raw_ica.plot()  
# If we compare the two graphs... MAGIC! Blinks are gone :)

#%%
# Now let's grasp the events from the annotations, so we can work with them. 

events = mne.events_from_annotations(raw)
events
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')  

# with the events we acquire the epochs  
epochs = mne.Epochs(raw_ica, events[0], event_id = events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = picks)
# It plots all the events in one plot
epochs.plot()


try: 
                                    
    epochs['right'].plot_image(picks = ['C3'])
    e_r = epochs['right']
    e_l = epochs['left']
    # e_p = epochs['pass']
    
    e_r.plot_psd(picks = ['C3','Cz','C4'], dB=True)
    e_l.plot_psd(picks = ['C3','Cz','C4'], dB=True)
    
    # random_right_event = mne.evoked.EvokedArray(e_r[np.random.randint(0, e_r.get_data().shape[0]+1)].get_data().reshape(21,201), raw.info)
    right = e_r.average()
    left  = e_l.average()
    
    
    right_csd =  mne.preprocessing.compute_current_source_density(right)
    left_csd =  mne.preprocessing.compute_current_source_density(left)
    
    
    right_csd.plot_joint(title='Current Source Density (RIGHT))')
    left_csd.plot_joint(title='Current Source Density (LEFT)')
    
except:
    pass
#%%
right = e_r.average()
left  = e_l.average()

right.plot_joint(title= 'Right')
left.plot_joint( title = 'Left')

#%% COMMON SPATIAL PATTERN (CSP)
from mne.decoding import CSP

labels = epochs.events[:, -1] # In the documentation they put a -2 after the []
epochs_train = epochs.copy()
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

#IMPORTANT, set_baseline, must be 0
# reg may be 'auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
csp = CSP(n_components=10, reg='oas', rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur  
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)



#%%
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC                         # SVM     
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.naive_bayes import GaussianNB          # Naive Bayesian
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #QDA                         # QDA 
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, precision_score, recall_score, auc, roc_curve


scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(n_splits=100, test_size=0.2)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier, the best one is the LDA

lda = LinearDiscriminantAnalysis()
svm = SVC()
knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
NB  = GaussianNB()
RF  = RandomForestClassifier()
qda = QuadraticDiscriminantAnalysis()

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train[:500], labels[:500], cv=cv, n_jobs=1) # Let's train the first 500 epochs
(learning_curve(clf, epochs_data_train[:500], labels[:500], cv=cv, scoring='accuracy'))
clf.fit(epochs_data_train[:500], labels[:500])  # We fit the model for making it usable 


testing_score = clf.score(epochs_data_train[500:],labels[500:])
print('Accuracy: ', testing_score)
print('Precision: ', precision_score(labels[500:], clf.predict(epochs_data_train[500:],))) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
print('Recall: ', recall_score(labels[500:], clf.predict(epochs_data_train[500:],))) 


asa = clf.predict(epochs_data_train[[0]]) # This is an example of a prediction. Notice the doubl [[]]

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f +- %f / Chance level: %f" % (np.mean(scores),np.std(scores),
                                                          class_balance))


plot_confusion_matrix(clf, epochs_data_train[500:], labels[500:])
plt.show()
#%% Save the model.

import pickle

with open('LeftRight_Classification.pkl','wb') as f:
    pickle.dump(clf,f)
    
# # load
# with open('model.pkl', 'rb') as f:
#     clf2 = pickle.load(f)

# clf2.predict(X[0:1])
#%%
# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)



sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()

#%% This part will create an optimal window for each subject

# tim = {'t_min':0, 't_max':0.8}
# ite = np.arange(tim['t_max'],3.4,0.1)

# total_acc = []
# acc  = 0.0
# best = [0,0]


# for i in range(len(ite)):
#     tim['t_min']+=0.1
#     tim['t_max']+=0.1
    
#     events = mne.events_from_annotations(raw)
#     events
#     picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
#                            exclude='bads')  
  
#     epochs = mne.Epochs(raw_ica, events[0], event_id = events[1], preload = True, tmin=tim['t_min'], tmax=tim['t_max'], baseline=None, picks = picks)
    
   

#     #COMMON SPATIAL PATTERN (CSP)
#     from mne.decoding import CSP

#     labels = epochs.events[:, -1] # In documentation they put a -2 after the []
#     epochs_train = epochs.copy()
#     epochs_data = epochs.get_data()
#     epochs_data_train = epochs_train.get_data()
    
#     #IMPORTANT, set_baseline, must be 0
#     # reg may be 'auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
#     csp = CSP(n_components=10, reg='oas', rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur  
#     csp.fit_transform(epochs_data, labels)
    
        
#     from sklearn.pipeline import Pipeline
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.model_selection import ShuffleSplit, cross_val_score
#     import matplotlib.pyplot as plt

#     scores = []
#     epochs_data = epochs.get_data()
#     epochs_data_train = epochs_train.get_data()
#     cv = ShuffleSplit(100, test_size=0.2, random_state=None)
#     cv_split = cv.split(epochs_data_train)

#     # Assemble a classifier
#     lda = LinearDiscriminantAnalysis()

  
#     # Use scikit-learn Pipeline with cross_val_score function
#     clf = Pipeline([('CSP', csp), ('LDA', lda)])
#     scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    
#     # Printing the results
#     class_balance = np.mean(labels == labels[0])
#     class_balance = max(class_balance, 1. - class_balance)
#     print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                           class_balance))
#     total_acc.append(np.mean(scores))
#     actual_acc = np.mean(scores)
    
#     if actual_acc > acc:
#         acc = actual_acc
#         best = [tim['t_min'], tim['t_max']]


# plt.plot(total_acc)

#%% In this section, 3 classifiers (l-r, r-p, l-p) will be computed: 
# labels ---->  lr ={l = 1, r = 2}, lp ={l = 1, p = 2}, rp ={r = 2, p = 1},
# for each type we use lda, svm and qda for more robustness

models_names = ['lr','lp','rp']
models_acc   = {'lr': 0, 'lp': 0, 'rp': 0}


    
for i in range(3):
    if i == 0:
        raw.set_annotations(annotations_l_r)
    if i == 1:
        raw.set_annotations(annotations_l_p)
    if i == 2: 
        raw.set_annotations(annotations_r_p)
        
    events = mne.events_from_annotations(raw)
    picks  = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')  
    # we separate the epochs of train, by just having the first 85% of epochs.  
    
    epochs_train = mne.Epochs(raw_ica, events[0][:int(len(events[0])*0.80)], event_id = events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = picks)
    epochs_test = mne.Epochs(raw_ica, events[0][int(len(events[0])*0.80):], event_id = events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = picks)
    
    labels_train = epochs_train.events[:, -1] # lr ={l = 1, r = 2}, lp ={l = 1, p = 2}, rp ={r = 2, p = 1},
    labels_test  = epochs_test.events[:, -1]
    
   
    epochs_data_train = epochs_train.get_data()
    epochs_data_test  = epochs_test.get_data()
    
    
    #IMPORTANT, set_baseline, must be 0
    # reg may be 'auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
    csp = CSP(n_components=10, reg='oas', rank = 'info'); # Very important to set rank to info, otherwise a rank problem may occur  
    csp.fit_transform(epochs_data, labels);
        
    scores = []
    scores_1 = []
    scores_2 = []
    scores_3 = []
    
    #
    
    
    cv = ShuffleSplit(100, test_size=0.2)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier, the best one is the LDA

    lda = LinearDiscriminantAnalysis()
    svc = SVC()
    knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
    NB  = GaussianNB()
    RF  = RandomForestClassifier()
    qda = QuadraticDiscriminantAnalysis()
    
    
    if i == 0:
        clf_lr_lda = Pipeline([('CSP', csp), ('LDA', lda)])
        scores_1 = cross_val_score(clf_lr_lda, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_lr_lda.fit(epochs_data_train, labels_train)  # We fit the model for making it usable 
        
        clf_lr_svc = Pipeline([('CSP', csp), ('SVC', svc)])
        scores_2 = cross_val_score(clf_lr_svc, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_lr_svc.fit(epochs_data_train, labels_train)  # We fit the model for making it usable 
        
        clf_lr_qda = Pipeline([('CSP', csp), ('QDA', qda)])
        scores_3 = cross_val_score(clf_lr_qda, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_lr_qda.fit(epochs_data_train, labels_train)  # We fit the model for making it usable 
        
    if i == 1:
        clf_lp_lda = Pipeline([('CSP', csp), ('LDA', lda)])
        scores_1 = cross_val_score(clf_lp_lda, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_lp_lda.fit(epochs_data_train, labels_train)  # We fit the model for making it usable
        
        clf_lp_svc = Pipeline([('CSP', csp), ('SVC', svc)])
        scores_2 = cross_val_score(clf_lp_svc, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_lp_svc.fit(epochs_data_train, labels_train)  # We fit the model for making it usable 
        
        clf_lp_qda = Pipeline([('CSP', csp), ('QDA', qda)])
        scores_3 = cross_val_score(clf_lp_qda, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_lp_qda.fit(epochs_data_train, labels_train)  # We fit the model for making it usable 
        
        
    if i == 2: 
        clf_rp_lda = Pipeline([('CSP', csp), ('LDA', lda)])
        scores_1 = cross_val_score(clf_rp_lda, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_rp_lda.fit(epochs_data_train, labels_train)  # We fit the model for making it usable
        
        clf_rp_svc = Pipeline([('CSP', csp), ('SVC', svc)])
        scores_2 = cross_val_score(clf_rp_svc, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_rp_svc.fit(epochs_data_train, labels_train)  # We fit the model for making it usable 
        
        clf_rp_qda = Pipeline([('CSP', csp), ('QDA', qda)])
        scores_3 = cross_val_score(clf_rp_qda, epochs_data_train, labels_train, cv=cv, n_jobs=1)
        clf_rp_qda.fit(epochs_data_train, labels_train)  # We fit the model for making it usable  
        
        
    
    
    asa = clf_lr_lda.predict(epochs_data_test[[77]]) # This is an example of a prediction. Notice the doubl [[]]

    # Printing the results
    class_balance = np.mean(labels_train == labels_train[0])
    class_balance = max(class_balance, 1. - class_balance)
    
    models_acc[models_names[i]] = np.mean(scores_1)
    
    print("Classification accuracy: %f +- %f / Chance level: %f" % (np.mean(scores_1),np.std(scores_1),
                                                          class_balance))
    
# Let's test

annotations = mne.Annotations(time, 1.0, mark) # this is the annotation of every class
raw.set_annotations(annotations)    
events = mne.events_from_annotations(raw)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')  
  
epochs = mne.Epochs(raw_ica, events[0], event_id = events[1], preload = True, tmin=tmin , tmax=tmax, baseline=None, picks = picks)
labels = epochs.events[:, -1]  

epochs_data = epochs.get_data()

#from sklearn.model_selection import train_test_split
# [epochs_data_train, epochs_data_test, labels_train, labels_test] = train_test_split(epochs_data, labels, test_size= 0.15)

epochs_train = mne.Epochs(raw_ica, events[0][:int(len(events[0])*0.80)], event_id = events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = picks)
epochs_test = mne.Epochs(raw_ica, events[0][int(len(events[0])*0.80):], event_id = events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = picks)
    
labels_train = epochs_train.events[:, -1] # lr ={l = 1, r = 2}, lp ={l = 1, p = 2}, rp ={r = 2, p = 1},
labels_test  = epochs_test.events[:, -1]

epochs_data_train = epochs_train.get_data()
epochs_data_test  = epochs_test.get_data()    



classifiers =['clf_lr_lda', 'clf_lr_svc', 'clf_lr_qda',
              'clf_lp_lda', 'clf_lp_svc', 'clf_lp_qda',
              'clf_rp_lda', 'clf_rp_svc', 'clf_rp_qda']

classifiers_dic = {'clf_lr_lda': clf_lr_lda, 'clf_lr_svc': clf_lr_svc, 'clf_lr_qda': clf_lr_qda,
                   'clf_lp_lda': clf_lp_lda, 'clf_lp_svc': clf_lp_svc, 'clf_lp_qda': clf_lp_qda,
                   'clf_rp_lda': clf_rp_lda, 'clf_rp_svc': clf_rp_svc, 'clf_rp_qda': clf_rp_qda}

final_scores = []

for z in range(len(labels_test)): 
    
    lrp_score = {'left_score': 0, 'right_score': 0, 'pass_score': 0}
    
    for j in np.arange(1, 2, int(len(classifiers)/3)): #  3 because there are three classes
        
        
        aaa1_lr = classifiers_dic[classifiers[j]].predict(epochs_data_test[[z]])
        aaa2_lp = classifiers_dic[classifiers[j+3]].predict(epochs_data_test[[z]])
        aaa3_rp = classifiers_dic[classifiers[j+6]].predict(epochs_data_test[[z]])
        aaa4_lab = labels_test[z]
        

        if aaa1_lr == 1:
            lrp_score['left_score']  += 1
        else:
            lrp_score['right_score'] += 1
    
        if aaa2_lp == 1:
            lrp_score['left_score']  += 1
        else:
            lrp_score['pass_score']  += 1
    
        if aaa3_rp == 1:
            lrp_score['pass_score']  += 1
        else:
            lrp_score['right_score'] += 1
    
    a_max_key = max(lrp_score, key=lrp_score.get) # lets get the best trial
        
    if a_max_key == 'left_score':
        final_scores.append(1)
    if a_max_key == 'pass_score':
        final_scores.append(2)  
    if a_max_key == 'right_score':
        final_scores.append(3)
    
final_scores = np.array(final_scores)    
substr = final_scores-labels_test # if the substraction ressults in 0, the guess is correct
count = 0
l_count = 0
p_count = 0
r_count = 0

for i, j in enumerate(substr):
    if j == 0:
        count += 1
        if final_scores[i] == 1:
            l_count += 1
        if final_scores[i] == 2:
            p_count += 1
        if final_scores[i] == 3:
            r_count += 1
    else: 
        continue
    
l_len = 0
p_len = 0
r_len = 0

for i in labels_test:
    if i == 1:
        l_len +=1 
    if i == 2:
        p_len +=1
    if i == 3:
        r_len +=1
    
final_accuracy = count/len(substr)
final_accuracy_l = l_count/l_len
final_accuracy_p = p_count/p_len
final_accuracy_r = r_count/r_len

print('The total accuracy is: ',final_accuracy, '\n',
       'Left accuracy:  ', final_accuracy_l, '\n'
       ' Right accuracy: ', final_accuracy_r, '\n',
       'Pass accuracy:  ', final_accuracy_p, '.')

 
