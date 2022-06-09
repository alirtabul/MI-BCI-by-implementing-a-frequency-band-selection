# -*- coding: utf-8 -*-
"""
In this code, features are extracted from epochs in order to distinguish the rest state and the 
imagery state
We will use the following statistical features: 
    - Skewness
    - Logarithm energy entropy
    - Energy
    - Shannon entropy
    - Kurtosis.

These features will be extracted to C3 and C4 channels and their neighbours.

"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import pymatreader
import scipy.stats

# load and preprocess data ####################################################
path = '../Data/CLA/CLASubjectA1601083StLRHand.mat' # best time 0.6  1.4, BW = 22-26
path = '../Data/CLA/CLASubjectC1512233StLRHand.mat' # best time 2.1  2.9 # AWSOME DATA
#path = '../Data/CLA/CLASubjectC1512163StLRHand.mat' # best time 2.1  2.9
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
raw.filter(15., 21., fir_design='firwin', skip_by_annotation='edge', filter_length='auto')


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

markers =  np.array(struct['marker']).transpose()
[mark, pos, time] = type_pos_pas(markers)

# We create a function which will separate left/right marks from pass,
# left/pass marks  from right, and righ/pass marks from left.

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

raw.set_annotations(annotations_l_p)

events = mne.events_from_annotations(raw)

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])

# epoch data ##################################################################
tmin, tmax = 2.1, 3  # define epochs around events (in s)

epochs = mne.Epochs(raw, events[0], events[1], tmin, tmax,
                    picks=picks, baseline=None, preload=True)


epochs_data = epochs.get_data()

epochs_data[2]
epochs
#%% 
# Herein, the probability density function is ploted (for visual purposes).
from scipy.stats import norm



data = epochs_data[154] # Choose any epoch
ch = epochs.ch_names

plt.style.use('ggplot')


fig, axs = plt.subplots(1, (data.shape[0]), figsize=(12,5))
fig.suptitle('Probability density function (PDF)', fontsize=30)


pdf_all = []
for i in range((data.shape[0])):
    
    # A channel is selected from an epoch ['C3' --> 0, 'C4' --> 1, 'Cz' --> 2]
    
    gau = scipy.stats.gaussian_kde(data[i])
     
    
    # Values in which the gausian will be evaluated
    dist_space = np.linspace( min(data[i]), max(data[i]), 200 )

    gau_total = np.sum(gau(dist_space))# Normalize, this is the integral underthe curve

    pdf = gau(dist_space)/gau_total
   
   
    pdf_all.append(pdf)
    axs[i].plot(dist_space, pdf,lw=3, color= [210/255,90/255,150/255])
    axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i].tick_params(axis='x', labelsize=15)
    axs[i].tick_params(axis='y', labelsize=15)
    axs[i].set_title(ch[i], color = 'k', fontsize=25)
    axs[i].set_xlabel('Voltage (V)', fontsize=20)
    axs[i].set_ylabel('Probability density', fontsize=20)
plt.show()


#%% Skewness

# data = epochs_data[36]


skw_epoch = []
for i in range((data.shape[0])):
    skw_epoch.append(scipy.stats.skew(data[i], axis=0, bias=True))
    
print('skewness: ', skw_epoch)

#%% Kurtosis

# data = epochs_data[36]


krt_epoch = []
for i in range((data.shape[0])):
    krt_epoch.append(scipy.stats.kurtosis(data[i], axis=0, fisher=True, bias=True, nan_policy='propagate'))

print('Kurtosis:', krt_epoch)
#%% Energy
# Formulas:   np.sum((x))**2 

# data = epochs_data[36]

def energy(signal):
    energy = 0.0
    
    for x in signal:
        energy += (x)**2    
    return energy

energy_epoch = []
for i in range((data.shape[0])):
    energy_epoch.append(energy(data[i]))

print('Enery: ', energy_epoch)

#%% Shannon Entropy, Entropy & Log Energy Entropy
# Formulas:
    # Entropy:             -np.sum(freq * np.log2(freq))           freq---> is each valueo of the pdf
    # Shannon Entropy:     -np.sum((freq)**2 * (np.log2(freq))**2)
    # Log Energy Entropy:  -np.sum((np.log2(freq))**2)


# formulas are found in the following paper: DOI: 10.1007/s10439-009-9795-x


def all_entropy(pdf):
    '''
    pdf    ---> The Probability density function calculated above. 
    
    RETURNS: entropy, shannon_entropy, log_energy_entropy
    '''
    shannon_entropy = 0.0
    entropy = 0.0
    log_energy_entropy = 0.0
    
    for freq in pdf:
        entropy += freq * np.log2(freq)
        shannon_entropy += (freq)**2 * (np.log2(freq))**2
        log_energy_entropy += (np.log2(freq))**2
    entropy = -entropy    
    shannon_entropy = -shannon_entropy
    log_energy_entropy = -log_energy_entropy
    
    return entropy,shannon_entropy, log_energy_entropy

entropy_epoch = []
shannon_entropy_epoch = []
log_energy_entropy_epoch = []

for i in range((data.shape[0])):
    # S = -sum(pk * log(pk)) --> pk is the Probability Density Function 
    entr, shan, log_en= all_entropy(pdf_all[i])
    
    entropy_epoch.append(entr)
    shannon_entropy_epoch.append(shan)
    log_energy_entropy_epoch.append(log_en)

print(' Entropy: {} \n Shannon: {} \n Log Energy Entropy: {}'.format(entropy_epoch,  shannon_entropy_epoch,  log_energy_entropy_epoch))
#%% Let's create a function that gathers all the features in an array and for more eopchs
import pywt

def feature_extracter(epochs): 
    ch = ["C3", "Cz", "C4"]
    
    skw_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
    krt_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
    energy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
    entropy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
    shannon_entropy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
    log_energy_entropy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
    
    for epoch in epochs:
        
        data = epoch
        
        pdf_all = []
        for i in range((data.shape[0])):
            
            # A channel is selected from an epoch ['C3' --> 0, 'C4' --> 1, 'Cz' --> 2]
            
            gau = scipy.stats.gaussian_kde(data[i])
             
            
            # Values in which the gausian will be evaluated
            dist_space = np.linspace( min(data[i]), max(data[i]), 100 )
        
            gau_total = np.sum(gau(dist_space))# Normalize, this is the integral underthe curve
        
            pdf = gau(dist_space)/gau_total
            
            pdf_all.append(pdf)
            
        skw_epoch = []        
        for i in range((data.shape[0])):
            skw_epoch.append(scipy.stats.skew(data[i], axis=0, bias=True))
        krt_epoch = []
        for i in range((data.shape[0])):
            krt_epoch.append(scipy.stats.kurtosis(data[i], axis=0, fisher=True, bias=True, nan_policy='propagate'))
        
        def energy(signal):
            energy = 0.0
            
            for x in signal:
                energy += (x)**2    
            return energy
        
        energy_epoch = []
        for i in range((data.shape[0])):
            energy_epoch.append(energy(data[i]))    
            
        
        
        Pi_all = [] # here we save the values for each channel
        # We need Wavelets in order to perform the Shannon Energy, for instance.
        for i in range((data.shape[0])):
           
            coef,_= pywt.cwt(data[i],  np.arange(7,20,0.5), 'morl') # Apply the Wavelet transform
            
            # now we use the following formulas: 'https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy'
            [M,N] = coef.shape; # M --> scale number, N --> time segments
            Ej = []
            for j in range(M):
                Ej.append(sum(abs(coef[j,:])));
                
            Etot=sum(Ej);
            
            pi = []
            for i in Ej:
                pi.append(i/Etot)
            Pi_all.append(pi)
        Pi_all = np.array(Pi_all)   
        
        
        
        def all_entropy(pdf):
            '''
            pdf    ---> The Probability density function calculated above. 
            
            RETURNS: entropy, shannon_entropy, log_energy_entropy
            '''
            shannon_entropy = 0.0
            entropy = 0.0
            log_energy_entropy = 0.0
            
            for freq in pdf:
                entropy += freq * np.log2(freq)
                shannon_entropy += (freq)**2 * (np.log2(freq)**2)
                log_energy_entropy += (np.log2(freq))**2
            entropy = -entropy    
            shannon_entropy = -shannon_entropy
            log_energy_entropy = -log_energy_entropy
            
            return entropy,shannon_entropy, log_energy_entropy
        entropy_epoch = []
        shannon_entropy_epoch = []
        log_energy_entropy_epoch = []
        for i in range((data.shape[0])):
            # S = -sum(pk * log(pk)) --> pk is the Probability Density Function 
            entr, shan, log_en = all_entropy(pdf_all[i])
            
            entropy_epoch.append(entr)
            shannon_entropy_epoch.append(shan)
            log_energy_entropy_epoch.append(log_en)    
        
        skw_epoch_dic[ch[0]].append(skw_epoch[0]), skw_epoch_dic[ch[1]].append(skw_epoch[1]), skw_epoch_dic[ch[2]].append(skw_epoch[2]) 
        krt_epoch_dic[ch[0]].append(krt_epoch[0]), krt_epoch_dic[ch[1]].append(krt_epoch[1]), krt_epoch_dic[ch[2]].append(krt_epoch[2]) 
        energy_epoch_dic[ch[0]].append(energy_epoch[0]), energy_epoch_dic[ch[1]].append(energy_epoch[1]), energy_epoch_dic[ch[2]].append(energy_epoch[2])
        shannon_entropy_epoch_dic[ch[0]].append(shannon_entropy_epoch[0]), shannon_entropy_epoch_dic[ch[1]].append(shannon_entropy_epoch[1]), shannon_entropy_epoch_dic[ch[2]].append(shannon_entropy_epoch[2])
        log_energy_entropy_epoch_dic[ch[0]].append(log_energy_entropy_epoch[0]), log_energy_entropy_epoch_dic[ch[1]].append(log_energy_entropy_epoch[1]), log_energy_entropy_epoch_dic[ch[2]].append(log_energy_entropy_epoch[2])
        entropy_epoch_dic[ch[0]].append(entropy_epoch[0]), entropy_epoch_dic[ch[1]].append(entropy_epoch[1]), entropy_epoch_dic[ch[2]].append(entropy_epoch[2]) 
        
    return skw_epoch_dic, krt_epoch_dic, energy_epoch_dic, entropy_epoch_dic, shannon_entropy_epoch_dic, log_energy_entropy_epoch_dic

# Separate the data:
e_p = epochs['pass']
e_l = epochs['left']

A_skw, A_krt, A_energy, A_entropy, A_shannon, A_log= feature_extracter(e_p) # pass
B_skw, B_krt, B_energy, B_entropy, B_shannon, B_log= feature_extracter(e_l)  # left
C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log= feature_extracter(epochs) # all epochs



plt.scatter( A_log['C4'], A_shannon['C4'], color = 'k', marker = 'x',  s= 0.6, edgecolors= 'k')
plt.scatter( B_log['C4'], B_shannon['C4'], color = 'm', marker = 'x',  s= 0.6, edgecolors= 'k')
plt.show()

plt.scatter( A_log['C4'], A_energy['C4'], color = 'k', marker = 'x',  s= 0.6, edgecolors= 'k')
plt.scatter( B_log['C4'], B_energy['C4'], color = 'm', marker = 'x',  s= 0.6, edgecolors= 'k')
plt.show()

# Now we extract the featuers for each epoch:

def feature_ex(skw, krt, energy, entropy, shannon, log):
    big_feat = [] 
    ch = ['C3','C4', 'Cz']
    for j in range(len(skw['C3'])):
        small_feat = []
        for i in range(len(ch)):
            
            new_small_feat = [skw[ch[i]][j], krt[ch[i]][j], energy[ch[i]][j], entropy[ch[i]][j], shannon[ch[i]][j], log[ch[i]][j]]
            small_feat.extend(new_small_feat)
        big_feat.append(small_feat)
    return big_feat  
             
FEAT = feature_ex(C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log)
FEAT = np.array(FEAT)
A = FEAT


#%% With the features we classify the data. CV will be applied to classify the data

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC                         # SVM     
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.naive_bayes import GaussianNB          # Naive Bayesian
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #QDA                         # QDA 
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, precision_score, recall_score, auc, roc_curve


labels = epochs.events[:, -1]

scores = []
epochs_data = epochs.get_data()

cv = ShuffleSplit(100, test_size=0.20)
cv_split = cv.split(epochs_data)

# Assemble a classifier, the best one is the LDA

lda = LinearDiscriminantAnalysis()
svm = SVC()
knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
NB  = GaussianNB()
RF  = RandomForestClassifier(n_estimators = 250, #Optimized with CVSearch
                             min_samples_split = 5,
                              min_samples_leaf = 2,
                              max_features = 'sqrt',
                              max_depth = 6,
                              bootstrap = True, class_weight={1:0.4,2:0.6}) # With class weight we give more importance to the pass class
qda = QuadraticDiscriminantAnalysis()

# Normalize
scale = StandardScaler()

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('scale', scale),('RF', RF)])
scores = cross_val_score(clf, FEAT[:500], labels[:500], cv=cv, n_jobs=1)
clf.fit(FEAT[:500], labels[:500])  # We fit the model for making it usable 


asa = clf.predict(FEAT[[16]]) # This is an example of a prediction. Notice the doubl [[]]

testing_score = clf.score(FEAT[500:],labels[500:])
print('Accuracy: ', testing_score)
print('Precision: ', precision_score(labels[500:], clf.predict(FEAT[500:],))) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
print('Recall: ', recall_score(labels[500:], clf.predict(FEAT[500:],))) 

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy (CV)): %f +- %f / Chance level: %f" % (np.mean(scores),np.std(scores),
                                                          class_balance))

plot_confusion_matrix(clf, FEAT[500:], labels[500:])
plt.show()
#%% Lets find the best parameters for the Random Forest with GridSearch
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 15, num = 13)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
RF_opt = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = RF_opt, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(FEAT, labels)
print(rf_random.best_params_)
#%% Save the model.

import pickle

with open('LeftPass_Classification.pkl','wb') as f:
    pickle.dump(clf,f)

#%% WAVELET FOR SHANNON ENTROPY


import pywt



# This two functions extracts the best scales for the wavelet. 
# This two functions are given by: ' https://github.com/TNTLFreiburg/brainfeatures/blob/91d12949c71bc3e1c8e67d6081a8274e79f4c35d/brainfeatures/feature_generation/feature_generator.py'
def freq_to_scale(freq, wavelet, sfreq):
    """ compute cwt scale to given frequency
    see: https://de.mathworks.com/help/wavelet/ref/scal2frq.html """
    central_freq = pywt.central_frequency(wavelet)
    assert freq > 0, "freq smaller or equal to zero!"
    scale = central_freq / freq
    return scale * sfreq

def freqs_to_scale(freqs, wavelet, sfreq):
    """ compute cwt scales to given frequencies """
    scales = []
    for freq in freqs:
        scale = freq_to_scale(freq, wavelet, sfreq)
        scales.append(scale)
    return scales

scale = freqs_to_scale(np.arange(7,30), 'morl', 200)

# Let's check if the frquencies are the desired 
dt = 1/200  # 200 Hz sampling
frequencies = pywt.scale2frequency('morl', scale) / dt # Lets find the frquencies that are more usefull tu us (30-7)
frequencies
# Now let's compute the coeficiencies of the WT, applying a MORLET


Pi_all = [] # here we save the the values for each channel
for i in range((data.shape[0])):
   
    coef,_= pywt.cwt(data[i],  np.arange(7,20,0.5), 'morl')
    
    # now we use the following formulas: 'https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy'
    [M,N] = coef.shape; # M --> scale number, N --> time segments
    Ej = []
    for j in range(M):
        Ej.append(sum(abs(coef[j,:])));
        
    Etot=sum(Ej);
    
    pi = []
    for i in Ej:
        pi.append(i/Etot)
    Pi_all.append(pi)
Pi_all = np.array(Pi_all)    
 

#%% If we want to add dimensionality reduction we can use PCA. However the results are not good.
# FEAT_norm = StandardScaler().fit_transform(FEAT)
# from sklearn.decomposition import PCA
# import pandas as pd
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(FEAT_norm)
# principalDf = pd.DataFrame(data = principalComponents
#               , columns = ['principal component 1', 'principal component 2'])
# finalDf = pd.concat([principalDf, pd.Series(labels, name = 'target')], axis = 1)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = [1, 2]
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                 , finalDf.loc[indicesToKeep, 'principal component 2']
#                 , c = color
#                 , s = 50)
# ax.legend(targets)
# ax.grid()

# Let's try the model (IT WORKS REALLY GREAT!)(It also detects the right :) )

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
Acisco = raw.get_data(picks=picks)
times = [0]
for i in np.arange(2990, 3310,0.1): # first value is the starting time, second value the ending time and th third one the time step
    vala = i
    ACA = [Acisco[:,int(vala*200):int(vala*200+180)]] # Remember that the sampling frequency is 200. The 180 are the samples between 2.1 and 3 second (0.9*200)
    CA_skw, CA_krt, CA_energy, CA_entropy, CA_shannon, CA_log = feature_extracter(ACA)
    FEAT = feature_ex(CA_skw, CA_krt, CA_energy, CA_entropy, CA_shannon, CA_log)
    FEAT = np.array(FEAT)
    pred = list(clf.predict(FEAT))
    
    if pred == [1]:
        if i > (times[0]+2):
            times[0] = i
            print(i)
