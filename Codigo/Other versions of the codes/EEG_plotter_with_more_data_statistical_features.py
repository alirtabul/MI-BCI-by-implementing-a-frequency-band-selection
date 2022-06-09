#%%
"""
In this script we are going to mix two dataset from the same subject in order to train the model 
with more data.  
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

# %matplotlib inline # displaying in the same code
# %matplotlib qt # displaying in a separate windows
#%%
# Firstly, we must locate the files. 
# Firstly, we must locate the files. 
fs= 200
path1_C = '../Data/CLA/CLASubjectC1512233StLRHand.mat' # best time 2.1  3.9
path2_C = '../Data/CLA/CLASubjectC1512163StLRHand.mat' # best time 2.1  3.9
path3_C = '../Data/CLA/CLASubjectC1511263StLRHand.mat'



path1_B = '../Data/CLA/CLASubjectB1510193StLRHand.mat' # best time 2.1  3.9
path2_B = '../Data/CLA/CLASubjectB1510203StLRHand.mat' # best time 2.1  3.9
path3_B = '../Data/CLA/CLASubjectB1512153StLRHand.mat'


path1_E = '../Data/CLA/CLASubjectE1512253StLRHand.mat' # best time 2.1  3.9
path2_E = '../Data/CLA/CLASubjectE1601193StLRHand.mat' # best time 2.1  3.9
path3_E = '../Data/CLA/CLASubjectE1601223StLRHand.mat'


path1_F = '../Data/CLA/CLASubjectF1509163StLRHand.mat' # best time 2.1  3.9
path2_F = '../Data/CLA/CLASubjectF1509173StLRHand.mat' # best time 2.1  3.9
path3_F = '../Data/CLA/CLASubjectF1509283StLRHand.mat'


def epoch_event(path, tmin= 2.1, tmax=3, LP=15,HP=21):

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
    raw.filter(LP, HP, fir_design='firwin', skip_by_annotation='edge')
    
    
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
    
    raw.set_annotations(annotations_l_r)
    
    events = mne.events_from_annotations(raw)
    
    picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
    
    # Let's concatenate all Left and right epochs ##################################################################
    
    
    epochs = mne.Epochs(raw, events[0], events[1], tmin, tmax,
                        picks=picks, baseline=None, preload=True)
    
    Left_Right_epochs_data = epochs.get_data()[:475]
    Left_Right_epochs_data_test = epochs.get_data()[475:]
    # Let's create the Pass epochs by having the exact marker times and 2 secons after it ##################################################################
    
    
    raw.set_annotations(annotations_l_p)
    
    events = mne.events_from_annotations(raw)
    
    picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
    
    epochs1 = mne.Epochs(raw, events[0], events[1], 0, 0.9,
                        picks=picks, baseline=None, preload=True)
    
    epochs2 = mne.Epochs(raw, events[0], events[1], tmin, tmax,
                        picks=picks, baseline=None, preload=True)
    
    e_p1 = epochs1['pass'].get_data()
    e_p2 = epochs2['pass'].get_data()
    
    
    Pass_epochs_data = np.concatenate((e_p1,e_p2))[:475]
    Pass_epochs_data_test = np.concatenate((e_p1,e_p2))[475:]
    
    All_epochs = np.concatenate((Left_Right_epochs_data,Pass_epochs_data))
    All_epochs_test = np.concatenate((Left_Right_epochs_data_test,Pass_epochs_data_test))
    
    labels1 = np.array([1 for i in range(len(Left_Right_epochs_data))])
    labels2 = np.array([2 for i in range(len(Pass_epochs_data))])
    labels = np.concatenate((labels1,labels2))
    
    labels1 = np.array([1 for i in range(len(Left_Right_epochs_data_test))])
    labels2 = np.array([2 for i in range(len(Pass_epochs_data_test))])
    labels_test = np.concatenate((labels1,labels2))
    
    return All_epochs, labels, All_epochs_test, labels_test 

[epoch1_C, label1_C, epoch1_C_test, label1_C_test] = epoch_event(path1_C, tmin= 2.1, tmax=3, LP=15,HP=26)   # LP=15,HP=21
[epoch2_C, label2_C, epoch2_C_test, label2_C_test] = epoch_event(path2_C, tmin= 2.1, tmax=3, LP=15,HP=26)  
[epoch3_C, label3_C, epoch3_C_test, label3_C_test] = epoch_event(path3_C, tmin= 2.1, tmax=3, LP=15,HP=26) 

[epoch1_B, label1_B, epoch1_B_test, label1_B_test] = epoch_event(path1_B, tmin= 2.1, tmax=3, LP=15,HP=26)   
[epoch2_B, label2_B, epoch2_B_test, label2_B_test] = epoch_event(path2_B, tmin= 2, tmax=2.9, LP=15,HP=26)  
[epoch3_B, label3_B, epoch3_B_test, label3_B_test] = epoch_event(path3_B, tmin= 2.3, tmax=3.2, LP=15,HP=26)
[epoch1_E, label1_E, epoch1_E_test, label1_E_test] = epoch_event(path1_E, tmin= 2.2, tmax=3.1, LP=15,HP=26)   
[epoch2_E, label2_E, epoch2_E_test, label2_E_test] = epoch_event(path2_E, tmin= 2, tmax=2.9, LP=15,HP=26)  
[epoch3_E, label3_E, epoch3_E_test, label3_E_test] = epoch_event(path3_E, tmin= 2.2, tmax=3.1, LP=15,HP=26)
#[epoch1_F, label1_F, epoch1_F_test, label1_F_test] = epoch_event(path1_F, tmin= 1, tmax=1.9, LP=15,HP=26)   
[epoch2_F, label2_F, epoch2_F_test, label2_F_test] = epoch_event(path2_F, tmin= 1.3, tmax=2.2, LP=15,HP=26)  
[epoch3_F, label3_F, epoch3_F_test, label3_F_test] = epoch_event(path3_F, tmin= 1.4, tmax=2.3, LP=15,HP=26)

All_epochs = np.concatenate((epoch1_C, epoch2_C, epoch3_F, epoch2_F))
labels = np.concatenate((label1_C, label2_C, label3_F, label2_F))
All_epochs_test = np.concatenate((epoch1_C_test, epoch2_C_test, epoch3_F_test, epoch2_F_test))
labels_test = np.concatenate((label1_C_test, label2_C_test, label3_F_test, label2_F_test))

# -*- coding: utf-8 -*-
"""
In this program we are going to mix the Right and Left Classes and create new pass classes
in order to augment the data that we have for the Probabilistic and the statistical features program.
"""

#%% Now let's run the Probabilistic_and_statistical_features.py with the new data

from scipy.stats import norm

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



C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log= feature_extracter(All_epochs) # all epochs
D_skw, D_krt, D_energy, D_entropy, D_shannon, D_log= feature_extracter(All_epochs_test)

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

FEAT_test = feature_ex(D_skw, D_krt, D_energy, D_entropy, D_shannon, D_log)
FEAT_test = np.array(FEAT_test)


# mean_FEAT = []
# for i in FEAT:
#     A = [0,0,0,0,0,0]
#     A[0] = (i[0]+i[6]+i[12])/3
#     A[1] = (i[1]+i[7]+i[13])/3
#     A[2] = (i[2]+i[8]+i[14])/3
#     A[3] = (i[3]+i[9]+i[15])/3
#     A[4] = (i[4]+i[10]+i[16])/3
#     A[5] = (i[5]+i[11]+i[17])/3  
#     mean_FEAT.append(A)   
# FEAT = np.array(mean_FEAT)        


# mean_FEAT_test = []
# for i in FEAT_test:
#     A = [0,0,0,0,0,0]
#     A[0] = (i[0]+i[6]+i[12])/3
#     A[1] = (i[1]+i[7]+i[13])/3
#     A[2] = (i[2]+i[8]+i[14])/3
#     A[3] = (i[3]+i[9]+i[15])/3
#     A[4] = (i[4]+i[10]+i[16])/3
#     A[5] = (i[5]+i[11]+i[17])/3  
#     mean_FEAT_test.append(A)   
# FEAT_test = np.array(mean_FEAT_test)         
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



scores = []
epochs_data = All_epochs

cv = ShuffleSplit(10, test_size=0.20)
cv_split = cv.split(epochs_data)

# Assemble a classifier, the best one is the LDA

lda = LinearDiscriminantAnalysis()
svm = SVC(kernel ='rbf', gamma = 0.00001, C = 10000)
knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
NB  = GaussianNB()
RF  = RandomForestClassifier(n_estimators = 100, #Optimized with CVSearch
                             min_samples_split = 5,
                              min_samples_leaf = 4,
                              max_features = 'sqrt',
                              max_depth = 11,
                              bootstrap = False, class_weight={1:0.4,2:0.6}) # With class weight we give more importance to the pass class
qda = QuadraticDiscriminantAnalysis()

# Normalize
scale = StandardScaler()

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('scale', scale),('RF', RF)])
scores = cross_val_score(clf, FEAT[:], labels[:], cv=cv, n_jobs=1)
clf.fit(FEAT[:], labels[:])  # We fit the model for making it usable 


asa = clf.predict(FEAT[[16]]) # This is an example of a prediction. Notice the doubl [[]]
testing_score = clf.score(FEAT_test[:],labels_test[:])
print('Accuracy: ', testing_score)
print('Precision: ', precision_score(labels_test[:], clf.predict(FEAT_test[:],))) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
print('Recall: ', recall_score(labels_test[:], clf.predict(FEAT_test[:],)))  

# We can plot the ROC and calculate the AUC
fig= plt.figure()
fpr, tpr, threshold = roc_curve(labels[:]-1, clf.predict(FEAT[:],)-1)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f +- %f / Chance level: %f" % (np.mean(scores),np.std(scores),
                                                          class_balance))


plot_confusion_matrix(clf, FEAT_test[:], labels_test[:])
plt.show()
#%% Let's see which are the features with more relevance to our prediction with the Shapley valeus

import shap
features_names = ['skw[C3]', 'krt[C3]', 'energy[C3]', 'entropy[C3]', 'shannon[C3]', 'log[C3]']

explainer = shap.TreeExplainer(clf.named_steps["RF"])
shap_values = explainer.shap_values(FEAT)
shap_values

shap.summary_plot(shap_values, FEAT, feature_names= features_names, plot_type="bar")


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
#%% Save the model.x

import pickle

with open('RightLeft_vs_Pass_Classification_all.pkl','wb') as f:
    pickle.dump(clf,f)

 

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

