# -*- coding: utf-8 -*-
"""
Main script for training and testing the AI models for the Left vs Right-hand Imagery.

@author: Ali Abdul Ameer Abbas
"""

# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Import the modules that were created in Left_vs_Right.py
from Left_vs_Right import PreprocessDataset, ProcessDataset, Model_LR

#%% Plot configuration.
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rc('axes', titlesize=30, labelsize=25)

#%%
# Firstly, we must define the paths of dataset' files. 

path1_C = '../Data/CLA/CLASubjectC1512233StLRHand.mat' 
path2_C = '../Data/CLA/CLASubjectC1512163StLRHand.mat' 
path3_C = '../Data/CLA/CLASubjectC1511263StLRHand.mat'


path1_B = '../Data/CLA/CLASubjectB1510193StLRHand.mat' 
path2_B = '../Data/CLA/CLASubjectB1510203StLRHand.mat' 
path3_B = '../Data/CLA/CLASubjectB1512153StLRHand.mat'


path1_E = '../Data/CLA/CLASubjectE1512253StLRHand.mat' 
path2_E = '../Data/CLA/CLASubjectE1601193StLRHand.mat' 
path3_E = '../Data/CLA/CLASubjectE1601223StLRHand.mat'


path1_F = '../Data/CLA/CLASubjectF1509163StLRHand.mat' 
path2_F = '../Data/CLA/CLASubjectF1509173StLRHand.mat' 
path3_F = '../Data/CLA/CLASubjectF1509283StLRHand.mat'

fs= 200 # Define the sampling frequency.

#%%    
# Apply the preprocessing class to the data.

#Initialize the classes for each subject.

SubjC_1 = PreprocessDataset(path = path1_C, show_plots=True)
SubjC_2 = PreprocessDataset(path = path2_C)
SubjC_3 = PreprocessDataset(path = path3_C)

SubjE_1 = PreprocessDataset(path = path1_E)
SubjE_2 = PreprocessDataset(path = path2_E)
SubjE_3 = PreprocessDataset(path = path3_E)

SubjF_1 = PreprocessDataset(path = path1_F)
SubjF_2 = PreprocessDataset(path = path2_F)
SubjF_3 = PreprocessDataset(path = path3_F)

# Set the reference.
SubjC_1.channel_reference() 
SubjC_2.channel_reference() 
SubjC_3.channel_reference() 

SubjE_1.channel_reference()
SubjE_2.channel_reference() 
SubjE_3.channel_reference()

SubjF_1.channel_reference()  
SubjF_2.channel_reference() 
SubjF_3.channel_reference()

# Filter the data.

SubjC_1.MNE_bandpass_filter(HP = 15, LP = 26, show_plots=True) 
SubjC_2.MNE_bandpass_filter(HP = 15, LP = 26) 
SubjC_3.MNE_bandpass_filter(HP = 15, LP = 26) 

SubjE_1.MNE_bandpass_filter(HP = 15, LP = 26)
SubjE_2.MNE_bandpass_filter(HP = 15, LP = 26) 
SubjE_3.MNE_bandpass_filter(HP = 15, LP = 26)

SubjF_1.MNE_bandpass_filter(HP = 15, LP = 26)  
SubjF_2.MNE_bandpass_filter(HP = 15, LP = 26) 
SubjF_3.MNE_bandpass_filter(HP = 15, LP = 26)

# Add annotations.

SubjC_1.add_annotations() 
SubjC_2.add_annotations() 
SubjC_3.add_annotations() 

SubjE_1.add_annotations()
SubjE_2.add_annotations() 
SubjE_3.add_annotations()

SubjF_1.add_annotations()  
SubjF_2.add_annotations() 
SubjF_3.add_annotations()
    
# Apply ICA.

ica_C_1 = SubjC_1.apply_ICA(ICA_exclude = [], show_plots=True) 
ica_C_2 = SubjC_2.apply_ICA(ICA_exclude = [])  
ica_C_3 = SubjC_3.apply_ICA(ICA_exclude = [])  

ica_E_1 = SubjE_1.apply_ICA(ICA_exclude = []) 
ica_E_2 = SubjE_2.apply_ICA(ICA_exclude = [])  
ica_E_3 = SubjE_3.apply_ICA(ICA_exclude = []) 

ica_F_1 = SubjF_1.apply_ICA(ICA_exclude = [])   
ica_F_2 = SubjF_2.apply_ICA(ICA_exclude = [])  
ica_F_3 = SubjF_3.apply_ICA(ICA_exclude = [])    
    
# Create Epochs.

[epoch1_C, label1_C] = SubjC_1.create_epochs(ica_C_1, tmin = 2.1, tmax = 3, show_plots=True) 
[epoch2_C, label2_C] = SubjC_2.create_epochs(ica_C_2, tmin = 2.1, tmax = 3)
[epoch3_C, label3_C] = SubjC_3.create_epochs(ica_C_3, tmin = 2.1, tmax = 3) 

[epoch1_E, label1_E] = SubjE_1.create_epochs(ica_E_1, tmin = 2.2, tmax = 3.1)
[epoch2_E, label2_E] = SubjE_2.create_epochs(ica_E_2, tmin = 2, tmax = 2.9) 
[epoch3_E, label3_E] = SubjE_3.create_epochs(ica_E_3, tmin = 2.2, tmax = 3.1)

[epoch1_F, label1_F] = SubjF_1.create_epochs(ica_F_1, tmin = 1, tmax = 1.9) 
[epoch2_F, label2_F] = SubjF_2.create_epochs(ica_F_2, tmin = 1.3, tmax = 2.2) 
[epoch3_F, label3_F] = SubjF_3.create_epochs(ica_F_3, tmin = 1.4, tmax = 2.3)

# Plot the elctrode positions and the PSD in dB and not dB.
SubjC_1.plot_electrodes_and_psd()

# Now, the data of all the subjects will be concatenated, and separated in training and testing.

epochs_train = np.concatenate((epoch1_C, epoch2_C, epoch1_E,epoch2_E, epoch1_F,epoch2_F))  # Epochs for training.
labels = np.concatenate((label1_C, label2_C, label1_E, label2_E, label1_F,label2_F))       # Labels for training.
epochs_test = np.concatenate((epoch3_C, epoch3_E, epoch3_F))                               # Epochs for testing.
label_test = np.concatenate((label3_C, label3_E, label3_F))                                # Labels for testing.



#%% Obtain features.
       
Train_feats = ProcessDataset(SubjC_1, epochs_train, labels)  # Create a class to process the data.      
Test_feats = ProcessDataset(SubjC_1, epochs_test, label_test)  # Create a class to process the data.

# Obteain the CSP object.
csp = Train_feats.create_CSP(show_plot = True)# Create the CSP




#%% Train and test the models

Subj_All = Model_LR()

# Test and train all the subject at once.
clf = Subj_All.ML_classification(csp, epochs_train, labels, epochs_test, label_test, classif = 'RF')

print('\nAll Subjects:')
# Test Subject C.
Subj_All.test_model(epochs_test, label_test, show_plot = True)
print('\nSubject C:')
# Test Subject C.
Subj_All.test_model(epoch3_C, label3_C )
print('\nSubject E:')
# Test Subject E.
Subj_All.test_model(epoch3_E, label3_E )
print('\nSubject F:')
# Test Subject F.
Subj_All.test_model(epoch3_F, label3_F )

# Save the model.

# with open('LeftRight_Classification_More_Data_all.pkl','wb') as f:
#     pickle.dump(clf,f)


















