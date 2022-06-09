# -*- coding: utf-8 -*-
"""
Main script for training and testing the AI models for the Imagery vs Resting state.

@author: Ali Abdul Ameer Abbas
"""

# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pickle

# Import the modules that were created in Imagery_vs_Resting.py
from Imagery_vs_Resting import PreprocessDatasetPass, ProcessDatasetPass, Model_IR

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

#Initialize the classes for each subject.

SubjC_1 = PreprocessDatasetPass(path = path1_C, show_plots=True)
SubjC_2 = PreprocessDatasetPass(path = path2_C)
SubjC_3 = PreprocessDatasetPass(path = path3_C)

SubjE_1 = PreprocessDatasetPass(path = path1_E)
SubjE_2 = PreprocessDatasetPass(path = path2_E)
SubjE_3 = PreprocessDatasetPass(path = path3_E)

SubjF_1 = PreprocessDatasetPass(path = path1_F)
SubjF_2 = PreprocessDatasetPass(path = path2_F)
SubjF_3 = PreprocessDatasetPass(path = path3_F)

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

# Add annotations and create epochs. 
    
# Create Epochs.

[epoch1_C, label1_C] = SubjC_1.add_annotations_and_epochs(tmin = 2.1, tmax = 3) 
[epoch2_C, label2_C] = SubjC_2.add_annotations_and_epochs(tmin = 2.1, tmax = 3)
[epoch3_C, label3_C] = SubjC_3.add_annotations_and_epochs(tmin = 2.1, tmax = 3) 

[epoch1_E, label1_E] = SubjE_1.add_annotations_and_epochs(tmin = 2.2, tmax = 3.1)
[epoch2_E, label2_E] = SubjE_2.add_annotations_and_epochs(tmin = 2, tmax = 2.9) 
[epoch3_E, label3_E] = SubjE_3.add_annotations_and_epochs(tmin = 2.2, tmax = 3.1)

#[epoch1_F, label1_F] = SubjF_1.add_annotations_and_epochs(tmin = 1, tmax = 1.9) 
[epoch2_F, label2_F] = SubjF_2.add_annotations_and_epochs(tmin = 1.3, tmax = 2.2) 
[epoch3_F, label3_F] = SubjF_3.add_annotations_and_epochs(tmin = 1.4, tmax = 2.3)


# Now, the data of all the subjects will be concatenated, and separated in training and testing.

All_epochs = np.concatenate((epoch1_C, epoch2_C)) # Epochs for training.
labels = np.concatenate((label1_C, label2_C))     # Labels for training.
All_epochs_test = epoch3_C                        # Epochs for testing.
label_test = label3_C                             # Labels for testing.     


# All_epochs = np.concatenate((epoch1_C, epoch2_C, epoch1_E,epoch2_E, epoch2_F))  # Epochs for training.
# labels = np.concatenate((label1_C, label2_C, label1_E, label2_E, label2_F))     # Labels for training.
# All_epochs_test = np.concatenate((epoch3_C, epoch3_E, epoch3_F))                # Epochs for testing.
# label_test = np.concatenate((label3_C, label3_E, label3_F))                     # Labels for testing. 

#%% Obtain features.
       
Train_feats = ProcessDatasetPass(SubjC_1, All_epochs, labels)  # Create a class to process the data.      
Test_feats = ProcessDatasetPass(SubjC_1, All_epochs_test, label_test)  # Create a class to process the data.


# Obtain the Statistical features.
C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log = Train_feats.feature_extracter() # All epochs train.
D_skw, D_krt, D_energy, D_entropy, D_shannon, D_log= Test_feats.feature_extracter()   # All epochs test.

FEAT = Train_feats.feature_ex(C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log) 
FEAT = np.array(FEAT) # Training features.

FEAT_test = Test_feats.feature_ex(D_skw, D_krt, D_energy, D_entropy, D_shannon, D_log)
FEAT_test = np.array(FEAT_test) # Testing features.        



# Obtain the STFT coeficients.
# Train coefs.
Coef_train = Train_feats.ShortTimeFourierTransform(fs=200.0, window='hann',nperseg=181, noverlap=180)

# Test coefs.
Coef_test = Test_feats.ShortTimeFourierTransform(fs=200.0, window='hann',nperseg=181, noverlap=180)

# Plot a random STFT spectrogram 
for i in range(3):
    frec, tim, Zx =stft(All_epochs[-1,i], fs=200.0, window='hann',nperseg=181, noverlap=180)
    
    plt.figure()
    plt.pcolormesh(tim, frec, np.abs(Zx), shading='gouraud', cmap='jet',vmin=-0,vmax=0.8e-6)
    plt.title('STFT Magnitude {}'.format(["C3", "Cz", "C4"][i]), fontsize=30)
    plt.ylim([5,30])
    plt.ylabel('Frequency [Hz]', fontsize=20)
    plt.xlabel('Time [sec]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.colorbar()
    plt.show()
    
#%%
Subj_train = Model_IR()
## Train the ML model with Statitical Features.
clf = Subj_train.ML_classification(FEAT, labels, FEAT_test, label_test, classif = 'RF')

# Save the ML model:

# with open('RightLeft_vs_Pass_Classification_all.pkl','wb') as f:
#     pickle.dump(clf,f)

#%% Train the CNN model with STFT.
model, scaler = Subj_train.CNN_classification(Coef_train, labels, Coef_test, label_test) 

# Save the CNN model:

# model.save('Pass_vs_Left_Right_with_DL_all.h5')

# #MinMaxScaler
# with open('Pass_vs_Left_Right_with_DL_Scaler_all.pkl','wb') as f:
#     pickle.dump(scaler,f)  