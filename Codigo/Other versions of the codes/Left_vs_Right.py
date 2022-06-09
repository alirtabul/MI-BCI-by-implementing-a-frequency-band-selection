# -*- coding: utf-8 -*-
"""
LEFT VS RIGHT-HAND IMAGERY.
In this script, numerous classes used for preprocessing, processing, and classifying left vs right-hand imageries
will be coded.


@author: Ali Abdul Ameer Abbas
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import pymatreader
import scipy.stats
import pandas as pd
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC                         # SVM     
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.naive_bayes import GaussianNB          # Naive Bayesian
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #QDA                         # QDA 
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, precision_score, recall_score, auc, roc_curve
from sklearn.model_selection import RandomizedSearchCV


from scipy.stats import norm
import pywt



#%%        
class PreprocessDataset:
    """
    This class will preprocess a file containing EEG recordings. 
    It assigns a common reference, apply an ICA, filter the data. It also segments and annotates the data.   
           
    """
    
    def __init__(self, path, montage = 'standard_1020', show_plots = False, bad_chan = []):
        
        """
        Intialize the class by reading and reconstructing the signals.
        INPUTS:
            montage --> The reference selected.
            path --> The path of the file.
            bad_channels --> Bad channels to eliminate.
            show_plots --> To show the plots of the EEG recording. 
        """
        # Convert the matlab files into a python compatible file.
        self.struct = pymatreader.read_mat(path, ignore_fields=['previous'], variable_names = 'o')
        self.struct = self.struct['o']
        
        
        self.chann = self.struct['chnames'] # Save the channels names.
        self.chann = self.chann[:-1]        # Eliminate the X3 channel, which is only used for the data adquisition.
        
        #Set important properties for the dataset. 
        self.info  = mne.create_info(ch_names = self.chann, 
                               sfreq = float(self.struct['sampFreq']),ch_types='eeg',
                               montage= montage, verbose=None)
        
        #Create the data variable.
        self.data   = np.array(self.struct['data'])
        self.data   = self.data[:,:-1]
        self.data_V = self.data*1e-6        # mne reads units are in Volts, convert it to uV.
        
        # Create the raw instance for working with the data in the MNE library.
        self.raw   = mne.io.RawArray(self.data_V.transpose(),self.info, copy = "both")
        self.raw.info['bads'] = bad_chan # exclude bad channels
        
        # Show the raw data if show_plot is true.
        if show_plots:
            self.raw.plot(title = 'Raw signal')

    def channel_reference(self, ref = 'average' ):
        
        """
        Set a reference to the data. 
        INPUTS:
            ref --> Reference.
        
        OUTPUTS: 
            raw --> Referenced data.   
        """
        
        self.raw = self.raw.set_eeg_reference(ref, projection= False)
        
        return(self.raw)
        
    def MNE_bandpass_filter(self, HP = 15, LP = 26, filt_type = 'firwin', show_plots = False, skip_by_annotation='edge'):
        
        """
        Apply a BP FIR filter. 
        
        INPUTS:
            LP,HP --> Low Pass filter, High Pass filter.
            filt_type --> filter type.
            show_plots --> To show the plots of the EEG recording. 
        OUTPUTS: 
            raw --> Filtered data.
        """
        self.raw.filter(HP, LP, fir_design = filt_type, skip_by_annotation = 'edge')        
        
        if show_plots:
            self.raw.plot(title = 'Filtered Raw')
            self.raw.plot_psd()
            
            
        return(self.raw)
        
    
    def apply_ICA(self, n_components = 10, ICA_exclude = [0], show_plots = False): 
        
        """
        Apply the ICA for removing eye blinking, EMG signals, EKG signals, among other artifacts.
        
        INPUTS:
            n_components --> Number of components of the ICA.
            ICA_exclude --> Array used to eliminate components from teh signals.
            show_plots --> To show the plots of the EEG recording. 
        OUTPUTS: 
            raw_ica --> Data after ICA.
        """
        # Apply ICA for blink removal 
        self.ica = mne.preprocessing.ICA(n_components = n_components, random_state = 0,  method='fastica')
        
        # The MNE documentation recommends to HP filter the ICA at 1 Hz. 
        filt_raw = self.raw.copy()                 # Create a copy of teh data
        filt_raw.filter(l_freq=1., h_freq=None)    # HP filter at 1 Hz 
        self.ica.fit(filt_raw)                     # Find the ICA parameters. 

        # Plot the ICA components to detect artifacts.   
        if show_plots:
            self.ica.plot_components(outlines = 'skirt', colorbar=True, contours = 0)
            self.ica.plot_sources(self.raw) 
        
        # Once the bad components are detected, we procede to remove them.
        self.ica.exclude = ICA_exclude

        self.raw_ica = self.ica.apply(self.raw.copy(), exclude = self.ica.exclude)
        
        # Plot the results. 
        if show_plots:
            self.raw.plot()
            self.raw_ica.plot()  
        
        return self.raw_ica
    
    def type_pos(self, sfreq = 200.0):
        """
        Gets the marker position. 
        Just left and right positions, NO PASS marker
        Only takes into account the right and left class, not the passive class.
        
        INPUTS:
            sfreq --> Sampling frequency.
        OUTPUTS: 
            mark --> markers.
            pos --> position of the marker.
            time --> time of the marker.        
        """
        self.markers =  np.array(self.struct['marker']).transpose() # Include the markers
        
        # Intitialize the output arrays. 
        mark = []
        pos  = []
        time = []
        
        desc = ['left', 'right'] # for assigning the movements--> left = 1, right = 2, pass = 3
        
        #Evaluate the markers in order to find the position and time of each marker, as well as ¡ts type.
        for i in range(len(self.markers)-1):
            if self.markers[i]==0 and self.markers[i+1] != 0 and (self.markers[i+1] in [1,2]):
                
                mark.append(desc[self.markers[i+1]-1])
                pos.append((i+2))
                time.append((i+2)/sfreq)
            else:
                continue
            
        return mark, pos, time 
    
    def add_annotations(self, duration = 1):
        """
        This function uses the tye_pos function to annotate the markers.
        INPUTS:
            duration -->Its the duration of the annotation.
        """            
        
        [self.mark, self.pos, self.time] = self.type_pos()                 # Use the typo pos function. 

        self.annotations = mne.Annotations(self.time, duration, self.mark) # Annotate the markers.
        self.raw.set_annotations(self.annotations)                         # Set the annotation. 
     
    def create_epochs(self, raw_ica, tmin= 2.1, tmax= 3, show_plots = False):    
        """'
        Segment and label all the data. 
        INPUTS:
            raw_ica --> Preprocessed data.
            tmin, tmax --> times for segmeting the data.
            show_plots --> To show the plots of the EEG recording. 
        OUTPUTS:
            epochs.get_data() --> get the epochs.
            labels --> get the labels.
        """
        self.events = mne.events_from_annotations(self.raw) # extract the events from the annotations.
        
        self.picks = mne.pick_types(self.raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')  # Exclude the bad events.
        
        # Save the epochs inside a variable.
        self.epochs = mne.Epochs(self.raw_ica, self.events[0], event_id = self.events[1],
                                 preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = self.picks)
        # Save the labels inside a variable.
        self.labels = self.epochs.events[:, -1]
        
        # Plot the segmented data.
        if show_plots:
            self.epochs.plot()
        
        return(self.epochs.get_data(), self.labels)
    
    def plot_electrodes_and_psd(self):
        """
        Plot the position of the electrodes and the psd.
        """
    
        fig = self.raw.plot_psd(dB=False, xscale='linear', estimate='power')  # show the power spectrum density 
        fig.suptitle('Power spectral density (PSD)',  fontsize = 30)
        plt.show()
        
        fig = self.raw.plot_psd(dB=True, xscale='linear', average = False )  # show the power spectrum density 
        fig.suptitle('Power spectral density (PSD) (dB)', fontsize = 30)
        plt.show()
        
        
        
        
        fig = plt.figure()
        ax2d = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')
        self.raw.plot_sensors(show_names = True, axes = ax2d)
        self.raw.plot_sensors(show_names = False, axes = ax3d, kind = '3d')
        # Make a sphere to project the 3d electrodes
        u = np.linspace(0, 2 * np.pi, 200)
        v = np.linspace(0, np.pi, 200)
        x = 0.1 * np.outer(np.cos(u), np.sin(v))
        y = 0.1 * np.outer(np.sin(u), np.sin(v))
        z = 0.15 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax3d.plot_surface(x, y, z,alpha = 0.8, color='navajowhite')
        plt.show()
        

      
        
        

#%%    

# Now a class for processing the data will be created. Mainly, this class will perform the CSP, the statistical features, and the STFT.
    
class ProcessDataset():
    """
    This class will process a file containing EEG recordings. 
    It is a subclass of the PreprocessDataset class.
    Mainly, this class will perform the CSP.   
    
       
    """
    
    def __init__(self, class_pre, epochs, labels):
                
        """
        Intialize the class by getting the labels and epochs.
        INPUTS:
            class_pre --> Introduce any class of PreprocessDataset so that important information (info, chnames, ...) are defined. 
            epochs --> Epochs obtained from the PreprocessingDataset class. This epochs should combine multiple subjects.
            labels --> Labels obtained from the PreprocessingDataset class. This epochs should combine multiple subjects.
        """
        # Define important parameters for the class:
 
        self.struct = class_pre.struct
        self.chann  = class_pre.chann
        self.info   = class_pre.info
        self.data   = class_pre.data
        self.data_V = class_pre.data_V
        self.raw    = class_pre.raw
        self.raw_ica= class_pre.raw_ica
        self.ica    = class_pre.ica
        self.markers= class_pre.markers
        self.mark   = class_pre.mark
        self.time   = class_pre.time
        self.pos    = class_pre.pos
        
        self.epochs = epochs
        self.labels = labels      
        #Only done to plot the CSP.
        self.epochs_subj = class_pre.epochs.get_data()
        self.labels_subj = class_pre.labels 
        
        
    # This method is for the CSP.     
    def create_CSP(self, reg = 'oas', n_components = 10, show_plot = False):
        
        """
        Create the CSP to obtain spatial featuers. 
        INPUTS: 
            reg --> reg function. reg may be 'auto', 'empirical', 'diagonal_fixed', 
                    'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
            n_components --> Number of components.
            show_plot --> Compute the CSP for the Subject selected and display the CSP.
                          Notice that the CSP plotted are only from the selected subject
        OUTPUTS:
            csp --> The CSP.
        """
        # Create the CSP structure.
        self.csp = CSP(n_components= n_components, reg = reg, rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur
        
        # Plot the CSP.
        if show_plot: 
            #IMPORTANT, set_baseline, must be 0
            csp_subj = CSP(n_components= n_components, reg = reg, rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur.  
            csp_subj.fit_transform(self.epochs_subj, self.labels_subj)
    
            csp_subj.plot_patterns(self.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
            
        return(self.csp)         

    
 

#%% 

class Model_LR:
    """
    Model Left/Right. Class to organize the AI models with the best reults used to distinguis between Left and Right-hand imageries.
    """
    
    def ML_classification(self, csp, epochs_train, labels_train, epochs_test, labels_test, classif = 'RF'):
        """
        Uses ML classifiers in conjuction with CSP to train and test the motor imageries.
        
        INPUTS:
            csp --> Introduce the CSP.
            epochs_train, labels_train --> Introduce the epochs and labels for training the models.
            epochs_test, labels_test --> Introduce the epochs and labels for testing the models.
            classif --> The classifier used. (LDA, RF, SVM, KNN, NB, QDA)
        OUTPUTS:
            clf --> return the model.
        """
        
        # Initialize a score array.
        self.scores = []
        
        self.labels = labels_train 
        self.epochs_data_train = epochs_train
        
        self.epochs_test = epochs_test
        self.labels_test = labels_test
        
        self.csp = csp
        # set the validation and test data 
        self.cv = ShuffleSplit(10, test_size=0.2, random_state=None)
        self.cv_split = self.cv.split(self.epochs_data_train)

        # Assemble a classifier
           
        self.lda = LinearDiscriminantAnalysis()
        self.svm = SVC(kernel ='rbf', gamma = 0.00001, C = 10000)
        self.knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
        self.NB  = GaussianNB()
        self.RF  = RandomForestClassifier(n_estimators = 100, 
                                     min_samples_split = 16, 
                                     min_samples_leaf = 4, 
                                     max_features = 'sqrt', 
                                     max_depth = 8, 
                                     bootstrap = True)
        self.qda = QuadraticDiscriminantAnalysis()

        


        # Use scikit-learn Pipeline with cross_val_score function
        if classif == 'LDA':
            self.clf = Pipeline([('CSP', self.csp), ('LDA', self.lda)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'QDA':
            self.clf = Pipeline([('CSP', self.csp), ('QDA', self.qda)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'SVM':
            self.clf = Pipeline([('CSP', self.csp), ('SVM', self.svm)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'KNN':
            self.clf = Pipeline([('CSP', self.csp), ('KNN', self.knn)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'NB':
            self.clf = Pipeline([('CSP', self.csp), ('NB', self.NB)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        else:
            self.clf = Pipeline([('CSP', self.csp), ('RF', self.RF)])

        self.scores = cross_val_score(self.clf, self.epochs_data_train, self.labels, cv=self.cv, n_jobs=1)
        
        self.clf.fit(self.epochs_data_train[:], self.labels[:])  # We fit the model for making it usable 
        
        
        self.testing_score = self.clf.score(self.epochs_test, self.labels_test) #Testing Data (Never seen before)
        print('\nAccuracy: ', self.testing_score)
        print('Precision: ', precision_score(self.labels_test[:], self.clf.predict(self.epochs_test[:],))) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
        print('Recall: ', recall_score(self.labels_test[:], self.clf.predict(self.epochs_test,))) 

        # Printing the results
        self.class_balance = np.mean(self.labels == self.labels[0])
        self.class_balance = max(self.class_balance, 1. - self.class_balance)
        print("Classification accuracy: %f +- %f / Chance level: %f" % (np.mean(self.scores), np.std(self.scores),
                                                                  self.class_balance))
        plt.rc('xtick', labelsize=20) 
        plt.rc('ytick', labelsize=20)
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', titlesize=30, labelsize=25)
        
        plot_confusion_matrix(self.clf, self.epochs_test, self.labels_test)
        plt.show()
        
        return self.clf
    
    def test_model(self, epochs_test, labels_test, show_plot = False):
        """
        The only purpose of this function is to test the models for each subject.
        It assess accuracy, precision, and recall.
        Also plots the confusion matrix.
        Always run this method after the ML_classification function.
        INPUTS:
            epochs_test --> Epochs to test.
            labels_test --> Labels to have the ground truth.
            show_plot --> To show the confunsion matrix.
        """
        
        self.testing_score = self.clf.score(epochs_test, labels_test) #Testing Data (Never seen before)
        print('Accuracy: ', self.testing_score)
        print('Precision: ', precision_score(labels_test[:], self.clf.predict(epochs_test[:],))) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
        print('Recall: ', recall_score(labels_test[:], self.clf.predict(epochs_test,))) 
        
        if show_plot:
            plt.rc('xtick', labelsize=20) 
            plt.rc('ytick', labelsize=20)
            plt.rcParams.update({'font.size': 16})
            plt.rc('axes', titlesize=30, labelsize=25)
        
            plot_confusion_matrix(self.clf, epochs_test, labels_test)
            plt.show()





       
#############################################################################################################
#%% Example on what to execute on the Main file (Main_Left_vs_Right.py)

# # Firstly, we must define the paths of dataset' files. 

# path1_C = '../Data/CLA/CLASubjectC1512233StLRHand.mat' 
# path2_C = '../Data/CLA/CLASubjectC1512163StLRHand.mat' 
# path3_C = '../Data/CLA/CLASubjectC1511263StLRHand.mat'


# path1_B = '../Data/CLA/CLASubjectB1510193StLRHand.mat' 
# path2_B = '../Data/CLA/CLASubjectB1510203StLRHand.mat' 
# path3_B = '../Data/CLA/CLASubjectB1512153StLRHand.mat'


# path1_E = '../Data/CLA/CLASubjectE1512253StLRHand.mat' 
# path2_E = '../Data/CLA/CLASubjectE1601193StLRHand.mat' 
# path3_E = '../Data/CLA/CLASubjectE1601223StLRHand.mat'


# path1_F = '../Data/CLA/CLASubjectF1509163StLRHand.mat' 
# path2_F = '../Data/CLA/CLASubjectF1509173StLRHand.mat' 
# path3_F = '../Data/CLA/CLASubjectF1509283StLRHand.mat'

# fs= 200 # Define the sampling frequency.
# #%% Create a class to preprocess the data.
# plt.rc('xtick', labelsize=20) 
# plt.rc('ytick', labelsize=20)
# plt.rc('axes', titlesize=30, labelsize=25)
 


# #%%    
# # Apply the preprocessing class to the data.

# #Initialize the classes for each subject.

# SubjC_1 = PreprocessDataset(path = path1_C, show_plots=True)
# SubjC_2 = PreprocessDataset(path = path2_C)
# SubjC_3 = PreprocessDataset(path = path3_C)

# SubjE_1 = PreprocessDataset(path = path1_E)
# SubjE_2 = PreprocessDataset(path = path2_E)
# SubjE_3 = PreprocessDataset(path = path3_E)

# SubjF_1 = PreprocessDataset(path = path1_F)
# SubjF_2 = PreprocessDataset(path = path2_F)
# SubjF_3 = PreprocessDataset(path = path3_F)

# # Set the reference.
# SubjC_1.channel_reference() 
# SubjC_2.channel_reference() 
# SubjC_3.channel_reference() 

# SubjE_1.channel_reference()
# SubjE_2.channel_reference() 
# SubjE_3.channel_reference()

# SubjF_1.channel_reference()  
# SubjF_2.channel_reference() 
# SubjF_3.channel_reference()

# # Filter the data.

# SubjC_1.MNE_bandpass_filter(HP = 15, LP = 26, show_plots=True) 
# SubjC_2.MNE_bandpass_filter(HP = 15, LP = 26) 
# SubjC_3.MNE_bandpass_filter(HP = 15, LP = 26) 

# SubjE_1.MNE_bandpass_filter(HP = 15, LP = 26)
# SubjE_2.MNE_bandpass_filter(HP = 15, LP = 26) 
# SubjE_3.MNE_bandpass_filter(HP = 15, LP = 26)

# SubjF_1.MNE_bandpass_filter(HP = 15, LP = 26)  
# SubjF_2.MNE_bandpass_filter(HP = 15, LP = 26) 
# SubjF_3.MNE_bandpass_filter(HP = 15, LP = 26)

# # Add annotations.

# SubjC_1.add_annotations() 
# SubjC_2.add_annotations() 
# SubjC_3.add_annotations() 

# SubjE_1.add_annotations()
# SubjE_2.add_annotations() 
# SubjE_3.add_annotations()

# SubjF_1.add_annotations()  
# SubjF_2.add_annotations() 
# SubjF_3.add_annotations()
    
# # Apply ICA.

# ica_C_1 = SubjC_1.apply_ICA(ICA_exclude = [], show_plots=True) 
# ica_C_2 = SubjC_2.apply_ICA(ICA_exclude = [])  
# ica_C_3 = SubjC_3.apply_ICA(ICA_exclude = [])  

# ica_E_1 = SubjE_1.apply_ICA(ICA_exclude = []) 
# ica_E_2 = SubjE_2.apply_ICA(ICA_exclude = [])  
# ica_E_3 = SubjE_3.apply_ICA(ICA_exclude = []) 

# ica_F_1 = SubjF_1.apply_ICA(ICA_exclude = [])   
# ica_F_2 = SubjF_2.apply_ICA(ICA_exclude = [])  
# ica_F_3 = SubjF_3.apply_ICA(ICA_exclude = [])    
    
# # Create Epochs.

# [epoch1_C, label1_C] = SubjC_1.create_epochs(ica_C_1, tmin = 2.1, tmax = 3, show_plots=True) 
# [epoch2_C, label2_C] = SubjC_2.create_epochs(ica_C_2, tmin = 2.1, tmax = 3)
# [epoch3_C, label3_C] = SubjC_3.create_epochs(ica_C_3, tmin = 2.1, tmax = 3) 

# [epoch1_E, label1_E] = SubjE_1.create_epochs(ica_E_1, tmin = 2.2, tmax = 3.1)
# [epoch2_E, label2_E] = SubjE_2.create_epochs(ica_E_2, tmin = 2, tmax = 2.9) 
# [epoch3_E, label3_E] = SubjE_3.create_epochs(ica_E_3, tmin = 2.2, tmax = 3.1)

# [epoch1_F, label1_F] = SubjF_1.create_epochs(ica_F_1, tmin = 1, tmax = 1.9) 
# [epoch2_F, label2_F] = SubjF_2.create_epochs(ica_F_2, tmin = 1.3, tmax = 2.2) 
# [epoch3_F, label3_F] = SubjF_3.create_epochs(ica_F_3, tmin = 1.4, tmax = 2.3)

# # Plot the elctrode positions and the PSD in dB and not dB.
# SubjC_1.plot_electrodes_and_psd()

# # Now, the data of all the subjects will be concatenated, and separated in training and testing.

# epochs_train = np.concatenate((epoch1_C, epoch2_C, epoch1_E,epoch2_E, epoch1_F,epoch2_F))  # Epochs for training.
# labels = np.concatenate((label1_C, label2_C, label1_E, label2_E, label1_F,label2_F))       # Labels for training.
# epochs_test = np.concatenate((epoch3_C, epoch3_E, epoch3_F))                               # Epochs for testing.
# label_test = np.concatenate((label3_C, label3_E, label3_F))                                # Labels for testing.   




# #%% Obtain features.
      
# Train_feats = ProcessDataset(SubjC_1, epochs_train, labels)  # Create a class to process the data.      
# Test_feats = ProcessDataset(SubjC_1, epochs_test, label_test)  # Create a class to process the data.

# # Obteain the CSP object.
# csp = Train_feats.create_CSP(show_plot = True)# Create the CSP



# #%% Train and test the models
# Subj_All = Model_LR()

# # Test and train all the subject at once.
# clf = Subj_All.ML_classification(csp, epochs_train, labels, epochs_test, label_test, classif = 'RF')

# print('\nAll Subjects:')
# # Test Subject C.
# Subj_All.test_model(epochs_test, label_test, show_plot = True)
# print('\nSubject C:')
# # Test Subject C.
# Subj_All.test_model(epoch3_C, label3_C )
# print('\nSubject E:')
# # Test Subject E.
# Subj_All.test_model(epoch3_E, label3_E )
# print('\nSubject F:')
# # Test Subject F.
# Subj_All.test_model(epoch3_F, label3_F )



