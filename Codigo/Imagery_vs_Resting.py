# -*- coding: utf-8 -*-
"""
IMAGERY VS NON-IMAGERY.
In this script, numerous classes used for preprocessing, processing, and classifying imageries vs resting states
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
from sklearn.preprocessing     import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Activation, TimeDistributed, LSTM, concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Embedding, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.layers.normalization import BatchNormalization

from scipy.signal import stft
from scipy.stats import norm
import pywt

#%%
        
class PreprocessDatasetPass:
    """
    This class will preprocess a file containing EEG recordings. 
    It assigns a common reference, apply an ICA, filter the data. It also segments and annotates the data.
    It is used for Imagery vs Non-imagery.
           
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
    
    def type_pos_pas(self, sfreq = 200.0):
        """
        Gets the marker position. 
        Just left and right positions, NO PASS marker
        I takes into account the right, left, and pass class.
        
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
        
        desc = ['left', 'right', 'pass'] # for assigning the movements--> left = 1, right = 2, pass = 3
        
        #Evaluate the markers in order to find the position and time of each marker, as well as ¡ts type.
        for i in range(len(self.markers)-1):
            if self.markers[i]==0 and self.markers[i+1] != 0 and (self.markers[i+1] in [1,2,3]):
                
                mark.append(desc[self.markers[i+1]-1])
                pos.append((i+2))
                time.append((i+2)/sfreq)
            else:
                continue
            
        return mark, pos, time 
    
    def multi_annotations(self, marks, time, window_time = 1.0):
        """
        This method is used in otder to obtain all the possible pairs of annotation possible
        for a future distinctions between Left/Right classes relative to Pass classes.
        INPUTS: 
            marks, time --> The EEG markers and the times thereof.  
            window_time --> Length of the annotation window.
        OUTPUTS:
            annotations_left_right, annotations_left_pass, annotations_right_pass --> Annotations for each pair.
        """
        # Create annotations for Left and Right.
        mark_left_right = []
        time_left_right = []
        for mar in range(len(marks)):
           if marks[mar] != 'pass':
               mark_left_right.append(marks[mar]) 
               time_left_right.append(time[mar]) 
         
        # Create annotations for Left and Pass.
        mark_left_pass = []
        time_left_pass = []
        for mar in range(len(marks)):
           if marks[mar] != 'right':
               mark_left_pass.append(marks[mar]) 
               time_left_pass.append(time[mar]) 
        
        # Create annotations for Right and Pass.
        mark_right_pass = []
        time_right_pass = []
        for mar in range(len(marks)):
           if marks[mar] != 'left':
               mark_right_pass.append(marks[mar]) 
               time_right_pass.append(time[mar]) 
        
        # Save the annotations in a variable.
        self.annotations_left_right = mne.Annotations(time_left_right, window_time, mark_left_right)
        self.annotations_left_pass  = mne.Annotations(time_left_pass, window_time,  mark_left_pass)
        self.annotations_right_pass = mne.Annotations(time_right_pass, window_time, mark_right_pass)     
        
        return self.annotations_left_right, self.annotations_left_pass, self.annotations_right_pass
    
    def add_annotations_and_epochs(self, tmin= 2.1, tmax= 3,  duration = 1):
        """
        This function uses the tye_pos_pas function to annotate the markers obtained from all the combinations
        of the multi_annotations function. In other words, it stacks together Left and Right imageries, and
        differentiate them from the Pass imagery. Furthermore, it creates epochs and balances the data. 
        
        INPUTS:
            tmin, tmax --> Times for segmeting the data.
            duration -->Its the duration of the annotation.
        OUTPUTS:
            All_epochs --> Get the epochs.
            labels --> Get the labels.
        """            
        
        # Use the typo_pos_pas function to obbtain markers. 
        [self.mark, self.pos, self.time] = self.type_pos_pas()                 
        # Use the multiannotation function. 
        [self.annotations_l_r, self.annotations_l_p, self.annotations_r_p] = self.multi_annotations(self.mark, self.time, window_time = 1.0)
        
        # The following lines will join Left and Right movements.
        self.raw.set_annotations(self.annotations_l_r)                         # Set the Left/Right annotation. 
        self.events = mne.events_from_annotations(self.raw)
        self.picks = mne.pick_channels(self.raw.info["ch_names"], ["C3", "Cz", "C4"]) # Only channels C3, C4 and Cz will be considered.
        
        # Concatenate all Left and right epochs 
        self.epochs = mne.Epochs(self.raw, self.events[0], self.events[1], tmin, tmax,
                            picks=self.picks, baseline=None, preload=True)
        # Save the imagery data in a variable.
        self.Left_Right_epochs_data = self.epochs.get_data()
        
        
        # To balance the data, new Pass classes will be created by finding moments in which there are no imageries.
        self.raw.set_annotations(self.annotations_l_p) # Find the Pass classes, for instance, in Left vs Pass.
        self.events = mne.events_from_annotations(self.raw)
        self.picks = mne.pick_channels(self.raw.info["ch_names"], ["C3", "Cz", "C4"]) # Only channels C3, C4 and Cz will be considered.
        
        epochs1 = mne.Epochs(self.raw, self.events[0], self.events[1], tmin, tmax,
                            picks=self.picks, baseline=None, preload=True)
        
        epochs2 = mne.Epochs(self.raw, self.events[0], self.events[1], tmin-tmin, tmax-tmin, # This is done to create new Pass classes.
                            picks=self.picks, baseline=None, preload=True)
        
        e_p1 = epochs1['pass'].get_data()
        e_p2 = epochs2['pass'].get_data()
        # Save the pass data in a variable.
        self.Pass_epochs_data = np.concatenate((e_p1,e_p2))
        
        # concatenate all the data.
        self.All_epochs = np.concatenate((self.Left_Right_epochs_data,self.Pass_epochs_data))
        
        # Set labels (Left/Right --> 1; Pass --> 2)
        labels1 = np.array([1 for i in range(len(self.Left_Right_epochs_data))])
        labels2 = np.array([2 for i in range(len(self.Pass_epochs_data))])
        self.labels = np.concatenate((labels1,labels2))
        
        return self.All_epochs, self.labels
    

#%%    
class ProcessDatasetPass():
    """
    This class will process a file containing EEG recordings. 
    It is a subclass of the PreprocessDataset class.
    Mainly, this class will perform tthe statistical features and the STFT.   
    
       
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
        self.markers= class_pre.markers
        self.mark   = class_pre.mark
        self.time   = class_pre.time
        self.pos    = class_pre.pos
        
        self.epochs = epochs
        self.labels = labels      
        #Only done to plot the CSP.
        self.epochs_subj = class_pre.epochs.get_data()
        self.labels_subj = class_pre.labels 
    
    # The following methods are used for extracting the statistical features.
    def energy(self, signal):
        """
        Used for calculating the Energy of the signnal.
        INPUTS:
            signal --> Signal used to obtain the energy.
        OUTPUTS:
            energy_value --> Energy of the signal.
            
        """
        
        energy_value = 0.0
        
        for x in signal:
            energy_value += (x)**2  # Apply the Energy formula.  
        return energy_value
    
    def all_entropy(self, pdf):
        """
        Calculate the Entropies of the signals.
        INPUTS:
            pdf  ---> The Probability density function. 
        
        RETURNS: 
            entropy, shannon_entropy, log_energy_entropy
        """
        self.shannon_entropy = 0.0
        self.entropy = 0.0
        self.log_energy_entropy = 0.0
        self.pdf = pdf
        
        for freq in self.pdf:
            self.entropy += freq * np.log2(freq)
            self.shannon_entropy += (freq)**2 * (np.log2(freq)**2)
            self.log_energy_entropy += (np.log2(freq))**2
            
        self.entropy = -self.entropy    
        self.shannon_entropy = -self.shannon_entropy
        self.log_energy_entropy = -self.log_energy_entropy
        
        return self.entropy,self.shannon_entropy, self.log_energy_entropy
    
    def feature_extracter(self):
        """
        Extract the Features of channels C3, Cz, and C4.
        OUTPUTS:
            skw_epoch_dic, krt_epoch_dic, energy_epoch_dic, 
            entropy_epoch_dic, shannon_entropy_epoch_dic, log_energy_entropy_epoch_dic --> dictionaries of all the features.
            
        """
        ch = ["C3", "Cz", "C4"] # Define the channels.
        
        # Initialize the dictianaries for each feature.
        self.skw_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
        self.krt_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
        self.energy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
        self.entropy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
        self.shannon_entropy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
        self.log_energy_entropy_epoch_dic = {ch[0]: [], ch[1]: [], ch[2]: []}
        
        # Calculate the features for each epoch. 
        for epoch in self.epochs:
            
            data = epoch
            
            self.pdf_all = [] # Initialize a list for saving the PDF of each epoch.   
            for i in range((data.shape[0])):
                # A channel is selected from an epoch ['C3' --> 0, 'C4' --> 1, 'Cz' --> 2]
                gau = scipy.stats.gaussian_kde(data[i])
                 
                # Values in which the gausian will be evaluated
                dist_space = np.linspace( min(data[i]), max(data[i]), 100 )
                gau_total = np.sum(gau(dist_space))# Normalize, this is the integral under the curve.
                pdf = gau(dist_space)/gau_total
                self.pdf_all.append(pdf)
                
                
            self.skw_epoch = [] # Initialize a list for saving the Skewness of each epoch.  
            for i in range((data.shape[0])):
                self.skw_epoch.append(scipy.stats.skew(data[i], axis=0, bias=True)) # Calculate skw.
                
                
            self.krt_epoch = [] # Initialize a list for saving the Kurtosis of each epoch.    
            for i in range((data.shape[0])):
                self.krt_epoch.append(scipy.stats.kurtosis(data[i], axis=0, fisher=True, bias=True, nan_policy='propagate')) # Calculate krt.
            

            self.energy_epoch = [] # Initialize a list for saving the Energy of each epoch.  
            for i in range((data.shape[0])):
                energy_value = self.energy(data[i]) # Call the Energy method.
                self.energy_epoch.append(energy_value)    
            
        
        
            self.Pi_all = [] # here we save the values for each channel
            
            # Wavelets are needed in order to perform the Shannon Energy, for instance.
            for i in range((data.shape[0])):
               
                coef,_= pywt.cwt(data[i],  np.arange(7,20,0.5), 'morl') # Apply the Wavelet transform
                
                # Now, the following formulas are used:
                    #'https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy'
                [M,N] = coef.shape; # M --> scale number, N --> time segments
                Ej = []
                for j in range(M):
                    Ej.append(sum(abs(coef[j,:])));
                    
                Etot=sum(Ej);
                
                pi = []
                for i in Ej:
                    pi.append(i/Etot)
                self.Pi_all.append(pi)
            self.Pi_all = np.array(self.Pi_all)   
    
            # Save the entropies in lists.
            self.entropy_epoch = []
            self.shannon_entropy_epoch = []
            self.log_energy_entropy_epoch = []
            for i in range((data.shape[0])):
                # S = -sum(pk * log(pk)) --> pk is the Probability Density Function 
                entr, shan, log_en = self.all_entropy(self.pdf_all[i]) # Call the all_entropy method.
                
                self.entropy_epoch.append(entr)
                self.shannon_entropy_epoch.append(shan)
                self.log_energy_entropy_epoch.append(log_en)    
            
            # Save all the calues in dictionaries. 
            self.skw_epoch_dic[ch[0]].append(self.skw_epoch[0]), self.skw_epoch_dic[ch[1]].append(self.skw_epoch[1]), self.skw_epoch_dic[ch[2]].append(self.skw_epoch[2]) 
            self.krt_epoch_dic[ch[0]].append(self.krt_epoch[0]), self.krt_epoch_dic[ch[1]].append(self.krt_epoch[1]), self.krt_epoch_dic[ch[2]].append(self.krt_epoch[2]) 
            self.energy_epoch_dic[ch[0]].append(self.energy_epoch[0]), self.energy_epoch_dic[ch[1]].append(self.energy_epoch[1]), self.energy_epoch_dic[ch[2]].append(self.energy_epoch[2])
            self.shannon_entropy_epoch_dic[ch[0]].append(self.shannon_entropy_epoch[0]), self.shannon_entropy_epoch_dic[ch[1]].append(self.shannon_entropy_epoch[1]), self.shannon_entropy_epoch_dic[ch[2]].append(self.shannon_entropy_epoch[2])
            self.log_energy_entropy_epoch_dic[ch[0]].append(self.log_energy_entropy_epoch[0]), self.log_energy_entropy_epoch_dic[ch[1]].append(self.log_energy_entropy_epoch[1]), self.log_energy_entropy_epoch_dic[ch[2]].append(self.log_energy_entropy_epoch[2])
            self.entropy_epoch_dic[ch[0]].append(self.entropy_epoch[0]), self.entropy_epoch_dic[ch[1]].append(self.entropy_epoch[1]), self.entropy_epoch_dic[ch[2]].append(self.entropy_epoch[2]) 
            
        return self.skw_epoch_dic, self.krt_epoch_dic, self.energy_epoch_dic, self.entropy_epoch_dic, self.shannon_entropy_epoch_dic, self.log_energy_entropy_epoch_dic
    
    def feature_ex(self, skw, krt, energy, entropy, shannon, log):
        """
        Method for organizing the features of all the epochs.
        INPUTS:
            skw, krt, energy, entropy, shannon, log --> dictionaries of each feature.
        OUTPUTS:
            big_feat --> Organized features.
        """
        
        big_feat = []  # Initialize the big_feat list. 
        ch = ['C3','C4', 'Cz'] # Define the channels.
        # Organize the Features.
        for j in range(len(skw['C3'])):
            small_feat = []
            for i in range(len(ch)):
                
                new_small_feat = [skw[ch[i]][j], krt[ch[i]][j], energy[ch[i]][j], entropy[ch[i]][j], shannon[ch[i]][j], log[ch[i]][j]]
                small_feat.extend(new_small_feat)
            big_feat.append(small_feat)
        return big_feat  
    
    # Define the STFT features.
    def ShortTimeFourierTransform(self, fs=200.0, window='hann',nperseg=181, noverlap=180):
        """
        Obtain the  STFT features.
        INPUTS:
            fs --> Sampling frequency.
            window --> Window type.
            nperseg --> Samples per segment.
            noverlap --> Overlap allowance.
        OUTPUT:
            coef --> STFT Coeficients
        """
        # Compute the STFT.
        frec, tim, self.coef = stft(self.epochs, fs = fs, window = window, nperseg = nperseg, noverlap = noverlap)
        self.coef= np.abs(self.coef) # Get the absolute value.
        
        return self.coef
    


#%%
class Model_IR:
    """
    Model Imagery/Resting. Class to organize the AI models with the best reults used to distinguis between an Imagery 
    and an Imagery.
    """
    
    def ML_classification(self, epochs_train, labels_train, epochs_test, labels_test, classif = 'RF'):
        """
        Uses ML classifiers in conjuction with the statistical_features to train and test the motor imageries.
        
        INPUTS:
            epochs_train, labels_train --> Introduce the statistical features and labels for training the models.
            epochs_test, labels_test --> Introduce the statistical features  and labels for testing the models.
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
        
        # set the validation and test data 
        self.cv = ShuffleSplit(10, test_size=0.2, random_state=None)
        self.cv_split = self.cv.split(self.epochs_data_train)
        
        # Assemble a classifier
           
        self.lda = LinearDiscriminantAnalysis()
        self.svm = SVC(kernel ='rbf', gamma = 0.00001, C = 10000)
        self.knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
        self.NB  = GaussianNB()
        self.qda = QuadraticDiscriminantAnalysis()
        self.RF  = RandomForestClassifier(n_estimators = 100, 
                                 min_samples_split = 16, 
                                 min_samples_leaf = 4, 
                                 max_features = 'sqrt', 
                                 max_depth = 8, 
                                 bootstrap = True)
        
        
        # Define a normalizer.
        
        self.scale = StandardScaler()
        
        
        # Use scikit-learn Pipeline with cross_val_score function
        if classif == 'LDA':
            self.clf = Pipeline([('scale', self.scale), ('LDA', self.lda)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'QDA':
            self.clf = Pipeline([('scale', self.scale), ('QDA', self.qda)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'SVM':
            self.clf = Pipeline([('scale', self.scale), ('SVM', self.svm)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'KNN':
            self.clf = Pipeline([('scale', self.scale), ('KNN', self.knn)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        elif classif == 'NB':
            self.clf = Pipeline([('scale', self.scale), ('NB', self.NB)]) # Pipeline firstly extracts features from CSP and then does the LDA classification
        else:
            self.clf = Pipeline([('scale', self.scale), ('RF', self.RF)])
      
        
        self.scores = cross_val_score(self.clf, self.epochs_data_train, self.labels, cv=self.cv, n_jobs=1)
        
        self.clf.fit(self.epochs_data_train[:], self.labels[:])  # We fit the model for making it usable 
        
        
        self.testing_score = self.clf.score(self.epochs_test, self.labels_test) #Testing Data (Never seen before)
        print('Accuracy: ', self.testing_score)
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

        

    def CNN_classification(self, Coef_train, labels, Coef_test, label_test, input_shape = (3,91,181),
                          filter1 = 120, filter2 = 240, dense1 = 164, dense2 = 128, kern_size = (3,3),
                          padding = 'same', activation1 = 'relu', activation2 = 'sigmoid', drop = 0.2,
                          pool_size1 = 2, pool_size2 = 1, lr = 0.0001, dec = 300, mom = 0.8,
                          bch_size = 40, epo = 260):
        """
        
        Uses CNN classifiers in conjuction with STFT to train and test imageries vs the resting state.
        This CNN has 2 CNN layers and 3 dense layers.
        
        INPUTS:
            Coef_train, labels --> Introduce the STFT Coeficients and labels for training the models.
            Coef_test, label_test --> Introduce the STFT Coeficients and labels for testing the models.
            input_shape --> Shape inserted on the first layer of teh CNN.
            filter1, filter2 --> Filter size of layers 1 and 2 of the CNN.
            dense1, dense2 --> Size of Dense layers 1 and 2 of the CNN.
            ker_size --> Kernel size.
            padding --> Padding type.
            activation1 --> Activation function of all the layers except the last one.
            activation2 --> Activation function of the last layer.
            drop --> Dropout ratio.
            pool_size1, pool_size2 --> Pools size of layers 1 and to of the CNN. 
            lr --> Learning rate.
            dec --> Denominator for the decay rate DECAY = lr/dec.
            mom --> Momentum.
            bch_size --> Batch Size.
            epo --> Epochs of the CNN (Iterations).
        OUTPUTS:
            model --> Return the model.
            scale -> Returns the scaler.

        """             
        # Activate the GPU.
        tf.config.list_physical_devices('GPU')
        
        # Initialize the model.
        self.model = Sequential()
        
        # First layer of the CNN.
        self.model.add(Conv2D(filters = filter1, kernel_size = kern_size, padding = padding, activation = activation1,input_shape = input_shape)) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
        self.model.add(MaxPooling2D(pool_size = pool_size1))
        self.model.add(Dropout(drop))
        
        # Second layer of the CNN.
        self. model.add(Conv2D(filters = filter2, kernel_size = kern_size, padding = padding, activation = activation1)) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
        self.model.add(MaxPooling2D(pool_size = pool_size2))
        self.model.add(Dropout(drop))
        
        # Flatten layer.
        self.model.add(Flatten())
     
        # First Dense layer.
        self.model.add(Dense(dense1, activation = activation1))
        self.model.add(Dropout(drop))
        
        # Second Dense layer.
        self.model.add(Dense(dense2, activation = activation1))
        self.model.add(Dropout(drop))
        
        # Last Dense layer.
        self.model.add(Dense(1, activation = activation2))
        
        # Define some parameters (leraning rate, decay rate, and momentum).
        learning_rate = lr 
        decay_rate = learning_rate/dec
        momentum = mom
        
        # Create the optimizer. 
        self.opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        #self.pt = RMSprop(learning_rate=learning_rate, momentum=0.2, decay=decay_rate)
        #self.opt = Adam(learning_rate=learning_rate, decay=decay_rate)
        
        #Compile the model.
        self.model.compile(loss='binary_crossentropy', optimizer= self.opt, metrics=['accuracy']) 
        
        self.model.summary() # Show a summary.
        
        # Separate the data into test and train.
        X_train = Coef_train
        y_train = labels-1
        X_test = Coef_test
        y_test = label_test-1
        
        
        # It's very important to normalize the data and also reshape it in order to correctely fit the model.
        # Reshape it according to --> (NUMBER OP EPOCHS, NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS)
        X_train_re = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.X_train_norm = self.scaler.fit_transform(X_train_re)
        self.X_train_norm = self.X_train_norm.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])
        
        X_test_re = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
        self.X_test_norm = self.scaler.transform(X_test_re)
        self.X_test_norm = self.X_test_norm.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2],X_test.shape[3])
        
        # Fit the model.
        history = self.model.fit(self.X_train_norm, y_train, batch_size = bch_size, epochs = epo, validation_data=(self.X_test_norm, y_test))
        
        # Plot the historical behavior of the model along the interations.
        history_df = pd.DataFrame(self.model.history.history).rename(columns={"loss":"train_loss", "accuracy":"train_accuracy"})
        history_df.plot(figsize=(8,8))
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        
        # Test teh model.
        self.predictions = (self.model.predict(self.X_test_norm) > 0.5).astype("int32")
        self.testing_score = self.model.evaluate(self.X_test_norm, y_test) #Testing Data (Never seen before)
        print('Accuracy: ', self.testing_score)
        print(recall_score(y_test, self.predictions)) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
        print(precision_score(y_test, self.predictions)) 

        return self.model, self.scaler
            
        
#############################################################################################################
#%% Example on what to execute on the Main file (Main_Imagery_vs_Resting.py)

# Firstly, we must define the paths of dataset' files. 

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


#%%  

# #Initialize the classes for each subject.

# SubjC_1 = PreprocessDatasetPass(path = path1_C, show_plots=True)
# SubjC_2 = PreprocessDatasetPass(path = path2_C)
# SubjC_3 = PreprocessDatasetPass(path = path3_C)

# SubjE_1 = PreprocessDatasetPass(path = path1_E)
# SubjE_2 = PreprocessDatasetPass(path = path2_E)
# SubjE_3 = PreprocessDatasetPass(path = path3_E)

# SubjF_1 = PreprocessDatasetPass(path = path1_F)
# SubjF_2 = PreprocessDatasetPass(path = path2_F)
# SubjF_3 = PreprocessDatasetPass(path = path3_F)

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

# # Add annotations and create epochs. 
    
# # Create Epochs.

# [epoch1_C, label1_C] = SubjC_1.add_annotations_and_epochs(tmin = 2.1, tmax = 3) 
# [epoch2_C, label2_C] = SubjC_2.add_annotations_and_epochs(tmin = 2.1, tmax = 3)
# [epoch3_C, label3_C] = SubjC_3.add_annotations_and_epochs(tmin = 2.1, tmax = 3) 

# [epoch1_E, label1_E] = SubjE_1.add_annotations_and_epochs(tmin = 2.2, tmax = 3.1)
# [epoch2_E, label2_E] = SubjE_2.add_annotations_and_epochs(tmin = 2, tmax = 2.9) 
# [epoch3_E, label3_E] = SubjE_3.add_annotations_and_epochs(tmin = 2.2, tmax = 3.1)

# #[epoch1_F, label1_F] = SubjF_1.add_annotations_and_epochs(tmin = 1, tmax = 1.9) 
# [epoch2_F, label2_F] = SubjF_2.add_annotations_and_epochs(tmin = 1.3, tmax = 2.2) 
# [epoch3_F, label3_F] = SubjF_3.add_annotations_and_epochs(tmin = 1.4, tmax = 2.3)


# # Now, the data of all the subjects will be concatenated, and separated in training and testing.

# All_epochs = np.concatenate((epoch1_C, epoch2_C)) # Epochs for training.
# labels = np.concatenate((label1_C, label2_C))     # Labels for training.
# All_epochs_test = epoch3_C                        # Epochs for testing.
# label_test = label3_C                             # Labels for testing.     


# # All_epochs = np.concatenate((epoch1_C, epoch2_C, epoch1_E,epoch2_E, epoch2_F))  # Epochs for training.
# # labels = np.concatenate((label1_C, label2_C, label1_E, label2_E, label2_F))     # Labels for training.
# # All_epochs_test = np.concatenate((epoch3_C, epoch3_E, epoch3_F))                # Epochs for testing.
# # label_test = np.concatenate((label3_C, label3_E, label3_F))                     # Labels for testing. 


#%% Obtain features.
       
# Train_feats = ProcessDatasetPass(SubjC_1, All_epochs, labels)  # Create a class to process the data.      
# Test_feats = ProcessDatasetPass(SubjC_1, All_epochs_test, label_test)  # Create a class to process the data.


# # Obtain the Statistical features.
# C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log = Train_feats.feature_extracter() # All epochs train.
# D_skw, D_krt, D_energy, D_entropy, D_shannon, D_log= Test_feats.feature_extracter()   # All epochs test.

# FEAT = Train_feats.feature_ex(C_skw, C_krt, C_energy, C_entropy, C_shannon, C_log) 
# FEAT = np.array(FEAT) # Training features.

# FEAT_test = Test_feats.feature_ex(D_skw, D_krt, D_energy, D_entropy, D_shannon, D_log)
# FEAT_test = np.array(FEAT_test) # Testing features.        



# # Obtain the STFT coeficients.
# # Train coefs.
# Coef_train = Train_feats.ShortTimeFourierTransform(fs=200.0, window='hann',nperseg=181, noverlap=180)

# # Test coefs.
# Coef_test = Test_feats.ShortTimeFourierTransform(fs=200.0, window='hann',nperseg=181, noverlap=180)

# # Plot a random STFT spectrogram 
# for i in range(3):
#     frec, tim, Zx =stft(All_epochs[-1,i], fs=200.0, window='hann',nperseg=181, noverlap=180)
    
#     plt.figure()
#     plt.pcolormesh(tim, frec, np.abs(Zx), shading='gouraud', cmap='jet',vmin=-0,vmax=0.8e-6)
#     plt.title('STFT Magnitude {}'.format(["C3", "Cz", "C4"][i]), fontsize=30)
#     plt.ylim([5,30])
#     plt.ylabel('Frequency [Hz]', fontsize=20)
#     plt.xlabel('Time [sec]', fontsize=20)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.colorbar()
#     plt.show()

#%%
# Subj_train = Model_IR()

# clf = Subj_train.ML_classification(FEAT, labels, FEAT_test, label_test, classif = 'RF')


# model, scaler = Subj_train.CNN_classification(Coef_train, labels, Coef_test, label_test) 




