# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:45:50 2020

@author: aliab
"""
import mne
import matplotlib # IMPORTANT--> It must be version 3.2.2
import pylab
import pymatreader 
import pandas as pd
import numpy as np

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import matplotlib.pyplot as plt

class MotorImagery:
    
    import mne
    import matplotlib # IMPORTANT--> It must be version 3.2.2
    import pylab
    import pymatreader 
    import pandas as pd
    import numpy as np

    from mne.decoding import CSP
    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    import matplotlib.pyplot as plt

    def __init__(self, path, montage = 'standard_1020', show_plots = True, bad_chan = []):
        
        self.struct = pymatreader.read_mat(path, ignore_fields=['previous'], variable_names = 'o')
        self.struct = self.struct['o']
        
        self.chann = self.struct['chnames']
        self.chann = self.chann[:-1]            # pop th X3 channel, which is only used for the data adquisition.
        self.info  = mne.create_info(ch_names = self.chann, 
                               sfreq = float(self.struct['sampFreq']),ch_types='eeg', montage= montage, verbose=None)
        
        self.data   = np.array(self.struct['data'])
        self.data   = self.data[:,:-1]
        self.data_V = self.data*1e-6            # mne reads units are in Volts
        
        self.raw   = mne.io.RawArray(self.data_V.transpose(),self.info, copy = "both")
        # exclude bad channels
        self.raw.info['bads'] = bad_chan
        
        
        if show_plots:
            self.raw.plot(title = 'Raw signal')
    
    def channel_reference(self, ref = 'average' ):
        
        self.raw = self.raw.set_eeg_reference(ref, projection= False)
        
        return(self.raw)
        
    def bandpass_filter(self, HP = 7, LP = 30, filt_type = 'firwin', ref = True, show_plots = True):
        
        if ref:                     # set reference if desired
            self.channel_reference(ref = 'average' )
        
        else:
            pass
        
        self.raw.filter(HP, LP, fir_design = filt_type, skip_by_annotation = 'edge')        
        
        if show_plots:
            self.raw.plot(title = 'Filtered Raw')
            self.raw.plot_psd()
            
            
        return(self.raw)
    
    def apply_ICA(self, n_components = 10, ICA_exclude = [0], show_plots = True): 
        
        # ICA for blink removal 

        self.ica = mne.preprocessing.ICA(n_components = n_components, random_state = 0,  method='fastica')
        
        self.filt_raw = self.raw.copy()
        self.filt_raw.filter(l_freq=1., h_freq=None) # it's recommended to HP filter at 1 Hz 
        self.ica.fit(self.filt_raw)

        # We plot (also topographically) the ICA components to exclude the blinkings.  
        self.ica.plot_components(outlines = 'skirt', colorbar=True, contours = 0)
        self.ica.plot_sources(self.raw) 
        
        # Once the bad components are detected, we procide to remove them 

        self.ica.exclude = ICA_exclude

        self.raw_ica = self.ica.apply(self.raw.copy(), exclude = self.ica.exclude)
        
        self.raw.plot()
        self.raw_ica.plot()  
        # If we compare the two graphs... MAGIC! Blinks are gone :)
        
        return self.raw_ica
    
    def type_pos(self, markers, sfreq = 200.0):
        """
        Gets the marker position. 
        Just left and right positions, NO PASS marker
        Only takes into account the right and left class, not the passive class
        """
    
        self.mark = []
        self.pos  = []
        self.time = []
        self.desc = ['left', 'right'] # for assigning the movements--> left = 1, right = 2, pass = 3
        for i in range(len(markers)-1):
            if markers[i]==0 and markers[i+1] != 0 and (markers[i+1] in [1,2]):
                
                self.mark.append(self.desc[markers[i+1]-1])
                self.pos.append((i+2))
                self.time.append((i+2)/sfreq)
            else:
                continue
            
        return self.mark, self.pos, self.time 
    
    def add_annotations(self, duration = 1):
                    
        
        self.markers =  np.array(self.struct['marker']).transpose()
        [self.mark, self.pos, self.time] = self.type_pos(self.markers)

        self.annotations = mne.Annotations(self.time, duration, self.mark)
        self.raw.set_annotations(self.annotations)
         
    def create_epochs(self, raw_ica, tmin= 0.6, tmax=1.4 ):    
        
        self.events = mne.events_from_annotations(self.raw)
        self.events
        self.picks = mne.pick_types(self.raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')  
  
        self.epochs = mne.Epochs(self.raw_ica, self.events[0], event_id = self.events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = self.picks)
        self.epochs.plot()
        
        return(self.epochs)
        
    def evoked_potentials(self, epochs, Laplacian = True, show_plots = True):
        
        self.e_r = epochs['right']
        self.e_l = epochs['left']
        
        self.right = self.e_r.average()
        self.left  = self.e_l.average()
        
        if Laplacian:
            
            self.right_csd =  mne.preprocessing.compute_current_source_density(self.right)
            self.left_csd =  mne.preprocessing.compute_current_source_density(self.left)


            self.right_csd.plot_joint(title='Current Source Density (RIGHT)')
            self.left_csd.plot_joint(title ='Current Source Density (LEFT)')

    def apply_CSP(self, epochs, reg = 'oas', n_components = 10 ):
        
        self.labels       = epochs.events[:, -1] # In documentation they put a -2 after the []
        self.epochs_train = epochs.copy()
        self.epochs_data  = epochs.get_data()
        self.epochs_data_train = self.epochs_train.get_data()

        #IMPORTANT, set_baseline, must be 0
        # reg may be 'auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
        self.csp = CSP(n_components= n_components, reg = reg, rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur  
        self.csp.fit_transform(self.epochs_data, self.labels)

        self.csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        
        return(self.csp)
    
    def LDA_classification(self, csp, epochs, over_time = True):
        
        self.scores = []
        
        self.labels       = epochs.events[:, -1] # In documentation they put a -2 after the []
        self.epochs_train = epochs.copy()
        self.epochs_data  = epochs.get_data()
        self.epochs_data_train = self.epochs_train.get_data()
        
        # set the validation and test data 
        self.cv = ShuffleSplit(100, test_size=0.2, random_state=None)
        self.cv_split = self.cv.split(self.epochs_data_train)

        # Assemble a classifier
           
        
        lda = LinearDiscriminantAnalysis()
        


        # Use scikit-learn Pipeline with cross_val_score function
        self.clf = Pipeline([('CSP', csp), ('LDA', lda)]) # Pipepline firstly extracts features from CSP and then does the LDA classification
        self.scores = cross_val_score(self.clf, self.epochs_data_train, self.labels, cv=self.cv, n_jobs=1)

        # Printing the results
        self.class_balance = np.mean(self.labels == self.labels[0])
        self.class_balance = max(self.class_balance, 1. - self.class_balance)
        print("Classification accuracy: %f / Chance level: %f" % (np.mean(self.scores),
                                                                  self.class_balance))
        
        if over_time:
            
            # plot CSP patterns estimated on full data for visualization
            csp.fit_transform(self.epochs_data, self.labels)

            csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)


            self.sfreq = self.raw.info['sfreq']
            self.w_length = int(self.sfreq * 0.5)   # running classifier: window length
            self.w_step = int(self.sfreq * 0.1)  # running classifier: window step size
            self.w_start = np.arange(0, self.epochs_data.shape[2] - self.w_length, self.w_step)

            self.scores_windows = []

            for train_idx, test_idx in self.cv_split:
                self.y_train, self.y_test = self.labels[train_idx], self.labels[test_idx]

                self.X_train = csp.fit_transform(self.epochs_data_train[train_idx], self.y_train)
                self.X_test = csp.transform(self.epochs_data_train[test_idx])

                # fit classifier
                lda.fit(self.X_train, self.y_train)

                # running classifier: test classifier on sliding window
                self.score_this_window = []
                for n in self.w_start:
                    self.X_test = csp.transform(self.epochs_data[test_idx][:, :, n:(n + self.w_length)])
                    self.score_this_window.append(lda.score(self.X_test, self.y_test))
                self.scores_windows.append(self.score_this_window)

            # Plot scores over time
            self.w_times = (self.w_start + self.w_length / 2.) / self.sfreq + epochs.tmin

            plt.figure()
            plt.plot(self.w_times, np.mean(self.scores_windows, 0), label='Score')
            plt.axvline(0, linestyle='--', color='k', label='Onset')
            plt.axhline(0.5, linestyle='-', color='k', label='Chance')
            plt.xlabel('time (s)')
            plt.ylabel('classification accuracy')
            plt.title('Classification score over time')
            plt.legend(loc='lower right')
            plt.show()

        return np.mean(self.scores)
        
     
        
    
    
#%% Algorithm for finding the most suitable band-width   
path = '../Data/CLA/CLASubjectA1601083StLRHand.mat'
path_2 = '../Data/CLA/CLASubjectC1512233StLRHand.mat'
subj1 = MotorImagery(path, show_plots = False)
HP = 8 
LP = 12
accur = []
for i in range(7):
    subj1 = MotorImagery(path, show_plots = False) # IMPORTANT TO REINITIALIZE RAW
    subj1.bandpass_filter( HP=HP, LP =LP, show_plots = False)
    HP += 2
    LP += 2
    subj1.add_annotations()

    subj1.raw.annotations

    raw_ICA_1 = subj1.apply_ICA(ICA_exclude = [])

    epochs = subj1.create_epochs(raw_ICA_1, tmin= 0.6, tmax= 1.4)
    epochs

    subj1.evoked_potentials(epochs)

    csp = subj1.apply_CSP(epochs)

    acc = subj1.LDA_classification(csp,epochs, over_time = False)
    accur.append(acc)
    
    plt.plot(accur)


subj1 = MotorImagery(path, show_plots = False)
subj1.bandpass_filter( HP=16, LP =20, show_plots = False)

subj1.add_annotations()
subj1.raw.annotations

raw_ICA_1 = subj1.apply_ICA(ICA_exclude = [])

epochs = subj1.create_epochs(raw_ICA_1, tmin= 2.4, tmax=3.2)
epochs

csp = subj1.apply_CSP(epochs,n_components = 10)
acc = subj1.LDA_classification(csp, epochs, over_time = False)

#%%
