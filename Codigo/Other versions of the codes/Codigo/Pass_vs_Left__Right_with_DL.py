# -*- coding: utf-8 -*-
"""
In this program we are going to mix the Right and Left Classes and create new pass classes
in order to augment the data that we have. Then we are going to classifing using the spectrum of the electrodes.
Moreover, we will train the model with two different sessions from Pacient C.
"""


import numpy as np
import matplotlib.pyplot as plt
import mne
import pymatreader
import scipy.stats
import pandas as pd

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

def epoch_event_pass(path, tmin= 2.1, tmax=3, LP=15,HP=21):
    '''
    This function extracts the events and epochs from a certain file.
    LP,HP --> Low Pass filter, High Pass filter
    
    '''
    
    # Now we simply pass the matlab files into a python compatible file.
    struct = pymatreader.read_mat(path, ignore_fields=['previous'], variable_names = 'o')
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
    raw.filter(LP, HP, fir_design='firwin', skip_by_annotation='edge') # normally 7-30 Hz

    
    mne.compute_rank(raw, rank='info')
    
        
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

    Left_Right_epochs_data = epochs.get_data()
    
    # Let's create the Pass epochs by having the exact marker times and 2 secons after it ##################################################################
    
    
    raw.set_annotations(annotations_l_p)
    
    events = mne.events_from_annotations(raw)
    
    picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
    
    epochs1 = mne.Epochs(raw, events[0], events[1], tmin, tmax,
                        picks=picks, baseline=None, preload=True)
    
    epochs2 = mne.Epochs(raw, events[0], events[1], tmin-tmin, tmax-tmin,
                        picks=picks, baseline=None, preload=True)
    
    e_p1 = epochs1['pass'].get_data()
    e_p2 = epochs2['pass'].get_data()

    
    Pass_epochs_data = np.concatenate((e_p1,e_p2))
    

    All_epochs = np.concatenate((Left_Right_epochs_data,Pass_epochs_data))
    
    labels1 = np.array([1 for i in range(len(Left_Right_epochs_data))])
    labels2 = np.array([2 for i in range(len(Pass_epochs_data))])
    labels = np.concatenate((labels1,labels2))
    
    return All_epochs, labels 


[epoch1, label1] = epoch_event_pass(path1_C, tmin= 2.1, tmax=3, LP=15,HP=26)   
[epoch2, label2] = epoch_event_pass(path2_C, tmin= 2.1, tmax=3, LP=15,HP=26)  
[All_epochs_test, label_test] = epoch_event_pass(path3_C, tmin= 2.1, tmax=3, LP=15,HP=26)# for testing
All_epochs = np.concatenate((epoch1,epoch2))
labels = np.concatenate((label1,label2))

#%% Let's compute the Short Time Fourier Transform (STFT).

from scipy.signal import stft

#Example
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

#%% Let's do the Fourier Transform to all our data
frec, tim, Coef_train =stft(All_epochs, fs=200.0, window='hann',nperseg=181, noverlap=180)
print(frec.shape,tim.shape, Coef_train.shape)
Coef_train= np.abs(Coef_train)

frec, tim, Coef_test =stft(All_epochs_test, fs=200.0, window='hann',nperseg=181, noverlap=180)
Coef_test= np.abs(Coef_test)


#%% Let's rty to train a hybrid Deep learning model with CNN and LSTM
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Activation, TimeDistributed, LSTM, concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Embedding, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, precision_score, recall_score, auc, roc_curve


tf.config.list_physical_devices('GPU')

model = Sequential()

model.add(Conv2D(filters=120, kernel_size=(3,3), padding= 'same', activation='relu',input_shape=(3,91,181))) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=240, kernel_size=(3,3), padding= 'same', activation='relu')) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))


# model.add(Conv1D(filters=120, kernel_size=2, padding= 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=1))
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=60, kernel_size=2, padding= 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=1))
# model.add(Dropout(0.3))


# model.add(LSTM(40,return_sequences=True, activation='tanh'))
# model.add(Dropout(0.2))

model.add(Flatten())


# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(12,activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(164,activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))


learning_rate = 0.0001
decay_rate = learning_rate/300
momentum = 0.8
opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#opt = RMSprop(learning_rate=learning_rate, momentum=0.2, decay=decay_rate)
#opt = Adam(learning_rate=learning_rate, decay=decay_rate)
model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy']) 

model.summary()

#Normalize the data
X_train = Coef_train
y_train = labels-1
X_test = Coef_test
y_test = label_test-1


# It's very important to normalize the data and also reshape it in order to correctely fit the model
# To reshape it we must put (NUMBER OP EPOCHS, NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS)
X_train_re = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
scaler = MinMaxScaler(feature_range=(0,1))
X_train_norm = scaler.fit_transform(X_train_re)
X_train_norm = X_train_norm.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])

X_test_re = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
X_test_norm = scaler.transform(X_test_re)
X_test_norm = X_test_norm.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2],X_test.shape[3])


history = model.fit(X_train_norm, y_train, batch_size=40, epochs=260, validation_data=(X_test_norm, y_test))

predictions = (model.predict(X_test_norm) > 0.5).astype("int32")

history_df = pd.DataFrame(model.history.history).rename(columns={"loss":"train_loss", "accuracy":"train_accuracy"})
history_df.plot(figsize=(8,8))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

testing_score = model.evaluate(X_test_norm, y_test) #Testing Data (Never seen before)
print('Accuracy: ', testing_score)
print(recall_score(y_test, predictions)) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
print(precision_score(y_test, predictions)) 



#%%
import pickle

model.save('Pass_vs_Left_Right_with_DL_all.h5')

#MinMaxScaler
with open('Pass_vs_Left_Right_with_DL_Scaler_all.pkl','wb') as f:
    pickle.dump(scaler,f)  









