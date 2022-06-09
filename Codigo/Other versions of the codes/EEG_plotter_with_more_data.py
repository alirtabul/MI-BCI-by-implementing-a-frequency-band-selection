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


# %matplotlib inline # displaying in the same code
# %matplotlib qt # displaying in a separate windows
#%%
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
    [mark, pos, time] = type_pos(markers)

    annotations = mne.Annotations(time, 1.0, mark) # this is the annotation of every class
    raw.set_annotations(annotations)


    # ICA for blink removal 
    
    ica = mne.preprocessing.ICA(n_components = 10, random_state = 0,  method='fastica')
    
    filt_raw = raw.copy()
    filt_raw.filter(l_freq=1., h_freq=None) # it's recommended to HP filter at 1 Hz 
    ica.fit(filt_raw)
    
    ica.exclude = [] # We must wisely choose the bad components based on the ICA sources plot
    
    raw_ica = ica.apply(raw.copy(), exclude = ica.exclude)

    
    # Now let's grasp the events from the annotations, so we can work with them. 
    
    events = mne.events_from_annotations(raw)
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')  
    
    # with the events we acquire the epochs  
    epochs = mne.Epochs(raw_ica, events[0], event_id = events[1], preload = True, tmin=tmin, tmax=tmax, baseline=None, picks = picks)
    

    labels = epochs.events[:, -1] # In the documentation they put a -2 after the []
    
    return epochs.get_data(),labels

[epoch1_C, label1_C] = epoch_event(path1_C, tmin= 2.1, tmax=3, LP=15,HP=26)   # LP=15,HP=21
[epoch2_C, label2_C] = epoch_event(path2_C, tmin= 2.1, tmax=3, LP=15,HP=26)  
[epoch3_C, label3_C] = epoch_event(path3_C, tmin= 2.1, tmax=3, LP=15,HP=26) 

[epoch1_B, label1_B] = epoch_event(path1_B, tmin= 2.1, tmax=3, LP=15,HP=26)   
[epoch2_B, label2_B] = epoch_event(path2_B, tmin= 2, tmax=2.9, LP=15,HP=26)  
[epoch3_B, label3_B] = epoch_event(path3_B, tmin= 2.3, tmax=3.2, LP=15,HP=26)
[epoch1_E, label1_E] = epoch_event(path1_E, tmin= 2.2, tmax=3.1, LP=15,HP=26)   
[epoch2_E, label2_E] = epoch_event(path2_E, tmin= 2, tmax=2.9, LP=15,HP=26)  
[epoch3_E, label3_E] = epoch_event(path3_E, tmin= 2.2, tmax=3.1, LP=15,HP=26)
[epoch1_F, label1_F] = epoch_event(path1_F, tmin= 1, tmax=1.9, LP=15,HP=26)   
[epoch2_F, label2_F] = epoch_event(path2_F, tmin= 1.3, tmax=2.2, LP=15,HP=26)  
[epoch3_F, label3_F] = epoch_event(path3_F, tmin= 1.4, tmax=2.3, LP=15,HP=26)

epochs = np.concatenate((epoch1_C, epoch2_C, epoch1_E,epoch2_E, epoch1_F,epoch2_F))
labels = np.concatenate((label1_C, label2_C, label1_E, label2_E, label1_F,label2_F))
epoch3 = np.concatenate((epoch3_C, epoch3_E, epoch3_F))
label3 = np.concatenate((label3_C, label3_E, label3_F))

# epoch3 = epoch3_F
# label3 = label3_F
#%% Apply the CSP

from mne.decoding import CSP
#train
epochs_train = epochs
epochs_data = epochs
epochs_data_train = epochs_train
#train with psd (power spectral density)
epochs_data_train_fft = np.fft.rfft(epochs_data_train)
epochs_data_train_fft_abs = np.abs(epochs_data_train_fft)
epochs_data_train_psd = np.square(epochs_data_train_fft_abs)


#test
epochs_test = epoch3
epochs_data_test = epoch3
#test with psd
epochs_data_test_fft = np.fft.rfft(epochs_data_test)
epochs_data_test_fft_abs = np.abs(epochs_data_test_fft)
epochs_data_test_psd = np.square(epochs_data_test_fft_abs)


csp = CSP(n_components=10, reg='oas', rank = 'info') # Very important to set rank to info, otherwise a rank problem may occur  
#csp.fit_transform(epochs_data_train[:], labels)
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
epochs_data = epochs
epochs_data_train = epochs_train
cv = ShuffleSplit(n_splits=10, test_size=0.2)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier

lda = LinearDiscriminantAnalysis()
svm = SVC(kernel ='rbf', gamma = 0.00001, C = 10000)
knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
NB  = GaussianNB()
RF  = RandomForestClassifier(n_estimators = 100, 
                             min_samples_split = 16, 
                             min_samples_leaf = 4, 
                             max_features = 'sqrt', 
                             max_depth = 8, 
                             bootstrap = True)
qda = QuadraticDiscriminantAnalysis()

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('RF', RF)])
scores = cross_val_score(clf, epochs_data_train[:], labels[:], cv=cv, n_jobs=1) # Let's train the first 500 epochs
(learning_curve(clf, epochs_data_train[:], labels[:], cv=cv, scoring='accuracy'))
clf.fit(epochs_data_train[:], labels[:])  # We fit the model for making it usable 


testing_score = clf.score(epoch3, label3) #Testing Data (Never seen before)
print('Accuracy: ', testing_score)
print('Precision: ', precision_score(label3[:], clf.predict(epoch3[:],))) # It gives more importance that the pass class is detected correctly. TruePositive/(TruePositives+FalsePositives)
print('Recall: ', recall_score(label3[:], clf.predict(epoch3[:],))) 


asa = clf.predict(epochs_data_train[[0]]) # This is an example of a prediction. Notice the doubl [[]]

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f +- %f / Chance level: %f" % (np.mean(scores),np.std(scores),
                                                          class_balance))


plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rcParams.update({'font.size': 16})
plt.rc('axes', titlesize=30, labelsize=25)

plot_confusion_matrix(clf, epoch3, label3)
plt.show()

#%% Lets find the best parameters for the Random Forest with GridSearch
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 100, num = 5)]
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
random_grid = [{},{'RF__n_estimators': n_estimators,
               'RF__max_features': max_features,
               'RF__max_depth': max_depth,
               'RF__min_samples_split': min_samples_split,
               'RF__min_samples_leaf': min_samples_leaf,
               'RF__bootstrap': bootstrap}]
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
RF_opt = Pipeline([('CSP', csp), ('RF', RandomForestClassifier())])
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = RF_opt, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(epochs_data_train[:], labels[:])
print(rf_random.best_params_)
print(rf_random.best_score_)


#%% Lets find the best parameters for the SVM
random_grid = {'SVM__C':[1,10,100,1000,5000, 10000,],'SVM__gamma':[0.001,0.0001,0.0001], 'SVM__kernel':['linear','rbf']}
print(random_grid) # Use the random grid to search for best hyperparameters

# First create the base model to tune
SVM_opt = Pipeline([('CSP', csp), ('SVM', SVC())])
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
SVM_random = RandomizedSearchCV(estimator = SVM_opt, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
SVM_random.fit(epochs_data_train[:], labels[:])
print(SVM_random.best_params_)
print(SVM_random.best_score_)
#%% Let's rty to train a hybrid Deep learning model with CNN and LSTM
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Activation, LSTM, concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Embedding, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


model = Sequential()

model.add(Conv1D(filters=240, kernel_size=2, padding= 'same', activation='relu',input_shape=(181,21))) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))




# model.add(Conv1D(filters=120, kernel_size=2, padding= 'same', activation='relu'))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Dropout(0.3))

model.add(LSTM(40,return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))

model.add(Flatten())


# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(12,activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))


learning_rate = 0.001
decay_rate = learning_rate/300
momentum = 0.8
opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#opt = RMSprop(learning_rate=0.0001, )
model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy']) 

model.summary()

#Normalize the data
X_train = epochs_data_train
y_train = labels-1
X_test = epoch3
y_test = label3-1


# It's very important to normalize the data and also reshape it in order to correctely fit the model
# To reshape it we must put (NUMBER OP EPOCHS, NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS)
X_train_re = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
scaler = MinMaxScaler(feature_range=(0,1))
X_train_norm = scaler.fit_transform(X_train_re)
X_train_norm = X_train_norm.reshape(X_train.shape[0], X_train.shape[2],X_train.shape[1])

X_test_re = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
X_test_norm = scaler.transform(X_test_re)
X_test_norm = X_test_norm.reshape(X_test.shape[0], X_test.shape[2],X_test.shape[1])


history = model.fit(X_train_norm, y_train, batch_size=40, epochs=500, validation_data=(X_test_norm, y_test))




history_df = pd.DataFrame(model.history.history).rename(columns={"loss":"train_loss", "accuracy":"train_accuracy"})
history_df.plot(figsize=(8,8))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


#%% Let's rty to train a hybrid Deep learning model with CNN and LSTM. (This time the inputs are the PSD not a time series.)

model_psd = Sequential()

model_psd.add(Conv1D(filters=440, kernel_size=2, padding= 'same', activation='relu',input_shape=(int(np.ceil(181/2)),21))) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
model_psd.add(MaxPooling1D(pool_size=2))
model_psd.add(Dropout(0.2))




model_psd.add(Conv1D(filters=220, kernel_size=2, padding= 'same', activation='relu'))
model_psd.add(MaxPooling1D(pool_size=1))
model_psd.add(Dropout(0.3))

model_psd.add(LSTM(20,return_sequences=True, activation='tanh'))
model_psd.add(Dropout(0.2))

model_psd.add(Flatten())


model_psd.add(Dense(128,activation='relu'))
model_psd.add(Dropout(0.3))

model_psd.add(Dense(64,activation='relu'))
model_psd.add(Dropout(0.1))

model_psd.add(Dense(24,activation='relu'))
model_psd.add(Dropout(0.3))

model_psd.add(Dense(1,activation='sigmoid'))


learning_rate = 0.001
decay_rate = learning_rate/300
momentum = 0.8
opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#opt = RMSprop(learning_rate=0.0001, )
model_psd.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy']) 

model_psd.summary()

#Normalize the data
X_train = epochs_data_train_psd
y_train = labels-1
X_test = epochs_data_test_psd
y_test = label3-1


# It's very important to normalize the data and also reshape it in order to correctely fit the model
# To reshape it we must put (NUMBER OF EPOCHS, NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS)
X_train_re = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
scaler_psd = MinMaxScaler(feature_range=(0,1))
X_train_norm = scaler_psd.fit_transform(X_train_re)
X_train_norm = X_train_norm.reshape(X_train.shape[0], X_train.shape[2],X_train.shape[1])

X_test_re = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
X_test_norm = scaler_psd.transform(X_test_re)
X_test_norm = X_test_norm.reshape(X_test.shape[0], X_test.shape[2],X_test.shape[1])


history = model_psd.fit(X_train_norm, y_train, batch_size=40, epochs=500, validation_data=(X_test_norm, y_test))
# 



history_df = pd.DataFrame(model_psd.history.history).rename(columns={"loss":"train_loss", "accuracy":"train_accuracy"})
history_df.plot(figsize=(8,8))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#%% Lets optimize the model:

from keras_tuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from keras_tuner.engine.hyperparameters import HyperParameter

def build_model(hp):
    model = Sequential()

    model.add(Conv1D(hp.Int("input_units", min_value=32,max_value =256,step=16), kernel_size=hp.Int("n_ker_1",2,4,1), padding= 'same', activation='relu',input_shape=(181,21))) #INPUTS_SIZE =(NUMBER OF SAMPLES IN EACH EPOCH, NUMBER OF CHANNELS
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(hp.Choice("Dropout_1", values=[0.1,0.2,0.3])))
    
    
    # for i in range(hp.Int('Conv_layers',1,1)):
    #     model.add(Conv1D(filters=hp.Int("input_units_2_layer_"+str(i), min_value=32,max_value =256,step=16), kernel_size=hp.Int("n_ker_2",2,4,1), padding= 'same', activation='relu'))
    #     model.add(MaxPooling1D(pool_size=3))
    #     model.add(Dropout(hp.Choice("Dropout_2_"+str(i), values=[0.1,0.2,0.3])))
    
        
    # for i in range(hp.Int('LSTM_layers',1,1)):
    #     model.add(LSTM(hp.Int("LSTM_input_units", min_value=10,max_value =100,step=10),return_sequences=True, activation='tanh'))
    #     model.add(Dropout(hp.Choice("Dropout_3_"+str(i), values=[0.1,0.2,0.3])))
    #     # model.add(LSTM(64,return_sequences=True, activation='tanh'))
    #     # model.add(Dropout(0.25))
    model.add(LSTM(hp.Int("LSTM_input_units", min_value=10,max_value =50,step=10),return_sequences=True, activation='tanh'))
    model.add(Dropout(hp.Choice("Dropout_3", values=[0.1,0.2,0.3])))
        
    model.add(Flatten())
    
    
    # model.add(Dense(64,activation='relu'))
    # model.add(Dropout(0.3))
    # for i in range(hp.Int('Dense_layers',1,1)):
    #     model.add(Dense(hp.Int("Dense"+str(i), min_value=32,max_value =256,step=16), activation='relu'))
    #     model.add(Dropout(hp.Choice("Dropout_4_"+str(i), values=[0.1,0.2,0.3])))
        
    
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(1,activation=hp.Choice("Final Activation", ['sigmoid','relu'])))
    
    
    learning_rate = hp.Choice("learning_rate", values=[0.1,0.01,0.001,0.0001])
    decay_rate = learning_rate/500
    momentum = 0.8
    opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    # opt2 = Adam(lr=learning_rate,decay=decay_rate)
    #opt = RMSprop(learning_rate=0.0001, )
    model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy']) 
    
    return model
    

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 

tuner = RandomSearch(build_model,
                     objective = 'val_accuracy',
                     max_trials = 200,
                     executions_per_trial=1,
                     directory="C:/Users/hussa/OneDrive/Escritorio",
                     project_name=('RandomSearchWiner12'))
tuner.search_space_summary()

tuner.search(x=X_train_norm,
             y=y_train,
             epochs=160,
             batch_size=50,
             validation_data=(X_test_norm, y_test))
             # callbacks=[stop_early])

tuner.results_summary()
#%% Save the model.

import pickle
#ML model
with open('LeftRight_Classification_More_Data_all.pkl','wb') as f:
    pickle.dump(clf,f)

#MinMaxScaler
with open('Scaler.pkl','wb') as f:
    pickle.dump(scaler,f)    
#DL model    
model.save('RightLeft_vs_Pass_Classification_More_Data_DL.h5')
    
# # load
# with open('model.pkl', 'rb') as f:
#     clf2 = pickle.load(f)

# clf2.predict(X[0:1])