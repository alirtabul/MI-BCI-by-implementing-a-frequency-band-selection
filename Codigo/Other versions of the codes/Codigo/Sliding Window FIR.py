# -*- coding: utf-8 -*-
"""
In this script we will do a first approach for detecting the LEFT, RIGHT and PASS
classes while inserting a segment of the EEG recording. 
"""
# %% Load the EEG recording

import pickle
import pywt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import pymatreader
import scipy.stats
import tensorflow as tf
from scipy.signal import firwin, lfilter
# load and preprocess data ####################################################
# best time 0.6  1.4, BW = 22-26
path = '../Data/CLA/CLASubjectA1601083StLRHand.mat'
# best time 2.1  2.9 # AWSOME DATA
path = '../Data/CLA/CLASubjectC1512233StLRHand.mat'
path = '../Data/CLA/CLASubjectC1512163StLRHand.mat'  # best time 2.1  2.9
path = '../Data/CLA/CLASubjectC1511263StLRHand.mat'
struct = pymatreader.read_mat(
    path, ignore_fields=['previous'], variable_names='o')
struct = struct['o']


chann = struct['chnames']
# pop th X3 channel, which is only used for the data adquisition.
chann = chann[:-1]
info = mne.create_info(ch_names=chann,
                       sfreq=float(struct['sampFreq']), ch_types='eeg', montage='standard_1020', verbose=None)

data = np.array(struct['data'])
data = data[:, :-1]
data_V = data*1e-6  # mne reads units are in Volts

raw = mne.io.RawArray(data_V.transpose(), info, copy="both")

raw = raw.set_eeg_reference('average', projection=False)
#raw.filter(15., 21., fir_design='firwin', skip_by_annotation='edge')

# %% next step is to add the feature extracter funtion needed for the model


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
            dist_space = np.linspace(min(data[i]), max(data[i]), 100)

            # Normalize, this is the integral underthe curve
            gau_total = np.sum(gau(dist_space))

            pdf = gau(dist_space)/gau_total

            pdf_all.append(pdf)

        skw_epoch = []
        for i in range((data.shape[0])):
            skw_epoch.append(scipy.stats.skew(data[i], axis=0, bias=True))
        krt_epoch = []
        for i in range((data.shape[0])):
            krt_epoch.append(scipy.stats.kurtosis(
                data[i], axis=0, fisher=True, bias=True, nan_policy='propagate'))

        def energy(signal):
            energy = 0.0

            for x in signal:
                energy += (x)**2
            return energy

        energy_epoch = []
        for i in range((data.shape[0])):
            energy_epoch.append(energy(data[i]))

        Pi_all = []  # here we save the values for each channel
        # We need Wavelets in order to perform the Shannon Energy, for instance.
        for i in range((data.shape[0])):

            # Apply the Wavelet transform
            coef, _ = pywt.cwt(data[i],  np.arange(7, 20, 0.5), 'morl')

            # now we use the following formulas: 'https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy'
            [M, N] = coef.shape  # M --> scale number, N --> time segments
            Ej = []
            for j in range(M):
                Ej.append(sum(abs(coef[j, :])))

            Etot = sum(Ej)

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

            return entropy, shannon_entropy, log_energy_entropy
        entropy_epoch = []
        shannon_entropy_epoch = []
        log_energy_entropy_epoch = []
        for i in range((data.shape[0])):
            # S = -sum(pk * log(pk)) --> pk is the Probability Density Function
            entr, shan, log_en = all_entropy(pdf_all[i])

            entropy_epoch.append(entr)
            shannon_entropy_epoch.append(shan)
            log_energy_entropy_epoch.append(log_en)

        skw_epoch_dic[ch[0]].append(skw_epoch[0]), skw_epoch_dic[ch[1]].append(
            skw_epoch[1]), skw_epoch_dic[ch[2]].append(skw_epoch[2])
        krt_epoch_dic[ch[0]].append(krt_epoch[0]), krt_epoch_dic[ch[1]].append(
            krt_epoch[1]), krt_epoch_dic[ch[2]].append(krt_epoch[2])
        energy_epoch_dic[ch[0]].append(energy_epoch[0]), energy_epoch_dic[ch[1]].append(
            energy_epoch[1]), energy_epoch_dic[ch[2]].append(energy_epoch[2])
        shannon_entropy_epoch_dic[ch[0]].append(shannon_entropy_epoch[0]), shannon_entropy_epoch_dic[ch[1]].append(
            shannon_entropy_epoch[1]), shannon_entropy_epoch_dic[ch[2]].append(shannon_entropy_epoch[2])
        log_energy_entropy_epoch_dic[ch[0]].append(log_energy_entropy_epoch[0]), log_energy_entropy_epoch_dic[ch[1]].append(
            log_energy_entropy_epoch[1]), log_energy_entropy_epoch_dic[ch[2]].append(log_energy_entropy_epoch[2])
        entropy_epoch_dic[ch[0]].append(entropy_epoch[0]), entropy_epoch_dic[ch[1]].append(
            entropy_epoch[1]), entropy_epoch_dic[ch[2]].append(entropy_epoch[2])

    return skw_epoch_dic, krt_epoch_dic, energy_epoch_dic, entropy_epoch_dic, shannon_entropy_epoch_dic, log_energy_entropy_epoch_dic


def feature_ex(skw, krt, energy, entropy, shannon, log):
    big_feat = []
    ch = ['C3', 'C4', 'Cz']
    for j in range(len(skw['C3'])):
        small_feat = []
        for i in range(len(ch)):

            new_small_feat = [skw[ch[i]][j], krt[ch[i]][j], energy[ch[i]]
                              [j], entropy[ch[i]][j], shannon[ch[i]][j], log[ch[i]][j]]
            small_feat.extend(new_small_feat)
        big_feat.append(small_feat)
    return big_feat


# %% We proceed on loading the models created in the other scripts:
# load
with open('RightLeft_vs_Pass_Classification_More_Data_2.pkl', 'rb') as f:
    clf1 = pickle.load(f)
with open('LeftPass_Classification.pkl', 'rb') as f:
    clf2 = pickle.load(f)
with open('RightPass_Classification.pkl', 'rb') as f:
    clf3 = pickle.load(f)
with open('LeftRight_Classification_More_Data_all.pkl', 'rb') as f:
    clf4 = pickle.load(f)
#load scaler
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

DL_model = tf.keras.models.load_model('RightLeft_vs_Pass_Classification_More_Data_DL.h5')
DL_model.summary()


DL_model_Pass_detection = tf.keras.models.load_model('Pass_vs_Left_Right_with_DL_all.h5')

with open('Pass_vs_Left_Right_with_DL_Scaler_all.pkl', 'rb') as f:
    Pass_scaler = pickle.load(f)

# %% Now we create a sliding window for inserting the datapoints and classifying each window.
import cmath
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
Acisco = raw.get_data(picks=picks)
AciscoAll = raw.get_data()
times = [0]

prediction_markers = []
prediction_times = []

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hann'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps
#Article in how to design a FIR filter: 
#https://www.allaboutcircuits.com/technical-articles/design-examples-of-fir-filters-using-window-method/
filt_length = 150 # It says the order of the filter. 
delay = np.ceil((filt_length-1)/2) # the delay of the filter (in samples) is calculated as following.
FIR_Coeficients = bandpass_firwin(filt_length,15,26, 200, window='hamming')
plt.figure()
plt.title('Impulse Response of the FIR filter', fontsize="30")
plt.xlabel('Samples', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.plot(FIR_Coeficients)
plt.show()

w, mag = scipy.signal.freqz(FIR_Coeficients, 1, fs=200) # the numerator are the coeficients and de denominator is 1 (no poles). 


# plot the fase and the magnitud of the filter designed
fig, axs = plt.subplots(2)
fig.suptitle('FIR filter BODE', fontsize=30)
axs[0].plot(w, 20*np.log10(np.abs(mag)))   # Bode magnitude plot
axs[0].set_xlabel('Frequency (Hz)', fontsize=20)
axs[0].set_ylabel('Magnitude (dB)', fontsize=20)
#axs[0].grid(True)
axs[1].plot(w, np.unwrap(np.angle(mag,deg=False)))   # Bode phase plot
axs[1].set_xlabel('Frequency (Hz)', fontsize=20)
axs[1].set_ylabel('Phase (rad)', fontsize=20)
#axs[1].grid(True)
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('Before and after filtering', fontsize=30)

ACA = [Acisco[:, int(2500*200):int(2500*200+180+delay)]]
axs[0].plot(ACA[0][1])
axs[0].set_xlabel('Samples', fontsize=20)
axs[0].set_ylabel('Amplitude', fontsize=20)
axs[1].set_xlabel('Samples', fontsize=20)
axs[1].set_ylabel('Amplitude', fontsize=20)
ACA = lfilter(FIR_Coeficients, 1.0, ACA)
axs[1].plot(ACA[0][1])
#%% ML left_right / ML Pass_LeftRight

times = [0]

prediction_markers = []
prediction_times = []

# first value is the starting time, second value the ending time and th third one the time step
for i in np.arange(2500, 2730, 0.1):
    vala = i
    # Remember that the sampling frequency is 200. The 180 are the samples between 2.1 and 3 second (0.9*200)
    
    ACA = [Acisco[:, int(vala*200):int(vala*200+181+delay)]]
    ACA = lfilter(FIR_Coeficients, 1.0, ACA)
    ACA = np.array(ACA)[:,:,int(delay):] # Remove the delay for inserting the data to the models.
    ACA_All = [AciscoAll[:, int(vala*200):int(vala*200+181+delay)]]
    ACA_All = lfilter(FIR_Coeficients, 1.0, ACA_All)
    ACA_All = np.array(ACA_All)[:,:,int(delay):] # Remove the delay for inserting the data to the models.
    CA_skw, CA_krt, CA_energy, CA_entropy, CA_shannon, CA_log = feature_extracter(
        ACA)
    FEAT = feature_ex(CA_skw, CA_krt, CA_energy,
                      CA_entropy, CA_shannon, CA_log)
    FEAT = np.array(FEAT)
    pred = list(clf1.predict(FEAT))  
    
    if pred == [1] and clf1.predict_proba(FEAT)[0,0]>0.5: # we must be 50% sure that the prediction is okay.
        if i > (times[0]):

            if times[0]+0.5 < i:
                print('-----------------')

            pred2 = list(clf4.predict(np.array(ACA_All)))
            print(clf4.predict(np.array(ACA_All)), clf4.predict_proba(np.array(ACA_All)))
            if pred2 == [1] and clf4.predict_proba(np.array(ACA_All))[0,0]>0.5:
                print(round(i, 1), 'Left prediction')
                prediction_markers.append('Left prediction')
                prediction_times.append(i)
            if pred2 == [2] and clf4.predict_proba(np.array(ACA_All))[0,1]>0.5:
                print(round(i, 1), 'Right prediction')
                prediction_markers.append('Right prediction')
                prediction_times.append(i)
            times[0] = i

prediction_markers = np.array(prediction_markers)
prediction_times = np.array(prediction_times)

# Now let's  create a loop in which we can separate each imagery by their time and length.
new_prediction_markers = []
new_prediction_times = []
segment = []
times = prediction_times[0]
for i, time in enumerate(prediction_times):
    if time > times + 0.5:
        if len(segment) >= 2:
            segment_label = prediction_markers[segment]
            segment_time = prediction_times[segment]

            left_count = np.char.count(segment_label, 'Left prediction').sum()
            right_count = np.char.count(
                segment_label, 'Right prediction').sum()

            if left_count > right_count and right_count <= 4:
                new_prediction_markers.append('Left prediction')
                new_prediction_times.append(list(segment_time)[0])
            elif left_count < right_count and left_count <= 4:
                
                new_prediction_markers.append('Right prediction')
                new_prediction_times.append(list(segment_time)[0])
            else:
                pass

            segment = []
        else:
            segment = []
    else:
        segment.append(i)
    times = time


new_prediction_markers
new_prediction_times


# Now let's plot the predictions with the labels


def type_pos(markers, sfreq=200.0):
    """
    Just left and right positions, NO PASS marker
    """
    mark = []
    pos = []
    time = []
    # for assigning the movements--> left =1, right = 2, pass = 3
    desc = ['left', 'right']
    for i in range(len(markers)-1):
        if markers[i] == 0 and markers[i+1] != 0 and (markers[i+1] in [1, 2]):

            mark.append(desc[markers[i+1]-1])
            pos.append((i+2))
            time.append((i+2)/sfreq)
        else:
            continue
    return mark, pos, time


markers = np.array(struct['marker']).transpose()
[mark, pos, time] = type_pos(markers)

mark.extend(new_prediction_markers)
time.extend(new_prediction_times)

# Now, let's plot:

# this is the annotation of every class
annotations = mne.Annotations(time, 1.0, mark)
raw.set_annotations(annotations)
raw.plot(duration=20.0,start=200)


# %% (The same code of before but with Deep Learning) Now we create a sliding window for inserting the datapoints and classifying each window.

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
Acisco = raw.get_data(picks=picks)
AciscoAll = raw.get_data()
times = [0]

prediction_markers = []
prediction_times = []

# first value is the starting time, second value the ending time and th third one the time step
for i in np.arange(500, 1000, 0.1):
    vala = i
    # Remember that the sampling frequency is 200. The 180 are the samples between 2.1 and 3 second (0.9*200)
    ACA = [Acisco[:, int(vala*200):int(vala*200+181+delay)]]
    ACA = lfilter(FIR_Coeficients, 1.0, ACA)
    ACA = np.array(ACA)[:,:,int(delay):] # Remove the delay for inserting the data to the models.
    
    ACA_All = [AciscoAll[:, int(vala*200):int(vala*200+181+delay)]]
    ACA_All = lfilter(FIR_Coeficients, 1.0, ACA_All)
    ACA_All = np.array(ACA_All)[:,:,int(delay):] # Remove the delay for inserting the data to the models.
    
    CA_skw, CA_krt, CA_energy, CA_entropy, CA_shannon, CA_log = feature_extracter(
        ACA)
    FEAT = feature_ex(CA_skw, CA_krt, CA_energy,
                      CA_entropy, CA_shannon, CA_log)
    FEAT = np.array(FEAT)
    pred = list(clf1.predict(FEAT))

    if pred == [1]:
        if i > (times[0]):

            if times[0]+0.5 < i:
                print('-----------------')

            pred2 = [int((DL_model.predict(scaler.transform(np.array(ACA_All).reshape(1,181*21)).reshape(1,181,21))> 0.5).astype("int32"))+1] 
            if pred2 == [1]:
                print(round(i, 1), 'Left prediction')
                prediction_markers.append('Left prediction')
                prediction_times.append(i)
            if pred2 == [2]:
                print(round(i, 1), 'Right prediction')
                prediction_markers.append('Right prediction')
                prediction_times.append(i)
            times[0] = i

prediction_markers = np.array(prediction_markers)
prediction_times = np.array(prediction_times)

# Now let's  create a loop in which we can separate each imagery by their time and length.
new_prediction_markers = []
new_prediction_times = []
segment = []
times = prediction_times[0]
for i, time in enumerate(prediction_times):
    if time > times + 0.5:
        if len(segment) >= 4:
            segment_label = prediction_markers[segment]
            segment_time = prediction_times[segment]

            left_count = np.char.count(segment_label, 'Left prediction').sum()
            right_count = np.char.count(
                segment_label, 'Right prediction').sum()

            if left_count > right_count and right_count <= 4:
                new_prediction_markers.append('Left prediction')
                new_prediction_times.append(list(segment_time)[0])
            elif left_count < right_count and left_count <= 4:
                
                new_prediction_markers.append('Right prediction')
                new_prediction_times.append(list(segment_time)[0])
            else:
                pass

            segment = []
        else:
            segment = []
    else:
        segment.append(i)
    times = time


new_prediction_markers
new_prediction_times


# Now let's plot the predictions with the labels


def type_pos(markers, sfreq=200.0):
    """
    Just left and right positions, NO PASS marker
    """
    mark = []
    pos = []
    time = []
    # for assigning the movements--> left =1, right = 2, pass = 3
    desc = ['left', 'right']
    for i in range(len(markers)-1):
        if markers[i] == 0 and markers[i+1] != 0 and (markers[i+1] in [1, 2]):

            mark.append(desc[markers[i+1]-1])
            pos.append((i+2))
            time.append((i+2)/sfreq)
        else:
            continue
    return mark, pos, time


markers = np.array(struct['marker']).transpose()
[mark, pos, time] = type_pos(markers)

mark.extend(new_prediction_markers)
time.extend(new_prediction_times)

# Now, let's plot:

# this is the annotation of every class
annotations = mne.Annotations(time, 1.0, mark)
raw.set_annotations(annotations)
raw.plot(duration=20.0,start=200)

# %% Now we create a sliding window for inserting the datapoints and classifying each window.
# This time we do it with the DL Pass_vs_Left_Right_with_DL model
# Best case

from scipy.signal import stft

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
Acisco = raw.get_data(picks=picks)
AciscoAll = raw.get_data()
times = [0]

prediction_markers = []
prediction_times = []

# first value is the starting time, second value the ending time and th third one the time step
for i in np.arange(170, 3300, 0.1):
    vala = i
    # Remember that the sampling frequency is 200. The 180 are the samples between 2.1 and 3 second (0.9*200)
    ACA = [Acisco[:, int(vala*200):int(vala*200+181+delay)]]
    ACA = lfilter(FIR_Coeficients, 1.0, ACA)
    ACA = np.array(ACA)[:,:,int(delay):] # Remove the delay for inserting the data to the models.
    
    ACA_All = [AciscoAll[:, int(vala*200):int(vala*200+181+delay)]]
    ACA_All = lfilter(FIR_Coeficients, 1.0, ACA_All)
    ACA_All = np.array(ACA_All)[:,:,int(delay):] # Remove the delay for inserting the data to the models.
    
    frec, tim, Zx =stft(ACA, fs=200.0, window='hann',nperseg=181, noverlap=180)
    
    FEAT = np.abs(Zx)
    pred = [int((DL_model_Pass_detection.predict(Pass_scaler.transform(np.array(FEAT).reshape(1,3*91*181)).reshape(1,3,91,181))> 0.2).astype("int32"))+1]
    
    if pred == [1]:
        if i > (times[0]):

            if times[0]+0.5 < i:
                print('-----------------')

            pred2 = list(clf4.predict(np.array(ACA_All)))
            pred2_prob = clf4.predict_proba(np.array(ACA_All))
            print(clf4.predict(np.array(ACA_All)), clf4.predict_proba(np.array(ACA_All)))
            if pred2 == [1]: # and pred2_prob[0][0]>0.7:
                print(round(i, 1), 'Left prediction')
                prediction_markers.append('Left prediction')
                prediction_times.append(i)
            if pred2 == [2]: # and pred2_prob[0][1]>0.7:
                print(round(i, 1), 'Right prediction')
                prediction_markers.append('Right prediction')
                prediction_times.append(i)
            times[0] = i

prediction_markers = np.array(prediction_markers)
prediction_times = np.array(prediction_times)

# Now let's  create a loop in which we can separate each imagery by their time and length.
new_prediction_markers = []
new_prediction_times = []
segment = []
times = prediction_times[0]
for i, time in enumerate(prediction_times):
    if time > times + 0.5:
        if len(segment) >= 4:
            segment_label = prediction_markers[segment]
            segment_time = prediction_times[segment]

            left_count = np.char.count(segment_label, 'Left prediction').sum()
            right_count = np.char.count(
                segment_label, 'Right prediction').sum()

            if left_count > right_count and right_count <= 4:
                new_prediction_markers.append('Left prediction')
                new_prediction_times.append(list(segment_time)[0])
            elif left_count < right_count and left_count <= 4:
                
                new_prediction_markers.append('Right prediction')
                new_prediction_times.append(list(segment_time)[0])
            else:
                pass

            segment = []
        else:
            segment = []
    else:
        segment.append(i)
    times = time


new_prediction_markers
new_prediction_times


# Now let's plot the predictions with the labels


def type_pos(markers, sfreq=200.0):
    """
    Just left and right positions, NO PASS marker
    """
    mark = []
    pos = []
    time = []
    # for assigning the movements--> left =1, right = 2, pass = 3
    desc = ['left', 'right']
    for i in range(len(markers)-1):
        if markers[i] == 0 and markers[i+1] != 0 and (markers[i+1] in [1, 2]):

            mark.append(desc[markers[i+1]-1])
            pos.append((i+2))
            time.append((i+2)/sfreq)
        else:
            continue
    return mark, pos, time


markers = np.array(struct['marker']).transpose()
[mark, pos, time] = type_pos(markers)

mark.extend(new_prediction_markers)
time.extend(new_prediction_times)

# Now, let's plot:

# this is the annotation of every class
annotations = mne.Annotations(time, 1.0, mark)
raw.set_annotations(annotations)
raw.plot(duration=20.0,start=500.0)
