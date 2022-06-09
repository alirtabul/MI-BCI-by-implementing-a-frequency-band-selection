# -*- coding: utf-8 -*-
"""
In this script, the trained AI models will be tested on one entire session of Subject C, by sliding 
the session inside a Window. Both AI models will work cooperatively to classify each signal the same
way as it would do on a real-time implementation.
 

@author: Ali Abdul Ameer Abbas
"""

# Import important libraries.
import pickle
import pywt
import numpy as np
import matplotlib.pyplot as plt
import mne
import pymatreader
import scipy.stats
import tensorflow as tf
from scipy.signal import firwin, lfilter
from scipy.signal import stft
import serial
import time 

# Import the modules that were created in Left_vs_Right.py and Imagery_vs_Resting.py
from Left_vs_Right import PreprocessDataset, ProcessDataset
from Imagery_vs_Resting import PreprocessDatasetPass, ProcessDatasetPass

#%% Plot configuration.
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rc('axes', titlesize=30, labelsize=25)

#%% load and preprocess data

path = '../Data/CLA/CLASubjectC1511263StLRHand.mat' # Only an unseen session of Subject C will be used.

#Initialize the class

Subject_C = PreprocessDataset(path)

# Set the reference.
Subject_C.channel_reference() 

#%% Proceed on loading the models created in the other scripts:
    
# Load the Left vs Right imagery chosen Model.

with open('LeftRight_Classification_More_Data_all.pkl', 'rb') as f:
    clf4 = pickle.load(f)
    
# Load the Imagery vs Resting state chosen Model.
DL_model_Pass_detection = tf.keras.models.load_model('Pass_vs_Left_Right_with_DL_all.h5')
# Load CNN's Scaler.
with open('Pass_vs_Left_Right_with_DL_Scaler_all.pkl', 'rb') as f:
    Pass_scaler = pickle.load(f)


#%% Get the data from the signals.
# Get the data for all channels.
AciscoAll = Subject_C.raw.get_data()

# Get the data for the C3, Cz, and C4 channels. Useful for the statistical features and STFT analysis.
picks = mne.pick_channels(Subject_C.raw.info["ch_names"], ["C3", "Cz", "C4"])
Acisco = Subject_C.raw.get_data(picks=picks)


#%% Define and design the FIR filter used in the real-time simulation.
#Article in how to design a FIR filter: 
#https://www.allaboutcircuits.com/technical-articles/design-examples-of-fir-filters-using-window-method/

def bandpass_firwin(ntaps, lowcut, highcut, fs = 200, window = 'hann'):
    '''
    This function enables the user to define the parameters for BP FIR filter.
    
    INPUTS:
        ntaps --> Lenght of the filter.
        lowcut --> Low frequency.
        highcut --> High frequency.
        fs --> Sampling frequency,
        window --> Window type. (Hanning, Hamming, etc.)
    OUTPUTS:
        taps --> Filter coeficients.
    '''
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

# Design the filter.
filt_length = 150 # It says the order of the filter. 
lowcut = 15       # Low frequency.
highcut = 26      # High frequency.

delay = np.ceil((filt_length-1)/2) # the delay of the filter (in samples) is calculated as following.

FIR_Coeficients = bandpass_firwin(filt_length,lowcut,highcut, fs = 200, window='hamming') 

# Compute the frequency response of the digital filter.
w, mag = scipy.signal.freqz(FIR_Coeficients, 1, fs=200) # The numerator are the coeficients and the denominator is 1 (no poles). 

# Plot the filter's impulse response.

plt.figure()
plt.title('Impulse Response of the FIR filter', fontsize="30")
plt.xlabel('Samples', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.plot(FIR_Coeficients)
plt.show()

# Plot the Bode.

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

# Plot an example of the filter's effect.


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

#%% Create a sliding window for inserting the datapoints and classifying each window
# by using the saved RF model with CSP (Left vs Right --> clf4) 
# and the CNN with STFT (Imagery vs Resting --> DL_model_Pass_detection, Pass_scaler).

# Define the serial communication with Arduino.

# Ensure that the Arduino port is initially closed. Otherwise, errors may occur.
try:
    arduino.close()
except:
    pass

# Open the Serial Communication with Arduino.
try:
    arduino = serial.Serial('COM3', timeout=None, baudrate=9600)
    time.sleep(2)
except:
    print('Arduino not connected')

# Initialize parameters.

times = [0]
prediction_markers = [] # Save the predictions.
prediction_times = []   # Save the times of the prediction,

# Define the times of the session to analyze.
start_time = 170 # Seconds.
stop_time = 500  # Seconds.
step = 0.1       # Seconds. Step of the window at each iteration (Always 0.1). 
  
# Start the analysis and save the data in the vectors.

for i in np.arange(start_time, stop_time, step):
    vala = i
    # Encapsulate the data in the time window.
    # Remember that the sampling frequency is 200. Since the window has size 0.9, there are 180 samples (0.9*200).
    ACA_All = [AciscoAll[:, int(vala*200):int(vala*200+181+delay)]]   # Keep the data that is insede the time window.  
    ACA_All = lfilter(FIR_Coeficients, 1.0, ACA_All)                  # Filter the data 
    ACA_All = np.array(ACA_All)[:,:,int(delay):]                      # Remove the delay for inserting the data to the models.
    # Do the same but just for channels C3, Cz, and C4. 
    ACA = [Acisco[:, int(vala*200):int(vala*200+181+delay)]]
    ACA = lfilter(FIR_Coeficients, 1.0, ACA)
    ACA = np.array(ACA)[:,:,int(delay):] 
    
    # Extract the STFT features.
    frec, tim, Zx =stft(ACA, fs=200.0, window='hann',nperseg=181, noverlap=180)
    FEAT = np.abs(Zx)
    
    # Predict if the window is in an Imagery or Resting state.
    pred = [int((DL_model_Pass_detection.predict(Pass_scaler.transform(np.array(FEAT).reshape(1,3*91*181)).reshape(1,3,91,181))> 0.2).astype("int32"))+1]
    
    # If the prediction is an imagery, apply the Left vs Right prediction (pred2)
    if pred == [1]:
        if i > (times[0]):
            # Print the prediction and the probability of the prediction
            pred2 = list(clf4.predict(np.array(ACA_All)))
            pred2_prob = clf4.predict_proba(np.array(ACA_All))
            print(clf4.predict(np.array(ACA_All)), clf4.predict_proba(np.array(ACA_All)))
            
            # Send the pred2 variable to Arduino in order to move the Robotic Manipulator (Left or Right).
            try:
                arduino.write((str(pred2[0])+'\n').encode())
                #print('Data sent.')
            except:
                pass

            
            # The following lines are used for later plotting the results in a graph.
            if times[0]+0.5 < i:   #If there is a difference of 0.5 sec between predictions, it is considered another imagery. 
                print('-----------------')
                # Send a reset indication to Arduino in order to reset the Left and Right Counters (see arduino code).
                try:
                    arduino.write(('Reset'+'\n').encode())
    
                except:
                    pass
           
            if pred2 == [1]: # If Left is detected.
                print(round(i, 1), 'Left prediction')
                prediction_markers.append('Left prediction') # Add the prediction to the list.
                prediction_times.append(i)                   # Add the prediction time to the list.
                
            if pred2 == [2]: # If Right is detected.
                print(round(i, 1), 'Right prediction')
                prediction_markers.append('Right prediction') # Add the prediction to the list.
                prediction_times.append(i)                    # Add the prediction time to the list.
                
            times[0] = i # Udapte the time. 

# Close the Serial communication.
try:
    arduino.close()
    print('Data sent.')
except:
    pass

# Convert the lists to numpy arrays.
prediction_markers = np.array(prediction_markers)
prediction_times = np.array(prediction_times)

# Now, create a loop in which each imagery be separated their time and length.
# Initialize the needed variables.
new_prediction_markers = []
new_prediction_times = []
segment = []                # It will separate different imageries events.
times = prediction_times[0]

# Create a loop for organizing the lists.
for i, timer in enumerate(prediction_times):
    # If there is a difference of 0.5 sec between predictions, it is considered another imagery. 
    if timer > times + 0.5:
        # To be more robust, consider a correct imagery when the model has predicted a class 4 times consecutively.
        if len(segment) >= 4:  
            segment_label = prediction_markers[segment]
            segment_time = prediction_times[segment]

            left_count = np.char.count(segment_label, 'Left prediction').sum()
            right_count = np.char.count(
                segment_label, 'Right prediction').sum()
            # If Left prediction are more abundant thanright ones, consider Left.
            if left_count > right_count and right_count <= 4:
                
                new_prediction_markers.append('Left prediction')
                new_prediction_times.append(list(segment_time)[0])
                
            # If Right prediction are more abundant than left ones, consider Right.
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
    times = timer

# Now plot the results.

# Put the ground truth markers as well as the neewly generated prediction markers.

[mark, pos, timer] = Subject_C.type_pos() # Obtain the ground truth markers.

mark.extend(new_prediction_markers) # Add the new markers.
timer.extend(new_prediction_times)   # Add the new markers' times.

# Set annotations.

annotations = mne.Annotations(timer, 1.0, mark)
Subject_C.raw.set_annotations(annotations)
Subject_C.raw.plot(duration=20.0,start=500.0)













