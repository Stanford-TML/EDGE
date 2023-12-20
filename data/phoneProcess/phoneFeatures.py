import pandas as pd
import numpy as np
import pickle as pkl
import os
import librosa
from customFeatureExtract import *

from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from scipy.signal import find_peaks
import pywt  # PyWavelets library for wavelet transform

from sklearn.preprocessing import MinMaxScaler

#Getting scaler values to rescale custom input
aist = "/Users/pdealcan/Documents/github/EDGEk/data/test/positionsPhone/"
aistFiles = os.listdir(aist)

#Scaler for accelerometer
aistC = [pd.read_csv(f"{aist}{k}").iloc[:,0:3] for k in aistFiles]
maxes = [np.max(np.max(k)) for k in aistC]
mins = [np.min(np.min(k)) for k in aistC]

minAccel = np.median(mins)
maxesAccel = np.median(maxes)

#Scaler for gyroscope
aistC = [pd.read_csv(f"{aist}{k}").iloc[:,3:6] for k in aistFiles]
maxes = [np.max(np.max(k)) for k in aistC]
mins = [np.min(np.min(k)) for k in aistC]

minGyro = np.median(mins)
maxesGyro = np.median(maxes)

#Receives IMU data (accelerometer and gyro)
def extractPhoneFeatures(directoryIn, directoryOut, brigitta = False):
    #Read phone and ectract features
    os.makedirs(directoryOut, exist_ok=True)
    files = os.listdir(directoryIn)
    for k in files:
        phone = pd.read_csv(f"{directoryIn}{k}")
        assert phone.shape[1] == 6 #3 dimensions for accelerometer, 3 for gyroscope
    
        #Downsampling to half
        phone = phone.to_numpy()

        if brigitta:
            print("Brigitta's dataset. Scaling data.")
            scaler = MinMaxScaler(feature_range=(minAccel, maxesAccel))
            scaled_accel = scaler.fit_transform(phone[:, 0:3])
            phone[:, 0:3] = scaled_accel

#            scaler = MinMaxScaler(feature_range=(minGyro, maxesGyro))
#            scaled_gyro = scaler.fit_transform(phone[:, 3:6])
#            phone[:, 3:6] = scaled_gyro

        phone = phone[:: 2, :]
        df = extractFeats(phone, phone.shape[0])
        df = np.float32(df)
        fName = f"{directoryOut}{k}"
        fName = fName.replace(".csv", ".npy")
        np.save(fName, df)
        print(f"Wrote file: {fName}")

#Processing AIST++ dataset
#for k in ["train", "test"]:
#    directoryIn = f"../../data/{k}/positionsPhone/"
#    directoryOut = f"../../data/{k}/baseline_feats/"
#    extractPhoneFeatures(directoryIn, directoryOut, False)

#Processing brigitta's dataset
dirIn = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/phoneIMU/"
dirOut = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/phoneFeatures/"
extractPhoneFeatures(dirIn, dirOut, True)
