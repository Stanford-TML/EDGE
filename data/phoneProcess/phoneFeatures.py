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

def createPhone(df, markerIndex, markerIndex2):
    phone = (df[:,markerIndex:markerIndex+3] + df[:,markerIndex2:markerIndex2+3])/2 #Phone root calculated between right hip and knee markers
    return phone

def extractPhoneFeatures(directoryIn, directoryOut):
    #Read phone and ectract features
    files = os.listdir(directoryIn)
    for k in files:
        fName = f"{directoryOut}{k}"
        df = pd.read_csv(f"{directoryIn}{k}")
        fName = f"{directoryOut}{k}"
        fName = fName.replace("csv", "npy")
        df = df.to_numpy()
        df = df[:: 2, :]
        df = get_second_derivative(df)
        df = extractFeats(df, df.shape[0])
        df = np.float32(df)
        np.save(fName, df)
        print(f"Wrote file: {k}")

##Getting Watch+Phone features from AIST++
train = ["test", "test"]
fType = ["accel"]
for k in train:
    for l in fType:
        directoryIn = f"../{l}/{k}/positionsWatch/"
        directoryOut = f"../{l}/{k}/baseline_feats_watch/"
        extractPhoneFeatures(directoryIn, directoryOut)

if False:
    ##Getting features from AMASS (Accel only)
    dirIn = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/accelPositions/"
    dirOut = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/phoneFeatures/"
    extractPhoneFeatures(dirIn, dirOut)

    ##Getting features from AMASS (Gyro)
    dirIn = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/gyroPositions/"
    dirOut = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/phoneFeaturesGyro/"
    extractPhoneFeatures(dirIn, dirOut)

    ##Getting features from AIST++
    train = ["test", "test"]
    fType = ["accel", "gyro"]
    for k in train:
        for l in fType:
            directoryIn = f"../{l}/{k}/positionsPhone/"
            directoryOut = f"../{l}/{k}/baseline_feats/"
            extractPhoneFeatures(directoryIn, directoryOut)

    ##Extracting features from Brigitta
    fType = ["accel", "gyro"]
    for k in fType:
        directoryIn = f"/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/{k}/phoneIMU/"
        directoryOut = f"/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/{k}/phoneFeatures/"
        extractPhoneFeatures(directoryIn, directoryOut)
