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

def createPhone(df):
    phone = (df[:,6:9] + df[:,15:18])/2 #Phone root calculated between right hip and knee markers
    return phone

def writePhonePositions(train):
    directoryIn = f"../../data/{train}/motions_sliced/"
    directoryOut = f"../../data/{train}/positionsPhone/"
    os.makedirs(directoryOut, exist_ok=True)
    fNames = os.listdir(directoryIn)
    for x in fNames:
        nIn = f"{directoryIn}{x}"
        nOut = f"{directoryOut}{x}"
        df = pd.read_pickle(nIn)
        ph = createPhone(df)
        fileObject = open(nOut, 'wb')
        pkl.dump(ph, fileObject)
        print(f"Wrote: {x}")

def extractPhoneFeatures(train):
    #Read phone and ectract features
    directoryIn = f"../../data/{train}/positionsPhone/"
    directoryOut = f"../../data/{train}/baseline_feats/"
    os.makedirs(directoryOut, exist_ok=True)
    files = os.listdir(directoryIn)
    for k in files:
        df = pd.read_pickle(f"{directoryIn}{k}")
        df = df[:: 2, :] #Downsampling to half
        df = get_second_derivative(df)
        df = extractFeats(df, df.shape[0])
        df = np.float32(df)
        fName = f"{directoryOut}{k}"
        fName = fName.replace("pkl", "npy")
        np.save(fName, df)
        print(f"Wrote file: {k}")

train = "test"
writePhonePositions(train)
extractPhoneFeatures(train)

train = "train"
writePhonePositions(train)
extractPhoneFeatures(train)
