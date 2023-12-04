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

#df = pd.read_pickle("./data/dataset_backups/processed_train_data.pkl")
train = "test"
directoryIn = f"./data/{train}/phonePositions/"
directoryOut = f"./data/{train}/baseline_feats/"
fNames = os.listdir(directoryIn)
phoneData = [np.load(f"{directoryIn}{k}") for k in fNames]

# Convert position to acceleration
accelerations = [get_second_derivative(x) for x in phoneData]

#Feature extractions
accelerations = [extractFeats(x, 150) for x in accelerations]

#Print or use the extracted features as needed
for k in range(len(fNames)):
    fName = f"{directoryOut}{fNames[k]}"
    d = accelerations[k]
    fileObject = open(fName, 'wb')
    pkl.dump(d, fileObject)
    print(f"Wrote file: {k}")
