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

#marker starts at 0
def getMarker(npMatrix, mNumber):
    markerIndexes = 3 * mNumber + np.arange(-2, 1)
    npMatrix = npMatrix[:,markerIndexes]
    return npMatrix

def createPhone(df, rightHipMarker = 2, rightKneeMarker = 5, leftHipMarker = 1):
    rightHip = getMarker(df, rightHipMarker)
    rightKnee = getMarker(df, rightKneeMarker)
    leftHip = getMarker(df, leftHipMarker)
    phone = np.add(rightHip, rightKnee)/2 #Phone root calculated between right hip and knee markers
    phoneUp = np.add(phone, rightHip)/2 #Up of phone calculated between root of phone and right hip
    middleHip = np.add(leftHip, rightHip)/2
    phoneLateral = np.add(middleHip, rightKnee)/2;
    entirePhone = np.concatenate((phone, phoneUp, phoneLateral), axis = 1)
    return entirePhone

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

def extractGyro(justPhone): #justPhone is a 9 array matrix representing simulated markers for the phone
    #Indices of axis
    xInd = 0
    yInd = 2
    zInd = 1
    # Getting each marker of virtual phone. 1=root; 2=up; 3=side
    m1 = getMarker(justPhone, 0) #Root
    m2 = getMarker(justPhone, 1) #Upper part of phone
    m3 = getMarker(justPhone, 2) #Side of phone
    a1 = m1[:, xInd] - m2[:, xInd]
    a2 = m1[:, zInd] - m2[:, zInd]
    a3 = m2[:, zInd] - m3[:, zInd]
    dM1M2 = np.sqrt(np.sum((m1 - m2) ** 2, axis=1))
    dM2M3 = np.sqrt(np.sum((m2 - m3) ** 2, axis=1))
    pitch = np.arcsin(a1 / dM1M2)
    yaw = np.arcsin(a2 / dM1M2)
    roll = np.arcsin(a3 / dM2M3)
    # Convert angles from radians to degrees
    pitch_degrees = np.degrees(pitch)
    yaw_degrees = np.degrees(yaw)
    roll_degrees = np.degrees(roll)
    gyro = np.column_stack((roll_degrees, yaw_degrees, roll_degrees))
    return gyro

def extractPhoneFeatures(train):
    #Read phone and ectract features
    directoryIn = f"../../data/{train}/positionsPhone/"
    directoryOut = f"../../data/{train}/baseline_feats/"
    os.makedirs(directoryOut, exist_ok=True)
    files = os.listdir(directoryIn)
    for k in files:
        phone = pd.read_csv(f"{directoryIn}{k}")
#        gyro = extractGyro(phone)
#        phone = getMarker(phone, 0) #Getting root for acceleration
        #Concatenate phone root + gyro. Should result in a six dimensions array
#        phone = np.concatenate((phone, gyro), axis = 1)
        assert phone.shape[1] == 6
        #Downsampling to half
        phone = phone.to_numpy()
        phone = phone[:: 2, :]         
        #Position to accel and angles to angular velocity
#        df = get_second_derivative(phone)
        df = extractFeats(phone, phone.shape[0])
        df = np.float32(df)
        fName = f"{directoryOut}{k}"
        fName = fName.replace(".csv", ".npy")
        np.save(fName, df)
        print(f"Wrote file: {fName}")

#Processing AIST++ dataset
for k in ["train", "test"]:
    extractPhoneFeatures(k)
#for k in ["test"]:
#    writePhonePositions(k)
#    print(f"Finished extracting features of {k}")
#    print(f"Finished extracting features of {k}")

#Processing Brigitta's dataset
#dirIn = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/convertedCSV/"
#dirOut = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/phoneExtracted/"
#allFiles = os.listdir(dirIn)
#rightHipMarker = 1; rightKneeMarker = 2; leftHipMarker = 5 #Confirmed from Briggitta's data
#for k in allFiles:
#    df = pd.read_csv(f"{dirIn}{k}")
#    df = np.array(df)
#    ph = createPhone(df, rightHipMarker, rightKneeMarker, leftHipMarker)
#    gyro = extractGyro(ph)
#    ph = getMarker(ph, 0) #Getting root of phone for acceleration
#    ph = ph/100 #Rescaling to match dataset
#    ph = np.concatenate((ph, gyro), axis = 1)
#    assert ph.shape[1] == 6
#    ph = ph[:: 2, :]         
#    df = get_second_derivative(ph)
#    df = extractFeats(df, df.shape[0])
#    df = np.float32(df)
#    fName = f"{dirOut}{k}"
#    fName = fName.replace("csv", "npy")
#    if ("beat" in fName) == False:
#        np.save(fName, df)
#    else:
#        print("Not saving cause of beat")


