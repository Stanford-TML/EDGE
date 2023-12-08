import numpy as np

from sklearn.decomposition import PCA

from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from scipy.signal import find_peaks
import pywt  # PyWavelets library for wavelet transform

sr = 30

def getPCAproj(x):
    pca = PCA(n_components=3)
    pca.fit(x)
    proj = pca.transform(x)
    return proj

def get_second_derivative(position):
    dt = 1/30 #1 divided by frames per second
    velocity = np.gradient(position, dt, axis = 0)
    acceleration = np.gradient(velocity, dt, axis = 0)
    return acceleration

# Peak Features
def fP(x):
    x, _ = find_peaks(x)
    return x

def extractFeats(acceleration_data, windowLength):
    #Descriptives accross euclidien dimensions (x, y, z)
    mean_values = np.mean(acceleration_data, axis=1)
    std_dev_values = np.std(acceleration_data, axis=1)
    skewness_values = skew(acceleration_data, axis=1)
    kurtosis_values = kurtosis(acceleration_data, axis=1)

    descriptivesColumns = np.column_stack([mean_values, std_dev_values, skewness_values, kurtosis_values])

    #Descriptives accross time
    mean_values = np.mean(acceleration_data, axis=0)
    std_dev_values = np.std(acceleration_data, axis=0)
    skewness_values = skew(acceleration_data, axis=0)
    kurtosis_values = kurtosis(acceleration_data, axis=0)

    descriptivesRows = np.concatenate([mean_values, std_dev_values, skewness_values, kurtosis_values])

    all0 = np.tile(descriptivesRows, (windowLength, 1))

    # Time-domain Features
    min_values = np.min(acceleration_data, axis=0)
    max_values = np.max(acceleration_data, axis=0)
    range_values = np.ptp(acceleration_data, axis=0)
    rms_values = np.sqrt(np.mean(acceleration_data**2, axis=0))
    energy_values = np.sum(acceleration_data**2, axis=0)
    variance_values = np.var(acceleration_data, axis=0)

    timeDescriptives = np.concatenate([min_values, max_values, range_values, rms_values, energy_values, variance_values])
    all1 = np.tile(timeDescriptives, (windowLength, 1))

    # Frequency-domain Features
    fft_values = np.abs(fft(acceleration_data, axis=0))

    # Time-Frequency Features (using Wavelet Transform)
    coeffs, _ = pywt.cwt(acceleration_data, np.arange(1, 10), 'gaus1')
    coeffs = np.column_stack(coeffs)

    frequencyDescriptives = np.column_stack([fft_values, coeffs])
    all2 = frequencyDescriptives

    # Signal Magnitude Area (SMA)
    sma_values = np.sum(np.abs(acceleration_data), axis=0)

    # Zero Crossing Rate
    zero_crossing_rate_values = np.sum(np.diff(np.sign(acceleration_data), axis=0) != 0, axis=0)

    magZero = np.concatenate([sma_values, zero_crossing_rate_values])
    all3 = np.tile(magZero, (windowLength, 1))

    # Correlation between Axes
    correlation_matrix = np.corrcoef(acceleration_data, rowvar=False)
    correlation_xy = correlation_matrix[0, 1]
    correlation_yz = correlation_matrix[1, 2]
    correlation_xz = correlation_matrix[0, 2]

    correlations = np.array([correlation_xy, correlation_yz, correlation_xz])
    correlations = np.tile(correlations, (windowLength, 1))

    peaks = [fP(acceleration_data[:, x]) for x in range(3)]  # Replace 0 with the axis of interest
    number_of_peaks = np.array([len(x) for x in peaks])

    nPeaks = np.tile(number_of_peaks, (windowLength, 1))
    nPeaks = np.column_stack([correlations, nPeaks])

    #PCA projections
    pcas = getPCAproj(acceleration_data)

    all4 = np.column_stack([nPeaks, pcas])

    allFeatures = np.column_stack([all0, all1, all2, all3, all4])

    return allFeatures

