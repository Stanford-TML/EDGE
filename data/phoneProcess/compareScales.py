import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

aist = "/Users/pdealcan/Documents/github/EDGEk/data/test/positionsPhone/"

aistFiles = os.listdir(aist)
aistC = [pd.read_csv(f"{aist}{k}").iloc[:,0:3] for k in aistFiles]

maxes = [np.max(np.max(k)) for k in aistC]
mins = [np.min(np.min(k)) for k in aistC]

minAccel = np.median(mins)
maxesAccel = np.median(maxes)

scaled_matrix_A = scaler.fit_transform(matrix_A)
