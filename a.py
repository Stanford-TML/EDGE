import pandas as pd
import numpy as np
import os

fNames = os.listdir("./data/test/motions_sliced/")
fNames = [x.replace(".pkl", "") for x in fNames]

index = np.random.randint(0, len(fNames))

entire = pd.read_pickle(f"./data/test/motions_sliced/{fNames[index]}.pkl")
phone = pd.read_csv(f"./data/test/positionsPhone/{fNames[index]}.csv")
phone = phone.iloc[:, 0:3]
phone = phone.to_numpy()

a = np.concatenate((entire, phone), axis =1)
pd.DataFrame(a).to_csv("/Users/pdealcan/Downloads/out1.csv", index = False, header = False)
