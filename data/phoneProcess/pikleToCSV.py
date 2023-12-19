import os
import pandas as pd
import numpy as np

train = ["train", "test"]
for q in train:
    dirIn = f"../{q}/motions_sliced/"
    dirOut = f"../{q}/motions_sliced_csv/"
    fileNames = os.listdir(dirIn)
    fileNames = [x.replace(".pkl", "") for x in fileNames]

    for k in fileNames:
        print(k)
        df = pd.read_pickle(f"{dirIn}{k}.pkl")
        df = pd.DataFrame(df)
        df.to_csv(f"{dirOut}{k}.csv", index = False, header = False)
        print(k)

    print("Finished test")
