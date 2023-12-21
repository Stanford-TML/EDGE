import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

from data.phoneProcess.customFeatureExtract import *

import pandas as pd
import numpy as np

import random

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0

stringintkey = cmp_to_key(stringintcmp_)

def test(opt):
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    print("Using precomputed features")

    dirPhoneInput = "../data/CoE/accel/brigittaData/phoneExtracted/"
    phoneInput = os.listdir(dirPhoneInput)
    dirFullBody = "../data/CoE/accel/brigittaData/convertedCSV/"

    matchIndex, unMatchIndex = random.sample(range(100), 2)

    caseRead = phoneInput[matchIndex].replace(".npy", "")

    features = np.load(f"{dirPhoneInput}{caseRead}.npy")
    features = pd.DataFrame(features)
    fullBody = pd.read_csv(f"{dirFullBody}{caseRead}.csv")

    randomInit = np.random.randint(100, features.shape[0]-150)
    features = features.iloc[randomInit:(randomInit+151), :]
    fullBody = fullBody.iloc[randomInit:(randomInit+301), :]
    fullBody = fullBody.to_numpy()
    fullBody = fullBody[:: 2, :] #Downsample
    fullBody = pd.DataFrame(fullBody)

    features.to_csv("./generatedDance/custom/phoneNormalizedMatch.csv", index = False, header = False)
    fullBody.to_csv("./generatedDance/custom/groundTruthMatch.csv", index = False, header = False)

    inputsPath = "./data/generatedDance/custom/"
    fNames = ["phoneNormalizedMatch.csv"]

    file = [] 
    for j in fNames:
       #Extract features
        df = pd.read_csv(f"./generatedDance/custom/{j}")
        df = df.to_numpy()
        df = np.float32(df)
        print(f"Shape of input is: {df.shape}")
        df = torch.from_numpy(df)
        file.append(df)

    fk_out = None

    model = EDGE(opt.feature_type, "./weights/train_checkpoint_gyro_1000.pt")
    model.eval()

    fileNames = ["./generatedDance/custom/predictedFullMatch.csv"]
    render_dir = "./generatedDance/custom/"

    print("Generating dances")
    for k in range(len(fileNames)):
        data_tuple = None, file[k], [fileNames[k]]
        model.render_sample(
            data_tuple, k, render_dir, render_count=-1, fk_out=fk_out, render=True
        )

    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
