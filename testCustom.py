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

    inputsPath = "./generatedDance/custom/"
    fNames = ["phoneNormalizedUnMatch.csv", "phoneNormalizedMatch.csv"]
  
    file = [] 
    for j in fNames:
       #Extract features
        df = pd.read_csv(f"{inputsPath}{j}")
        df = df.to_numpy()

        print(df.shape)
        df = get_second_derivative(df)
        df = extractFeats(df, df.shape[0])
        print(f"Shape of input is: {df.shape}")

        filE = np.array([df])
        filE = np.float32(filE)
        filE = torch.from_numpy(filE)

        print(filE.shape)
        filE = filE.reshape(2, 150, -1)
        print(filE.shape)

        file.append(filE)

    fk_out = None

    model = EDGE(opt.feature_type, "./weights/train_checkpoint.pt")
    model.eval()

    fileNames = ["./generatedDance/custom/predictedFullUnMatch.csv", "./generatedDance/custom/predictedFullMatch.csv"]
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
