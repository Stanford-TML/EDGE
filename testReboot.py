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
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    
    sample_size = 3
    print(f"Sample size {sample_size}")

    inputsPath = "./generatedDance/custom/"
#    all_filenames = ["phoneNormalizedUnMatch.csv", "phoneNormalizedMatch.csv"]
    all_filenames = ["phoneNormalizedMatch.csv"]

    print("Using precomputed features")
    # all subdirectories

  
    all_cond = [] 
    for j in all_filenames:
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
        fileE = filE.reshape(2, 150, -1)

        all_cond.append(fileE)



    model = EDGE(opt.feature_type, "./weights/train_checkpoint.pt")
    model.eval()

    # directory for saving the dances
    fk_out = None
    render_dir = "./generatedDance/custom/"

    fileNames = ["./generatedDance/custom/predictedFullMatch.csv"]

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], [fileNames[i]]
        print(f"Inputing file: {all_filenames[i]}")
        model.render_sample(
#            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
            data_tuple, i, render_dir, render_count=-1, fk_out=fk_out, render=True
        )
    print("Done")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
