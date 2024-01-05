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

dataset_details = {
  "amass": {"accel": {"in": "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/phoneFeatures/",
                      "out": "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/predicted/",
                      "weights": "./weights/train_checkpointFirstTrain.pt",
                      "nFeatures": 75},
            "gyro": {"in": "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/phoneFeaturesGyro/",
                     "out": "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/predictedGyro/",
                     "weights": "./weights/train_checkpoint_gyro_current5.pt",
                     "nFeatures": 141},
  },
  "aist": {"accel": {"in": "/Users/pdealcan/Documents/github/EDGEk/data/accel/test/baseline_feats/",
                     "out": "/Users/pdealcan/Documents/github/EDGEk/data/accel/test/predicted/",
                     "weights": "./weights/train_checkpointFirstTrain.pt",
                     "nFeatures": 75},
           "gyro": {"in": "/Users/pdealcan/Documents/github/EDGEk/data/gyro/test/baseline_feats/",
                    "out": "/Users/pdealcan/Documents/github/EDGEk/data/gyro/test/predicted/",
                    "weights": "./weights/train_checkpoint_gyro_current5.pt",
                    "nFeatures": 141},
  },
  "brigitta": {"accel": {"in": "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/accel/phoneFeatures/",
                         "out": "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/accel/predicted/",
                         "weights": "./weights/train_checkpointFirstTrain.pt",
                         "nFeatures": 75},
               "gyro": {"in": "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/gyro/phoneFeatures/",
                        "out": "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/gyro/predicted/",
                        "weights": "./weights/train_checkpoint_gyro_current5.pt",
                        "nFeatures": 141},
  }
}

feature = "accel"
dataset = "amass"

def test(opt):
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    
    sample_size = 1
    print(f"Sample size {sample_size}")

    inputsPath = dataset_details[dataset][feature]["in"]
    render_dir = dataset_details[dataset][feature]["out"]
    nFeatures = dataset_details[dataset][feature]["nFeatures"]

    fNames = os.listdir(inputsPath) #Name of input file
    fileNames = [f"{render_dir}{x}".replace("npy", "csv") for x in fNames] #Name of output file

    model_weights = dataset_details[dataset][feature]["weights"]

    all_cond = [] 
    for j in fNames:
        df = np.load(f"{inputsPath}{j}")

        filE = np.float32(df)
        filE = torch.from_numpy(filE)
            
        fileE = filE.reshape(1, 150, -1)

        all_cond.append(fileE)
    
    model = EDGE(nFeatures, model_weights)
    model.eval()

    # directory for saving the dances
    fk_out = None
        
    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], [fileNames[i]]
        print(f"Inputing file: {fNames[i]}")
        model.render_sample(
#            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
            data_tuple, i, render_dir, render_count=-1, fk_out=fk_out, render=True
        )
    print("Done")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
