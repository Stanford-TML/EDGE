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
#    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
#    sample_length = opt.out_length
#    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    print("Using precomputed features")
    # all subdirectories
#    dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
    inputsPath = "./data/test/baseline_feats/"
    fNames = os.listdir(inputsPath)
    index = np.random.randint(len(fNames))

    file = [np.load(f"{inputsPath}{fNames[index]}")]
    file = torch.from_numpy(np.array(file))

    file.shape

    model = EDGE(opt.feature_type, "./weights/train_checkpoint_gyro.pt")
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    fileName = [fNames[index].replace(".npy", "")]
    render_dir = "./generatedDance/pairGenerated/"

    os.makedirs(render_dir, exist_ok=True)

    print("Generating dances")
    data_tuple = None, file, fileName
    model.render_sample(
        data_tuple, fileName, render_dir, render_count=-1, fk_out=fk_out, render=True
    )

    f = f"{fNames[index]}".replace("npy", "pkl")
    d = pd.read_pickle(f"./data/test/motions_sliced/{f}")
    d = d[:: 2, :]
    d = pd.DataFrame(d)
    f2 = f.replace("pkl", "csv")
    d.to_csv(f"{render_dir}true_{f2}", index = False, header = False)

    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
