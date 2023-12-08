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
    file = ["./data/test/baseline_feats/gBR_sBM_cAll_d04_mBR0_ch02_slice0.npy", "./data/test/baseline_feats/gBR_sBM_cAll_d04_mBR0_ch02_slice1.npy"]
    file = [np.load(x) for x in file]
    file = torch.from_numpy(np.array(file))

    model = EDGE(opt.feature_type, "./weights/train-1.pt")
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    fileName = ["gBR_sBM_cAll_d04_mBR0_ch02_slice0.npy", "gBR_sBM_cAll_d04_mBR0_ch02_slice1.npy"]
    render_dir = "./generatedDance/"
    print("Generating dances")
    data_tuple = None, file, fileName
    model.render_sample(
        data_tuple, "test", render_dir, render_count=-1, fk_out=fk_out, render=True
    )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
