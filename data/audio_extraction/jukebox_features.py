import os
from functools import partial
from pathlib import Path

import jukemirlib
import numpy as np
from tqdm import tqdm

FPS = 30
LAYER = 66


def extract(fpath, skip_completed=True, dest_dir="aist_juke_feats"):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    audio = jukemirlib.load_audio(fpath)
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)

    #np.save(save_path, reps[LAYER])
    return reps[LAYER], save_path


def extract_folder(src, dest):
    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, skip_completed=False, dest_dir=dest)
    for fpath in tqdm(fpaths):
        rep, path = extract_(fpath)
        np.save(path, rep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")

    args = parser.parse_args()

    extract_folder(args.src, args.dest)
