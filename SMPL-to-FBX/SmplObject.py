import glob
import os
import pickle
from typing import Dict, Tuple

import numpy as np


class SmplObjects(object):
    joints = [
        "m_avg_Pelvis",
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ]

    def __init__(self, read_path):
        self.files = {}

        paths = sorted(glob.glob(os.path.join(read_path, "*.pkl")))
        for path in paths:
            filename = path.split("/")[-1]
            with open(path, "rb") as fp:
                data = pickle.load(fp)
            self.files[filename] = {
                "smpl_poses": data["smpl_poses"],
                "smpl_trans": data["smpl_trans"],
            }
        self.keys = [key for key in self.files.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        key = self.keys[idx]
        return key, self.files[key]
