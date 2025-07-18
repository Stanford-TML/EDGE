import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.preprocess import Normalizer, vectorize_many
from vis import SMPLSkeleton


class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 30
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        # backup_path is a str, ensure directory exists
        if not os.path.exists(backup_path):
            os.makedirs(backup_path, exist_ok=True)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos3df'].shape}, Rot6d: {data['rot6df'].shape}, Music: {data['music'].shape}"
        )

        # process data, convert to 6dof etc
        pose_input = self.process_dataset_dd100(data["pos3df"], data["rot6df"], data["pos3df"])
        self.data = {
            "pose": pose_input,
            "music": data["music"],
            "filenames": data["fname"],
        }
        assert len(pose_input) == len(data["music"])
        
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        # music feature is already loaded as numpy array, so just convert to torch
        music_feature = torch.from_numpy(self.data["music"][idx]).float()
        return self.data["pose"][idx], music_feature, filename_

    def load_aistpp(self):
        # 这里假设DD100lfAll的输出目录结构如下：
        # data_path/
        #   |- train/ 或 test/
        #       |- pos3df.npy
        #       |- rot6df.npy
        #       |- music.npy

        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        pos3df_path = os.path.join(split_data_path, "motion")
        music_path = os.path.join(split_data_path, "music")

        # 直接加载全部数据
        pos3df = np.load(pos3df_path)  # N x seq x 3
        music = np.load(music_path)    # N x seq x feat_dim (已是jukebox特征)

        # 下采样
        pos3df = pos3df[:, :: self.data_stride, :]
        rot6df = rot6df[:, :: self.data_stride, :]
        music = music[:, :: self.data_stride, :]

        data = {
            "pos3df": pos3df,
            "rot6df": rot6df,
            "music": music,
            
        }
        return data

    def process_dataset_dd100(self, root_pos, rot6d, global_pos):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        rot6d = torch.Tensor(rot6d)
        global_pos = torch.Tensor(global_pos)
        # to ax
        bs, sq, c = rot6d.shape
        rot6d = rot6d.reshape((bs, sq, -1, 6))
        
        feet = global_pos[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(rot6d)  # cast to right dtype

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, rot6d]
        global_pose_vec_input = vectorize_many(l).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input
   

class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "baseline",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4",
                "mKR2",
                "mBR0",
                "mLO2",
                "mJB5",
                "mWA0",
                "mJS3",
                "mMH3",
                "mHO5",
                "mPO1",
            ]
        )
        self.train = train

        # if not aist, then set train to true to ignore test split logic
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True

        self.data = self.load_music()  # Call this last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        # now we have a batch of filenames
        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        # get the features
        features = np.array([np.load(x) for x in filenames])

        return torch.Tensor(features), seq_slice

    def load_music(self):
        # open data path
        split_data_path = os.path.join(self.data_path)
        music_path = os.path.join(
            split_data_path,
            f"{self.data_name}_baseline_feats"
            if self.feature_type == "baseline"
            else f"{self.data_name}_juke_feats/juke_66",
        )
        self.music_path = music_path
        # get the music filenames strided, with each subsequent item 5 slices (2.5 seconds) apart
        all_names = []

        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
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

        for features in glob.glob(os.path.join(music_path, "*.npy")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict
