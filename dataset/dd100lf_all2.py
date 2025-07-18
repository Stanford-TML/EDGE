import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import os
import random
from tqdm import tqdm
from dataset.preprocess import vectorize_many
from dataset.preprocess import Normalizer

def paired_collate_fn(insts):
    mo_seq = list(zip(*insts))
    print(mo_seq.shape)
    mo_seq = torch.FloatTensor(mo_seq)
    print('Here!')
    print(mo_seq.size())

    return mo_seq

def swap_left_right_position(data):
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    # left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    # right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]
    left_hand_chain = [i for i in range(25, 40)]
    right_hand_chain = [i for i in range(40, 55)]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 22:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right(data):
    T, n_joints = data.shape[0], data.shape[1] // 3
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    positions = swap_left_right_position(positions).reshape(T, -1)

    return positions

def custom_collate_fn(batch):
    """
    Collate function for DD100lfAll dataset.
    Adapts to the output of DD100lfAll.__getitem__ (dict with keys: pos3dl, pos3df, music, [fname], [rot6dl], [rot6df])
    """
    # batch: list of dicts
    out = {}
    # Always present
    out['pos3dl'] = torch.stack([item['pos3dl'] for item in batch], dim=0)
    out['pos3df'] = torch.stack([item['pos3df'] for item in batch], dim=0)
    out['music'] = torch.stack([item['music'] for item in batch], dim=0)
    out['fname'] = [item['fname'] for item in batch]
    # Optional: rot6d
    if 'rot6dl' in batch[0] and batch[0]['rot6dl'] is not None:
        out['rot6dl'] = torch.stack([item['rot6dl'] for item in batch], dim=0)
    if 'rot6df' in batch[0] and batch[0]['rot6df'] is not None:
        out['rot6df'] = torch.stack([item['rot6df'] for item in batch], dim=0)
    return out

music_dtype_dict = {
    'librosa': 'feature',
    'jukebox': 'jukebox'
}

num_joints = 24

class DD100lfAll(Dataset):
    def __init__(self, feature_type, music_root, motion_root, split='train', interval=-1, dtype='pos3d', move=16, embeded_pos3d=False, backup_path=None):
        self.feature_type = feature_type
        self.music_dtype = music_dtype_dict[self.feature_type]
        
        # Add both pos3d and rot6d if dtype is 'both'
        if dtype == 'both':
            dtypes = ['pos3d', 'rot6d']
        elif isinstance(dtype, list):
            dtypes = dtype
        else:
            dtypes = [dtype]
        self.dtype = dtype
        self.names = []
        self.embeded_pos3d = embeded_pos3d
        self.train = split == 'train'
        self.interval = interval
        dances = {'pos3dl':[], 'pos3df':[], 'rot6dl':[], 'rot6df':[], 'music':[]}
        music_files = {}
        agent_files = {'leader':{}, 'follower':{}}

        assert move > 0 and move % 4 == 0, "move must be divisible by 4"
    
        def _process_files(agent_files, np_music):       
            def _process_dance(np_dance, dtype_folder):
                if dtype_folder == 'pos3d' and not self.embeded_pos3d:
                    root = np_dance[:, :3]  # the root
                    np_dance = np_dance - np.tile(root, (1, num_joints))  # Calculate relative offset with respect to root
                    np_dance[:, :3] = root
                # For rot6d, no processing needed
                else:
                    np_dance = np_dance[:, :num_joints*6]
                return np_dance

            this_pair = {}
            for agent in agent_files:
                # For each dtype_folder, load the corresponding file
                for dtype_folder in dtypes:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    if not os.path.isfile(dance_path):
                        continue
                    np_dance = np.load(dance_path)
                    np_dance = np_dance[:, :np_dance.shape[1]//55*num_joints]
                    np_dance = np_dance[:len(np_dance) - len(np_dance)%4, :] # to fit encodec down sample stragegy
                    np_dance = _process_dance(np_dance, dtype_folder)
                    this_pair[(agent, dtype_folder)] = np_dance

            # Only proceed if both leader and follower exist for all dtypes
            for dtype_folder in dtypes:
                if (('leader', dtype_folder) not in this_pair) or (('follower', dtype_folder) not in this_pair):
                    continue
                ldance = this_pair[('leader', dtype_folder)]
                fdance = this_pair[('follower', dtype_folder)]
                
                # # directly move the lroot to origin and froot correspond to it
                # if dtype_folder == 'pos3d' and self.embeded_pos3d:
                #     lroot = ldance[0, :3]
                #     fdance = fdance - np.tile(lroot, (fdance.shape[0], num_joints))
                #     ldance = ldance - np.tile(lroot, (ldance.shape[0], num_joints))
                
                
                seq_len =  min(len(ldance), len(fdance), len(np_music))
                if (interval is not None) and ( interval != 'None' ):
                    segment_length = self.interval
                    # context_length+1但不修改成员变量值，适配rigid_transform_dd100lf的尾部frame drop操作
                    for i in range(0, seq_len - segment_length, move):
                        np_dance_sub_seq_l = ldance[i: i + segment_length]
                        np_dance_sub_seq_f = fdance[i: i + segment_length]
                        np_music_sub_seq = np_music[i:i + segment_length]

                        if len(np_dance_sub_seq_l) != segment_length or len(np_dance_sub_seq_f) != segment_length or len(np_music_sub_seq) != segment_length:
                            continue

                        dances[dtype_folder+'l'].append(np_dance_sub_seq_l)
                        dances[dtype_folder+'f'].append(np_dance_sub_seq_f)
                        if dtype_folder == dtypes[0]:
                            dances['music'].append(np_music_sub_seq)
                            self.names.append(take)
                    done = 1
                else:
                    dances[dtype_folder+'l'].append(ldance[:seq_len//move*move])
                    dances[dtype_folder+'f'].append(fdance[:seq_len//move*move])
                    if dtype_folder == dtypes[0]:
                        dances['music'].append(np_music[:seq_len//move*move])
                        self.names.append(take)
                done = 1

        fnames = os.listdir(os.path.join(motion_root, 'pos3d', split))
        mnames = os.listdir(os.path.join(music_root,  self.music_dtype, split))

        for mname in mnames:
            path = os.path.join(music_root, self.music_dtype, split, mname)
            music_files[mname[:-4]] = path

        for fname in fnames:
            path = os.path.join(motion_root, 'pos3d', split, fname)
            if path.endswith('_00.npy'):
                agent_files['follower'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                agent_files['leader'][fname[:-7]] = path

        for take in tqdm(agent_files['follower'], desc='Processing files'):
            if take not in agent_files['leader'] or take not in music_files:
                continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32)
            
            # For each dtype, process files
            _process_files(agent_files, np_music)

        pose_input = self.process_dataset_dd100(dances['rot6df'], dances['pos3df'])
        self.data = {
            'pose': pose_input,
            'music': dances['music'],
        }

        assert len(pose_input) == len(dances['music'])
        assert len(pose_input) == len(self.names)
        self.length = len(pose_input)
        print('DD100lfAll dataset loaded!')   

    def process_dataset_dd100(self, rot6d, global_pos):
        # to Tensor
        rot6d = torch.from_numpy(np.array(rot6d))
        global_pos = torch.from_numpy(np.array(global_pos))
        root_pos = global_pos[:, :, :3]
        # to ax
        bs, sq, c = rot6d.shape
        rot6d = rot6d.reshape((bs, sq, -1, 6))
        
        feet = global_pos[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1])
        contacts = (feetv < 0.01).to(rot6d.dtype)  # cast to right dtype

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, rot6d]
        global_pose_vec_input = vectorize_many(l).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        # if self.train:
        self.normalizer = Normalizer(global_pose_vec_input)
        # else:
            # assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.interval > 0:
            global_pose_vec_input = global_pose_vec_input[:, :self.interval]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pose = self.data["pose"][index]
        music = self.data["music"][index]
        filename_ = self.names[index]
        
        return pose, music, filename_

