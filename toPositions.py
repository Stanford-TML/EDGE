import pandas as pd
import numpy as np
import torch
import sys
import os

import json

import pickle as pkl

from dataset.dance_dataset import *

from vis import SMPLSkeleton

from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)

# CONVERTS SMPL REPRESENTATIONS TO POSITIONS, AND CREATES A ROOT FOR THE CENTER OF
# A PHONE AS THE AVERAGE OF HIP AND KNEE MARKERS

#markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]
name1 = 'pos'
name2 = 'q'
def smplToPosition(df, name1 = 'pos', name2 = 'q', aist = True):
    smpl = SMPLSkeleton()
    # to Tensor
    root_pos = torch.Tensor(np.array([df[name1]]))
    local_q = torch.Tensor(np.array([df[name2]]))
    # to ax
    bs, sq, c = local_q.shape
    local_q = local_q.reshape((bs, sq, -1, 3))
    # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
    root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor(
        [0.7071068, 0.7071068, 0, 0]
    )  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    local_q[:, :, :1, :] = root_q
    # don't forget to rotate the root position too 😩
    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(
        root_pos
    )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
    # do FK
    print(local_q.shape)
    print(root_pos.shape)
    positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
    #positions = positions.reshape(17733, 150, -1)
    #positions = positions.reshape(186, 150, -1)
    positions = positions[0]
    positions = positions.numpy()
    if aist:
        positions = positions.reshape(300, 72)
#    positions = positions.reshape(-1, 150, 72)
    #Tenho que transformar de volta em numpy, não Torch tensor
    return positions


def wrapMotionConverter(train):
    directoryIn = f"./data/{train}/motions_sliced_original/"
    directoryOut = f"./data/{train}/motions_sliced/"
    os.makedirs(directoryOut, exist_ok=True)
    files = os.listdir(directoryIn) #SMPL files
    for k in range(len(files)):
        df = pd.read_pickle(f"{directoryIn}{files[k]}")
        df = smplToPosition(df)
    #        df = df.numpy()
        fName = f"{directoryOut}{files[k]}"
        fileObject = open(fName, 'wb')
        pkl.dump(df, fileObject)
        print(f"Wrote file: {k}")

#Converting AIST++ to positions
#train = "test"
#wrapMotionConverter(train)
#train = "train"
#wrapMotionConverter(train)

#Converting AMASS to position
a = np.load("/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDB/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_poses.npz")
[print(b) for b in a]

df = {}
df['q'] = a['poses'][:,3:75]
df['pos'] = a['poses'][:,0:3]
df['pos'].shape
df['q'].shape
#df = smplToPosition(df, aist = False)
#df = df.reshape(-1, 72)

#df = pd.DataFrame(df)
smpl = SMPLSkeleton()
root_pos = torch.Tensor(np.array([df[name1]]))
local_q = torch.Tensor(np.array([df[name2]]))

bs, sq, c = local_q.shape
local_q = local_q.reshape((bs, sq, -1, 3))

positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
positions = positions.reshape(bs, sq, 72)
positions = positions[0]

df = pd.DataFrame(positions)
df.to_csv("/Users/pdealcan/Documents/github/EDGEk/test.csv", index = False, header = False)





#Converting Phantom to positions
pathIn = "/Users/pdealcan/Documents/github/data/CoE/accel/phantomDanceData/motion/"
files = os.listdir(pathIn)
f = open(f"{pathIn}{files[60]}")
df = json.load(f)
df['pos'] = df.pop('root_positions')
df['q'] = df.pop('rotations')
df = {'pos': np.array(df['pos']),
      'q': np.array(df['q'])}
#df['pos'].shape
#df['q'].shape
df['q'] = df['q'][:, :, 0:3]
df['q'] = df['q'].reshape(-1, 72)
df['q'].shape

df = smplToPosition(df, aist = False)
df = df.reshape(-1, 72)
df = pd.DataFrame(df)
df.to_csv("/Users/pdealcan/Documents/github/EDGEk/test.csv", index = False, header = False)





#Converting AIST++ to positions
#train = "test"
directoryIn = f"/Users/pdealcan/Documents/github/motions_sliced_original_test/"
files = os.listdir(directoryIn) #SMPL files
df = pd.read_pickle(f"{directoryIn}{files[50]}")
df['pos'].shape
df['q'].shape

df = smplToPosition(df)
df = pd.DataFrame(df)

df.to_csv("/Users/pdealcan/Documents/github/EDGEk/test.csv", index = False, header = False)
