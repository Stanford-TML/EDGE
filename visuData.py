import pandas as pd
import numpy as np
import torch

import pickle as pkl

from dataset.dance_dataset import *

from vis import SMPLSkeleton

from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)

def createPhone(df):
    phone = (df[:,6:9] + df[:,15:18])/2
#    df = np.concatenate([df, phone], axis = 1)
    return phone

#df = pd.read_pickle("./data/dataset_backups/processed_train_data.pkl")
#df = pd.read_pickle("./data/dataset_backups/processed_test_data.pkl")
df = pd.read_pickle("./data/train/motions_sliced/gBR_sBM_cAll_d04_mBR1_ch03_slice0.pkl")

markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]

smpl = SMPLSkeleton()


[print(x) for x in df]
# to Tensor
root_pos = torch.Tensor(df['pos'])
local_q = torch.Tensor(df['q'])

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
positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
#positions = positions.reshape(17733, 150, -1)
positions = positions.reshape(186, 150, -1)
#Tenho que transformar de volta em numpy, não Torch tensor

phone = np.array(list(map(createPhone, positions)))
#Writting phone data to train folder
directoryOut = "/Users/pdealcan/Documents/github/EDGE/"
for k in range(phone.shape[0]):
    fName = f"{directoryOut}{df['filenames'][k]}"
    d = phone[k]
    with open(fName, 'wb') as f:
        np.save(f, d)
    print(f"Wrote file: {k}")

train = "train"
#Writting train movement data as positions instead of quaternions
fNames = [x.replace(f"data/{train}/baseline_feats", "") for x in df['filenames']]
fNames = [x.replace("npy", "pkl") for x in fNames]
directoryOut = f"/Users/pdealcan/Documents/github/EDGE/data/{train}/motions_sliced"

positions = positions.numpy()
for k in range(len(fNames)):
    fName = f"{directoryOut}{fNames[k]}"
    d = positions[k]
    fileObject = open(fName, 'wb')
    pkl.dump(d, fileObject)
    print(f"Wrote file: {k}")
    
#pd.DataFrame(positions[0]).to_csv("./temp.csv", index = False, header = False)

