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
    
    sample_size = 1
    print(f"Sample size {sample_size}")

    brigitta = True

    if brigitta:
        dirPhoneInput = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/phoneFeatures/"
        phoneInput = os.listdir(dirPhoneInput)
        dirFullBody = "../data/CoE/accel/brigittaData/convertedCSV/"


        #Selected subset of dances for eval
        phoneInput = ["58-LilyLaine_16", "12-HannaKuisma_21", "16-MikkoValtonen_ 03", "37-LeenaHjelt_beat_140", "50-JattaSavolainen_24", "36-SuviKulo_02", "46-KarlollinaKatainen_12", "46-KarlollinaKatainen_06", "47-EeviAntila_21", "57-AnnaNuoritimo_01", "01-MarjoPeltomaa_19", "48-MariaLaakso_18", "37-LeenaHjelt_02", "14-JohannaLaukkanen_20", "50-JattaSavolainen_16", "58-LilyLaine_20", "01-MarjoPeltomaa_06", "47-EeviAntila_14", "50-JattaSavolainen_ 03", "06-SariLiukkonen_ 03", "24-HeliTikkala_beat_140", "48-MariaLaakso_beatpractice_110", "11-JonnaAaltonen_24", "25-MesimaariaLammi_01", "11-JonnaAaltonen_beatpractice_110", "25-MesimaariaLammi_beatpractice_110", "03-HannaMarkuksela_beat_120", "30-KaisaPeltonen_12", "30-KaisaPeltonen_09", "47-EeviAntila_23", "24-HeliTikkala_10"]

        matchIndex, unMatchIndex = random.sample(range(len(phoneInput)), 2)
        caseRead = phoneInput[matchIndex].replace(".npy", "")

        features = np.load(f"{dirPhoneInput}{caseRead}.npy")
        features = pd.DataFrame(features)
        fullBody = pd.read_csv(f"{dirFullBody}{caseRead}.csv")
        
        randomInit = np.random.randint(600, features.shape[0]-150)
        features = features.iloc[randomInit:(randomInit+150), :]
        fullBody = fullBody.iloc[randomInit:(randomInit+300), :]
        fullBody = fullBody.to_numpy()
        fullBody = fullBody[:: 2, :] #Downsample
        fullBody = pd.DataFrame(fullBody)

        features.to_csv("./generatedDance/custom/phoneNormalizedMatch.csv", index = False, header = False)
        fullBody.to_csv("./generatedDance/custom/groundTruthMatch.csv", index = False, header = False)

        inputsPath = "./generatedDance/custom/"

    else:

        dirPhoneInput = "./data/test/baseline_feats/"
        phoneInput = os.listdir(dirPhoneInput)
        dirFullBody = "./data/test/motions_sliced_csv/"

        matchIndex, unMatchIndex = random.sample(range(100), 2)

        caseRead = phoneInput[matchIndex].replace(".npy", "")

        features = np.load(f"{dirPhoneInput}{caseRead}.npy")
        features = pd.DataFrame(features)
        fullBody = pd.read_csv(f"{dirFullBody}{caseRead}.csv")
        
        fullBody = fullBody.to_numpy()
        fullBody = fullBody[:: 2, :] #Downsample
        fullBody = pd.DataFrame(fullBody)

        features.to_csv("./generatedDance/originalAist/phoneNormalizedMatch.csv", index = False, header = False)
        fullBody.to_csv("./generatedDance/originalAist/groundTruthMatch.csv", index = False, header = False)

        inputsPath = "./generatedDance/originalAist/"

    fNames = ["phoneNormalizedMatch.csv"]
  
    all_cond = [] 
    for j in fNames:
       #Extract features
        df = pd.read_csv(f"{inputsPath}{j}", header=None)
        df = df.to_numpy()
        print(df.shape)

        filE = np.float32(df)
        filE = torch.from_numpy(filE)
            
        print(filE.shape)
        fileE = filE.reshape(1, 150, -1)
        print(filE.shape)

        all_cond.append(fileE)


    model = EDGE(opt.feature_type, "./weights/train_checkpoint_gyro_1000.pt")
    model.eval()

    # directory for saving the dances
    fk_out = None

    if brigitta:
        render_dir = "./generatedDance/custom/"
        fileNames = ["./generatedDance/custom/predictedFullMatch.csv"]
    else:
        render_dir = "./generatedDance/originalAist/"
        fileNames = ["./generatedDance/originalAist/predictedFullMatch.csv"]
        
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
