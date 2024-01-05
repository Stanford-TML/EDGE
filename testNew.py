import pandas as pd
import numpy as np
import os
import json

pathIn = "/Users/pdealcan/Documents/github/data/CoE/accel/phantomDanceData/motion/"
files = os.listdir(pathIn)

f = open(f"{pathIn}{files[0]}")
df = json.load(f)
[print(x) for x in df]
df['root_positions']

pd.DataFrame(df)
