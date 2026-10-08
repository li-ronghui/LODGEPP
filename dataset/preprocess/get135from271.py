import sys
from networkx import out_degree_centrality
import numpy as np
import os
from tqdm import tqdm
modir="/data2/lrh/dataset/HumanML3D/mofeatures_full271"
outdir="/data2/lrh/dataset/HumanML3D/mofeatures_135"

if not os.path.exists(outdir):
    os.makedirs(outdir)

for file in tqdm(os.listdir(modir)):
    mofile = os.path.join(modir, file)
    modata = np.load(mofile)

    root = modata[:,3+22*6:3+22*6+3]
    pose = modata[:,3:3+22*6]

    data = np.concatenate([root,pose], axis=-1)
    np.save(os.path.join(outdir, file), data)