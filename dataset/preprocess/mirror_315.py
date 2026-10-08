import sys
from finedance_preprocess import swap_left_right
from finedance_preprocess import ax_to_6v
import numpy as np
import os
import torch

modir = "dataset/data/origin/motion_feature315"
outdir = "dataset/data/origin/motion_feature315mirror"
if not os.path.exists(outdir):
    os.makedirs(outdir)

for file in os.listdir(modir):
    mofile = os.path.join(modir, file)
    modata = np.load(mofile)
    modata = torch.from_numpy(modata).float().cuda()

    root = modata[:,:3]
    pose = modata[:,3:]

    MirrorMotion = swap_left_right(pose)
    MirrorMotion = ax_to_6v(MirrorMotion).view(MirrorMotion.shape[0], 52*6)
    print("MirrorMotion.shape", MirrorMotion.shape)
    Mroot = root.clone()
    Mroot[:,0] *= -1
    MirrorMotion = torch.cat([Mroot, MirrorMotion], dim=1).detach().cpu().numpy()

    np.save(os.path.join(outdir, 'M'+file), MirrorMotion)
    # sys.exit(0)
