import os
import torch
import numpy as np
from dld.data.utils.smplfk import set_on_ground_139, SMPLX_Skeleton
# d = {'a':1,'b':2}
# print(d)
# out = {}
# for key in d.keys():
#     print(key)
#     out[key] = d[key]

# print(out)

# ckptpath = '/data2/lrh/project/dance/edge_pylight/experiments/Edge_Module/Double266_0103_Norm_128len_266_diff_bc768/checkpoints/epoch=1999.ckpt'
# dict = torch.load(ckptpath)
# print(dict.keys())

# data = np.load("/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/FineDance_relative_Norm_GenreDis_bc190/val299/samples_2024-03-05-10-53-46/000_299.npy")
# print(data.shape)

smplx_model = SMPLX_Skeleton()
data_gound = np.load('/data2/lrh/dataset/fine_dance/gound/mofea319/001.npy')[:10, :139]
motion = torch.from_numpy(data_gound)
motion = set_on_ground_139(motion, smplx_model, -1.2)
motion = motion.detach().cpu().numpy()
print(motion.shape)
# if "":
#     print('  is true')
# else:
#     print(" flase")