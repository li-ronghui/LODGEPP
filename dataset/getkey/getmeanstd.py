import numpy as np
import sys
import os
from os.path import join as pjoin
from tqdm import tqdm


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in tqdm(file_list):
        if file[-3:] != 'npy':
            continue
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:3] = Std[0:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4:5] = Std[4:5].mean() / 1.0
    Std[5:7] = Std[5:7].mean() / 1.0
    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    data_dir = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea290_ori/traj_4'
    save_dir = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea290_ori/traj_4/Norm'
    mean, std = mean_variance(data_dir, save_dir)