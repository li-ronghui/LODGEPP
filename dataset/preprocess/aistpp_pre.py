# 还存在的问题: 音频数据的对齐, 是否和其他 motion 位姿一样等
import sys
sys.path.append('/data2/lrh/dyq/debug_exp/dataset/') # 放在这个目录下使用
from utils.format import ax_to_6v
from aistplusplus_api.aist_plusplus.loader import AISTDataset
from tqdm import tqdm
import torch
import numpy as np
import os

def get_filenames_without_extension(folder_path):
    filenames = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            filename_without_extension, _ = os.path.splitext(file_name)
            filenames.append(filename_without_extension)
    return filenames

# save_dir
save_dir = '/data2/lrh/dyq/debug_exp/dataset/data/mofeature135/mofeature_aistpp/'
# aist++ 文件夹目录
anno_dir = '/data2/lrh/human_datasets/aist_plusplus_final//'
# 得到所有的 video_name
aist_dataset = AISTDataset(anno_dir)
motion_names = get_filenames_without_extension(aist_dataset.motion_dir)

aist_fps = 60
down_fps = 20
down_sample = aist_fps//down_fps

# 读取 AIST++ 数据集
# aistpp_motion = []
for seq_name in tqdm(motion_names, desc='AIST++'):
    # SMPL joints
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
        aist_dataset.motion_dir, seq_name)
    # print(smpl_poses.shape) # (2302, 72)
    # print(smpl_trans.shape) # (2302, 3)

    # 降采样
    smpl_poses = smpl_poses[::down_sample]
    smpl_trans = smpl_trans[::down_sample]

    smpl_poses = torch.from_numpy(smpl_poses).float()
    smpl_trans = torch.from_numpy(smpl_trans).float()
    
    
    # 处理成 3 + 22*6 的格式
        # SMPL 格式转换 (*,24 x 3) 轴角axis -> (*, 22 x 6) rot6d 格式
    smpl_poses = smpl_poses[:,:-6].reshape(-1,22,3) # 去掉最后 2*3 的 SMPL 关节点数据
    smpl_poses = ax_to_6v(smpl_poses) # torch.Size([2302, 22, 6]) rot6d 格式
    smpl_poses = smpl_poses.reshape(-1,22*6)

    dance_data = torch.cat([smpl_trans,smpl_poses],dim = 1)[:,:135] # 3 + 22*6(135) or 268 

    # aistpp_motion.append(dance_data) # 利用 list 能够处理变长数据 | 为了能够和 music 对齐, 保留 motion 的分段性, 使用时随机截取部分
    dance_data = dance_data.cpu().numpy()
    np.save(save_dir + seq_name + '.npy', dance_data) # 保存到这个文件夹内