import pickle
import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
import json
import random
# import torchgeometry as tgy
import sys
import codecs as cs
from .utils.format import ax_to_6v
from .aistplusplus_api.aist_plusplus.loader import AISTDataset


SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]

SMPLX_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13,
                        15, 17, 16, 19, 18, 21, 20, 22, 24, 23,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
SMPLX_POSE_FLIP_PERM = []
for i in SMPLX_JOINTS_FLIP_PERM:
    SMPLX_POSE_FLIP_PERM.append(3*i)
    SMPLX_POSE_FLIP_PERM.append(3*i+1)
    SMPLX_POSE_FLIP_PERM.append(3*i+2)

def flip_pose(pose):
    #Flip pose.The flipping is based on SMPLX parameters.
    pose = pose[:,SMPLX_POSE_FLIP_PERM]
    # we also negate the second and the third dimension of the axis-angle
    pose[:,1::3] = -pose[:,1::3]
    pose[:,2::3] = -pose[:,2::3]
    return pose


class motion_Smpl(data.Dataset):
    def __init__(self, args, istrain): # 通过 istrain 来选择使用的数据集
        self.args = args
        self.aistpp_motion = []
        self.FineDance_motion = []
        self.amass_motion = []

        if "all" in args.dataset or "aistpp" in args.dataset:
            # AIST++ motion
                # 得到 AIST++ dance data from aistpp_api/demos/extract_motion_feats.py
            anno_dir = args.aistpp_dir

                # 得到所有的 video_name
            aist_dataset = AISTDataset(anno_dir)
            # motion_names = get_filenames_without_extension(aist_dataset.motion_dir)
            if istrain:
                split_file = anno_dir + 'splits/crossmodal_train.txt'
                print('..training dataset loading')
            else:
                split_file = anno_dir + 'splits/crossmodal_test.txt'
                print('..test dataset loading')
            motion_names = []

            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    motion_names.append(line.strip())
            
                # 读取 AIST++ 数据集
            
            for seq_name in tqdm(motion_names, desc='AIST++'):
                # SMPL joints
                smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
                    aist_dataset.motion_dir, seq_name)
                # print(smpl_poses.shape) # (2302, 72)
                # print(smpl_trans.shape) # (2302, 3)
                smpl_poses = torch.from_numpy(smpl_poses).float()
                smpl_trans = torch.from_numpy(smpl_trans).float()
                
                # 处理成 3 + 22*6 的格式
                    # SMPL 格式转换 (*,24 x 3) 轴角axis -> (*, 22 x 6) rot6d 格式
                smpl_poses = smpl_poses[:,:-6].reshape(-1,22,3) # 去掉最后 2*3 的 SMPL 关节点数据
                smpl_poses = ax_to_6v(smpl_poses) # torch.Size([2302, 22, 6]) rot6d 格式
                smpl_poses = smpl_poses.reshape(-1,22*6)

                dance_data = torch.cat([smpl_trans,smpl_poses],dim = 1)[:,:args.DATA_SETTING.nfeats] # 3 + 22*6(135) or 268 


                self.aistpp_motion.append(dance_data) # 利用 list 能够处理变长数据 | 为了能够和 music 对齐, 保留 motion 的分段性, 使用时随机截取部分

            print(f'AIST++ has {len(self.aistpp_motion)} samples..')


        elif "all" in args.dataset or "finedance" in args.dataset:
            # FineDance
            self.fd_motion_dir = args.fd_motion_dir
            self.istrain = istrain
            self.motion_index = []
            self.name = []
            motion_all = []
            key_idx = {}
            with open("tools/musicpeak.pkl", "rb") as f1:
                music_peak = pickle.load(f1)
            with open("tools/motionbeat.pkl", "rb") as f2:
                motion_beat = pickle.load(f2)

         
            for key_ in motion_beat.keys():
                key = key_[1:] if key_[0] == 'M' else key_
                key_idx[key_] = list(set(music_peak[key]['beat_idxs'].tolist()      \
                                        + music_peak[key]['onset_idxs'].tolist()    \
                                        + motion_beat[key_].tolist()                \
                                                ))
                key_idx[key_].sort()
                
            #     print(len(key_idx[key_] ))
            # print("key_num", key_num)
    

            ignor_list, train_list, test_list = self.get_train_test_list()
            if self.istrain:
                self.datalist= train_list + test_list
            else:
                self.datalist = test_list

            key_num = 0
            for name in tqdm(self.datalist, desc='FineDance'):
                name = name + ".npy"
                if name[:-4] in ignor_list:
                    continue
                
            
                if args.DATA_SETTING.nfeats == 132:
                    motion = np.load(os.path.join(self.fd_motion_dir, name))[:,3:3+args.DATA_SETTING.nfeats]
                elif args.DATA_SETTING.nfeats == 144:
                    mo_temp = np.load(os.path.join(self.fd_motion_dir, name))
                    rot_append = np.zeros([mo_temp.shape[0], 6], dtype=np.float32)
                    rot_append[:,:3] = mo_temp[:, 3+22*6:3+22*6+3]

                    padding = np.zeros([mo_temp.shape[0], 6], dtype=np.float32)
                    motion = np.concatenate([rot_append, mo_temp[:, 3:22*6+3], padding], axis=-1)
                elif args.DATA_SETTING.nfeats == 135:
                    mo_temp = np.load(os.path.join(self.fd_motion_dir, name))
                    motion = mo_temp[:, :135]
                elif args.DATA_SETTING.nfeats == 263:
                    mo_temp = np.load(os.path.join(self.fd_motion_dir, name))
                    assert mo_temp.shape[-1] == 263
                    motion = mo_temp
                else:
                    raise("error of args.DATA_SETTING.nfeats")
                motion = torch.from_numpy(motion).float() # 转为 tensor

                child_list = [x for x in key_idx[name[:-4]] if x < motion.shape[0]]
                key_num += len(child_list)

                motion_all.append((motion, name,  child_list))
            print("key_num is :", key_num)

                
            # self.FineDance_motion = np.concatenate(motion_all, axis=0).astype(np.float32) # 为了能够和 music 对齐, 保留 motion 的分段性, 使用时随机截取部分
            self.FineDance_motion = motion_all


            print(f'FineDance has {len(self.FineDance_motion)} samples..')
        
        elif "all" in args.dataset or "aistpp" in args.dataset:
            # HumanML3D(AMASS) 
            if istrain:
                split_file = args.o_h3d_dir +  'train.txt'
            else:
                split_file = args.o_h3d_dir +  'test.txt'

                # 得到所有 train 文件的路径
            self.amass_motion = []
            self.lengths = []
            id_list = []
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

                # 读取对应的处理后的 HumanML3D(amass+huamnact12) 数据
            for name in tqdm(id_list,desc='HumanML3D'):
                try:
                    motion =  np.load(args.p_h3d_dir + name + '.npy', allow_pickle=True)[:,:args.DATA_SETTING.nfeats]
                    motion = torch.from_numpy(motion).float() # 转为 tensor
                    self.amass_motion.append(motion)
                except:
                    # Some motion may not exist in KIT dataset
                    pass
                # self.amass_motion = torch.cat(self.amass_motion,dim=0) # 为了能够和 text 对齐, 保留 motion 的分段性, 使用时随机截取部分
            print(f'HumanML3D has {len(self.aistpp_motion)} samples..')
        
        # 为使不同长度的 motion 数据能被处理, 参考 T2M-GPT: dataset/dataset_VQ.py
            # 合并 FineDance, AIST++ 和 AMASS 三个数据集的 motion 数据
        self.window_size = args.window_size
        self.data = [self.aistpp_motion,self.FineDance_motion,self.amass_motion]
        self.lengths = []

        # self.data = torch.cat(self.data,dim = 0)

        self.motion_data = []
        for dataset in tqdm(self.data, desc='processing mixed data:'):
            for motion, name, key_idx in dataset:
                try:
                    if motion.shape[0] < self.window_size: # 过滤掉不符合长度要求的数据
                        continue
                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.motion_data.append((motion, name, key_idx))
                except:
                    # Some motion may not exist in KIT dataset
                    pass
        print("done")

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, index): # 参考 T2M-GPT/dataset/dataset_VQ.py
        motion, name, key_idx = self.motion_data[index]
        # idx_list = np.random.randint(0, motion.shape[0] - self.window_size, 16) #.numpy() #.tolist()      # 从所有帧训练      
        # child_list = [x for x in key_idx if x < motion.shape[0]]
        assert int(key_idx[-1]) < motion.shape[0]

        # 从关键帧训练 随机选择16个关键帧
        # idx_list = random.sample(key_idx, 16)                
        # Rmotion = motion[idx_list].squeeze()

        idx = random.sample(key_idx, 1)[0]
        if idx >= len(motion)-5:
            idx = len(motion)-5
        if idx <= 4:
            idx = 4
        idx_list = []
        for i in range(idx-4,idx+4):
            idx_list.append(i)
        # if int(index)<10:
        #     print("index is {}, sampled idx is {}".format(index, idx))
        Rmotion = motion[idx_list].squeeze()
     
        return Rmotion # torch.Size([128, 64, 135]) | 统一输出格式为 3+22*6 的 SMPL 格式
    
    def get_train_test_list(self):
        all_list = []
        train_list = []
        for i in range(1,212):
            all_list.append(str(i).zfill(3))

        test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
        ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]

        for one in all_list:
            if one not in test_list:
                train_list.append(one)
        temp = []
        for one in train_list:
            temp.append("M" + one)
        train_list = train_list + temp

        temp = []
        for one in ignor_list:
            temp.append("M" + one)
        ignor_list = ignor_list + temp

        temp = []
        for one in test_list:
            temp.append("M" + one)
        test_list = test_list + temp

        return ignor_list, train_list, test_list


def DATALoader(args, isTrain, batch_size = 128, num_workers = 8):
    # 定义数据集（施工中）
    trainSet = motion_Smpl(args, isTrain)
    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)
    
    return train_loader

# 把 train_loader 变成迭代器, 加 while true 变成能一直迭代
def cycle(iterable):
    while True:
        # print("1")
        for x in iterable:
            yield x
