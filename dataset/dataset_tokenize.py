'''和dataset_vq区别: 返回文件名, 用于保存新的token.pth文件'''
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


class motion_token_Smpl(data.Dataset):
    def __init__(self, args, istrain): # 通过 istrain 来选择使用的数据集
        data_dict = {}
        name_list = []
        min_motion_len = 40

        # # AIST++ motion
        #     # 得到 AIST++ dance data from aistpp_api/demos/extract_motion_feats.py
        # anno_dir = args.aistpp_dir

        #     # 得到所有的 video_name
        # aist_dataset = AISTDataset(anno_dir)
        # # aistpp_motion_names = get_filenames_without_extension(aist_dataset.motion_dir)
        # if istrain:
        #     split_file = anno_dir + 'splits/crossmodal_train.txt'
        #     print('..training dataset loading')
        # else:
        #     split_file = anno_dir + 'splits/crossmodal_test.txt'
        #     print('..test dataset loading')
        
        # aistpp_motion_names = []
        # with cs.open(split_file, 'r') as f:
        #     for line in f.readlines():
        #         aistpp_motion_names.append(line.strip())
        
        #     # 读取 AIST++ 数据集
        # self.aistpp_motion = []
        # aistpp_motion_names = aistpp_motion_names[:2] # debug

        # for seq_name in tqdm(aistpp_motion_names, desc='AIST++'):
        #     dance_data = np.load(args.aistpp_p_dir + seq_name + '.npy')

        #     if (len(dance_data)) < min_motion_len: # 不能太短
        #         continue

        #     dance_data = torch.from_numpy(dance_data).float() # 转为 tensor
        #     self.aistpp_motion.append(dance_data) # 利用 list 能够处理变长数据 | 为了能够和 music 对齐, 保留 motion 的分段性, 使用时随机截取部分

        #     data_dict[seq_name] = {'motion': dance_data,
        #                             'length': len(dance_data),
        #                             'name': seq_name
        #                             }
        #     name_list.append(seq_name)
            
        # print(f'AIST++ has {len(self.aistpp_motion)} samples..')

        # FineDance
        self.fd_motion_dir = args.fd_motion_dir
        self.istrain = istrain


        ignor_list, train_list, test_list = self.get_train_test_list()
        if self.istrain:
            fd_name_list= train_list
        else:
            fd_name_list = test_list

        fd_name_list = fd_name_list[:2] # debug

        motion_all = []
        for name in tqdm(fd_name_list, desc='FineDance'):
            name_npy = name + ".npy"
            if name_npy[:-4] in ignor_list:
                continue
     
            motion = np.load(os.path.join(self.fd_motion_dir, name_npy))[:,:args.DATA_SETTING.nfeats] # 加载预处理的数据 | FineDance 取 3+22*6 + 别的部分 263
            motion = torch.from_numpy(motion).float() # 转为 tensor
            
            if (len(motion)) < min_motion_len: # 不能太短
                continue
            
            motion_all.append(motion)

            data_dict[name] = {'motion': motion,
                                'length': len(motion),
                                'name': name
                                }
            name_list.append(name)

        # self.FineDance_motion = np.concatenate(motion_all, axis=0).astype(np.float32) # 为了能够和 music 对齐, 保留 motion 的分段性, 使用时随机截取部分
        self.FineDance_motion = motion_all
        self.fd_names = fd_name_list

        print(f'FineDance has {len(self.FineDance_motion)} samples..')
        

        # HumanML3D(AMASS) 
        if istrain:
            split_file = args.o_h3d_dir +  'train.txt'
        else:
            split_file = args.o_h3d_dir +  'test.txt'

            # 得到所有 train 文件的路径
        self.amass_motion = []
        self.lengths = []
        h3d_id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                h3d_id_list.append(line.strip())

        h3d_id_list = h3d_id_list[:2] # debug

            # 读取对应的处理后的 HumanML3D(amass+huamnact12) 数据
        for name in tqdm(h3d_id_list,desc='HumanML3D'):
            try:
                motion =  np.load(args.p_h3d_dir + name + '.npy', allow_pickle=True)[:,:args.DATA_SETTING.nfeats]
                motion = torch.from_numpy(motion).float() # 转为 tensor

                if (len(motion)) < min_motion_len: # 不能太短
                    continue

                self.amass_motion.append(motion)

                if (len(motion)) < min_motion_len: # 不能太短
                        continue
                
                data_dict[name] = {'motion': motion,
                                'length': len(motion),
                                'name': name
                                }
                name_list.append(name)
            except:
                # Some motion may not exist in KIT dataset
                pass

            # self.amass_motion = torch.cat(self.amass_motion,dim=0) # 为了能够和 text 对齐, 保留 motion 的分段性, 使用时随机截取部分
        print(f'HumanML3D has {len(self.amass_motion)} samples..')


        self.unit_length = 4 # from T2M-GPT
        self.data_dict = data_dict
        self.name_list = name_list
        


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index): # 参考 T2M-GPT/dataset/dataset_VQ.py
        name = self.name_list[index]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]
        motion = motion[0:m_length]

        return motion, name
    
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


def DATALoader(args, isTrain, batch_size = 1, num_workers = 8):
    # 定义数据集（施工中）
    trainSet = motion_token_Smpl(args, isTrain)
    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last = True)
    
    return train_loader


