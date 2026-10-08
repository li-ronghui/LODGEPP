import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
import cv2
import json
import copy
import scipy
import models.c2d_trans_frame as trans
import models.vqvae_frame as vqvae
import sys

sys.path.append("/data2/lrh/project/dance/long")
# from render import ax_to_6v, ax_from_6v
# from dataset.dataset_gpt_m2d_frame import Genres_fd
from dataset.preprocess.foot_process.set_on_groud import set_on_ground, set_on_ground_139
from data.utils.smplfk import SMPLX_Skeleton, do_smplxfk

Debug = False

# from utils.parser_util import args

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



def music2genre(label_dir):
    music_genre = {}
    for file in os.listdir(label_dir):
        name = file.split(".")[0]
        jsonfile = os.path.join(label_dir, file)
        with open(jsonfile,"r") as f:
            genredict = json.load(f)
        genre = genredict['style2']

        music_genre[name] = genre

    return music_genre

def flip_pose(pose):
    #Flip pose.The flipping is based on SMPLX parameters.
    pose = pose[:,SMPLX_POSE_FLIP_PERM]
    # we also negate the second and the third dimension of the axis-angle
    pose[:,1::3] = -pose[:,1::3]
    pose[:,2::3] = -pose[:,2::3]
    return pose

class FineDance_Smpl(data.Dataset):
    def __init__(self, args, istrain):
        ipadd = "10.103.11.40"
        if ipadd == "10.103.11.45":
            self.motion_dir = '/data2/lrh/dataset/fine_dance/gound/mofea319/'
            self.music_dir = '/data2/lrh/dataset/fine_dance/gound/musicfea_edge/'
            # self.motion_dir = '/data2/lrh/dataset/fine_dance/origin/motion_feature319'
            # self.music_dir = '/data2/lrh/dataset/fine_dance/origin/music_feature35_edge'
            self.music2genre = music2genre("/data2/lrh/dataset/fine_dance/origin/label_json")
        elif ipadd == "10.103.11.40":
            # self.motion_dir = "/data/lrh/datasets/fine_dance/gound/mofea319/"
            self.motion_dir = "/data/lrh/datasets/fine_dance/gound/smplx_mofea266/new_joint_vecs/"
            self.music_dir = '/data/lrh/datasets/fine_dance/gound/musicfea_edge/'
            self.music2genre = music2genre("/data/lrh/datasets/fine_dance/origin/label_json")
        else:
            print("ipadd", ipadd)
            raise("error of machine ip")
        if args.normdir is not None:
            NormMean = np.load(os.path.join(args.normdir, 'Mean.npy'))
            NormStd = np.load(os.path.join(args.normdir, 'Std.npy'))
            NormMean = torch.from_numpy(NormMean).to(f"cuda:{args.gpu}")
            NormStd = torch.from_numpy(NormStd).to(f"cuda:{args.gpu}")
        
        self.istrain = istrain
        self.args = args
        self.seq_len = args.window_size * args.clip_frames
        slide = self.seq_len // args.windows

        self.motion_index = []
        self.music_index = []
        self.name = []
        motion_all = []
        music_all = []
        
        ignor_list, train_list, test_list = self.get_train_test_list()
        if self.istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list
            

        total_length = 0            # 将数据集中的所有motion用同一个index索引

        if Debug:
            self.datalist = ['001']
        debug_num = 0
        for name in tqdm(self.datalist):
            save_name = name
            name = name + ".npy"
            # debug_num += 1
            # if debug_num>6:
            #     break


            if name[:-4] in ignor_list:
                continue
            if int(name[:-4]) >= 72 and int(name[:-4]) <= 83:
                continue
            # if self.music2genre[name[:-4]] in ['Miao','Dai']:
            #     continue
            
            motion = np.load(os.path.join(self.motion_dir, name))
            motion = torch.from_numpy(motion).to(f"cuda:{args.gpu}")
            motion = ((motion - NormMean) / NormStd ).detach().cpu().numpy()
            # FK
            # smplx_model = SMPLX_Skeleton()
            # motion = torch.from_numpy(motion)
            # motion = set_on_ground_139(motion, smplx_model, -1.2)
            # motion = motion.detach().cpu().numpy()
                
            music = np.load(os.path.join(self.music_dir, name))
            min_all_len = min(motion.shape[0], music.shape[0])
            motion = motion[:min_all_len]
    
            if args.DATA_SETTING.nfeats == 139:
                motion = motion[:, :139]
            else:
                print("motion.shape", motion.shape)
                # raise("input motion shape error! not 168 or 319!")
            music = music[:min_all_len]         # motion = motion[:min_all_len]
            nums = (min_all_len-self.seq_len) // slide + 1          # 舍弃了最后一段不满seq_len的motion

            if self.istrain:
                clip_index = []
                for i in range(nums):
                    clip_index.append(i)
                index = np.array(clip_index) * slide + total_length     # clip_index为local index
                index_ = np.array(clip_index) * slide
            else:
                index = np.arange(nums) * slide + total_length
                index_ = np.arange(nums) * slide 

            motion_all.append(motion)
            music_all.append(music)
            
            if args.mix:
                motion_index = []
                music_index = []
                num = (len(index) - 1) // 8 + 1
                for i in range(num):
                    motion_index_tmp, music_index_tmp = np.meshgrid(index[i*8:(i+1)*8], index[i*8:(i+1)*8])         # 这里i有问题？似乎没有
                    motion_index += motion_index_tmp.reshape((-1)).tolist()
                    music_index += music_index_tmp.reshape((-1)).tolist()
                    index_tmp = np.meshgrid(index_[i*8:(i+1)*8])
                    index_ += index_tmp.reshape((-1)).tolist()
            else:
                motion_index = index.tolist()
                music_index = index.tolist()
                index_ = index_.tolist()
            index_ = [save_name + "_" + str(element).zfill(5) for element in index_]
            
            self.motion_index += motion_index
            self.music_index += music_index
            total_length += min_all_len
            self.name += index_
            
        self.motion = np.concatenate(motion_all, axis=0).astype(np.float32)
        self.music = np.concatenate(music_all, axis=0).astype(np.float32)

        self.len = len(self.motion_index)
        print(f'FineDance has {self.len} samples..')


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        motion_index = self.motion_index[index]
        music_index = self.music_index[index]
        filename = self.name[index]

        motion = self.motion[motion_index:motion_index+self.seq_len]
        if motion.shape[-1] == 139:
            # motion[:, 4:7]  = motion[:, 4:7] - motion[:1, 4:7]           # The first 4 dimension are foot contact
            motion[:, 4]  = motion[:, 4] - motion[:1, 4] 
            motion[:, 6]  = motion[:, 6] - motion[:1, 6] 
        elif motion.shape[-1] == 266 or motion.shape[-1] == 338:
            motion[:, [0,2]]  = motion[:, [0,2]] - motion[:1, [0,2]] 

        music = self.music[music_index:music_index+self.seq_len]
        
        
        return motion, music, filename
        
    
    def get_train_test_list(self):
        all_list = []
        train_list = []
        for i in range(1,212):
            all_list.append(str(i).zfill(3))
        # print(all_list)
        # print('all_list is :', len(all_list))

        test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
        ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
        # flist = ['106','008','011', '012','017','018','019','021','023','024','025','026','027','028','030','031','034','035','038','039','040','043','044','045','048','105','104','108','109','110','111','112','115','148','155','170'] + ['190'] + ['208','200','185','180','173','152'] + ['154'] + ['046','113', '153']
        tradition_list = ['005', '007', '008', '015', '017', '018', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '032', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '126', '127', '132', '133', '134',  '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '151', '152', '153', '154', '155', '170']
        morden_list = []
        for one in all_list:
            if one not in tradition_list:
                morden_list.append(one)

        ignor_list = ignor_list
        for one in all_list:
            if one not in test_list:
                train_list.append(one)
        

        if self.args.partial == 'full':
            return ignor_list, train_list, test_list
        elif self.args.partial == 'morden':
            for one in train_list:
                if one in tradition_list:
                    train_list.remove(one)
            for one in test_list:
                if one in tradition_list:
                    test_list.remove(one)
            return ignor_list, train_list, test_list
        elif self.args.partial == 'tradition':
            for one in train_list:
                if one in morden_list:
                    train_list.remove(one)
            for one in test_list:
                if one in morden_list:
                    test_list.remove(one)
            return ignor_list, train_list, test_list