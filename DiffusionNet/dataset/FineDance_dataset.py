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
from utils.smplfk import SMPLX_Skeleton, do_smplxfk

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
        # self.motion_dir = '/data2/lrh/dataset/fine_dance/origin/motion_feature319'
        # self.motion_dir = args.fd_motion_dir
        # self.music_dir = '/data2/lrh/dataset/fine_dance/origin/music_feature35_edge'
        # self.music2genre = music2genre("/data2/lrh/dataset/fine_dance/origin/label_json")
        # ipadd = getip()
        ipadd = "10.103.11.45"
        if ipadd == "10.103.11.45":
            self.motion_dir = '/data2/lrh/dataset/fine_dance/gound/mofea319/'
            self.music_dir = '/data2/lrh/dataset/fine_dance/gound/musicfea_edge/'
            # self.motion_dir = '/data2/lrh/dataset/fine_dance/origin/motion_feature319'
            # self.music_dir = '/data2/lrh/dataset/fine_dance/origin/music_feature35_edge'
            self.music2genre = music2genre("/data2/lrh/dataset/fine_dance/origin/label_json")
        elif ipadd == "10.103.11.40":
            self.motion_dir = "/data/lrh/datasets/fine_dance/origin/motion_feature319mirror/"
            self.music_dir = '/data/lrh/datasets/fine_dance/origin/music_feature35_edge/'
            self.music2genre = music2genre("/data/lrh/datasets/fine_dance/origin/label_json")
        else:
            print("ipadd", ipadd)
            raise("error of machine ip")
        
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
            if self.music2genre[name[:-4]] in ['Miao','Dai']:
                continue
            
            motion = np.load(os.path.join(self.motion_dir, name))
            # FK
            smplx_model = SMPLX_Skeleton()
            motion = torch.from_numpy(motion)[:, :139]
            motion = set_on_ground_139(motion, smplx_model, -1.2)
            motion = motion.detach().cpu().numpy()

            music = np.load(os.path.join(self.music_dir, name))

            min_all_len = min(motion.shape[0], music.shape[0])
            motion = motion[:min_all_len]
            if motion.shape[-1] == 168:
                motion = np.concatenate([motion[:,:69], motion[:,78:]], axis=1)     # 22,  25
            elif motion.shape[-1] == 319:
                if args.DATA_SETTING.nfeats == 139:
                    motion = motion[:, :139]
            elif motion.shape[-1] == 315 or motion.shape[-1] == 263  or motion.shape[-1] == 139:
                pass
                # motion = np.concatenate([motion[:,:135], motion[:,153:]], axis=1)    #
            else:
                print("motion.shape", motion.shape)
                raise("input motion shape error! not 168 or 319!")
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

        if self.args.keymotion_dir:
            print("keymotion_dir")
            print("keymotion_dir")
            print("keymotion_dir")
            temp_list = copy.deepcopy(self.motion_index)
            key_len = []
            self.keylength = {}
            for i in range(len(temp_list)):
                name_ = self.name[i]
                try:
                    data = np.load(os.path.join(self.args.keymotion_dir, name_+'.npy'))
                    key_len.append(int(data.shape[0]))
                    self.keylength[name_] = int(data.shape[0])
                except:
                    print(os.path.join(self.args.keymotion_dir, name_+'.npy' + " not exits"))
                    self.motion_index.pop(i)
                    self.music_index.pop(i)
                    self.name.pop(i)

            self.len = len(self.motion_index)
            print(f'FineDance has {self.len} samples..')
            print("max of key length is: ", max(key_len))
            print("min of key length is: ", min(key_len))

    def get_keymotion(self, net, trans_encoder, left, right):
        args = self.args
        genre_dict = self.music2genre
        print(genre_dict)
        device = f"cuda:{args.gpu}"
        key_savepath = os.path.join(os.path.join(args.out_dir, f'{args.exp_name}'), "keymotion")
        if not os.path.exists(key_savepath):
            os.makedirs(key_savepath)
        # musicclip_savepath = os.path.join(os.path.join(args.out_dir, f'{args.exp_name}'), "musicclip")
        # if not os.path.exists(musicclip_savepath):
        #     os.makedirs(musicclip_savepath)

        key_nums = []

        for i in tqdm(range(self.len)):
            music_index = self.music_index[i]
            music_clip = self.music[music_index:music_index+self.seq_len]
            name = self.name[i]
            songname = name.split("_")[0]

            music_savedir = "/data2/lrh/dataset/fine_dance/div_by_time/music_25_920"
            os.makedirs(music_savedir, exist_ok=True)
            np.save(os.path.join(music_savedir, name + ".npy"), music_clip)
            # continue


            if not (int(songname)>left and int(songname)<=right):
                continue

            if os.path.exists(os.path.join(args.key_savepath, name + ".npy") ):
                print("continue  because exists")
                continue

            genre = genre_dict[name.split("_")[0]]
            # print("genre1 ", genre)
            genre = np.array(Genres_fd[genre])
            genre = torch.from_numpy(genre).unsqueeze(0)
            # print("genre2 ", genre)

            noise = torch.randn(music_clip.shape[0], 256).to(device)
            music_clip = torch.from_numpy(music_clip).unsqueeze(0).to(device)
            print("music_clip.shape", music_clip.shape)
            print("name", name)

            # debug!!!
            cls_pred = trans_encoder.sample(noise = noise, feature = music_clip, genre = genre, masked_token_seq = None)
            
            try:
                cls_pred = cls_pred.squeeze(0)
            except:
                print("name", name)
                print("cls_pred.shape", cls_pred.shape)
                raise("error of ", name)
                # with open(os.path.join(args.out_dir, "ignore.txt"), "a") as file:
                #     file.write(name+"\n")
                # print("continue because None")
                # continue

            reshape_dim = int(self.args.clip_frames / self.args.vq_scale)
            if cls_pred.shape[0] % reshape_dim!=0:
                print("jishu cls_pred.shape error ",cls_pred.shape)
                cls_pred = cls_pred[:cls_pred.shape[0]-1]
            if cls_pred.shape[0] % reshape_dim != 0:
                print("cls_pred.shape error ", cls_pred.shape)
                print(cls_pred.shape)
                raise("error of cls_pred shape")
            cls_pred = cls_pred.reshape(-1, reshape_dim)

            
            print("cls_pred.shape", cls_pred.shape)
            key_nums.append(cls_pred.shape[0])
            keymotion = net.vqvae.my_forward_decoder(cls_pred).detach().cpu().numpy()
            print("before keymotion.shape", keymotion.shape)
            keymotion = keymotion.reshape(-1, keymotion.shape[-1])
            print("keymotion.shape", keymotion.shape)
            print("keymotion save at ", os.path.join(args.key_savepath, name + ".npy"))
            np.save(os.path.join(args.key_savepath, name + ".npy"), keymotion)

            print("cls_pred.shape", cls_pred.shape)
            # sys.exit(0)
        print("max key num", max(key_nums))
            
    def get_full_gptmotion(self, net, trans_encoder, left, right):
        args = self.args
        genre_dict = self.music2genre
        print(genre_dict)
        device = f"cuda:{args.gpu}"
        full_savepath = os.path.join(os.path.join(args.out_dir, f'{args.exp_name}'), "fullmotion")
        if not os.path.exists(full_savepath):
            os.makedirs(full_savepath)
        # musicclip_savepath = os.path.join(os.path.join(args.out_dir, f'{args.exp_name}'), "musicclip")
        # if not os.path.exists(musicclip_savepath):
        #     os.makedirs(musicclip_savepath)

        key_nums = []

        for i in tqdm(range(self.len)):
            music_index = self.music_index[i]
            music_clip = self.music[music_index:music_index+self.seq_len]
            name = self.name[i]
            songname = name.split("_")[0]

            music_savedir = "/data2/lrh/dataset/fine_dance/div_by_time/music_25_920"
            os.makedirs(music_savedir, exist_ok=True)
            np.save(os.path.join(music_savedir, name + ".npy"), music_clip)
            # continue


            if not (int(songname)>left and int(songname)<=right):
                continue

            if os.path.exists(os.path.join(args.key_savepath, name + ".npy") ):
                print("continue  because exists")
                continue

            genre = genre_dict[name.split("_")[0]]
            # print("genre1 ", genre)
            genre = np.array(Genres_fd[genre])
            genre = torch.from_numpy(genre).unsqueeze(0)
            # print("genre2 ", genre)

            noise = torch.randn(music_clip.shape[0], 256).to(device)
            music_clip = torch.from_numpy(music_clip).unsqueeze(0).to(device)
            print("music_clip.shape", music_clip.shape)
            print("name", name)

            # debug!!!
            cls_pred = trans_encoder.sample(noise = noise, feature = music_clip, genre = genre, masked_token_seq = None)
            
            try:
                cls_pred = cls_pred.squeeze(0)
            except:
                print("name", name)
                print("cls_pred.shape", cls_pred.shape)
                raise("error of ", name)
                # with open(os.path.join(args.out_dir, "ignore.txt"), "a") as file:
                #     file.write(name+"\n")
                # print("continue because None")
                # continue

            reshape_dim = int(self.args.clip_frames / self.args.vq_scale)
            if cls_pred.shape[0] % reshape_dim!=0:
                print("jishu cls_pred.shape error ",cls_pred.shape)
                cls_pred = cls_pred[:cls_pred.shape[0]-1]
            if cls_pred.shape[0] % reshape_dim != 0:
                print("cls_pred.shape error ", cls_pred.shape)
                print(cls_pred.shape)
                raise("error of cls_pred shape")
            cls_pred = cls_pred.reshape(-1, reshape_dim)

            
            print("cls_pred.shape", cls_pred.shape)
            key_nums.append(cls_pred.shape[0])
            keymotion = net.vqvae.my_forward_decoder(cls_pred).detach().cpu().numpy()
            print("before keymotion.shape", keymotion.shape)
            keymotion = keymotion.reshape(-1, keymotion.shape[-1])
            print("keymotion.shape", keymotion.shape)
            print("keymotion save at ", os.path.join(args.key_savepath, name + ".npy"))
            np.save(os.path.join(args.key_savepath, name + ".npy"), keymotion)

            print("cls_pred.shape", cls_pred.shape)
            sys.exit(0)
        print("max key num", max(key_nums))


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        motion_index = self.motion_index[index]
        music_index = self.music_index[index]
        filename = self.name[index]

        motion = self.motion[motion_index:motion_index+self.seq_len]
        if motion.shape[-1] != 263:
            if motion.shape[-1] == 319 or motion.shape[-1] == 139:
                # motion[:, 4:7]  = motion[:, 4:7] - motion[:1, 4:7]           # The first 4 dimension are foot contact
                motion[:, 4]  = motion[:, 4] - motion[:1, 4] 
                motion[:, 6]  = motion[:, 6] - motion[:1, 6] 
                
                # with torch.no_grad():
                #     smplx_modle = SMPLX_Skeleton()
                #     positions = do_smplxfk(torch.from_numpy(motion[:,:139]), smplx_modle)
                #     ltoe = positions[0:150, 10,  1]
                # print("read mean of y", ltoe.mean())
                # print("read min of y", min(ltoe) )
                # print("read max of y", max(ltoe) )
                
            else:
                motion[:, 0] = motion[:, 0] - motion[:1, 0]
                motion[:, 2] = motion[:, 2] - motion[:1, 2]
        music = self.music[music_index:music_index+self.seq_len]
        
        if self.args.keymotion_dir:
            try:
                key_motion = np.load(os.path.join(self.args.keymotion_dir, filename+'.npy'))
    
                scale = self.args.window_size/key_motion.shape[0]
                key_motion = scipy.ndimage.zoom(key_motion, zoom=[scale, 1], order=0)
                # np.interp()
                # interpolated_data = cv2.resize(key_motion, None, fx=scale_factors[1], fy=scale_factors[0], interpolation=cv2.INTER_NEAREST)
                # print(key_motion.shape)
            except:
                print("file not exits: " + os.path.join(self.args.keymotion_dir, filename+'.npy'))
                raise
            return motion, music, filename, key_motion
        else:
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