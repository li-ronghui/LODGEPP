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
from dataset.aistplusplus_api.aist_plusplus.loader import AISTDataset


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

# Genres = {
#     'gBR': 0,
#     'gPO': 1,
#     'gLO': 2,
#     'gMH': 3,
#     'gLH': 4,
#     'gHO': 5,
#     'gWA': 6,
#     'gKR': 7,
#     'gJS': 8,
#     'gJB': 9,
# }

Genres = {'gBR': 0,             # Breaking
          'Breaking': 0,
          'gPO': 1,             # 
          'Popping': 1,
          'gLO': 2,
          'Locking': 2,
          'Hiphop':3,
          'gMH': 3,
          'gLH': 3,
          'Urban':4,
          'gHO': 5,
          'gWA': 6,
          'gKR': 7,
          'gJS': 8,
          'gJB': 8,
          'Jazz':8,
          'jazz':8,

          'Tai':9,
          'Uighur':10,
          'Hmong':11,

          'HanTang':12,
          'ShenYun':13,
          'Kun':14,
          'DunHuang':15,

          'Rumba':16,
          'Samba':17,
          'Waltz':18,
          'Tango':19,
          'Cha-cha':20,
          'Cowboy':21,
          
          'Korean':22,
          'Choreography':23,
          'Chinese':24,
          'Dai':25,
          'Wei':26,
          'Miao':27,

}

Genres_aist = {
    'gBR': 0,
    'gPO': 1,
    'gLO': 2,
    'gMH': 3,
    'gLH': 4,
    'gHO': 5,
    'gWA': 6,
    'gKR': 7,
    'gJS': 8,
    'gJB': 9,
}


Genres_fd = {            # Breaking
          'Breaking': 0,
          'Popping': 1,
          'Locking': 2,
          'Hiphop':3,
          'Urban':4,
          'Jazz':5,
          'jazz':5,

          'Tai':6,
          'Uighur':7,
          'Hmong':8,
          'Dai':6,
          'Wei':7,
          'Miao':8,

          'HanTang':9,
          'ShenYun':10,
          'Kun':11,
          'DunHuang':12,

          'Korean':13,
          'Choreography':14,
          'Chinese':15,
}



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

class m2d_Smpl(data.Dataset):
    def __init__(self, args, istrain): # 通过 istrain 来选择使用的数据集
        
        data_dict = {}
        name_list = []
        self.args = args
        self.window_size = args.DATA_SETTING.full_seq_len
        self.block_size = self.window_size +1
        self.clip_frames = 4

        dataname = args.dataname[0]
        self.tkdir =  args.tkdir     #'/data/lrh/datasets/fine_dance/origin/
        self.music_dir = eval(f"args.DATASET.{dataname.upper()}.MUSIC") 

        if 'FINEDANCE' in dataname:
            self.music2genre = music2genre(eval(f"args.DATASET.{dataname.upper()}.LABEL"))

        # max_motion_len = 0

        # temp param
        # self.window_size = 120 # 每次截取的 music 和 motion 长度, 太长则满足长度的数据少
        
        # # AIST++ motion
        #     # 得到 AIST++ dance data from aistpp_api/demos/extract_motion_feats.py
        # anno_dir = args.aistpp_dir

        #     # 得到所有的 video_name
        # # aist_dataset = AISTDataset(anno_dir)
        # # motion_names = get_filenames_without_extension(aist_dataset.motion_dir)
        # if istrain:
        #     split_file = anno_dir + 'splits/crossmodal_train.txt'
        #     print('..training dataset loading')
        # else:
        #     split_file = anno_dir + 'splits/crossmodal_test.txt'
        #     print('..test dataset loading')
        # motion_names = []

        # with cs.open(split_file, 'r') as f:
        #     for line in f.readlines():
        #         motion_names.append(line.strip())
        
        # motion_names = motion_names[:2] # debug

        #     # 读取 AIST++ token 和 music 文件
        # aistpp_motion = []
        # aistpp_music = []
        # for seq_name in tqdm(motion_names, desc='AIST++'):
        #     # read motion_token
        #     dance_data = np.load(args.tkdir + f'f_{args.DATA_SETTING.nfeats}/' + seq_name + '.npy')
        #     dance_data = torch.from_numpy(dance_data).float() # 转为 tensor
            
            
        #     if dance_data.shape[0]>max_motion_len: # 寻找最大 len
        #         max_motion_len = dance_data.shape[0]
        #     if dance_data.shape[0]<self.window_size: # 过滤掉太短的
        #         continue

        #     # Read music
        #     audio_name = seq_name.split('.')[0].split('_')[4] # 得到对应的 music 的名称
        #     audio_path = os.path.join(args.aistpp_p_music_dir, audio_name + '.npy')
        #     music = np.load(audio_path)
        #     music = torch.from_numpy(music).float() # 转为 tensor

        #     # genre
        #     genre_label = seq_name.split('.')[0].split('_')[0]
        #     genre = torch.tensor(Genres[genre_label])

        #     aistpp_motion.append(dance_data) # 利用 list 能够处理变长数据 | 为了能够和 music 对齐, 保留 motion 的分段性, 使用时随机截取部分
        #     aistpp_music.append(music)

        #     data_dict[seq_name] = {'motion_token': dance_data,
        #                             'music':music,
        #                             'genre':genre}
            
        #     name_list.append(seq_name)
    
        # self.aistpp_motion = aistpp_motion
        # self.aistpp_music = aistpp_music

        # print(f'AIST++ has {len(self.aistpp_motion)} samples..')



        # FineDance
        self.istrain = istrain
        self.motion_index = []
        self.name = []
        motion_all = []
        music_all = []

        ignor_list, train_list, test_list = self.get_train_test_list(dataset = args.dataname[0])
        if self.istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list

        # self.datalist = self.datalist[:2] # debug
        unexist_list = []
        for name in tqdm(self.datalist, desc=args.dataname[0]):
            if name[0] == 'M':
                continue

            name = name[1:] if name[0]=='M' else name
            name_npy = name + ".npy"
            if name_npy[:-4] in ignor_list:
                continue
            jsonname = name[1:] if name[0]=='M' else name
            # 加载 motion
            if not os.path.exists(os.path.join(args.tkdir, name_npy)):
                print('unexist', os.path.join(args.tkdir, name_npy))
                unexist_list.append(name_npy)
                continue
            motion = np.load(os.path.join(args.tkdir, name_npy)) # 加载预处理的数据 | FineDance 取 3+22*6 + 别的部分 263
            motion = torch.from_numpy(motion).float().reshape(-1) # 转为 tensor

            if motion.shape[0]<self.window_size: # 过滤掉太短的
                continue
            
            input_name_npy = name_npy
            if name_npy[0] =='M':
                input_name_npy = name_npy[1:]

            # 加载 music
            music = np.load(os.path.join(self.music_dir, input_name_npy))
            music = torch.from_numpy(music).float() # 转为 tensor

            # # 加载 genre
            # with open(args.fd_g_dir + name +'.json') as file:
            #     lable_json = json.load(file)
            # # 获取 style2 genre 的值
            # genre_label = lable_json['style2']
            # # print(name,':',genre_label) # debug 查看 genre
            # genre = torch.tensor(Genres_fd[genre_label])

            if 'FINEDANCE' in dataname:
                genre_label = self.music2genre[name.split(".")[0]]
                # print("genre1 ", genre)
                genre = torch.tensor(Genres_fd[genre_label])
            elif 'AISTPP' in dataname:
                genre_label = name.split('_')[0]
                genre = torch.tensor(Genres_aist[genre_label])


            min_all_len = min(motion.shape[0]*self.clip_frames, music.shape[0])

            # if motion.shape[0]>max_motion_len: # 寻找最大 len
            #     max_motion_len = motion.shape[0]

            motion = motion[: int(min_all_len//self.clip_frames)]
            music = music[:min_all_len]

            data_dict[name] = {'motion_token': motion,
                                'music':music,
                                'genre':genre
                                }
            name_list.append(name)

            motion_all.append(motion)
            music_all.append(music)

        print(f'There are {len(unexist_list)} samples unexist')
        print(f'Dataset has {len(motion_all)} samples..')
        
        self.data_dict = data_dict
        self.name_list = name_list
        self.feature_dim = args.DATA_SETTING.nfeats

        # 参考 T2M-GPT
        # self.max_motion_length = max_motion_len
        self.mot_end_idx = args.nb_code
        self.mot_pad_idx = args.nb_code + 1


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index): # 参考 T2M-GPT/dataset/dataset_VQ.py
        data = self.data_dict[self.name_list[index]]
        m_tokens, music, genre = data['motion_token'], data['music'], data['genre']
        
        m_tokens = m_tokens.reshape(-1) # [1,*] -> [*] # 当只有一个维度, 使用这个就不能看到当前维度的实际值了
        seq_len = m_tokens.shape[0]

        # 仅使用 fd dataset, 受到 vq-vae影响 music 需要上采样 4 倍  | 截取 music 和 motion 中随机的一段 
        idx = random.randint(0, seq_len - self.window_size)
        
        m_tokens = m_tokens[idx:idx+self.window_size]
        music_range = min(music.shape[0],idx* self.clip_frames + self.window_size* self.clip_frames)
        music = music[idx*self.clip_frames:music_range] # 音乐的 feature 应该是 m_tokens 的四倍
        
        if music.shape[0] < self.window_size*self.clip_frames: # 补齐 music feature: torch.Size([120, 35])
            zero_emb = torch.zeros([self.window_size * self.clip_frames -music.shape[0], music.shape[1]])
            music = torch.cat([music,zero_emb],dim=0) # 使用 0 补齐剩下的维度
        


        # # 选 m_token | 一个数
        # coin = np.random.choice([False, False, True])
        # # print(len(m_tokens))
        # if coin:
        #     # 随机选择序列, 随机剪掉头或尾 | drop one token at the head or tail
        #     coin2 = np.random.choice([True, False])
        #     if coin2:
        #         m_tokens = m_tokens[:-1]
        #     else:
        #         m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        # if m_tokens_len+1 < self.window_size+1: # t2d 用, 记得改成 torch: 这种是 121 | 补1, 是错误 motion
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.window_size-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) # 用 1 补上剩下的
        # else: # 已经到达了末端, 为 max length
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0) # 补上一个 end_inx

        # if m_tokens_len+1 < self.max_motion_length: # m_tokens_len + end_inx 小于 self.max_motion_length
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) # 用 1 补上剩下的
        # else: # 已经到达了末端, 为 max length
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0) # 补上一个 end_inx
        
        # if m_tokens_len < self.window_size: #  到达 music 末端, padding 补齐
        #     m_tokens = torch.cat([m_tokens, torch.ones((1), dtype=int) * self.mot_end_idx, torch.ones((self.window_size-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) # 用 1 补上剩下的
        # else: # 还没有到达末端
        #     pass

        assert m_tokens_len == self.window_size

        if m_tokens_len < self.block_size: #  到达 music 末端, padding 补齐
            # print(" < ")
            m_tokens = torch.cat([m_tokens.view(-1), torch.ones((1), dtype=int) * self.mot_end_idx, torch.ones((self.block_size-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) # 用 1 补上剩下的
            # print("m_tokens.shape", m_tokens.shape)
        elif m_tokens_len == self.block_size:
            m_tokens = torch.cat([m_tokens.view(-1), torch.ones((1), dtype=int) * self.mot_end_idx], axis=0)
            # print(" == ")
            # print("m_tokens.shape", m_tokens.shape)
        else: # 还没有到达末端
            print("error info !!")
            print("m_tokens.shape",m_tokens.shape)
            print("final_len", m_tokens_len)
            raise("error of max")
        
        if music.shape[0] < self.window_size*self.clip_frames: # 补齐 music feature: torch.Size([120, 35])
            zero_emb = torch.zeros([int(self.window_size*self.clip_frames -music.shape[0]), music.shape[1]])
            music = torch.cat([music,zero_emb],dim=0) # 使用 0 补齐剩下的维度

        return music, m_tokens.reshape(-1), genre, m_tokens_len   # torch.Size([128, 64, 135]) | 统一输出格式为 3+22*6 的 SMPL 格式
        

    def get_train_test_list(self, dataset="FineDance"):
        if dataset in ["AISTPP", "AISTPP_60FPS", "AISTPP_151", "AISTPP_266", "AISTPP_290"]:
            train = []
            test = []
            ignore = []

            train_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_train.txt', 'r')
            for fname in train_file.readlines():
                train.append(fname.strip())
            train_file.close()

            test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_test.txt', 'r')
            for fname in test_file.readlines():
                test.append(fname.strip())
            test_file.close()
                              
            test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_val.txt', 'r')
            for fname in test_file.readlines():
                test.append(fname.strip())
            test_file.close()

            ignore_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/ignore_list.txt', 'r')
            for fname in ignore_file.readlines():
                ignore.append(fname.strip())
            ignore_file.close()

            return ignore, train, test

        elif dataset == "AISTPP_LONG263":
            train = []
            test = []
            ignore = []
            print("modir", self.motion_dir)
            for file in os.listdir(self.motion_dir):
                if file[-4:] != '.npy':
                    continue
                file = file.split('.')[0]
                if file.split('_')[-1] in ['mLH5', 'mJS4', 'mBR3', 'mMH2', 'mPO1', 'mWA0']:
                    test.append(file)
                else:
                    train.append(file)

            return  ignore, train, test


        else:
            all_list = []
            train_list = []
            for i in range(1,212):
                all_list.append(str(i).zfill(3))
    
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
            
            if self.args.DATA_SETTING.partial == 'full':
                return ignor_list, train_list, test_list
            elif self.args.DATA_SETTING.partial == 'morden':
                for one in train_list:
                    if one in tradition_list:
                        train_list.remove(one)
                for one in test_list:
                    if one in tradition_list:
                        test_list.remove(one)
                return ignor_list, train_list, test_list
            elif self.args.DATA_SETTING.partial == 'tradition':
                for one in train_list:
                    if one in morden_list:
                        train_list.remove(one)
                for one in test_list:
                    if one in morden_list:
                        test_list.remove(one)
                return ignor_list, train_list, test_list


def DATALoader(args, isTrain, batch_size = 128, num_workers = 8):
    # 定义数据集
    trainSet = m2d_Smpl(args, isTrain)
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
        for x in iterable:
            yield x
