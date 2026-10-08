import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import random
# import torchgeometry as tgy
import codecs as cs
import json
import os

class mask2d_Smpl(data.Dataset):
    def __init__(self, args, istrain): # 通过 istrain 来选择使用的数据集

        self.window_size = args.window_size
        new_name_list = []
        data_dict = {}
        self.lengths = []
        id_list = []

        ignor_list, train_list, test_list = self.get_train_test_list()
        if istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list

        self.datalist = self.datalist[:2] # debug

        # finedance
        for name in tqdm(self.datalist, desc='FineDance'):
            name_npy = name + ".npy"
            if name_npy[:-4] in ignor_list:
                continue
     
            # 加载 motion
            motion = np.load(os.path.join(args.tk_dir, name_npy)) # 加载预处理的数据 | FineDance 取 3+22*6 + 别的部分 263
            motion = torch.from_numpy(motion).float() # 转为 tensor

            if motion.shape[1]<self.window_size: # 过滤掉太短的
                continue
            
            input_name_npy = name_npy
            if name_npy[0] =='M':
                input_name_npy = name_npy[1:]

            # 加载 music
            music = np.load(os.path.join(args.fd_music_dir, input_name_npy))
            music = torch.from_numpy(music).float() # 转为 tensor


            min_all_len = min(motion.shape[1], music.shape[0])

            # if motion.shape[1] > max_motion_len: # 寻找最大 len
            #     max_motion_len = motion.shape[1]

            motion = motion[:min_all_len]
            music = music[:min_all_len]

            data_dict[name] = {'motion_token': motion,
                                'text':None,
                                'music':music,
                                }
            new_name_list.append(name)

        print(f'FineDance has {len(new_name_list)} samples..')

        # HumanML3D(AMASS + Humanact12) 
        if istrain:
            split_file = args.o_h3d_dir +  'train.txt'
        else:
            split_file = args.o_h3d_dir +  'test.txt'

        # HumanML3D
        # 同 T2M-GPT
        fps = 20
        unit_length = 4
        # self.max_motion_length = 51
        self.mot_end_idx = args.nb_code
        self.mot_pad_idx = args.nb_code + 1

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        id_list = id_list[:2] # debug

            # 读取对应的处理后的 HumanML3D(amass+huamnact12) 数据
        for name in tqdm(id_list,desc='HumanML3D'):
            try:
                # read motion_token
                motion =  np.load(args.tk_dir + f'/f_{args.DATA_SETTING.nfeats}/' + name + '.npy', allow_pickle=True)[:,:args.DATA_SETTING.nfeats]
                motion = torch.from_numpy(motion).float()

                if motion.shape[1]<self.window_size: # 过滤掉太短的
                    continue

                # Read text
                with cs.open(args.h3d_texts_dir + name + '.txt') as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else: # 非正常情况命名 | 对一段数据中的某一段动作进行额外的标注
                                motion_sub = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in motion if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(motion_sub) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                motion_sub = torch.from_numpy(motion_sub).float() # 转为 tensor
                                
                                data_dict[new_name] = {'motion_token': motion_sub,
                                                       'text':[text_dict],
                                                       'music':None}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag: # 正常情况命名: 一整段 motion 对应一个 caption
                    data_dict[name] = {'motion_token': motion,
                                       'text':text_data,
                                       'music':None}
                    new_name_list.append(name)
            except:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
            # self.h3d_motion = torch.cat(self.h3d_motion,dim=0) # 为了能够和 text 对齐, 保留 motion 的分段性, 使用时随机截取部分
        print(f'HumanML3D has {len(self.data_dict)} samples..')
    
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

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index): # 参考 T2M-GPT/dataset/dataset_VQ.py
        data = self.data_dict[self.name_list[index]]
        m_token_list, text_list, music = data['motion_token'], data['text'], data['music']

        # 随机选一个动作
        m_tokens = random.choice(m_token_list)

        # 截取某一段
        m_tokens = m_tokens.reshape(-1)
        seq_len = m_tokens.shape[0]
        idx = random.randint(0, seq_len - self.window_size)
        m_tokens = m_tokens[idx:idx+self.window_size]

        text_data, caption = None, None
        # 如果是 text-motion
        if text_list is not None:
            # 选文本
            text_data = random.choice(text_list)
            caption= text_data['caption'] # 除了 caption(全文), 还有单词(token)
        # 如果是 music-dance
        if music is not None:
            music_range = min(music.shape[0],idx*4 + self.window_size*4)
            music = music[idx*4:music_range] # 音乐的 feature 应该是 m_tokens 的四倍
        
            if music.shape[0] < self.window_size*4: # 补齐 music feature: torch.Size([120, 35])
                zero_emb = torch.zeros([self.window_size*4 -music.shape[0], music.shape[1]])
                music = torch.cat([music,zero_emb],dim=0) # 使用 0 补齐剩下的维度

        # 选 m_token | 一个数
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # 随机选择序列, 随机剪掉头或尾 | drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len < self.window_size: #  padding 补齐, 不补 ending
            m_tokens = torch.cat([m_tokens, torch.ones((self.window_size-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) # 用 1 补上剩下的
        else: # 不需要补
            pass
        
        cond = None
        is_music = False
        if music is not None:
            cond = music
            is_music = True
        else:
            cond = caption

        return cond, m_tokens.reshape(-1), m_tokens_len, is_music   # torch.Size([128, 64, 135]) | 统一输出格式为 3+22*6 的 SMPL 格式
    


def DATALoader(args, isTrain, batch_size = 128, num_workers = 8):
    # 定义数据集
    trainSet = mask2d_Smpl(args, isTrain)
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
