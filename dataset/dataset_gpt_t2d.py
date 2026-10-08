import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import random
# import torchgeometry as tgy
import codecs as cs


class t2d_Smpl(data.Dataset):
    def __init__(self, args, istrain): # 通过 istrain 来选择使用的数据集
        # HumanML3D(AMASS + Humanact12) 
        if istrain:
            split_file = args.o_h3d_dir +  'train.txt'
        else:
            split_file = args.o_h3d_dir +  'test.txt'

            # 得到所有 train 文件的路径
        self.window_size = args.window_size
        new_name_list = []
        data_dict = {}
        self.lengths = []
        id_list = []

        # 同 T2M-GPT
        fps = 20
        unit_length = 4
        self.max_motion_length = 51
        self.mot_end_idx = args.nb_code
        self.mot_pad_idx = args.nb_code + 1

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # id_list = id_list[:129] # debug

            # 读取对应的处理后的 HumanML3D(amass+huamnact12) 数据
        for name in tqdm(id_list,desc='HumanML3D'):
            try:
                # read motion_token
                motion =  np.load(args.tk_dir + f'/f_{args.DATA_SETTING.nfeats}/' + name + '.npy', allow_pickle=True)[:,:args.DATA_SETTING.nfeats]

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
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag: # 正常情况命名: 一整段 motion 对应一个 caption
                    data_dict[name] = {'motion_token': motion,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
            # self.h3d_motion = torch.cat(self.h3d_motion,dim=0) # 为了能够和 text 对齐, 保留 motion 的分段性, 使用时随机截取部分
        print(f'HumanML3D has {len(self.data_dict)} samples..')
        

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index): # 参考 T2M-GPT/dataset/dataset_VQ.py
        data = self.data_dict[self.name_list[index]]
        m_token_list, text_list = data['motion_token'], data['text']

        # 随机选一个动作
        m_tokens = random.choice(m_token_list)

        # 选文本
        text_data = random.choice(text_list)
        caption= text_data['caption'] # 除了 caption(全文), 还有单词(token)

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

        if m_tokens_len+1 < self.max_motion_length: # m_tokens_len + end_inx 小于 self.max_motion_length
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) # 用 1 补上剩下的
        else: # 已经到达了末端, 为 max length
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0) # 补上一个 end_inx
        
        return caption, m_tokens.reshape(-1), m_tokens_len   # torch.Size([128, 64, 135]) | 统一输出格式为 3+22*6 的 SMPL 格式
    


def DATALoader(args, isTrain, batch_size = 128, num_workers = 8):
    # 定义数据集
    trainSet = t2d_Smpl(args, isTrain)
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
