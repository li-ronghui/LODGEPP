'''
此程序功能为针对预处理的特征，从root_tran中对xz积分，得到gloabl trans
'''
import torch

def get_data(data):
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, :2]
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 2]
    
    rot6d = data[:,3: 3+22*6]
    
