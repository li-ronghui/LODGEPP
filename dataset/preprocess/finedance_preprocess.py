import os
import numpy as np
import torch
import sys
sys.path.append('/data2/lrh/dyq/debug_exp/')
from utils.quaternion import ax_from_6v, quat_slerp, ax_to_6v
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
from utils.smplfk import SMPLSkeleton
from tqdm import tqdm
# device = f'cuda:{args.gpu}'
device = f'cuda:{0}'

# 路径准备
motiondir = "/data2/lrh/dyq/debug_exp/dataset/data/origin/motion_feature319"
outdir = "/data2/lrh/dyq/debug_exp/dataset/data/origin/mofeature_vq_full"
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
#FK模型
smplfk = SMPLSkeleton()

def get_SMPL_positions(smplfk, root_trans, root_rot6d, pose_rot6d):
    pass
    t, c = root_trans.shape    
    # trans = motion[:, 4:7].view(-1, 3)   # root position
    root_trans = root_trans.unsqueeze(0)
    rotation = torch.cat([root_rot6d, pose_rot6d], dim=-1)
    rotation = ax_from_6v(rotation.reshape(t, -1, 6)).unsqueeze(0)         
    positions = smplfk.forward(rotation, root_trans).squeeze(0)         # t, 24, 3
    positions = positions[:, :22, :].reshape(t, 66)
    return positions

def swap_left_right(data):   
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    
    if data.shape[-1] == 22*6:
        device_ = data.device
        t,c= data.shape
        data = ax_from_6v(data.view(t,22,6))
    elif data.shape[-1] == 52*6:
        t,c= data.shape
        data = ax_from_6v(data.view(t,52,6))
    assert len(data.shape) == 3 and data.shape[-1] == 3
    pose = data.clone()
    
    # pose = data[:,1:,:].clone()
    tmp = pose[:, right_chain].clone()
    pose[:, right_chain] = pose[:, left_chain].clone()
    pose[:, left_chain] = tmp.clone()
    if pose.shape[1] > 24:
        tmp = pose[:, right_hand_chain].clone()
        pose[:, right_hand_chain] = pose[:, left_hand_chain].clone()
        pose[:, left_hand_chain] = tmp.clone()
        
    pose[:,:,1:3] *= -1
    return pose



# def swap_left_right(data):   
#     right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
#     left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
#     left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
#     right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    
#     if data.shape[-1] == 3+22*6:
#         device_ = data.device
#         t,c= data.shape
#         root = data[:,:3].unsqueeze(1)
#         pose6d = ax_from_6v(data[:,3:].view(t,22,6))
#         data = torch.cat([root, pose6d], dim=1)         # .detach().cpu().numpy()
#     # data的shape为seqlen, j, 3   j为关节点数量
#     # elif data.shape[-1] == 3:
#     assert len(data.shape) == 3 and data.shape[-1] == 3
#     data = data.clone()
#     root = data[:, 0:1, :].clone()
#     root[:, 0:1, 0] *= -1
#     # data[..., 0, 0] = (data[..., 0, 0] * -1).clone()  # 原始代码对xyz都镜像，这里改为只对x镜像
#     # data[..., 0, 2] = (data[..., 0, 2] * -1).clone()
    
#     pose = data[:,1:,:].clone()
#     tmp = pose[:, right_chain].clone()
#     pose[:, right_chain] = pose[:, left_chain].clone()
#     pose[:, left_chain] = tmp.clone()
#     if pose.shape[1] > 24:
#         tmp = pose[:, right_hand_chain].clone()
#         pose[:, right_hand_chain] = pose[:, left_hand_chain].clone()
#         pose[:, left_hand_chain] = tmp.clone()
        
#     pose[:,:,1:3] *= -1
#     data = torch.cat([root, pose], dim=1)
#     return data

def get_mofeatures(motion):
    motion = motion.clone()
    t,c = motion.shape  
    
    if motion.shape[-1] == 319 or motion.shape[-1] == 139:           # The first 4 dimension are foot contact
        motion[:, 4:7]  = motion[:, 4:7] - motion[:1, 4:7]    
        trans_pos =  motion[:, 4:7]     
    else:
        raise("motion shape error!")
        # motion[:, :3] = motion[:, :3] - motion[:1, :3]              # 第一帧的xyz坐标归一化到原点
        # trans_pos =  motion[:, :3]
    
    y_height = trans_pos[:-1, 1:2].clone()       # y的高度
    xz_vel = (trans_pos[1:, :] - trans_pos[:-1, :]).clone()
    
    # 将第一帧的全局朝向初始化          由于舞蹈中存在背对着的动作，直接初始化角度而不初始化全局位置会出错
    # root_mat = rotation_6d_to_matrix(motion[:,7:7+6])           # t, 3, 3
    # rot0 = torch.linalg.inv(root_mat[:1,:]).expand(t,3,3)          
    # root_mat = torch.einsum('tij,tjk -> tik', rot0, root_mat)    # 旋转矩阵A左×B，表示将B旋转A的角度   
    # root_rot6d = matrix_to_rotation_6d(root_mat).view(t, 6)
    # motion[:,7:7+6] = root_rot6d
    
    positions = get_SMPL_positions(smplfk, root_trans=motion[:, 4:7], root_rot6d=motion[:,7:7+6], pose_rot6d=motion[:, 13:13+21*6])                             # seqlen, 22*6
    positions = positions[:, :22*3]
    positions_vel = positions[1:,:] - positions[:-1,:]          # 22*3
    
    # 计算脚接地情况  这里采用EDGE的计算方式，没有用Humanml3d的计算方式
    feet = positions.view(t, 22, 3)[:, (7, 8, 10, 11)]          # seqlen, 4, 3
    # feetv = torch.zeros(feet.shape[:2], device=device)          # seqlen, 4
    feetv = (feet[1:] - feet[:-1]).norm(dim=-1)
    foot_ctc = (feetv < 0.01).to(device)  # cast to right dtype        # seqlen-1, 4
    
    root_trans = torch.cat([xz_vel[:,:1], y_height, xz_vel[:,2:3]], dim=-1)              # 3
    root_rot6d = motion[:-1,7:7+6]                                  # 6
    pose_rot6d = motion[:-1, 13:13+21*6]                            # seqlen-1, 126
    positions = positions[:-1,:]                                   # seqlen-1, 66
    mofeature = torch.cat([root_trans, root_rot6d, pose_rot6d, positions, positions_vel, foot_ctc], dim = -1)

    return mofeature

 
ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
for file in tqdm(os.listdir(motiondir)):
    if file.split(".")[0] in ignor_list:
        continue
    filepath = os.path.join(motiondir, file)
    motion = torch.from_numpy(np.load(filepath)).float().to(device)
    seqlen = motion.shape[0]
    
    mofeature = get_mofeatures(motion)
    mofeature = mofeature.detach().cpu().numpy()
    np.save(os.path.join(outdir, file), mofeature)
    
    motion = torch.from_numpy(np.load(filepath)).float().to(device)
    
    MirrorMotion = swap_left_right(motion[:,7:7+22*6])
    Mroot = motion[:,4:7].clone()
    Mroot[:,0] *= -1
    
    Mfoot_ctc = torch.zeros([seqlen, 4]).to(motion)            # 用于格式匹配，没有实际意义
    Mtemp= ax_to_6v(MirrorMotion[:,:,:]).view(seqlen, 132)
    MirrorMotion = torch.cat([Mfoot_ctc, Mroot, Mtemp],dim=1) #  4+3+132=139数据维度和motion保持一致
    Mmofeature = get_mofeatures(MirrorMotion)
    
    
    Mmofeature = Mmofeature.detach().cpu().numpy()
    np.save(os.path.join(outdir, 'M'+file), Mmofeature)
    
    
