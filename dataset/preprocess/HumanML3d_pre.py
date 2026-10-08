# Humanml3D 转成 Finedance 同一格式
import os
import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import time
import pandas as pd
from os.path import join as pjoin
# from dataset.preprocess.finedance_preprocess import Mmofeature
os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append('/data2/lrh/dyq/debug_exp/')
from data.utils.quaternion import ax_from_6v, quat_slerp, ax_to_6v
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
from utils.smplfk import SMPLSkeleton
from tqdm import tqdm


def swap_left_right(data):   
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    
    if data.shape[-1] == 22*6:
        device_ = data.device
        t,c= data.shape
        data = ax_from_6v(data.view(t,22,6))
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

def amass_to_motion(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)
    fps = 0
    ex_fps = 20
    try:
        fps = bdata['mocap_framerate']
        frame_number = bdata['trans'].shape[0]
    except:
#         print(list(bdata.keys()))
        return fps
    
    fId = 0 # frame id of the mocap sequence
    # pose_seq = []
    trans = []
    root_orient = []
    pose_body = []
    # pose_hand = []
    
    
    down_sample = int(fps / ex_fps)
#     print(frame_number)
#     print(fps)
    
    # with torch.no_grad():
    for fId in range(0, frame_number, down_sample):
        root_orient.append(torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(device)) # controls the global root orientation
        pose_body.append(torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(device)) # controls the body
        # pose_hand.append(torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(device)) # controls the finger articulation
        # betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(device) # controls the body shape
        trans.append(torch.Tensor(bdata['trans'][fId:fId+1]).to(device))   

        
    trans = torch.cat(trans, dim=0)   
    seqlen = trans.shape[0]
    root_orient = ax_to_6v(torch.cat(root_orient, dim=0)).view(seqlen, -1) 
    pose_body = ax_to_6v(torch.cat(pose_body, dim=0).view(seqlen, 21, 3)).view(seqlen, -1) 
    # pose_hand = ax_to_6v(torch.cat(pose_hand, dim=0).view(seqlen, -1, 3)).view(seqlen, -1) 
    
    motion = torch.cat([trans, root_orient, pose_body[:,:(numjoints-1)*6]], dim = -1)
    motion = motion.detach().cpu().numpy()
    np.save(save_path, motion)
    
    return fps


def get_mofeatures(motion, smplfk):
    motion = motion.clone()
    assert motion.shape[-1] == 135
    motion[:, :3] = motion[:, :3] - motion[:1, :3]              # 第一帧的xyz坐标归一化到原点
    trans_pos =  motion[:, :3]
  
    y_height = trans_pos[:-1, 1:2].clone()       # y的高度
    xz_vel = (trans_pos[1:, :] - trans_pos[:-1, :]).clone()
    
    # 将第一帧的全局朝向初始化
    t,c = motion.shape
    root_mat = rotation_6d_to_matrix(motion[:,3:3+6])           # t, 3, 3
    rot0 = torch.linalg.inv(root_mat[:1,:]).expand(t,3,3)          
    root_mat = torch.einsum('tij,tjk -> tik', rot0, root_mat)    # 旋转矩阵A左×B，表示将B旋转A的角度   
    root_rot6d = matrix_to_rotation_6d(root_mat).view(t, 6)
    motion[:,3:3+6] = root_rot6d
    
    positions = get_SMPL_positions(smplfk, root_trans=motion[:, :3], root_rot6d=motion[:,3:3+6], pose_rot6d=motion[:, 9:9+21*6])                             # seqlen, 22*6
    positions = positions[:, :22*3]
    positions_vel = positions[1:,:] - positions[:-1,:]          # 22*3
    
    # 计算脚接地情况  这里采用EDGE的计算方式，没有用Humanml3d的计算方式
    feet = positions.view(t, 22, 3)[:, (7, 8, 10, 11)]          # seqlen, 4, 3
    # feetv = torch.zeros(feet.shape[:2], device=device)          # seqlen, 4
    feetv = (feet[1:] - feet[:-1]).norm(dim=-1)
    foot_ctc = (feetv < 0.01).to(device)  # cast to right dtype        # seqlen-1, 4
    
    root_trans = torch.cat([xz_vel[:,:1], y_height, xz_vel[:,2:3]], dim=-1)              # 3
    root_rot6d = root_rot6d[:-1,:]                                  # 6
    pose_rot6d = motion[:-1, 9:9+21*6]                            # seqlen-1, 126
    positions = positions[:-1,:]                                   # seqlen-1, 66
    mofeature = torch.cat([root_trans, root_rot6d, pose_rot6d, positions, positions_vel, foot_ctc], dim = -1)

    return mofeature

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
    
if __name__ == '__main__':
    # device = f'cuda:{args.gpu}'
    device = f'cuda:{2}'
    numjoints = 22              # 算上root一共22个关节点

    #FK模型
    smplfk = SMPLSkeleton()
    
    # 路径准备
    motiondir = "/data2/lrh/dataset/HumanML3D/origin_data"
    outdir = "/data2/lrh/dataset/HumanML3D/motion135"      
        
    paths = []
    folders = []
    dataset_names = []
    for root, dirs, files in os.walk(motiondir):
    #     print(root, dirs, files)
    #     for folder in dirs:
    #         folders.append(os.path.join(root, folder))
        folders.append(root)
        for name in files:
            if name[-3:] == 'txt':
                continue
            dataset_name = root.split('/')[-2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))
            
    
    save_root = '/data2/lrh/dataset/HumanML3D/motion135'
    save_folders = [folder.replace('origin_data', 'pose_data') for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]

    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0

    # 将npz文件转化为npy文件，npy文件的motion维度是135=3+6+21*6：trans + root_rot6d + pose_rot6d
    for paths in group_path:
        dataset_name = paths[0].split('/')[-3]
        # if not dataset_name == 'BMLhandball':       # 用于debug，查看某一特定数据集
        #     continue
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s'%dataset_name)

        for path in pbar:
            save_path = path.replace('origin_data', 'pose_data')
            save_path = save_path[:-3] + 'npy'
            
            fps = amass_to_motion(path, save_path)  
        cur_count += len(paths)
        print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
        # time.sleep(0.5)
        
        
    
    # 提取motion特征
    index_path = '/data2/lrh/dataset/HumanML3D/index.csv'
    save_dir = '/data2/lrh/dataset/HumanML3D/mofeatures_full271'
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    fps = 20
    
    print("now extract mofeatures!")
    for i in tqdm(range(total_amount)):
        source_path = index_file.loc[i]['source_path']
        # if "SSM_synced" not in source_path:                 # debug用
        #     continue
        new_name = index_file.loc[i]['new_name']
        source_path =  "/data2/lrh/dataset/HumanML3D" + source_path[1:]
        data = np.load(source_path)
        start_frame = index_file.loc[i]['start_frame']
        end_frame = index_file.loc[i]['end_frame']
        # 注意humanact12是Humanml3d已经处理成npy的，直接拷贝到pose_data目录下
        if 'humanact12' in source_path:
            data = torch.from_numpy(data).to(device)
            root = data[:,0,:]
            pose = ax_to_6v(data[:,1:23,:]).view(root.shape[0], 132)
            data = torch.cat([root, pose], dim=-1)
        elif 'humanact12' not in source_path:
            if 'Eyes_Japan_Dataset' in source_path:
                data = data[3*fps:]
            if 'MPI_HDM05' in source_path:
                data = data[3*fps:]
            if 'TotalCapture' in source_path:
                data = data[1*fps:]
            if 'MPI_Limits' in source_path:
                data = data[1*fps:]
            if 'Transitions_mocap' in source_path:
                data = data[int(0.5*fps):]
            data = data[start_frame:end_frame]
            data = torch.from_numpy(data).to(device)
        else:
            raise("error")
 
        # print(data.shape)
        seqlen = data.shape[0]
        data_ = data.clone()
        mofeature = get_mofeatures(data_, smplfk).detach().cpu().numpy()
        
        data_m = swap_left_right(data[:,3:3+22*6])
        # print("data_m.shape",data_m.shape)
        Mroot = data[:,:3]
        Mroot[:,0] *= -1
        # print("Mroot.shape",Mroot.shape)
        
        Mtemp= ax_to_6v(data_m[:,:,:]).view(seqlen, 132)
        data_m = torch.cat([Mroot, Mtemp],dim=1)          #  3+132=135数据维度和motion保持一致
        
        # np.save(pjoin(save_dir, 'dataM'+new_name), data_m.detach().cpu().numpy())
        
        Mmofeature = get_mofeatures(data_m, smplfk).detach().cpu().numpy()
           
    #     save_path = pjoin(save_dir, )
        
        np.save(pjoin(save_dir, new_name), mofeature)
        np.save(pjoin(save_dir, 'M'+new_name), Mmofeature)



    
# for dataset in (os.listdir(motiondir)):
#     for split in (os.listdir(os.path.join(motiondir, dataset))):
#         for file in (os.listdir(os.path.join(motiondir, dataset, split))):
#             filepath = os.path.join(motiondir, file)
#             motion = torch.from_numpy(np.load(filepath)).float().to(device)
#             if motion.shape[-1] == 319 or motion.shape[-1] == 139:           # The first 4 dimension are foot contact
#                 motion[:, 4:7]  = motion[:, 4:7] - motion[:1, 4:7]    
#                 trans_pos =  motion[:, 4:7]     
#             else:
#                 motion[:, :3] = motion[:, :3] - motion[:1, :3]              # 第一帧的xyz坐标归一化到原点
#                 trans_pos =  motion[:, :3]
            
#             y_height = trans_pos[:-1, 1:2].clone()       # y的高度
#             xz_vel = (trans_pos[1:, :] - trans_pos[:-1, :]).clone()
#             xz_vel = torch.cat([xz_vel[:,:1], xz_vel[:,2:3]], dim=-1)
            
#             # 将第一帧的全局朝向初始化
#             t,c = motion.shape
#             root_mat = rotation_6d_to_matrix(motion[:,7:7+6])           # t, 3, 3
#             rot0 = torch.linalg.inv(root_mat[:1,:]).expand(t,3,3)          
#             root_mat = torch.einsum('tij,tjk -> tik', rot0, root_mat)    # 旋转矩阵A左×B，表示将B旋转A的角度   
#             root_rot6d = matrix_to_rotation_6d(root_mat).view(t, 6)
#             motion[:,7:7+6] = root_rot6d
            
#             positions = get_SMPL_positions(smplfk, root_trans=motion[:, 4:7], root_rot6d=motion[:,7:7+6],pose_rot6d=motion[:, 13:13+21*6])                             # seqlen, 22*6
#             positions = positions[:, :22*3]
#             positions_vel = positions[1:,:] - positions[:-1,:]          # 22*3
            
#             # 计算脚接地情况  这里采用EDGE的计算方式，没有用Humanml3d的计算方式
#             feet = positions.view(t, 22, 3)[:, (7, 8, 10, 11)]          # seqlen, 4, 3
#             # feetv = torch.zeros(feet.shape[:2], device=device)          # seqlen, 4
#             feetv = (feet[1:] - feet[:-1]).norm(dim=-1)
#             foot_ctc = (feetv < 0.01).to(device)  # cast to right dtype        # seqlen-1, 4
            
#             root_trans = torch.cat([xz_vel, y_height], dim=-1)              # 3
#             root_rot6d = root_rot6d[:-1,:]                                  # 6
#             pose_rot6d = motion[:-1, 13:13+21*6]                            # seqlen-1, 126
#             positions = positions[:-1,3:]                                   # seqlen-1, 63
            
#             mofeature = torch.cat([root_trans, root_rot6d, pose_rot6d, positions, positions_vel, foot_ctc], dim = -1)
#             # mofeature = torch.cat([root_trans, root_rot6d, pose_rot6d], dim = -1)
#             mofeature = mofeature.detach().cpu().numpy()
            
#             np.save(os.path.join(outdir, file), mofeature)
                
    
       
    
    
      
    
                 
    
    
        
    
    