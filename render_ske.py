import sys, os
import torch, os, pickle
# from data.utils.motion_process import recover_from_ric,recover_from_smplx, recover_from_ric266, recover_from_ric266v, recover_from_smplx_v
import numpy as np
from DanceDiffusion.dld.data.utils.motion_process import recover_from_ric266v_grad, recover_from_ricv_norotinit_grad, recover_from_ric266
from DanceDiffusion.dld.data.utils.smplfk import SMPLX_Skeleton, do_smplxfk
import DanceDiffusion.dld.data.utils.plot_3d_global as plot_3d

import argparse
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
import smplx

def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat
def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat
def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat
def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--modir", type=str, default="/data2/lrh/dataset/fine_dance/mofea266/new_joint_vecs/")
    parser.add_argument("--fps", type=int, default=30)
    # parser.add_argument("--nfeats", type=int, default=263)
    # parser.add_argument("--joints_num", type=int, default=22)
    parser.add_argument("--outdir", type=str, default=None)
    # parser.add_argument('--unnorm', action='store_true')            # add --unnorm to unnormalize the motion feature
    args = parser.parse_args()
    print(args.gpu)
    # joints_num = args.joints_num

    smplx_model = SMPLX_Skeleton(Jpath = './smplx_neu_J_1.npy')
    device = f'cuda:{args.gpu}'
    modir = args.modir
    if args.outdir is None:
        outdir = os.path.join(modir, 'ske_55_v')
    else:
        outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    molist = os.listdir(modir) 
    # if args.unnorm:
    #     if args.nfeats == 263:
    #         mean = np.load(os.path.join('/data2/lrh/dataset/fine_dance/mofea263cut', 'Mean.npy'))
    #         std = np.load(os.path.join('/data2/lrh/dataset/fine_dance/mofea263cut', 'Std.npy'))
    #     elif args.nfeats == 266:
    #         mean = np.load(os.path.join('/data2/lrh/dataset/fine_dance/mofea266', 'Mean.npy'))
    #         std = np.load(os.path.join('/data2/lrh/dataset/fine_dance/mofea266', 'Std.npy'))

    joints3d = []
    file_li = []
    print( os.listdir(modir))
    for file in os.listdir(modir):
        # if not '098' in file:
        #     continue
        # if '001' not in file:
        #     continue
        if not file.split('.')[-1] in  ['pkl', 'npy']:
            continue
        if file[0] == "M":
            continue
        # if file[:3] == "gpt":
        #     continue
        # if int(file.split('_')[0]) < 1700:
        #     continue
        print("file", file)
        if file[-3:] == 'pkl':
            normdata = np.load('/data2/lrh/dataset/fine_dance/total/mofea338/new_joint_vecs/001.npy')
            normdata = recover_from_smplx(torch.from_numpy(normdata), 55).detach().cpu().numpy()
            print("normdata", normdata.shape)
            normleg = normdata[0,8] - normdata[0,5]
            print("normleg", normleg.shape)
            print("normleg", normleg)
            normleg = np.linalg.norm(normleg, ord=2)
            print("normleg", normleg)

            pkl_data = pickle.load(open(os.path.join(modir, file), 'rb'))
            file_li.append(os.path.join(outdir,file.replace('pkl', 'mp4')))
            smpl_poses = torch.from_numpy(pkl_data["pose"]).reshape(-1, 52, 3)
            joints = torch.from_numpy(pkl_data["xyz"])  
            joints = joints[3000:4000]
            joints = joints[::4,:52,:]
            leg = joints[0,8] - joints[0,5]
            leg = np.linalg.norm(leg, ord=2)
            joints_num = 52
            scale = leg / normleg
            joints = joints / scale
            joints[:,:,0] = joints[:,:,0]  - joints[0,0,0]
            joints[:,:,2] = joints[:,:,0]  - joints[0,0,2]
            
        elif file[-3:] == 'npy':
            motion = np.load(os.path.join(modir, file))
            if len(motion.shape) == 3 and motion.shape[0] == 1:
                motion = motion[0]
            print('-----------motion-------0', motion.shape)
            # if args.unnorm:
            #     motion = (motion * std) + mean
            file_li.append(os.path.join(outdir,file.replace('npy', 'mp4')))
            print("motion shape", motion.shape)
            nfeats = motion.shape[1]
            # if motion.shape[1] == 263 or motion.shape[1] == 266:
            #     print("npy , 263")
            #     joints_num = 22
            #     motion = torch.from_numpy(motion).to(torch.float32)
            # elif  motion.shape[1] == 290:
            #     joints_num = 24
            #     motion = torch.from_numpy(motion).to(torch.float32)
            # elif len(motion.shape) == 3 and motion.shape[-1] == 266:
            #     print('motion--------0', motion.shape)
            #     motion = torch.from_numpy(motion).to(torch.float32)
            #     motion = motion.reshape(-1, 266)
                # print('motion--------1', motion.shape)
            if motion.shape[1] == 263:
                joints_num = 22
                motion = torch.from_numpy(motion).to(torch.float32)
                joints = recover_from_ric(motion, joints_num)#.detach().cpu().numpy()
            elif motion.shape[1] == 266:
                joints_num = 22
                motion = torch.from_numpy(motion).to(torch.float32)
                # debug 
                if '60fps' in modir:
                    motion = motion[::2]
                print('motion--------2', motion.shape)
                joints_num = 22
                joints = recover_from_ric266v_grad(motion, joints_num)
                # joints = recover_from_ric266(motion, joints_num)
                print('joints', joints.shape)
            elif motion.shape[1] == 290:
                joints_num = 24
                motion = torch.from_numpy(motion).to(torch.float32)
                # debug 
                if '60fps' in modir:
                    motion = motion[::2]
                print('motion--------2', motion.shape)
                joints = recover_from_ricv_norotinit_grad(motion, joints_num)
                # joints = recover_from_ric266(motion, joints_num)
            elif motion.shape[1] == 338:
                joints_num = args.joints_num = 55
                motion = torch.from_numpy(motion)
                joints = recover_from_smplx_v(motion, joints_num)
                # joints = recover_from_smplx(motion, joints_num)
            elif motion.shape[1] == 319:
                # /data/lrh/datasets/fine_dance/gound/mofea319
                joints_num = 55
                motion = torch.from_numpy(motion)
                trans = motion[:,4:7]
                rot6d = motion[:,7:].reshape(motion.shape[0], 52, 6)
                axis = ax_from_6v(rot6d).reshape(motion.shape[0], 156)
                joints = smplx_model.forward(axis, trans)
            elif motion.shape[1] == 24 and motion.shape[2] == 3:
                joints_num = 24
                joints = torch.from_numpy(motion)

        print("joints0", joints[0,0,:])
        joints = joints.reshape(joints.shape[0], joints_num*3).detach().cpu().numpy()
        roott = joints[:1, :3]  # the root Tx72 (Tx(24x3))
        joints = joints - np.tile(roott, (1, joints_num)) 
        joints = joints.reshape(-1, joints_num, 3)
        joints3d.append(joints[:400, :])

    pose_vis = plot_3d.draw_to_batch(joints3d, None, file_li, joints_num)




