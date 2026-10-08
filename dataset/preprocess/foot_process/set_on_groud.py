import argparse
import os
from pydoc import doc
from cv2 import mean
import numpy as np
from pathlib import Path
import torch
import sys
import glob
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) 
from dataset.quaternion import ax_to_6v, ax_from_6v
# from dataset.preprocess import Normalizer, vectorize_many
# from data.utils.smplfk import SMPLX_Skeleton, do_smplxfk
# from train import train


floor_height = 0


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt

def set_on_ground(root_pos, local_q_156, smplx_model):
    # root_pos = root_pos[:, :] - root_pos[:1, :]
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    positions = smplx_model.forward(local_q_156, root_pos)
    positions = positions.view(length, -1, 3)   # bxt, j, 3
    
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos, local_q_156

def set_on_ground_139(data, smplx_model, ground_h=0):
    length = data.shape[0]
    assert len(data.shape) == 2
    assert data.shape[1] == 139
    positions = do_smplxfk(data, smplx_model)
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    data[:, 5] = data[:, 5] - (height -  ground_h)

    return data

def motion_feats_extract(moinputs_dir, mooutputs_dir, music_indir, music_outdir):
    ignor_list, train_list, test_list = get_train_test_list()

    device = "cpu"
    print("extracting")
    raw_fps = 30
    data_fps = 30
    data_fps <= raw_fps
    data_stride = raw_fps // data_fps
    device = "cpu"
    smplx_model = SMPLX_Skeleton() # smplx.SMPLX(model_path='/data/lrh/smpl_model/smplx', ext='npz', gender='neutral',
                                    #num_betas=10, flat_hand_mean=True, num_expression_coeffs=10, use_pca=False).eval()
        
    motions = sorted(glob.glob(os.path.join(moinputs_dir, "*.npy")))
    # assert len(motions) == len(features)
    for motion in tqdm(motions):
        # if len(all_pos) >= 20:
        #     break
        print(motion)
        # make sure name is matching
        # f_name = os.path.join(mooutputs_dir, "jukebox_feats", name + ".npy")
        # load motion
        data = np.load(motion)
        fname = os.path.basename(motion).split(".")[0]
        mname = fname if 'M' not in fname else fname[1:]
        music_fea = np.load(os.path.join(music_indir, mname+".npy"))
        if mname in ["010", "014"]:
            data = data[3:]
            music_fea = music_fea[3:]
        # if mname in ['105', '110', '113' , '153', '211']:
        if mname == '004':
            data = data[8*30:]
            music_fea = music_fea[8*30:]
        if mname == '005':
            data = data[10*30:]
            music_fea = music_fea[10*30:]
        if mname == '067':
            data = data[6*30:]
            music_fea = music_fea[6*30:]
        if mname == '105':
            data = data[19*30:]
            music_fea = music_fea[19*30:]
        if mname == '110':
            data = data[14*30:]
            music_fea = music_fea[14*30:]
        if mname == '113':
            data = data[29*30:]
            music_fea = music_fea[29*30:]
        if mname == '153':
            data = data[52*30:]
            music_fea = music_fea[52*30:]
        if mname == '211':
            data = data[22*30:]
            music_fea = music_fea[22*30:]
        
        # 004:8
        # 005:10
        # 067:6
        # 105:19
        # 110:14
        # 113:29
        # 153:52
        # 211:22

        # np.save(os.path.join(music_outdir, mname+".npy"), music_fea)

        pos = data[:, 4:7]   # length, c
        q = data[:, 7:]
        print("data.shape", data.shape)
        print("pos.shape", pos.shape)
        print("q.shape", q.shape)
        # root_pos = pos
        root_pos = torch.Tensor(pos).to(device) # 150, 3
        local_q = torch.Tensor(q).to(device).view(q.shape[0], 52, 6)    # 150, 165
        local_q = ax_from_6v(local_q)
        length = root_pos.shape[0]
        local_q = local_q.view(length, -1, 3)  
        print("local_q", local_q.shape)
        # local_q_156 = torch.concat((local_q[:, :22, :], local_q[:, 25:, :]), dim=1)
        local_q_156 = local_q.view(length, 156)

        root_pos, local_q_156 = set_on_ground(root_pos, local_q_156, smplx_model)

        positions = smplx_model.forward(local_q_156, root_pos)

        positions = positions.view(length, -1, 3)   # bxt, j, 3
        

        # contacts

        feet = positions[:, (7, 8, 10, 11)]  # # 150, 4, 3

        contacts_d_ankle = (feet[:,:2,1] < 0.12).to(local_q_156)
        contacts_d_teo = (feet[:,2:,1] < 0.05).to(local_q_156)

        contacts_d = torch.cat([contacts_d_ankle, contacts_d_teo], dim=-1).detach().cpu().numpy()


        local_q_156 = local_q_156.view(length, 52, 3)  
        local_q_156 = ax_to_6v(local_q_156).view(length,312).detach().cpu().numpy()

        print("contacts_d.shape", contacts_d.shape)
        print("root_pos.shape", root_pos.shape)
        print("local_q_156.shape", local_q_156.shape)
        mofeats_input = np.concatenate( [contacts_d, root_pos, local_q_156] ,axis=-1)
        # np.save(os.path.join(mooutputs_dir, fname+".npy"), mofeats_input)
        print("mofeats_input", mofeats_input.shape)


        if positions.shape[0] < 600:
            continue
        x = np.arange(0, 600) 
        lankle  = positions[0:600, 7,  1]
        # l1cv = contacts[0:600, 0]/10+0.4
        
        ltoe = positions[0:600, 10,  1]
        rtoe = positions[0:600, 11,  1]
        lanklecd = contacts_d[0:600, 0]/10 + 0.2
        ltoecd = contacts_d[0:600, 2]/10 + 0.3
        # l1anklecd = contacts_d[0:600, 0]/10 + 0.2

        # l2 = positions[0:600, 8,  1] 
        plt.plot(x,  ltoe, 0.01,'o', label='ltoe')
        plt.plot(x,  rtoe, 0.01,'o', label='rtoe')
        plt.plot(x,  lankle, 0.01,'o', label='lankle')
        # plt.plot(x, l2, 0.1, 'o', label='l2')

        # plt.plot(x,  l1cv, 0.01,'o', label='l1cv')
        plt.plot(x,  ltoecd, 0.01,'o', label='ltoecd')
        plt.plot(x,  lanklecd, 0.01,'o', label='lanklecd')
        plt.plot(x,  lankle-ltoe + 0.5, 0.01,'o', label='cha')


        # plt.plot(x, r2c, 'o', label='r2c')
        # 设置标题、轴标签和图例
        plt.title("四条随机数曲线")
        plt.xlabel("横轴 (1-100的整数)")
        plt.ylabel("纵轴 (随机小数)")
        plt.legend()
        # 保存图像到文件
        plt.savefig("abandon/pic3/" + os.path.basename(motion).replace("npy", "png"), dpi=600)
        print("wait for a long time, now loading done!")
        plt.close()
        # all_l = np.array(all_l)  # N x seq x 319


        
        

    # data = {"mofeats_input": mofeats_input, "filenames": all_names, "music_features": all_music_features}
    
    # with open(os.path.join(mooutputs_dir, "alldata_rot6d_jukebox.pkl"), "wb") as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return


def get_train_test_list():
    path = '/data/human/datasets/data/motion_npy_smpl'
    all_list = []
    train_list = []
    for i in range(1,212):
        all_list.append(str(i).zfill(3))
    print(all_list)
    print('all_list is :', len(all_list))

    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123"]
    for one in all_list:
        if one not in test_list:
            train_list.append(one)
    print(train_list)
    print('train_list is :', len(train_list))
    
    return ignor_list, train_list, test_list


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument("--dance_path", type=str, default="/data/human/datasets/data/motion_npy_smpl", help="raw dance data path")
    parser.add_argument("--music_path", type=str, default="/data/human/datasets/data/music", help="raw music data path")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    motion_feats_extract(moinputs_dir='/data2/lrh/dataset/fine_dance/origin/motion_feature319mirror/', mooutputs_dir="/data2/lrh/dataset/fine_dance/gound/mofea319", music_indir="/data2/lrh/dataset/fine_dance/origin/music_feature35_edge", music_outdir="/data2/lrh/dataset/fine_dance/gound/musicfea_edge", )