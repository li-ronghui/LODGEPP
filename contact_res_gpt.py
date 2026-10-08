import glob
import os
import pickle
import numpy as np
import argparse
from omegaconf import OmegaConf
import sys, glob
from pathlib import Path

from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
import torch
from render_ske import ax_from_6v,ax_to_6v

def get_songlist(pkldir):
    songlist = []
    for file in os.listdir(pkldir):
        if file[-3:] != 'npy':
            continue
        if len(file.split('_')) > 2:
            song = file.split('_')[2].split("g")[0]
        if song not in songlist:
            songlist.append(song)

    return songlist

def get_repeatnum(pkldir):
    num = 0
    for file in os.listdir(pkldir):
        if not file[-3:] == 'pkl':
            continue
        # print(file)
        if not "_r" in file:
            return 1
        else:
            if num < int(file.split('.')[0].split('_r')[-1]):
                num = int(file.split('.')[0].split('_r')[-1])
    return num+1


def contact_res(modir):
    songlist = get_songlist(modir)
    repeatnum = get_repeatnum(modir)
    print("repeatnum is {}".format(str(repeatnum) ))
    print('songlist', songlist)


    catdir = os.path.join(Path(modir), "concat" , 'npy')        # 合并后的保存目录
    if not os.path.exists(catdir):
        os.makedirs(catdir)
    
    # if repeatnum == 0:
    for song in songlist:
        # if '009' in song or '005' in song or '004' in song  or '002' in song or '003' in song  or '007' in song:
        #     continue
        # one_song 保存了一个歌曲的所有pkl的路劲
        one_song = sorted(glob.glob(os.path.join(modir, 'dod' + '*'+ song + 'g' + '*')))
        print(one_song)
        print("total num", len(one_song))
        idx = 0
        total_num = len(one_song)

        for idx in range(total_num):
            gi = idx //4
            li = idx %4 
            print("query is : --------------")
            print(os.path.join(modir, 'dod' + '*'+ song + 'g' + str(gi).zfill(3) + '_l' + str(li).zfill(3) + '_r0.pkl'))
            local_fineme = sorted(glob.glob(os.path.join(modir, 'dod' + '*'+ song + 'g' + str(gi).zfill(3) + '_l' + str(li).zfill(3) + '_r0.npy')))
            if len(local_fineme) == 1:
                local_fineme = local_fineme[0]
            print("local_fineme", local_fineme)
            modata = np.load(open(local_fineme, "rb"))
        
            if idx == 0:
                dance = modata
            else:
                dance = np.concatenate((dance, modata), axis=0)
            print(idx)

        print("danceshape", dance.shape)
        np.save(os.path.join(catdir, song+'.npy'), dance)
        # root = dance[:,:3]
        # rot6d = torch.from_numpy(dance[:,3:]).reshape(dance.shape[0], 22, 6)
        # qua = rotation_6d_to_matrix(rot6d)
        # qua = matrix_to_quaternion(qua).view(dance.shape[0], -1).detach().cpu().numpy()
        # print("qua.shape", qua.shape)
        # qua_data = np.concatenate([root, qua], axis=1)
        # print("qua_data.shape", qua_data.shape)
        # np.save(os.path.join(quadir, song+'.npy'), qua_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--modir", type=str, default='experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint0_trans') 
    # "experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/test2")

    
    # parser.add_argument(
    #     "--transit",
    #     action="store_true",
    #     help="weather optmize transition between two 512 dance clips",
    # )
    # if parser.parse_args().transit:
    #     parser.add_argument("--gpu", type=str, default="2")

    args = parser.parse_args()
    args = OmegaConf.create(vars(args))


    contact_res(args.modir)
    



    