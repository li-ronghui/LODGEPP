import numpy as np
import torch
import sys, os
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

# from EDGE import data
sys.path.append(os.getcwd())
from DanceDiffusion.dld.data.utils.motion_process import recover_from_ric266v_grad, recover_from_ricv_norotinit_grad
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

mode = 'traj_4' # traj_4    traj_7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daname", type=str,
            required=False,
            default="FINEDANCE_266CUT",     # #  FINEDANCE_266CUT  AISTPP_290
            help="name of dataset",
        )
    parser.add_argument("--assets", type=str,
            required=False,
            default="/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/configs/45assets.yaml",
            help="config file for asset paths",
        )
    args = parser.parse_args()
    args = OmegaConf.create(vars(args))
    cfg_assets = OmegaConf.load(args.assets)
    args = OmegaConf.merge(cfg_assets, args)
    
    dataname = args.daname
    datarootdir = eval(f"args.DATASET.{dataname.upper()}.ROOT")
    traj_savedir = os.path.join(datarootdir, mode)
    os.makedirs(traj_savedir, exist_ok=True)
    modir = eval(f"args.DATASET.{dataname.upper()}.MOTION")
    print('modir', modir)
    mean = np.load( eval(f"args.DATASET.{dataname.upper()}.normalizer.params.mean") ) 

    num = 0
    for file in os.listdir(modir):
        mofile = os.path.join(modir, file)
        modata = np.load(mofile)
        if modata.shape[1] == 266:
            print('data.shape[1] is 266')
            if mode == 'traj_7':
                joint_index = [7,8 , 15,16,17,20,21]
            elif mode == 'traj_4':
                joint_index = [7,8 , 20,21]
            # motion = torch.from_numpy(modata).to(torch.float32)
            # joints_num = 22
            # joints = recover_from_ric266v_grad(motion, joints_num)
        elif modata.shape[1] == 290:
            if mode == 'traj_7':
                joint_index = [7,8 , 15,16,17,22,23]
            elif mode == 'traj_4':
                joint_index = [7,8 , 22,23]
            print('data.shape[1] is 290')
        motion = modata[:,:7]
        for index_ in joint_index:
            motion = np.concatenate( [motion, modata[:,7+ (index_-1)*3 : 7+(index_*3)]], axis=1 )
            # print('7+ (index_-1)*3', 7+ (index_-1)*3)
            # print('7+(index_*3)', 7+(index_*3))
            # print('modata[:,:7+ (index_-1)*3 : 7+(index_*3)]', modata[:,7+ (index_-1)*3 : 7+(index_*3)].shape)
        print('motion', motion.shape)

        np.save(os.path.join(traj_savedir, file), motion)
        # print('motion_beats', motion_beats[0].shape)
        # print('len(kinetic_vel)', len(kinetic_vel))
