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
from DanceDiffusion.dld.data.utils.smplfk import SMPLX_Skeleton

from DanceDiffusion.dld.data.utils.smplfk import do_smplxfk



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daname", type=str,
            required=False,
            default="AISTPP_151",     # #  FINEDANCE_266CUT  AISTPP_290   FINEDANCE_139CUT    AISTPP_151
            help="name of dataset",
        )
    parser.add_argument("--assets", type=str,
            required=False,
            default="DanceDiffusion/configs/45assets.yaml",
            help="config file for asset paths",
        )
    args = parser.parse_args()
    args = OmegaConf.create(vars(args))
    cfg_assets = OmegaConf.load(args.assets)
    args = OmegaConf.merge(cfg_assets, args)
    smplx_model = SMPLX_Skeleton()
    
    dataname = args.daname
    datarootdir = eval(f"args.DATASET.{dataname.upper()}.ROOT")
    keymo_savedir = os.path.join(datarootdir, 'keymo')
    os.makedirs(keymo_savedir, exist_ok=True)
    modir = eval(f"args.DATASET.{dataname.upper()}.MOTION")
    print('modir', modir)
    # mean = np.load( eval(f"args.DATASET.{dataname.upper()}.normalizer.params.mean") ) 

    num = 0
    for file in os.listdir(modir):
        if file.split('.')[-1] != 'npy':
            continue
        mofile = os.path.join(modir, file)
        modata = np.load(mofile)
        if modata.shape[1] == 266:
            print('data.shape[1] is 266')
            motion = torch.from_numpy(modata).to(torch.float32)
            joints_num = 22
            joints = recover_from_ric266v_grad(motion, joints_num)
        elif modata.shape[1] == 290:
            print('data.shape[1] is 290')
            motion = torch.from_numpy(modata).to(torch.float32)
            joints_num = 24
            joints = recover_from_ricv_norotinit_grad(motion, joints_num)
        elif modata.shape[1] == 319:
            if args.daname == 'FINEDANCE_139CUT':
                print('data.shape[1] is 139')
                motion = torch.from_numpy(modata).to(torch.float32)
                motion = motion[...,:139]
                joints_num = 22
                joints = do_smplxfk(motion, smplx_model)[:,:joints_num,:]
        elif modata.shape[1] == 151:
            if args.daname == 'AISTPP_151':
                print('data.shape[1] is 139')
                motion = torch.from_numpy(modata).to(torch.float32)
                motion = motion[...,:151]
                joints_num = 22
                joints = do_smplxfk(motion.clone()[...,:139], smplx_model)[:,:joints_num,:]
        print('joints', joints.shape)

        mobeatlist = []
        keypoints = joints.detach().cpu().numpy().reshape(-1, joints_num, 3)
        kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
        kinetic_vel = G(kinetic_vel, 5)
        motion_beats = argrelextrema(kinetic_vel, np.less)[0]

        motion_beats = [x for x in motion_beats if (x>3 and x<joints.shape[0]-4)]
        keymotion = None
        for beat in motion_beats:
            if keymotion is None:
                keymotion = modata[beat-4:beat+4]
            else:
                keymotion = np.concatenate([keymotion,  modata[beat-4:beat+4]])
        print('keymotion', keymotion.shape)
        keymotion = keymotion.reshape(-1,8,modata.shape[1])
        print('keymotion', keymotion.shape)
        num += keymotion.shape[0]
        np.save(os.path.join(keymo_savedir, file), keymotion)
        # print('motion_beats', motion_beats[0].shape)
        # print('len(kinetic_vel)', len(kinetic_vel))
    print('num', num)
