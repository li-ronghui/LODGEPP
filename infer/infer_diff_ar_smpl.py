"""
Use auto regressive manner to generate long dance
"""
from email.errors import FirstHeaderLineIsContinuationDefect
import glob
import pickle

import os,sys
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import argparse
import time
from omegaconf import OmegaConf
from datetime import datetime

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())
sys.path.append('/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion/')
# from options.infer_gpt_diff import parse_args_diffusion

from DanceDiffusion.dld.data.utils.audio import extract
from DanceDiffusion.dld.data.FineDance_dataset import music2genre
from DanceDiffusion.dld.models.get_model import get_module
from DanceDiffusion.dld.data.get_data import get_datasets
from DanceDiffusion.dld.utils.logger import create_logger
from DanceDiffusion.dld.data.utils.audio import slice_audio
from DanceDiffusion.dld.data.utils.audio import extract as extract_music35
# from contact_res import contact_res
from omegaconf import OmegaConf
from render import ax_to_6v, ax_from_6v

print(torch.cuda.device_count())
print(os.getenv('CUDA_VISIBLE_DEVICES'))




# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])
test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092",  "037", "109", "204", "144", "120"]
# test_list = ["063", "144"]

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0

def load_modata(keymopath, device):
    if keymopath[-3:] == 'pkl':
        pkl_data = pickle.load(open(keymopath, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        T,C = smpl_poses.shape
        smpl_poses = smpl_poses.reshape(T, -1, 3)
        smpl_poses = torch.from_numpy(smpl_poses).to(device)
        smpl_rot = ax_to_6v(smpl_poses ).reshape(T, -1)
        smpl_trans = torch.from_numpy(pkl_data["smpl_trans"]).to(device)
        assert smpl_rot.shape[1] == 132

        modata = torch.cat( [ smpl_trans, smpl_rot], dim=1)
    elif keymopath[-3:] == 'npy':
        modata  = np.load(keymopath)

    print(modata.shape)

    return modata



def test(args):
    music_dir = eval(f"args.DATASET.{dataname.upper()}.MUSIC")
    motion_dir = eval(f"args.DATASET.{dataname.upper()}.MOTION")
    if 'FINEDANCE' in dataname:
        genre_lable = eval(f"args.DATASET.{dataname.upper()}.LABEL")
        music2genre_lable = music2genre(genre_lable)
        test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    elif 'AISTPP' in dataname:
        test_list = []
        test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_test.txt', 'r')
        for fname in test_file.readlines():
            test_list.append(fname.strip())
        test_file.close()
                              
        test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_val.txt', 'r')
        for fname in test_file.readlines():
            test_list.append(fname.strip())
        test_file.close()

    
    count = 1
    device = f'cuda:{args.gpu}'
    print(device)
    fk_out = args.outdir


    dataset = get_datasets(args.diff, logger=logger, phase="test")[0]

    print('DATA_SETTING', args.diff.DATA_SETTING.nfeats)
    args.diff.model.DanceDecoder.params.nfeats = args.diff.DATA_SETTING.nfeats
    args.diff.model.DanceDecoder.params.seq_len = args.diff.DATA_SETTING.full_seq_len
    args.diff.model.DanceDiscriminator.params.nfeats = args.diff.DATA_SETTING.nfeats
    args.diff.model.DanceDiscriminator.params.seq_len = args.diff.DATA_SETTING.full_seq_len
    args.diff.model.diffusion.params.repr_dim = args.diff.DATA_SETTING.nfeats
    args.diff.model.diffusion.params.horizon = args.diff.DATA_SETTING.full_seq_len
    model_fine = get_module(args.diff, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.checkpoint2))
    state_dict = torch.load(args.checkpoint2,
                            map_location="cpu")["state_dict"]

    model_fine.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(args.diff.model.model_type))
    model_fine.to(device)
    model_fine.eval()

    for file in os.listdir(music_dir):
        if 'FINEDANCE' in dataname:
            if not file[:3] in test_list:
                continue
        else:
            if not file.split('.')[0] in test_list:
                continue

        file_name = file[:-4]
        mufile = os.path.join(music_dir, file)
        # args.diff.cache_features = args.cache_features
        if args.cache_features:
            music_fea_full = np.load(mufile)
        else:
            music_fea_full,_ = extract(fpath=mufile)
        
        print("music_fea_full", music_fea_full.shape)
        
        start = time.time()
        all_filenames = []
        point = 0
        GTmotion = np.load(os.path.join(motion_dir, file))
        GTmotion = np.tile(GTmotion, (10, 1))
        # GTmotion = torch.from_numpy(GTmotion).to(device).to(dtype=torch.float32)
        # GTmotion = normalizer.normalize(GTmotion) # (GTmotion - mean ) / std

        T_length = music_fea_full.shape[0]
        print('T_length',T_length)
        point = 0
        while(True):
            if point+args.length > T_length:
                np.save(os.path.join(fk_out, file_name + '.npy'), outmotion.detach().cpu().numpy())
                break
            
            music_diff_clip = music_fea_full[point:point + args.length]
            music_beat = music_diff_clip[:,-1]
            music_peak = music_diff_clip[:,-2]
            print('music_beat', music_beat)
            print('music_peak', music_peak)
            ind_beat = [index for index, value in enumerate(music_beat) if value >0.5]
            ind_peak = [index for index, value in enumerate(music_peak) if value >0.5]
            print('ind_beat', ind_beat)
            print('ind_peak', ind_peak)
            
            music_diff_clip = torch.from_numpy(music_diff_clip).unsqueeze(0).to(device).to(dtype=torch.float32)
            gt_motion_clip = GTmotion[point:point + args.length]
            motion_diff_clip = torch.zeros([music_diff_clip.shape[0], music_diff_clip.shape[1], args.diff.DATA_SETTING.nfeats]).to(music_diff_clip).detach().cpu().numpy()

            # lastbeat = 4
            # for onebeat in ind_beat:
            #     if onebeat> 12 and onebeat<120 and onebeat>(lastbeat+8):
            #         print('motion_diff_clip[:, onebeat-4:onebeat+4]', motion_diff_clip[:, onebeat-4:onebeat+4].shape)
            #         print('gt_motion_clip[:, onebeat-4:onebeat+4]', gt_motion_clip[onebeat-4:onebeat+4].shape)
            #         motion_diff_clip[:, onebeat-4:onebeat+4] = gt_motion_clip[onebeat-4:onebeat+4]
            #         lastbeat = onebeat
            # motion_diff_clip = [motion_diff_clip, ind_beat]

            if point==0:
                Firstcond = None
            else:
                print('outmotion', outmotion.shape)
                Firstcond = normalizer.normalize(outmotion[-4:].clone() )  #(outmotion[-4:] - mean ) / std

            data_tuple = None, music_diff_clip, all_filenames, motion_diff_clip
            outmotion_clip = model_fine.render_sample(
                        data_tuple, args.diff.soft_hint, None, render_count=count, fk_out=fk_out, render=True, setmode=args.setmode, cons=None, Returnfull=True, soft_hint=args.diff.soft_hint, Firstcond=Firstcond, genre=None,   # all_orikey
                    )
            if point==0:
                outmotion = outmotion_clip.squeeze(0)
            else:
                outmotion = torch.cat([outmotion, outmotion_clip.squeeze(0)[4:]], dim=0)
            print('outmotion_clip', outmotion_clip.shape)
            print('outmotion', outmotion.shape)
            point = point + args.length - 4
        end = time.time()
        print("used time", end - start)
        # sys.exit(0)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfgdiff", type=str,
            required=False,
            default="experiments/Local_Module/FineDance_relative_Norm_GenreDis_bc190/config_2024-03-05-22-09-52_train.yaml",
            help="config file",
        )
    parser.add_argument("--checkpoint2", type=str,
            default="experiments/Local_Module/FineDance_relative_Norm_GenreDis_bc190/checkpoints/epoch=1599.ckpt")
    parser.add_argument("--assets", type=str,
            required=False,
            default="/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion/configs/45assets.yaml",
            help="config file for asset paths",
        )
    parser.add_argument("--outdir", type=str,
            required=False,
            default="./tempoutdir",
            help="config file for asset paths",
        )
    parser.add_argument("--setmode", type=str,
            required=False,
            default="inpaint_soft_ddim",        # inpaint_soft_loop, inpaint_key
        )
    parser.add_argument("--gpu", type=int,
            required=False,
            default=5,
        )
    parser.add_argument("--soft_hint", type=str, default="autore")
    parser.add_argument('--cache_features', dest='cache_features', action='store_false', help='布尔标志，默认为 True')
    args = parser.parse_args()
    # conf = OmegaConf.create()
    args = OmegaConf.create(vars(args))
    cfgdiff = OmegaConf.load(args.cfgdiff)
    cfgdiff.soft_hint = args.soft_hint
    cfg_assets = OmegaConf.load(args.assets)
    conf = OmegaConf.create({
        "diff": cfgdiff,
    })
    cfg = OmegaConf.merge(conf, args)
    args = OmegaConf.merge(cfg, cfg_assets)
    
    # args.diff.Name = "demo--" + cfg.NAME
    
    current_time = datetime.now()
    print(args.keys())
    print(args.diff.keys())
    dataname = args.diff.TEST.DATASETS[0]
    print('dataname', dataname)
    normalizer = torch.load(eval(f"args.DATASET.{dataname.upper()}.normalizer"))  
    # mean = np.load(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.mean"))
    # std= np.load(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.std"))
    # if 'AISTPP' in dataname:
    #     args.length1 = 256
    # else:
    #     args.length1 = 512
    args.length = args.diff.length2 = args.diff.DATA_SETTING.full_seq_len

    args.diff.NAME = args.outdir
    args.diff.Name = "demo--" + args.diff.NAME
    logger, _ = create_logger(args.diff, phase="demo")

    
    # opt = OmegaConf.create(vars(opt))
    command = ' '.join(sys.argv)
    if not os.path.exists(args.outdir):
        os.makedirs((args.outdir), exist_ok=False)
    yaml_path = os.path.join(args.outdir, 'parameters.yaml')
    OmegaConf.save(args, yaml_path)
    with open(os.path.join(args.outdir, 'command.txt'), 'a') as f:
        f.write('\n')
        f.write(command)

    test(args)


'''
python inferaist_gpt_diffu_whole_ar.py --outdir abandon/aist_gpttrans_ar5
'''