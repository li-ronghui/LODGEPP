import glob
import os,sys
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import argparse
from omegaconf import OmegaConf
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from models.edge.args import FineDance_parse_test_opt
# from data.slice import slice_audio
from EDGE import EDGE
import datetime
# from data.audio_extraction.baseline_features import extract as baseline_extract
# from data.audio_extraction.jukebox_features import extract as juke_extract
# from dataset.FineDance_dataset import get_train_test_dict


def render_sample(
    model, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, mode="normal",cons=None
):
    _, cond, wavname, orikey = data_tuple
    assert len(cond.shape) == 3
    # if render_count < 0:
    render_count = len(cond)
    shape = (render_count, model.opt.full_seq_len, model.opt.nfeats)
    cond = cond.cuda().float()
    
    if cond.shape[-1] == 170:
        constraint={}
        constraint["value"] = torch.cat([torch.zeros(cond.shape[0], cond.shape[1], 4).to(cond), cond[:,:,35:]], dim=-1)
        constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
        constraint["mask"][:,0,:] = 1 
        constraint["mask"][:,-1,:] = 1 
        constraint["mask"][:,:,:4] = 0 
    elif cond.shape[-1] == 174:
        constraint={}
        constraint["value"] = torch.zeros_like(cond[:,:,35:])
        constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
        # print("constraint[value].shape", constraint["value"].shape)
        for i in range(cond.shape[0]):
            constraint["value"][i, :4, :] = torch.from_numpy(orikey[i][4:8,:]).to(cond)
            constraint["value"][i, -4:, :] = torch.from_numpy(orikey[i][-8:-4,:]).to(cond)
            constraint["mask"][i, :4,:] = 1 
            constraint["mask"][i, -4:,:] = 1 
        # constraint["mask"][:,:,:4] = 0 
    elif cond.shape[-1] == 35:
        constraint={}
        constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
        constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
        # print("constraint[value].shape", constraint["value"].shape)
        print("orikey.shape", orikey.shape)
        for i in range(cond.shape[0]):
            constraint["value"][i, :4, :] = torch.from_numpy(orikey[i][4:8,:]).to(cond)
            constraint["value"][i, -4:, :] = torch.from_numpy(orikey[i][-8:-4,:]).to(cond)
            constraint["mask"][i, :4,:] = 1 
            constraint["mask"][i, -4:,:] = 1 
        
        # constraint={}
        # constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139).to(cond)
        # for i in range(cond.shape[0]):
        #     constraint["value"][i, 0, 4:] = cons[i][0, :]
        #     constraint["value"][i, -1, 4:] = cons[i][-1, :]
        # constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
        # constraint["mask"][:,0,:] = 1 
        # constraint["mask"][:,-1,:] = 1 
        # constraint["mask"][:,:,:4] = 0 
    else:
        raise("cond fea error!")
    # constraint = None
    model.render_sample(
        shape,
        cond[:render_count],
        None,
        label,
        render_dir,
        name=wavname[:render_count],
        sound=True,
        constraint=constraint,
        mode=mode,            # 这里设置        default is long
        fk_out=fk_out,
        render=render
    )

def render_sample_ori(
    model, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True):
    _, cond, wavname = data_tuple
    assert len(cond.shape) == 3
    if render_count < 0:
        render_count = len(cond)

    if model.opt.full_seq_len == 1024:
        shape = (render_count, 72, model.opt.nfeats)
    else: 
        shape = (render_count, model.opt.full_seq_len, model.opt.nfeats)
    cond = cond.cuda().float()
    model.render_sample(
        shape,
        cond[:render_count],
        None,
        label,
        render_dir,
        name=wavname[:render_count],
        sound=True,
        mode="normal",            # 这里设置        default is long
        fk_out=fk_out,
        render=render
    )


# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

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



def test(opt):
    if opt.fullpt:
        model = torch.load(opt.checkpoint)
        model.eval()
    else:
        model = EDGE(opt)
        model.eval()
    
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    for file in os.listdir(music_dir):
        if not file[:3] in test_list:
            continue

        file_name = file[:-4]
        music_fea = np.load(os.path.join(music_dir, file))
        music_fea = torch.from_numpy(music_fea).cuda().unsqueeze(0)
        music_fea = music_fea.repeat(count, 1, 1)

        mofea   =  np.load(os.path.join(modir, file))[:, :139]
        mofea[:, 4:7] = mofea[:, 4:7] - mofea[:1, 4:7]
        print("mofea.shape", mofea.shape)
        mofea = torch.from_numpy(mofea).cuda().unsqueeze(0)
        mofea = mofea.repeat(count, 1, 1).detach().cpu().numpy()

        all_filenames = [file_name]*count
        # directory for optionally saving the dances for eval
        fk_out = None
        if opt.save_motions:
            fk_out = opt.motion_save_dir

            if not os.path.exists(fk_out):
                os.makedirs((fk_out), exist_ok=False)
            with open(os.path.join(fk_out, 'command.txt'), 'w') as f:
                f.write(command)

        
        if opt.fullpt:
            # data_tuple = None, music_fea, all_filenames
            # print("music_fea.shape", music_fea.shape)
            # render_sample_ori(
            #         model, data_tuple, "test", opt.render_dir, render_count=count, fk_out=fk_out, render=not opt.no_render
            #     )
            if int(length) > 512:
                data_tuple = None, music_fea, all_filenames
                render_sample_ori(
                    model, data_tuple, "test", opt.render_dir, render_count=count, fk_out=fk_out, render=not opt.no_render
                )
            else:
                data_tuple = None, music_fea, all_filenames, mofea
                render_sample(
                            model, data_tuple, "test_inp", opt.render_dir, render_count=count, fk_out=fk_out, render=not opt.no_render, mode='inpaint', cons=mofea,      # all_orikey
                        )
                print("Done")
                print("OK")

        else:
            data_tuple = None, music_fea, all_filenames
            print("music_fea.shape", music_fea.shape)
            model.render_sample_ori(
                    data_tuple, "test", opt.render_dir, render_count=count, fk_out=fk_out, render=not opt.no_render
                )
            
            data_tuple = None, music_fea, all_filenames, mofea
            model.render_sample(
                        data_tuple, "test_inp", opt.render_dir, render_count=count, fk_out=fk_out, render=not opt.no_render, mode='inpaint', cons=mofea,      # all_orikey
                    )
            print("Done")
            print("OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
            required=False,
            default="./experiments/EDGE_aist266/len128_win2_266originloss2/parameters.yaml",
            help="your experimental config file",
        )
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    args.fullpt = False
    args = OmegaConf.create(vars(args))
    cfg_exp = OmegaConf.load(args.cfg)
    cfg = OmegaConf.merge(cfg_exp, args)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y%m%d_%H%M%S')
    fk_out = str(Path(cfg.cfg).parent) + '/infer_' +  time_str
    os.makedirs(fk_out, exist_ok=True)
    
    # opt = FineDance_parse_test_opt()
    command = ' '.join(sys.argv)
    with open(os.path.join(fk_out, 'command.txt'), 'w') as f:
        f.write(command)

    music_dir = "/data2/lrh/dataset/aist/data/origin/30fps/fullset/div_bytime/musics35_30fps_"
    # modir = "/data2/lrh/dataset/fine_dance/div_by_time/motion_fea319_"
    count = 1
    length = cfg.DATA_SETTING.full_seq_len
    # modir = modir + str(length)
    music_dir = music_dir + str(length)


    test(cfg)


'''
CUDA_VISIBLE_DEVICES=2  python tool/infer/infer_edge.py --cond_feadim  35 --full_seq_len 512 --nfeats 139  --keymotion_dir '' --checkpoint experiments/0918_edge139_512/40-1004edge139_512_35-bce_fc3/Full-train-4390.pt  --motion_save_dir experiments/0918_edge139_512/40-1004edge139_512_35-bce_fc3/inf4390 --save_motions --fullpt


CUDA_VISIBLE_DEVICES=2  python infer_fullpt.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/1004_edge139_128/train/Awith_fc_bce2/weights/Full-train-4190.pt  --motion_save_dir experiments/1004_edge139_128/inf4190 --save_motions --fullpt


coarse
CUDA_VISIBLE_DEVICES=2  python infer_coarse.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/edge139_Coarse1024/train/first2/weights/Full-train-630.pt  --motion_save_dir experiments/edge139_Coarse1024/inf --save_motions --fullpt




CUDA_VISIBLE_DEVICES=2  python infer_edge128.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/EDGE_128_139/trainEdge128-2500.pt --motion_save_dir experiments/1003_edge139_128/inf/finedance_128_2500 --save_motions

# accelerate launch  edge.py --cond_feadim  35 --full_seq_len 128 --windows 20 --batch_size 800 --epochs 2500 --nfeats 139 --wandb_pj_name edge139_128 --project experiments/1003_edge139_128/train --exp_name edge_128 --render_dir experiments/1003_edge139_128/renders --keymotion_dir '' --fd-motion-dir /data2/lrh/dataset/fine_dance/origin/motion_feature319mirror/ --checkpoint experiments/EDGE_128_139/trainEdge128-2500.pt





CUDA_VISIBLE_DEVICES=2  python infer_edge128.py --cond_feadim  35 --full_seq_len 512 --nfeats 139  --keymotion_dir '' --checkpoint /data2/lrh/project/dance/long/experiments/0918_edge139_512/train/edge_5122/weights/train-2920.pt  --motion_save_dir experiments/0918_edge139_512/infer --save_motions
'''