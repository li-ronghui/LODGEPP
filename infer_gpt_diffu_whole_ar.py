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

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())
import models.vqvae as vqvae
import models.c2d_trans_frame as trans
from dataset.FineDance_dataset_back import music2genre
from dataset.dataset_gpt_m2d import Genres_fd

from options.infer_gpt_diff import parse_args_diffusion
from utils.music_fea35 import extract
from render_ske import ax_to_6v, ax_from_6v

from DanceDiffusion.dld.models.get_model import get_module
from DanceDiffusion.dld.data.get_data import get_datasets
from DanceDiffusion.dld.utils.logger import create_logger
from DanceDiffusion.dld.data.utils.audio import slice_audio
from DanceDiffusion.dld.data.utils.audio import extract as extract_music35
from DanceDiffusion.contact_res import contact_res

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



def test(opt,cfg):
    # music_dir = "/data2/lrh/dataset/fine_dance/div_by_time/music_fea35edge_"
    music_dir = "/data/lrh/datasets/fine_dance/origin/music"
    genre_lable = "/data/lrh/datasets/fine_dance/origin/label_json"
    music2genre_lable = music2genre(genre_lable)
    count = 1
    length_co = opt.length1
    length_fi = opt.length2
    # music_dir = music_dir + str(opt.length1)
    device = f'cuda:{opt.gpu}'
    print(device)

    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir
        with open(os.path.join(fk_out, 'command.txt'), 'w') as f:
            f.write(command)
    # print(f"fk_out: {fk_out}")
    ##### ---- Network ---- #####
    net = vqvae.HumanVQVAE(opt, ## use args to define different parameters in different quantizers
                        opt.nb_code,
                        opt.code_dim,
                        opt.output_emb_width,
                        opt.down_t,
                        opt.stride_t,
                        opt.width,
                        opt.depth,
                        opt.dilation_growth_rate,
                        opt.vq_act,
                        opt.vq_norm)

    trans_encoder = trans.c2d_Transformer(args=opt,
                                    num_vq=opt.nb_code, 
                                    window_size=opt.gpt_window_size,
                                    embed_dim=opt.embed_dim_gpt, 
                                    clip_dim=opt.clip_dim, 
                                    block_size=opt.block_size, 
                                    num_layers=opt.num_layers, 
                                    n_head=opt.n_head_gpt, 
                                    drop_out_rate=opt.drop_out_rate, 
                                    fc_rate=opt.ff_rate)


    if opt.resume_trans is not None: # 有才加载
        print ('loading transformer checkpoint from {}'.format(opt.resume_trans))
        ckpt = torch.load(opt.resume_trans, map_location='cpu')
        trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.to(device)

    if opt.resume_pth : # 有vq-vae才加载
        print('loading checkpoint from {}'.format(opt.resume_pth))
        ckpt = torch.load(opt.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.to(device)
    

    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    model_fine = get_module(cfg, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.checkpoint2))
    state_dict = torch.load(cfg.checkpoint2,
                            map_location="cpu")["state_dict"]

    model_fine.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model_fine.to(device)
    model_fine.eval()
    
    # test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092",  "037", "109", "204", "144"]
    # test_list = ['098']
    for file in os.listdir(music_dir):
        if not file[:3] in test_list:
            continue

        file_name = file[:-4]
        mufile = os.path.join(music_dir, file)
        if opt.cache_features:
            music_fea_full = np.load(mufile)
        else:
            music_fea_full,_ = extract(fpath=mufile)
        
        print("music_fea_full", music_fea_full.shape)
        number = int(music_fea_full.shape[0]//opt.length1)
        music_fea_full = music_fea_full[:number * opt.length1]
        
        start = time.time()
        all_filenames = []
        for current_muidx in range(number):
            music_clip = music_fea_full[current_muidx*opt.length1: (current_muidx+1)*opt.length1]
            music_clip = torch.from_numpy(music_clip).unsqueeze(0).to(device).to(dtype=torch.float32)
            # music_clip = music_clip.repeat(count, 1, 1)
            noise = torch.randn(music_clip.shape[0], 256).to(music_clip)
            print("name is {}, current,{}".format(file_name, current_muidx) )

            genre = music2genre_lable[file_name[:3]]
            genre = torch.tensor(Genres_fd[genre]).to(device)
            print("genre.shape", genre.shape)
            genre = genre.repeat(1)
            print("genre", genre)
            print("genre.shape", genre.shape)

            if current_muidx == 0:
                before_x = None
            else:
                before_x = cls_pred.view(-1)[-4:].unsqueeze(0)
            cls_pred = trans_encoder.sample(noise = noise, feature = music_clip, genre = genre, masked_token_seq = None, before_x=before_x)
        
            # print("cls_pred1.shape", cls_pred.shape)
            if cls_pred.shape[1] > opt.gpt_window_size:
                cls_pred = cls_pred[:, :opt.gpt_window_size]
            print('cls_pred', cls_pred)
            pred_motion = net.vqvae.my_forward_decoder(cls_pred)[0]
            pred_motion = pred_motion.detach().cpu().numpy()
            print("pred_motion", pred_motion.shape)
            # print(f"fk_out: {fk_out}, file_name: {file_name}, current_muidx: {current_muidx}")
            # if fk_out is None or file_name is None:
            #     raise ValueError("fk_out or file_name is None!")

            saved_motion = (pred_motion * std266) + mean266
            np.save(os.path.join(fk_out, 'gpt_' + file_name + 'g' + str(current_muidx).zfill(3) + '.npy'), saved_motion)
            # sys.exit(0)
            
            music_clip = music_clip.reshape(-1, length_fi, music_clip.shape[-1])
            pred_motion = pred_motion.reshape(-1, length_fi, pred_motion.shape[-1])

            if current_muidx == 0:
                music_fea_whole = music_clip
                pred_motion_whole = pred_motion
            else:
                music_fea_whole = torch.cat([music_fea_whole, music_clip] , dim = 0)
                pred_motion_whole = np.concatenate([pred_motion_whole, pred_motion] , axis = 0)
            
            for l_i in range(music_clip.shape[0]):
                for c_i in range(count):
                    all_filenames.append(file_name[:-4] + file_name + 'g' + str(current_muidx).zfill(3) + '_l' + str(l_i).zfill(3) + '_r' + str(c_i))
        
        music_fea_whole = music_fea_whole.reshape(-1, music_fea_whole.shape[-1])
        pred_motion_whole = pred_motion_whole.reshape(-1, pred_motion_whole.shape[-1])
        print("music_fea_whole", music_fea_whole.shape)
        print("pred_motion_whole", pred_motion_whole.shape)

        T_length = pred_motion_whole.shape[0]
        print('T_length',T_length)
        # Firstcond = False
        gpt_time = time.time()
        print("GPT used time", gpt_time - start)
        point = 0
        while(True):
            if point+128 > T_length:
                np.save(os.path.join(fk_out, file_name + '.npy'), outmotion.detach().cpu().numpy())
                break
            music_diff_clip = music_fea_whole[point:point+128].unsqueeze(0)
            motion_diff_clip = np.expand_dims(pred_motion_whole[point:point+128], 0)
            if point==0:
                Firstcond = None
            else:
                Firstcond = outmotion[-4:]

            data_tuple = None, music_diff_clip, all_filenames, motion_diff_clip
            outmotion_clip = model_fine.render_sample(
                        data_tuple, "dod", None, render_count=count, fk_out=fk_out, render=not opt.no_render, setmode="inpaint_soft_ddim", cons=pred_motion, Returnfull=True, soft_hint=opt.soft_hint, Firstcond=Firstcond, genre=None,   # all_orikey
                    )
            if point==0:
                outmotion = outmotion_clip.squeeze(0)
            else:
                outmotion = torch.cat([outmotion, outmotion_clip.squeeze(0)[4:]], dim=0)
            print('outmotion_clip', outmotion_clip.shape)
            print('outmotion', outmotion.shape)
            point = point + 124
        end = time.time()
        print("used time", end - start)
        # sys.exit(0)
     

if __name__ == "__main__":
    parser, cfg = parse_args_diffusion(phase="demo", condition=True)

    mean266 = np.load(os.path.join('/data/lrh/datasets/fine_dance/gound/smplx_mofea266/noMirror_Norm/', 'Mean.npy'))
    std266 = np.load(os.path.join('/data/lrh/datasets/fine_dance/gound/smplx_mofea266/noMirror_Norm/', 'Std.npy'))
    
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.length1 = 512
    cfg.length2 = 128
    cfg.Name = "demo--" + cfg.NAME
    cfg.checkpoint2 = '/data/lrh/project/dance/LongV2/LongV2Exp1/DanceDiffusion/experiments/Local_Module/0406_newNorm_128len_338_diff_bc768/checkpoints/epoch=2999.ckpt'
    # cfg.checkpoint2 = '/data/lrh/project/dance/LongV2/LongV2Exp1/DanceDiffusion/experiments/Local_Module/0406FineTune_newNorm_128len_338_diff_bc768/checkpoints/epoch=1799.ckpt'
    # cfg.checkpoint2 = '/data/lrh/project/dance/LongV2/LongV2Exp1/DanceDiffusion/experiments/Local_Module/0406FineTune_footloss_aloss_newNorm_128len_338/checkpoints/epoch=399.ckpt'
    logger, final_output_dir = create_logger(cfg, phase="demo")

    # opt_new = parser.parse_args()                             
    opt = parser.parse_args()
    opt.do_normalize = False
    opt.feature_dim = opt.nfeats
    print("soft_hint", opt.soft_hint)
    print("hint", opt.hint)

    # setmode = "inpaint_soft"
    setmode = "inpaint_soft_ddim"
    
    opt = OmegaConf.create(vars(opt))
    command = ' '.join(sys.argv)
    if not os.path.exists(opt.motion_save_dir):
        os.makedirs((opt.motion_save_dir), exist_ok=False)
    yaml_path = os.path.join(opt.motion_save_dir, 'parameters.yaml')
    OmegaConf.save(opt, yaml_path)
    with open(os.path.join(opt.motion_save_dir, 'command.txt'), 'a') as f:
        f.write(command)

    test(opt,cfg)


'''



python tool/infer/infer_gpt_diffu_whole.py --cond_feadim  35 --nfeats 139  --keymotion_dir ''  --motion_save_dir experiments/infer/clip8_139/FineDance_1007_1024_win128/1107/test1 --save_motions --fullpt


dod
python tool/infer/infer_gpt_diffu.py --cond_feadim  35 --nfeats 139  --keymotion_dir ''  --motion_save_dir experiments/infer/clip8_139/FineDance_1007_1024_win128/1012/test1 --save_motions


coarse
CUDA_VISIBLE_DEVICES=2  python infer_coarse.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/edge139_Coarse1024/train/first2/weights/Full-train-630.pt  --motion_save_dir experiments/edge139_Coarse1024/inf --save_motions --fullpt




CUDA_VISIBLE_DEVICES=2  python infer_edge128.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/EDGE_128_139/trainEdge128-2500.pt --motion_save_dir experiments/1003_edge139_128/inf/finedance_128_2500 --save_motions

# accelerate launch  edge.py --cond_feadim  35 --full_seq_len 128 --windows 20 --batch_size 800 --epochs 2500 --nfeats 139 --wandb_pj_name edge139_128 --project experiments/1003_edge139_128/train --exp_name edge_128 --render_dir experiments/1003_edge139_128/renders --keymotion_dir '' --fd-motion-dir /data2/lrh/dataset/fine_dance/origin/motion_feature319mirror/ --checkpoint experiments/EDGE_128_139/trainEdge128-2500.pt





CUDA_VISIBLE_DEVICES=2  python infer_edge128.py --cond_feadim  35 --full_seq_len 512 --nfeats 139  --keymotion_dir '' --checkpoint /data2/lrh/project/dance/long/experiments/0918_edge139_512/train/edge_5122/weights/train-2920.pt  --motion_save_dir experiments/0918_edge139_512/infer --save_motions
'''