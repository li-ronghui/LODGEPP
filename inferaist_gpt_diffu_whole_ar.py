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
import models.vqvae as vqvae
import models.c2d_trans_frame as trans
from dataset.FineDance_dataset_back import music2genre
from dataset.dataset_gpt_m2d import Genres_fd, Genres_aist

# from options.infer_gpt_diff import parse_args_diffusion
from utils.music_fea35 import extract
from render_ske import ax_to_6v, ax_from_6v

from DanceDiffusion.dld.models.get_model import get_module
from DanceDiffusion.dld.data.get_data import get_datasets
from DanceDiffusion.dld.utils.logger import create_logger
from DanceDiffusion.dld.data.utils.audio import slice_audio
from DanceDiffusion.dld.data.utils.audio import extract as extract_music35
from DanceDiffusion.contact_res import contact_res
from omegaconf import OmegaConf

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
    # genre_lable = eval(f"args.DATASET.{dataname.upper()}.MUSIC")
    if 'FINEDANCE' in dataname:
        # genre_lable = "/data/lrh/datasets/fine_dance/origin/label_json"
        genre_lable = "/data2/lrh/dataset/fine_dance/origin/label_json"
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

    # music_dir = music_dir + str(opt.length1)
    device = f'cuda:{args.gpu}'
    print(device)

    fk_out = args.outdir
    # if opt.save_motions:
    #     fk_out = opt.outdir
    #     with open(os.path.join(fk_out, 'command.txt'), 'w') as f:
    #         f.write(command)
    # print(f"fk_out: {fk_out}")
    ##### ---- Network ---- #####
    net = vqvae.HumanVQVAE(args.vq, ## use args to define different parameters in different quantizers
                        args.vq.nb_code,
                        args.vq.code_dim,
                        args.vq.output_emb_width,
                        args.vq.down_t,
                        args.vq.stride_t,
                        args.vq.width,
                        args.vq.depth,
                        args.vq.dilation_growth_rate,
                        args.vq.vq_act,
                        args.vq.vq_norm)

    trans_encoder = trans.c2d_Transformer(args=args.gpt,
                                    num_vq=args.gpt.nb_code, 
                                    window_size=args.gpt.DATA_SETTING.full_seq_len,
                                    embed_dim=args.gpt.embed_dim_gpt, 
                                    clip_dim=args.gpt.clip_dim, 
                                    block_size=args.gpt.block_size, 
                                    num_layers=args.gpt.num_layers, 
                                    n_head=args.gpt.n_head_gpt, 
                                    drop_out_rate=args.gpt.drop_out_rate, 
                                    fc_rate=args.gpt.ff_rate)


    if args.gpt.resume_trans is not None: # 有才加载
        print ('loading transformer checkpoint from {}'.format(args.gpt.resume_trans))
        ckpt = torch.load(args.gpt.resume_trans, map_location='cpu')
        trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.to(device)

    if args.vq.resume_pth : # 有vq-vae才加载
        print('loading checkpoint from {}'.format(args.vq.resume_pth))
        ckpt = torch.load(args.vq.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.to(device)
    

    dataset = get_datasets(args.diff, logger=logger, phase="test")[0]

    print('DATA_SETTING', args.diff.DATA_SETTING.nfeats)
    args.diff.model.DanceDecoder.params.nfeats = args.diff.DATA_SETTING.nfeats
    args.diff.model.DanceDecoder.params.seq_len = args.diff.DATA_SETTING.full_seq_len
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
        number = int(music_fea_full.shape[0]//args.length1)
        music_fea_full = music_fea_full[:number * args.length1]
        
        start = time.time()
        all_filenames = []
        point = 0
        while(True):
            if point+args.length1 > music_fea_full.shape[0]:
                break
            music_clip = music_fea_full[point: point + args.length1]
            music_clip = torch.from_numpy(music_clip).unsqueeze(0).to(device).to(dtype=torch.float32)
            noise = torch.randn(music_clip.shape[0], 256).to(music_clip)

        # for current_muidx in range(number):
        #     music_clip = music_fea_full[current_muidx*args.length1: (current_muidx+1)*args.length1]
        #     music_clip = torch.from_numpy(music_clip).unsqueeze(0).to(device).to(dtype=torch.float32)
            # music_clip = music_clip.repeat(count, 1, 1)
            # noise = torch.randn(music_clip.shape[0], args.length1).to(music_clip)
            print("name is {}, current,{}".format(file_name, point) )

            # genre = music2genre_lable[file_name[:3]]
            # genre = torch.tensor(Genres_fd[genre]).to(device)


            if 'FINEDANCE' in dataname:
                genre = music2genre_lable[file_name.split(".")[0]]
                genre = torch.tensor(Genres_fd[genre]).to(device)
            elif 'AISTPP' in dataname:
                genre = file_name.split('_')[0]
                genre = torch.tensor(Genres_aist[genre]).to(device)
            
            print("genre.shape", genre.shape)
            genre = genre.repeat(1)
            print("genre", genre)
            print("genre.shape", genre.shape)

            if point == 0:
                before_x = None
            else:
                before_x = cls_pred.view(-1)[-4:].unsqueeze(0)
            cls_pred = trans_encoder.sample(noise = noise, feature = music_clip, genre = genre, masked_token_seq = None, before_x=before_x)
        
            print("cls_pred1.shape", cls_pred.shape)
            if cls_pred.shape[1] >args.gpt.DATA_SETTING.full_seq_len:
                cls_pred = cls_pred[:, :args.gpt.DATA_SETTING.full_seq_len]
            print("cls_pred2.shape", cls_pred.shape)
            print('cls_pred', cls_pred)

            if point == 0:
                music_fea_whole = music_clip.squeeze(0)
                pred_motion_idx = cls_pred[0].detach().cpu().numpy().tolist()
            else:
                music_fea_whole = torch.cat([music_fea_whole, music_clip.squeeze(0)[16:]] , dim = 0)
                pred_motion_idx = pred_motion_idx + cls_pred[0,4:].detach().cpu().numpy().tolist()
            print("music_fea_whole", music_fea_whole.shape)
            print("pred_motion_idx", len(pred_motion_idx))
            print("pred_motion_idx", pred_motion_idx)
            
            for l_i in range(music_clip.shape[0]):
                for c_i in range(count):
                    all_filenames.append(file_name[:-4] + file_name + 'g' + str(point).zfill(3) + '_l' + str(l_i).zfill(3) + '_r' + str(c_i))
            point = point + args.length1 - 16       # 4 idx corresponding 16 frames

        pred_motion_idx = torch.tensor(pred_motion_idx).unsqueeze(0).to(device)
        print('pred_motion_idx', pred_motion_idx.shape)
        print('pred_motion_idx', pred_motion_idx)
        pred_motion = net.vqvae.my_forward_decoder(pred_motion_idx)[0]
        pred_motion = pred_motion.detach().cpu().numpy()
        print("pred_motion", pred_motion.shape)

        if args.vq.Norm:
            saved_motion = (pred_motion * std266) + mean266
            np.save(os.path.join(fk_out, 'gpt_' + file_name + '.npy'), saved_motion)
        else:
            saved_motion = pred_motion.copy()
            np.save(os.path.join(fk_out, 'gpt_' + file_name + '.npy'), saved_motion)
            pred_motion = (pred_motion-mean266) / std266
        
        print("music_fea_whole", music_fea_whole.shape)
        print("pred_motion", pred_motion.shape)

        T_length = pred_motion.shape[0]
        print('T_length',T_length)
        # Firstcond = False
        gpt_time = time.time()
        print("GPT used time", gpt_time - start)
        point = 0
        while(True):
            if point+args.length2 > T_length:
                np.save(os.path.join(fk_out, file_name + '.npy'), outmotion.detach().cpu().numpy())
                break
            music_diff_clip = music_fea_whole[point:point+128].unsqueeze(0)
            motion_diff_clip = np.expand_dims(pred_motion[point:point+128], 0)
            if point==0:
                Firstcond = None
            else:
                Firstcond = (outmotion[-4:] - mean266 ) / std266

            data_tuple = None, music_diff_clip, all_filenames, motion_diff_clip
            outmotion_clip = model_fine.render_sample(
                        data_tuple, "dod", None, render_count=count, fk_out=fk_out, render=True, setmode=args.setmode, cons=pred_motion, Returnfull=True, soft_hint=args.diff.soft_hint, Firstcond=Firstcond, genre=None,   # all_orikey
                    )
            if point==0:
                outmotion = outmotion_clip.squeeze(0)
            else:
                outmotion = torch.cat([outmotion, outmotion_clip.squeeze(0)[4:]], dim=0)
            print('outmotion_clip', outmotion_clip.shape)
            print('outmotion', outmotion.shape)
            point = point + args.length2 - 4
        end = time.time()
        print("used time", end - start)
        # sys.exit(0)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgvq", type=str,
            required=False,
            default="./experiments290/vqvae/AISTPP_290/0617_resume2_win64_Norm_bc1024/train_20240617_030343.yaml",
            help="config file",
        )
    parser.add_argument("--cfggpt", type=str,
            required=False,
            default="./experiments/gpt/AISTPP_266/mo266_0608_win64_Norm_bc1024_flat/traingpt_20240609_064827.yaml",
            help="config file",
        )
    parser.add_argument("--cfgdiff", type=str,
            required=False,
            default="./DanceDiffusion/experiments/Local_Module/0524_Norm_loss266origin_diff_bc768/config_2024-06-05-03-10-55_train.yaml",
            help="config file",
        )
    parser.add_argument("--checkpoint2", type=str,
            default="DanceDiffusion/experiments/Local_Module/0524_Norm_loss266origin_diff_bc768/checkpoints/epoch=1999.ckpt")
    parser.add_argument("--assets", type=str,
            required=False,
            default="/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/configs/45assets.yaml",
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
    parser.add_argument("--soft_hint", type=str, default="gpt")
    parser.add_argument('--cache_features', dest='cache_features', action='store_false', help='布尔标志，默认为 True')
    args = parser.parse_args()
    # conf = OmegaConf.create()
    args = OmegaConf.create(vars(args))
    cfgvq = OmegaConf.load(args.cfgvq)
    cfggpt = OmegaConf.load(args.cfggpt)
    cfgdiff = OmegaConf.load(args.cfgdiff)
    cfgdiff.soft_hint = args.soft_hint
    cfg_assets = OmegaConf.load(args.assets)
    conf = OmegaConf.create({
        "vq": cfgvq,
        "gpt": cfggpt,
        "diff": cfgdiff,
    })
    cfg = OmegaConf.merge(conf, args)
    args = OmegaConf.merge(cfg, cfg_assets)
    
    # args.diff.Name = "demo--" + cfg.NAME
    
    current_time = datetime.now()

    assert args.vq.dataname[0] == args.gpt.dataname[0]
    dataname = args.vq.dataname[0]
    print('dataname', dataname)
    print(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.mean"))
    mean266 = np.load(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.mean"))
    std266 = np.load(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.std"))
    # if 'AISTPP' in dataname:
    #     args.length1 = 256
    # else:
    args.length1 = args.gpt.DATA_SETTING.full_seq_len * 4
    args.length2 = args.diff.length2 = 128

    # args.diff.NAME = args.outdir
    args.diff.Name = "demo--" + args.diff.NAME
    logger, _ = create_logger(args.diff, phase="demo")

    
    # opt = OmegaConf.create(vars(opt))
    command = ' '.join(sys.argv)
    if not os.path.exists(args.outdir):
        os.makedirs((args.outdir), exist_ok=False)
    yaml_path = os.path.join(args.outdir, 'parameters.yaml')
    OmegaConf.save(args, yaml_path)
    with open(os.path.join(args.outdir, 'command.txt'), 'a') as f:
        f.write(command)

    os.system("cp -r DanceDiffusion/dld/models/architectures " +  args.outdir)

    test(args)


'''
python inferaist_gpt_diffu_whole_ar.py --outdir abandon/aist_gpttrans_ar5
'''