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
# sys.path.append('/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion/')
# sys.path.append("/data2/lrh/project/dance/long/")
import models.vqvae as vqvae
import models.c2d_trans_frame as trans
from DanceDiffusion.dld.data.FineDance_dataset import music2genre
from dataset.dataset_gpt_m2d import Genres_fd
# sys.path.append("/data2/lrh/project/dance/long/")
from models.edge.args import FineDance_parse_test_opt
# from data.slice import slice_audio
# from edge import EDGE
# sys.path.append("/data2/lrh/project/dance/long/")
from utils.music_fea35 import extract
from render import ax_to_6v, ax_from_6v

def swap_left_right(data):   
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    
    if data.shape[-1] == 22*6:
        device_ = data.device
        t,c= data.shape
        data = ax_from_6v(data.view(t,22,6))
    elif data.shape[-1] == 52*6:
        t,c= data.shape
        data = ax_from_6v(data.view(t,52,6))
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

def render_sample(
    model, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, mode="normal",cons=None, device=None, Returnfull=False, soft_hint=False
):
    _, cond, wavname, orikey = data_tuple
    assert len(cond.shape) == 3
    # if render_count < 0:
    render_count = len(cond)
    shape = (render_count, model.opt.full_seq_len, model.opt.nfeats)
    cond = cond.to(device).float()
    
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
        if soft_hint == 'gpt':
            constraint={}
            constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
            constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
            # print("constraint[value].shape", constraint["value"].shape)
            # print("orikey.shape", orikey.shape)
            for i in range(cond.shape[0]):
                mocond_1 = orikey[i][:4,:]
                mocond_1[:, 4:7]  = mocond_1[:, 4:7] - mocond_1[:1, 4:7]  
                mocond_2 = orikey[i][-4:,:]
                mocond_2[:, 4:7]  = mocond_2[:, 4:7] - mocond_2[:1, 4:7]  
                constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                constraint["value"][i, 4:-4, :4] = torch.from_numpy(orikey[i][4:-4, :4]).to(cond)
                constraint["value"][i, 4:-4, 7:] = torch.from_numpy(orikey[i][4:-4, 7:]).to(cond)
                constraint["mask"][i, :,:] = 1 
                constraint["mask"][i, :,:] = 1
                # constraint["mask"][i, 4:-4, 4:7] = 0 
                # constraint["mask"][i, 4:-4, 4:7] = 0
        elif soft_hint == 'dod':
            constraint={}
            constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
            constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
            # print("constraint[value].shape", constraint["value"].shape)
            # print("orikey.shape", orikey.shape)
            for i in range(cond.shape[0]):
                mocond_1 = orikey[i][:4,:]
                print("orikey i .shape", orikey[i].shape)
                # mocond_1[:, 4:7]  = mocond_1[:, 4:7] - mocond_1[:1, 4:7]  
                mocond_1[:, 4]  = mocond_1[:, 4] - mocond_1[:1, 4]  
                mocond_1[:, 6]  = mocond_1[:, 6] - mocond_1[:1, 6]  
                mocond_2 = orikey[i][-4:,:]
                # mocond_2[:, 4:7]  = mocond_2[:, 4:7] - mocond_2[:1, 4:7] 
                mocond_2[:, 4]  = mocond_2[:, 4] - mocond_2[:1, 4]  
                mocond_2[:, 6]  = mocond_2[:, 6] - mocond_2[:1, 6]   
                mid = torch.from_numpy(orikey[i][4:-4]).to(cond)
                Mmid_pose = swap_left_right(mid[:, 7:])
                print("Mmid_pose.shape", Mmid_pose.shape)
                Mmid_pose = ax_to_6v(Mmid_pose).view(-1, 132)
                print("Mmid_pose.shape", Mmid_pose.shape)
                # Mmid_pose = Mmid_pose.view(Mmid_pose.shape[0], -1)
                Mmid_root = mid[:, 4:7].clone()
                Mmid_root[:,0] *= -1
                Mmid_foot = torch.cat( [mid[:, 1:2], mid[:, 0:1], mid[:, 3:4], mid[:, 2:3] ] , dim=-1)
                Mmid = torch.cat([Mmid_foot, Mmid_root, Mmid_pose], dim=-1)
                assert mid.shape[0] == 16
                constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                constraint["value"][i, 28:36, :] = mid[:8]
                constraint["value"][i, 60:68, :] = Mmid[:8]
                constraint["value"][i, 156:164, :] = mid[-8:]
                constraint["value"][i, 188:196, :] = Mmid[-8:]
                constraint["mask"][i, :, :] = 0
                constraint["mask"][i, 28:36, :4] =1
                constraint["mask"][i, 28:36, 7:] =1
                constraint["mask"][i, 60:68, :4] =1
                constraint["mask"][i, 60:68, 7:] =1
                constraint["mask"][i, 156:164, :4] =1
                constraint["mask"][i, 156:164, 7:] =1
                constraint["mask"][i, 188:196, :4] =1
                constraint["mask"][i, 188:196, 7:] =1
                constraint["mask"][i, :4,:] = 1 
                constraint["mask"][i, -4:,:] = 1 
        else:
            constraint={}
            constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
            constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
            # print("constraint[value].shape", constraint["value"].shape)
            # print("orikey.shape", orikey.shape)
            for i in range(cond.shape[0]):
                mocond_1 = orikey[i][:4,:]
                mocond_1[:, 4:7]  = mocond_1[:, 4:7] - mocond_1[:1, 4:7]  
                mocond_2 = orikey[i][-4:,:]
                mocond_2[:, 4:7]  = mocond_2[:, 4:7] - mocond_2[:1, 4:7]  
                constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                constraint["mask"][i, :4,:] = 1 
                constraint["mask"][i, -4:,:] = 1 
    else:
        raise("cond fea error!")
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
        render=render,
    )

def render_sample_ori(
    model, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, device=None, Returnfull=False):
    _, cond, wavname = data_tuple
    assert len(cond.shape) == 3
    if render_count < 0:
        render_count = len(cond)

    if model.opt.full_seq_len == 1024:
        shape = (render_count, 72, model.opt.nfeats)
    else: 
        shape = (render_count, model.opt.full_seq_len, model.opt.nfeats)
    cond = cond.to(device).float()
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
        render=render,
        Returnfull=Returnfull
    )


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



def test(opt):
    music_dir = "/data2/lrh/dataset/fine_dance/origin/music"
    # music_dir = "/data2/lrh/project/dance/long/abandon/20240508"
    # music_dir = "/data2/lrh/project/dance/long/eval/music/wav"
    genre_lable = "/data2/lrh/dataset/fine_dance/origin/label_json"
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
    


    if opt.fullpt:
        model_fine = torch.load(opt.checkpoint2)
        model_fine.eval()
    else:
        opt.full_seq_len = length_fi
        model_fine =  EDGE(opt, opt.feature_type, opt.checkpoint2)
        model_fine.eval()
    
    # test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092",  "037", "109", "204", "144"]
    for file in os.listdir(music_dir):
        # if not file[:3] in test_list:
        #     continue

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

            # debug!!!
            # genre = music2genre_lable[file_name[:3]]
            # genre = torch.tensor(Genres_fd[genre]).to(device)
            genre = torch.tensor(15).to(device)         # Breaking0 Locking1 Poping 2 Hantang9 Shenyun10  Chinese15
            print("genre.shape", genre.shape)
            genre = genre.repeat(1)
            print("genre", genre)
            print("genre.shape", genre.shape)

            if current_muidx == 0:
                before_x = None
            else:
                before_x = cls_pred.view(-1)[-4:].unsqueeze(0)
            cls_pred = trans_encoder.sample(noise = noise, feature = music_clip, genre = genre, masked_token_seq = None, before_x=before_x)
        
            print("cls_pred1.shape", cls_pred.shape)
            if cls_pred.shape[1] > opt.gpt_window_size:
                cls_pred = cls_pred[:, :opt.gpt_window_size]
            pred_motion = net.vqvae.my_forward_decoder(cls_pred)[0]
            pred_motion = pred_motion.detach().cpu().numpy()
            print("pred_motion", pred_motion.shape)
            np.save(os.path.join(fk_out, 'gpt_' + file_name + 'g' + str(current_muidx).zfill(3) + '.npy'), pred_motion)
            # sys.exit(0)
            
            music_clip = music_clip.reshape(-1, length_fi, music_clip.shape[-1])
            pred_motion = pred_motion.reshape(-1, length_fi, pred_motion.shape[-1]) # .detach().cpu().numpy()

            if current_muidx == 0:
                music_fea_whole = music_clip
                pred_motion_whole = pred_motion
            else:
                music_fea_whole = torch.cat([music_fea_whole, music_clip] , dim = 0)
                pred_motion_whole = np.concatenate([pred_motion_whole, pred_motion] , axis = 0)
            
            
            
            for l_i in range(music_clip.shape[0]):
                for c_i in range(count):
                    all_filenames.append(file_name[:-4] + file_name + 'g' + str(current_muidx).zfill(3) + '_l' + str(l_i).zfill(3) + '_r' + str(c_i))
            
        print("music_fea_whole", music_fea_whole.shape)
        music_fea_whole = music_fea_whole.repeat(count, 1,1)
        print("music_fea_whole", music_fea_whole.shape)
        print("pred_motion_whole", pred_motion_whole.shape)
        pred_motion_whole = np.repeat(pred_motion_whole, count, axis=0)
        print("pred_motion_whole", pred_motion_whole.shape)

        gpt_time = time.time()
        print("GPT used time", gpt_time - start)

        # data_tuple = None, music_fea_whole, all_filenames, pred_motion_whole
        # if opt.fullpt:
        #     render_sample(
        #             model_fine, data_tuple, "dod", None, render_count=count, fk_out=fk_out, render=not opt.no_render, mode=opt.mode, cons=pred_motion, device=device, Returnfull=True, soft_hint=opt.soft_hint    # all_orikey
        #         )
        # else:
        #     model_fine.render_sample(
        #             data_tuple, "dod", None, render_count=count, fk_out=fk_out, render=not opt.no_render, mode=opt.mode, cons=pred_motion, soft_hint=opt.soft_hint      # all_orikey
        #         )
        # end = time.time()
        # print("used time", end - start)

     

if __name__ == "__main__":
    parser = FineDance_parse_test_opt(condition=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, default="inpaint_soft")
    parser.add_argument("--soft_hint", type=str, default="gpt")
    parser.add_argument("--length1", type=int, default=1024)
    parser.add_argument("--length2", type=int, default=256)
    parser.add_argument("--gpt_window_size", type=int, default=256)
    parser.add_argument("--clip_frames", type=int, default=4)
    # parser.add_argument("--checkpoint1", type=str, default="experiments/edge139_Coarse1024/train/mu2565/weights/Full-train-3320.pt")
    # parser.add_argument("--checkpoint2", type=str, default="experiments/edge139_256/1005edge139_256_35/train/weights/train-2500.pt")
    parser.add_argument('--checkpoint2', type=str, default='experiments/1023edge139_256_35/train/bce_fc2/weights/Full-train-3070.pt')
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding") # 数据集扩大, 增大一倍
    parser.add_argument("--code-dim", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--output-emb-width", type=int, default=1024, help="output embedding width")
    parser.add_argument("--tk_dir", type=str, default="/data2/lrh/project/dance/long/experiments/vqvae/output_vq/clip8_139/FineDance_1007_1024_win128/eval_idx512")
    # good
    parser.add_argument("--resume-trans", type=str, default="/data2/lrh/project/dance/long/experiments/vqvae/gpt/clip8_139/FineDance_1007_1024_win128/ckpt/mask_best_acc.pth")
    parser.add_argument("--resume-pth", type=str, default="/data2/lrh/project/dance/long/experiments/vqvae/output_vq/clip8_139/FineDance_1007_1024_win128/vqvae/f_139/best_recon.pth")
    # compare
    # parser.add_argument("--resume-trans", type=str, default="experiments/vqvae/gpt/clip8_139/FineDance_1009_gpt_win128/ckpt/mask_last.pth")
    # parser.add_argument("--resume-pth", type=str, default="experiments/vqvae/output_vq/clip8_139/FineDance_1011_2048_win128/vqvae/f_139/best_conmit.pth")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=1024, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument("--block-size", type=int, default=2049)
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")

    # opt_new = parser.parse_args()                             
    opt = parser.parse_args()
    opt.do_normalize = False
    opt.feature_dim = opt.nfeats
    print("soft_hint", opt.soft_hint)
    print("hint", opt.hint)
    

    
    # args = option_trains.get_args_parser()
    opt = OmegaConf.create(vars(opt))
    command = ' '.join(sys.argv)

    if not os.path.exists(opt.motion_save_dir):
        os.makedirs((opt.motion_save_dir), exist_ok=False)
    yaml_path = os.path.join(opt.motion_save_dir, 'parameters.yaml')
    OmegaConf.save(opt, yaml_path)
    with open(os.path.join(opt.motion_save_dir, 'command.txt'), 'a') as f:
        f.write(command)

    test(opt)


'''
python  tool/infer/infer_gpt_diffu_whole.py --cond_feadim 35 --nfeats 139 --keymotion_dir ''  --motion_save_dir /data2/lrh/project/dance/long/experiments/infer/clip8_139/FineDance_1007_1024_win128/20240508/Chinese_demos_h5  --save_motions --fullpt --checkpoint2 experiments/1023edge139_256_35/train/bce_fc2/weights/Full-train-3070.pt --gpu 6




python tool/infer/infer_gpt_diffu_whole.py --cond_feadim  35 --nfeats 139  --keymotion_dir ''  --motion_save_dir experiments/infer/clip8_139/FineDance_1007_1024_win128/1107/test1 --save_motions --fullpt


dod
python tool/infer/infer_gpt_diffu.py --cond_feadim  35 --nfeats 139  --keymotion_dir ''  --motion_save_dir experiments/infer/clip8_139/FineDance_1007_1024_win128/1012/test1 --save_motions


coarse
CUDA_VISIBLE_DEVICES=2  python infer_coarse.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/edge139_Coarse1024/train/first2/weights/Full-train-630.pt  --motion_save_dir experiments/edge139_Coarse1024/inf --save_motions --fullpt




CUDA_VISIBLE_DEVICES=2  python infer_edge128.py --cond_feadim  35 --full_seq_len 128 --nfeats 139  --keymotion_dir '' --checkpoint experiments/EDGE_128_139/trainEdge128-2500.pt --motion_save_dir experiments/1003_edge139_128/inf/finedance_128_2500 --save_motions

# accelerate launch  edge.py --cond_feadim  35 --full_seq_len 128 --windows 20 --batch_size 800 --epochs 2500 --nfeats 139 --wandb_pj_name edge139_128 --project experiments/1003_edge139_128/train --exp_name edge_128 --render_dir experiments/1003_edge139_128/renders --keymotion_dir '' --fd-motion-dir /data2/lrh/dataset/fine_dance/origin/motion_feature319mirror/ --checkpoint experiments/EDGE_128_139/trainEdge128-2500.pt





CUDA_VISIBLE_DEVICES=2  python infer_edge128.py --cond_feadim  35 --full_seq_len 512 --nfeats 139  --keymotion_dir '' --checkpoint /data2/lrh/project/dance/long/experiments/0918_edge139_512/train/edge_5122/weights/train-2920.pt  --motion_save_dir experiments/0918_edge139_512/infer --save_motions
'''