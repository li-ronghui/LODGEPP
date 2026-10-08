
import os
import json
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_vq
# import utils.eval_trans as eval_trans
# from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
import argparse
from omegaconf import OmegaConf
from datetime import datetime
sys.path.append('/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion')
 


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str,
        required=False,
        default="experiments/vqvae/AISTPP_266/mo266_0607_win64_Norm/train_20240607_113600.yaml",
        help="config file",
    )
parser.add_argument("--assets", type=str,
        required=False,
        default="/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/configs/45assets.yaml",
        help="config file for asset paths",
    )
parser.add_argument("--resume-pth", type=str,
        required=False,
        default="/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/experiments/vqvae/AISTPP_266/mo266_0607_win64_Norm/vqvae/f_266_6590/best_recon.pth",
        help="config file for asset paths",
    )
parser.add_argument('--gpu', type=int, default=1, help='gpu')

args = parser.parse_args()
args = OmegaConf.create(vars(args))
cfg_exp = OmegaConf.load(args.cfg)
cfg_assets = OmegaConf.load(args.assets)
cfg = OmegaConf.merge(cfg_exp, cfg_assets)
args = OmegaConf.merge(cfg, args)
current_time = datetime.now()

##### ---- Exp dirs ---- #####
# args = option_vq.get_args_parser()
device = f"cuda:{args.gpu}"
torch.manual_seed(args.seed)
# sys.exit(0)
outdir_root = args.out_dir
out_dir = os.path.join(outdir_root, f'{args.exp_name}', "eval")
out_dir_motion = os.path.join(out_dir, "eval_motion")
out_dir_codeidx = os.path.join(out_dir, "eval_idx")
os.makedirs(out_dir, exist_ok = True) # 创建 out-dir 文件夹, 用于保存 loger 数据
os.makedirs(out_dir_motion, exist_ok = True) # 创建 out_dir_motion 文件夹, 保存encoder又decoder后的motioin数据。用于检查可视化效果
os.makedirs(out_dir_codeidx, exist_ok = True) # 创建 out-dir 文件夹, 用于保存 loger 数据
print('out_dir', out_dir)

dataname = args.dataname[0]
print('dataname', dataname)
if args.Norm:
    if '139' in dataname or '151' in dataname:
        normalizer = torch.load(eval(f"args.DATASET.{dataname.upper()}.normalizer"))
    else:
        print(eval(f"args.DATASET.{dataname.upper()}"))
        NormMean = np.load(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.mean"))
        NormStd = np.load(eval(f"args.DATASET.{dataname.upper()}.normalizer.params.std"))


##### ---- Logger ---- #####
time_str = current_time.strftime('%Y%m%d_%H%M%S')
OmegaConf.save(args, os.path.join(out_dir, 'test_' + time_str + '.yaml'))

logger = utils_model.get_logger(out_dir) 
writer = SummaryWriter(out_dir) 


##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)


if args.resume_pth : # 有才加载
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(device)

Loss = losses.ReConsLoss(args.recons_loss,args)
length_clip = 512
# modir = "/data2/lrh/dataset/fine_dance/origin/motion_feature319"

modir = eval(f"args.DATASET.{dataname.upper()}.MOTION")
# modir = '/data/lrh/datasets/fine_dance/gound/mofea266/new_joint_vecs'

for file in os.listdir(modir):
    if file[-3:] != "npy":
        continue
    # if file[0] == "M":
        # continue
    mofile = os.path.join(modir, file)
    gt_motion = np.load(mofile)
    if '60fps' in modir:
        print('jiang caiyang!')
        gt_motion = gt_motion[::2]
    if length_clip > gt_motion.shape[0]:
        continue
    T, C = gt_motion.shape
    if gt_motion.shape[-1] == 319:
        C = 139
        gt_motion = gt_motion[:,:C]
    
    # gt_motion = gt_motion[:,:135]
    # print("gt_motion.shape",gt_motion.shape)
    tail = T % length_clip
    gt_motion = gt_motion[:T-tail]
    assert gt_motion.shape[0] % length_clip == 0
    if args.Norm:
        if '139' in dataname or '151' in dataname:
            gt_motion = normalizer.normalize(torch.from_numpy(gt_motion)).detach().cpu().numpy()
        else:
            gt_motion = (gt_motion - NormMean) / NormStd
    gt_motion = gt_motion.reshape(-1, length_clip, C)

    gt_motion = torch.from_numpy(gt_motion).to(torch.float32).to(device)    #.unsqueeze(0)
    print("gt_motion.shape",gt_motion.shape)
    code_idx = net.vqvae.my_encode(gt_motion)

    code_idx_np = code_idx.detach().cpu().numpy()
    np.save(os.path.join(out_dir_codeidx, file), code_idx_np.reshape(-1))

    pred_motion = net.vqvae.my_forward_decoder(code_idx)
    print("pred_motion",pred_motion.shape)
    pred_motion = pred_motion.detach().cpu().numpy()
    if args.Norm:
        if '139' in dataname or '151' in dataname:
            pred_motion = normalizer.unnormalize(torch.from_numpy(pred_motion)).detach().cpu().numpy()
        else:
            pred_motion = (pred_motion * NormStd)  + NormMean

    if len(pred_motion.shape) == 3 and pred_motion.shape[0] == 1:
        pred_motion = pred_motion.squeeze(0)
    np.save(os.path.join(out_dir_motion, file), pred_motion.reshape(-1, C))
    # print(pred_motion.shape)
    #
sys.exit(0)
##### ------ warm-up ------- #####
print('start warming up...')
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in tqdm(range(1, args.warm_up_iter)):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = next(train_loader_iter) # motion, music, filename
    gt_motion = gt_motion.to(device).float() # (bs, 64 args.window_size, dim)

    pred_motion, loss_commit, perplexity = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

# 记录一些数值
recon_min, comit_min,vel_min,preb_min = 1000,1000,1000,1000

for nb_iter in tqdm(range(1, args.total_iter + 1)):
    
    gt_motion = next(train_loader_iter) # motion, music, filename | motion:torch.Size([128, 128, 319])
    gt_motion = gt_motion.to(device).float() # bs, nb_joints, joints_dim, seq_len
    
    pred_motion, loss_commit, perplexity = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} :  lr {current_lr:.5f}  \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,
    
    # Save checkpoint
    if nb_iter % args.save_iter == 0 or nb_iter == 1:
        torch.save({'net' : net.state_dict()}, os.path.join(args.ckpt_dir, 'vqvae/' + f'f_{args.DATA_SETTING.nfeats}' + '/last.pth'))
        print('saved...')

    if loss_motion.item() < recon_min:
        torch.save({'net' : net.state_dict()}, os.path.join(args.ckpt_dir, 'vqvae/' + f'f_{args.DATA_SETTING.nfeats}' + '/best_recon.pth'))
        recon_min = loss_motion.item()
        print('recon updated...')

    if loss_vel.item() < vel_min:
        torch.save({'net' : net.state_dict()}, os.path.join(args.ckpt_dir, 'vqvae/' + f'f_{args.DATA_SETTING.nfeats}' + '/best_vel.pth'))
        vel_min = loss_vel.item()
        print('vel updated...')

    if loss_commit.item() < comit_min:
        torch.save({'net' : net.state_dict()}, os.path.join(args.ckpt_dir, 'vqvae/' + f'f_{args.DATA_SETTING.nfeats}' + '/best_conmit.pth'))
        comit_min = loss_commit.item()
        print('commit updated...')

    if perplexity.item() < preb_min:
        torch.save({'net' : net.state_dict()}, os.path.join(args.ckpt_dir, 'vqvae/' + f'f_{args.DATA_SETTING.nfeats}' + '/best_preb.pth'))
        preb_min = perplexity.item()
        print('perplexity updated...')
    
    # eval
    # if nb_iter % args.eval_iter==0 :
        
    