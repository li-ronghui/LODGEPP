import os
import json
import sys
from sklearn.preprocessing import Normalizer
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
import models.vqvae as vqvae
# from test_vq_clip8 import NormStd
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
# from dataset.FineDance_dataset import FineDance_Smpl
from  DanceDiffusion.dld.data.FineDance_dataset import FineDance_Smpl

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import argparse
sys.path.append('/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion')

# import utils.eval_trans as eval_trans
# from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
# warnings.filterwarnings('ignore')

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def train(args):
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(out_dir) 
    writer = SummaryWriter(out_dir) 

    ##### ---- Dataloader ---- #####
    train_dataset = FineDance_Smpl(
                args=args,            # data/
                istrain=True,
                dataname=args.dataname[0],
            )
    # test_dataset = FineDance_Smpl(
    #     args=args,          
    #             istrain=False,
            # )
    train_data_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=32,      # num_workers=min(int(num_cpus * 0.75), 32), 
                pin_memory=True,
                drop_last=True,
            )
    # test_data_loader = DataLoader(
    #             test_dataset,
    #             batch_size=args.batch_size,
    #             shuffle=True,
    #             num_workers=2,
    #             pin_memory=True,
    #             drop_last=True,
    #         )

    # 验证集
    # val_loader = dataset_eval.DATALoader(args, True) # isTrain

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
    net.train()
    net.to(device)

    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    print("lr scheduler", args.lr_scheduler)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    Loss = losses.ReConsLoss(args.recons_loss,args)

    ##### ------ warm-up ------- #####
    print('start warming up...')
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    dataname = args.dataname[0]
    if args.Norm:
        print(f"args.DATASET.{dataname.upper()}.normalizer.params.mean")
        normalizer = torch.load(eval(f"args.DATASET.{dataname.upper()}.normalizer"))
    iter_num = 0
    # for nb_iter in tqdm(range(1, args.warm_up_iter)):
    #     for step, (gt_motion, cond, filename) in enumerate(train_data_loader):
    #         iter_num += 1
    #         optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
            
    #         gt_motion = gt_motion[:,:,:args.DATA_SETTING.nfeats]
    #         gt_motion = gt_motion.to(device).float() # (bs, 64 args.window_size, dim)
    #         # print("gt_motion", gt_motion[0,0,:3])
    #         if args.Norm:
    #             gt_motion = normalizer.normalize(gt_motion) # (gt_motion - NormMean) / NormStd 
    #         # print("gt_motion", gt_motion.shape)
    #         # print("gt_motion", gt_motion[0,0,:3])
    #         pred_motion, loss_commit, perplexity = net(gt_motion)
            
    #         if gt_motion.shape[-1] in [266,290]:
    #             loss_motion = Loss(pred_motion, gt_motion)
    #             loss_vel = Loss.forward_vel(pred_motion, gt_motion, args.DATA_SETTING.njoints)
    #             loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    #         elif gt_motion.shape[-1] == 135:
    #             loss_motion = Loss.forward_cliprot(pred_motion, gt_motion)
    #             loss_fk = Loss.forward_clipfk(pred_motion, gt_motion)         # 计算motion clip的loss
    #             loss = loss_motion + args.commit * loss_commit +  loss_fk
    #         elif gt_motion.shape[-1] == 139:
    #             loss_motion = Loss.forward_edgeloss139(pred_motion, gt_motion)
    #             loss = loss_motion + args.commit * loss_commit
    #         else:
    #             loss_motion = Loss(pred_motion, gt_motion)
    #             loss = loss_motion + args.commit * loss_commit
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if nb_iter % args.print_iter ==  0 :
    #             avg_recons += loss_motion.item()
    #             avg_perplexity += perplexity.item()
    #             avg_commit += loss_commit.item()

    #             # avg_recons /= args.print_iter
    #             # avg_perplexity /= args.print_iter
    #             # avg_commit /= args.print_iter
                
    #             logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
                
    #             avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    # 记录一些数值
    recon_min, comit_min,vel_min,preb_min = 1000,1000,1000,1000

    for nb_iter in tqdm(range(1, args.total_iter + 1)):
        scheduler.step()
        for step, (gt_motion, cond, filename) in enumerate(train_data_loader):
            gt_motion = gt_motion.to(device).float()[:,:,:args.DATA_SETTING.nfeats] # bs, nb_joints, joints_dim, seq_len
            if args.Norm:
                gt_motion = normalizer.normalize(gt_motion) #(gt_motion - NormMean) / NormStd 
            
            pred_motion, loss_commit, perplexity = net(gt_motion)
            if gt_motion.shape[-1] in [266,290]:
                loss_motion = Loss(pred_motion, gt_motion)
                loss_vel = Loss.forward_vel(pred_motion, gt_motion, args.DATA_SETTING.njoints)
                loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
            elif gt_motion.shape[-1] == 135:
                loss_motion = Loss.forward_cliprot(pred_motion, gt_motion)
                loss_fk = Loss.forward_clipfk(pred_motion, gt_motion) 
                loss = loss_motion + args.commit * loss_commit +  loss_fk
            elif gt_motion.shape[-1] == 139:
                loss_motion = Loss.forward_edgeloss139(pred_motion, gt_motion)
                loss = loss_motion + args.commit * loss_commit
            else:
                loss_motion = Loss(pred_motion, gt_motion)
                loss = loss_motion + args.commit * loss_commit
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if nb_iter % args.print_iter ==  0 :
                avg_recons += loss_motion.item()
                avg_perplexity += perplexity.item()
                avg_commit += loss_commit.item()

                # avg_recons /= args.print_iter
                # avg_perplexity /= args.print_iter
                # avg_commit /= args.print_iter
                
                writer.add_scalar('./Train/L1', avg_recons, nb_iter)
                writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
                writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
                
                logger.info(f"Train. Iter {nb_iter} : lr {scheduler.get_lr()[0]:.8f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
                
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,
        
            # Save checkpoint
            if nb_iter % args.save_iter == 0:
                torch.save({'net' : net.state_dict()}, ckpt_save_dir + '/last.pth')
                print('saved...')

                if loss_motion.item() < recon_min:
                    torch.save({'net' : net.state_dict()}, ckpt_save_dir + '/best_recon.pth')
                    recon_min = loss_motion.item()
                    print('recon updated...')

                if loss_commit.item() < comit_min:
                    torch.save({'net' : net.state_dict()}, ckpt_save_dir + '/best_conmit.pth')
                    comit_min = loss_commit.item()
                    print('commit updated...')

                if perplexity.item() < preb_min:
                    torch.save({'net' : net.state_dict()}, ckpt_save_dir + '/best_preb.pth')
                    preb_min = perplexity.item()
                    print('perplexity updated...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
            required=False,
            default="./configs/vqvae/aistpp290.yaml",
            help="config file",
        )
    parser.add_argument("--assets", type=str,
            required=False,
            default="DanceDiffusion/configs/45assets.yaml",
            help="config file for asset paths",
        )
    args = parser.parse_args()
    args = OmegaConf.create(vars(args))
    cfg_exp = OmegaConf.load(args.cfg)
    cfg_assets = OmegaConf.load(args.assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    args = OmegaConf.merge(cfg, args)
    current_time = datetime.now()
    out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(out_dir, exist_ok = True) # 创建 out-dir 文件夹, 用于保存 loger 数据
    
    time_str = current_time.strftime('%Y%m%d_%H%M%S')
    OmegaConf.save(args, os.path.join(out_dir, 'train_' + time_str + '.yaml'))


    command = ' '.join(sys.argv)
    # 保存执行命令到文件
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(command)

    ckpt_save_dir = os.path.join(out_dir, 'vqvae/' + f'f_{args.DATA_SETTING.nfeats}')
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    # args.clip_frames = 1
    # args.keymotion_dir = None
    device = f"cuda:{args.gpu}"
    print(args.dataname)
    torch.manual_seed(args.seed)

    train(args)
            

    
            
        