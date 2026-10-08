'''text and music masked seq generetion'''
import os, sys
import json
from torch.distributions import Categorical
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import models.c2d_trans_frame as trans

import clip
import options.option_trans_frame as option_trains
# import options.option_vq_frame as option_vq
import utils.utils_model as utils_model
# from dataset import dataset_gpt
from dataset import dataset_gpt_m2d as dataset_gpt_m2d_clip
import numpy as np
import random
import copy

import warnings
# warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # 让报错信息的位置准确
# CUDA_VISIBLE_DEVICES=0
import argparse
from omegaconf import OmegaConf
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str,
        required=False,
        default="./configs/gpt/aistpp290.yaml",
        help="config file",
    )
parser.add_argument("--assets", type=str,
        required=False,
        default="DanceDiffusion/configs/45assets.yaml",
        help="config file for asset paths",
    )
parser.add_argument("--tkdir", type=str,
        required=False,
        default="experiments/vqvae/AISTPP_266/mo266_0607_win64_Norm/eval_idx_f_266_5710",
        help="config file for asset paths",
    )
parser.add_argument('--gpu', type=int, default=2, help='gpu')
args = parser.parse_args()
args = OmegaConf.create(vars(args))
cfg_exp = OmegaConf.load(args.cfg)
cfg_assets = OmegaConf.load(args.assets)
cfg = OmegaConf.merge(cfg_exp, cfg_assets)
args = OmegaConf.merge(cfg, args)
current_time = datetime.now()

out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(out_dir, exist_ok = True) # 创建 out-dir 文件夹, 用于保存 loger 数据
outckpt_dir = os.path.join(out_dir, 'ckpt')
os.makedirs(outckpt_dir, exist_ok = True) 
time_str = current_time.strftime('%Y%m%d_%H%M%S')
OmegaConf.save(args, os.path.join(out_dir, 'traingpt_' + time_str + '.yaml'))

command = ' '.join(sys.argv)
# 保存执行命令到文件
with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
    f.write(command)

device = f"cuda:{args.gpu}"
torch.manual_seed(args.seed)


##### ---- Logger ---- #####
logger = utils_model.get_logger(out_dir) 
writer = SummaryWriter(out_dir) 
# logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader = dataset_gpt_m2d_clip.DATALoader(args, 
                                    isTrain = True,
                                    batch_size = args.batch_size)

train_loader_iter = dataset_gpt_m2d_clip.cycle(train_loader)

##### ---- Network ---- #####
trans_encoder = trans.c2d_Transformer(args=args,
                                num_vq=args.nb_code, 
                                window_size=args.DATA_SETTING.full_seq_len,
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


if args.resume_trans is not None: # 有才加载
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.to(device)


##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss() 

##### ---- Training ---- #####
nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

best_avg_loss_cls, best_avg_acc = 10000, 0 # t2d 用 | ckpt 筛选
while nb_iter <= args.total_iter:
    batch = next(train_loader_iter)
    cond, m_tokens, genre, m_tokens_len = batch

    # continue
    cond, m_tokens, genre, m_tokens_len = cond.to(device), m_tokens.to(device), genre.to(device), m_tokens_len.to(device)
    
    bs = m_tokens.shape[0]
    target = copy.deepcopy(m_tokens) # 不进行 deepcopy 会变成协同变量 | (bs, 26)
    target = target.to(device)


    input_index = target[:,:-1]
    # 生成初始idx 
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                         device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                         device=input_index.device))
    mask = mask.round().to(dtype=torch.int64) # 随机替换掉部分 idx, 增加模型鲁棒性
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices 

    # t2d 用 | T2M-GPT loss | predict accuracy
    # predict idx
    noise = torch.randn(bs, 256).to(device)
    
    if cond.shape[2] == 35: # music feature
        # cls_pred = trans_encoder(a_indices, noise = noise, feature = cond, genre = genre, masked_token_seq = None) # m_tokens
        cls_pred = trans_encoder(a_indices, noise = noise, feature = cond, genre = genre, masked_token_seq = None)
    else:
        raise("error of input condition")
    cls_pred = cls_pred.contiguous()        #.long()    torch.Size([2, 2049, 513])
    target = target.long()                  # b, 2048

    loss_cls = 0.0
    for i in range(bs): # 计算每个 batch 
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs
        # loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i]], target[i][:m_tokens_len[i]])[mask_idx] / bs # only compute masked items' loss
        # loss_cls += loss_ce(cls_pred[i][mask_idx], target[i][mask_idx].long()) / bs # only compute masked items' loss
        # torch.Size([2, 65, 1025]) torch.Size([2, 64])
        # Accuracy | 
        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1) # torch.Size([2, 121, 512]), 可见第二个 dim 对应 seq_len

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)
        else: 
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()
    
    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()


    # args.print_iter = 20
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} :  lr {scheduler.get_lr()[0]:.5f} , Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    
        # Save checkpoint
        if nb_iter % args.save_iter == 0 or nb_iter == 1:
            torch.save({'trans' : trans_encoder.state_dict(),
                        # 'dis' : D_net.state_dict()
                        }, 
                        os.path.join(outckpt_dir, 'mask_last.pth')) 
            print('saved... ')

            if avg_loss_cls < best_avg_loss_cls:
                torch.save({'trans' : trans_encoder.state_dict(),
                            # 'dis' : D_net.state_dict()
                            }, 
                            os.path.join(outckpt_dir, 'mask_best_loss.pth')) 
                best_avg_loss_cls = avg_loss_cls
                print('best loss .pth updated...')
            
            if avg_acc > best_avg_acc:
                    torch.save({'trans' : trans_encoder.state_dict(),
                                # 'dis' : D_net.state_dict()
                                }, 
                            
                                os.path.join(outckpt_dir, 'mask_best_acc.pth')) 
                    best_avg_acc = avg_acc
                    print('acc updated...')

    nb_iter += 1


        
    
