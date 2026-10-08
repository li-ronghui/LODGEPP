'''text and music masked seq generetion'''
import os
import json
from regex import F
from torch.distributions import Categorical
import torch
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter
import models.c2d_trans_frame as trans
import models.vqvae as vqvae

import clip
import options.option_trans_frame as option_trains
# import options.option_vq_frame as option_vq
import utils.utils_model as utils_model
# from dataset import dataset_gpt
from dataset import dataset_gpt_m2d
import numpy as np
import random
import copy

import warnings
# warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # 让报错信息的位置准确
# CUDA_VISIBLE_DEVICES=0

##### ---- Exp dirs ---- #####
args = option_trains.get_args_parser()
args.clip_frames = 4
device = f"cuda:{args.gpu}"
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True) # 创建 out-dir 文件夹, 用于保存 loger 数据
os.makedirs(os.path.join(args.out_dir, "result"), exist_ok = True)

# outckpt_dir = os.path.join(args.ckpt_dir, 'gpt/' + f'f_{args.DATA_SETTING.nfeats}' + '/' + args.exp_name )
# if not os.path.
# exists(outckpt_dir):
#     os.makedirs(outckpt_dir)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir) 
writer = SummaryWriter(args.out_dir) 
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


##### ---- Dataloader ---- #####
test_loader = dataset_gpt_m2d.DATALoader(args, 
                                    isTrain = False,
                                    batch_size = args.batch_size)

test_loader_iter = dataset_gpt_m2d.cycle(test_loader)


##### ---- Network ---- #####
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

trans_encoder = trans.c2d_Transformer(args=args,
                                num_vq=args.nb_code, 
                                window_size=args.window_size,
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

if args.resume_pth : # 有vq-vae才加载
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(device)


##### ---- Optimizer & Scheduler ---- #####
# optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss() 

##### ---- Training ---- #####
nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

best_avg_loss_cls, best_avg_acc = 10000, 0 # t2d 用 | ckpt 筛选


while nb_iter <= args.total_iter:

    batch = next(test_loader_iter)
    cond, m_tokens, genre, m_tokens_len = batch
    cond, m_tokens, genre, m_tokens_len = cond.to(device), m_tokens.to(device), genre.to(device), m_tokens_len.to(device)
    bs = m_tokens.shape[0]

    print("cond.shape", cond.shape)
    print(genre)
    print("genre", genre.shape)
    # sys.exit(0)
    target = copy.deepcopy(m_tokens) # 不进行 deepcopy 会变成协同变量 | (bs, 26)
    print("target.shape", target.shape)
    # target = target.to(device)


    # t2d 用 | T2M-GPT loss | predict accuracy
    # predict idx
    noise = torch.randn(bs, 256).to(device)
    
    if cond.shape[2] == 35: # music feature
        # cls_pred = trans_encoder(a_indices, noise = noise, feature = cond, genre = genre, masked_token_seq = None) # m_tokens
        cls_pred = trans_encoder.sample(noise = noise, feature = cond, genre = genre, masked_token_seq = None)
    else:
        raise("error of input condition")


    cls_pred = cls_pred[0].contiguous().unsqueeze(0)        #.long()    torch.Size([2, 2049, 513])
    print("cls_pred.shape", cls_pred.shape)
    if cls_pred.shape[1] > args.window_size:
        cls_pred = cls_pred[:, :args.window_size]


    pred_motion = net.vqvae.my_forward_decoder(cls_pred)
    pred_motion = pred_motion.detach().cpu().numpy()
    print("shape of pred_motion", pred_motion.shape)


    np.save(os.path.join(args.out_dir, "result", str(nb_iter)+".npy"), pred_motion)
    nb_iter += 1


