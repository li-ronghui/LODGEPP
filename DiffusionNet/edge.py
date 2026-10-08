import multiprocessing
import os
from zlib import Z_FULL_FLUSH

import numpy as np
os.environ["WANDB_API_KEY"] = "8703d7348effc00c329d216eb3eed220e0e2912d"
os.environ["WANDB_MODE"] = "online"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from functools import partial
from pathlib import Path
# from dataset.preprocess import My_Normalizer as Normalizer        # do not use Normalizer
from models.edge.args import FineDance_parse_train_opt, save_arguments_to_yaml
import sys

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import re

from dataset.FineDance_dataset_back import FineDance_Smpl
from models.edge.adan import Adan
from models.edge.diffusion import GaussianDiffusion
from models.edge.model import DanceDecoder, Refine_DanceDecoder
from utils.smplfk import SMPLX_Skeleton
from render import ax_from_6v, ax_to_6v

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

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}

def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)

class EDGE:
    def __init__(
        self,
        opt,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):
        self.opt = opt
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
    
        self.repr_dim = repr_dim = opt.nfeats
        feature_dim = opt.cond_feadim
   
        self.horizon = horizon = opt.full_seq_len

        self.accelerator.wait_for_everyone()

        self.resume_num = 0
        checkpoint = None
        self.normalizer = None
        if checkpoint_path != "":
            print("checkpoint_path", checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.resume_num = int(os.path.basename(checkpoint_path).split("-")[-1].split(".")[0])      # int(os.path.basenam

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )
        # model = Refine_DanceDecoder(
        #     nfeats=repr_dim,
        #     seq_len=horizon,
        #     latent_dim=512,
        #     ff_size=1024,
        #     num_layers=8,
        #     num_heads=8,
        #     dropout=0.1,
        #     cond_feature_dim=feature_dim,
        #     activation=F.gelu,
        # )

        diffusion = GaussianDiffusion(
            model,
            opt,
            horizon,
            repr_dim,
            smplx_model = None,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
            do_normalize = opt.do_normalize
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)                                            
        self.diffusion = diffusion.to(self.accelerator.device)                              # 为什么这里不需要prepare
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            print("only load model")
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )
            print("loaded only model sucessfully!!!")


    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        print("train_dataset = FineDance_Dataset ")  
        train_dataset = FineDance_Smpl(
            args=opt,            # data/
            istrain=True,
        )
        test_dataset = FineDance_Smpl(
            args=opt,          
            istrain=False,
        )

        num_cpus = multiprocessing.cpu_count()
        print("batchsize=:", opt.batch_size)
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.5), 60),      # num_workers=min(int(num_cpus * 0.75), 32), 
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            wandb.save("params.yaml")  # 保存wandb配置到文件
            yaml_path = os.path.join(wdir, 'parameters.yaml')
            save_arguments_to_yaml(opt, yaml_path)
            with open(os.path.join(wdir, 'command.txt'), 'w') as f:
                f.write(command)


        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            print("epoch:", epoch+self.resume_num)
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_vfkloss = 0
            avg_footloss = 0
            avg_foot_normalloss = 0
            avg_foot_bce_loss = 0
            avg_fc_loss = 0
            avg_root_loss = 0


            # train
            self.train()
            
            for step, batchdata in enumerate(
                load_loop(train_data_loader)
            ):
                if self.opt.keymotion_dir:
                    x, cond, filename, key_motion = batchdata
                    # key_motion = torch.cat((key_motion[:,:,:3], key_motion[:,:,6:6+22*6]), dim=-1)
                    assert key_motion.shape[2] == 135 or key_motion.shape[2] == 139
                    cond = torch.cat([cond, key_motion], dim=-1)
                else:
                    x, cond, filename = batchdata

                if opt.do_normalize:
                    with torch.no_grad():
                        x = self.normalizer.normalize(x)
                
                if opt.nfeats == 139:
                    x = x[:, :, :opt.nfeats]

                    # with torch.no_grad():
                    #     smplx_modle = SMPLX_Skeleton()
                    #     positions = do_smplxfk(x[0, :,:139], smplx_modle)
                    #     ltoe = positions[0:150, 10,  1]
                    # print("batch mean of y", ltoe.mean())
                    # print("batch min of y", min(ltoe) )
                    # print("batch max of y", max(ltoe) )


            # for step, (x, cond, filename, wavnames) in enumerate(
                # train_data_loader
            # ):
                if opt.nfeats == 263:
                    total_loss, (loss, v_loss) = self.diffusion(
                        x, cond, t_override=None
                    )
                    fk_loss = torch.zeros_like(loss).to(loss)
                    foot_loss = torch.zeros_like(loss).to(loss)
                    fk_loss_v = torch.zeros_like(loss).to(loss)
                    fk_loss_a = torch.zeros_like(loss).to(loss)
                    root_loss = torch.zeros_like(loss).to(loss)
                else:
                    total_loss, (loss, v_loss, fk_loss, foot_loss, foot_normalloss, foot_bce_loss, fc_loss, fk_loss_v, fk_loss_a, root_loss) = self.diffusion(
                        x, cond, t_override=None
                    )
                # print("3")
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_vfkloss += fk_loss_v.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    avg_foot_normalloss += foot_normalloss.detach().cpu().numpy()
                    avg_foot_bce_loss += foot_bce_loss.detach().cpu().numpy()
                    avg_fc_loss  += fc_loss.detach().cpu().numpy()
                    avg_root_loss += root_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )

            #-----------------------------------------------------------------------------------------------------------
            # test
            # Save model
            if ((epoch+self.resume_num) % opt.save_interval) == 0  or epoch<=1:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                self.eval()     # debug!
                # save only if on main thread
                if self.accelerator.is_main_process:
                    # self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_vfkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    avg_foot_normalloss /= len(train_data_loader)
                    avg_foot_bce_loss /= len(train_data_loader)
                    avg_fc_loss /= len(train_data_loader)
                    avg_root_loss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "FK V Loss": avg_vfkloss,
                        "Foot Loss": avg_footloss,
                        "Foot Loss Norm": avg_foot_normalloss,
                        "bce_loss": avg_foot_bce_loss,
                        "fc_loss": avg_fc_loss,
                        "root_loss": avg_root_loss,
                    }
                    wandb.log(log_dict)

                    if opt.fullpt:
                        torch.save(self.diffusion, os.path.join(wdir, f"Full-train-{epoch+self.resume_num}.pt"))
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),     # 经过accelerate prepare的模型，在保存时需要unwrap，反之不需要
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                        }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch+self.resume_num}.pt"))
                    print(f"[MODEL SAVED at Epoch {epoch+self.resume_num}]")
                   
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.opt.nfeats)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    if self.opt.keymotion_dir:
                        (x, cond, filename, key_motion) = next(iter(test_data_loader))
                        # if frames
                        # key_motion = torch.cat((key_motion[:,:,:3], key_motion[:,:,6:6+22*6]), dim=-1)
                        assert key_motion.shape[2] == 135 or key_motion.shape[2] == 139
                        cond = torch.cat([cond, key_motion], dim=-1)
                    else:
                        (x, cond, filename) = next(iter(test_data_loader))
                    if opt.do_normalize:
                        with torch.no_grad():
                            x = self.normalizer.normalize(x)
                            
                    if opt.nfeats == 139:
                        x = x[:, :, :opt.nfeats]
                    
                    cond = cond.to(self.accelerator.device)
                    # name_iter = name_iter+1
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch+self.resume_num,
                        render_out = os.path.join(opt.render_dir, "train_" + opt.exp_name),      # render out
                        fk_out = os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=filename[:render_count],
                        # name = str(epoch) + str(name_iter).zfill(3)
                        sound=True,
                    )
            #-----------------------------------------------------------------------------------------------------------
            
                        
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, mode="normal",cons=None, soft_hint=False,
    ):
        _, cond, wavname, orikey = data_tuple
        # print("orikey",orikey.shape)
        print(cond.shape)
        assert len(cond.shape) == 3

        # if render_count < 0:
        render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device).float()
        
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
                    mocond_1[:, 4]  = mocond_1[:, 4] - mocond_1[:1, 4]  
                    mocond_1[:, 6]  = mocond_1[:, 6] - mocond_1[:1, 6]  
                    mocond_2 = orikey[i][-4:,:]
                    mocond_2[:, 4]  = mocond_2[:, 4] - mocond_2[:1, 4]  
                    mocond_2[:, 6]  = mocond_2[:, 6] - mocond_2[:1, 6] 

                    constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                    constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                    constraint["value"][i, 4:-4, :4] = torch.from_numpy(orikey[i][4:-4, :4]).to(cond)
                    constraint["value"][i, 4:-4, 7:] = torch.from_numpy(orikey[i][4:-4, 7:]).to(cond)
                    constraint["mask"][i, :,:] = 1 
                    constraint["mask"][i, :,:] = 1
                    constraint["mask"][i, :, 4:7] = 0 
                    constraint["mask"][i, :, 4:7] = 0
            elif soft_hint == 'dod':
                constraint={}
                constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
                constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
                # print("constraint[value].shape", constraint["value"].shape)
                # print("orikey.shape", orikey.shape)
                for i in range(cond.shape[0]):
                    mocond_1 = orikey[i][:4,:]
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
                    # mocond_1[:, 4:7]  = mocond_1[:, 4:7] - mocond_1[:1, 4:7]  
                    mocond_1[:, 4]  = mocond_1[:, 4] - mocond_1[:1, 4]  
                    mocond_1[:, 6]  = mocond_1[:, 6] - mocond_1[:1, 6]  
                    mocond_2 = orikey[i][-4:,:]
                    # mocond_2[:, 4:7]  = mocond_2[:, 4:7] - mocond_2[:1, 4:7]  
                    mocond_2[:, 4]  = mocond_2[:, 4] - mocond_2[:1, 4]  
                    mocond_2[:, 6]  = mocond_2[:, 6] - mocond_2[:1, 6] 

                    constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                    constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                    constraint["mask"][i, :4,:] = 1 
                    constraint["mask"][i, -4:,:] = 1 
                
        else:
            raise("cond fea error!")
        # constraint = None

        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
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
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device).float()
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="normal",            # 这里设置        default is long
            fk_out=fk_out,
            render=render
        )



def train(opt):
    model = EDGE(opt, opt.feature_type, opt.checkpoint)
    model.train_loop(opt)
    
if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=7  python edge.py --full_seq_len 512 --windows 20 --batch_size 120  --epochs 2500 --nfeats 139 --wandb_pj_name 0821_edge512decoder --project experiments/0821_edge512decoder/train  --exp_name edge_512   --render_dir experiments/0821_edge512decoder/renders
    opt = FineDance_parse_train_opt()
    opt.window_size = opt.full_seq_len
    opt.feature_dim = opt.nfeats
    opt.clip_frames = 1

    # train(opt)
    # # 获取执行命令
    command = ' '.join(sys.argv)
    # 保存执行命令到文件
    if not os.path.exists(os.path.join(opt.project, opt.exp_name)):
        os.makedirs(os.path.join(opt.project, opt.exp_name), exist_ok=False)
            
    train(opt)