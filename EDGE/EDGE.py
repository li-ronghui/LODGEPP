import multiprocessing
import os, sys
import pickle
from functools import partial
from pathlib import Path
from tkinter import FALSE

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

# from dataset.dance_dataset import AISTPPDataset
from dataset.dataset import FineDance_Smpl
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton
from configs.config import instantiate_from_config, get_obj_from_str
from omegaconf import OmegaConf
import datetime





def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        cfg,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):
        self.cfg = cfg
        feature_type = cfg.feature_type
        use_baseline_feats = feature_type == cfg.feature_type

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        self.repr_dim = repr_dim = cfg.DATA_SETTING.nfeats

        FPS = 30
        self.horizon = cfg.DATA_SETTING.full_seq_len

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            print('EDGE load checkpoint! :  ', checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            # self.normalizer = checkpoint["normalizer"]

        
        if cfg.Norm:
            dataname = str(cfg.TRAIN.DATASETS[0])
            # self.normalizer = torch.load(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))  
            self.normalizer = instantiate_from_config(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))
        else:
            self.normalizer = None


        model = instantiate_from_config(cfg.model.DanceDecoder)
        # model = get_obj_from_str(cfg.model.DanceDecoder["target"])(smplx_model=self.smplx_fk, normalizer=self.normalizer, genre_num=self.cfg.DATA_SETTING.GENRE_NUM,**cfg.model.DanceDecoder.get("params", dict()))

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = get_obj_from_str(cfg.model.diffusion["target"])(cfg=cfg, model=model, normalizer=self.normalizer, smpl=smpl,  **cfg.model.diffusion.get("params", dict()))
        '''
        model = DanceDecoder(
            nfeats = cfg.model.DanceDecoder.nfeats,
            seq_len = cfg.model.DanceDecoder.seq_len,
            latent_dim = cfg.model.DanceDecoder.latent_dim,
            ff_size = cfg.model.DanceDecoder.ff_size,
            num_layers = cfg.model.DanceDecoder.num_layers,
            num_heads = 8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )
        diffusion = GaussianDiffusion(
            model,
            self.horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )'''

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        cfgim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.cfgim = self.accelerator.prepare(cfgim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, cfg):
        # load datasets
        # train_tensor_dataset_path = os.path.join(
        #     cfg.processed_data_dir, f"train_tensor_dataset.pkl"
        # )
        # test_tensor_dataset_path = os.path.join(
        #     cfg.processed_data_dir, f"test_tensor_dataset.pkl"
        # )
        # if (
        #     not cfg.no_cache
        #     and os.path.isfile(train_tensor_dataset_path)
        #     and os.path.isfile(test_tensor_dataset_path)
        # ):
        #     train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
        #     test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        # else:
        train_dataset = FineDance_Smpl(
            args=cfg,
            istrain=True,
            dataname=cfg.TRAIN.DATASETS[0]
            # data_path=cfg.data_path,
            # backup_path=cfg.processed_data_dir,
            # train=True,
            # force_reload=cfg.force_reload,
        )
        test_dataset = FineDance_Smpl(
            args=cfg,
            istrain=False,
            dataname=cfg.TEST.DATASETS[0]
        )
            # cache the dataset in case
            # if self.accelerator.is_main_process:
            #     pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
            #     pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))


        # data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=min(cfg.TRAIN.NUM_WORKERS, 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=cfg.EVAL.BATCH_SIZE,
            shuffle=True,          # origin is Ture
            num_workers=cfg.EVAL.NUM_WORKERS,
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
            save_dir = str(increment_path(Path(cfg.project) / cfg.exp_name))
            print('exp_name 11',cfg.exp_name)
            save_dir = Path(save_dir)
            cfg.exp_name = save_dir.name
            print('exp_name 1-1',cfg.exp_name)
            wandb.init(project=cfg.wandb_pj_name, name=cfg.exp_name)
            print('save_dir 11', save_dir)
            print('exp_name 12',cfg.exp_name)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

            current_time = datetime.datetime.now()
            time_str = current_time.strftime('%Y%m%d_%H%M%S')
            OmegaConf.save(cfg, os.path.join(save_dir, 'train_' + time_str + '.yaml'))

            command = ' '.join(sys.argv)
            with open(os.path.join(save_dir, 'command.txt'), 'w') as f:
                f.write(command)


        self.accelerator.wait_for_everyone()
        for epoch in range(1, cfg.TRAIN.END_EPOCH + 1):
            avg_lossdict = {}
            avg_lossdict['loss'] = torch.tensor(0.0)
            if cfg.LOSS.TYPE == 'loss_266_origin':
                avg_lossdict['mseloss'] = torch.tensor(0.0)
                avg_lossdict['v_loss'] = torch.tensor(0.0)
            elif cfg.LOSS.TYPE == 'loss_266':
                avg_lossdict['mseloss'] = torch.tensor(0.0)
                avg_lossdict['v_loss'] = torch.tensor(0.0)
                avg_lossdict['foot_loss'] =torch.tensor(0.0)
                avg_lossdict['a_loss'] = torch.tensor(0.0)
            elif cfg.LOSS.TYPE == 'loss_vec_vel':
                avg_lossdict['mseloss'] = torch.tensor(0.0)
                avg_lossdict['local_loss'] = torch.tensor(0.0)
                avg_lossdict['bone_loss'] = torch.tensor(0.0)
                avg_lossdict['v_loss'] = torch.tensor(0.0)
                avg_lossdict['a_loss'] = torch.tensor(0.0)
                avg_lossdict['wave_loss_ll'] = torch.tensor(0.0)
                avg_lossdict['wave_loss_lh'] = torch.tensor(0.0)
                avg_lossdict['wave_loss_hl'] = torch.tensor(0.0)
                avg_lossdict['wave_loss_hh'] = torch.tensor(0.0)
            # train
            self.train()
            for step, (x, cond, genre, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                if cfg.Norm:
                    x = self.normalizer.normalize(x)
                # -------------------Training-------------------
                loss_dict = self.diffusion(
                    x, cond, t_override=None
                )

                self.cfgim.zero_grad()
                self.accelerator.backward(loss_dict['loss'])        # loss is the total loss

                self.cfgim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    Cp_loss_dict = loss_dict.copy()
                    for key in loss_dict.keys():
                        avg_lossdict[key] += Cp_loss_dict[key].detach().cpu().numpy()
                    # avg_loss += loss.detach().cpu().numpy()
                    # avg_vloss += v_loss.detach().cpu().numpy()
                    # avg_fkloss += fk_loss.detach().cpu().numpy()
                    # avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % cfg.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )

            # Save model
            if (epoch % cfg.log_interval) == 0 or epoch==1:
                print('now in testing , epoch is: ', epoch)
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    for key in avg_lossdict.keys():
                        avg_lossdict[key] /= len(train_data_loader)
                    # avg_loss /= len(train_data_loader)
                    # avg_vloss /= len(train_data_loader)
                    # avg_fkloss /= len(train_data_loader)
                    # avg_footloss /= len(train_data_loader)
                    # log_dict = {
                    #     "Train Loss": avg_loss,
                    #     "V Loss": avg_vloss,
                    #     "FK Loss": avg_fkloss,
                    #     "Foot Loss": avg_footloss,
                    # }
                    wandb.log(avg_lossdict)
                    if (epoch % cfg.save_interval) == 0 or epoch==1:
                        ckpt = {
                            "ema_state_dict": self.diffusion.master_model.state_dict(),
                            "model_state_dict": self.accelerator.unwrap_model(
                                self.model
                            ).state_dict(),
                            "cfgimizer_state_dict": self.cfgim.state_dict(),
                            # "normalizer": self.normalizer,
                        }
                        torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                        # generate a sample
                        render_count = 2
                        shape = (render_count, self.horizon, self.repr_dim)
                        print("Generating Sample")
                        # draw a music from the test dataset
                        (x, cond, genre, wavnames) = next(iter(test_data_loader))
                        cond = cond.to(self.accelerator.device)
                        print('exp_name 13',cfg.exp_name)
                        self.diffusion.render_sample(
                            shape,
                            cond[:render_count],
                            self.normalizer,
                            epoch,
                            os.path.join(cfg.project, cfg.exp_name , 'render'),  # render_out
                            os.path.join(cfg.project, cfg.exp_name , 'render'),  # fk_out
                            name=wavnames[:render_count],
                            sound=True,
                        )
                        print(f"[MODEL SAVED at Epoch {epoch}]")


                
            
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, mode="long", 
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode=mode,
            fk_out=fk_out,
            render=render
        )

    def render_sample_new(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, setmode="long", Returnfull=True,  Firstcond=None, genre=None, 
    ):
        _, cond, wavname, orikey = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        if genre is not None:
            genre = genre.repeat(shape[0]).squeeze()


        if cond.shape[-1] == 35:
            if isinstance(orikey, list):
                beatlist = orikey[1]
                orikey = orikey[0]
            else:
                beatlist = None

            if orikey[0].shape[-1] == 290:
                constraint={}
                constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], self.cfg.DATA_SETTING.nfeats).to(cond)
                constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], self.cfg.DATA_SETTING.nfeats).to(cond)
                for i in range(cond.shape[0]):
                    if self.cfg.get('Remode', 'default_value') != 'default_value':
                        if self.cfg.Remode == 'replace_key':
                            # constraint["value"][i, :, :7]  =  torch.from_numpy(orikey[i][:, :7]).to(cond)
                            try:
                                if self.cfg.Traj == 'TRAJ7':
                                    joint_index = [7,8 , 15,16,17,22,23]
                                elif self.cfg.Traj == 'TRAJ4':
                                    joint_index = [7,8 , 22,23]
                            except:
                                joint_index = [7,8 , 15,16,17,22,23]
                            jointsnum = 24
                            for index_ in joint_index:
                                constraint["value"][i, :, 7+ (index_-1)*3 : 7+index_*3]  =  torch.from_numpy(orikey[i][:, 7+ (index_-1)*3 : 7+index_*3]  ).to(cond) # 
                                constraint["value"][i, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  torch.from_numpy(orikey[i][:, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  ).to(cond)   # 
                                constraint["value"][i, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  torch.from_numpy(orikey[i][:, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  ).to(cond)

                    if beatlist is not None:
                        for beatidx in beatlist:
                            constraint["value"][i, beatidx-4:beatidx+4, :] = torch.from_numpy(orikey[i][beatidx-4:beatidx+4]).to(cond)
                            constraint["mask"][i, beatidx-4:beatidx+4,:] = 1 
                            constraint["mask"][i, beatidx-4:beatidx+4, :7] = 0 
                            
                    if Firstcond is not None:
                        constraint["value"][i, :4] = Firstcond
                        constraint["mask"][i, :4] = 1 
                        # constraint["mask"][i, 4:20] = 0 

            elif orikey[0].shape[-1] == 266:
                constraint={}
                constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], self.cfg.DATA_SETTING.nfeats).to(cond)
                constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], self.cfg.DATA_SETTING.nfeats).to(cond)
                for i in range(cond.shape[0]):
                    # 采用gpt的trans
                    if self.cfg.DATA_SETTING.nfeats == 338:
                        constraint["value"][i, :, :7+21*3] = torch.from_numpy(orikey[i][:, :7+21*3]).to(cond)
                        constraint["value"][i, :, 7+54*3+1:7+54*3+22*3] = torch.from_numpy(orikey[i][:,7+21*3+21*6+1:7+21*3+21*6+22*3]).to(cond)
                        constraint["mask"][i, :,:7+21*3] = 1 
                        constraint["mask"][i, :,7+54*3+1:7+54*3+22*3] = 1 
                    elif self.cfg.DATA_SETTING.nfeats == 266:
                        if self.cfg.get('Remode', 'default_value') != 'default_value':
                            if self.cfg.Remode == 'replace_key':
                                print('constraint["value"] device', constraint["value"].device)
                                # constraint["value"][i, :, :7]  =  torch.from_numpy(orikey[i][:, :7]).to(cond)
                                try:
                                    if self.cfg.Traj == 'TRAJ7':
                                        joint_index = [7,8 , 15,16,17,20,21]
                                    elif self.cfg.Traj == 'TRAJ4':
                                        joint_index = [7,8 , 20,21]
                                except:
                                    joint_index = [7,8 , 15,16,17,20,21]
                                jointsnum = 22
                                for index_ in joint_index:
                                    constraint["value"][i, :, 7+ (index_-1)*3 : 7+index_*3]  =  torch.from_numpy(orikey[i][:, 7+ (index_-1)*3 : 7+index_*3]  ).to(cond) # 
                                    constraint["value"][i, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  torch.from_numpy(orikey[i][:, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  ).to(cond)   # 
                                    constraint["value"][i, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  torch.from_numpy(orikey[i][:, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  ).to(cond)
                            elif self.cfg.Remode == 'add_key':
                                try:
                                    if self.cfg.Traj == 'TRAJ7':
                                        joint_index = [7,8 , 15,16,17,20,21]
                                    elif self.cfg.Traj == 'TRAJ4':
                                        joint_index = [7,8 , 20,21]
                                except:
                                    joint_index = [7,8 , 15,16,17,20,21]
                                
                                jointsnum = 22
                                for index_ in joint_index:
                                    constraint["value"][i, :, 7+ (index_-1)*3 : 7+index_*3]  +=  torch.from_numpy(orikey[i][:, 7+ (index_-1)*3 : 7+index_*3]  ).to(cond) # 
                                    constraint["value"][i, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  +=  torch.from_numpy(orikey[i][:, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  ).to(cond)   # 
                                    constraint["value"][i, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  +=  torch.from_numpy(orikey[i][:, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  ).to(cond)
                        if beatlist is not None: 
                            for beatidx in beatlist:
                                constraint["value"][i, beatidx-4:beatidx+4, :] = torch.from_numpy(orikey[i][beatidx-4:beatidx+4]).to(cond)
                                constraint["mask"][i, beatidx-4:beatidx+4,:] = 1 
                                constraint["mask"][i, beatidx-4:beatidx+4, :7] = 0 

                    if Firstcond is not None:
                        constraint["value"][i, :4] = Firstcond
                        constraint["mask"][i, :4] = 1 
                        constraint["mask"][i, 4:20] = 0 



        if Returnfull:
            return self.diffusion.render_sample(
                shape,
                cond[:render_count],
                genre,
                self.normalizer,
                label,
                render_dir,
                constraint = constraint,
                name=wavname[:render_count],
                sound=True,
                mode=setmode,
                fk_out=fk_out,
                render=render,
                Returnfull=Returnfull,
            )
        else:
            self.diffusion.render_sample(
                shape,
                cond[:render_count],
                genre,
                self.normalizer,
                label,
                render_dir,
                constraint = constraint,
                name=wavname[:render_count],
                sound=True,
                mode=setmode,
                fk_out=fk_out,
                render=render,
                Returnfull=Returnfull,
            )
        
