from ast import If
import copy
import os, sys
import pickle
from pathlib import Path
from functools import partial
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

sys.path.append('/data/lrh/project/dance/LongV2/LongV2Exp1/')
from dld.data.utils.preprocess import ax_from_6v, quat_slerp
# from vis import skeleton_render
from dld.data.utils.smplfk import SMPLX_Skeleton, ax_to_6v
from .utils import extract, make_beta_schedule
from dld.data.utils.motion_process import recover_from_ric266v_grad, recover_from_ricv_norotinit_grad, recover_from_ric266
from dld.DWT_IDWT.DWT_IDWT_layer import DWT_2D
from dld.losses.double_react import InterLoss

kinematic_chain22 = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21],
                 [9, 13, 16, 18, 20]]

kinematic_chain24 = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21, 23],
                 [9, 13, 16, 18, 20, 22]]

def identity(t, *args, **kwargs):
    return t

def replace_or_addkey(cfg, x_noisy, x_start):
    Remode = cfg.Remode
    try:
        Traj = cfg.Traj
    except:
        Traj = 'TRAJ7'

    if x_start.shape[-1] == 266:
        if Traj == 'TRAJ7':
            joint_index = [7,8 , 15,16,17,20,21]
        elif Traj == 'TRAJ4':
            joint_index = [7,8 , 20,21]
        jointsnum = 22
    elif x_start.shape[-1] == 290:
        if Traj == 'TRAJ7':
            joint_index = [7,8 , 15,16,17,22,23]
        elif Traj == 'TRAJ4':
            joint_index = [7,8 , 22,23]
        jointsnum = 24
    elif x_start.shape[-1] == 139 or x_start.shape[-1] == 151:
        if Traj == 'TRAJ8':
            joint_index = [1,2 , 4,5,16,17,18,19]
        elif Traj == 'TRAJ4':
            joint_index = [1,2 , 16,17]
        else:
            print('Traj', Traj)
            raise
        jointsnum = 22


    if Remode == 'replace_key':
        # 7 + 21*3 + 21*6 + 22*3 + 4
        # x_noisy[:, :, :7]  =  x_start[:, :, :7]
        for index_ in joint_index:          # :7+ (index_-1)*3 : 7+index_*3
            x_noisy[:, :, 7+ (index_-1)*3 : 7+index_*3]  =  x_start[:, :, 7+ (index_-1)*3 : 7+index_*3]   # 
            x_noisy[:, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  x_start[:, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]   # 
            x_noisy[:, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]

    elif Remode == 'replace_32_128':
        for index_ in joint_index:
            x_noisy[:, :4, 7+ (index_-1)*3 : 7+index_*3]  =  x_start[:, :4, 7+ (index_-1)*3 : 7+index_*3]
            x_noisy[:, :4, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  x_start[:, :4, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]
            x_noisy[:, :4, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, :4, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]

            x_noisy[:, -4:, 7+ (index_-1)*3 : 7+index_*3]  =  x_start[:, -4:, 7+ (index_-1)*3 : 7+index_*3]
            x_noisy[:, -4:, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  x_start[:, -4:, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]
            x_noisy[:, -4:, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, -4:, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]

            x_noisy[:, 28:36, 7+ (index_-1)*3 : 7+index_*3]  =  x_start[:, 28:36, 7+ (index_-1)*3 : 7+index_*3]
            x_noisy[:, 28:36, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  x_start[:, 28:36, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]
            x_noisy[:, 28:36, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, 28:36, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]


            x_noisy[:, 60:68, 7+ (index_-1)*3 : 7+index_*3]  =  x_start[:, 60:68, 7+ (index_-1)*3 : 7+index_*3]
            x_noisy[:, 60:68, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  x_start[:, 60:68, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]
            x_noisy[:, 60:68, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, 60:68, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]

            x_noisy[:, 92:100, 7+ (index_-1)*3 : 7+index_*3]  =  x_start[:, 92:100, 7+ (index_-1)*3 : 7+index_*3]
            x_noisy[:, 92:100, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  =  x_start[:, 92:100, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]
            x_noisy[:, 92:100, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, 92:100, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]
    elif Remode == 'replace_32_128_fea139':
        for index_ in joint_index:
            # x_noisy[:, :4, 7+ index_*3 : 7+(index_+1)*3]  =  x_start[:, :4, 7+ index_*3 : 7+(index_+1)*3]
            # x_noisy[:, -4:, 7+ index_*3 : 7+(index_+1)*3]  =  x_start[:, -4:, 7+ index_*3 : 7+(index_+1)*3]
            # x_noisy[:, 28:36, 7+ index_*3 : 7+(index_+1)*3]  =  x_start[:, 28:36, 7+ index_*3 : 7+(index_+1)*3]
            # x_noisy[:, 60:68, 7+ index_*3 : 7+(index_+1)*3]  =  x_start[:, 60:68, 7+ index_*3 : 7+(index_+1)*3]
            # x_noisy[:, 92:100, 7+ index_*3 : 7+(index_+1)*3]  =  x_start[:, 92:100, 7+ index_*3 : 7+(index_+1)*3]

            x_noisy[:, :4, 7+ index_*6 : 7+(index_+1)*6]  =  x_start[:, :4, 7+ index_*6 : 7+(index_+1)*6]
            x_noisy[:, -4:, 7+ index_*6 : 7+(index_+1)*6]  =  x_start[:, -4:, 7+ index_*6 : 7+(index_+1)*6]
            x_noisy[:, 28:36, 7+ index_*6 : 7+(index_+1)*6]  =  x_start[:, 28:36, 7+ index_*6 : 7+(index_+1)*6]
            x_noisy[:, 60:68, 7+ index_*6 : 7+(index_+1)*6]  =  x_start[:, 60:68, 7+ index_*6 : 7+(index_+1)*6]
            x_noisy[:, 92:100, 7+ index_*6 : 7+(index_+1)*6]  =  x_start[:, 92:100, 7+ index_*6 : 7+(index_+1)*6]


    elif Remode == 'add_key':
        # x_noisy[:, :, :7]  +=  x_start[:, :, :7]
        for index_ in joint_index:
            x_noisy[:, :, 7+ (index_-1)*3 : 7+index_*3]  +=  x_start[:, :, 7+ (index_-1)*3 : 7+index_*3]   # 
            x_noisy[:, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]  +=  x_start[:, :, 7+(jointsnum-1)*3 + (index_-1)*3 : 7+(jointsnum-1)*3 + index_*3]   # 
            x_noisy[:, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]  =  x_start[:, :, 7+(jointsnum-1)*3 + (jointsnum-1)*3+ index_*6:7+(jointsnum-1)*3 + (jointsnum-1)*3+ (index_+1)*6]   # 
    return x_noisy
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        cfg, 
        model,
        normalizer,
        horizon,
        repr_dim,
        smplx_model,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        dis_model=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.joints_num = cfg.DATA_SETTING.njoints
        self.loss_type = loss_type
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.normalizer = normalizer
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        if dis_model is not None:
            self.dis_model=dis_model
            self.master_model_dis = copy.deepcopy(self.dis_model)
        

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smplx_fk = smplx_model

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon      # 设置为不预测噪声，直接预测x

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss
        self.dwt = DWT_2D("haar")

            

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, genre, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, genre, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, genre, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, genre, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, genre, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, genre=genre, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    @torch.no_grad()
    def ddim_sample(self, shape, cond, genre, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):        # 
        for time, time_next in time_pairs: 
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x
    
    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, genre, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        if batch == 1:
            return self.ddim_sample(shape, cond, genre)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)
        
        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, genre, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        


    @torch.no_grad()
    def inpaint_key(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels
        print('x shape', x.shape)
        print('value shape', value.shape)
        if self.cfg.Replace:           
            x[:, :4, :]  =  value[:, :4, :]
            # x[:, -4:, :] =  value[:, -4:, :]

        x = replace_or_addkey(self.cfg, x, value)
        # x = value * mask + (1.0 - mask) * x

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):        # 
        # for time, time_next in time_pairs: 
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            if time > 0:
                x_start = value * mask + (1.0 - mask) * x_start
            # if time > 20:   
            #     x_start = value * mask + (1.0 - mask) * x_start

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            

            if self.cfg.Replace:           
                x[:, :4, :]  =  value[:, :4, :]
                # x[:, -4:, :] =  value[:, -4:, :]
            
            # if time > 0:   
                # x = value * mask + (1.0 - mask) * x
            x = replace_or_addkey(self.cfg, x, value)

            
        return x
        
        
    @torch.no_grad()
    def inpaint_soft_loop(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            if int(i) > 0:  # self.opt.hint:        # 数字越小控制soft hint效果越强
                # fill with i
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

                # sample x from step i to step i-1
                x, _ = self.p_sample(x, cond, genre, timesteps)      # p_sample
                # enforce constraint between each denoising step
                value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
                x = value_ * mask + (1.0 - mask) * x
                # x[:, :4, :] = value[:, :4, :]  * mask[:, :4, :]  + (1.0 - mask[:, :4, :] ) * x[:, :4, :] 
                # x[:, -4:, :] = value[:, -4:, :]  * mask[:, -4:, :]  + (1.0 - mask[:, -4:, :] ) * x[:, -4:, :]
            else:
                # fill with i
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                # sample x from step i to step i-1
                x, _ = self.p_sample(x, cond, genre, timesteps)
                # enforce constraint between each denoising step
                value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
                # x = value_ * mask + (1.0 - mask) * x
                x[:, :4, :] = value[:, :4, :]  * mask[:, :4, :]  + (1.0 - mask[:, :4, :] ) * x[:, :4, :] 
                x[:, -4:, :] = value[:, -4:, :]  * mask[:, -4:, :]  + (1.0 - mask[:, -4:, :] ) * x[:, -4:, :]
            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        

    @torch.no_grad()
    def inpaint_soft_ddim(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        print("inpaint_soft_ddim")
        # device = self.betas.device
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        # if batch == 1:
        #     return self.ddim_sample(shape, cond, genre)
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        # assert batch > 1
        # assert x.shape[1] % 2 == 0
        # half = x.shape[1] // 2
        x_start = None

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels
        # np.save('/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/abandon/value2.npy', value.detach().cpu().numpy())

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            # if time > 0:
            #     x_start = value * mask + (1.0 - mask) * x_start

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            # if time > 0:
            #     # the first half of each sequence is the second half of the previous one
            #     x[1:, :half] = x[:-1, half:]
            if time > 0:
                x = value * mask + (1.0 - mask) * x
            # x[:, :4, :] = value[:, :4, :]  * mask[:, :4, :]  + (1.0 - mask[:, :4, :] ) * x[:, :4, :] 
            # x[:, -4:, :] = value[:, -4:, :]  * mask[:, -4:, :]  + (1.0 - mask[:, -4:, :] ) * x[:, -4:, :]
        return x
    

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    def q_sample_res(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
             x_start  + extract(self.sqrt_alphas_cumprod, t, x_start.shape) * (y - x_start)
             + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            # + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )
    

    def loss_263(self, model_out, target, t): 
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        v_loss = self.loss_fn(model_out[..., 4 : (22 - 1) * 3 + 4], target[..., 4 : (22 - 1) * 3 + 4], reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
 
        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
            )
        total_loss, (mseloss, v_loss) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                })
        return loss_dict
    

    def loss_266_origin(self, model_out, target, t): 
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        v_loss = self.loss_fn(model_out[..., 7 : (22 - 1) * 3 + 7], target[..., 7 : (22 - 1) * 3 + 7], reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
            )
        total_loss, (mseloss, v_loss) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                })
        
        return loss_dict
    
    def loss_origin_addfre(self, model_out, target, t): 
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        model_fft =  torch.fft.fft(model_out, dim=0).to(model_out)
        target_fft =  torch.fft.fft(target, dim=0).to(model_out)
        
        model_vel = (model_out[..., 1:] - model_out[..., :-1])
        target_vel = (target[..., 1:] - target[..., :-1])

        model_acc = (model_vel[..., 1:] - model_vel[..., :-1])
        target_acc = (target_vel[..., 1:] - target_vel[..., :-1])

        v_loss = self.loss_fn(model_vel, target_vel, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        a_loss = self.loss_fn(model_acc, target_acc, reduction="none")
        a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
        a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)

        T = model_fft.shape[1]
        fft_loss_low = self.loss_fn(model_fft[:, :T//2], target_fft[:, :T//2], reduction="none")
        fft_loss_low = reduce(fft_loss_low, "b ... -> b (...)", "mean")
        fft_loss_low = fft_loss_low * extract(self.p2_loss_weight, t, fft_loss_low.shape)

        fft_loss_high = self.loss_fn(model_fft[:, T//2 :], target_fft[:, T//2:], reduction="none")
        fft_loss_high = reduce(fft_loss_high, "b ... -> b (...)", "mean")
        fft_loss_high = fft_loss_high * extract(self.p2_loss_weight, t, fft_loss_high.shape)

        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
                self.cfg.LOSS.LAMBDA_A              * a_loss.mean(),
                self.cfg.LOSS.LAMBDA_FFT_LOW              * fft_loss_low.mean(),
                self.cfg.LOSS.LAMBDA_FFT_HIGH              * fft_loss_high.mean(),
            )
        total_loss, (mseloss, v_loss, a_loss, fft_loss_low, fft_loss_high) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                    "a_loss": a_loss,
                    # "fft_loss": fft_loss,
                    "fft_loss_low": fft_loss_low,
                    "fft_loss_high": fft_loss_high,
                })

        
        return loss_dict


    def loss_with_wave(self, model_out, target, t): 
        B,T,C = model_out.shape
        if C == 266:
            jointsnum = 22
        elif C == 290:
            jointsnum = 24
        else:
            print('model_out', model_out.shape)
            raise('error of model_out shape')
        
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)
        mseloss = mseloss.mean()

        local_loss = self.loss_fn(model_out[..., : (jointsnum - 1) * 3 + 7], target[..., : (jointsnum - 1) * 3 + 7], reduction="none")
        local_loss = reduce(local_loss, "b ... -> b (...)", "mean")
        local_loss = local_loss * extract(self.p2_loss_weight, t, local_loss.shape)
        local_loss = local_loss.mean()

        
        
        # model_xzy = recover_from_ricv_norotinit_grad(model_out, jointsnum)
        # target_xzy = recover_from_ricv_norotinit_grad(target, jointsnum)

        if self.cfg.LOSS.LAMBDA_WAVE_LL + self.cfg.LOSS.LAMBDA_WAVE_LH + self.cfg.LOSS.LAMBDA_WAVE_HL + self.cfg.LOSS.LAMBDA_WAVE_HH>0:
            mll, mlh, mhl, mhh = self.dwt(model_out.permute(0,3,1,2).contiguous())
            tll, tlh, thl, thh = self.dwt(target.permute(0,3,1,2).contiguous())
        
        if self.cfg.LOSS.LAMBDA_V>0:
            model_vel = (model_out[..., 1:, 7 : (jointsnum - 1) * 3 + 7] - model_out[..., :-1, 7 : (jointsnum - 1) * 3 + 7])
            target_vel = (target[..., 1:, 7 : (jointsnum - 1) * 3 + 7] - target[..., :-1, 7 : (jointsnum - 1) * 3 + 7])
            v_loss = self.loss_fn(model_vel, target_vel, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
            v_loss = v_loss.mean()
        else:
            v_loss = 0.0

        if self.cfg.LOSS.LAMBDA_A>0:
            model_acc = (model_vel[..., 1:] - model_vel[..., :-1])
            target_acc = (target_vel[..., 1:] - target_vel[..., :-1])

            a_loss = self.loss_fn(model_acc, target_acc, reduction="none")
            a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
            a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)
            a_loss = a_loss.mean()
        else:
            a_loss  = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LL>0:
            wave_loss_ll = self.loss_fn(mll, tll, reduction="none")
            wave_loss_ll = reduce(wave_loss_ll, "b ... -> b (...)", "mean")
            wave_loss_ll = wave_loss_ll * extract(self.p2_loss_weight, t, wave_loss_ll.shape)
            wave_loss_ll = wave_loss_ll.mean()
        else:
            wave_loss_ll = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LH>0:
            wave_loss_lh = self.loss_fn(mlh, tlh, reduction="none")
            wave_loss_lh = reduce(wave_loss_lh, "b ... -> b (...)", "mean")
            wave_loss_lh = wave_loss_lh * extract(self.p2_loss_weight, t, wave_loss_lh.shape)
            wave_loss_lh = wave_loss_lh.mean()
        else:
            wave_loss_lh = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_HL>0:
            wave_loss_hl = self.loss_fn(mhl, thl, reduction="none")
            wave_loss_hl = reduce(wave_loss_hl, "b ... -> b (...)", "mean")
            wave_loss_hl = wave_loss_hl * extract(self.p2_loss_weight, t, wave_loss_hl.shape)
            wave_loss_hl = wave_loss_hl.mean()
        else:
            wave_loss_hl = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LH>0:
            wave_loss_hh = self.loss_fn(mhh, thh, reduction="none")
            wave_loss_hh = reduce(wave_loss_hh, "b ... -> b (...)", "mean")
            wave_loss_hh = wave_loss_hh * extract(self.p2_loss_weight, t, wave_loss_hh.shape)
            wave_loss_hh = wave_loss_hh.mean()
        else:
            wave_loss_hh = 0.0

       
        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss,
                self.cfg.LOSS.LAMBDA_LOCAL          * local_loss,
                self.cfg.LOSS.LAMBDA_V              * v_loss,
                self.cfg.LOSS.LAMBDA_A              * a_loss,
                self.cfg.LOSS.LAMBDA_WAVE_LL             * wave_loss_ll,
                self.cfg.LOSS.LAMBDA_WAVE_LH              * wave_loss_lh,
                self.cfg.LOSS.LAMBDA_WAVE_HL              * wave_loss_hl,
                self.cfg.LOSS.LAMBDA_WAVE_HH              * wave_loss_hh,
            )
        total_loss, (mseloss, local_loss, v_loss, a_loss, wave_loss_ll, wave_loss_lh,wave_loss_hl,wave_loss_hh) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "local_loss": local_loss,
                    "v_loss": v_loss,
                    "a_loss": a_loss,
                    "wave_loss_ll": wave_loss_ll,
                    "wave_loss_lh": wave_loss_lh,
                    "wave_loss_hl": wave_loss_hl,
                    "wave_loss_hh": wave_loss_hh,
                })

        
        return loss_dict

    def loss_vec_position(self, model_out, target, t): 
        B,T,C = model_out.shape
        if C == 266:
            jointsnum = 22
            kinematic_chain = kinematic_chain22
        elif C == 290:
            jointsnum = 24
            kinematic_chain = kinematic_chain24
        else:
            print('model_out', model_out.shape)
            raise('error of model_out shape')
        
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)
        mseloss = mseloss.mean()

        local_loss = self.loss_fn(model_out[..., 7 : (jointsnum - 1) * 3 + 7], target[...,7 : (jointsnum - 1) * 3 + 7], reduction="none")
        local_loss = reduce(local_loss, "b ... -> b (...)", "mean")
        local_loss = local_loss * extract(self.p2_loss_weight, t, local_loss.shape)
        local_loss = local_loss.mean()

        model_out_ = self.normalizer.unnormalize(model_out)
        target_ = self.normalizer.unnormalize(target)
        model_xyz = recover_from_ric266(model_out_, jointsnum)
        target_xyz = recover_from_ric266(target_, jointsnum)

        # local_loss = self.loss_fn(model_xyz, target_xyz, reduction="none")
        # local_loss = reduce(local_loss, "b ... -> b (...)", "mean")
        # local_loss = local_loss * extract(self.p2_loss_weight, t, local_loss.shape)
        # local_loss = local_loss.mean()

        if self.cfg.LOSS.LAMBDA_BONE>0:
            pred_bones = []
            tgt_bones = []
            for chain in kinematic_chain:
                for i, joint in enumerate(chain[:-1]):
                    pred_bone = (model_xyz[..., chain[i], :] - model_xyz[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                            keepdim=True)  # [B,T,P,1]
                    tgt_bone = (target_xyz[..., chain[i], :] - target_xyz[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                        keepdim=True)
                    pred_bones.append(pred_bone)
                    tgt_bones.append(tgt_bone)
            pred_bones = torch.cat(pred_bones, dim=-1)
            tgt_bones = torch.cat(tgt_bones, dim=-1)

            bone_loss = self.loss_fn(pred_bones, tgt_bones, reduction="none")
            bone_loss = reduce(bone_loss, "b ... -> b (...)", "mean")
            bone_loss = bone_loss * extract(self.p2_loss_weight, t, bone_loss.shape)
            bone_loss = bone_loss.mean()
        else:
            bone_loss = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LL + self.cfg.LOSS.LAMBDA_WAVE_LH + self.cfg.LOSS.LAMBDA_WAVE_HL + self.cfg.LOSS.LAMBDA_WAVE_HH>0:
            mll, mlh, mhl, mhh = self.dwt(model_xyz.permute(0,3,1,2).contiguous())
            tll, tlh, thl, thh = self.dwt(target_xyz.permute(0,3,1,2).contiguous())
        
        if self.cfg.LOSS.LAMBDA_V>0:
            model_vel = model_xyz[:, 1:] - model_xyz[:, :-1]
            target_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
            v_loss = self.loss_fn(model_vel, target_vel, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
            v_loss = v_loss.mean()
        else:
            v_loss = 0.0

        if self.cfg.LOSS.LAMBDA_A>0:
            model_acc = (model_vel[:, 1:] - model_vel[:, :-1])
            target_acc = (target_vel[:, 1:] - target_vel[:, :-1])

            a_loss = self.loss_fn(model_acc, target_acc, reduction="none")
            a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
            a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)
            a_loss = a_loss.mean()
        else:
            a_loss  = 0.0

        if self.cfg.LOSS.LAMBDA_FC>0:
            fids = [7, 10, 8, 11]
            feet_vel = model_xyz[:, 1:, fids, :] - model_xyz[:, :-1, fids,:]
            feet_h = model_xyz[:, :-1, fids, 1]
            contact = self.foot_detect(feet_vel, feet_h, 0.001) # refer InterGen
            contact = contact.unsqueeze(-1).repeat(1,1,1,3)
            feet_vel = feet_vel*contact
            # feet_vel[~contact] = 0 
            foot_loss = self.loss_fn(  
                feet_vel, torch.zeros_like(feet_vel), reduction="none"
            )
            foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
            foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)
        else:
            foot_loss = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LL>0:
            wave_loss_ll = self.loss_fn(mll, tll, reduction="none")
            wave_loss_ll = reduce(wave_loss_ll, "b ... -> b (...)", "mean")
            wave_loss_ll = wave_loss_ll * extract(self.p2_loss_weight, t, wave_loss_ll.shape)
            wave_loss_ll = wave_loss_ll.mean()
        else:
            wave_loss_ll = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LH>0:
            wave_loss_lh = self.loss_fn(mlh, tlh, reduction="none")
            wave_loss_lh = reduce(wave_loss_lh, "b ... -> b (...)", "mean")
            wave_loss_lh = wave_loss_lh * extract(self.p2_loss_weight, t, wave_loss_lh.shape)
            wave_loss_lh = wave_loss_lh.mean()
        else:
            wave_loss_lh = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_HL>0:
            wave_loss_hl = self.loss_fn(mhl, thl, reduction="none")
            wave_loss_hl = reduce(wave_loss_hl, "b ... -> b (...)", "mean")
            wave_loss_hl = wave_loss_hl * extract(self.p2_loss_weight, t, wave_loss_hl.shape)
            wave_loss_hl = wave_loss_hl.mean()
        else:
            wave_loss_hl = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LH>0:
            wave_loss_hh = self.loss_fn(mhh, thh, reduction="none")
            wave_loss_hh = reduce(wave_loss_hh, "b ... -> b (...)", "mean")
            wave_loss_hh = wave_loss_hh * extract(self.p2_loss_weight, t, wave_loss_hh.shape)
            wave_loss_hh = wave_loss_hh.mean()
        else:
            wave_loss_hh = 0.0

       
        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss,
                self.cfg.LOSS.LAMBDA_LOCAL          * local_loss,
                self.cfg.LOSS.LAMBDA_BONE           * bone_loss,
                self.cfg.LOSS.LAMBDA_FC             * foot_loss,
                self.cfg.LOSS.LAMBDA_V              * v_loss,
                self.cfg.LOSS.LAMBDA_A              * a_loss,
                self.cfg.LOSS.LAMBDA_WAVE_LL             * wave_loss_ll,
                self.cfg.LOSS.LAMBDA_WAVE_LH              * wave_loss_lh,
                self.cfg.LOSS.LAMBDA_WAVE_HL              * wave_loss_hl,
                self.cfg.LOSS.LAMBDA_WAVE_HH              * wave_loss_hh,
            )
        total_loss, (mseloss, local_loss, bone_loss, foot_loss, v_loss, a_loss, wave_loss_ll, wave_loss_lh,wave_loss_hl,wave_loss_hh) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "local_loss": local_loss,
                    "bone_loss": bone_loss,
                    "foot_loss": foot_loss,
                    "v_loss": v_loss,
                    "a_loss": a_loss,
                    "wave_loss_ll": wave_loss_ll,
                    "wave_loss_lh": wave_loss_lh,
                    "wave_loss_hl": wave_loss_hl,
                    "wave_loss_hh": wave_loss_hh,
                })

        
        return loss_dict
    

    def loss_vec_vel(self, model_out, target, t): 
        B,T,C = model_out.shape
        if C == 266:
            jointsnum = 22
            kinematic_chain = kinematic_chain22
        elif C == 290:
            jointsnum = 24
            kinematic_chain = kinematic_chain24
        else:
            print('model_out', model_out.shape)
            raise('error of model_out shape')
        
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)
        mseloss = mseloss.mean()

        local_loss = self.loss_fn(model_out[..., : (jointsnum - 1) * 3 + 7], target[..., : (jointsnum - 1) * 3 + 7], reduction="none")
        local_loss = reduce(local_loss, "b ... -> b (...)", "mean")
        local_loss = local_loss * extract(self.p2_loss_weight, t, local_loss.shape)
        local_loss = local_loss.mean()

        model_xyz = torch.cat( [torch.zeros([B,T,3]).to(model_out), model_out[...,7 : (jointsnum - 1) * 3 + 7]],dim=-1 ).reshape(B,T,jointsnum,3)
        target_xyz = torch.cat( [torch.zeros([B,T,3]).to(model_out), target[...,7 : (jointsnum - 1) * 3 + 7]],dim=-1 ).reshape(B,T,jointsnum,3)

        # local_loss = self.loss_fn(model_xyz, target_xyz, reduction="none")
        # local_loss = reduce(local_loss, "b ... -> b (...)", "mean")
        # local_loss = local_loss * extract(self.p2_loss_weight, t, local_loss.shape)
        # local_loss = local_loss.mean()

        if self.cfg.LOSS.LAMBDA_BONE>0:
            pred_bones = []
            tgt_bones = []
            for chain in kinematic_chain:
                for i, joint in enumerate(chain[:-1]):
                    pred_bone = (model_xyz[..., chain[i], :] - model_xyz[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                            keepdim=True)  # [B,T,P,1]
                    tgt_bone = (target_xyz[..., chain[i], :] - target_xyz[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                        keepdim=True)
                    pred_bones.append(pred_bone)
                    tgt_bones.append(tgt_bone)
            pred_bones = torch.cat(pred_bones, dim=-1)
            tgt_bones = torch.cat(tgt_bones, dim=-1)

            bone_loss = self.loss_fn(pred_bones, tgt_bones, reduction="none")
            bone_loss = reduce(bone_loss, "b ... -> b (...)", "mean")
            bone_loss = bone_loss * extract(self.p2_loss_weight, t, bone_loss.shape)
            bone_loss = bone_loss.mean()
        else:
            bone_loss = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LL + self.cfg.LOSS.LAMBDA_WAVE_LH + self.cfg.LOSS.LAMBDA_WAVE_HL + self.cfg.LOSS.LAMBDA_WAVE_HH>0:
            mll, mlh, mhl, mhh = self.dwt(model_xyz.permute(0,3,1,2).contiguous())
            tll, tlh, thl, thh = self.dwt(target_xyz.permute(0,3,1,2).contiguous())
        
        if self.cfg.LOSS.LAMBDA_V>0:
            model_vel = model_out[..., 1:, : (jointsnum - 1) * 3 + 7] - model_out[..., :-1, : (jointsnum - 1) * 3 + 7]
            target_vel = target[..., 1:, : (jointsnum - 1) * 3 + 7] - target[..., :-1, : (jointsnum - 1) * 3 + 7]
            v_loss = self.loss_fn(model_vel, target_vel, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
            v_loss = v_loss.mean()
        else:
            v_loss = 0.0

        if self.cfg.LOSS.LAMBDA_A>0:
            model_acc = (model_vel[:, 1:] - model_vel[:, :-1])
            target_acc = (target_vel[:, 1:] - target_vel[:, :-1])

            a_loss = self.loss_fn(model_acc, target_acc, reduction="none")
            a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
            a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)
            a_loss = a_loss.mean()
        else:
            a_loss  = 0.0

        # if self.cfg.LOSS.LAMBDA_FC>0:
        #     fids = [7, 10, 8, 11]
        #     feet_vel = model_xyz[:, 1:, fids, :] - model_xyz[:, :-1, fids,:]
        #     feet_h = model_xyz[:, :-1, fids, 1]
        #     contact = self.foot_detect(feet_vel, feet_h, 0.001) # refer InterGen
        #     contact = contact.unsqueeze(-1).repeat(1,1,1,3)
        #     feet_vel = feet_vel*contact
        #     # feet_vel[~contact] = 0 
        #     foot_loss = self.loss_fn(  
        #         feet_vel, torch.zeros_like(feet_vel), reduction="none"
        #     )
        #     foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
        #     foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)
        # else:
        #     foot_loss = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LL>0:
            wave_loss_ll = self.loss_fn(mll, tll, reduction="none")
            wave_loss_ll = reduce(wave_loss_ll, "b ... -> b (...)", "mean")
            wave_loss_ll = wave_loss_ll * extract(self.p2_loss_weight, t, wave_loss_ll.shape)
            wave_loss_ll = wave_loss_ll.mean()
        else:
            wave_loss_ll = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LH>0:
            wave_loss_lh = self.loss_fn(mlh, tlh, reduction="none")
            wave_loss_lh = reduce(wave_loss_lh, "b ... -> b (...)", "mean")
            wave_loss_lh = wave_loss_lh * extract(self.p2_loss_weight, t, wave_loss_lh.shape)
            wave_loss_lh = wave_loss_lh.mean()
        else:
            wave_loss_lh = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_HL>0:
            wave_loss_hl = self.loss_fn(mhl, thl, reduction="none")
            wave_loss_hl = reduce(wave_loss_hl, "b ... -> b (...)", "mean")
            wave_loss_hl = wave_loss_hl * extract(self.p2_loss_weight, t, wave_loss_hl.shape)
            wave_loss_hl = wave_loss_hl.mean()
        else:
            wave_loss_hl = 0.0

        if self.cfg.LOSS.LAMBDA_WAVE_LH>0:
            wave_loss_hh = self.loss_fn(mhh, thh, reduction="none")
            wave_loss_hh = reduce(wave_loss_hh, "b ... -> b (...)", "mean")
            wave_loss_hh = wave_loss_hh * extract(self.p2_loss_weight, t, wave_loss_hh.shape)
            wave_loss_hh = wave_loss_hh.mean()
        else:
            wave_loss_hh = 0.0

       
        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss,
                self.cfg.LOSS.LAMBDA_LOCAL          * local_loss,
                self.cfg.LOSS.LAMBDA_BONE           * bone_loss,
                self.cfg.LOSS.LAMBDA_V              * v_loss,
                self.cfg.LOSS.LAMBDA_A              * a_loss,
                self.cfg.LOSS.LAMBDA_WAVE_LL             * wave_loss_ll,
                self.cfg.LOSS.LAMBDA_WAVE_LH              * wave_loss_lh,
                self.cfg.LOSS.LAMBDA_WAVE_HL              * wave_loss_hl,
                self.cfg.LOSS.LAMBDA_WAVE_HH              * wave_loss_hh,
            )
        total_loss, (mseloss, local_loss, v_loss, bone_loss, a_loss, wave_loss_ll, wave_loss_lh,wave_loss_hl,wave_loss_hh) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "local_loss": local_loss,
                    "bone_loss": bone_loss,
                    "v_loss": v_loss,
                    "a_loss": a_loss,
                    "wave_loss_ll": wave_loss_ll,
                    "wave_loss_lh": wave_loss_lh,
                    "wave_loss_hl": wave_loss_hl,
                    "wave_loss_hh": wave_loss_hh,
                })

        
        return loss_dict
    

    def loss_266(self, model_out, target, t): 
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        # v_loss = self.loss_fn(model_out[..., 7 : (22 - 1) * 3 + 7], target[..., 7 : (22 - 1) * 3 + 7], reduction="none")
        # v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        # v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
 
        model_out = self.normalizer.unnormalize(model_out)
        target = self.normalizer.unnormalize(target)
        model_xyz = recover_from_ric266v_grad(model_out, 55)
        target_xyz = recover_from_ric266v_grad(target, 55)

        target_vel = (target_xyz[..., 1:] - target_xyz[..., :-1])
        model_vel = (model_xyz[..., 1:] - model_xyz[..., :-1])

        v_loss = self.loss_fn(model_vel, target_vel, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        target_acc = (target_vel[..., 1:] - target_vel[..., :-1])
        model_out_acc = (model_vel[..., 1:] - model_vel[..., :-1])
        a_loss = self.loss_fn(model_out_acc, target_acc, reduction="none")
        a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
        a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)

        fids = [7, 10, 8, 11]
        feet_vel = model_xyz[:, 1:, fids, :] - model_xyz[:, :-1, fids,:]
        feet_h = model_xyz[:, :-1, fids, 1]
        contact = self.foot_detect(feet_vel, feet_h, 0.001)
        contact = contact.unsqueeze(-1).repeat(1,1,1,3)
        feet_vel = feet_vel*contact
        # feet_vel[~contact] = 0 
        foot_loss = self.loss_fn(  
            feet_vel, torch.zeros_like(feet_vel), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
        foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)

        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
                self.cfg.LOSS.LAMBDA_A              * a_loss.mean(),
                self.cfg.LOSS.LAMBDA_FC             * foot_loss.mean(),
            )
        total_loss, (mseloss, v_loss, a_loss, foot_loss) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                    "a_loss": a_loss,
                    "foot_loss": foot_loss,
                })
        
        return loss_dict
    
    def loss_338(self, model_out, target, t): 
        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        model_out = self.normalizer.unnormalize(model_out)
        target = self.normalizer.unnormalize(target)
        model_xyz = recover_from_smplx_v_grad(model_out, 55)
        target_xyz = recover_from_smplx_v_grad(target, 55)

        target_vel = (target_xyz[..., 1:] - target_xyz[..., :-1])
        model_vel = (model_xyz[..., 1:] - model_xyz[..., :-1])

        v_loss = self.loss_fn(model_vel, target_vel, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)


        target_acc = (target_vel[..., 1:] - target_vel[..., :-1])
        model_out_acc = (model_vel[..., 1:] - model_vel[..., :-1])
        a_loss = self.loss_fn(model_out_acc, target_acc, reduction="none")
        a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
        a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)

        fids = [7, 10, 8, 11]
        feet_vel = model_xyz[:, 1:, fids, :] - model_xyz[:, :-1, fids,:]
        feet_h = model_xyz[:, :-1, fids, 1]
        contact = self.foot_detect(feet_vel, feet_h, 0.001)
        contact = contact.unsqueeze(-1).repeat(1,1,1,3)
        # print('feet_vel', feet_vel.shape)
        # print('contact', contact.shape)
        feet_vel = feet_vel*contact
        # feet_vel[~contact] = 0 
        foot_loss = self.loss_fn(  
            feet_vel, torch.zeros_like(feet_vel), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
        foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)

        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
                self.cfg.LOSS.LAMBDA_A              * a_loss.mean(),
                self.cfg.LOSS.LAMBDA_FC             * foot_loss.mean(),
            )
        total_loss, (mseloss, v_loss, a_loss, foot_loss) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                    "a_loss": a_loss,
                    "foot_loss": foot_loss,
                })
        return loss_dict

    def foot_detect(self, feet_vel, feet_h, thres):
        velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(
            [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

        feet_x = (feet_vel[..., 0]) ** 2
        feet_y = (feet_vel[..., 1]) ** 2
        feet_z = (feet_vel[..., 2]) ** 2

        contact = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)).float()
        return contact
    
    def loss_263_Coarse(self, model_out, target, t): 
        B, T, C = model_out.shape
        model_out = model_out.reshape(-1, 8, C)
        target = target.reshape(-1, 8, C)
        t = t.repeat(int(target.shape[0]//B)  )

        mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        v_loss = self.loss_fn(model_out[..., 4 : (22 - 1) * 3 + 4], target[..., 4 : (22 - 1) * 3 + 4], reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
 
        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
            )
        total_loss, (mseloss, v_loss) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                })
        return loss_dict


    def loss_266_double_react(self, model_out, cond, target, t): 
        input_motion = cond[..., 35:]
        if input_motion.shape[-1] == 266:
            joints_num = 22
        elif input_motion.shape[-1] == 4268:
            joints_num = 710
        input_pos, input_ang = input_motion[..., :3], input_motion[..., 3:4]
        input_root = torch.cat([input_pos, input_ang],dim=-1)
        pred_pos, pred_ang = model_out[..., :3], model_out[..., 3:4]
        pred_root = torch.cat([pred_pos, pred_ang],dim=-1)
        gt_pos, gt_ang = target[..., :3], target[..., 3:4]
        gt_root = torch.cat([gt_pos, gt_ang],dim=-1)

        loss_root_g = self.loss_fn(pred_root, gt_root, reduction="none")           
        loss_root_g = reduce(loss_root_g, "b ... -> b (...)", "mean")
        loss_root_g = loss_root_g * extract(self.p2_loss_weight, t, loss_root_g.shape)

        # velocity loss
        pred_root_v = pred_root[:, 1:] - pred_root[:, :-1]
        gt_root_v = gt_root[:, 1:] - gt_root[:, :-1]
        loss_root_v = self.loss_fn(pred_root_v, gt_root_v, reduction="none")           
        loss_root_v = reduce(loss_root_v, "b ... -> b (...)", "mean")
        loss_root_v = loss_root_v * extract(self.p2_loss_weight, t, loss_root_v.shape)

        # relative_loss
        pred_root_r = pred_root - input_root
        gt_root_r = gt_root - input_root
        loss_root_r = self.loss_fn(pred_root_r, gt_root_r, reduction="none")           
        loss_root_r = reduce(loss_root_r, "b ... -> b (...)", "mean")
        loss_root_r = loss_root_r * extract(self.p2_loss_weight, t, loss_root_r.shape)

        mseloss = self.loss_fn(model_out[..., 4:], target[..., 4:], reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        v_loss = self.loss_fn(model_out[..., 4 : (joints_num - 1) * 3 + 4], target[..., 4 : (joints_num - 1) * 3 + 4], reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        LAMBDA_LOSS        = self.cfg.LOSS.LAMBDA_LOSS        
        LAMBDA_V           = self.cfg.LOSS.LAMBDA_V           
        LAMBDA_ROOT_Global = self.cfg.LOSS.LAMBDA_ROOT_Global 
        LAMBDA_ROOT_V      = self.cfg.LOSS.LAMBDA_ROOT_V      
        LAMBDA_ROOT_R      = self.cfg.LOSS.LAMBDA_ROOT_R      

        losses = (
                LAMBDA_LOSS           * mseloss.mean(),
                LAMBDA_V              * v_loss.mean(),
                LAMBDA_ROOT_Global    * loss_root_g.mean(),
                LAMBDA_ROOT_V         * loss_root_v.mean(),
                LAMBDA_ROOT_R         * loss_root_r.mean(),
            )
        total_loss, (mseloss, v_loss, root_Global, root_V, root_R) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                    "ROOT_Global": root_Global,
                    "ROOT_V": root_V,
                    "ROOT_R": root_R,
                })
        return loss_dict
    

    def loss_double_react(self, model_out, cond, target, t): 
        # mseloss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        # mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        # mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)
        

        # loss_dict  = {}
        # loss_dict['loss'] = mseloss.mean()

        interloss_manager = InterLoss(self.cfg, self.loss_type, nb_joints=self.joints_num, normalizer=self.normalizer, p2_loss_weight=self.p2_loss_weight)
        interloss_manager.forward(model_out, cond, target, t)
        # loss_dict = self.interloss_manager.forward(model_out, cond, target, t)
        loss_dict = {}
        loss_dict.update(interloss_manager.losses)
        return loss_dict
    

    def loss_266_double(self, model_out, target, t): 
        assert model_out.shape[-1] == 266*2
        pred_motion_a = model_out[..., :266]
        pred_motion_b = model_out[..., 266:]
        pred_pos_a, pred_ang_a = pred_motion_a[..., :3], pred_motion_a[..., 3:4]
        pred_root_a = torch.cat([pred_pos_a, pred_ang_a],dim=-1)
        pred_pos_b, pred_ang_b = pred_motion_b[..., :3], pred_motion_b[..., 3:4]
        pred_root_b = torch.cat([pred_pos_b, pred_ang_b],dim=-1)

        gt_motion_a = target[..., :266]
        gt_motion_b = target[..., 266:]
        gt_pos_a, gt_ang_a = gt_motion_a[..., :3], gt_motion_a[..., 3:4]
        gt_root_a = torch.cat([gt_pos_a, gt_ang_a],dim=-1)
        gt_pos_b, gt_ang_b = gt_motion_b[..., :3], gt_motion_b[..., 3:4]
        gt_root_b = torch.cat([gt_pos_b, gt_ang_b],dim=-1)

        loss_root_g = self.loss_fn(pred_root_a+pred_root_b, gt_root_a+gt_root_b, reduction="none")           
        loss_root_g = reduce(loss_root_g, "b ... -> b (...)", "mean")
        loss_root_g = loss_root_g * extract(self.p2_loss_weight, t, loss_root_g.shape)

        # velocity loss
        pred_root_a_v = pred_root_a[:, 1:] - pred_root_a[:, :-1]
        gt_root_a_v = gt_root_a[:, 1:] - gt_root_a[:, :-1]
        pred_root_b_v = pred_root_b[:, 1:] - pred_root_a[:, :-1]
        gt_root_b_v = gt_root_b[:, 1:] - gt_root_a[:, :-1]
        loss_root_v = self.loss_fn(pred_root_a_v+pred_root_b_v, gt_root_a_v+gt_root_b_v, reduction="none")           
        loss_root_v = reduce(loss_root_v, "b ... -> b (...)", "mean")
        loss_root_v = loss_root_v * extract(self.p2_loss_weight, t, loss_root_v.shape)

        # relative_loss
        pred_root_ar = pred_root_b - pred_root_a
        gt_root_ar = gt_root_b - gt_root_a
        loss_root_r = self.loss_fn(pred_root_ar, gt_root_ar, reduction="none")           
        loss_root_r = reduce(loss_root_r, "b ... -> b (...)", "mean")
        loss_root_r = loss_root_r * extract(self.p2_loss_weight, t, loss_root_r.shape)

        mseloss = self.loss_fn(pred_motion_a[..., 4:]+pred_motion_b[..., 4:], gt_motion_a[..., 4:]+gt_motion_b[..., 4:], reduction="none")            #  mse loss
        mseloss = reduce(mseloss, "b ... -> b (...)", "mean")
        mseloss = mseloss * extract(self.p2_loss_weight, t, mseloss.shape)

        v_loss = self.loss_fn(pred_motion_a[..., 4 : (22 - 1) * 3 + 4]+pred_motion_b[..., 4 : (22 - 1) * 3 + 4], gt_motion_a[..., 4 : (22 - 1) * 3 + 4]+gt_motion_b[..., 4 : (22 - 1) * 3 + 4], reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape) 

        losses = (
                self.cfg.LOSS.LAMBDA_LOSS           * mseloss.mean(),
                self.cfg.LOSS.LAMBDA_V              * v_loss.mean(),
                self.cfg.LOSS.LAMBDA_ROOT_Global    * loss_root_g.mean(),
                self.cfg.LOSS.LAMBDA_ROOT_V         * loss_root_v.mean(),
                self.cfg.LOSS.LAMBDA_ROOT_R         * loss_root_r.mean(),
            )
        total_loss, (mseloss, v_loss, root_Global, root_V, root_R) = sum(losses), losses
        loss_dict = {}
        loss_dict.update({
                    "loss": total_loss,
                    "mseloss": mseloss,
                    "v_loss": v_loss,
                    "ROOT_Global": root_Global,
                    "ROOT_V": root_V,
                    "ROOT_R": root_R,
                })
        return loss_dict
    

    def smpl_loss(self, model_out_ori, target_ori, t):
        # full reconstruction loss
        loss = self.loss_fn(model_out_ori, target_ori, reduction="none")            # mse loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # velocity loss
        target_v = target_ori[:, 1:, 4:] - target_ori[:, :-1, 4:]
        model_out_v = model_out_ori[:, 1:, 4:] - model_out_ori[:, :-1, 4:]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out_ori.shape      
        # unnormalize
        if self.normalizer is not None:
            model_out_ori = self.normalizer.unnormalize(model_out_ori)
            target_ori = self.normalizer.unnormalize(target_ori)
        # split off contact from the rest
        model_contact, model_out = torch.split(model_out_ori, (4, model_out_ori.shape[2] - 4), dim=2)  # 前4维是foot contact
        target_contact, target = torch.split(target_ori, (4, target_ori.shape[2] - 4), dim=2)       # b, length, jxc
        
        # model_x为root position, model_q为rotation
        model_x = model_out[:, :, :3]   # root position
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))
        b, s, nums, c_ = model_q.shape

        model_xp = self.smplx_fk.forward(model_q, model_x)
        target_xp = self.smplx_fk.forward(target_q, target_x)
        model_xp = model_xp.view(b, s, -1, 3)
        target_xp = target_xp.view(b, s, -1, 3) 

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0               # 不计算动态帧
        foot_loss = self.loss_fn(                   # 静态的foot，让它的速度为0
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")


        losses = (
            self.cfg.LOSS.LAMBDA_MSE  * loss.mean(),
            self.cfg.LOSS.LAMBDA_V    * v_loss.mean(),
            self.cfg.LOSS.LAMBDA_FK   * fk_loss.mean(),
            self.cfg.LOSS.LAMBDA_FOOT * foot_loss.mean(),
        )

        loss_dict = {}
        loss_dict.update({
                    "loss": sum(losses),
                    "mseloss": losses[0],
                    "Vloss": losses[1],
                    "fk_loss": losses[2],
                    "foot_loss": losses[3],
                })
        
        return loss_dict
    


    def smpl_loss_relative(self, model_out_ori, target_ori, t):
        # full reconstruction loss
        loss = self.loss_fn(model_out_ori, target_ori, reduction="none")            # mse loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)


        # velocity loss
        target_v = target_ori[:, 1:, 4:] - target_ori[:, :-1, 4:]
        model_out_v = model_out_ori[:, 1:, 4:] - model_out_ori[:, :-1, 4:]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out_ori.shape      
        # unnormalize
        if self.normalizer is not None:
            model_out_ori = self.normalizer.unnormalize(model_out_ori)
            target_ori = self.normalizer.unnormalize(target_ori)
        # split off contact from the rest
        model_contact, model_out = torch.split(model_out_ori, (4, model_out_ori.shape[2] - 4), dim=2)  # 前4维是foot contact
        target_contact, target = torch.split(target_ori, (4, target_ori.shape[2] - 4), dim=2)       # b, length, jxc


        # model_x为root position, model_q为rotation
        model_x = model_out[:, :, :3]   # root position
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))
        b, s, nums, c_ = model_q.shape

        model_xp = self.smplx_fk.forward(model_q, model_x)
        target_xp = self.smplx_fk.forward(target_q, target_x)
        model_xp = model_xp.view(b, s, -1, 3)
        target_xp = target_xp.view(b, s, -1, 3)

        # model_xp[:,:,:,:] = model_xp[:,:,:,:] - model_xp[:,:,0:1,:]
        # target_xp[:,:,:,:] = target_xp[:,:,:,:] - target_xp[:,:,0:1,:]    # B,T,J,3
        fk_loss = self.loss_fn(model_xp[:,:,:,:] - model_xp[:,:,0:1,:], target_xp[:,:,:,:] - target_xp[:,:,0:1,:], reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]       # l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx].clone()  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v_fc = model_foot_v.clone()
        model_foot_v[~static_idx] = 0               # 不计算动态帧
        foot_loss = self.loss_fn(                   # 静态的foot，让它的速度为0
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        if self.cfg.LOSS.LAMBDA_FC > 0.0:
            foot_y_ankle = model_xp[:, :, [7, 8], 1]
            foot_y_toe = model_xp[:, :, [10, 11], 1]
            ground_height = 0
            velocity_foot_normal = model_foot_v_fc[:, :, :, 1:2].clone()
            fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (0.08+ground_height)), dim=3).repeat(1, 1, 1, 3)     # ground height is 0
            fc_mask_teo = torch.unsqueeze((foot_y_toe <= (0.05+ground_height)), dim=3).repeat(1, 1, 1, 3)
            fc_mask_y = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2)
            model_foot_v_fc[~fc_mask_y] = 0
            velocity_foot_normal_v = torch.zeros_like(velocity_foot_normal)
            velocity_foot_normal_v[:, :-1] = (velocity_foot_normal[:, 1:, :, :] - velocity_foot_normal[:, :-1, :, :]) 
            normal_mask = (velocity_foot_normal_v<0)
            model_foot_v_fc[:,:,:,1:2][~normal_mask] = 0        # double check here
            fc_loss = self.loss_fn(                   # 静态的foot，让它的速度为0
                model_foot_v_fc, torch.zeros_like(model_foot_v_fc), reduction="none"
            )
            fc_loss = reduce(fc_loss, "b ... -> b (...)", "mean")
            fc_loss = fc_loss * extract(self.p2_loss_weight, t, fc_loss.shape)
            fc_loss = fc_loss.mean()
        else:
            fc_loss = 0.0


        model_x_v = model_x[:, 1:, :] - model_x[:, :-1, :]
        model_x_a = model_x_v[:, 1:, :] - model_x_v[:, :-1, :]
        target_x_v = target_x[:, 1:, :] - target_x[:, :-1, :]
        target_x_a = target_x_v[:, 1:, :] - target_x_v[:, :-1, :]

        transl_loss_v = self.loss_fn(model_x_v , target_x_v , reduction="none")
        transl_loss_v = reduce(transl_loss_v, "b ... -> b (...)", "mean")
        transl_loss_v = transl_loss_v * extract(self.p2_loss_weight, t, transl_loss_v.shape)
        transl_loss_a = self.loss_fn(model_x_a , target_x_a, reduction="none")
        transl_loss_a = reduce(transl_loss_v, "b ... -> b (...)", "mean")
        transl_loss_a = transl_loss_a * extract(self.p2_loss_weight, t, transl_loss_a.shape)
        trans_loss = 0.3*transl_loss_v.mean() + 0.6 * transl_loss_a.mean()

        if self.cfg.LOSS.LAMBDA_FK_V>0:
            model_xp_v = model_xp[:, 1:, :] - model_xp[:, :-1, :]
            target_xp_v = target_xp[:, 1:, :] - target_xp[:, :-1, :]
            fk_loss_v = self.loss_fn(model_xp_v , target_xp_v , reduction="none")
            fk_loss_v = reduce(fk_loss_v, "b ... -> b (...)", "mean")
            fk_loss_v = fk_loss_v * extract(self.p2_loss_weight, t, fk_loss_v.shape)
            fk_loss_v = fk_loss_v.mean()
        else:
            fk_loss_v = 0.0
        if self.cfg.LOSS.LAMBDA_FK_A>0:
            model_xp_a = model_xp_v[:, 1:, :] - model_xp_v[:, :-1, :]
            target_xp_a = target_xp_v[:, 1:, :] - target_xp_v[:, :-1, :]
            fk_loss_a = self.loss_fn(model_xp_a , target_xp_a, reduction="none")
            fk_loss_a = reduce(fk_loss_a, "b ... -> b (...)", "mean")
            fk_loss_a = fk_loss_a * extract(self.p2_loss_weight, t, fk_loss_a.shape)
            fk_loss_a = fk_loss_a.mean()
        else:
            fk_loss_a = 0.0


        losses = (
        self.cfg.LOSS.LAMBDA_MSE  * loss.mean(),
        self.cfg.LOSS.LAMBDA_V    * v_loss.mean(),
        self.cfg.LOSS.LAMBDA_FK   * fk_loss.mean(),
        self.cfg.LOSS.LAMBDA_FK_V   * fk_loss_v,
        self.cfg.LOSS.LAMBDA_FK_A   * fk_loss_a,
        self.cfg.LOSS.LAMBDA_FOOT * foot_loss.mean(),
        self.cfg.LOSS.LAMBDA_FC * fc_loss,
        self.cfg.LOSS.LAMBDA_TRANS * trans_loss,
        )
        loss_dict = {}
        loss_dict.update({
                    "loss": sum(losses),
                    "mseloss": losses[0],
                    "Vloss": losses[1],
                    "fk_loss": losses[2],
                    "fk_loss_v": losses[3],
                    "fk_loss_a": losses[4],
                    "foot_loss": losses[5],
                    "fc_loss": losses[6],
                    "trans_loss": losses[7],
                })
        # else:
        #     losses = (
        #     self.cfg.LOSS.LAMBDA_MSE  * loss.mean(),
        #     self.cfg.LOSS.LAMBDA_V    * v_loss.mean(),
        #     self.cfg.LOSS.LAMBDA_FK   * fk_loss.mean(),
        #     self.cfg.LOSS.LAMBDA_FOOT * foot_loss.mean(),
        #     )
        #     loss_dict = {}
        #     loss_dict.update({
        #                 "loss": sum(losses),
        #                 "mseloss": losses[0],
        #                 "Vloss": losses[1],
        #                 "fk_loss": losses[2],
        #                 "foot_loss": losses[3],
        #             })
        
        return loss_dict

    def p_losses(self, x_start, cond, genre_id, t, isgen=False):
        noise = torch.randn_like(x_start)           
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      # denoise x0

        if self.cfg.Replace:           
            x_noisy[:, :4, :]  =  x_start[:, :4, :]
            x_noisy[:, -4:, :] =  x_start[:, -4:, :]
        x_noisy = replace_or_addkey(self.cfg, x_noisy, x_start)
        x_recon = self.model(x_noisy, cond, genre_id, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if isgen:
            return model_out
        
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # if self.transition_dim == 139 or self.transition_dim == 135 or self.transition_dim == 319  or self.transition_dim == 315:
        #     total_loss, losses = self.smpl_loss(model_out, target, t)
        # elif self.transition_dim == 263 or self.transition_dim == 266:
        #     total_loss, losses = self.loss_263(model_out, target, t)
      
        if self.cfg.LOSS.TYPE == 'smpl_loss':
            loss_dict = self.smpl_loss(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'smpl_loss_relative':
            loss_dict = self.smpl_loss_relative(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_263':
            loss_dict = self.loss_263(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_266':
            loss_dict = self.loss_266(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_266_origin':
            loss_dict = self.loss_266_origin(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_origin_addfre':
            loss_dict = self.loss_origin_addfre(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_with_wave':
            loss_dict = self.loss_with_wave(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_vec_position':
            loss_dict = self.loss_vec_position(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_vec_vel':    
            loss_dict = self.loss_vec_vel(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_338':
            loss_dict = self.loss_338(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_266_double_react':
            loss_dict = self.loss_266_double_react(model_out, cond, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_double_react':
            loss_dict = self.loss_double_react(model_out, cond, target, t)
        elif self.cfg.LOSS.TYPE == 'loss_266_double':
            loss_dict = self.loss_266_double(model_out, target, t)
    

        if self.cfg.Discriminator:
            return loss_dict, model_out
        else:
            return loss_dict
            
            
    

    def loss(self, x, cond, genre_id, t_override=None, isgen=False):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, genre_id, t, isgen)

    def forward(self, x, cond, genre_id=None, t_override=None, isgen=False):
        return self.loss(x, cond, genre_id, t_override, isgen)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x
    
    def smplxmodel_fk(self, local_q, root_pos):      # input
        b, s, nums, c = local_q.shape
        local_q = local_q.view(b*s, -1)
        full_pose = self.smplx_model(
                    betas = torch.zeros([b*s, 10], device=local_q.device, dtype=torch.float32),
                    transl = root_pos.view(b*s, -1),        # global translation
                    global_orient = local_q[:, :3],
                    body_pose = local_q[:, 3:66],           # 21
                    jaw_pose = torch.zeros([b*s, 3], device=local_q.device, dtype=torch.float32),         # 1
                    leye_pose = torch.zeros([b*s,  3], device=local_q.device, dtype=torch.float32),        # 1
                    reye_pose= torch.zeros([b*s,  3], device=local_q.device, dtype=torch.float32),          # 1
                    left_hand_pose = local_q[:, 66:111],   # 15
                    right_hand_pose = local_q[:, 111:], # 15
                    expression = torch.zeros([b*s, 10], device=local_q.device, dtype=torch.float32),
                    return_verts = False
            )
        full_pose = full_pose.joints.view(b, s, -1, 3)   # b, s, 55, 3
        
        return full_pose    #full_pose_52
        

    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        genre=None,
        Returnfull=False,
        # do_normalize=False,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "inpaint_soft":
                func_class = self.inpaint_soft_loop
            elif mode == "inpaint_key":
                func_class = self.inpaint_key
            elif mode == "inpaint_soft_ddim":
                func_class = self.inpaint_soft_ddim      # inpaint_soft_ddim         # my_inpaint_soft_ddim
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    genre,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape


        if self.cfg.DATA_SETTING.nfeats != 263 and self.cfg.DATA_SETTING.nfeats != 266 and self.cfg.DATA_SETTING.nfeats != 338 and self.cfg.DATA_SETTING.nfeats != 290:
            if self.cfg.DATA_SETTING.nfeats == 139 or self.cfg.DATA_SETTING.nfeats==135:
                reshape_size = 66
            elif self.cfg.DATA_SETTING.nfeats == 151:
                reshape_size = 72
            elif self.cfg.DATA_SETTING.nfeats == 319 or self.cfg.DATA_SETTING.nfeats==315:
                reshape_size = 156
            else:
                raise("error of nfeats")
            
            if self.cfg.Norm:
                samples = normalizer.unnormalize(samples)
            samples_ori = samples.clone()

            if samples.shape[2] == 319 or samples.shape[2] == 151 or samples.shape[2] == 139:                 # debug if samples.shape[2] == 151:    
                sample_contacts, samples = torch.split(
                    samples, (4, samples.shape[2] - 4), dim=2
                )
                sample_contacts = sample_contacts.to(cond.device)
            else:
                sample_contact = None
            # do the FK all at once
            b, s, c = samples.shape
            pos = samples[:, :, :3].to(cond.device)  # np.zeros((sample.shape[0], 3))
            q = samples[:, :, 3:].reshape(b, s, -1, 6)      # debug 24
            # go 6d to ax
            q = ax_from_6v(q).to(cond.device)
        else:
            samples = normalizer.unnormalize(samples)
        
        if mode == "long":
            if self.cfg.DATA_SETTING.nfeats == 263:
                pass
                print("need to be added")
                b,s,c = samples.shape
                assert c == 263
                assert s % 2 == 0
                half = s // 2
                pos = recover_from_ric(samples, 22).clone()
                # q = samples.clone()
                pos = pos.reshape(b,s,66)
                print("pos", pos.shape)

                if b > 1:
                    # if long mode, stitch position using linear interp
                    fade_out = torch.ones((1, s, 1)).to(pos.device)
                    fade_in = torch.ones((1, s, 1)).to(pos.device)
                    fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                        pos.device
                    )
                    fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                        pos.device
                    )
                    pos[:-1] *= fade_out
                    pos[1:] *= fade_in

                    full_pos = torch.zeros((s + half * (b - 1), 66)).to(pos.device)
                    idx = 0
                    for pos_slice in pos:
                        full_pos[idx : idx + s] += pos_slice
                        idx += half

                    full_pos = full_pos.unsqueeze(0)
                else:
                    full_pos = pos
                    # full_q = q

                Path(fk_out).mkdir(parents=True, exist_ok=True)
                for num, (full_pos_one, filename) in enumerate(zip(full_pos, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    outname = f"{epoch}_{num}_{filename}.npy"
                    np.save(f"{fk_out}/{outname}", full_pos_one)

            else:
                b, s, c1, c2 = q.shape
                assert s % 2 == 0
                half = s // 2
                if b > 1:
                    # if long mode, stitch position using linear interp

                    fade_out = torch.ones((1, s, 1)).to(pos.device)
                    fade_in = torch.ones((1, s, 1)).to(pos.device)
                    fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                        pos.device
                    )
                    fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                        pos.device
                    )

                    pos[:-1] *= fade_out
                    pos[1:] *= fade_in

                    full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                    idx = 0
                    for pos_slice in pos:
                        full_pos[idx : idx + s] += pos_slice
                        idx += half

                    # stitch joint angles with slerp
                    slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                    left, right = q[:-1, half:], q[1:, :half]
                    # convert to quat
                    left, right = (
                        axis_angle_to_quaternion(left),
                        axis_angle_to_quaternion(right),
                    )
                    merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                    # convert back
                    merged = quaternion_to_axis_angle(merged)

                    full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                    full_q[:half] += q[0, :half]
                    idx = half
                    for q_slice in merged:
                        full_q[idx : idx + half] += q_slice
                        idx += half
                    full_q[idx : idx + half] += q[-1, half:]

                    # unsqueeze for fk
                    full_pos = full_pos.unsqueeze(0)
                    full_q = full_q.unsqueeze(0)
                else:
                    full_pos = pos
                    full_q = q
                    
                # model_xp = self.smplx_fk.forward(model_q, model_x)
                # full_pose = self.smplx_fk.forward(full_q, full_pos).view(b, s, -1, 3).detach().cpu().numpy()      # # b, s, 52, 3
                # b_, s_, j_, c_ =  full_pose.shape
                # full_pose = (
                #     self.smpl.forward(full_q, full_pos).detach().cpu().numpy()
                # )  # b, s, 24, 3
                # squeeze the batch dimension away and render
                
                # skeleton_render(
                #     full_pose[0],
                #     epoch=f"{epoch}",
                #     out=render_out,
                #     name=name,
                #     sound=sound,
                #     sound_folder=sound_folder,
                #     render=render,
                #     stitch=False,
                #     smpl_mode="smplx"
                # )
                
                if fk_out is not None:
                    outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'  # f'{epoch}_{"_".join(name)}.pkl' #
                    Path(fk_out).mkdir(parents=True, exist_ok=True)
                    pickle.dump(
                        {
                            "smpl_poses": full_q.squeeze(0).reshape((-1, reshape_size)).cpu().numpy(),    # local rotations      # debug!!
                            "smpl_trans": full_pos.squeeze(0).cpu().numpy(),                    # root translation
                            # "full_pose": full_pose[0],                                          # 3d positions
                        },
                        open(os.path.join(fk_out, outname), "wb"),
                    )
                return

        # print("before smplx_fk.forward")
        # poses = self.smplx_fk.forward(q, pos).detach().cpu().numpy()
        # print("after smplx_fk.forward")
        # sample_contact = (
        #     sample_contact.detach().cpu().numpy()
        #     if sample_contact is not None
        #     else None
        # )
        # def inner(xx):
        #     num, pose = xx
        #     filename = name[num] if name is not None else None
        #     contact = sample_contact[num] if sample_contact is not None else None
        #     skeleton_render(
        #         pose,
        #         epoch=f"e{epoch}_b{num}",
        #         out=render_out,
        #         name=filename,
        #         sound=sound,
        #         contact=contact,
        #     )

        # p_map(inner, enumerate(poses))      # poses: 2, 150, 52, 3
        # print("4")

        if self.cfg.DATA_SETTING.nfeats != 263 and self.cfg.DATA_SETTING.nfeats != 266 and self.cfg.DATA_SETTING.nfeats != 338 and self.cfg.DATA_SETTING.nfeats != 290:
            if fk_out is not None and mode != "long":
                # print("saving data!")
                # print("fk_out is", fk_out)
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                # for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                for num, (sample_contact, qq, pos_, filename) in enumerate(zip(sample_contacts, q, pos, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    # outname = f"{epoch}_{num}_{filename}.pkl"
                    # print('saved pos_ shape', pos_.shape)
                    # print(" ")
                    # pickle.dump(
                    #     {
                    #         "smpl_poses": qq.reshape((-1, reshape_size)).cpu().numpy(),
                    #         "smpl_trans": pos_.cpu().numpy(),
                    #         # "full_pose": pose,
                    #     },
                    #     open(f"{fk_out}/{outname}", "wb"),
                    # )
                    
                    outname = f"{epoch}_{num}_{filename}.npy"
                    qq_rot6d = ax_to_6v(qq).reshape(pos_.shape[0], -1)
                    print("sample_contact", sample_contact.shape)
                    print("pos_", pos_.shape)
                    print("qq_rot6d", qq_rot6d.shape)
                    result_data = torch.cat([sample_contact, pos_, qq_rot6d], dim=1).detach().cpu().numpy()
                    Path(fk_out).mkdir(parents=True, exist_ok=True)
                    if not Returnfull:
                        np.save(os.path.join(fk_out, outname), result_data)
                if Returnfull:
                    return samples_ori
        else:
            if fk_out is not None and mode != "long":
                # print("saving data!")
                # print("fk_out is", fk_out)
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                # for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                for num, (sample, filename) in enumerate(zip(samples, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    outname = f"{epoch}_{num}_{filename}.npy"
                    if not Returnfull:
                        np.save(f"{fk_out}/{outname}", sample)
                if Returnfull:
                    return samples
                    
