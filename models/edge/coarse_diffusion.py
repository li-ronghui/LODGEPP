import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from utils.smplfk import skeleton_render
from utils.smplfk  import do_smplxfk, SMPLX_Skeleton
# from dataset.preprocess import My_Normalizer as Normalizer

from .utils import extract, make_beta_schedule

def identity(t, *args, **kwargs):
    return t

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
        model,
        opt,
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
        do_normalize=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
   
        self.normalizer = None
        self.do_normalize = do_normalize
        self.opt = opt

        self.cond_drop_prob = cond_drop_prob


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
        self.contact_bce = nn.BCEWithLogitsLoss()

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
    
    def model_predictions(self, x, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight)
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

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
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
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
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
    def ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

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
    def long_ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        if batch == 1:
            return self.ddim_sample(shape, cond)

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
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

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
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            # value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            # x = value_ * mask + (1.0 - mask) * x
            x = value * mask + (1.0 - mask) * x

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

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)           
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      # 将x0加噪到xt


        # 强制添加首尾帧
        # if x_start.shape[-1] == 139 or x_start.shape[-1]==319:
        #     firstidx = 4
        # else:
        #     firstidx = 0

        # if self.opt.cond_feadim == 174 or self.opt.cond_feadim == 35:
        #     x_noisy[:, :4, :]  =  x_start[:, :4, :]
        #     x_noisy[:, -4:, :] =  x_start[:, -4:, :]
        # elif self.opt.cond_feadim == 170:
        #     x_noisy[:, 0, :firstidx]  =  x_start[:, 0, :firstidx]
        #     x_noisy[:, -1, :firstidx] =  x_start[:, -1, :firstidx]
        # else:
        #     raise("error")
        #     print("No training restrict")
        #     print("self.opt.cond_feadim", self.opt.cond_feadim)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        if self.opt.nfeats == 139 or self.opt.nfeats == 135 or self.opt.nfeats == 319  or self.opt.nfeats == 315:
            total_loss, losses = self.smpl_loss(model_out, target, t)
        elif self.opt.nfeats == 263:
            total_loss, losses = self.loss_263(model_out, target, t)

        return total_loss, losses
    
    def loss_263(self, model_out, target, t): 
        loss = self.loss_fn(model_out, target, reduction="none")            #  mse loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        v_loss = self.loss_fn(model_out[..., 4 : (22 - 1) * 3 + 4], target[..., 4 : (22 - 1) * 3 + 4], reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
 
        losses = (
                loss.mean(),
                0.5 * v_loss.mean(),
            )
        return sum(losses), losses
    
    def smpl_loss(self, model_out, target, t):
        B, T, C = model_out.shape

        model_out = model_out.reshape(-1, 8, C)
        target = target.reshape(-1, 8, C)
        t = t.repeat(int(target.shape[0]//B)  )

        if self.opt.nfeats == 139 or self.opt.nfeats==135:
            with torch.no_grad():
                smplx_fk = SMPLX_Skeleton()
                model_xp = do_smplxfk(model_out, smplx_fk)
                target_xp = do_smplxfk(target, smplx_fk)

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")            # mse loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        model_contact, model_out = torch.split(model_out, (4, model_out.shape[2] - 4), dim=2)
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)       # b, length, jxc


        root_rloss = self.loss_fn(model_out[:, :, 3:9], target[:, :, 3:9], reduction="none")   
        root_rloss = reduce(root_rloss, "b ... -> b (...)", "mean")
        root_rloss = root_rloss * extract(self.p2_loss_weight, t, root_rloss.shape)

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        if not self.opt.no_vloss:
            v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)



        # FK loss
        b, s, c = model_out.shape     
        foot_bce_loss =  self.contact_bce(model_contact.clone(), target_contact.clone())
        
        # model_x为root position, model_q为rotation
        model_x = model_out[:, :, :3]   # root position
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))
        b, s, nums, c_ = model_q.shape

        if self.opt.nfeats == 139 or self.opt.nfeats==135:
            pass
        elif self.opt.nfeats == 319 or self.opt.nfeats==315:
            model_q = model_q.view(b*s, -1)
            target_q = target_q.view(b*s, -1)
            model_x = model_x.view(-1, 3)
            target_x = target_x.view(-1, 3)
            model_xp = self.smplx_fk.forward(model_q, model_x)
            target_xp = self.smplx_fk.forward(target_q, target_x)
            model_xp = model_xp.view(b, s, -1, 3)
            target_xp = target_xp.view(b, s, -1, 3)
        else:
            raise("error of nfeats!")
        

        l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
        relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
        gt_joint_xyz = target_xp[:, :, relevant_joints, :]  
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, 1:, :, :] - gt_joint_xyz[:, :-1, :, :], axis=3)  
        fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=3).repeat(1, 1, 1, 3)
        pred_joint_xyz = model_xp[:, :, relevant_joints, :] 
        pred_vel = pred_joint_xyz[:, 1:, :, :] - pred_joint_xyz[:, :-1, :, :]
        pred_vel[~fc_mask] = 0
        fc_loss = self.loss_fn(pred_vel, torch.zeros(pred_vel.shape, device=pred_vel.device), reduction="none")
        fc_loss = reduce(fc_loss, "b ... -> b (...)", "mean")
        fc_loss = fc_loss * extract(self.p2_loss_weight, t, fc_loss.shape)
        # self.masked_l2(pred_vel,
        #                              torch.zeros(pred_vel.shape, device=pred_vel.device),
        #                              mask[:, :, :, 1:])


        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        # print("fk_loss.shape",fk_loss.shape)
        fk_loss_root = fk_loss[:,:,0,:]
        fk_loss_root = reduce(fk_loss_root, "b ... -> b (...)", "mean")
        fk_loss_root = fk_loss_root * extract(self.p2_loss_weight, t, fk_loss_root.shape)
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)
        
        model_xp_v = model_xp[:, 1:, :, :] - model_xp[:, :-1, :, :]
        model_xp_a = model_xp_v[:, 1:, :, :] - model_xp_v[:, :-1, :, :]
        target_xp_v = target_xp[:, 1:, :, :] - target_xp[:, :-1, :, :]
        target_xp_a = target_xp_v[:, 1:, :, :] - target_xp_v[:, :-1, :, :]

        fk_loss_v = self.loss_fn(model_xp_v, target_xp_v, reduction="none")
        fk_loss_v_root = fk_loss_v[:,:,0,:]
        fk_loss_v_root = reduce(fk_loss_v_root, "b ... -> b (...)", "mean")
        fk_loss_v_root = fk_loss_v_root * extract(self.p2_loss_weight, t, fk_loss_v_root.shape)
        fk_loss_v = reduce(fk_loss_v, "b ... -> b (...)", "mean")
        fk_loss_v = fk_loss_v * extract(self.p2_loss_weight, t, fk_loss_v.shape)

        fk_loss_a = self.loss_fn(model_xp_a, target_xp_a, reduction="none")
        fk_loss_a_root = fk_loss_a[:,:,0,:]
        fk_loss_a_root = reduce(fk_loss_a_root, "b ... -> b (...)", "mean")
        fk_loss_a_root = fk_loss_a_root * extract(self.p2_loss_weight, t, fk_loss_a_root.shape)
        fk_loss_a = reduce(fk_loss_a, "b ... -> b (...)", "mean")
        fk_loss_a = fk_loss_a * extract(self.p2_loss_weight, t, fk_loss_a.shape)
        
        if not self.opt.no_vloss:        
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

            # losses = (
            #     0.636 * loss.mean(),
            #     2.964 * v_loss.mean(),
            #     0.646 * fk_loss.mean(),
            #     10.942 * foot_loss.mean(),
            #     1 * fk_loss_a.mean(),
            #     2 * (fk_loss_root.mean() + fk_loss_v_root.mean()*2 + fk_loss_a_root.mean()*2),
            # )

            # 2000 以后
            # losses = (
            #     8 * loss.mean(),            # 1 4 10
            #     2.964 * v_loss.mean(),
            #     0.646 * fk_loss.mean(),
            #     10 * foot_loss.mean(),
            #     0.2 * foot_bce_loss.mean(),
            #     8 * fc_loss.mean(),
            #     1 * fk_loss_v.mean(),
            #     1 * fk_loss_a.mean(),
            #     1 * (fk_loss_root.mean() + fk_loss_v_root.mean() + fk_loss_a_root.mean()*2 +  root_rloss.mean()),
            # )
            losses = (
                10 * loss.mean(),            # 1 4 10
                6 * v_loss.mean(),
                6 * fk_loss.mean(),
                15 * foot_loss.mean(),
                0.2 * foot_bce_loss.mean(),
                8 * fc_loss.mean(),
                1 * fk_loss_v.mean(),
                1 * fk_loss_a.mean(),
                1 * (fk_loss_root.mean() + fk_loss_v_root.mean() + fk_loss_a_root.mean()*2 +  root_rloss.mean()),
            )

        else:
            losses = (
                loss.mean(),
                fk_loss.mean(),
            )
        return sum(losses), losses

    def loss(self, x, cond, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

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
        # full_q = full_q.view(b, s, nums, c_)
        # full_pose_52 = torch.concat((full_pose[:, :, :22, :], full_pose[:, :, 25:, :]), dim=2)
        
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
        # do_normalize=True,
        Returnfull = False,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples_ori = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples_ori = shape


        if self.opt.nfeats != 263:
            if self.opt.nfeats == 139 or self.opt.nfeats==135:
                reshape_size = 66
            elif self.opt.nfeats == 319 or self.opt.nfeats==315:
                reshape_size = 156
            else:
                raise("error of nfeats")
            
            if self.do_normalize:
                with torch.no_grad():
                    samples_ori = normalizer.unnormalize(samples_ori)

            if samples_ori.shape[2] == 319 or samples_ori.shape[2] == 151 or samples_ori.shape[2] == 139:                 # debug if samples.shape[2] == 151:    
                sample_contact, samples = torch.split(
                    samples_ori, (4, samples_ori.shape[2] - 4), dim=2
                )
            else:
                sample_contact = None
            # do the FK all at once
            b, s, c = samples.shape
            pos = samples[:, :, :3].to(cond.device)  # np.zeros((sample.shape[0], 3))
            q = samples[:, :, 3:].reshape(b, s, -1, 6)      # debug 24
            # go 6d to ax
            q = ax_from_6v(q).to(cond.device)
        else:
            print("nfeats if not 263")
        
        if mode == "long":
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

        if self.opt.nfeats != 263:
            if fk_out is not None and mode != "long":
                # print("saving data!")
                # print("fk_out is", fk_out)
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                # for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                if Returnfull:
                    for num, (sample, filename) in enumerate(zip(samples_ori, name)):
                        filename = os.path.basename(filename).split(".")[0]
                        outname = f"{epoch}_{num}_{filename}.npy"
                        np.save(f"{fk_out}/{outname}", sample)
                else:
                    for num, (qq, pos_, filename) in enumerate(zip(q, pos, name)):
                        filename = os.path.basename(filename).split(".")[0]
                        outname = f"{epoch}_{num}_{filename}.pkl"
                        pickle.dump(
                            {
                                "smpl_poses": qq.reshape((-1, reshape_size)).cpu().numpy(),
                                "smpl_trans": pos_.cpu().numpy(),
                                # "full_pose": pose,
                            },
                            open(f"{fk_out}/{outname}", "wb"),
                        )
        else:
            if fk_out is not None and mode != "long":
                # print("saving data!")
                # print("fk_out is", fk_out)
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                # for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                for num, (sample, filename) in enumerate(zip(samples, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    outname = f"{epoch}_{num}_{filename}.npy"
                    np.save(f"{fk_out}/{outname}", sample)
