import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from dld.models.architectures.utils import extract
from dld.data.utils.common.quaternion import *
from dld.data.utils.motion_process import recover_root_from7, recover_from_smplx_globalinit_v


class InterLoss(nn.Module):
    def __init__(self, cfg, loss_mode, nb_joints, normalizer, p2_loss_weight):
        super(InterLoss, self).__init__()
        self.joints_num = nb_joints
        if loss_mode == 'l1':
            self.Loss = F.l1_loss
        elif loss_mode == 'l2':
            self.Loss = F.mse_loss

        self.normalizer = normalizer
        self.p2_loss_weight = p2_loss_weight

        self.weights = {}
        self.weights["RO"] = cfg.LOSS.RO  #0.01
        self.weights["RTrans"] = cfg.LOSS.RTrans  #0.01
        self.weights["JA"] = cfg.LOSS.JA  #3
        self.weights["DM"] = cfg.LOSS.DM  #3

        self.weights["mse_loss"] =  cfg.LOSS.MSE    # 30
        self.weights["v_loss"] =   cfg.LOSS.VEL     # 10
 
        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss
    
    def diffu_mse(self, prediction, target, t, dm_mask=None,):
        if dm_mask is not None:
            loss = self.Loss(prediction, target, reduction="none") * dm_mask
        else:
            loss = self.Loss(prediction, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = (loss * extract(self.p2_loss_weight, t, loss.shape) ).mean()
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    # def forward(self, motion_pred, motion_gt, mask, timestep_mask): 
    def forward(self, model_out, cond, target, t): 
        leader = cond[..., 35:]
        self.p2_loss_weight = self.p2_loss_weight.to(target.device) 

        self.losses["mse_loss"] = self.diffu_mse(model_out, target, t)  * self.weights["mse_loss"]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        target_v = target[:, 1:] - target[:, :-1]
        self.losses["v_loss"] = self.diffu_mse(model_out_v, target_v, t)  * self.weights["v_loss"]

        if self.normalizer is not None:
            leader = self.normalizer.unnormalize(leader)
            model_out = self.normalizer.unnormalize(model_out)
            target = self.normalizer.unnormalize(target)
        leader_y_ang, leader_trans = recover_root_from7(leader[:,:7])
        follower_y_ang, follower_trans = recover_root_from7(model_out[:,:7])
        target_y_ang, target_trans = recover_root_from7(target[:,:7])

        leader_xyz_gt = recover_from_smplx_globalinit_v(leader, self.joints_num)
        follower_xyz_pred = recover_from_smplx_globalinit_v(model_out, self.joints_num)
        follower_xyz_gt  = recover_from_smplx_globalinit_v(target, self.joints_num)

        # self.forward_distance_map(follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh=1, t=t, lossname="DM")
        # self.forward_joint_affinity(follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh=0.05, t=t, lossname="JA")     # origin is 0.1
        self.forward_relatvie(follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, t=t, lossname="DM")
        self.forward_relatvie(follower_y_ang, leader_y_ang, target_y_ang, t=t, lossname="RO")
        self.forward_relatvie(follower_trans, leader_trans, target_trans, t=t, lossname="RTrans")

        self.accum_loss()

        # return self.losses

    def forward_relatvie(self, follower_y_ang_pred, leader_y_ang_gt, follower_y_ang_gt, t, lossname):
        pred_relative = follower_y_ang_pred - leader_y_ang_gt
        tgt_relative = follower_y_ang_gt - leader_y_ang_gt
        self.losses[lossname] = self.diffu_mse(pred_relative, tgt_relative, t) * self.weights[lossname]

    def forward_relatvie_rot(self, t):
        r_hip, l_hip, sdr_r, sdr_l = [2, 1, 17, 16]
        across = self.pred_g_joints[..., r_hip, :] - self.pred_g_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        across_gt = self.tgt_g_joints[..., r_hip, :] - self.tgt_g_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

        pred_relative_rot = qbetween(forward[..., 0, :], forward[..., 1, :])
        tgt_relative_rot = qbetween(forward_gt[..., 0, :], forward_gt[..., 1, :])

        self.losses["RO"] = self.mix_masked_mse(pred_relative_rot[..., [0, 2]], tgt_relative_rot[..., [0, 2]], t) * self.weights["RO"]

    def forward_distance_map(self, follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh, t, lossname):
        pred_distance_matrix = torch.cdist(follower_xyz_pred, leader_xyz_gt).reshape(
            follower_xyz_pred.shape[:-2] + (1, -1,))
        tgt_distance_matrix = torch.cdist(follower_xyz_gt, leader_xyz_gt).reshape(
            follower_xyz_pred.shape[:-2] + (1, -1,))
        print('pred_distance_matrix', pred_distance_matrix.shape)

        distance_matrix_mask = (pred_distance_matrix < thresh).float()
        self.losses[lossname] = self.diffu_mse(pred_distance_matrix, tgt_distance_matrix, t, distance_matrix_mask) * self.weights[lossname]

    def forward_joint_affinity(self, follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh, t, lossname):
        pred_distance_matrix = torch.cdist(follower_xyz_pred, leader_xyz_gt).reshape(
            follower_xyz_pred.shape[:-2] + (1, -1,))
        tgt_distance_matrix = torch.cdist(follower_xyz_gt, leader_xyz_gt).reshape(
            follower_xyz_pred.shape[:-2] + (1, -1,))

        distance_matrix_mask = (tgt_distance_matrix < thresh).float()

        self.losses[lossname] = self.diffu_mse(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix), t, distance_matrix_mask) * self.weights[lossname]

    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses["loss"] = loss
        return self.losses



# class InterLoss(nn.Module):
#     def __init__(self, cfg, loss_mode, nb_joints, normalizer, p2_loss_weight):
#         super(InterLoss, self).__init__()
#         self.joints_num = nb_joints
#         if loss_mode == 'l1':
#             self.Loss = F.l1_loss
#         elif loss_mode == 'l2':
#             self.Loss = F.mse_loss

#         self.normalizer = normalizer
#         self.p2_loss_weight = p2_loss_weight

#         self.weights = {}
#         self.weights["RO"] = cfg.LOSS.RO  #0.01
#         self.weights["RTrans"] = cfg.LOSS.RTrans  #0.01
#         self.weights["JA"] = cfg.LOSS.JA  #3
#         self.weights["DM"] = cfg.LOSS.DM  #3

#         self.weights["mse_loss"] =  cfg.LOSS.MSE    # 30
#         self.weights["v_loss"] =   cfg.LOSS.VEL     # 10
 
#         self.losses = {}

#     def seq_masked_mse(self, prediction, target, mask):
#         loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
#         loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
#         return loss
    
#     def diffu_mse(self, prediction, target, t, dm_mask=None,):
#         if dm_mask is not None:
#             loss = self.Loss(prediction, target, reduction="none") * dm_mask
#         else:
#             loss = self.Loss(prediction, target, reduction="none")
#         loss = reduce(loss, "b ... -> b (...)", "mean")
#         loss = (loss * extract(self.p2_loss_weight, t, loss.shape) ).mean()
#         return loss

#     def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
#         if dm_mask is not None:
#             loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
#         else:
#             loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
#         if contact_mask is not None:
#             loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
#         loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
#         loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

#         return loss

#     # def forward(self, motion_pred, motion_gt, mask, timestep_mask): 
#     def forward(self, model_out, cond, target, t): 
#         leader = cond[..., 35:]
#         self.p2_loss_weight = self.p2_loss_weight.to(target.device) 

#         self.losses["mse_loss"] = self.diffu_mse(model_out, target, t)  * self.weights["mse_loss"]
#         model_out_v = model_out[:, 1:] - model_out[:, :-1]
#         target_v = target[:, 1:] - target[:, :-1]
#         self.losses["v_loss"] = self.diffu_mse(model_out_v, target_v, t)  * self.weights["v_loss"]

#         if self.normalizer is not None:
#             leader = self.normalizer.unnormalize(leader)
#             model_out = self.normalizer.unnormalize(model_out)
#             target = self.normalizer.unnormalize(target)
#         leader_y_ang, leader_trans = recover_root_from7(leader[:,:7])
#         follower_y_ang, follower_trans = recover_root_from7(model_out[:,:7])
#         target_y_ang, target_trans = recover_root_from7(target[:,:7])

#         leader_xyz_gt = recover_from_smplx_globalinit_v(leader, self.joints_num)
#         follower_xyz_pred = recover_from_smplx_globalinit_v(model_out, self.joints_num)
#         follower_xyz_gt  = recover_from_smplx_globalinit_v(target, self.joints_num)

#         self.follower_xyz_pred = follower_xyz_pred
#         self.leader_xyz_gt = leader_xyz_gt
#         self.follower_xyz_gt = follower_xyz_gt

#         self.forward_distance_map(follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh=1, t=t, lossname="DM")
#         self.forward_joint_affinity(follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh=0.05, t=t, lossname="JA")     # origin is 0.1
#         self.forward_relatvie(follower_y_ang, leader_y_ang, target_y_ang, t=t, lossname="RO")
#         self.forward_relatvie(follower_trans, leader_trans, target_trans, t=t, lossname="RTrans")

#         self.accum_loss()

#         return self.losses

#     def forward_relatvie(self, follower_y_ang_pred, leader_y_ang_gt, follower_y_ang_gt, t, lossname):
#         pred_relative = follower_y_ang_pred - leader_y_ang_gt
#         tgt_relative = follower_y_ang_gt - leader_y_ang_gt
#         self.losses[lossname] = self.diffu_mse(pred_relative, tgt_relative, t) * self.weights[lossname]

#     def forward_relatvie_rot(self, t):
#         r_hip, l_hip, sdr_r, sdr_l = [2, 1, 17, 16]
#         across = self.pred_g_joints[..., r_hip, :] - self.pred_g_joints[..., l_hip, :]
#         across = across / across.norm(dim=-1, keepdim=True)
#         across_gt = self.tgt_g_joints[..., r_hip, :] - self.tgt_g_joints[..., l_hip, :]
#         across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

#         y_axis = torch.zeros_like(across)
#         y_axis[..., 1] = 1

#         forward = torch.cross(y_axis, across, axis=-1)
#         forward = forward / forward.norm(dim=-1, keepdim=True)
#         forward_gt = torch.cross(y_axis, across_gt, axis=-1)
#         forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

#         pred_relative_rot = qbetween(forward[..., 0, :], forward[..., 1, :])
#         tgt_relative_rot = qbetween(forward_gt[..., 0, :], forward_gt[..., 1, :])

        
#         self.losses["RO"] = self.mix_masked_mse(pred_relative_rot[..., [0, 2]], tgt_relative_rot[..., [0, 2]], t) * self.weights["RO"]


#     def forward_distance_map(self, follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh, t, lossname):
#         pred_distance_matrix = torch.cdist(follower_xyz_pred, leader_xyz_gt).reshape(
#             follower_xyz_pred.shape[:-2] + (1, -1,))
#         tgt_distance_matrix = torch.cdist(follower_xyz_gt, leader_xyz_gt).reshape(
#             follower_xyz_pred.shape[:-2] + (1, -1,))

#         distance_matrix_mask = (pred_distance_matrix < thresh).float()
#         self.losses[lossname] = self.diffu_mse(pred_distance_matrix, tgt_distance_matrix, t, distance_matrix_mask) * self.weights[lossname]

#     def forward_joint_affinity(self, follower_xyz_pred, leader_xyz_gt, follower_xyz_gt, thresh, t, lossname):
#         pred_distance_matrix = torch.cdist(follower_xyz_pred, leader_xyz_gt).reshape(
#             follower_xyz_pred.shape[:-2] + (1, -1,))
#         tgt_distance_matrix = torch.cdist(follower_xyz_gt, leader_xyz_gt).reshape(
#             follower_xyz_pred.shape[:-2] + (1, -1,))

#         distance_matrix_mask = (tgt_distance_matrix < thresh).float()

#         self.losses[lossname] = self.diffu_mse(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix), t, distance_matrix_mask) * self.weights[lossname]

#     def accum_loss(self):
#         loss = 0
#         for term in self.losses.keys():
#             loss += self.losses[term]
#         self.losses["loss"] = loss
#         return self.losses
