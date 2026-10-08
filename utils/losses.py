import torch
import torch.nn as nn
from einops import reduce

from DanceDiffusion.dld.data.utils.smplfk import SMPLSkeleton, SMPLX_Skeleton, ax_from_6v
import torch.nn.functional as F



smplfk = SMPLSkeleton()

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, args): # 更改：把第二个变量去掉了
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        # self.nb_joints = nb_joints
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        self.motion_dim = args.DATA_SETTING.nfeats
        self.smplx_fk = smplfk
        
    def forward(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        loss = self.Loss(motion_pred, motion_gt)
        return loss
    
    def forward_cliprot(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred, motion_gt)
        # print("motion_pred.shape",motion_pred.shape)
        pred_v = motion_pred[:,1:,:] - motion_pred[:,:-1,:]
        gt_v = motion_gt[:,1:,:] - motion_gt[:,:-1,:]

        loss_v = self.Loss(pred_v, gt_v)
        return loss+loss_v
    
    def forward_clipfk(self, motion_pred, motion_gt) : 
        # 身体坐标
        B,T,C = motion_pred.shape
        if motion_gt.shape[1] == 135: 
            asix_pred = ax_from_6v(motion_pred[:,:, 3:].reshape(B,T,22,6))
            root_pred = motion_pred[:,:,:3]
            positions_pred = smplfk.forward(asix_pred, root_pred)
            asix_gt = ax_from_6v(motion_gt[:,:, 3:].reshape(B,T,22,6))
            root_gt = motion_gt[:,:,:3]
            positions_gt = smplfk.forward(asix_gt, root_gt)
        elif motion_gt.shape[1] == 139: 
            asix_pred = ax_from_6v(motion_pred[:,:, 4+3:].reshape(B,T,22,6))
            root_pred = motion_pred[:,:,4:4+3]
            positions_pred = smplfk.forward(asix_pred, root_pred)
            asix_gt = ax_from_6v(motion_gt[:,:, 4+3:].reshape(B,T,22,6))
            root_gt = motion_gt[:,:,4:4+3]
            positions_gt = smplfk.forward(asix_gt, root_gt)

        # 计算loss
        loss = self.Loss(positions_pred, positions_gt)
        
        pred_v = positions_pred[:,1:,:,:] - positions_pred[:,:-1,:,:]
        gt_v = positions_gt[:,1:,:,:] - positions_gt[:,:-1,:,:]
        loss_v = self.Loss(pred_v, gt_v)

        final_loss = loss + loss_v
        return final_loss
    
    def forward_fk(self, motion_pred, motion_gt) : 
        # 身体坐标
        B,C = motion_pred.shape
        motion_pred = motion_pred.view(B, -1, 6)
        motion_gt = motion_gt.view(B, -1, 6)

        asix_pred = ax_from_6v(motion_pred[:,1:, :]).unsqueeze(1)
        root_pred = motion_pred[:,0,:3].unsqueeze(1)
        positions_pred = smplfk.forward(asix_pred, root_pred)

        asix_gt = ax_from_6v(motion_gt[:,1:, :]).unsqueeze(1)
        root_gt = motion_gt[:,0,:3].unsqueeze(1)
        positions_gt = smplfk.forward(asix_gt, root_gt)


        loss = self.Loss(positions_pred, positions_gt)
        return loss
    
    def forward_vel(self, motion_pred, motion_gt,jointnum) : 
        loss = self.Loss(motion_pred[..., 7 : (jointnum - 1) * 3 + 7], motion_gt[..., 7 : (jointnum - 1) * 3 + 7])
        return loss
    

    def forward_edgeloss139(self, model_out, target): 
         # full reconstruction loss
        loss = self.Loss(model_out, target)            # mse loss
       

        model_contact, model_out = torch.split(model_out, (4, model_out.shape[2] - 4), dim=2)
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)       # b, length, jxc

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        v_loss = self.Loss(model_out_v, target_v)
  
        # FK loss
        b, s, c = model_out.shape      
        # model_x为root position, model_q为rotation
        model_x = model_out[:, :, :3]   # root position
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))
        b, s, nums, c_ = model_q.shape

        model_xp = self.smplx_fk.forward(model_q, model_x)
        target_xp = self.smplx_fk.forward(target_q, target_x)


        fk_loss = self.Loss(model_xp, target_xp)
        '''          v, a relative losses
        fk_loss_root = self.Loss(model_xp[:,:,0,:], target_xp[:,:,0,:])
        
        model_xp_v = model_xp[:, 1:, :, :] - model_xp[:, :-1, :, :]
        model_xp_a = model_xp_v[:, 1:, :, :] - model_xp_v[:, :-1, :, :]
        target_xp_v = target_xp[:, 1:, :, :] - target_xp[:, :-1, :, :]
        target_xp_a = target_xp_v[:, 1:, :, :] - target_xp_v[:, :-1, :, :]

        fk_loss_v = self.Loss(model_xp_v, target_xp_v)
        fk_loss_v_root = self.Loss(model_xp_v[:,:,0,:], target_xp_v[:,:,0,:])

        fk_loss_a = self.Loss(model_xp_a, target_xp_a)
        fk_loss_a_root = self.Loss(model_xp_a[:,:,0,:], target_xp_a[:,:,0,:])
        '''

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
        foot_loss = self.Loss(                   # 静态的foot，让它的速度为0
            model_foot_v, torch.zeros_like(model_foot_v)
        )
        # losses = (
        #     0.636 * loss.mean(),
        #     2.964 * v_loss.mean(),
        #     0.646 * fk_loss.mean(),
        #     10.942 * foot_loss.mean(),
        #     1 * fk_loss_v.mean(),
        #     1 * fk_loss_a.mean(),
        #     2 * (fk_loss_root.mean() + fk_loss_v_root.mean()*2 + fk_loss_a_root.mean()*2),
        # )
        losses = (
            0.636 * loss.mean(),            # 1 4 10
            2.964 * v_loss.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
            # 1 * fk_loss_v.mean(),
            # 1 * fk_loss_a.mean(),
            # 2 * (fk_loss_root.mean() + fk_loss_v_root.mean()*2 + fk_loss_a_root.mean()*2),
        )
        return sum(losses)
    



class Decoder_loss(nn.Module):
    def __init__(self, args): # 更改：把第二个变量去掉了
        super(Decoder_loss, self).__init__()
        self.loss_fn = F.mse_loss #if args.decoder.loss_type == "l2" else F.l1_loss
        self.args = args
        self.smplx_fk = SMPLX_Skeleton(batch=args.batch_size*512)
        self.smpl_fk = SMPLSkeleton()
    
    def forward(self, x_pred, x):
        return self.calculate_decoder_loss(x_pred, x)
        

    def calculate_decoder_loss(self, x_pred, x):
        # full reconstruction loss
        loss_full = self.loss_fn(x_pred, x, reduction="mean")            # mse loss
        
        if x_pred.shape[-1] == 319 or x_pred.shape[-1] == 139:
            foot_contact, x_pred = torch.split(x_pred, (4, x_pred.shape[2] - 4), dim=2               # 前4维是foot contact
            )
            _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)       # b, length, jxc
        
        x_v = x[:, 1:] - x[:, :-1]
        x_pred_v = x_pred[:, 1:] - x_pred[:, :-1]
        v_loss_full = self.loss_fn(x_pred_v, x_v, reduction="mean")
        
        if self.args.outfeats == 319 or self.args.outfeats == 315:
            b, s, c = x_pred.shape    
            x_pred_trans = x_pred[:, :, :3].view(-1, 3)   # root position
            x_pred_rot = ax_from_6v(x_pred[:, :, 3:].reshape(b, s, -1, 6)).view(b*s, -1)      # 以rot6d方式训练
            x_trans = x[:, :, :3].view(-1, 3)
            x_rot = ax_from_6v(x[:, :, 3:].reshape(b, s, -1, 6)).view(b*s, -1)
        
            x_pred_xyz = self.smplx_fk.forward(x_pred_rot, x_pred_trans).view(b, s, -1, 3)
            x_xyz = self.smplx_fk.forward(x_rot, x_trans).view(b, s, -1, 3)
            
        elif self.args.outfeats == 139 or self.args.outfeats == 135:
            b, s, c = x_pred.shape    
            x_pred_trans = x_pred[:, :, :3]   # root position
            x_pred_rot = ax_from_6v(x_pred[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
            x_trans = x[:, :, :3]
            x_rot = ax_from_6v(x[:, :, 3:].reshape(b, s, -1, 6))
            
            x_pred_xyz = self.smpl_fk.forward(x_pred_rot, x_pred_trans)    #.view(b, s, -1, 3)
            x_xyz = self.smpl_fk.forward(x_rot, x_trans)   # .view(b, s, -1, 3) 
        else:
            raise("error!")
        
        fk_loss_full = self.loss_fn(x_pred_xyz, x_xyz, reduction="mean")
        
        # foot skate loss
        foot_idx = [7, 8, 10, 11]
        # find static indices consistent with model's own predictions
        static_idx = foot_contact > 0.95  # N x S x 4
        model_feet = x_pred_xyz[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0               # 不计算动态帧
        foot_loss_full = self.loss_fn(                   # 静态的foot，让它的速度为0
            model_foot_v, torch.zeros_like(model_foot_v), reduction="mean")

        losses_full = (
            0.636 * loss_full.mean(),
            2.964 * v_loss_full.mean(),
            0.646 * fk_loss_full.mean(),
            10.942 * foot_loss_full.mean(),
        )
        return sum(losses_full), losses_full
    
    