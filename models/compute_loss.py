import torch
import torch.nn as nn

from smplx import SMPL
from einops import rearrange
# import rotation_conversionseometry # 这个文件不要也记得删了


class ComputeLoss(nn.Module):
    def __init__(self, loss_dict, rot_6d = False, smpl_path=None): 
        super(ComputeLoss, self).__init__()
        self.rot_6d = rot_6d # 暂定不用:verLoss | 是否是 rot_6d 格式, VerticesLoss 用

        self.gen_loss_dict = loss_dict # 用到的 loss name: 数字
        self.dis_loss_dict = {} # 根据 gen_loss_dict 进行调整

        # 都用了, 还要额外用一个 l_sfc_loss, 和
        for loss_name in loss_dict.keys():
            if loss_name == 'l_rec_loss' or 'l_smt_loss': # 重建损失
                self.l2_loss = nn.MSELoss()
            # if loss_name == 'l_vet_loss': # 使用了 VQ-VAE 不需要考虑动作关节是否到位了
            #     self.vet_loss = VerticesLoss(smpl_path, rot_6d)
            if loss_name == 'l_div_loss': # 多样性
                self.l1_loss = nn.L1Loss()
            if loss_name == 'l_adv_loss' or 'l_sfc_loss': # 经典 GAN 损失 + 风格聚焦项
                self.adv_loss = AdversarialLoss('hinge')
                self.dis_loss_dict.update({loss_name: loss_dict[loss_name]})

    def __call__(self, **kwargs):
        state = kwargs['state']
        if state in ["train_gen", "valid"]:
            return self.gen_loss(**kwargs)
        elif state in ["train_dis"]:
            return self.dis_loss(**kwargs)

    def gen_loss(self, **kwargs):
        state = kwargs['state'] # log 用
        music, gt_tokens, genre, a_indices = kwargs['music'], kwargs['m_tokens'], kwargs['genre'],kwargs['a_indices']

        b = music.shape[0]
        device = music.device

        # predict
        noise = torch.randn(b, 256).to(device)
        # gen_tokens = kwargs['gen'](a_indices, noise = noise, audio = music, genre = genre)
        # gen_tokens = gen_tokens.contiguous()
        gen_tokens = kwargs['gen'].sample(noise = noise, feature = music, genre = genre.cpu(), if_categorial = False)

        # # 输出是 idx 不需要这个:l_vet_loss
        # if self.rot_6d:
        #     dance_f = geometry.matTOrot6d(dance_f) # _f 是 target

        loss, log_dict = 0.0, {}
        for key, value in self.gen_loss_dict.items():
            if key == 'l_adv_loss': # 经典 GAN loss: discriminator 判断真假
                f_logit = kwargs['dis'](noise, music, gen_tokens, genre.cpu()) # 要改
                l_adv_loss = self.adv_loss(f_logit, True, False)
                loss += (l_adv_loss * value)

                log_dict.update({f'{state}/l_adv_loss': l_adv_loss})

            elif key == 'l_rec_loss': # idx 重建损失 | 等下检查一下对不对: iter1: 138505.6875 * 1 -> 0.5636 * 1
                print("gt_tokens.shape",gt_tokens.shape)
                print("gen_tokens.shape",gen_tokens.shape)
                a_ = gt_tokens.detach().cpu().numpy()
                b_ = gen_tokens.detach().cpu().numpy()
                l_rec_loss = self.l2_loss(gt_tokens, gen_tokens) # torch.Size([2, 121]) | torch.Size([2, 120]) | 判断生成的对不对
                # 得做个归一化gt_tokens
                l_rec_loss = l_rec_loss/ (gt_tokens.shape[0]*gt_tokens.shape[1] * kwargs['gen'].num_vq)
                loss += (l_rec_loss * value)

                log_dict.update({f'{state}/l_rec_loss': l_rec_loss})

            # elif key == 'l_vet_loss':  # 关节点 loss
            #     l_vet_loss = self.vet_loss(gt_tokens, gen_tokens)
            #     loss += (l_vet_loss * value)
                # log_dict.update({f'{state}/l_vet_loss': l_vet_loss})

            elif key == 'l_div_loss': # 多样性 loss: L1Loss : iter0: 60.8903 * 0.5
                noise1 = torch.randn(b, 256).to(device)
                noise2 = torch.randn(b, 256).to(device)

                style1 = kwargs['gen'].mapping(noise1, genre) # 此时 genre 已经变成了一个数字 id(tensor.long(1.))
                style2 = kwargs['gen'].mapping(noise2, genre) # torch.Size([2, 512])

                l_div_loss = self.l1_loss(noise1, noise2) / self.l1_loss(style1, style2)
                loss += (l_div_loss * value)

                log_dict.update({f'{state}/l_div_loss': l_div_loss})

            # 这里一个 bug, CUDA: not in same device
            elif key == 'l_sfc_loss': # style focus: 我们采用风格聚焦项: 保证生成的运动序列正确地保留特定于域的风格，而不管种子运动 xj 如何
                # 随机选 batch_size 个不同风格的 id
                gen_num = 28
                g_tensor = torch.arange(gen_num).long().to(device) # 27 是风格类别
                dance_id_ = [g_tensor[g_tensor != id][torch.randperm(gen_num-1)[0]] for id in genre]
                other_genre = torch.stack(dance_id_, dim=0).cpu() # 从别的类别随便挑一个, 共 batch_size 个

                other_genre = other_genre
                # dance_g_sfc = kwargs['gen'](a_indices, noise = noise, audio = music, genre = other_genre) # forward
                dance_g_sfc = kwargs['gen'].sample(noise = noise, feature = music, genre = other_genre, if_categorial = False)
                r_logit, f_logit = kwargs['dis'](noise, music, dance_g_sfc, other_genre, genre_ = genre) # in progress: 看来 dis 得多处理一种情况

                l_sfc_real = self.adv_loss(r_logit, True, False)
                l_sfc_fake = self.adv_loss(f_logit, False, False)

                l_sfc_loss = l_sfc_real + l_sfc_fake
                loss += (l_sfc_loss * value)

                log_dict.update({f'{state}/l_sfc_loss': l_sfc_loss})

        log_dict.update({f'{state}/loss': loss})
        return loss, log_dict

    def dis_loss(self, **kwargs):
        state = kwargs['state']
        music, gt_tokens, genre, a_indices = kwargs['music'], kwargs['m_tokens'], kwargs['genre'], kwargs['a_indices']

        b = music.shape[0]
        device = music.device

        noise = torch.randn(b, 256).to(device)
        # gen_tokens = kwargs['gen'](a_indices, noise = noise, audio = music, genre = genre) # forward
        gen_tokens = kwargs['gen'].sample(noise = noise, feature = music, genre = genre, if_categorial = False)

        # 输出是 idx 不需要这个
        # if kwargs['gen'].rot_6d:
        #     pose, trans = dance_f[:, :, :-3], dance_f[:, :, -3:]
        #     pose = rearrange(pose, 'b t (d c) -> (b t) d c', d=24, c=3)
        #     pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        #     pose = rearrange(pose, '(b t) d c -> b t (d c)', b=trans.shape[0], t=trans.shape[1])
        #     dance_f = torch.cat([pose, trans], dim=2)

        loss, log_dict = 0.0, {}
        for key, value in self.dis_loss_dict.items():
            if key == 'l_adv_loss':
                r_logit = kwargs['dis'](noise, music, gt_tokens, genre) # 真样本
                f_logit = kwargs['dis'](noise, music, gen_tokens.detach(), genre) # 假样本, detach 去掉梯度计算

                l_dis_real = self.adv_loss(r_logit, True, True)
                l_dis_fake = self.adv_loss(f_logit, False, True)

                l_adv_loss = ((l_dis_real + l_dis_fake) / 2)
                loss += (l_adv_loss * value)

                log_dict.update({f'{state}/l_real_loss': l_dis_real})
                log_dict.update({f'{state}/l_fake_loss': l_dis_fake})

        log_dict.update({f'{state}/loss': loss})
        return loss, log_dict


class AdversarialLoss(nn.Module): # 经典 GAN Loss
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type="nsgan"): 
        """ type = nsgan | lsgan | hinge """
        super(AdversarialLoss, self).__init__()
        self.type = type
        if type == "nsgan":
            self.criterion = nn.BCELoss()
        elif type == "lsgan":
            self.criterion = nn.MSELoss()
        elif type == "hinge": # 此分支
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None): # Gen:f_logit, True, False 
        if self.type == "hinge": # 执行
            if is_disc: # Gen 不执行
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else: # Gen 执行
                return (-outputs).mean()

        else:
            labels = torch.ones_like(outputs) if is_real else torch.zeros_like(outputs)
            loss = self.criterion(outputs, labels)
            return loss

# 暂定不用
class VerticesLoss(nn.Module): # 计算关节点偏差的 loss | 这段代码通过计算两个姿势和平移对应的3D人体模型的顶点之间的重建损失，得到一个平均损失值
    def __init__(self, model_path, rot_6d):
        super(VerticesLoss, self).__init__()
        # 参考用法 /opt/data/private/control2dance/debug_exp/dataset/aistplusplus_api/demos/extract_motion_feats.py
        self.smpl = SMPL(model_path=model_path, gender='MALE', batch_size=1).eval()
        self.rec_loss = nn.MSELoss()
        self.rot_6d = rot_6d

    def __call__(self, target, output):
        l_vet_loss = []

        if self.rot_6d:
            target = geometry.rot6dTOmat(target)
            output = geometry.rot6dTOmat(output)

        for tar, out in zip(target, output):
            pose, trans = tar[:, :-3].view(-1, 24, 3), tar[:, -3:]
            pose_, trans_ = out[:, :-3].view(-1, 24, 3), out[:, -3:]

            # smpl_poses = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))
            # smpl_poses_ = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose_))

            with torch.no_grad():
                vertices = self.smpl.forward(
                    global_orient=pose[:, 0:1],
                    body_pose=pose[:, 1:],
                    transl=trans
                ).vertices

                vertices_ = self.smpl.forward(
                    global_orient=pose_[:, 0:1],
                    body_pose=pose_[:, 1:],
                    transl=trans_
                ).vertices

                l_vet_loss.append(self.rec_loss(vertices, vertices_))

        l_vet_loss = torch.mean(torch.stack(l_vet_loss))
        return l_vet_loss
