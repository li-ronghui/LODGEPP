# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x                # torch.Size([10, 32, 64, 64])

        out = self.conv1(x)         # torch.Size([10, 32, 64, 64])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)       # # torch.Size([10, 32, 64, 64])
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class MyBatchNorm1D(nn.Module):
    def __init__(self, channel):
        super(MyBatchNorm1D, self).__init__()
        self.channel = channel
        self.bn = nn.BatchNorm1d(channel)
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn(x)
        x = x.permute(0,2,1)
        
        return x
    
class MyUpDownTimeSample(nn.Module):
    def __init__(self, time_len):
        super(MyUpDownTimeSample, self).__init__()
        # 输入维度为B,T,C
        self.time_len = time_len
        
    def forward(self, x):
        B,T,C = x.shape
        if self.time_len > T:
            modetype = "bilinear"
        elif self.time_len < T:
            modetype = "nearest"
        else:
            raise("time can not equal")
        
        x = nn.functional.interpolate(x.permute(0,2,1).unsqueeze(-1), size=(self.time_len ,1), mode=modetype)
        x = x.squeeze().permute(0,2,1)
        
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DanceHighResolutionModule(nn.Module):
    def __init__(self, cfg, num_branches, blocks, num_blocks, 
                 num_channels, time_lens, fuse_method, multi_scale_output=True):
        super(DanceHighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels)

        self.cfg = cfg
        self.infer_dim = cfg.MODEL.infer_dim
        self.num_channels = num_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.time_lens = time_lens

        self.multi_scale_output = multi_scale_output
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.infer_dim, nhead=4, dim_feedforward=2048,batch_first=True)
    
        self.branches = self._make_branches(
            num_branches, encoder_layer, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        layers.append(
            nn.Sequential(
                nn.TransformerEncoderLayer(d_model=num_channels[0], nhead=4, dim_feedforward=256, batch_first=True),
                # nn.TransformerEncoder(block, num_layers=num_blocks[0]),         # 这里yaml中的num_blocks全是0
                MyBatchNorm1D(num_channels[0]),                                 # # num_channels一样
                nn.ReLU(inplace=True),
            )
        )
        
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        # for i in range(num_branches if self.multi_scale_output else 1):
            # if not self.multi_scale_output:
                # i = num_branches-1
        for i in range(num_branches):
            if (not self.multi_scale_output) and (i != num_branches-1):
                fuse_layers.append(None)
                continue
            fuse_layer = []
            for j in range(num_branches):
                if j > i:           # downsample
                    fuse_layer.append(
                        nn.Sequential(
                            MyUpDownTimeSample(self.time_lens[i]),
                            nn.TransformerEncoderLayer(d_model=num_channels[i], nhead=4, dim_feedforward=256,batch_first=True),
                            MyBatchNorm1D(num_channels[i]),
                            nn.ReLU(inplace=True),
                        ))
                elif j == i:
                    fuse_layer.append(None)
                else:           # j < i
                    fuse_layer.append(
                        nn.Sequential(
                            MyUpDownTimeSample(self.time_lens[i]),
                            nn.TransformerEncoderLayer(d_model=num_channels[i], nhead=4,dim_feedforward=256,batch_first=True),
                            MyBatchNorm1D(num_channels[i]),
                            nn.ReLU(inplace=True),
                        ))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels
    
    def get_num_time_lens(self):
        return self.time_lens

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])      
            # x[0]维持这个维度不变torch.Size([10, 32, 64, 64])      x[1] [10, 64, 32, 32]   x[2] [10, 128, 16, 16]
        x_fuse = []

        for i in range(len(self.fuse_layers)):
            if (not self.multi_scale_output) and (i != len(self.fuse_layers)-1):
                continue
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'TransformerEncoderLayer': nn.TransformerEncoderLayer(d_model=160, nhead=4, dim_feedforward=256,batch_first=True),
}


class DanceHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(DanceHighResolutionNet, self).__init__()

        # stem net
        self.cfg = cfg
        self.infer_dim = cfg.MODEL.infer_dim
        self.l1 = nn.Linear(139, self.infer_dim)
        
        self.encoder_1 = nn.TransformerEncoderLayer(d_model=self.infer_dim, nhead=4, dim_feedforward=256,batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.infer_dim, momentum=BN_MOMENTUM)
        
        self.bn2 = nn.BatchNorm1d(self.infer_dim, momentum=BN_MOMENTUM)
        
    
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        time_lens = self.stage2_cfg['TIME_LENGTH']
        self.transition1 = self._make_transition_layer([128], time_lens, num_channels)         # [32,64]
        self.stage2, pre_stage_times = self._make_stage(
            self.stage2_cfg)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        time_lens = self.stage3_cfg['TIME_LENGTH']
        self.transition2 = self._make_transition_layer(
            pre_stage_times, time_lens, num_channels)
        self.stage3, pre_stage_times = self._make_stage(
            self.stage3_cfg, multi_scale_output=False)
        

        self.final_layer = nn.Linear(
            in_features=self.infer_dim,
            out_features=cfg['MODEL']['out_C'],
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_time_pre_layer, num_time_cur_layer, num_channels):      # [256]  [32,64]
        
        num_branches_pre = len(num_time_pre_layer)                  # 1、
        num_branches_cur = len(num_time_cur_layer)                  # 2

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_time_cur_layer[i] != num_time_pre_layer[i]:      #如果时间步长不相等
                    # raise("time not equal")
                    transition_layers.append(
                        nn.Sequential(
                                MyUpDownTimeSample(time_len=out_time_lens, mode='nearest'),
                                nn.TransformerEncoderLayer(d_model=self.infer_dim, nhead=4,dim_feedforward=256,batch_first=True),
                                MyBatchNorm1D(num_channels[i]),
                                nn.ReLU(inplace=True),
                            ) 
                    )
                else:       
                    transition_layers.append(None)                  # 如果时间步长相等,则不做处理，直接赋值
            else:
                trans_time_uplayer = []
                for j in range(i+1-num_branches_pre):
                    in_time_lens = num_time_pre_layer[-1]                     # 256
                    out_time_lens = num_time_cur_layer[i] if j == i-num_branches_pre else in_time_lens
                    if out_time_lens > in_time_lens:
                        # x = nn.functional.interpolate(x.permute(0,2,1).unsqueeze(-1), size=(out_time_lens,1), mode='bilinear', align_corners=False)
                        # x = x.squeeze().permute(0,2,1)
                        trans_time_uplayer.append(
                            nn.Sequential(
                                MyUpDownTimeSample(time_len=out_time_lens),
                                nn.TransformerEncoderLayer(d_model=self.infer_dim, nhead=4,dim_feedforward=256,batch_first=True),
                                MyBatchNorm1D(num_channels[0]),
                                nn.ReLU(inplace=True),
                            ) 
                        )
                    elif out_time_lens == in_time_lens:
                        transition_layers.append(None) 
                    else:
                        trans_time_uplayer.append(
                            nn.Sequential(
                                MyUpDownTimeSample(time_len=out_time_lens),
                                nn.TransformerEncoderLayer(d_model=self.infer_dim, nhead=4,dim_feedforward=256,batch_first=True),
                                MyBatchNorm1D(num_channels[0]),
                                nn.ReLU(inplace=True),
                            ) 
                        )
                  
                transition_layers.append(nn.Sequential(*trans_time_uplayer))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config,
                    multi_scale_output=True):               # 默认为True，stage2和stage3都设置True，Stage4设置为False
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        time_lens = layer_config['TIME_LENGTH']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False            # 这里设定是否输出多尺度特征，最后一个moduel选择True
            else:
                reset_multi_scale_output = True             

            modules.append(
                DanceHighResolutionModule(
                    self.cfg,
                    num_branches,
                    block,
                    num_blocks,
                    # num_inchannels,
                    num_channels,
                    time_lens,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            previous_time_lens = modules[-1].get_num_time_lens()

        return nn.Sequential(*modules), previous_time_lens

    def forward(self, x):           # B, 128, 139
        x = self.l1(x)              # B, 128, 256
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu(x) 
        
        x = self.encoder_1(x)       # B, 128, 139
        x = self.bn2(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu(x)            # # B, 128, 139
        
        x_list = []                                                 # experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
        for i in range(self.stage2_cfg['NUM_BRANCHES']):            # 2
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))               # 10, 32, 64, 64.    10, 64, 32, 32              
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)                                # N21:10, 32, 64, 64.    N22:10, 64, 32, 32    

        x_list = []                                                 # transition2为 None， None，Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])                            # 第一个进入None    N31:10, 32, 64, 64。  N32: 10, 64, 32, 32 .    N33: 10,128,16,16
        y_list = self.stage3(x_list)

        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # y_list = self.stage4(x_list)
                                # N41:10, 32, 64, 64。  N42: 10, 64, 32, 32 .    N43: 10,128,16,16.  N44: 10,256,8,8
        x = self.final_layer(y_list[0])     # y_list[0] N41:10, 32, 64, 64

        return x                            # x:10, 16, 64, 64

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_dance_net(cfg, is_train, **kwargs):
    model = DanceHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model




class DoubleLinear(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # self.double_conv = nn.Sequential(
        #     nn.Linear(in_channels, mid_channels),
        #     nn.BatchNorm1d(mid_channels),           # 第二维度做batchnorm
        #     nn.ReLU(inplace=True),
        #     nn.Linear(mid_channels, out_channels),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.l1  = nn.Linear(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.r1  = nn.ReLU(inplace=True)
        
        self.l2  = nn.Linear(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.r2  = nn.ReLU(inplace=True)

    def forward(self, x):
        B, T, C = x.shape
        x = self.l1(x)
        x = self.bn1(x.view(B*T, -1)).view(B, T, -1)
        x = self.r1(x)
        
        x = self.l2(x)
        x = self.bn2(x.view(B*T, -1)).view(B, T, -1)
        x = self.r2(x)
        
        
        return x

class MotionDecoderMix4(nn.Module):
    def __init__(self,
                 nfeats, scale,
                 dim: list = [1024, 768, 512, 315],
                 **kwargs):
        super().__init__()
        # upsampling
        assert scale == 4
        
        self.l1 = nn.Linear(nfeats, dim[0])           # 159 ---69
        self.doublelinear1 = DoubleLinear(dim[0], dim[1])
        self.doublelinear2 = DoubleLinear(dim[1], dim[2])
        self.doublelinear3 = DoubleLinear(dim[2], dim[3])
        self.l2 = nn.Linear(dim[3], dim[3])

        self.ml1 = nn.Linear(512, 128)
        # self.trans1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)

    def forward(self, x, music):
        # B, 150, 23, 3 = B, 150, 69
        # B, 900, 53, 3 = B, 900, 159
        music = music.permute(0, 2, 1)
        music = self.ml1(music)
        music = music.permute(0, 2, 1)

        x = torch.cat([x, music], dim=-1)

        x = self.l1(x)
        x = x.permute(0, 2, 1)
        B,C,T = x.shape
        x = nn.functional.interpolate(x.unsqueeze(2), scale_factor=(1,2), mode='bilinear', align_corners=False)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.doublelinear1(x)
        
        x = x.permute(0, 2, 1)
        x = nn.functional.interpolate(x.unsqueeze(2), scale_factor=(1,2), mode='bilinear', align_corners=False)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.doublelinear2(x)
        
        x = self.doublelinear3(x)
        x = self.l2(x)
        return x

if __name__ == 'main':
    print("pose_hrnet")
    data = torch.rand([10, 256,256,3], dtype=torch.float32)
    # model = DanceHighResolutionNet()
    
    
    # python tools/train_2.py  --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
    # python tools/train_ori.py  --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml