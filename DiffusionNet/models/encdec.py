import torch
import torch.nn as nn
from models.resnet import Resnet1D, Resnet1D_Linear

class EncoderFrame(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.output_emb_width = output_emb_width
        
        self.l1 = nn.Linear(input_emb_width, 24)
        self.res1 = Resnet1D_Linear(24, n_depth=depth)
        Tlayer1 = nn.TransformerEncoderLayer(d_model=24, nhead=4, batch_first=True)
        self.t1 = nn.TransformerEncoder(Tlayer1, num_layers=1)
        self.r1 = nn.ReLU()

        for i in range(down_t):
            input_dim = width = 24
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

        self.l2 = nn.Linear(output_emb_width, output_emb_width)
        self.res2 = Resnet1D_Linear(output_emb_width, n_depth=depth)
        Tlayer2 = nn.TransformerEncoderLayer(d_model=output_emb_width, nhead=2, batch_first=True)
        self.t2 = nn.TransformerEncoder(Tlayer2, num_layers=2)
        self.r2 = nn.ReLU()

        self.l3 = nn.Linear(output_emb_width, output_emb_width)


    def forward(self, x):
        B, J, C = x.shape       # 23,6
       

        x = self.res1(self.l1(x))       # 23,24
        x = self.r1(self.t1(x))

        x = x.permute(0,2,1)
        x = self.model(x) 
        x = x.permute(0,2,1)

        x = self.res2(self.l2(x))
        x = self.r2(self.t2(x))

        x = self.l3(x)

        # x = self.l1(x)
        # x = self.b1(x)
        # x = self.r1(x)

        # x = self.l2(x)
        # x = self.b2(x)
        # x = self.r2(x)

        # x = self.l3(x)
        # x = self.r3(self.b3(x))
        # x = self.l3_(x)
        # x = self.r3_(self.b3_(x))

        # x = self.l4(x)
        # x = self.r4(self.b4(x))

        # x = self.l5(x)
        return x


class DecoderFrame(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        self.output_emb_width = output_emb_width

        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)


        self.l2 = nn.Linear(output_emb_width, input_emb_width)
        self.res2 = Resnet1D_Linear(input_emb_width, n_depth=depth)
        Tlayer2 = nn.TransformerEncoderLayer(d_model=input_emb_width, nhead=1, batch_first=True)
        self.t2 = nn.TransformerEncoder(Tlayer2, num_layers=3)
        self.r2 = nn.ReLU()

        self.l3 = nn.Linear(input_emb_width, input_emb_width)


        # self.l1 = nn.Linear(input_emb_width, 256)
        # self.res1 = Resnet1D_Linear(256, n_depth=depth)

        # self.l2 = nn.Linear(256, 512)
        # self.res2 = Resnet1D_Linear(512, n_depth=depth)

        # self.l3 = nn.Linear(512, output_emb_width)
        # self.res3 = Resnet1D_Linear(output_emb_width, n_depth=depth)
        # self.b3 = nn.BatchNorm1d(output_emb_width)
        # self.r3 = nn.ReLU()

        # self.l4 = nn.Linear(output_emb_width, output_emb_width)
        
        # self.l1 = nn.Linear(input_emb_width, 512)
        # self.b1 = nn.BatchNorm1d(512)
        # self.r1 = nn.ReLU()

        # self.l2 = nn.Linear(512, 512)
        # self.b2 = nn.BatchNorm1d(512)
        # self.r2 = nn.ReLU()

        # self.l3 = nn.Linear(512, 512)
        # self.b3 = nn.BatchNorm1d(512)
        # self.r3 = nn.ReLU()

        # self.l4 = nn.Linear(512, output_emb_width)
        # self.b4 = nn.BatchNorm1d(output_emb_width)
        # self.r4 = nn.ReLU()
        # self.l5 = nn.Linear(output_emb_width, output_emb_width)

    def forward(self, x):
        B, C, J = x.shape           # B, 12，6

        x = self.model(x)           # B, 12, 24

        x = x.permute(0,2,1).contiguous()
        x = self.res2(self.l2(x))
        x = self.r2(self.t2(x))

        x = self.l3(x)

        return x

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
