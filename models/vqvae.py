import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        # else 应该是 AIST++ 的 dim, 似乎改这一个就行
        print('-----------------args.DATA_SETTING.nfeats-------------', args.DATA_SETTING.nfeats)
        self.encoder = Encoder(args.DATA_SETTING.nfeats, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(args.DATA_SETTING.nfeats, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset": # 默认是这个, 消融实验证明该优化方式最好
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x) # 注意, 处理的是 width, 所以要换到中间
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        # code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)           # 128,256,64
        # Encode
        x_encoder = self.encoder(x_in)      # 128.512.16        128,135,8
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)      #  x_quantized 128,512,16

        ## decoder
        x_decoder = self.decoder(x_quantized)  # x_decoder   torch.Size([128, 268, 64])
        x_out = self.postprocess(x_decoder)    # torch.Size([128, 64, 268])
        return x_out, loss, perplexity
    
    def my_encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x) # 注意, 处理的是 width, 所以要换到中间
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx
    
    def my_forward_decoder(self, x):
        N, C = x.shape
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(N, C, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    



class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        # else 应该是 AIST++
        # self.nb_joints = 21 if args.dataname == 'kit' else None
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        