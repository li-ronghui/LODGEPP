import torch.nn as nn
from models.encdec import EncoderFrame, DecoderFrame
from models.quantize_cnn import QuantizeEMAReset_Frame, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 input_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim = output_emb_width
        self.num_code = nb_code
        self.quant = args.quantizer
        # else 应该是 AIST++ 的 dim, 似乎改这一个就行
        self.encoder = EncoderFrame(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = DecoderFrame(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset": # 默认是这个, 消融实验证明该优化方式最好
            self.quantizer = QuantizeEMAReset_Frame(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bT, Jx6) -> (Bt, J, 6)
        # x = x.permute(0,2,1).float()
        if len(x.shape) == 2:
            B, C = x.shape
            x = x.view(B, 24, 6)
        return x

    def postprocess(self, x):
        B, J , C = x.shape
        x = x.view(B, -1)
        return x

    def encode(self, x):
        # N, T, _ = x.shape
        # N, _ = x.shape
        x_in = self.preprocess(x) 
        x_encoder = self.encoder(x_in)          # B, 6, 12
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1]) 
        code_idx = self.quantizer.quantize(x_encoder)
        # code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)           # 64,268
        # Encode
        x_encoder = self.encoder(x_in)      # 64,128
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)      #  x_quantized 128,512,16

        ## decoder
        x_decoder = self.decoder(x_quantized)  # x_decoder   torch.Size([128, 268, 64])
        x_decoder = self.postprocess(x_decoder)
        return x_decoder, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)      # B, C
        N, C = x_d.shape
        x_d = x_d.view(-1, 6, C).permute(0,2,1).contiguous()        # B ,12, 6
        # decoder
        x_decoder = self.decoder(x_d)
        x_decoder = self.postprocess(x_decoder)
        return x_decoder



class HumanVQVAE_frame(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 input_dim=6,
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
        self.vqvae = VQVAE_251(args, nb_code, input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

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
        