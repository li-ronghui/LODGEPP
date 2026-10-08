import sys
from typing import Any, Callable, List, Optional, Union

import numpy as np
from regex import R
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from .rotary_embedding_torch import RotaryEmbedding
from .utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
sys.path.append('/data2/lrh/project/dance/long/')
from utils.smplfk import SMPLX_Skeleton, do_smplxfk


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x



class SeqModel(nn.Module):
    def __init__(self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 35,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:
        super().__init__()
     
        self.network = nn.ModuleDict()
        self.network['body_net'] = DanceDecoder(
            nfeats=4+3+22*6,
            seq_len=seq_len,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=cond_feature_dim,
            activation=activation
        )
        self.network['hand_net'] = DanceDecoder(
            nfeats=30*6,
            seq_len=seq_len,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=35+139,      # debug !
            activation=activation
        )


    def forward(self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0):
        x_body_start = x[:,:,:4+135]
        x_hand_start = x[:,:,4+135:]
        body_output = self.network['body_net'](x_body_start, cond_embed, times, cond_drop_prob)

        cond_embed = torch.cat([body_output, cond_embed], dim = -1)
        hand_output = self.network['hand_net'](x_hand_start, cond_embed, times, cond_drop_prob)

        output = torch.cat([body_output, hand_output], dim=-1)   
        return output
    
    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)

        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)
        return unc + (conditioned - unc) * guidance_weight


class Refine_DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 35+139,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        # self.sig = nn.Sigmoid()
        # self.contact_bce = nn.BCEWithLogitsLoss()

        output_feats = nfeats
        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        # self.r_null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        # self.r_null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.norm_cond = nn.LayerNorm(latent_dim)
        self.refine_norm_cond1 = nn.LayerNorm(latent_dim)
        self.refine_norm_cond2 = nn.LayerNorm(latent_dim)
        self.refine_norm_cond3 = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)      # debug cond_feature_dim
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer = nn.Linear(latent_dim, output_feats)
        
        self.refine_final_layer1 = nn.Linear(latent_dim, output_feats)
        self.refine_final_layer2 = nn.Linear(latent_dim, output_feats)
        self.refine_final_layer3 = nn.Linear(latent_dim, output_feats)
        self.refine_input_projection1 = nn.Linear(nfeats, latent_dim)
        self.refine_input_projection2 = nn.Linear(nfeats, latent_dim)
        self.refine_input_projection3 = nn.Linear(nfeats, latent_dim)
        self.refine_cond_projection1 = nn.Linear(48, latent_dim)   
        self.refine_cond_projection2 = nn.Linear(48, latent_dim)   
        self.refine_cond_projection3 = nn.Linear(48, latent_dim)   
        # self.refine_non_attn_cond_projection = nn.Sequential(
        #     nn.LayerNorm(latent_dim),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.SiLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )
        refine_decoderstack = nn.ModuleList([])
        for _ in range(1):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.refine_seqTransDecoder1 = DecoderLayerStack(refine_decoderstack)
        self.refine_seqTransDecoder2 = DecoderLayerStack(refine_decoderstack)
        self.refine_seqTransDecoder3 = DecoderLayerStack(refine_decoderstack)
        # self.refine_final_layer = nn.Linear(output_feats, output_feats)
    
        

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight
    

    def get_rcond(self, output):
        with torch.no_grad():
            smplxfk = SMPLX_Skeleton()
            joints3d = do_smplxfk(output, smplxfk)[:,:,:22,:]
            B,T,J,_ = joints3d.shape
            l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
            relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
            pred_foot = joints3d[:, :, relevant_joints, :]          # B,T,J,4
            foot_vel = torch.zeros_like(pred_foot)
            foot_vel[:, :-1] = (
                pred_foot[:, 1:, :, :] - pred_foot[:, :-1, :, :]
            )  # (N, S-1, 4, 3)
            foot_y_ankle = pred_foot[:, :, :2, 1]
            foot_y_toe = pred_foot[:, :, 2:, 1]
            fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (-1.2+0.012)), dim=3).repeat(1, 1, 1, 3)
            fc_mask_teo = torch.unsqueeze((foot_y_toe <= (-1.2+0.05)), dim=3).repeat(1, 1, 1, 3)
            contact_lable = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2).int().to(output).reshape(B, T, -1)

            contact_toe_thresh, contact_ankle_thresh, contact_vel_thresh = -1.2+0.08, -1.2+0.015, 0.3 / 30           # 30 is fps
            contact_score_toe = torch.sigmoid((contact_toe_thresh - pred_foot[:, :, :2, 1])/contact_toe_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, :2, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_toe = torch.unsqueeze(contact_score_toe, dim=3).repeat(1, 1, 1, 3)
            contact_score_ankle = torch.sigmoid((contact_ankle_thresh - pred_foot[:, :, 2:, 1])/contact_ankle_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, 2:, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_ankle = torch.unsqueeze(contact_score_ankle, dim=3).repeat(1, 1, 1, 3)
            contact_score = torch.cat([contact_score_ankle, contact_score_ankle], dim = -2).reshape(B, T, -1)
            r_cond = torch.cat([contact_lable, contact_score, pred_foot.reshape(B,T,-1), foot_vel.reshape(B,T,-1)], dim = -1)  
        return r_cond

    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        # print("cond_embed.shape", cond_embed.shape)
        cond_tokens = self.cond_projection(cond_embed)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t1 = t + cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens_cated = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        # print("x", x.shape)
        output = self.seqTransDecoder(x, cond_tokens_cated, t1)
        output = self.final_layer(output)
        r_output = self.refine_input_projection1(output)
        r_cond = self.get_rcond(output)
        r_cond = self.refine_cond_projection1(r_cond)
        rc = torch.cat((r_cond, t_tokens), dim=-2)
        r_cond_tokens = self.refine_norm_cond1(rc)
        refine_output = self.refine_seqTransDecoder1(r_output, r_cond_tokens, t1)
        refine_output = self.refine_final_layer1(refine_output) / 10
        out = output + refine_output

        r_output = self.refine_input_projection2(output)
        r_cond = self.get_rcond(output)
        r_cond = self.refine_cond_projection2(r_cond)
        rc = torch.cat((r_cond, t_tokens), dim=-2)
        r_cond_tokens = self.refine_norm_cond2(rc)
        refine_output = self.refine_seqTransDecoder2(r_output, r_cond_tokens, t1)
        refine_output = self.refine_final_layer2(refine_output) / 5
        out = output + refine_output

        r_output = self.refine_input_projection3(output)
        r_cond = self.get_rcond(output)
        r_cond = self.refine_cond_projection3(r_cond)
        rc = torch.cat((r_cond, cond_tokens, t_tokens), dim=-2)
        r_cond_tokens = self.refine_norm_cond3(rc)
        refine_output = self.refine_seqTransDecoder3(r_output, r_cond_tokens, t1)
        refine_output = self.refine_final_layer3(refine_output) / 10
        out = output + refine_output

        return out
    


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 35,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        # self.sig = nn.Sigmoid()
        # self.contact_bce = nn.BCEWithLogitsLoss()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)      # debug cond_feature_dim
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        # print("cond_embed.shape", cond_embed.shape)
        cond_tokens = self.cond_projection(cond_embed)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)

        output = self.final_layer(output)

        return output

