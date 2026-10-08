import torch.nn as nn
import torch
from .c2d_trans_frame import GCrossCondTransBase,GCrossCondTransHead, MappingNet

# 参考 c2d_trans 搭建的 Dis
class DanceDiscriminator(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, # 需要等于 clip_dim, 因为要 concat
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                rot_6d=True):
        super().__init__()
        self.mlp_a = nn.Linear(35*4, embed_dim) # audio 被预处理成了 32 dim

        # 原 motion 处理
        # if rot_6d: # 走这个分支
        #     self.mlp_m = nn.Linear(22 * 6 + 3, dim) # 统一成了 22 个关节点, 不是 24
        # else:
        #     self.mlp_m = nn.Linear(22 * 3 + 3, dim)
        
        # 处理 Motion_token
        self.trans_base = GCrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) # 进行非线性变换和特征提取(含位置编码, 注意力)
        self.trans_head = GCrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) # 进行分类, 预测每个 idx 的概率

        # genre mapping
        self.mapping1 = MappingNet(256, embed_dim) # noise | 参考第一个 input 的 dim
        self.mapping2 = MappingNet(num_vq + 1, 1) # 使用索引张量 idx 和 genre 张量对 s 进行索引，返回索引后的结果
        self.emb_dim = embed_dim

    def forward(self, noise, audio, m_token, genre, genre_=None): # genre_ 是用来测多样性的
        # From t2m-gpt
        bs = audio.shape[0]
        audio = audio.reshape(-1,35*4) # 4 因为 vq-vae 提取信息时降采样 | 同一帧音频信息提取
        feature = self.mlp_a(audio) 
        feature = feature.reshape(bs,-1,self.emb_dim)

        g_feature = self.mapping1(noise, genre)[:, None] # 对应 mnet gen 的 s

        feat = self.trans_base(m_token.to(dtype=torch.int64), feature, g_feature) # torch.Size([2, 121, 512]) | 特征提取, 对应 mnet gen 的 tr_block()
        logits = self.trans_head(feat, g_feature) # torch.Size([2, 121, 1025]) | 预测下一个 motion_index 所对应的不同 codebook_index 的概率

        # from MNET
        x = logits.mean(dim=1) # # torch.Size([2, 1025]) | 这一步合理吗?

        if genre_ is None:
            return self.mapping2(x, genre)
        else:
            return self.mapping2(x, genre), self.mapping2(x, genre_)
        

class DanceDiscriminator_frame(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, # 需要等于 clip_dim, 因为要 concat
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                rot_6d=True):
        super().__init__()
        self.mlp_a = nn.Linear(35, embed_dim) # audio 被预处理成了 32 dim

        # 原 motion 处理
        # if rot_6d: # 走这个分支
        #     self.mlp_m = nn.Linear(22 * 6 + 3, dim) # 统一成了 22 个关节点, 不是 24
        # else:
        #     self.mlp_m = nn.Linear(22 * 3 + 3, dim)
        
        # 处理 Motion_token
        self.trans_base = GCrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) # 进行非线性变换和特征提取(含位置编码, 注意力)
        self.trans_head = GCrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) # 进行分类, 预测每个 idx 的概率

        # genre mapping
        self.mapping1 = MappingNet(256, embed_dim) # noise | 参考第一个 input 的 dim
        self.mapping2 = MappingNet(num_vq + 1, 1) # 使用索引张量 idx 和 genre 张量对 s 进行索引，返回索引后的结果
        self.emb_dim = embed_dim

    def forward(self, noise, audio, m_token, genre, genre_=None): # genre_ 是用来测多样性的
        # From t2m-gpt
        bs = audio.shape[0]
        # audio = audio.reshape(-1,35) # 4 因为 vq-vae 提取信息时降采样 | 同一帧音频信息提取
        feature = self.mlp_a(audio) 
        feature = feature.reshape(bs,-1,self.emb_dim)

        g_feature = self.mapping1(noise, genre)[:, None] # 对应 mnet gen 的 s

        feat = self.trans_base(m_token.to(dtype=torch.int64), feature, g_feature) # torch.Size([2, 121, 512]) | 特征提取, 对应 mnet gen 的 tr_block()
        logits = self.trans_head(feat, g_feature) # torch.Size([2, 121, 1025]) | 预测下一个 motion_index 所对应的不同 codebook_index 的概率

        # from MNET
        x = logits.mean(dim=1) # # torch.Size([2, 1025]) | 这一步合理吗?

        if genre_ is None:
            return self.mapping2(x, genre)
        else:
            return self.mapping2(x, genre), self.mapping2(x, genre_)
        