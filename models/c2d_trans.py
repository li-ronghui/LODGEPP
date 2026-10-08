import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding

# mnet attebtion pakege
from einops import rearrange
from torch import einsum

class c2d_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, # 需要等于 clip_dim, 因为要 concat
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        # seq generetion
        self.trans_base = GCrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) # | 进行非线性变换和特征提取(含位置编码, 注意力)
        self.trans_head = GCrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) # 进行分类, 预测每个 idx 的概率
        # mask prediction
        self.kf_trans_base = KFCrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate) #  naive self-attention
        self.kf_trans_head = KFCrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)  
        
        self.block_size = block_size
        self.num_vq = num_vq
        self.mask_token = num_vq + 2 # 这个不用 embedding, 会被与处理掉

        # Audio Process
        self.mlp_a = nn.Linear(35*4, embed_dim) # 考虑vq-vae, motion 降采样了 4 倍 | audio 被预处理成了 35 dim 
        # genre mapping
        self.mapping = MappingNet(256, embed_dim)
        self.emb_dim = embed_dim

    def get_block_size(self):
        return self.block_size


    def forward(self, idxs, noise = None, genre = None, feature=None, masked_token_seq = None):
        '''
        None: Seq_predict
        masked_token_seq: Masked Predict
        '''
        if feature.shape[2] == 35: # audio feature | (bz, seq_len, 32) -> (bz, seq_len, 512) -> (bz, 512)
            bs = feature.shape[0]
            feature = feature.reshape(-1,35*4) # 4 因为 vq-vae 提取信息时降采样 4 倍 | 同一帧音频信息提取
            feature = self.mlp_a(feature) # audio feature 和 motion 是等长的
            feature = feature.reshape(bs,-1,self.emb_dim)

        if masked_token_seq is None: # seq predict
            g_feature = self.mapping(noise, genre)[:, None] # 对应 mnet gen 的 s
            feat = self.trans_base(idxs.to(dtype=torch.int64), feature, g_feature) # torch.Size([2, 121, 512]) | 特征提取, 对应 mnet gen 的 tr_block()
            logits = self.trans_head(feat, g_feature) # 预测当前序列的下一个 motion_index 所对应的不同 codebook_index 的概率
        else: # mask predict | 是关键帧控制, 可以看出区别在于分类 logits 不同, 实际区别在于使用 padding mask(预处理 pipline 中, 这里没有) or sequence mask
            feat = self.kf_trans_base(idxs.to(dtype=torch.int64), feature, masked_token_seq.to(dtype=torch.int64)) # torch.Size([2, 121, 512]) | 特征提取, 对应 mnet gen 的 tr_block()
            logits = self.kf_trans_head(feat, masked_token_seq.to(dtype=torch.int64)) # 预测 masked 下一个 motion_index 所对应的不同 codebook_index 的概率
        return logits

    def sample(self, noise = None, genre = None, feature=None, if_categorial=False, before_x = None, masked_token_seq = None): # 这个部分估计不需要改, 只需要确保输入可以多模态(text/music embedding)就可以
        '''序列生成, 使用 seq_mask attention'''
        
        gen_num = self.block_size # seq len 
        if before_x is not None: # before_x 即初始的几个idx, 用于衔接上一段 music
            gen_num = self.block_size - before_x.shape[0]

        for k in range(gen_num):
            if k == 0: 
                if before_x is not None: # # 延续之前的 before_x 继续 gen
                    x = before_x
                else: # 从零开始 gen, seq gen or masked gen | 区别: 是否有 origin seq 参考, 使用的注意力, 生成序列的loss(不体现在这里)
                    # x = []
                    x = torch.tensor([]).to(dtype=torch.int64)
            else:
                x = xs

            logits = self.forward(x, noise, genre, feature = feature, masked_token_seq = masked_token_seq) # feature + [] or feature + masked_token_seq | 更改这个部分的 x 的输入, 增加关键帧控制 | 输入 [之前的 index], [clip feature], 预测下一个时刻的 index 的 prob
            logits = logits[:, -1, :] # 实际上会输出所有的 idx 的 prob
            probs = F.softmax(logits, dim=-1)
            if if_categorial: # predict 的时候不走该分支
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else: # predict 的时候走这个分支
                _, idx = torch.topk(probs, k=1, dim=-1)  # 根据 prob, 选择概率最大的 index 的序号
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1) # dim 1 才是 index 序列
            
            # if k == self.block_size - 1: # 如果到终点了, 去掉最后一个 idx 并返回
            if k == gen_num - 1: # 如果到终点了, 去掉最后一个 idx 并返回
                if masked_token_seq is None: 
                    return xs[:, :-1]
                else:
                    mask_idx_list = torch.nonzero(masked_token_seq == self.mask_token).reshape(-1)
                    masked_token_seq[mask_idx_list] = xs[mask_idx_list] # remove mask
                    return masked_token_seq

        # 应该执行不到这里? for 内部直接 return 了应该
        if masked_token_seq is None: 
                    return xs
        else:
            mask_idx_list = torch.nonzero(masked_token_seq == self.mask_token).reshape(-1)
            masked_token_seq[mask_idx_list] = xs[mask_idx_list] # remove mask
            return masked_token_seq    
        # return xs

    def music_sample(self, noise = None, genre = None, audio=None, if_categorial = False, before_motion_num = 10): 
        # Inference 用, 训练不用这个, 用 sample 就行, 只是改了 pipline, 重复调用sample, 并且参考之前的一段 motion
        # 同理, 如果需要在 music 中 mask 某一段并进行动作控制, 也只需要更改此部分 | In Progress 
        '''music loop seq_mask attention'''
        assert before_motion_num < self.block_size, "cannot refer before motion more than seq len(block_size)"
        seq_len = self.block_size

        xs = torch.tensor([]).to(dtype=torch.int64)
        idx = None
        for music_idx in range(0, audio.shape[0], step = seq_len):
            if music_idx <= before_motion_num: # 不参考之前 motino 的情况: 没有那么多

                # idx_to = min(music_idx + seq_len, audio.shape[0]) # 不能越界
                if audio.shape[0] < music_idx + seq_len: # 少的音乐信息用0补充
                    zero_emb = torch.zeros([music_idx + seq_len - audio.shape[0], audio.shape[1]])
                    audio = torch.cat([audio,zero_emb],dim=0) # 使用 0 补齐剩下的维度

                music_piece = audio[music_idx:music_idx + seq_len]
                idx = self.sample(noise = noise, genre = genre, feature = audio, if_categorial=False, before_x = None, masked_token_seq = None) # 在这里被初始化成 torch.tensor
                xs = torch.cat((xs, idx), dim=1) # dim 1 才是 motion_token 序列
            else:
                music_idx -= before_motion_num # 需要参考一些值, 所以 music 初始位置改变。这部分动作序列不会重复生成, 会继续生成, 音乐信息不会被参考, 仅用于对齐位置, 因为动作生成时仅参考当前帧音乐
                
                # idx_to = min(music_idx + seq_len, audio.shape[0]) # 不能越界
                if audio.shape[0] < music_idx + seq_len: # 少的音乐信息用0补充
                    zero_emb = torch.zeros([music_idx + seq_len - audio.shape[0], audio.shape[1]])
                    audio = torch.cat([audio,zero_emb],dim=0) # 使用 0 补齐剩下的维度

                music_piece = audio[music_idx:music_idx + seq_len]
                idx = self.sample(noise = noise, genre = genre, audio = music_piece, if_categorial=False, before_x = idx[:, :-before_motion_num])
                xs = torch.cat((xs, idx[:before_motion_num:]), dim=1) # 要从befor motion 之后的部分开始 concat
        return xs
    


class CausalCrossConditionalSelfAttention(nn.Module): # 注意力模块: seq attention

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # 实现 mask 的关键 | causal mask to ensure that attention is only applied to the left in the input sequence 
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)) # 下三角 1 矩阵, 上面全是 0
        self.n_head = n_head

    def forward(self, x): 
        B, T, C = x.size() # (1, emb_seg_len (idx_len + 1), emb(512) )

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # torch.Size([2, 1, 512]) | (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) # embedding 变成了可相乘矩阵: 第 1,2 表示 1和2 embedding 的关系, 第 i 行表示 第 i 个元素和各个 j 元素的联系
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf')) # 遮住 i<j 的 j 元素的权重, 设置为 -inf 即可
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class KFBlock(nn.Module): # self-atten

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        # MNET
        self.self_attn = Attention(embed_dim, n_head, drop_out_rate) 
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential( # 上采样又下采样
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, masked_x = None): # x:(1, emb_seg_len (idx_len + 1), emb(512) )
        if masked_x is None:
            mask_idx_list = torch.nonzero(masked_x == self.mask_token).reshape(-1)
            x = x + self.self_attn(self.ln1(x), k=self.ln1(masked_x), v=self.ln1(masked_x), mask_idx_list = mask_idx_list) # torch.Size([2, 121, 512]) 
        else:
            x = x + self.self_attn(self.ln1(x)) # torch.Size([2, 121, 512]) 
        x = x + self.mlp(self.ln2(x)) # torch.Size([2, 121, 512])
        return x


class GBlock(nn.Module): # seq-mask att + genre self-atten

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.g_attn = Attention(embed_dim, n_head, drop_out_rate) # genre_attention
        self.s_attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate) # seq_attention
        self.mlp = nn.Sequential( # 上采样又下采样
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )
    
    def forward(self, x, g_feature): # x:(1, emb_seg_len (idx_len + 1), emb(512) )
        x = x + self.s_attn(self.ln1(x)) # sequence mask generation attention: 
        x = x + self.g_attn(self.ln2(x), g_feature, g_feature) # torch.Size([2, 121, 512]) | 添加的
        x = x + self.mlp(self.ln3(x)) # torch.Size([2, 121, 512])
        return x

class Block(nn.Module): # lay off | 暂时不用了, 可以用来做消融实验 | seq-mask att only

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.s_attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate) # seq_attention
        self.mlp = nn.Sequential( # 上采样又下采样
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

        # MNET
        # self.ln2 = nn.LayerNorm(embed_dim)
        # self.g_attn = Attention(embed_dim, n_head, drop_out_rate) # genre_attention
    
    def forward(self, x): # x:(1, emb_seg_len (idx_len + 1), emb(512) )
        x = x + self.s_attn(self.ln1(x)) # torch.Size([2, 121, 512])
        # x = x + self.g_attn(self.ln2(x), g_feature, g_feature) # 添加的
        x = x + self.mlp(self.ln3(x)) # torch.Size([2, 121, 512])
        return x

class KFCrossCondTransBase(nn.Module): # self-attetion: KFBlock

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, # 等于 clip_dim, 因为需要做 embed concat
                clip_dim=512, # 看选的哪个 clip 模型
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 3, embed_dim) # padding token , end token , mask token |总共有 num_vq + 2 个单词(动作序列), 每个输入 token 都被映射为一个长度为 embedding_dim 的向量表示
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[KFBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)]) # block 内部涉及注意力计算
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, feature, masked_token_seq):
        # 预处理 feature
        if len(idx) == 0: # 如果是第一个序列 idx, 只能依靠 clip_feature 进行预测
            if len(feature.shape) == 2: # 如果是 clip [bs, 512]
                token_embeddings = self.cond_emb(feature).unsqueeze(1) # 如果是 clip [bs, 512]
            else: # music [bs, seq_len, 512]
                token_embeddings = self.cond_emb(feature[:,0,:].squeeze(1)).unsqueeze(1)
        else: 
            b, t = idx.size() # idx ([2, 120]) | idx concat 的是第二位
            assert t <= self.block_size, "Cannot forward, model block size is exhausted." # 动作序列超出 block_size 不能处理
            # forward the Trans model
            token_embeddings = self.tok_emb(idx) #  ([2, 120]) -> [2, 120, 512]) | (batch_size, idx_seq) -> (batch_size, idx_seq, embedding_dim) | 每个输入 idx 都被映射为一个长度为 embedding_dim 的向量表示
            
            # 如果是 clip: 1 + 120
            # 如果是 music: 480 + 120 -> 1 + 120 , 仅仅选择当前时刻 music feature
            if len(feature.shape) == 2: # 如果是 clip [bs, 512], 保留原格式
                pass 
            else: # music [bs, seq_len, 512], 选择当前时刻的 music
                feature = feature[:,t - 1,:]

            cond_embeddings = self.cond_emb(feature).unsqueeze(1) # 如果是 clip [bs, 512]
            token_embeddings = torch.cat([cond_embeddings, token_embeddings], dim=1) # torch.Size([2, 480 + 120, 512]) | (1, emb_seg_len (idx_len + 1), emb(512) ) |(1,1,512) cat (batch_size, idx_seq, embedding_dim), 所以 embedding_dim 收到 clip_dim 约束
            
        x = self.pos_embed(token_embeddings) # torch.Size([2, 121, 512]) | 位置编码

        # idx and audio/clip feature 特征提取
        # x = self.blocks(x) # 放入特征提取器, 进行注意力计算
        for block in self.blocks:
            x = block(x, masked_x = masked_token_seq) # 放入特征提取器, 进行注意力计算

        return x # torch.Size([ns, 1~121, 512])

class KFCrossCondTransHead(nn.Module): # self-attention

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[KFBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)]) # 去掉 seq mask
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, masked_token_seq = None):
        if masked_token_seq is not None:
            for block in self.blocks:
                x = block(x, masked_x = masked_token_seq) # 放入特征提取器, 进行注意力计算
        else:
            x = self.blocks(x) # self-attention | torch.Size([2, 121, 512]) 
        x = self.ln_f(x) # 层归一化
        logits = self.head(x) # 预测 mask 部分的 motion | torch.Size([2, 121, 512]) | 分类, 一个线性层 | 没有 softmax 的就叫 logits
        return logits


class GCrossCondTransBase(nn.Module): # 进行特征提取(添加Genre 需要更改)

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, # 等于 clip_dim, 因为需要做 embed concat
                clip_dim=512, # 看选的哪个 clip 模型
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim) # 总共有 num_vq + 2 个单词(动作序列), 每个输入 token 都被映射为一个长度为 embedding_dim 的向量表示
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[GBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)]) # block 内部涉及注意力计算
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, feature, g_feature):
        # 预处理 feature
        if len(idx) == 0: # 如果是第一个序列 idx, 只能依靠 clip_feature 进行预测
            if len(feature.shape) == 2: # 如果是 clip [bs, 512]
                token_embeddings = self.cond_emb(feature).unsqueeze(1) # 如果是 clip [bs, 512]
            else: # music [bs, seq_len, 512]
                token_embeddings = self.cond_emb(feature[:,0,:].squeeze(1)).unsqueeze(1)
        else: 
            b, t = idx.size() # idx ([2, 120]) | idx concat 的是第二位
            assert t <= self.block_size, "Cannot forward, model block size is exhausted." # 动作序列超出 block_size 不能处理
            # forward the Trans model
            token_embeddings = self.tok_emb(idx) #  ([2, 120]) -> [2, 120, 512]) | (batch_size, idx_seq) -> (batch_size, idx_seq, embedding_dim) | 每个输入 idx 都被映射为一个长度为 embedding_dim 的向量表示
            
            # 如果是 clip: 1 + 120
            # 如果是 music: 480 + 120 -> 1 + 120 , 仅仅选择当前时刻 music feature
            if len(feature.shape) == 2: # 如果是 clip [bs, 512], 保留原格式
                pass 
            else: # music [bs, seq_len, 512], 选择当前时刻的 music
                feature = feature[:,t - 1,:]

            cond_embeddings = self.cond_emb(feature).unsqueeze(1) # 如果是 clip [bs, 512]
            token_embeddings = torch.cat([cond_embeddings, token_embeddings], dim=1) # torch.Size([2, 480 + 120, 512]) | (1, emb_seg_len (idx_len + 1), emb(512) ) |(1,1,512) cat (batch_size, idx_seq, embedding_dim), 所以 embedding_dim 收到 clip_dim 约束
            
        x = self.pos_embed(token_embeddings) # torch.Size([2, 121, 512]) | 位置编码

        # idx and audio/clip feature 特征提取
        # x = self.blocks(x) # 放入特征提取器, 进行注意力计算
        for block in self.blocks:
            x = block(x, g_feature) # 放入特征提取器, 进行注意力计算

        return x # torch.Size([ns, 1~121, 512])


class GCrossCondTransHead(nn.Module): 

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[GBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, g_feature):
        # x = self.blocks(x) # torch.Size([2, 121, 512]) | 注意力特征提取
        for block in self.blocks:
            x = block(x, g_feature) # 放入特征提取器, 进行注意力计算
        x = self.ln_f(x) # 层归一化
        logits = self.head(x) # # torch.Size([2, 121, 512]) | 分类, 一个线性层 | 没有 softmax 的就叫 logits
        return logits


# from MNET
class MappingNet(nn.Module):
    def __init__(self, in_dim, dim): # in_dim 指的是 噪声的 dim
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        self.unshared = nn.ModuleList()
        # for _ in range(10): # AIST++ 10 个 genre
        for _ in range(28): # 共 28 个 genre, 0~27: CUDA error: device-side assert triggered
            self.unshared.append(nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, dim)
            ))

    def forward(self, x, genre):
        s = self.shared(x) # noise
        sList = []
        for unshare in self.unshared: # 一堆 MLP
            sList.append(unshare(s))
        s = torch.stack(sList, dim=1)
        idx = torch.LongTensor(range(len(genre))).to(genre.device)
        return s[idx, genre]

# adopted from MNET self-attention | 加了 mask 掩码
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        self.dim_head = int(dim_head) # mask 用
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.pe = nn.Parameter(torch.randn(480, 480))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k=None, v=None, mask_idx_list = None):
        b, n, _, h = *q.shape, self.heads
        # k = q if k is None else k
        # v = q if v is None else v
        
        mask = None
        if mask_idx_list is not None: # init mask
            assert k is not None and v is not None, 'please input the layernorm(masked_x) in to the k and v to preform mask attention'
            mask = torch.ones((q.shape[1], self.dim_head)) # seq_len x dim_head(dim // heads)
            mask[:, mask_idx_list] = 0
            mask[mask_idx_list, :] = 0
            mask = mask.view(1, 1, q.shape[1], self.dim_head)
        else: # naive self-attention | g attention for GBlock
            k = q if k is None else k
            v = q if v is None else v

        q = self.to_q(q) # # torch.Size([2, 121, 512]) 
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        _, _, dots_w, dots_h = dots.shape
        dots += self.pe[:dots_w, :dots_h]

        # 在这里进行 mask, softmax 之前
        if mask is not None:
            dots = dots.masked_fill(mask==0,float('-inf'))

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

