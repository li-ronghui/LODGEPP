# LODGEPP 权重与运行资产清单

本文档根据当前仓库的实际加载代码整理。模型文件尚未放入 Git，服务器部署目录中也尚未发现任何
`.ckpt`、`.pth` 或 `.pt` 权重。

服务器代码目录：

```text
/efs/nicorhli/project/LODGEPP
```

## 1. 第一优先级：标准 Lodge Global + Local 推理

标准 Lodge 是两阶段推理：Global Diffusion 先生成 characteristic dance primitives，Local
Diffusion 再生成完整动作。当前 `DanceDiffusion/infer_lodge.py` 会以 `strict=True` 分别加载
两个 Lightning checkpoint，因此两个文件缺一不可，而且必须与各自的训练配置匹配。

当前增强代码中写死的文件是：

| 阶段 | 当前代码要求的 checkpoint | 用途 |
| --- | --- | --- |
| Global | `FineDance_Coarse_Norm_139_WIN10/checkpoints/epoch=2999.ckpt` | 1024 帧全局动作/关键动作生成 |
| Local | `AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190_nofc/checkpoints/epoch=299.ckpt` | 256 帧局部细化和连接 |

如果保持当前脚本不改，建议最终放到：

```text
/efs/nicorhli/project/LODGEPP/DanceDiffusion/experiments/Global_Module/FineDance_Coarse_Norm_139_WIN10/checkpoints/epoch=2999.ckpt
/efs/nicorhli/project/LODGEPP/DanceDiffusion/experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190_nofc/checkpoints/epoch=299.ckpt
```

同时还需要 Global checkpoint 对应的完整训练配置：

```text
FineDance_Coarse_Norm_139_WIN10/config_2024-03-06-01-15-26_train.yaml
```

> 注意：上述 `AFineDance_..._nofc` 是当前私有增强代码选择的实验版本。目前没有在公开仓库中找到
> 这个精确 checkpoint 的公开下载地址，需要从原训练机、旧实验盘或作者备份中取回。

## 2. 官方可下载的 Lodge 权重包

官方公开的 Lodge 仓库提供了一个预训练包：

- 文件名：`exp.tar.gz`
- 已核验大小：`2,487,809,471` bytes，约 2.32 GiB
- [Google Drive 下载](https://drive.google.com/file/d/13Yp__EPAw0EjrSS898X5FtSQGmveBykA/view?usp=sharing)
- [百度网盘下载](https://pan.baidu.com/s/1twYAdqR5OjSPkIlT1AJafw?pwd=1mte)，提取码 `1mte`

官方推理脚本期待压缩包解压后至少提供：

```text
exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt
exp/Global_Module/FineDance_Global/global_train.yaml
exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt
exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml
```

这套权重是当前最值得先下载的完整基线。它对应公开版 Lodge 的 139D FineDance Global + Local
模型，但文件名和实验配置与当前 `Lodge_plus_smpl` 增强分支不完全相同。在没有检查
`state_dict` 键和张量形状前，不能声称它能直接替代上一节的私有增强 checkpoint。

建议下载后保留原始压缩包，并解压到：

```text
/efs/nicorhli/project/LODGEPP/pretrained/lodge_official/exp.tar.gz
/efs/nicorhli/project/LODGEPP/pretrained/lodge_official/exp/
```

## 3. 必需的非模型权重资产

### 3.1 FineDance 139D Normalizer

139D Global/Local 模型都会加载与训练时一致的 normalizer：

```text
Normalizer.pth
```

公开版文件只有约 4.5 KB，可从官方仓库取得：

- [Normalizer.pth](https://github.com/li-ronghui/LODGE/raw/refs/heads/main/data/Normalizer.pth)

建议保存为：

```text
/efs/nicorhli/project/LODGEPP/pretrained/lodge_official/data/Normalizer.pth
```

不能用当前 315D 原始 SMPL-X motion 临时计算出的统计量替代它；动作表示、维度与预处理必须与
checkpoint 完全一致。

### 3.2 SMPL-X 静态骨架

`smplx_neu_J_1.npy` 是前向运动学使用的静态关节模板，不是训练权重。它已经随当前仓库提供：

```text
DanceDiffusion/data/smplx_neu_J_1.npy
smplx_neu_J_1.npy
```

无需重复下载，但当前代码仍有旧机器绝对路径，需要部署时改为仓库内路径。

## 4. 第二优先级：VQ-VAE + GPT + Local Diffusion 实验路线

只有在准备运行 `infer/infer_gpt_diff_key.py` 时，才需要下面三个额外模型：

| 模型 | 代码中出现的文件 | checkpoint 字典键 |
| --- | --- | --- |
| FineDance 139D VQ-VAE | `best_recon.pth` | `net` |
| FineDance 139D GPT | `mask_best_acc.pth` | `trans` |
| 128 帧 Local Diffusion | `replace32128_win2_len128_relativeloss_Norm/checkpoints/epoch=2899.ckpt` | `state_dict` |

历史代码给出的 VQ-VAE/GPT 实验身份为：

```text
VQ-VAE: FineDance_1007_1024_win128/vqvae/f_139/best_recon.pth
GPT:    FineDance_1007_1024_win128/ckpt/mask_best_acc.pth
```

该路线还依赖以下匹配配置和中间资产：

```text
old-vqvae139.yaml
pld-gpt139.yaml
config_2024-07-06-11-00-05_train.yaml
FineDance 139D key-motion 库
FineDance 139D tokenized motion
Normalizer.pth
```

当前仓库只保留了加载路径，没有找到这三个模型的公开下载地址。它们不是运行官方
`infer_lodge.py` 基线的必需项，建议第二批再找。

## 5. 仅渲染网格时需要的 SMPL 系列模型

当前根目录 `render.py` 会初始化 SMPL、SMPL-H 和 SMPL-X。若使用它渲染人体网格，需要从
[SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/) 按其许可证下载：

```text
smpl/SMPL_MALE.pkl
smplh/SMPLH_MALE.pkl
smplx/SMPLX_NEUTRAL.npz
```

这些是受许可证约束的人体参数模型，不应提交到公开 GitHub。只生成/评估 139D 骨架动作时，
不必先下载全部三个文件。

## 6. 当前不需要下载的权重

- `hrnet_w32-36af842e.pth`：用于 AIST++ 视频姿态/分割预处理，不属于 FineDance Lodge 推理。
- Jukebox `vqvae.pth.tar`、`prior_level_2.pth.tar`：属于可选 Jukemirlib 音乐特征流程，主线
  35D FineDance 特征提取不需要。
- EDGE 目录中的 `Full-train-*.pt` / `train-*.pt`：旧基线和调试脚本引用，标准 Lodge
  Global + Local 推理不需要。
- 266D/263D/290D checkpoint：属于其他动作表示，不能与当前 139D 主线混用。

## 7. 推荐下载顺序

1. 下载官方 `exp.tar.gz`，不要改名。
2. 下载官方 `Normalizer.pth`。
3. 将两者上传到服务器 `pretrained/lodge_official/`，保留源文件校验值。
4. 解压后核对四个官方模型/配置文件是否齐全，再检查 `state_dict` 与当前代码兼容性。
5. 若目标是完整复现私有增强版，再寻找当前脚本精确指定的两个 checkpoint。
6. 只有要跑 GPT/key-motion 实验时，再寻找 VQ-VAE、GPT 和 `epoch=2899.ckpt`。

## 8. 下载后需要记录的信息

每个文件至少记录：

```text
原始下载链接
原始文件名
文件大小
SHA-256
解压目录
对应代码 commit
对应配置文件
```

只有文件存在还不代表可用；最终还要通过 checkpoint 结构检查、严格加载和单样本 GPU 推理。
