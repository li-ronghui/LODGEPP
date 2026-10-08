# LODGEPP 推理权重与资产清单

本文只针对当前仓库最后持续更新的 LODGEPP 推理入口：

```text
infer/infer_gpt_diff_key.py
```

该入口在源仓库 2024-07-10 加入，并持续更新至 2024-08-28。它不是公开版 Lodge
`DanceDiffusion/infer_lodge.py` 的 Global → Local 两阶段入口，而是：

```text
音乐 35D 特征
  → GPT 生成 VQ token
  → VQ-VAE 解码粗动作
  → FineDance key-motion 检索/替换
  → Local Diffusion 细化
  → 139D SMPL body motion
```

当前代码没有 LoRA、PEFT 或 adapter 权重加载逻辑。下面三项都是完整模型 checkpoint。

服务器代码目录：

```text
/efs/nicorhli/project/LODGEPP
```

## 1. 必须取得的三个 LODGEPP checkpoint

### 1.1 FineDance 139D VQ-VAE

历史 139D 推理运行记录中的候选文件：

```text
FineDance_1007_1024_win128/vqvae/f_139/best_recon.pth
```

旧 `Adebug/parameters.yaml` 留下的完整候选路径：

```text
/data2/lrh/project/dance/long/experiments/vqvae/output_vq/clip8_139/FineDance_1007_1024_win128/vqvae/f_139/best_recon.pth
```

加载契约：

```python
ckpt = torch.load(path, map_location="cpu")
net.load_state_dict(ckpt["net"], strict=True)
```

因此最终文件顶层必须包含 `net`。但 LODGEPP 实际使用的 `resume_pth` 保存在当前缺失的
`old-vqvae139.yaml` 中；取回该 YAML 前，不能把上述历史候选路径当作最终确认值。

### 1.2 FineDance 139D GPT

历史 139D 推理运行记录中的候选文件：

```text
FineDance_1007_1024_win128/ckpt/mask_best_acc.pth
```

旧 `Adebug/parameters.yaml` 留下的完整候选路径：

```text
/data2/lrh/project/dance/long/experiments/vqvae/gpt/clip8_139/FineDance_1007_1024_win128/ckpt/mask_best_acc.pth
```

加载契约：

```python
ckpt = torch.load(path, map_location="cpu")
trans_encoder.load_state_dict(ckpt["trans"], strict=True)
```

因此最终文件顶层必须包含 `trans`。但 LODGEPP 实际使用的 `resume_trans` 保存在当前缺失的
`pld-gpt139.yaml` 中；取回该 YAML 前，不能把上述历史候选路径当作最终确认值。

### 1.3 LODGEPP Local Diffusion

精确实验和 checkpoint：

```text
DanceDiffusion/experimentsFD139/FineDance0704/Local_Module/
replace32128_win2_len128_relativeloss_Norm/checkpoints/epoch=2899.ckpt
```

加载契约：

```python
state_dict = torch.load(path, map_location="cpu")["state_dict"]
model_fine.load_state_dict(state_dict, strict=True)
```

因此文件顶层必须包含 `state_dict`，并且必须与下面的 `cfgdiff` 完全匹配。

## 2. 三个 checkpoint 对应的配置文件

权重本身不够。推理脚本先从三个 YAML 构建网络，再以 `strict=True` 加载权重：

```text
DanceDiffusion/experimentsFD139/vqgpt/old-vqvae139.yaml
DanceDiffusion/experimentsFD139/vqgpt/pld-gpt139.yaml
DanceDiffusion/experimentsFD139/FineDance0704/Local_Module/
replace32128_win2_len128_relativeloss_Norm/config_2024-07-06-11-00-05_train.yaml
```

当前 Git 仓库中这三个实验 YAML 也不存在，必须和 checkpoint 一起找回。不能只用
`configs/vqvae/finedance139.yaml` 或 `configs/gpt/finedance139.yaml` 猜测替代，因为网络宽度、
codebook、序列长度或 normalization 设置只要有一项不同，严格加载就会失败。

## 3. 必须取得的非模型资产

### 3.1 FineDance 139D Normalizer

当前资产配置引用：

```text
/data2/lrh/project/dance/Lodge/lodge302/data/Normalizer.pth
```

推理时用于 VQ 输出反归一化、key-motion 归一化以及相邻 Local Diffusion 窗口衔接。必须使用
原训练时同一份文件，不能用当前 315D SMPL-X 原始动作临时生成的统计量替代。

### 3.2 FineDance 139D key-motion 库

当前资产配置引用：

```text
/data2/lrh/dataset/fine_dance/gound/mofea319/keymo
```

推理启动时会遍历其中所有 `.npy`，必要时从 319D 截取前 139D，然后建立检索库。这不是模型
权重，但缺少它时 LODGEPP 主入口无法启动。

### 3.3 FineDance 35D 音乐特征和标签

```text
/data2/lrh/dataset/fine_dance/gound/musicfea_edge
/data2/lrh/dataset/fine_dance/origin/label_json
```

输入也可以从音频现场提取 35D 特征，但当前默认资产路径仍需要迁移。服务器现有 FineDance
归档主要是原始/整理后数据，尚未证明包含与该推理代码完全一致的 `mofea319/keymo` 和
`musicfea_edge`。

## 4. 建议的服务器落盘位置

收到文件后，建议保留实验身份，不要把不同 checkpoint 混放：

```text
/efs/nicorhli/project/LODGEPP/pretrained/lodgepp/vqvae/
  best_recon.pth
  old-vqvae139.yaml

/efs/nicorhli/project/LODGEPP/pretrained/lodgepp/gpt/
  mask_best_acc.pth
  pld-gpt139.yaml

/efs/nicorhli/project/LODGEPP/pretrained/lodgepp/local_diffusion/
  epoch=2899.ckpt
  config_2024-07-06-11-00-05_train.yaml

/efs/nicorhli/project/LODGEPP/pretrained/lodgepp/data/
  Normalizer.pth
  keymo/
```

随后新增一份 EFS 专用资产配置，把旧 `/data2/lrh/...` 路径映射到上述目录，不覆盖历史配置。

## 5. 目前不要下载或混用的权重

- 公开 Lodge 的 `exp.tar.gz`：是公开版 Global → Local 基线，不是当前 LODGEPP 主入口需要的
  VQ-VAE + GPT + Local Diffusion 三权重组合。
- `FineDance_Global/checkpoints/epoch=2999.ckpt`：公开 Lodge Global Diffusion，LODGEPP 主入口
  没有加载它。
- `FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt`：公开 Lodge Local checkpoint，不是
  当前代码指定的 `replace32128.../epoch=2899.ckpt`。
- `experiments/1023edge139_256_35/.../Full-train-3070.pt`：2024-07-10 提交中的旧 DEBUG/EDGE
  路线运行记录；最新版 LODGEPP 主入口改用 Lightning `epoch=2899.ckpt`。
- 263D、266D、290D checkpoint：动作表示不一致，不能与当前 139D 主线混用。
- SMPL、SMPL-H、SMPL-X 参数模型：仅在需要网格渲染时另行准备，不属于这三个生成模型权重。

## 6. 当前来源状态

截至 2026-10-08：

- 当前本地代码、GitHub 仓库和服务器代码目录中均没有上述三个 checkpoint。
- 当前 Git 仓库也没有三份匹配实验 YAML。
- 没有在代码中发现这组三权重的公开下载链接。
- 原始旧路径均位于 `/data2/lrh/...`，最可靠的来源是原训练机备份或原工程归档。
- AWS EFS 上正在下载的 `/efs/nicorhli/lrh-20260828/project.zip.part-00` 很可能包含原工程和
  checkpoint，但下载尚未完成，当前不能将它作为已取得权重。

## 7. 请按这个清单查找或下载

```text
[必须] best_recon.pth
实验身份：FineDance_1007_1024_win128/vqvae/f_139

[必须] mask_best_acc.pth
实验身份：FineDance_1007_1024_win128/ckpt

[必须] epoch=2899.ckpt
实验身份：replace32128_win2_len128_relativeloss_Norm

[必须] old-vqvae139.yaml
[必须] pld-gpt139.yaml
[必须] config_2024-07-06-11-00-05_train.yaml
[必须] Normalizer.pth
[必须] mofea319/keymo 目录
```

如果能从旧机器或工程归档按目录取回，优先打包以下路径，而不是只凭同名文件挑选：

```text
/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion/experimentsFD139/vqgpt/

/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion/experimentsFD139/
FineDance0704/Local_Module/replace32128_win2_len128_relativeloss_Norm/

/data2/lrh/project/dance/long/experiments/vqvae/output_vq/clip8_139/
FineDance_1007_1024_win128/

/data2/lrh/project/dance/long/experiments/vqvae/gpt/clip8_139/
FineDance_1007_1024_win128/
```

取得后必须记录文件大小和 SHA-256，并依次检查：顶层字典键 → `strict=True` 加载 → 单音乐片段
GPU 推理 → 输出 139D motion 的形状和有限值。
