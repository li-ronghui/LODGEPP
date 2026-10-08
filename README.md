# LODGEPP

Lodge++: High-quality Long-duration Dance Generation with Robust Choreography.

> **当前状态：初始代码整理版。** 代码、配置和少量骨架辅助文件已经发布；
> 完整环境、训练特征、模型权重和可复现的一键运行入口仍在整理中。

## 1. 代码来源

本仓库的初始代码来自私有仓库 `li-ronghui/Lodge_plus_smpl`：

- 源 commit：`5d82c5d2ea5540e3cbcd18b434bcdac403bf778e`
- 初始导入日期：2026-10-08
- 已排除：Python 字节码、编辑器配置、实验输出、数据集和模型权重

## 2. 服务器部署

当前 AWS H200 环境使用永久 EFS 保存代码和数据：

```text
代码：/efs/nicorhli/project/LODGEPP
FineDance：/efs/nicorhli/data/dataset/dance/opensource/finedance
```

进入项目：

```bash
cd /efs/nicorhli/project/LODGEPP
```

更新代码：

```bash
git pull --ff-only origin main
```

不要把正式代码、数据、checkpoint 或实验结果只保存在 `/data/work`。
该目录是 Pod 本地临时 NVMe，Pod 重建后可能丢失。

## 3. 方法与代码结构

当前 LODGEPP 主推理链由三个生成阶段和一组数据/评测工具组成：

1. **VQ-VAE 动作表征**：学习离散动作 token。
2. **GPT/key-motion 建模**：生成或组织长序列中的关键动作 token。
3. **DanceDiffusion 局部细化**：结合音乐、检索到的 key-motion 和粗动作细化连续舞蹈动作。

主要目录：

| 路径 | 作用 |
| --- | --- |
| `train_vq_clip8.py` | VQ-VAE 训练入口 |
| `test_vq_clip8.py` / `test_vq_clip8_flat.py` | VQ-VAE 测试入口 |
| `train_gpt_mask.py` | GPT/key-motion 训练入口 |
| `DanceDiffusion/` | Lodge++ diffusion 训练、推理和评测代码 |
| `DiffusionNet/` | diffusion 网络与损失实现 |
| `EDGE/` | EDGE 基线及相关数据处理代码 |
| `dataset/` | FineDance/AIST++ 数据读取和预处理 |
| `configs/vqvae/` | VQ-VAE 配置 |
| `configs/gpt/` | GPT 配置 |
| `DanceDiffusion/configs/lodge/` | Lodge++ diffusion 配置 |
| `models/` | VQ-VAE、Transformer 和 diffusion 相关模型 |
| `render.py` / `render_ske.py` | 动作渲染工具 |

## 4. FineDance 数据

FineDance 不提交到 Git。服务器上的持久化数据目录为：

```text
/efs/nicorhli/data/dataset/dance/opensource/finedance
```

当前已核验布局：

| 目录 | 数量 | 内容 |
| --- | ---: | --- |
| `motion/` | 203 | neutral SMPL-X 参数，形状 `[T, 315]` |
| `music_wav/` | 207 | 音频 WAV |
| `music_npy/` | 207 | 音乐特征 NPY |
| `label_json/` | 211 | 风格/标签 JSON |
| `smplx_fix_foot/` | 203 | characterized + Fix Foot FBX |
| `vicon/` | 203 | Vicon retarget FBX |
| `finedance.rar` | 1 | 原始归档 |

原始归档：

```text
size: 4,447,681,161 bytes
sha256: d6ea20546150d76c240215fe6bc9c2aeddc5434e20cf805f3240565570d03a36
```

数据覆盖注意事项：

- `motion` 缺少编号 `116-123`。
- 两个音乐目录缺少 `117`、`118`、`121`、`122`。
- `label_json` 保留 `001-211`；没有为缺失动作补造数据。
- 当前目录是原始/整理后数据，不等同于旧配置要求的全部
  `139/263/266` 维训练特征、key-motion、Mean/Std normalizer。

## 5. 配置现状

代码中仍有大量来自旧机器的绝对路径，例如：

```text
/data2/lrh/...
/data/lrh/...
/home/lrh/...
```

集中出现于：

- `DanceDiffusion/configs/45assets.yaml`
- `configs/vqvae/*.yaml`
- `configs/gpt/*.yaml`
- 部分训练、推理和渲染脚本

因此，**不要直接把旧 README 或脚本中的命令当成已验证命令运行**。
正式训练前至少需要：

1. 将数据、SMPL/SMPL-H/SMPL-X 和 normalizer 路径改到当前 EFS。
2. 确认目标动作表示是 139、263、266 还是其他维度。
3. 生成对应的 motion/music features、key-motion 和 Mean/Std。
4. 补齐 LODGEPP 主线的 VQ-VAE、GPT 和 Local Diffusion checkpoint。
5. 固定 Python、PyTorch、PyTorch3D、PyTorch Lightning 等依赖版本。

## 6. 训练链路草案

以下只表示代码入口关系，不代表当前环境已经完成端到端验收。

### 6.1 VQ-VAE

```bash
python train_vq_clip8.py \
  --cfg configs/vqvae/finedance139.yaml \
  --assets DanceDiffusion/configs/45assets.yaml
```

### 6.2 GPT/key-motion

```bash
python train_gpt_mask.py \
  --cfg configs/gpt/finedance139.yaml \
  --assets DanceDiffusion/configs/45assets.yaml \
  --tkdir /path/to/tokenized_motion \
  --gpu 0
```

### 6.3 Lodge++ diffusion

```bash
cd DanceDiffusion
python train.py \
  --cfg configs/lodge/finedance/finedance_fea139.yaml \
  --cfg_assets configs/45assets.yaml \
  --device 0
```

LODGEPP 主推理入口：

- `infer/infer_gpt_diff_key.py`

它严格加载 FineDance 139D VQ-VAE、GPT 和 Local Diffusion 三个 checkpoint，并依赖匹配的
三份实验 YAML、Normalizer、key-motion 库和 35D 音乐特征。`DanceDiffusion/infer_lodge.py`
属于公开 Lodge 的旧 Global → Local 基线路线，不是当前 LODGEPP 主入口。详细文件身份见
[WEIGHTS.md](WEIGHTS.md)。

## 7. 当前验收状态

已完成：

- GitHub 初版发布。
- 260 个 Python 文件语法解析通过。
- FineDance 原始归档、解压目录和 FBX 后处理结果已在 EFS 核验。
- 代码和数据均有永久存储位置。

尚未完成：

- 可复现的环境文件（`environment.yml` 或完整 `requirements.txt`）。
- 当前 EFS 专用资产配置。
- 训练特征和 normalizer 的完整性检查。
- checkpoint 整理与下载说明。
- 单卡 smoke test、完整训练和推理视频验收。

## 8. 下一步建议

1. 新建一份只针对当前 EFS 的资产配置，不修改和删除历史配置。
2. 固定 Python/CUDA/PyTorch 依赖并建立独立环境。
3. 先跑数据读取和单 batch 前向 smoke test。
4. 再逐级验证 VQ-VAE → GPT → key-motion retrieval → Local Diffusion。
5. 补齐可下载 checkpoint、推理样例和结果视频。

模型权重、Normalizer、SMPL 资产以及下载优先级见 [WEIGHTS.md](WEIGHTS.md)。

## License

本仓库采用 [MIT License](LICENSE)。仓库内引入的第三方代码和数据集仍受各自许可证约束。
