# YOLO + SAM2(Hiera) 运行指南

NEU-DET 钢材表面缺陷检测。模型代码在 `model/`，架构见 `design.md`。

---
a100sh
module load cuda/12.8
source ~/miniconda3/bin/activate
conda activate defect
cd ~/industrialDefectData

## 0. 环境说明（只需读一次）

- **计算节点和登录节点的 Python 不是同一个**，训练一律在计算节点跑：

  | | `python3` | torch | ultralytics |
  | --- | --- | --- | --- |
  | 登录节点 | `/usr/bin/python3` 3.9.25 | 2.8.0 | ✗ 8.4 需要 ≥3.10 |
  | 计算节点（a100sh） | `/appl9/python/3.10.12` | 2.13.0 | ✓ |

  `pip install --user` 会按解释器版本分别装到 `~/.local/lib/python3.9/`
  和 `python3.10/`，两边互不可见——在登录节点装的包，计算节点不一定有。
- **不需要 conda**。`node_env` 是 Node.js 环境（跑 claude CLI 用的），里面没有
  Python，激活它反而可能干扰 PATH。
- SAM2 权重：`~/sam2/checkpoints/sam2.1_hiera_tiny.pt`（已下载，154 个张量验证可加载）。
  `~/sam2` 目录本身只有源码，且因缺 `hydra`/`iopath` 无法 import——模型不依赖它，
  `model/hiera.py` 是自包含实现，只是参数名与官方一致所以能直接读官方权重。
- 依赖仅 torch / numpy / PyYAML / Pillow，不需要 ultralytics、torchvision、timm。
<!-- source miniconda3/bin/activate
conda activate node_env -->
---

## 1. 直接跑（在 `a100sh` 交互会话里）

先申请一个交互式 A100（这条只能你自己在终端敲）：

```bash
a100sh
cd ~/industrialDefectData
```

以下所有命令都在这个会话里执行。

### 1.1 快速自检（约 1 分钟）

改完代码先跑这个：

```bash
python3 -m model.selftest --imgsz 640
```

会检查：所有配置能否构建、损失是否有限、**冻结参数是否真的没有梯度**、
adapter 开关是否只影响 adapter 参数量、mAP 计算是否正确、数据加载是否正常。

### 1.2 基线训练（100 epoch，约 30 分钟）

```bash
python3 -m model.train \
    --data neu-det-yolo \
    --variant tiny \
    --adapter-stages 1,2,3,4 \
    --neck panet \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --lr 1e-3 \
    --seed 0 \
    --workers 8 \
    --eval-every 5 \
    --verbose-eval \
    --sam2-ckpt ~/sam2/checkpoints/sam2.1_hiera_tiny.pt \
    --project runs \
    --name tiny_s1234_panet_seed0
```

一行版（方便复制）：

```bash
python3 -m model.train --data neu-det-yolo --variant tiny --adapter-stages 1,2,3,4 --epochs 100 --batch-size 16 --workers 8 --eval-every 5 --verbose-eval --sam2-ckpt ~/sam2/checkpoints/sam2.1_hiera_tiny.pt --name tiny_s1234_panet_seed0
```

产物在 `runs/tiny_s1234_panet_seed0/`：`best.pt`、`last.pt`、`results.json`
（含逐 epoch 历史、最佳 mAP、参数量/GFLOPs/FPS）。checkpoint 只存 adapter+neck+head，几 MB。

**怕断线就挂后台**（交互会话断了训练也不会死）：

```bash
nohup python3 -m model.train --data neu-det-yolo --epochs 100 --batch-size 16 \
    --workers 8 --sam2-ckpt ~/sam2/checkpoints/sam2.1_hiera_tiny.pt \
    --name tiny_s1234_panet_seed0 > train.log 2>&1 &

tail -f train.log
```

### 1.3 消融实验（核心，论文主图）

```bash
python3 -m model.ablation \
    --strategy add \
    --seeds 0,1,2 \
    --epochs 60 \
    --data neu-det-yolo \
    --variant tiny \
    --batch-size 16 \
    --workers 8 \
    --eval-every 10 \
    --sam2-ckpt ~/sam2/checkpoints/sam2.1_hiera_tiny.pt \
    --project runs/ablation_add
```

- `add` 策略：`()` → `(4)` → `(3,4)` → `(2,3,4)` → `(1,2,3,4)`，看加到第几层性价比最高
- 5 个配置 × 3 个种子 = 15 次训练，串行约 4–5 小时（建议配 `nohup`）
- 想先看趋势就用 `--seeds 0`，约 1.5 小时

输出：

- `runs/ablation_add/ablation.json` — 完整数据
- `runs/ablation_add/ablation.md` — 论文可直接用的表格（mAP ± std、可训练参数量、
  GFLOPs、FPS），拿来画"开启 stage 数 × mAP × FPS"权衡曲线

其他策略把 `--strategy` 换掉即可：

```bash
--strategy remove    # 从全开逐个关，看哪层最重要
--strategy single    # 每次只开一个
--strategy neck      # PANet / BiFPN / FPN
--strategy variant   # Hiera tiny / small / base_plus
```

### 1.4 评估某个 checkpoint

```bash
python3 -m model.val --weights runs/tiny_s1234_panet_seed0/best.pt --data neu-det-yolo
```

输出逐类 P / R / AP50 / AP50-95，以及参数量、GFLOPs、FPS。加 `--json out.json` 存文件。

### 1.5 只看效率指标（参数量 / GFLOPs / FPS）

```bash
python3 -c "
from model import build_model
m = build_model('model/configs/yolo_sam2_tiny.yaml').cuda()
print(m.param_counts())
print('GFLOPs', m.flops(640))
print('FPS b1', m.benchmark_fps(640, batch=1), 'FPS b8', m.benchmark_fps(640, batch=8))
"
```

---

## 2. 训练参数速查

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--variant` | `tiny` | Hiera 规模：`tiny` / `small` / `base_plus` / `large` |
| `--adapter-stages` | `1,2,3,4` | 哪几个 stage 插 adapter（**核心变量**），`none` 表示全不插 |
| `--adapter-ratio` | `4` | adapter 瓶颈比，越大越省参数 |
| `--neck` | `panet` | `panet` / `bifpn` / `fpn` |
| `--neck-width` | `192` | neck 通道数 |
| `--iou-type` | `ciou` | `ciou` / `siou` / `iou` |
| `--epochs` | `100` | 训练轮数 |
| `--batch-size` | `8` | A100 40GB 下 640px 可到 32 |
| `--imgsz` | `640` | 输入尺寸 |
| `--lr` / `--lrf` | `1e-3` / `0.01` | 初始学习率 / 末端衰减比例 |
| `--warmup-epochs` | `3` | 学习率预热 |
| `--seed` | `0` | 随机种子 |
| `--eval-every` | `5` | 每几个 epoch 评估一次 |
| `--sam2-ckpt` | 无 | **必填**，不给就是随机初始化主干 |
| `--train-norm` | 关 | 额外解冻主干 LayerNorm（轻度微调消融） |
| `--no-aug` / `--no-ema` / `--no-amp` | — | 关掉增强 / EMA / 混合精度 |
| `--name` / `--project` | `exp` / `runs` | 输出目录 |

---

## 4. 已实测的基准数据（A100-PCIE-40GB）

| 指标 | 数值 |
| --- | --- |
| 单 epoch 用时（640px, batch 16） | 约 15–22 秒 |
| GFLOPs @ 640 | 100.5 |
| FPS | 77.8（batch=1） / 109.9（batch=8） |
| 参数量 | 冻结主干 26.85M + 可训练 4.25M（其中 adapter 0.104M） |

注意：**mAP 目前还没有有效结果**。冒烟测试只跑了 2 epoch（mAP50 = 0.061），
只能证明流程通了，第一个能看的精度数字要等 100 epoch 基线跑完。

---

## 5. 其他 Hiera 权重下载

`--variant small` / `base_plus` 需要对应权重：

```bash
cd ~/sam2/checkpoints
curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

---

## 6. 实验顺序建议（design.md 的四层设计）

1. **骨干规模** — `--strategy variant`，先定 tiny / small / base_plus
2. **adapter 开关** — `--strategy add` + `--strategy remove`，这是核心贡献，做最细
3. **neck 结构** — `--strategy neck`，不是创新点，选个好的即可
4. **细节** — 损失函数 `--iou-type siou`、轻度解冻 `--train-norm`，补充消融

每次只变一个维度，固定数据划分、epoch 数、学习率。至少 3 个种子报 mean ± std
（`--seeds 0,1,2`），这是 design.md 引用的 Benchmarking 那篇（J. Intelligent
Manufacturing 2025）的要求。

---

## 7. 已知问题 / 注意事项

- **必须有 SAM2 权重**：不给 `--sam2-ckpt` 时主干是随机初始化的，代码能跑通但精度没有意义。
  会打 WARNING，别忽略。
- **epoch 数太少时学习率调度会退化**：warmup 默认 3 epoch，如果只跑 2 epoch，
  cosine 会立刻掉到 1e-5。冒烟测试 epoch 1 的 mAP 为 0 就是这个原因，正式训练无影响。
- **objectness 分支**：按 design.md 的架构图保留了 cls/reg/obj 三分支。现代 YOLO（v8/v10）
  已经去掉 obj，因为 TAL 的软标签本身就编码了定位质量。审稿人可能会问，
  想去掉就在 config 里设 `head.use_obj: false`，损失函数已经支持这条路径。
- **参数量争议**：总参数 31M（其中冻结 26.85M），比 Dynamic-YOLO 的 1.6M 重很多。
  论文里应该强调"只训练 4.25M 参数"，并把 FPS（78）和 GFLOPs（100.5）如实报出来，
  这是这个方案最容易被质疑的点。








python3 -m model.val --weights runs/f_cnn_baseline/best.pt --data neu-det-yolo --split train
class                n_gt        P        R     AP50   AP50-95
crazing               527    0.077    0.156   0.0463    0.0112
inclusion             852    0.691    0.585   0.6826    0.3652
patches               688    0.820    0.594   0.7362    0.4618
pitted_surface        345    0.061    0.017   0.0129    0.0061
rolled-in_scale       496    0.371    0.399   0.3559    0.1619
scratches             427    0.498    0.663   0.5622    0.2197
all                          0.420    0.402   0.3993    0.2043
{
  "trunk_frozen_M": 4.423,
  "adapters_M": 0.0,
  "neck_M": 2.533,
  "head_M": 1.516,
  "total_M": 8.472,
  "trainable_M": 8.472,
  "adapter_stages": [],
  "GFLOPs": 34.067,
  "FPS": 110.91
}