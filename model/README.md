# YOLO + SAM2(Hiera) 杂交检测模型 — NEU-DET

按 `design.md` 中的架构图实现：**冻结的 SAM2 Hiera 编码器 + 逐 stage 可插拔
Adapter（"分支增减"变量）+ YOLO 的 Neck 与解耦检测头**。

```
输入 640×640
   ↓
SAM2 Hiera 编码器（全程冻结）
   Stage1 ──Adapter①(开/关)──► C2  stride 4
   Stage2 ──Adapter②(开/关)──► C3  stride 8
   Stage3 ──Adapter③(开/关)──► C4  stride 16
   Stage4 ──Adapter④(开/关)──► C5  stride 32
   ↓  （C2 下采样后并入 P3，保留裂纹/划痕的高频细节）
Neck：PANet / BiFPN / FPN（可训练）→ P3 P4 P5
   ↓
解耦检测头：分类 / 定位(DFL) / 置信度 三分支（可训练）
   ↓  NMS
6 类缺陷检测框
```

可训练部分 = Adapter + Neck + Head；Hiera 主干（Tiny 约 27M）全程 `requires_grad=False`
且保持 eval 模式，这正是 SAM2-UNet 的训练配方，用来避免 1440 张训练图喂不饱大模型
而过拟合。

## 文件

| 文件 | 内容 |
| --- | --- |
| `hiera.py` | SAM2 Hiera trunk 的自包含实现（参数名与官方一致，可直接加载官方 `.pt`），带逐 stage adapter 插槽与冻结策略 |
| `adapter.py` | 轻量瓶颈 Adapter：`LN → 降维 → GELU → 升维 → GELU`，残差、零初始化（插入时等价恒等映射） |
| `neck.py` | `PANetNeck` / `BiFPNNeck`（可学习跨尺度权重）/ `FPNNeck`，统一消费 C2–C5 |
| `head.py` | Anchor-free 解耦头（cls / reg-DFL / obj）、anchor 生成、box 解码、纯 torch NMS |
| `loss.py` | TaskAlignedAssigner（TOOD/YOLOv8 式动态标签分配）+ BCE / CIoU / SIoU / DFL / objectness |
| `dataset.py` | NEU-DET YOLO txt 格式读取（仅依赖 PIL）、letterbox、翻转/旋转/亮度对比度增强 |
| `metrics.py` | COCO 式 101 点插值 AP：mAP@0.5、mAP@0.5:0.95、逐类 AP/P/R |
| `yolo_sam2.py` | 整体装配、配置系统、参数量 / GFLOPs / FPS 统计、权重保存（只存可训练部分） |
| `train.py` | 训练循环：AdamW + cosine + warmup + EMA + AMP + 梯度裁剪 |
| `val.py` | 评估与效率报告 |
| `ablation.py` | Adapter 开关消融扫描（`add` / `remove` / `single` / `full`）、Neck 对比、骨干规模对比，输出论文用 markdown 表格 |
| `configs/*.yaml` | tiny / small / base_plus / tiny_lite 四套配置 |

依赖只有 `torch`、`numpy`、`PyYAML`、`Pillow`（不需要 ultralytics / torchvision / timm）。

## 快速开始

```bash
cd industrialDefectData

# 1) 冒烟测试：构图 + 前向 + 损失 + 参数量
python -m model.selftest

# 2) 训练（四个 adapter 全开）
python -m model.train --data neu-det-yolo --variant tiny \
    --adapter-stages 1,2,3,4 --epochs 100 --batch-size 8 \
    --sam2-ckpt /path/to/sam2.1_hiera_tiny.pt --name t_all

# 3) 评估
python -m model.val --weights runs/t_all/best.pt --data neu-det-yolo

# 4) 核心消融：从全关开始逐层开启 adapter，3 个随机种子
python -m model.ablation --strategy add --seeds 0,1,2 --epochs 60 \
    --sam2-ckpt /path/to/sam2.1_hiera_tiny.pt
```

### SAM2 预训练权重

不给 `--sam2-ckpt` 时主干为随机初始化（结构可跑通，但没有 SAM2 的先验，精度会很差）。
官方权重从 Meta 的 SAM2 发布页下载（`sam2.1_hiera_tiny.pt` 等），
`hiera.py` 会自动剥掉 `model.image_encoder.trunk.` 前缀后加载。本仓库同级的
`~/sam2` 只是源码，未附带 checkpoint。

## 消融协议（对应 design.md 的四层实验设计）

| 层级 | 变量 | 命令 |
| --- | --- | --- |
| 一：骨干规模 | Hiera tiny / small / base_plus | `--strategy variant` |
| 二：Adapter 开关（核心贡献） | `()`→`(4)`→`(3,4)`→`(2,3,4)`→`(1,2,3,4)`；或从全开逐个关闭 | `--strategy add` / `--strategy remove` |
| 三：Neck 结构 | PANet / BiFPN / FPN | `--strategy neck` |
| 四：细节 | 损失函数 `--iou-type ciou/siou`、轻度解冻 `--train-norm` | 单跑 `model.train` |

`ablation.py` 每个配置固定数据划分、epoch 数、学习率（控制变量），可用 `--seeds 0,1,2`
重复多次并输出 `mean ± std`——这正是 design.md 里引用的 Benchmarking 论文所要求的
"别信单次跑分"。输出 `runs/ablation_*/ablation.json` 与 `ablation.md`（论文可直接用的表格，
同时给出 mAP、可训练参数量、GFLOPs、FPS，用来画"精度–参数量–速度"权衡曲线）。

## 设计说明

- **Adapter 零初始化**：`up` 层权重初始化为 0，插入 adapter 的瞬间输出与纯冻结主干完全一致，
  因此"开启更多 adapter"在训练初期不会破坏预训练特征，消融曲线的比较更公平。
- **C2 分支**：Hiera 的 stride-4 特征不直接建检测层（8400 个 anchor 已足够），而是下采样后与 C3
  融合进 P3，兼顾 crazing / scratches 这类细长低对比缺陷与显存开销。
- **objectness 分支**：按架构图保留三分支；obj 用 IoU 感知的 BCE 训练，推理时与类别分数相乘。
- **冻结主干保持 eval 模式**：`HieraAdapterBackbone.train()` 覆盖了默认行为，避免冻结块里的
  drop-path 在训练时引入无谓噪声。
- **权重保存**：`YOLOSAM2.save()` 只存 adapter/neck/head（几 MB），加载时再从配置重建主干，
  消融跑几十个配置也不会撑爆磁盘。
