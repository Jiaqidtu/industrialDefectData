**1. Information Fusion（中科院一区，IF≈15.5，Elsevier旗下顶刊）**

《A Real-Time Surface Defect Detection Model Based on Adaptive Feature Information Selection and Fusion》，2025年发表。做的是实时表面缺陷检测，核心是自适应特征信息选择与融合模块。这是目前找到的期刊等级最高的一篇。

链接：https://www.sciencedirect.com/science/article/pii/S1566253525011030

**2. Journal of Intelligent Manufacturing（中科院一区，IF≈7.7，Springer）**

- 《Benchmarking deep learning models for surface defect detection: a reproducible and statistically-rigorous approach》，2025年9月发表（Vol 37, Issue 7, pp 3001–3018）。作者Darío G. Lema等，明确在NEU-DET数据集上做了系统性对比，用四折交叉验证+ANOVA/Tukey检验评估了多个YOLO版本和自定义架构，指出很多论文声称的"性能提升"其实统计上不显著——这篇对你们做课题综述/方法论参考价值很高。
链接：https://link.springer.com/article/10.1007/s10845-025-02672-8
- 《AMFNet: aggregated multi-level feature interaction fusion network for defect detection on steel surfaces》，2025年发表，同刊。
链接：https://link.springer.com/article/10.1007/s10845-025-02613-5

**3. Engineering Applications of Artificial Intelligence（中科院一区，Elsevier）**

《A novel real-time steel surface defect detection method with enhanced feature extraction and adaptive fusion》

链接：https://www.sciencedirect.com/science/article/abs/pii/S0952197624014477

**4. Complex & Intelligent Systems（中科院一区，Springer，稍早一些）**

《A deep learning model for steel surface defect detection》，2023年发表，如果不严格限定"最新"也可以参考。

链接：https://link.springer.com/article/10.1007/s40747-023-01180-7

几点提醒：

NEU-DET本身是一个偏"工业视觉/缺陷检测"的小众数据集，很少出现在CVPR/ICCV/ECCV这类通用视觉顶会论文中（这类会议更常用COCO、ImageNet或自建大规模数据集），所以这个方向的顶级成果基本都集中在期刊（尤其是Information Fusion、IEEE TII、IEEE TIM、Pattern Recognition、Journal of Intelligent Manufacturing、Engineering Applications of AI这几本）。如果你们的要求里"CVPR等会议"是硬性的，可能需要放宽到看有没有相关的workshop论文，或者接受期刊为主的检索结果。

另外，"SCI二区以上"我这边是按JCR/中科院分区口径大致核实的（Information Fusion、J. Intelligent Manufacturing、EAAI、Complex & Intelligent Systems均为一区），但具体是否满足你们学校/学院认定的分区标准，建议再用中科院文献情报中心的分区表或者你们学校的期刊目录复核一遍，因为不同机构对"二区"的划分标准可能不完全一致。

我已经把五篇论文的正文（摘要+方法+架构图部分）都仔细读完了，现在信息很扎实，可以做一份准确的对比表。

| 维度 | CAT-YOLO（EAAI, 2024） | Dynamic-YOLO（Information Fusion, 2026） | AMFNet（J. Intelligent Manufacturing, 2025） | 改进YOLOv5s模型（Complex & Intelligent Systems, 2024） | 基准评测研究（J. Intelligent Manufacturing, 2025） |
| --- | --- | --- | --- | --- | --- |
| **作者/单位** | Jie Yang, Zhixin Liu（东北大学） | Li-Juan Liu等（北京语言大学/大连交通大学/米兰理工） | Changyun Wei等（河海大学/卡迪夫大学） | Zhaoguo Li等（齐鲁工业大学） | Darío G. Lema等（西班牙奥维耶多大学） |
| **检测范式** | 单阶段（YOLOv5s改进） | 单阶段（YOLOv8改进） | **两阶段**（Faster R-CNN + FPN + ResNet50） | 单阶段（YOLOv5s改进） | 不提出新模型，是**多模型统计对比研究**（YOLOv3~v10等9种模型） |
| **骨干网络** | CSP-FN block（融合FasterNet的PConv到CSP结构里，降算力保精度） | 沿用YOLOv8骨干，插入DDConv（双路下采样卷积） | ResNet50 + FPN（C1~C5多尺度特征金字塔） | 深度加深的YOLOv5s（stage比例1:2:3:1→2:2:4:2） | 直接用各原始模型的骨干（不改动） |
| **核心创新模块** | ① AF-SPPF：自适应融合空间金字塔池化，用softmax权重过滤不同感受野间的冲突信息② MTSCODE：任务特定解耦头，分类/定位分支分别用不同层级特征 | ① DDConv：双路下采样（Focus分支+MaxPool分支并行，捕获细节+全局）② Dy-CCFM：动态跨尺度特征融合，用高层特征动态加权低层特征（对标FPN/PANet/BiFPN） | ① BIM：分支交互模块，通道分组+不同膨胀率(3/5/7)空洞卷积+跳连② SCM：空间相关模块，Q·K点乘+sigmoid生成空间注意力权重③ DCM：膨胀上下文模块，多尺度感受野融合 | ① MSFE：三分支（1×1/3×3/5×5卷积）多尺度特征提取，替换C3② EFF：把骨干特征直接注入PANet自底向上路径，防止深层特征退化③ 简化Bottleneck：借鉴ConvNeXt，减少归一化/激活层数量 | 无新模块，核心贡献是**四折分层交叉验证+ANOVA/Tukey检验**的统计学评估框架 |
| **损失函数** | 未特别设计（沿用YOLO常规loss） | **MPDIoU**：基于最小点距离的IoU改进，专门优化小目标（裂纹、划痕）定位精度 | 未提及专用loss创新 | 常规YOLO loss | 不适用（评测研究） |
| **参数量/速度** | 27.7MB，102 FPS | 仅**1.6M参数**（极轻量），高推理速度 | 未强调轻量化，两阶段结构参数量相对更大 | 参数量小，未给出FPS，强调"精度-参数量"平衡 | 不适用 |
| **NEU-DET结果** | mAP 85.2%（比基线YOLOv5s高5.3%，比第二名高3.1%） | mAP 75.1%（比YOLOv8高2.1%，比DETR高5.0%） | 显著优于经典检测器（未给出具体数字，摘要中强调"significantly outperforms"） | mAP@0.5 = 73.08%（参数量最省） | 揭示很多论文声称的"提升"在统计上并不显著（即使2.4%的AP提升也可能不显著） |
| **设计哲学** | "看得更全"——多模块协同（特征强化+自适应池化+任务解耦），偏精度导向 | "选得更准"——动态特征选择与融合，强调对复杂背景/小目标的自适应性，是**轻量化和精度的双料优等生** | "两阶段的精细化"——回归Faster R-CNN路线，靠多层次特征交互+双重注意力机制榨取精度，计算成本更高 | "轻装上阵"——单阶段YOLO早期改进代表，模块设计相对简单直接，是这几篇里最"轻"的基线型工作 | "别急着信提升"——方法论批判性研究，提醒同行注意数据集划分和随机性对结果的影响，这篇不能拿来做"性能SOTA"参考，但**做文献综述/实验设计部分极有价值** |
| **期刊定位** | Q1（工程应用AI类） | Q1（信息融合理论顶刊，IF最高） | Q1（智能制造） | Q1（复杂智能系统） | Q1（智能制造，方法论/评测类） |

几个可以直接用在课题里的结论性观察：

这五篇大致能分成三类角色。AMFNet走的是"两阶段+多重注意力"的精度优先路线，代价是模型更重；Dynamic-YOLO和CAT-YOLO都是"单阶段YOLO+自适应特征融合"的思路，但Dynamic-YOLO更进一步做到了1.6M的极致轻量化，同时精度也不差，是这几篇里工程落地性价比最高的；改进YOLOv5s模型（Complex & Intelligent Systems, 2024）是相对早期、结构最朴素的基线型工作，适合作为"经典改进方法"的对照组。而基准评测研究这篇要单独看待——它不是提出新架构，而是在质疑整个领域"精度年年涨"的说法是否经得起统计检验，这一点如果你们的课题涉及方法对比或实验设计部分，引用价值会非常高（可以用来支撑"我们的对比实验采用了统计显著性检验"这类严谨性论证）。

**SAM2-UNet**（Visual Intelligence, 2025）已经做过"把SAM2的Hiera编码器当骨干网络"这件事，只不过用在医学/自然图像分割的U-Net上，没人把它搬到YOLO检测头+NEU-DET这个场景，这就是你们的空子。

先说这个先例怎么做的，因为可以直接借鉴：SAM2-UNet**冻结**Hiera编码器本体，只在每个多尺度block前面插入一个轻量adapter（降维线性层→GELU→升维线性层→GELU），只训练adapter和解码器部分。这个设计的意义在于——Hiera-L本身212M参数，如果全量微调，NEU-DET那点数据（1800张原图）根本喂不饱，会严重过拟合；但冻结主干、只调adapter，相当于把SAM2当一个"通用视觉特征提取器"来用，训练成本和过拟合风险都大大降低。这个思路你们可以原样搬过来，把它的U-Net解码器换成YOLO的neck+head就行，架构上是相通的。

**为什么Hiera这个编码器天然适合当YOLO骨干**：它本身就是分层结构，会输出四个不同尺度的特征图（类似FPN的C2-C5），这跟YOLO的neck（PANet/BiFPN）需要的多尺度输入格式几乎是天然对齐的，不需要额外设计特征对齐层，工程上改造成本比想象中低。

但有三个硬问题需要你们提前想清楚，也是审稿人一定会问的：

**第一，参数量和"实时性"的矛盾会被放大。** 之前那几篇NEU-DET顶刊论文都在拼轻量化（Dynamic-YOLO只有1.6M参数，CAT-YOLO 27.7MB），SAM2哪怕最小的Hiera-Tiny也有近40M参数，比YOLOv5s整个骨干（约7M）还大好几倍。你们要么明确把应用场景从"产线实时检测"改成"离线高精度质检"，要么走"训练时用大Hiera+adapter，训完之后蒸馏/剪枝出一个小模型再部署"的两阶段策略——后者其实是个不错的创新点，可以叫"SAM2引导的知识蒸馏"，比直接怼进去更站得住脚。

**第二，域差距（domain gap）是真实存在的。** SAM2是在自然图像/视频（SA-V数据集）上预训练的，NEU-DET是灰度纹理图像，视觉分布差异很大。SAM2-UNet能work是因为它验证场景还是自然图像和医学图像（跟自然图像更接近），钢材表面纹理这种工业灰度图迁移过去效果存疑，需要你们做实验验证——这也是论文里必须交代清楚的实验部分，不能想当然认为"大模型=好特征"。

**第三，回到你说的"增加或减少分支"**，我觉得这个点可以和adapter机制结合得很自然：Hiera一共4个stage，你们完全可以把"给几个stage插入adapter"设计成一个可调的架构变量——比如只在深层2个stage插adapter（更省参数、更快，适合抓大范围/低频缺陷如结疤、氧化）、还是四个stage全插（更精细，适合裂纹这种细节缺陷但更慢）。这样"分支增减"就不再是一个模糊的想法，而是一个具体的、可以做消融实验的架构选择，而且直接对应你们要写的"如何通过增减分支改进架构"这个核心贡献点，实验设计也很清楚：固定其他条件，只变adapter插入的stage数量，画一条"精度-参数量-FPS"的权衡曲线，这本身就是一个完整的实验章节。

综合下来，我建议的具体路线是：**冻结的Hiera-Tiny/Small编码器 + 逐stage可插拔adapter（对应"分支增减"）+ YOLO的neck和head**，训练时只更新adapter和neck/head，用消融实验证明"adapter插入的stage数量"这个变量对精度和速度的影响规律，这样你们的贡献点、实验设计、和现有工作的差异化都比较清楚了。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>YOLO + SAM2(Hiera)可插拔骨干网络架构</title>
<style>
  :root{
    --bg:#faf9f7;
    --ink:#1f2328;
    --sub:#6b7280;
    --frozen-fill:#eef0f3;
    --frozen-stroke:#9aa3ad;
    --train-fill:#fdf1e7;
    --train-stroke:#d97b3f;
    --neck-fill:#e9f2ef;
    --neck-stroke:#3f8f74;
    --head-fill:#eef0fb;
    --head-stroke:#5b5fc7;
    --out-fill:#fff;
    --out-stroke:#1f2328;
    --arrow:#8a8f98;
    --accent:#d97b3f;
  }
  *{box-sizing:border-box;}
  body{
    margin:0;
    background:var(--bg);
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;
    color:var(--ink);
    padding:32px 16px 48px;
  }
  .wrap{max-width:1180px;margin:0 auto;}
  h1{
    font-size:20px;
    font-weight:700;
    margin:0 0 4px;
    text-align:center;
  }
  .subtitle{
    text-align:center;
    color:var(--sub);
    font-size:13px;
    margin:0 0 28px;
  }
  .legend{
    display:flex;
    justify-content:center;
    gap:22px;
    flex-wrap:wrap;
    margin-bottom:18px;
    font-size:12.5px;
    color:var(--sub);
  }
  .legend-item{display:flex;align-items:center;gap:6px;}
  .swatch{width:13px;height:13px;border-radius:3px;display:inline-block;flex:none;}
  .diagram-card{
    background:#fff;
    border:1px solid #e7e5e1;
    border-radius:16px;
    padding:28px 20px 20px;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
  }
  svg text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;}
  .footnote{
    margin-top:18px;
    font-size:12.5px;
    color:var(--sub);
    line-height:1.7;
    border-top:1px solid #eee;
    padding-top:14px;
  }
  .footnote b{color:var(--ink);}
</style>
</head>
<body>
<div class="wrap">
  <h1>YOLO 检测头 + SAM2 (Hiera) 可插拔骨干网络架构</h1>
  <p class="subtitle">冻结主干 · 逐 Stage 可插拔 Adapter（对应"分支增减"）· 多尺度特征输入 Neck/Head</p>

  <div class="legend">
    <span class="legend-item"><span class="swatch" style="background:var(--frozen-fill);border:1.5px solid var(--frozen-stroke);"></span>冻结参数（不训练）</span>
    <span class="legend-item"><span class="swatch" style="background:var(--train-fill);border:1.5px solid var(--train-stroke);"></span>可训练 Adapter（分支开关）</span>
    <span class="legend-item"><span class="swatch" style="background:var(--neck-fill);border:1.5px solid var(--neck-stroke);"></span>Neck（特征融合，可训练）</span>
    <span class="legend-item"><span class="swatch" style="background:var(--head-fill);border:1.5px solid var(--head-stroke);"></span>Head（检测头，可训练）</span>
  </div>

  <div class="diagram-card">
    <svg viewBox="0 0 1140 760" width="100%" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#8a8f98"/>
        </marker>
        <marker id="arrowAccent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#d97b3f"/>
        </marker>
      </defs>

      <!-- Input -->
      <g>
        <rect x="20" y="330" width="110" height="70" rx="10" fill="#fff" stroke="#1f2328" stroke-width="1.6"/>
        <text x="75" y="358" text-anchor="middle" font-size="13" font-weight="600">输入图像</text>
        <text x="75" y="376" text-anchor="middle" font-size="10.5" fill="#6b7280">NEU-DET 200×200</text>
        <text x="75" y="390" text-anchor="middle" font-size="10.5" fill="#6b7280">灰度缺陷图</text>
      </g>
      <path d="M130 365 L168 365" stroke="#8a8f98" stroke-width="1.6" marker-end="url(#arrow)"/>

      <!-- Hiera Backbone container -->
      <rect x="172" y="60" width="330" height="640" rx="14" fill="none" stroke="#c7cbd1" stroke-width="1.4" stroke-dasharray="5 4"/>
      <text x="337" y="42" text-anchor="middle" font-size="13.5" font-weight="700">SAM2 Hiera 编码器（冻结主干，Hiera-T/S 二选一）</text>

      <!-- 4 Stages -->
      <!-- Stage 1 -->
      <g>
        <rect x="196" y="80" width="282" height="130" rx="10" fill="#eef0f3" stroke="#9aa3ad" stroke-width="1.4"/>
        <text x="210" y="102" font-size="12" font-weight="600">Stage 1 · 高分辨率</text>
        <text x="210" y="118" font-size="10" fill="#6b7280">下采样×4，通道数最少</text>
        <rect x="330" y="88" width="130" height="46" rx="8" fill="#fdf1e7" stroke="#d97b3f" stroke-width="1.4" stroke-dasharray="4 3"/>
        <text x="395" y="106" text-anchor="middle" font-size="10.5" font-weight="600" fill="#d97b3f">Adapter①</text>
        <text x="395" y="120" text-anchor="middle" font-size="9" fill="#b5673a">开关：ON/OFF</text>
        <rect x="330" y="148" width="130" height="46" rx="8" fill="#fff" stroke="#c7cbd1" stroke-width="1.2"/>
        <text x="395" y="166" text-anchor="middle" font-size="9.5" fill="#6b7280">↓降维→GELU</text>
        <text x="395" y="180" text-anchor="middle" font-size="9.5" fill="#6b7280">↑升维→GELU</text>
      </g>
      <text x="490" y="150" font-size="16" fill="#c7cbd1">⇣</text>

      <!-- Stage 2 -->
      <g>
        <rect x="196" y="222" width="282" height="130" rx="10" fill="#eef0f3" stroke="#9aa3ad" stroke-width="1.4"/>
        <text x="210" y="244" font-size="12" font-weight="600">Stage 2</text>
        <text x="210" y="260" font-size="10" fill="#6b7280">下采样×8</text>
        <rect x="330" y="230" width="130" height="46" rx="8" fill="#fdf1e7" stroke="#d97b3f" stroke-width="1.4" stroke-dasharray="4 3"/>
        <text x="395" y="248" text-anchor="middle" font-size="10.5" font-weight="600" fill="#d97b3f">Adapter②</text>
        <text x="395" y="262" text-anchor="middle" font-size="9" fill="#b5673a">开关：ON/OFF</text>
      </g>

      <!-- Stage 3 -->
      <g>
        <rect x="196" y="364" width="282" height="130" rx="10" fill="#eef0f3" stroke="#9aa3ad" stroke-width="1.4"/>
        <text x="210" y="386" font-size="12" font-weight="600">Stage 3</text>
        <text x="210" y="402" font-size="10" fill="#6b7280">下采样×16</text>
        <rect x="330" y="372" width="130" height="46" rx="8" fill="#fdf1e7" stroke="#d97b3f" stroke-width="1.4" stroke-dasharray="4 3"/>
        <text x="395" y="390" text-anchor="middle" font-size="10.5" font-weight="600" fill="#d97b3f">Adapter③</text>
        <text x="395" y="404" text-anchor="middle" font-size="9" fill="#b5673a">开关：ON/OFF</text>
      </g>

      <!-- Stage 4 -->
      <g>
        <rect x="196" y="506" width="282" height="130" rx="10" fill="#eef0f3" stroke="#9aa3ad" stroke-width="1.4"/>
        <text x="210" y="528" font-size="12" font-weight="600">Stage 4 · 低分辨率</text>
        <text x="210" y="544" font-size="10" fill="#6b7280">下采样×32，语义最强</text>
        <rect x="330" y="514" width="130" height="46" rx="8" fill="#fdf1e7" stroke="#d97b3f" stroke-width="1.4" stroke-dasharray="4 3"/>
        <text x="395" y="532" text-anchor="middle" font-size="10.5" font-weight="600" fill="#d97b3f">Adapter④</text>
        <text x="395" y="546" text-anchor="middle" font-size="9" fill="#b5673a">开关：ON/OFF</text>
      </g>

      <text x="337" y="668" text-anchor="middle" font-size="10.5" fill="#6b7280">四个 Stage 各自的 Adapter 独立开关 → 即"分支增减"消融变量</text>
      <text x="337" y="684" text-anchor="middle" font-size="10.5" fill="#6b7280">开启越多：精度↑ 参数量/延迟↑；开启越少：速度↑ 参数量↓</text>

      <!-- vertical connectors stage to stage -->
      <path d="M337 210 L337 222" stroke="#9aa3ad" stroke-width="1.4" marker-end="url(#arrow)"/>
      <path d="M337 352 L337 364" stroke="#9aa3ad" stroke-width="1.4" marker-end="url(#arrow)"/>
      <path d="M337 494 L337 506" stroke="#9aa3ad" stroke-width="1.4" marker-end="url(#arrow)"/>

      <!-- Feature outputs to Neck -->
      <path d="M478 145 L560 145 L560 250" stroke="#8a8f98" stroke-width="1.4" marker-end="url(#arrow)" fill="none"/>
      <path d="M478 287 L555 287 L555 310" stroke="#8a8f98" stroke-width="1.4" marker-end="url(#arrow)" fill="none"/>
      <path d="M478 429 L555 429 L555 390" stroke="#8a8f98" stroke-width="1.4" marker-end="url(#arrow)" fill="none"/>
      <path d="M478 571 L560 571 L560 460" stroke="#8a8f98" stroke-width="1.4" marker-end="url(#arrow)" fill="none"/>
      <text x="500" y="230" font-size="9.5" fill="#9aa3ad">C2</text>
      <text x="500" y="305" font-size="9.5" fill="#9aa3ad">C3</text>
      <text x="500" y="405" font-size="9.5" fill="#9aa3ad">C4</text>
      <text x="500" y="480" font-size="9.5" fill="#9aa3ad">C5</text>

      <!-- Neck -->
      <g>
        <rect x="570" y="230" width="200" height="260" rx="12" fill="#e9f2ef" stroke="#3f8f74" stroke-width="1.6"/>
        <text x="670" y="255" text-anchor="middle" font-size="13" font-weight="700" fill="#2f6e58">Neck</text>
        <text x="670" y="272" text-anchor="middle" font-size="10" fill="#4a7e6b">PANet / BiFPN 风格</text>
        <text x="670" y="288" text-anchor="middle" font-size="10" fill="#4a7e6b">多尺度特征融合</text>

        <rect x="592" y="308" width="156" height="34" rx="7" fill="#fff" stroke="#3f8f74" stroke-width="1.1"/>
        <text x="670" y="330" text-anchor="middle" font-size="10">自顶向下 · 语义传递</text>

        <rect x="592" y="352" width="156" height="34" rx="7" fill="#fff" stroke="#3f8f74" stroke-width="1.1"/>
        <text x="670" y="374" text-anchor="middle" font-size="10">自底向上 · 位置传递</text>

        <rect x="592" y="396" width="156" height="34" rx="7" fill="#fff" stroke="#3f8f74" stroke-width="1.1"/>
        <text x="670" y="418" text-anchor="middle" font-size="10">跨尺度加权融合</text>

        <text x="670" y="460" text-anchor="middle" font-size="10" fill="#4a7e6b">输出 P3 / P4 / P5</text>
        <text x="670" y="475" text-anchor="middle" font-size="9.5" fill="#6b8f80">（小/中/大目标三支路）</text>
      </g>

      <path d="M770 360 L810 360" stroke="#8a8f98" stroke-width="1.6" marker-end="url(#arrow)"/>

      <!-- Head -->
      <g>
        <rect x="815" y="150" width="220" height="420" rx="12" fill="#eef0fb" stroke="#5b5fc7" stroke-width="1.6"/>
        <text x="925" y="178" text-anchor="middle" font-size="13" font-weight="700" fill="#3d3fa8">解耦检测头 Head</text>

        <rect x="836" y="200" width="178" height="90" rx="8" fill="#fff" stroke="#5b5fc7" stroke-width="1.2"/>
        <text x="925" y="222" text-anchor="middle" font-size="11" font-weight="600">分类分支 Cls</text>
        <text x="925" y="240" text-anchor="middle" font-size="9.5" fill="#6b7280">Crazing / Inclusion</text>
        <text x="925" y="254" text-anchor="middle" font-size="9.5" fill="#6b7280">Patches / Pitted</text>
        <text x="925" y="268" text-anchor="middle" font-size="9.5" fill="#6b7280">Rolled-in / Scratches</text>

        <rect x="836" y="304" width="178" height="60" rx="8" fill="#fff" stroke="#5b5fc7" stroke-width="1.2"/>
        <text x="925" y="326" text-anchor="middle" font-size="11" font-weight="600">定位分支 Reg</text>
        <text x="925" y="344" text-anchor="middle" font-size="9.5" fill="#6b7280">Bounding Box (x,y,w,h)</text>

        <rect x="836" y="380" width="178" height="46" rx="8" fill="#fff" stroke="#5b5fc7" stroke-width="1.2"/>
        <text x="925" y="398" text-anchor="middle" font-size="11" font-weight="600">置信度分支 Obj</text>
        <text x="925" y="414" text-anchor="middle" font-size="9.5" fill="#6b7280">Confidence Score</text>

        <path d="M925 426 L925 448" stroke="#5b5fc7" stroke-width="1.4" marker-end="url(#arrow)"/>
        <rect x="836" y="452" width="178" height="46" rx="8" fill="#fff" stroke="#5b5fc7" stroke-width="1.2"/>
        <text x="925" y="470" text-anchor="middle" font-size="10.5" font-weight="600">NMS 非极大值抑制</text>
        <text x="925" y="484" text-anchor="middle" font-size="9" fill="#6b7280">筛选最终检测框</text>
      </g>

      <path d="M1035 475 L1075 475" stroke="#8a8f98" stroke-width="1.6" marker-end="url(#arrow)"/>

      <!-- Output -->
      <g>
        <rect x="1078" y="425" width="55" height="100" rx="10" fill="#fff" stroke="#1f2328" stroke-width="1.6"/>
        <text x="1105" y="470" text-anchor="middle" font-size="10.5" font-weight="600" transform="rotate(0 1105 470)">检测</text>
        <text x="1105" y="485" text-anchor="middle" font-size="10.5" font-weight="600">结果</text>
      </g>

      <!-- Bottom note bracket for trainable scope -->
      <path d="M330 700 L768 700" stroke="#d97b3f" stroke-width="1.6" stroke-dasharray="3 3"/>
      <text x="549" y="716" text-anchor="middle" font-size="10.5" fill="#b5673a">此区间（Adapter + Neck + Head）为可训练部分，Hiera 编码器主体全程冻结</text>
    </svg>
  </div>

  <div class="footnote">
    <b>读图说明：</b>灰色虚线大框内是 SAM2 的 Hiera 编码器，四个 Stage 依次降低分辨率、提升语义抽象层级，主体参数全程冻结不参与训练；每个 Stage 前挂一个橙色可插拔 Adapter（降维→GELU→升维→GELU 的轻量瓶颈结构），四个 Adapter 的开关组合就是论文里可以做消融实验的"分支增减"变量——开启的 Adapter 越多，特征适配能力越强、精度通常越高，但参数量和推理延迟也随之上升，可以画一条"开启 Stage 数 × mAP × FPS"的权衡曲线作为核心实验图。绿色 Neck 部分沿用 YOLO 常见的 PANet/BiFPN 双向融合结构，接收 C2–C5 四层特征；紫色 Head 部分是解耦检测头，分类、定位、置信度三条分支独立预测，最后经 NMS 输出六类缺陷（Crazing、Inclusion、Patches、Pitted_surface、Rolled-in_scale、Scratches）的检测框。
  </div>
</div>
</body>
</html>

```

对，这些选择基本都没法靠"拍脑袋"或者纯理论推导定下来，必须靠消融实验（ablation study）来决定——这也是这类架构改进论文能不能过审的关键部分，审稿人几乎一定会问"这个模块/超参数是怎么选出来的"。而且好消息是：把这些选择做成实验本身就是你们论文实验章节的主要内容，不是额外负担，是必需品。

具体到你们这个架构，需要通过实验决定的点大概分四类：

**第一层：骨干网络规模**——Hiera-Tiny / Small / Base+ 三选一（或几个都跑对比）。这个决定参数量和速度的基准线，应该最先确定，因为后面所有实验都要在这个基准上做。

**第二层：Adapter开关组合（你们的核心创新点）**——四个Stage、理论上有16种开关组合，但不需要穷举，常见做法是两种消融策略：一种是"从全开始逐个关闭"（看关掉哪个Stage的adapter掉点最多，说明哪层最重要），另一种是"从全关始逐个开启"（看加到第几层性价比最高）。这部分要做得最细，因为这是你们论文要重点讲的贡献，最后画一条"开启Stage数 vs mAP vs FPS"的曲线，这张图基本就是论文的核心实验图。

**第三层：Neck结构**——PANet / BiFPN / 简单FPN怎么选，也是实验对比，一般在前两层结果确定之后再做，因为Neck不是你们的创新点，选个效果好、写法标准的即可，不用花太多篇幅。

**第四层：其他细节**——损失函数（CIoU/SIoU/MPDIoU这类）、是否要给Hiera做轻度解冻微调（不是完全冻结）而不是纯adapter，这些可以放在最后做补充消融，属于"锦上添花"而非主干实验。

实验设计上有几个原则要注意：

一是**控制变量法**，每次只变一个维度，固定住其他所有设置（同样的数据划分、同样的训练轮数、同样的学习率），不然结果没法归因。二是**别信单次跑分**，之前给你们看的那篇Benchmarking论文（J. Intelligent Manufacturing）专门强调过，很多论文报的"精度提升"其实经不起统计检验——你们最好也采用类似的四折交叉验证，每个配置跑3-5次取均值±方差，有条件的话做个ANOVA或t检验，这样论文的说服力会强很多，也是审稿人越来越看重的点。三是**顺序上从粗到细**，先定骨干规模，再定adapter组合，再定Neck，最后调细节，不要一开始就想着把所有维度同时网格搜索，组合数会爆炸，你们的算力和时间大概率扛不住。

评估指标除了mAP@0.5和mAP@0.5:0.95，一定要同时报参数量、FLOPs、FPS这三个效率指标——因为你们这个方案最大的争议点就是"SAM2会不会太重"，如果精度提升了但效率数据不写清楚，审稿人第一反应就是质疑实用性。

如果你们现在算力/时间有限，我建议第一步就先用Hiera-Tiny把"adapter开关组合"这个核心消融做完（这是你们最独特的贡献），拿到初步结果之后再决定要不要往Small/Base+扩展去验证结论的可扩展性——这样即使后期资源不够，至少核心创新点的实验是完整的。

需要我帮你们把这套实验设计整理成一份更详细的实验方案文档（包括每组实验具体要跑哪些配置、大概需要多少组、怎么组织成论文里的消融表格）吗？