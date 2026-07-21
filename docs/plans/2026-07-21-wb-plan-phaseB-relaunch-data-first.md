# wb → cc:Phase B 回炉计划(data-first, 序列消融)

> 承接 cc brief `3d6ebef`(docs/handoff/2026-07-21-cc-direction-phaseB-relaunch-data-first.md, 已采纳)。
> 255 bug 已修(analyze_signal_presence.py 改布尔占比), 证据重出并核对: **<1% 无信号 = 39.3%**(median 1.6%, 85% crop 信号少数), 与 brief §1 真值一致。
> **本计划待 cc review 放行后再动手; 绝不擅自重挖数据集。先计划后训, gate 四关不过绝不接线。**

## 0. 证据基线(本计划的前提)

**数据质量(已三方互证, 采纳 39.3%):**
- walk/stand crop「信号色像素 <1%」比例 = **39.3%**; median 信号占比仅 **1.6%**; **85%** crop 信号是少数(<50%)。
- 分视频方差极大: 07=74% / 04=61% / 02=55% / 03=49% / 01=46% / 06=23% / 09=14% / 10=29% / **05=0% / 11=0%**。
  → 信号是否被框住高度视频依赖 → 对策是「提纯 + ROI 重定心」, 非「全量标错要重挖」。
- 诚实边界: 低信号占比部分是「小而远的信号在大框里」的固有现象, 非全是错框 → 不重挖全量, 只提纯+重定心。

**模型现状(v2, 已诊断):**
- gate 四关全崩: 关1 拒负例0.647 / 关2 07val 0.238 / 关3 04探针0.0 / 关4 val泛化失败; val_acc=0.397(训练0.682→val 0.397 的 gap 暗示过拟合/正则不足)。
- 消融(去 impostor_outside): 07 0.238→0.537(**+29.9pp, 超 5pp 阈值**)→ outside 毒化坐实; 但全去会掉负例拒识(-6pp)。

## 1. 回炉四步(按已证强度排序, 序列执行)

### Step 1 — 降权 impostor_outside 0.5(最强证据, 先做)
- **改动**: 复用已有 `downsample_source(rows, "impostor_outside", 0.5)`(train_classifier_retrain.py)。全删会掉负例拒识, 故**起步 0.5**(半量), 平衡毒化与负例拒识。
- **训练**: `train_classifier_retrain.py --downweight-source impostor_outside 0.5 --out models/ped_signal_v3.pt`(独立 v3, **不覆盖 v2**)。
- **gate**: diag_classifier_retrain.py `--model models/ped_signal_v3.pt --out-json data/output/diag_v3.json`。指标: 关1 拒负例应 ≥0.75(全删掉的 -6pp 退化须被 0.5 挡住), 07val 应明显高于 0.238。
- **交付**: v3 权重 + diag_v3.json + 与 v2 的 gate 对比。

### Step 2 — crop 信号提纯 + ROI 重定心(治 39.3% 无信号 / 85% 信号少数)
- **改动(mine_classifier_retrain.py, 需改脚本)**:
  1. **信号存在门**: 真信号 crop(walk=绿/stand=红)裁出后, 用生产同款 HSV 算信号占比; **< 0.5% 视为缺信号** → 先尝试 ROI 重定心(见下), 重定心后仍 <0.5% 则丢弃该正样本(不作 walk/stand, 也不硬塞 off)。阈值 0.5% 为起步值, cc 可调。
  2. **ROI 重定心**: 现用固定 `prior_roi` 中心框, 常偏/过大。改为: 取 `det.observe(frame)` 中**与目标色一致、且离 prior 最近**的候选 blob 中心作裁图中心, 并**收紧 roi_px**(如 120→96, 待数据定), 提升信号占比。
  3. 提纯/重定心逻辑抽成纯函数(`_crop_signal_ratio`, `_recentered_box`), 先写 TDD 再接 mining。
- **重挖**: 计划批准后**才**重跑 mining → `datasets/classifier_retrain_v2/`(新目录, 不覆盖现有)。
- **重抽检(关键盲区)**: 现有 892 verified 验的是「标签匹配帧级 GT」, **没验 crop 内是否真有信号**。重挖后**必须重跑画廊抽检**(make_classifier_retrain_gallery.py), Jacob 重点审 06/07 暗绿 + 低信号 crop。
- **gate**: diag_v4.json。新增成功指标: **提纯后 median 信号占比应从 1.6% 明显上升(目标 ≥10%)**, 证明 ROI 重定心有效。
- **交付**: 新数据集 + v4 权重 + diag_v4.json + median 信号占比对比 + 画廊抽检结论。

### Step 3 — 07 类暗绿(方法边界预警, 实验性)
- median 信号 1.6% + 48×48 下采样 → 微弱绿几乎不存活。
- **先试**: ①更大输入分辨率(48→64/96, 改 `_build_net` 输入尺寸 + 训练 preprocess); ②对比/CLAHE 归一化拉亮暗绿。在 v4 数据上单独训一版 v5 看 07 gate。
- **方法边界**: 若加大分辨率+归一化仍捕不住 07 → **承认 prior-ROI 裁图对 07 暗绿是方法边界**(早该知道), 据此决定 07 是否靠判别器(或降级为该视频不产绿/走其他路径), 写入报告交 cc 裁定。

### Step 4 — 容量最后评(正则 + 分辨率, 非通道数)
- 1+2 提纯 + 正则(dropout / weight decay, 治 train0.682→val0.397 的 gap)后重训 v6。
- **仅当 train_acc 仍卡低位才加容量**; 届时**优先输入分辨率**(微弱信号是被 48×48 抹掉的)而非通道数。
- gate 四关须全过才谈 Phase C。

## 2. 序列消融协议(一次一变量, 不捆版)
| 版本 | 仅含变动 | gate 关注 |
|---|---|---|
| v3 | Step1 降权 outside 0.5 | 关1 负例拒识不退化 + 07val 升 |
| v4 | v3 + Step2 提纯/ROI重定心 + 重挖 | median 信号占比↑ + 四关 |
| v5 | v4 + Step3 分辨率/对比(07) | 07val 是否破边界 |
| v6 | v5 + Step4 正则/容量 | 四关全过 + val_acc gap 收 |

**每个版本**: 独立 `.pt`(不覆盖) → diag JSON → 记录四关数字 + median 信号占比 → **cc review 通过才进下一步**。绝不允许「降 outside + 加容量」捆一版训(混淆归因)。

## 3. Gate 判定标准(四关 + 提纯指标)
- 关1 拒负例(01/10) ≥ 0.75
- 关2 06train·07val 暗绿收率 ≥ 0.90(07 val 诚实, 不当训练证据)
- 关3 04 探针窗口 walk ≥ 0.5
- 关4 val 泛化(01/07/11)
- **提纯指标**: median 信号占比较 1.6% 明显上升(目标 ≥10%)
- 任一关不过 → 不接线, 回计划调整。

## 4. 红线(不变)
- 先计划 cc review 再动手; **绝不擅自重挖数据集**(Step2 重挖只在放行后)。
- 一次一变量, 不捆版训; 版本各自独立 `.pt`, 不覆盖 ped_signal.pt 基线 / 不覆盖 v2/v3。
- gate 四关不过**绝不接线**; Phase C 生产模型全 11 重训; 不碰 `enforce_transition_limit`。
- 892 verified 盲区 → Step2 重挖后必须重跑画廊抽检。
- scoped git、署名、trunk main、TDD 先(挖矿信号门/ROI重定心纯函数 + 255 回归测已加)。

## 5. 里程碑 / 交付物
1. **本计划** → cc review 放行。
2. Step1: v3 + diag_v3.json(纯 CLI, 无脚本改动)。
3. Step2: 改 mine_classifier_retrain.py(纯函数+TDD) → 重挖数据集_v2 → 画廊重抽检 → v4 + diag_v4.json + median 信号占比。
4. Step3: v5(07 分辨率/对比实验) + 方法边界结论。
5. Step4: v6(正则/容量) + 最终四关报告。
6. 全过 → 出 Phase C 接线圈(另议)。

**下一步**: 等 cc 放行。放行后从 Step1(v3) 起, 前台分批跑(每批≤3视频防 SIGKILL), 跑完主动回报。
