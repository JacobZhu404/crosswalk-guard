# wb → cc 再诊断报告:Phase B 重训 gate(四关未过,进入回炉)

> 承接 `docs/plans/2026-07-21-wb-plan-phaseB-classifier-retrain.md`(cc 已批准,改 1 硬伤 + 4 澄清)。
> 主模型 `models/ped_signal_v2.pt` 已训(全量 5740, manifest split)。本报告指出:**四关未过,B 绝不硬接,进入回炉。**

## 0. TL;DR
- 主模型 v2 **四关全不过**(关1/关2-07/关3/关4 均 FAIL)。核心失败:**07(val) 真绿收率仅 0.238**(未见视频几乎全拒真绿)。
- 消融(去 `impostor_outside`)→ 07 收率 **0.238 → 0.537(+29.9pp,远超 §4.1 阈值 5pp)**,**坐实 cc 观察项1:outside 灌太多在毒化模型**。
- 但去 outside 后 07 仍 < 0.90,且类域混淆仍稀烂(walk 0.563/stand 0.413/off 0.625)→ **问题两层:①outside 主因(已证)②tiny-CNN 36KB 容量不足(cc 已预警)**。
- **结论:Phase B 主模型 gate 失败,按纪律 B 不过绝不硬接。回炉 = 去 outside + 加大容量。v2 不接线,等回炉版过 gate 再谈 Phase C。**

## 1. 主模型 v2 Gate 结果(全量,manifest split)
- train=4314 / val=1426; train_acc=0.682, **val_acc=0.397**(容量不足信号)。

| 关 | 目标 | 实测 | 结果 |
|---|---|---|---|
| 关1 拒负例(01/10) | ≥0.75 | **0.647** | ❌ |
| 关2 保暗绿 06(train) | ≥0.90 | 0.915 | ✅(train 弱证据,不当泛化证据) |
| 关2 保暗绿 **07(val)** | ≥0.90 | **0.238** | ❌ **致命** |
| 关3 过 04 试金石(42–43.2s) | walk≥0.5 | **0.0**(窗口 5 帧全 other,0 walk/0 off) | ❌ |
| 关4 val 泛化(01/07/11) | 拒≥0.75∧收≥0.90 | 01=0.63 / 07=0.24 / 11=0.90 | ❌ |

- **域混淆矩阵**(5740 crops): walk acc=0.618 / stand=0.535 / off=0.643。
- **Gate A 补 walk→off 误判率 = 0.245**(§4.1 要求加的哨兵):in-domain 真绿被过拒近 1/4,亮红灯。

## 2. 观察项1 消融(去 impostor_outside)

| 指标 | 主模型(全量) | 去 outside | 变化 |
|---|---|---|---|
| **07(val) 收率** | 0.238 | **0.537** | **+29.9pp** 🔺(超阈值 5pp) |
| 06(train) 收率 | 0.915 | 0.882 | -3.3pp |
| 关1 拒负例(01/10) | 0.647 | 0.588 | -5.9pp |
| 域混淆 walk→off | 0.245 | 0.276 | 略升 |

- **判定**:去 outside 后 07 收率回升 29.9pp ≥ §4.1 阈值 5pp → **触发降权/去 outside 结论成立**。outside 是 07 过拒的主因。
- 但去 outside 后 07 仍 0.537 < 0.90,且关1 拒负例反而降到 0.588 → **容量问题独立存在**,单去 outside 不够。

## 3. §4.1 四条澄清逐项答
1. **Gate 规则 + Phase C 不变式**:生产门控 = 二元 `off→否决 / {walk,stand}→放行`。代码注释已明示(`diag_classifier_retrain.py` gate 函数 + 计划 §4.1)。**Phase C 不变式立:判别器只做否决(off→unknown),绝不覆盖 `fuse_light` 绿/红颜色**——本报告所有 gate 用 `off` 占比判拒、`walk+stand` 占比判收,与此一致。
2. **06(train) 弱证据与 07(val) 分列**:报告 §1 表已分列。06=0.915 是 in-domain 近乎必过,**不计入泛化证据**;真正暗绿泛化关 = 07(val)=0.238,如实报、未用 06 掩 07。
3. **Gate A 加 walk→off 误判率**:已报 = 0.245(主模型)/ 0.276(去 outside)。这正是 in-domain 过拒真绿哨兵,单测 off 拒识率会漏掉这一项。
4. **消融触发阈值先定**:§4.1 定 N=5pp。实测去 outside 后 07 回升 29.9pp ≥ 5pp → 触发降权。结论非主观。

## 4. 交付物(本报告的物理产物)
- `models/ped_signal_v2.pt`(主模型,gate 失败,**不接线**)
- `models/ped_signal_v2_dropoutside.pt`(消融版,供对比)
- `data/output/diag_classifier_retrain_main.json`(主模型 gate 原始)
- `data/output/diag_classifier_retrain.json`(去 outside 版 gate + 消融对比)
- `scripts/train_classifier_retrain.py` / `scripts/diag_classifier_retrain.py`(薄封装,import 复用,零回归)
- `tests/test_train_classifier_retrain.py` / `tests/test_diag_classifier_retrain.py`(TDD 15 项全绿)

## 5. 可行性三选一(交 cc)
- **不可行(当前 v2)**:四关全不过,不接线。
- **回炉方案(推荐)**:① 训练去 `impostor_outside`(或 `--downweight-source impostor_outside 0.5` 看是否够)→ 已证 +29.9pp;② **加大 `_build_net` 容量**(加宽/加深,改单一真相源,如 3→16→32 通道 + 第三 maxpool,或加一层),解决 val_acc 0.397 的容量瓶颈。两者叠加重训一版再过 gate。
- **转位置/时序稳定门控(备选)**:若加大容量后仍分不开暗绿 vs impostor,放弃纯颜色判别,转 observe() 的位置/时序稳定门控(非颜色)——但这是另一条线,优先级低于先试回炉①②。

**cc 核后定回炉路径;回炉版过四关才谈 Phase C 接线。B 不过绝不硬接不变。**
