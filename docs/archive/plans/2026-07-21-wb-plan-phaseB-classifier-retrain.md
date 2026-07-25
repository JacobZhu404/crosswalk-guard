# wb → cc 执行计划:Phase B 灯态判别器重训 + 再诊断(gate)

> 承接 Phase A(commit `39e6863`) + cc 方向 brief(`docs/handoff/2026-07-21-cc-direction-phaseB-classifier-retrain.md`, `a6d860c`)。
> 本计划只到「产出 v2 模型 + 重跑 gate 诊断 + 出再诊断报告给 cc 核」为止。**先计划后训,gate 四关不过绝不接线。**

> **cc review 状态(2026-07-21)**:批准执行,1 硬伤修正 + 4 澄清(见 §2/§3/§4.1)。
> - cc 独立核过:三处硬伤属实、gate 测的分布对(scan_video:83-94 走 observe() 从 signal_prior 的 prior ROI 裁图,与生产门控插入点/训练分布一致)、复用函数全在、架构描述一致、diag 硬指旧数据集/模型。
> - 🔴 必修(已改):源降权机制——`CrossEntropyLoss(weight=...)` 是类级权重,压不到 off 类里的 `impostor_outside` 子源,且 `_rows_to_dataset` 不保留 source → 改「进 dataset 前**按行下采样**该源」,不动 loss。
> - 🟡 4 澄清(已补 §4.1):gate 规则写死 off→否决且判别器不覆盖颜色(Phase C 不变式)、06(train)弱证据须与 07(val)分列、Gate A 加 walk→off 误判率、消融触发阈值先定。

## 0. 目标与成功标准(锚定)
- 在 Phase A 数据集(`datasets/classifier_retrain/labels.csv`, 5740 行, 已排除 5 delete)上重训 `SignalStateClassifier`,产出**新模型** `models/ped_signal_v2.pt`(**不覆盖** `ped_signal.pt` 基线)。
- **成功 = 重训后重跑可行性诊断同口径四关全过**,不是训练集 acc 高(旧 pt 就是 in-domain 80% 却端到端屠真绿)。
- **唯一生产产物在 Phase C**:Phase B 的 train/val 模型只测泛化、验证可行性,**不接线**。

## 1. 复用策略(不重写,低回归风险)
现有脚本函数已可复用,采用「**import 复用 + 薄封装 + 加参数**」,不动旧数据集工作流:

| 现有模块 | 复用方式 |
|---|---|
| `signal_state_classifier._build_net` / `LABELS` / `_INPUT` | 直接 import,网络结构单一真相源 |
| `train_ped_signal.train_net` / `export_torch` / `_rows_to_dataset` | import 复用训练/导出逻辑 |
| `diag_classifier_feasibility.domain_confusion` / `scan_video` | import 复用端到端扫描逻辑 |

**新增两个薄脚本( scoped,不动旧脚本 )**:
- `scripts/train_classifier_retrain.py` — 包装上述训练函数 + manifest 固定 split 评测 + 源降权。
- `scripts/diag_classifier_retrain.py` — 包装上述诊断函数 + 重指向新数据集/新模型 + train/val 分组 + 04 探针 + impostor_outside 消融。

> 不修改 `train_ped_signal.py` / `diag_classifier_feasibility.py` 的默认行为,旧 `datasets/ped_signal` 工作流零回归。

## 2. TDD(先写测,cc 硬要求)
新增 `tests/test_train_classifier_retrain.py` + `tests/test_diag_classifier_retrain.py`:
- **split 无泄漏**:从 `manifest.json` 读 train/val → 两集合不交、并集=全 11 视频、与 labels.csv 实际视频一致。
- **排除 delete**:`label=="delete"` 行不进训练/评测(防 apply 删图残留污染)。
- **训练契约(smoke)**:合成数据 train → 导出 v2 → `SignalStateClassifier(v2).classify` 三类多数投票正确(复用 `train_ped_signal.run_smoke` 思路,但导出到 v2 路径)。
- **均衡/源下采样单元**:`--balanced` 过采样后三类样本数相等;`--downweight-source impostor_outside 0.5` 在 rows 进 `_rows_to_dataset` **之前按行下采样该源**(保留 keep_ratio 比例, seeded 可复现)——复用现有「先采样行再建 dataset」流程,**不动 loss / 不用 `CrossEntropyLoss(weight=...)`**(类级权重压不到 off 子源,且 `_rows_to_dataset` 不保留 source)。TDD 验证:下采样后该源行数 ≈ 原半数(±1)。
- **gate 判定单元**:四关阈值函数(`neg_off_ratio≥0.75` / `真绿收率≥0.90` / `04 窗口 walk≥0.5` / `val 泛化成立`)可独立断言。
- **04 探针窗口解析**:`--probe-window 违章04 42.0 43.2` 正确裁剪帧区间。

## 3. 训练配置(v2)
- **数据**:全量 `datasets/classifier_retrain/labels.csv`(5740,排除 5 delete)。**不**用纯 `--verified-only`(cc 禁);Jacob 修正已写回,尊重标签。
- **split**(manifest 固化):train={02,03,04,05,06,08,09,10}(8) / val={01,07,11}(3)。**严禁按帧切/LOVO**。
- **架构**:先用现 `_build_net`(48×48, 36KB)。**若 gate 卡"暗绿 vs impostor 分不开"→ 加大容量**(加宽/加深,改 `_build_net` 单一真相源,重训 v2)。
- **超参**:epochs=30(可 50 看饱和), lr=1e-3, Adam。`--balanced` 开(walk/stand 过采样到 off 量级;当前 off 2507/walk 1976/stand 1257 已较均衡,仍过采样防 walk 被压成少数类)。
- **源下采样(观察项1)**:`impostor_outside`=1395(最大最启发式负源,占 impostor 65%)。主训用全量;**消融版** 在进 dataset 前按行下采样该源(`--downweight-source impostor_outside 0.5` 或 `--drop-source impostor_outside`)重训一版,对比 06/07 真绿拒真率。**注意:不能用 `CrossEntropyLoss(weight=...)` 类级权重——impostor_outside 是 off 类的子源,类级权重会连 prior_off/真 impostor 一起压,且 `_rows_to_dataset` 不保留 source 无法 per-sample 加权。**
- **输出**:`models/ped_signal_v2.pt`(原子替换:先写临时文件再 `os.replace`,沿用 apply 事故教训)。

## 4. Gate:重跑可行性诊断(四关全过才算过)
`diag_classifier_retrain.py` 复用扫描逻辑,模型指向 v2,数据集指向 classifier_retrain:
- **A) 域混淆矩阵**(新数据集 crops):per-class acc,**重点 off 类在 impostor/impostor_outside 行的拒识率** = gate1「拒背心/环境绿」证据。
- **B) 端到端产绿区判别**(observe() 复现):按 train/val 分组报每视频 off/walk/stand。
  - **关1 拒负例**:01/10 负例拒识率 `neg_off_ratio ≥ 0.75`(旧 25% → 目标 >>75%)。
  - **关2 保暗绿**:06(train)/**07(val)** 真绿收率(walk+stand 占比)≥ 0.90,即拒真率 ≤ 0.10;**07 在 val 必须诚实泛化,不许 fudge**。
  - **关3 过 04 试金石**:`--probe-window 违章04 42.0 43.2` 短暗绿窗口内 walk 识别帧占比 ≥ 0.5(旧 8/14 拒 → 目标认出多数)。
  - **关4 val 泛化**:val 三视频(01/07/11)负例拒识率≥0.75 且正例(07)收率≥0.90。
- **观察项1 消融**:impostor_outside 减半/去除重训版,对比 06/07 真绿拒真率;若去除 outside 后 06/07 收率回升 → 证明确实是 outside 灌太多导致过拒,按 cc 指示降权/收窄重训。

### 4.1 cc review 补强(已批准,执行须落实)
- **Gate 规则写死 + Phase C 不变式**:生产门控 = 二元 `off→否决 / {walk,stand}→放行`。§4 关2「walk+stand 收率≥0.90」据此成立。**Phase C 不变式(现在立,防接错)**:判别器只做否决(`off→unknown`),**绝不覆盖 `fuse_light` 的绿/红颜色**——否则真绿帧被判 `stand` 会把颜色翻红。计划/代码注释须明示。
- **06(train)弱证据须与 07(val)分列**:06 在 train,收率近乎必过,**不得当泛化证据**;真正的暗绿泛化关是 07(val)。再诊断报告必须 06/07 分列,不许 06 掩 07。
- **Gate A 加 walk→off 误判率**:域混淆矩阵除报 off 拒 impostor 外,**必须同时报 walk→off 误判率**(in-domain 过拒真绿哨兵);只测拒背心 = 没测保真绿。
- **消融触发阈值先定**:「outside 灌太多」判据量化——去 outside 重训后若 07(val)收率回升 ≥ N 个百分点(初值 **N=5**)→ 才降权/收窄重训;结论不许主观。

## 5. 红线(不可违反,摘自 handoff §4)
- 不覆盖 `ped_signal.pt`(存 `ped_signal_v2.pt`);Phase C 生产模型须全 11 重训(val 折回)。
- 禁用纯 `--verified-only`;不碰 `enforce_transition_limit`;净回退不上(8 好视频不回退/07 暗绿不杀/负例 10 保持 0/01 转 0 fp)。
- B 不过 = 绝不硬接(回炉数据/模型 or 转位置/时序稳定门控)。
- 写回类脚本先写临时文件再原子替换(apply 事故教训)。
- scoped git(**绝不 `git add -A`**)、署名 `Co-Authored-By: Claude Opus 4.8`、trunk main、TDD 先。

## 6. 执行里程碑(顺序,等 cc 放行才进下一步)
1. **本计划 cc review 放行**(当前步)。
2. 写 TDD(`test_train_classifier_retrain.py` + `test_diag_classifier_retrain.py`)→ 跑通空跑/单测。
3. 写两个薄脚本(`train_classifier_retrain.py` / `diag_classifier_retrain.py`,import 复用)。
4. TDD 全绿后,前台训 v2(5740 图 tiny-CNN,秒级~分钟级)→ 存 `ped_signal_v2.pt`。
5. 重跑 gate 诊断 → 出 `docs/reports/2026-07-21-wb-phaseB-diagnosis.md`(四关证据 + outside 消融 + val 泛化 + §4.1 四澄清逐项答:gate 规则/off→否决不覆盖颜色、06/07 分列、Gate A walk→off 率、消融触发 N=5pp)。
6. **可行性三选一交 cc**:可行→Phase C 接线计划 / 不可行→回炉(加大容量 or 降 outside)/ 转回退门控。**cc 核后才谈 Phase C。**

## 7. 交付物
- `scripts/train_classifier_retrain.py`(新,薄封装)
- `scripts/diag_classifier_retrain.py`(新,薄封装)
- `tests/test_train_classifier_retrain.py` + `tests/test_diag_classifier_retrain.py`
- `models/ped_signal_v2.pt`(训练产物,非接线)
- `docs/reports/2026-07-21-wb-phaseB-diagnosis.md`(再诊断报告,gate 结论)
