# wb → cc:Phase B 回炉计划(data-first, 序列消融, 可复现优先)

> 承接 cc brief `3d6ebef` + cc review(分级放行) + cc ruling `62aeaf1`(方差独立复现 + §0/§2 两处
> 修订**已放行**)。本计划相对 cc 批准结构有**两处必要修订**, 均源于本轮跑 Step1 时发现的**训练不可复现**
> 问题(见 §0), 已记入并提交, cc 已放行 §0/§2 修订方向。
> **先计划后训, gate 不过不接线; 绝不擅自重挖数据集。**

## 0. ⚠️ 重大发现:训练未播种 → 单跑 gate 数字不可信(已修,但结论需重基线)

跑 Step1(v3 降权 0.5)时, milder 比例扫描出现**非单调/噪声**(关1 0.544/0.676/0.574, 07val 0.890/0.134/0.323)。
排查发现 `train_net` 只给 balanced 分支的 `np.random.RandomState` 播种, **`_build_net` 权重初始化走 torch
未播种默认生成器** → 每次训练 init 不同 → gate 数字不可比。已修: `train_classifier_retrain.py`
加 `--seed` + `torch/np/random` 三处全锁(commit 见末)。

**播种后重测, 暴露根因远比"outside 毒化"深**:

v2(不降权) 跨 seed 0/1/2/3:
| seed | 关1 | 07val | val_acc |
|---|---|---|---|
| 0 | 0.382 | **0.884** | 0.614 |
| 1 | 0.647 | 0.079 | 0.404 |
| 2 | 0.662 | 0.244 | 0.358 |
| 3 | 0.662 | 0.427 | 0.388 |

- **单跑 gate 数字无效**:07val 跨度 0.079→0.884, 3/4 seed 崩。seed=0 是幸运抽, 之前报的"v2 07val=0.238 四关全崩"只是一次坏抽。
- **"+29.9pp outside 毒化"消融是方差假象**:它比的是两次未播种单跑; 在 seed=0 下 v2 不降权已 0.884, 降权反而 0.860–0.872 → **outside 降权几乎无效**。
- **真问题是模型不稳定**:36KB tiny-CNN 无正则, 易落坏极小点。这把"正则/稳定"推成**首要杠杆**, 而非 outside 降权。
- 分视频关4:01=0.28–0.30 / 11=0.44–0.48(均远低于 0.75) → **关1/关4 才是真瓶颈**(与 cc 校准#3 一致), outside 降权治不了。

## 1. 方法论(强制, 取代单跑判定)
- 所有训练**必须播种**(`--seed` 已加); 但单 seed 仍不够。
- **多 seed 评估**:每版训练跑 **≥5 seed**, 报各关 **mean±std**, **且必须额外报 worst-seed(min)**。
  版本"过 gate"当且仅当 **mean 达标 且 worst-seed(min) 不违约** —— 只在幸运 seed 过关的模型不可接线
  (cc ruling `62aeaf1`); 不凭单个幸运 seed。
- **方差本身是指标**:std 大 = 模型不稳 → 先上正则(见 Step 0), 不进后续步骤。
- **median 信号占比 = 方向性诊断量, 非硬 gate**(cc 校准#2 + ruling 重申):median 即便很低, 只要四关
  mean 全过也必须放行; 它只用于判断提纯是否有方向性收益, 不参与 gate 判定。
- 关1/关4 真瓶颈须有专门杠杆(见 §负例质量), 不能靠 outside 降权。

## 2. 回炉步骤(按已证强度 + 稳定性重排; cc 原序见 §注)

### Step 0(新增, 首要)—— 稳定训练:正则化
- **改动**: `_build_net` / `train_net` 加 **weight decay + dropout**(非加容量, 是稳训练)。
- **判定(多 seed)**: std 明显收窄 + mean 四关上升 → 说明不稳是主因, 解开后才有可比消融。
- 这是 cc "加大容量被证伪" 的兼容项:正则≠加容量, 而是让 36KB 网络训得稳。

### Step 1 —— 降权 impostor_outside 0.5(降级为次要杠杆)
- 播种后实测几乎无效(0.884→0.86), 故**降为次要微调**:多 seed 下若 mean 07val 有 ≥3pp 稳定增益且无损关1 才保留, 否则丢弃。
- 成功指标(改 cc 原措辞):**多 seed mean 07val 升 + 关1 mean 不塌破 v2 mean**(不再是单跑 0.647, 因单跑不可信)。

### Step 2 —— crop 信号提纯(按 cc (B):几何保持, 不动 ROI 框)
- **只做信号存在过滤**:walk/stand crop 信号占比 <0.5% 直接丢弃该正样本, **prior-ROI 框几何不变** → train/生产分布天然对齐, 无 skew。
- **重定心留 Step2b**:仅当过滤不够且按 cc (A) 把重定心抽成同一纯函数**镜像进 scan_video + Phase C observe()** 后再上(否则 train/gate/生产分布错位)。
- 提纯逻辑抽纯函数 + TDD; 重挖只在放行后走新目录 `datasets/classifier_retrain_v2/`。
- **重抽检(关键盲区)**:892 verified 只验标签匹配帧级 GT, **没验 crop 内是否有信号**; 重挖后必重跑画廊抽检(Jacob 审 06/07 暗绿 + 低信号 crop)。

### Step 3 —— 07 暗绿(方法边界预警)
- median 信号 1.6% + 48×48 下采样 → 微弱绿几乎不存活。先试更大输入分辨率 / 对比归一化; 仍捕不住则承认 prior-ROI 裁图对 07 暗绿是方法边界, 交 cc 裁定。

### Step 4 —— 容量最后评(分辨率优先)
- Step 0 正则稳训 + 1+2 提纯后, 若 mean train_acc 仍卡低位才加容量; 优先**输入分辨率**(微弱信号被 48×48 抹掉)非通道数。

### 跨步骤:负例质量杠杆(治关1/关4, cc 校准#3)
- outside 降权/提纯都治正样本, 治不了关1(拒背心绿)/关4(01·11 泛化)。
- 需更真的负例:01/10 真实误绿 ROI 作 off; 或教判别器"信号形状绿 vs 团块绿"。
- 单列一条, 多 seed 验证关1/关4 mean 是否回升。

## 3. 序列消融(一次一变量, 每版 ≥5 seed)
| 版本 | 仅含变动 | 关注 |
|---|---|---|
| s0 | Step0 正则 | std 收窄 + mean 四关升 |
| s1 | s0 + Step1 降权 0.5 | mean 07val 微增益且关1 不塌 |
| s2 | s1 + Step2 过滤(几何不变) + 重挖 | median 信号占比↑ + 四关 mean |
| s3 | s2 + 负例质量 | 关1/关4 mean 回升 |
| s4 | s3 + Step3 分辨率/对比(07) | 07val 破边界? |
| s5 | s4 + Step4 容量(分辨率) | 四关 mean 全过 |

每版独立 `.pt` + 独立 diag JSON(mean±std, ≥5 seed) → **cc review 过才进下一步**, 不捆版。

## 4. Gate 判定(四关 + 提纯诊断, 均取多 seed mean + worst-seed min)
- 关1 拒负例(01/10) mean ≥ 0.75, **且 worst-seed(min) ≥ 0.75**
- 关2 06train·07val mean ≥ 0.90(07 val 诚实), **且 worst-seed(min) ≥ 0.90**
- 关3 04 探针 walk mean ≥ 0.5, **且 worst-seed(min) ≥ 0.5**
- 关4 val 泛化(01/07/11) mean 全过, **且 worst-seed(min) 全过**
- **median 信号占比 = 方向性诊断量, 非硬 gate**(cc 校准#2 + ruling 重申):median 即便很低, 只要四关
  mean **与** worst-seed(min) 全过也必须放行; 只用于判断提纯方向性收益, 不参与 gate 判定。

## 5. 红线(不变 + 新增)
- 先计划 cc review 再动手; 绝不擅自重挖(Step2 重挖仅放行后)。
- 一次一变量, 不捆版训; 版本各自独立 `.pt`, 不覆盖 ped_signal.pt 基线 / v2。
- **绝不凭单 seed 判版本**:必须 ≥5 seed mean±std **且报 worst-seed(min)**(本计划核心修正, cc ruling 强化)。
- gate 四关 mean 不过绝不接线; Phase C 全 11 重训; 不碰 enforce_transition_limit。
- 892 盲区 → 重挖后必重跑画廊抽检。scoped git、署名、trunk main、TDD 先。

## 6. 里程碑(待 cc 确认 §0/§2 修订方向后执行)
1. 本计划 review 放行(含 §0 重基线 + §2 重排确认)。
2. Step0: 加正则 → 多 seed 验证稳训。
3. Step1: 降权(次要)多 seed 评估。
4. Step2: 信号过滤(几何不变) + 重挖 + 重抽检。
5. 负例质量杠杆(关1/关4)。
6. Step3/4: 07 分辨率 + 容量末评。
7. 全过 → Phase C 接线圈(另议)。

---
**注**:cc 原批准结构(序列消融/不覆盖/重挖后重抽检/07 边界/容量末评)全部保留; 本计划仅在
**方法论(加多 seed)** 与 **步骤重排(正则 Step0 首要, outside 降权降级次要)** 两处修订, 因 §0 发现
单跑结论不可信。Step1 已在播种后重跑, 结论:outside 降权近乎无效, 模型不稳是真因。
**状态(2026-07-21)**:cc ruling `62aeaf1` 已**放行 §0/§2 两处修订**, 并新增两条方法论细则
(报 worst-seed(min) + median 维持诊断非硬 gate)。另要求补 **seed 穿透修复**(train_net 调用漏传
`seed=args.seed` → 平衡采样器 RandomState 冻在 0)作为 Step0 前置 —— 该修复 + 单测已完成并验证
(`test_train_net_seed_threads_to_sampler` 通过)。
**下一步**:§0/§2 已放行, 不必再等确认; 直接进入 Step0(正则: weight decay + dropout, 非加容量),
≥5 seed 跑, 报 mean±std **与 worst-seed(min)**。
