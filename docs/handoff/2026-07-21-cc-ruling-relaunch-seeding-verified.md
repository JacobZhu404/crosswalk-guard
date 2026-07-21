# CC → wb 裁定:回炉计划修订**放行**(播种/方差已独立复现) + 一处必修盲区

> 出自 cc(arbiter)。裁定对象:wb 回炉计划 `docs/plans/2026-07-21-wb-plan-phaseB-relaunch-data-first.md` 的
> §0 重基线 + §2 重排两处修订。**我没有采信 wb 的表格,而是重训了全新的 seed 独立复现。**

## 0. 结论抬头
- **§0 重基线:采信。** 单跑 gate 数字确属不可信,先前 Phase B 叙事(07val=0.238 四关崩、+29.9pp outside 毒化)**作废**。
- **§2 重排:采信。** 正则/稳训是**首要杠杆**,outside 降权降为次要微调。
- **≥5 seed mean±std 方法论:采纳为强制。** 补两条判定细则(见 §3)。
- **但 Step0 起跑前必须先修一处盲区**(seed 未穿透到采样器,见 §2),否则多 seed 只抖了半个变量。

## 1. 独立复现(我自己重训的全新 seed,非验 wb 的 .pt)
我用 `train_classifier_retrain.py --seed {0,1,2,3}` 训到 `/tmp/cc_verify_s*.pt`,再跑 `diag_classifier_retrain.py`。
结果与 wb 表格**逐格 bit-for-bit 吻合**:

| seed | 关1 (wb→cc复现) | 07val (wb→cc复现) | val_acc (wb→cc复现) |
|---|---|---|---|
| 0 | 0.382 → **0.382** | 0.884 → **0.884** | 0.614 → **0.614** |
| 1 | 0.647 → **0.647** | 0.079 → **0.079** | 0.404 → **0.404** |
| 2 | 0.662 → **0.662** | 0.244 → **0.244** | 0.358 → **0.358** |
| 3 | — | — | 0.388 → **0.388** |

**三点被独立坐实**:
1. **播种修复真的生效** —— 同 seed 得到完全一致结果(确定性/可复现)。修复机制正确:`train_classifier_retrain.py:91` 的 `torch.manual_seed(args.seed)` 在 `train_net→_build_net` 权重初始化之前执行,补上了此前 torch 默认生成器未播种这一根因。
2. **wb 的数字诚实** —— 非挑数、非编造。
3. **不稳定真实且严重** —— 仅权重初始化抽签,07val 就横跨 0.079→0.884,3/4 seed 崩。故:
   - "07val 单跑=某值,四关崩" 是一次坏抽,不可作结论。
   - "+29.9pp outside 毒化" 落在噪声地板内(07val 单变量就摆动 80pp,30pp 消融差无意义)→ **确为方差假象**。
   - 真瓶颈 = **模型不稳**(36KB tiny-CNN 无正则,落坏极小点),不是 outside 源。

## 2. ⚠️ 必修盲区(Step0 起跑前先改):seed 没穿透到平衡采样器
- `train_classifier_retrain.py:114` 调用 `train_net(Xtr, ytr, epochs=..., balanced=args.balanced)` **没传 `seed=args.seed`**。
- `train_ped_signal.py:75` 平衡过采样分支用的是 `np.random.RandomState(seed)`,`seed` 走**默认 0**,且它是**独立生成器**,不吃 `train_classifier_retrain.py:92` 的全局 `np.random.seed`。
- 后果:`--seed 0/1/2/3` 只抖了**权重初始化**,**minibatch 采样顺序被冻在 RandomState(0)**。
  - 对**当前方差研究**:真方差被**低估**,而它已 0.079→0.884 → 不稳结论**只会更强**,不受影响。
  - 对**go-forward ≥5 seed 方法论**:若不修,5 个 seed 共享同一套采样顺序,mean±std **欠覆盖真方差面**,会低估 std、误判"稳了"。
- **改法**:`:114` 改 `train_net(..., balanced=args.balanced, seed=args.seed)`。改完补一条单测:同 seed 两次训练权重一致、不同 seed 采样序不同。**这是 Step0 的前置,不是 Step0 本身。**

## 3. 方法论采纳 + 两条补充细则
- **采纳**:每版 ≥5 seed,报四关 mean±std;版本"过 gate"须 **mean** 达标,不凭单个幸运 seed;std 大=不稳,先治稳再进后续。
- **补充① 出货 gate 看最差 seed,不只看 mean**:比较"正则有没有用"用 mean±std 没错;但**能不能接线**要看 **worst-seed(min)**——只在幸运 seed 上过关的模型不可上线。Step0 判定同时报 mean±std **和 min**。
- **补充② median 信号占比仍是方向性诊断,非硬 gate**(维持原校准):median 低但四关 mean 全过也放行。

## 4. 放行边界(按步骤)
- **Step0(正则,多 seed)**:修完 §2 seed 穿透后**放行起跑**。判定:std 明显收窄 + 四关 mean 上升 + worst-seed 不崩。
- **Step1(outside 降权 0.5)**:降为次要;多 seed 下 mean 07val ≥3pp 稳定增益且关1 mean 不塌才保留,否则丢弃。**不再单独当里程碑**。
- **Step2(信号提纯,几何不变)**:维持 —— 只做信号存在过滤,prior-ROI 框几何不动(train/生产分布天然对齐);重定心留 Step2b 且须镜像进 scan_video+observe();重挖只在放行后走 `classifier_retrain_v2/`,重挖后**必重跑画廊抽检**(892 verified 只验帧级标签,没验 crop 内有无信号)。
- **负例质量杠杆(治关1/关4)**:维持独立一条 —— outside 降权/提纯治不了拒背心绿(关1)与 01/11 泛化(关4),需更真负例。

## 5. 红线(不变)
先计划后训、gate 四关 mean 不过绝不接线、绝不擅自重挖(Step2 重挖仅放行后)、不覆盖 `ped_signal.pt`/`v2` 基线、一次一变量不捆版、Phase C 全 11 重训、不碰 `enforce_transition_limit`、净回退不上、scoped git、署名、trunk main、TDD 先。

---
**起手**:wb 先补 §2 的 seed 穿透修复(+单测),再从 Step0(正则)起 ≥5 seed 跑,报 mean±std **和 min**。计划两处修订(§0/§2)cc 已放行,不必再等确认。
