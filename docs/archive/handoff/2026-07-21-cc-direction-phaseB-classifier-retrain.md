# CC → wb 方向:Phase B 灯态判别器重训 + 再诊断(gate)

> 出自 cc。承接 Phase A(`datasets/classifier_retrain/`,已收官并提交 commit 39e6863)。
> Phase A 独立核实通过:split 无泄漏、类平衡(全局 off 37.2%)、自动标签与 GT 全自洽(walk 全落绿帧 / stand 全落红帧 / impostor 全非绿)、Jacob 抽检 892 张 verified + 5 删除已写回。
> **Phase B = 重训 + 重跑可行性诊断当 gate。出再诊断报告给 cc,cc 核后才谈 Phase C 接线。**

## 0. 目标(锚定,别跑偏)
- 在 Phase A 数据集上重训 `SignalStateClassifier`(LABELS=[walk,stand,off]),产出**新模型**(存 `models/ped_signal_v2.pt`,**别覆盖** `ped_signal.pt` 基线)。
- **成功标准 = 重训后重跑 `scripts/diag_classifier_feasibility.py` 同口径能过**,不是训练集 acc 高。上一个 pt 就是 in-domain 80% 却端到端屠真绿。

## 1. 训练纪律
- **按视频 split(已在 manifest 固化)**:train={02,03,04,05,06,08,09,10} / val={01,07,11}。**严禁按帧切/泄漏**——这是旧 pt 过拟合的直接教训。
- **不要用纯 `--verified-only`**(cc 硬要求)。labels.csv 里 Jacob 的修正(892 verified + 5 删除 + 约 372 张信号改判 off)**已写回**,直接**用全量 labels.csv 训练(尊重已改标签、排除已删行)**。纯 verified-only 会丢掉 4848 张已证可靠的 bulk 正样本(prior_roi walk 全绿 / stand 全红,cc 已核)。
- **类平衡**:walk 是历史少数类(旧模型 20% → 偏置屠绿)。挖矿已做 walk 过采样,训练再确认 loss 加权或采样均衡,**别让 walk 再被压成少数类**。
- **容量**:现结构 36KB 容量存疑。若 gate 卡在"暗绿 vs impostor 分不开",考虑加大模型——先训现结构看 gate,再决定。

## 2. Gate:重跑可行性诊断(四关全过才算过)
重训后**重跑 `scripts/diag_classifier_feasibility.py` 同口径**,逐条配证据:
1. **拒背心/环境绿**:01/10 负例假绿显著拒(目标 >>75%,旧模型只 25%);
2. **保 06/07 暗绿**:不能再 0.98–0.99 拒真绿(06 在 train / **07 在 val**——07 若诚实泛化失败要如实报,不许 fudge);
3. **过 04 试金石**:42–43.2s 暗短绿被认出(旧模型 8/14 拒);
4. **未见视频(val 01/07/11)泛化**成立。

## 3. 两个观察项(cc 盯,Phase B 必答)
1. **`impostor_outside` 占 impostor 65%(1398,val/train 合计最大最启发式的负源)**。它是"信号外绿斑",与门控实际输入(prior ROI)可能不同分布。**若重训后模型又过度拒真绿(06/07 gate 崩),首要怀疑就是它灌太多 → 降权/收窄该源重训。** 再诊断报告里请**单列 impostor_outside 对 06/07 真绿拒真率的影响**(可做消融:去掉/减半 outside 再训一版对比)。
2. verified 语义:cc 已判定不 blanket 批量验证(见 §5),训练不依赖 verified flag。

## 4. 红线(不可违反)
- **GT/eval 产物只用于训练数据 curation,绝不进生产推理**;判别器不得引入 per-video 硬编码。
- **Phase C 最终接线用的生产模型必须在全 11 视频上重训**(val 折回训练)——Phase B 的 train/val 模型只为测泛化、验证可行性,不是最终产物。
- 不碰 `enforce_transition_limit`;**净回退不上**(Phase C 端到端:8 好视频不回退、07 暗绿不杀、负例 10 保持 0、01 期望转 0 fp、先验兄弟 02/03 不退步)。
- **B 不过 = 绝不硬接**:回炉数据/模型(加大容量 or 降 outside)或转位置/时序稳定门控(非颜色)。
- scoped git(**绝不 `git add -A`**)、署名 `Co-Authored-By: Claude Opus 4.8`、trunk main;TDD(训练/split/gate 评测逻辑先写测)。
- **回写类脚本先写临时文件再原子替换**——刚踩过 apply 脚本"打开即截断"清空 labels.csv 的坑,别重蹈。

## 5. verified 批量验证:cc 已决(不 blanket 接)
Jacob 问过是否批量验证剩余 4848 张。cc 判定**不做 blanket 批量接受**:
- prior_roi(2976)/prior_off(150)自动标 == GT 派生标,再标 verified 是**循环、零信息**;
- impostor(577)+ impostor_outside(1145)是**唯一可能标偏**的启发式负源,盲标 verified=1 会**污染 flag 语义**(把没人看过的混进"人工确认")并**重演过拒真绿风险**;
- 训练走全量 all-minus-flagged,flag 不 gate 训练 → 批量验证对 Phase B 零收益。
- **若要加人工覆盖,应做 impostor_outside 的定向抽检(而非 blanket 接受)**——那才是真正该看的最高风险源。此为可选,不阻塞 Phase B。

## 6. 交付
再诊断报告交回 cc:四关证据 + impostor_outside 消融 + val 泛化 + 可行性三选一(可行→Phase C 接线计划 / 不可行→回炉方案 / 转回退门控)。**cc 核后才谈 Phase C。**

---
**起手**:wb 先出 **Phase B 训练+再诊断执行计划**(训练配置 / loss 加权 / 模型存路径 / gate 评测脚本复用)→ cc review 放行 → 再训。**先计划后训,先诊断后接线。**
