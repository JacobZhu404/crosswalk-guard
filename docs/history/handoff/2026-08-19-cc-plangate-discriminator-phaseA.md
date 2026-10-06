# cc plan-gate:wb 判别器重训线 Phase A(诊断 + 数据集方案)

> 署名:cc(arbiter) 转交:Jacob(拍板 Phase B/C);抄送 wb
> gate 对象:wb 分支 `wb-discriminator-mining`@`af1cc85`(未 push/merge)
> ├ `docs/reports/2026-08-19-wb-discriminator-retrain-phaseA-diagnosis.md`
> └ `docs/plans/2026-08-19-wb-discriminator-retrain-dataset-plan.md`
> cc 独立复现:`git show` 只读取 wb 分支文档(不进 wb 活 worktree);另写 `scripts/_ccprobe_falsegreen.py` 纯 cv2/numpy 逐 bit 复刻 `_sample_roi`,零生产码调用,在主工作树(有 input_video)离线复采 09。

---

## 1. 诊断根因订正 = 确认(bit-for-bit + 视觉)

wb 推翻两条旧 memory(扩展因子吃车绿 / (0.50,0.10)=真灯位)。**cc 独立复现全部证实:**

| wb 结论 | cc 复现证据 | 判定 |
|---|---|---|
| 扩展因子不是主因(仅 3/34 帧触发) | cc 复采 09[72,106] 35帧:**紧160触发32 / 扩320触发3 / dark0**;绿票 紧160=32、扩320=3 → 紧ROI主导 | ✅ 确认 |
| 主导根因=紧160 ROI + 宽松HSV投票扫进非灯绿 | cc crop `_ccfg_09_t95_prior_t160.png` = 路面车流,右下**黄绿出租车车身**(hue落[35,95]绿区);`prior_e320.png` 更大视野绿全来自车身/树冠 | ✅ 确认 |
| (0.50,0.10)"真灯位"实为树叶陷阱 | cc crop `_ccfg_09_t95_alt_t160.png` = **纯树**(树干+绿冠),无灯 | ✅ 确认 |

**诊断部分 PASS。** wb 方法学(GT-锚定分类,只把 GT 显式非-green 帧计假绿、未标注帧丢弃不夸大规模;质心偏移区分集中灯 vs 弥散环境绿)cc 采信。旧 memory `light-classifier-retrain` / `prior-misframe-rootcause` 相应更正。

## 2. cc 发现:数据集方案的 load-bearing 矛盾(比 wb §8 风险更尖锐)

### 2.1 头号阻断 —— 09 结构性无解(判别器解不了它自己的动机案例)

**09 的 prior 全程偏框,不止假绿段。** wb 诊断 §4.1 自证:t=20 / t=40(GT=green,落在违章窗 [11-72] 内)的 160px ROI 同样是"树木/车辆混合,未见清晰行人灯"。结合 memory `postmerge-f1-875`(09 TP 本脆 = 偏框 prior 采碎绿):

> **09 的 TP-green(违章窗 [11-72])与 FP-green([72-106])来自同一偏框 ROI 的车辆/环境绿,唯一区分是 GT 时间标签,ROI 内没有任何视觉灯信号。**

后果:一个"看灯形/灯境"的判别器,对 09 的 ROI **要么学不出**(全程无灯可依),**要么靠场景/时序记忆把 [11-72] 和 [72-106] 分开 = 正是屠掉 `ped_signal.pt` 的过拟合**。方案 §5.2 的验收指标「09 [11-72] green_signal recall ≥95% ∧ [72-106] no_signal recall ≥90%」在"不改 prior 坐标"红线下**可能根本不可同时达成**——因为两段的像素来源同质。这个矛盾方案没有正面处理。

### 2.2 当前零记分 headroom —— 投入 vs 回报要重估

按 wb 自己 §3.2 的表:
- **01**:FP 已由 #3+dedup 归零(`021dfce`),无当前误差;
- **06 / 02**:definitive 假绿"不直接产生记分 FP"(wb 原文);
- **09**:当前是 TP;[72-106] 是 34s **持续**绿,#3(T=6)按设计不降级持续绿,但只要 09 episode 已记 TP 且未额外产出窗外 confirmed FP,抑制 [72-106] **F1 零增益、却拿 09 脆 TP 冒险**。

→ **判别器线当前没有明确的 F1 headroom,且对 09 是净风险。** 它是"环境绿"的治本储备(防未来场景/时序漂移把环境绿变成记分 FP),不是当下的 F1 修复。这一条重定位了它相对 b2(车牌归属 + 碎片化有明确当前价值)的优先级。

### 2.3 次级技术条件(方向若继续则须解决)

- **场景数=样本数瓶颈**:11 视频=11 场景,LOVO 留一。green_signal 正例集中在极少数 violation 视频,留一折可能近零正例。§3.2 的 350/400/1350 是"3fps 估算"非实测。**Phase B 前须先产出真实 per-video × per-class 帧数表**,证每折有足够正例,否则 LOVO 不成立。
- **灯占 ROI 极小**:crop 实测灯是 160px ROI 角落小暗块(<5% 像素)。MobileNet@160 下采样恐吞掉小灯 → 模型只能靠场景上下文判别 = 场景记忆。须验证裁法(更小居中裁 / 更高分辨 / 只在 prior 确实框到灯的视频上训)。
- **Phase C 接线交互**:须证不与 #3(T=6 时序门控)/Fix A(dedup)双重降级,不回退 01(FP 已归零)/11(真绿 10.27s 近阈)。

## 3. 采信项(无异议)

LOVO 而非随机划分(避免相邻帧泄漏)✓;保留颜色 vote 作 fallback(低置信回退)✓;不碰 prior 坐标 / `_sample_roi` 阈值 / `ped_signal.pt`✓;新权重 gate 前不入库✓;GT 只作训练/评估 oracle 不进推理✓;三类 recall 硬验收(不为压假绿屠真绿)方向 ✓。

## 4. 裁定

- **诊断报告:PASS**(cc bit-for-bit + 视觉复现全部证实,已更正旧 memory)。
- **数据集方案:HOLD —— 方向技术上成立,但 §2.1 的 09 结构性矛盾是 load-bearing,且 §2.2 显示当前零记分 headroom。不带着这个矛盾进 Phase B。**

**转 Jacob 拍板(三选一 + 优先级):**
1. **对 09 放宽"不改 prior 坐标"红线**,先重定位 09 真灯位(注意 wb 已证候选 (0.50,0.10) 是树 → 需重找,09 视角下可能不存在可用行人灯像素)。若 09 真灯位不可得,判别器对 09 无解应明说。
2. **接受判别器不解 09**,把线重定位为"01/02/06 环境绿治本储备",在明确记分 headroom 出现前降优先级,让位 b2(qw 已 plan-gate PASS、有当前价值)。
3. **换法**(如 prior ROI 形状约束 / 弥散绿空间判别,承 memory `01fp-falsegreen-fix1-disproved` 的"弥散绿空间判别"方向)。

无论哪条,Phase B 启动前须补 §2.3 三项(真实类别帧数表 / 裁法验证 / 接线交互)。

## 5. 放行状态

- **状态:诊断 PASS / 数据集方案 HOLD 待 Jacob 优先级 + 红线拍板。** wb 分支 `af1cc85` 保持独立不 push/merge。
- cc 未进 wb 活 worktree(只 `git show` 读已提交文档,复现走独立探针脚本)。
- 基线不变:全 11 F1=0.941 / P=1.000(`021dfce`)。b2 线(qw)不受本 gate 阻塞,并行推进。

---
*署名:cc(arbiter)。证据=cc 独立探针 09[72,106] 复采(紧160=32票/扩320=3票)+ 三 crop 视觉(车身/树冠/纯树,无灯)。承 `docs/handoff/2026-08-19-cc-plangate-b2-PASS-conditional.md`、[[light-classifier-retrain]]、[[prior-misframe-rootcause]]、[[postmerge-f1-875-09-regression]]。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
