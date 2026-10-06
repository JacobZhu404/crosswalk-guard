# CC 派 qw:斑马线检测召回天花板 —— 诊断(Phase A)→ qw 提方案→cc 审→放手修→cc 验→迭代(Phase B)

> 出自 cc(coordinator/arbiter)。**工作模式升级(Jacob 2026-08-03)**: qw 从只读诊断提级到"**诊断 → 自己规划路径/方案 → cc review → 通过后放手实现(write)→ cc check 效果 → 迭代**"。
> **Phase A(只读诊断)**:钉死"当前斑马线检测(v11)离完美 GT 还差多少、差在**召回(没检到条纹)还是几何(全宽横带形状)**"。**Phase B(qw 规划+实现,cc 双关口把关)**:基于 A 的结论,qw 自己提修法方案→cc 审→放手改→cc 独立验效果→迭代。目标最终是把端到端违章 F1 拉向 GT 天花板(0.941)。方法论复刻你已熟的召回天花板套路(但这里 GT 是**真 polygon,已标注**,比灯态更实)。
> **纪律**: **先把 Phase A 跑完交 cc 复核,通过后再进 Phase B;Phase B 里 cc 审方案通过前不许写任何生产代码。** 先诊断后修、先计划后动。

## 0. 为什么是这个任务(杠杆)
历史实验(`diag_gt_crosswalk_ceiling.py`,报告 `docs/history/plans/2026-07-17-wb-plan-v6-crosswalk-detector-upgrade.md:8-14`)已证:**把完美 GT 斑马线塞进管线 → 端到端违章 F1 0.636→0.941(denom=box)**。斑马线检测是当前最大瓶颈。但那实验只证"GT 完美就好",**没分解 v11 差在哪**。本任务补这一刀:把 v11 与 GT 的差距拆成可行动的方向。

## 1. 口径钉死(侦察已确认的 file:line,你必须自己复核这些点再动)
**检测器(生产默认)= v11 经典 CV,输出 (h,w) uint8 二值掩膜(255=斑马线),不是 polygon、不是 YOLO-seg**:
- `src/redlight/models/crosswalk.py`,类 `CrosswalkDetector`,`detect()` 在 **crosswalk.py:37-45** → 派发到 `_cv_v11()`(**48-169**)。
- **关键形状约束(= 召回天花板根因,代码自己写死的)**: `_cv_v11` 最后画的是**整幅宽度的水平矩形带** `cv2.rectangle(mask, (0, cy1), (w, cy2), 255, FILLED)`(**crosswalk.py:162-167**)→ 永远画不出透视梯形。这是要量的核心。
- 参数硬编码在代码里(非 config): `BAND_FRAC=0.30 / STEP_FRAC=0.12 / MIN_RUN_FRAC=0.06`(crosswalk.py:72-77)、自适应阈值 `mean+1.5*std`(:100)、亮度下限 `100`(:132)。

**实验性 v2(时序 running-max + 透视梯形拟合)—— 未接管线,仅 `eval_crosswalk_mask.py --detector v2` 可调**:
- `src/redlight/models/crosswalk_v2.py`,`detect()`(:39-80),`_fit_trapezoid()`(:82-110)。docstring 自述 Phase-1 探针、预期 mask-IoU<0.5。

**掩膜怎么用进违章判定(不是 IoU、不是点在多边形内,是占比+阈值)**:
- `compute_overlap_ratio(box, mask, footprint=1.0, denom="box")` @ `src/redlight/infrastructure/geometry.py:23-53`: `footprint=0.5` 取车框**下半**;`denom="mask"` 返回 `inside/mask_area`、`denom="box"` 返回 `inside/box_area`。
- 生产 batch 引擎: `BatchViolationEngine.accumulate()` 用 `footprint=0.5, denom=occ_denom(默认 "mask")` @ `violation_engine.py:178-180`;阈值 `overlap_thr = box_overlap if denom=="box" else overlap`(:143-144),balanced preset `overlap=0.20 / box_overlap=0.20`(`tracker.py:20-23`)。
- **⚠ 这里有个必须写进分析的交互**: `denom="mask"` 下占比 = `inside/mask_area`。**v11 全宽横带 → mask_area 巨大 → 占比被稀释**。历史实验 denom=box(0.941) > denom=mask(0.875) 正好和这一致 —— 全宽带形状不只压 IoU,还压 denom=mask 下的占比。

**GT(真 polygon,已标注,像素坐标)**:
- `datasets/gt/crosswalk/违章*.json`,schema `{video, frames:[{ts, poly:[[x,y],...], note}]}`,`poly` = 像素坐标 [x,y] 对(**非归一化**,值到 ~1078)。
- 覆盖 **9 个视频**(违章02/03/04/05/06/07/08/09/11),**每个 4 个 anchor 帧**;**违章08 只有 3/4 帧有 poly**(1 帧空,跳过);**违章01/10 无 GT 文件**(负例视频,不在本诊断)。
- 栅格化: `poly_to_mask()` @ `src/redlight/evaluation/module_metrics.py:34-45`(`cv2.fillPoly`)。度量函数 `mask_iou / mask_band / band_iou` 同文件(:9-59)。

**怎么跑单视频单帧(最简,复刻现成脚本)**:
```python
from redlight.infrastructure.config import load_config
from redlight.models.crosswalk import CrosswalkDetector
import cv2
cfg = load_config("configs/config.yaml"); det = CrosswalkDetector(cfg)
cap = cv2.VideoCapture("input_video/违章05.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, int(ts*1000)); ok, frame = cap.read()
mask = det.detect(frame)   # (h,w) uint8; 不传 vehicle_boxes(纯检测天花板,与 eval_crosswalk_mask.py:52 一致)
```
现成参考: `scripts/eval_crosswalk_mask.py`(`eval_video()` :32-62 已做 v11/v2 的 mask-IoU vs GT,支持 `--detector`)。**你可以直接复用/扩展它**,不用从零写。

## 2. 三个要回答的问题
**Q1 —— v11 掩膜相对 GT poly 的 IoU 天花板**: 逐视频逐 anchor 帧算 `mask_iou(v11_mask, gt_mask)`。哪些视频均值 <0.5?(复刻 eval_crosswalk_mask.py 的判据)给出 9 视频 × ≤4 帧的表。

**Q2 —— 把每个 miss 拆成"召回不足 vs 几何过宽"(核心,决定修哪)**: 对每帧另算两个方向量:
- **召回 recall = |v11 ∩ GT| / |GT|** —— v11 带子覆盖了多少真斑马线(低 = 真没检到 / 垂直没盖住条纹)。
- **精度 precision = |v11 ∩ GT| / |v11|** —— v11 带子里多少是真斑马线(低 = 全宽带向两侧/上下溢出)。
- 预期:全宽横带 → **recall 高、precision 低**(带子纵向盖住了但横向溢满全宽)。**若真是这样 → 瓶颈是形状/几何(该约束带子或上 v2 梯形),不是"没检到" → 修匹配/形状而非硬提检测灵敏度。** 反之若 recall 也低 → 是真召回失败(条纹没扫到)。逐视频给出 recall/precision,并明确每个 <0.5-IoU 视频落在哪一类。

**Q3 —— v11 vs v2 头对头**: 同样逐视频算 v2 的 IoU/recall/precision(`--detector v2`)。v2 梯形是不是可行的立项方向,还是像 docstring 说的 IoU<0.5 还需大改?给出对比表 + 一句方向判断。

## 3. 交付物
- 报告 `docs/reports/2026-08-03-qw-crosswalk-recall-ceiling.md`:三问裁断 + 逐视频聚合 + 逐 anchor 明细表 + 方法学 + caveat。
- 逐帧 CSV `data/output/qw/crosswalk_recall_ceiling_per_frame.csv`:列 `video, ts, note, v11_iou, v11_recall, v11_precision, v2_iou, v2_recall, v2_precision, gt_area, v11_area, v2_area`(供 cc bit-for-bit 复核)。
- **不改任何生产代码/GT**。经典 CV 确定性,**单次跑,无 seed**。

## 4. 陷阱(必须遵守,否则 cc 复核会打回)
- **GT 只有 4 anchor 帧/视频 → 这是 per-anchor 天花板,不是 per-window**。别声称"全窗口召回率",只说这些采样帧。
- **违章08 跳过空 poly 那帧**(3/4);**违章01/10 无 GT,不跑**。
- **poly 是像素坐标**,`poly_to_mask` 需要 (h,w) —— 用**实际读到的 frame.shape**,别硬编码 1080p(不同视频分辨率可能不同,自己 print 确认)。
- **v11 检测天花板默认不传 vehicle_boxes**(与 eval_crosswalk_mask.py:52 一致,保证可复现)。vehicle-anchor bonus(crosswalk.py:140-147)会掺入车辆信息、污染"纯检测能力"口径 —— 如果想额外看它,单开一列标清楚,别混进主口径。
- **Q2 的 recall/precision 是掩膜像素级的,不是违章召回**。denom=mask 对占比的稀释效应(§1 那条)属于**分析讨论**,不用你去跑 cli.run 端到端(那是历史实验已覆盖的一半);但要在报告里点出这个交互,把"全宽带 → mask_area 膨胀 → denom=mask 占比被压"和历史 denom=box>mask 的结果对上。
- **别和灯态选灯召回天花板(80.5%)混**(`diag_candidate_recall_ceiling.py` 是那个,不是这个)。
- 结论前先看数据分层:若某视频 v11 完全没检到(IoU≈0)vs 检到但形状错(IoU 中等、recall 高 precision 低),这俩是**不同的立项方向**,分开报,别用一个均值糊过去。

## 5. 签名(Phase A 交付)
交付时提交署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`;scoped add(只加你自己的报告 + CSV + 诊断脚本,**绝不 `git add -A`**)。cc 会 bit-for-bit 两层复核(聚合层从你 CSV 自算 + 重建层 cc 独立重跑若干帧的 v11/GT 掩膜 IoU)再裁定。**A 通过才进 B。**

## 6. Phase B —— 诊断后:qw 规划 → cc review → 放手实现 → cc check → 迭代
这是 Jacob 定的新工作模式:给你方向和北极星,**路径和方案由你自己规划**,cc 只在两个关口把关。

- **B0 北极星**: 把 v11 掩膜质量拉向 GT 天花板(历史证 GT→端到端违章 F1 0.941,`docs/history/plans/2026-07-17-...v6...:8-14`),让**端到端违章 F1 真升**。**修法不预设** —— 由 Phase A 结论决定(几何过宽→约束带子形状/接 v2 梯形;真召回不足→调检测灵敏度/条纹扫描)。
- **B1 qw 提方案(先计划后动)**: 写 `docs/plans/2026-08-xx-qw-crosswalk-fix-plan.md`,四点讲清:①**诊断指向哪类瓶颈**(直接引 Phase A 的 recall/precision 数据,别泛泛)②**具体改法**(改哪个文件/函数、加新类还是改 v11、是否用 flag 门控)③**怎么验**(近端 = mask-IoU before/after;北极星 = 端到端违章 P/R/F1)④**风险**(负例 01/10 会不会新误报、会不会碰别的模块/拖慢)。**这一步不写代码。**
- **B2 cc review 方案(第一道 gate)**: cc 审四条,过了才放手 ——
  (a) **打在真瓶颈上**:别 A 说"几何过宽"你却去调检测灵敏度;
  (b) **可验**:有明确 baseline + before/after 度量口径;
  (c) **加性/门控**:新检测器走**新类或 flag**,**不静默替换生产 v11**(cc 验完、Jacob 拍板才接线),别动 `configs/config.yaml` 默认;
  (d) **守约**:独立 worktree/clone(不共享 HEAD)、scoped add、qw 署名、**不碰** `ped_signal.pt`/B-only s1 缓存/`2026-07-31` 报告/别的 agent 产物、原子写(tmp+os.replace)。
- **B3 qw 实现**: 在**独立 worktree/clone** 里改(多 agent 共享 HEAD 会被别人 checkout 拽走),改动加性/flag 门控,scoped add,qw 署名。
- **B4 cc check 效果(第二道 gate,cc 独立复核)**: cc 独立重跑**真实 detector**(非 GT 注入),量三样:
  (1) **mask-IoU 逐视频 before/after**(近端必须升);
  (2) **端到端违章 P/R/F1**——复用 `diag_gt_crosswalk_ceiling.py` 对 `datasets/gt/events.csv` 的匹配口径,但喂**真 detector**;
  (3) **负例 01/10 零新增误报**(斑马线过检可能凭空造违章)。
  **接受条件(全满足): mask-IoU 升(必要) AND 端到端 F1 不回退且最好升 AND 负例零新误报 AND 不拖垮别的模块。** 任一不达标 → 回 B1。
- **B5 迭代**: 不过就带 cc 给的具体证据回炉,再提→再审→再实现,直到达标;若判定此路不通(如 v2 梯形需大改超出本轮),记为独立立项交 Jacob,不硬凑。

---
**一句话给 qw**: 先跑 **Phase A** 只读诊断——量 v11 全宽横带(`crosswalk.py:162-167`)相对 GT poly(`datasets/gt/crosswalk/`,9 视频×4 anchor)的差距,拆成召回(|v11∩GT|/|GT|)vs 几何过宽(|v11∩GT|/|v11|)+ v11 vs v2 头对头,复用 `scripts/eval_crosswalk_mask.py`,只读单次无 seed。交 cc 复核过关后进 **Phase B**:你自己写修法方案(`docs/plans/`)→ cc 审(打真瓶颈/可验/加性门控/守约)→ 通过后独立 worktree 里放手改 → cc 独立验(mask-IoU + 端到端 F1 + 负例零新误报)→ 迭代。**cc 审方案前不写生产代码;新检测器 flag 门控不静默替换 v11。**
