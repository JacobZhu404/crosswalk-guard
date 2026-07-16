# wb 计划 v4: 先修度量(端到端 + 模块级评测)

> 对应 brief: `docs/handoff/2026-07-16-cc-direction-fix-eval-measurement.md`
> 角色: wb(写码) 出计划, cc(规划+review) 复核放行后再写码。**本文件只出计划, 不改任何代码。**
> 依据: 重读 `eval_violations.py` / `violation_eval.py` / `videos.csv` / `events.csv` / `crosswalk.py` / `pipeline/tracker.py` / `violation_engine.py`(均于 2026-07-16 重新核对)。

## 0. 与 v3 的关系(声明)
v3(管线修复 04/11 flicker, 车辆锚定) **暂挂起**, 不在本计划内。理由: brief 实测指出当前尺子本身偏/缺(不跑负例、FP 不拆解、3 块模块无评测), 在偏尺子上优化 = 继续跑偏。等度量修好、看清"碎片占比 / 掩膜失准幅度"后再决定要不要做 v3 类管线修复。**本计划零管线逻辑改动。**

---

## Part A — 端到端度量修复(优先, 改动小, 先做)

### A0 现状核对(已读码确认)
- `eval_violations.py:83` `videos = args.videos or sorted(gt.keys())` —— `gt` 只含 `is_violation==1` 的视频 → **默认只跑 9 个正例, 负例 01/10 被排除**。✓ 印证 brief。
- `match_violation_events(preds, gts)` 是**纯函数、贪心 1:1**: `fp = len(preds)-len(used_p)`。当 `gts=[]`(负例)时, 所有 confirmed 自动变 `fp`, `tp=0, fn=0`。→ **负例的"任何 confirmed=FP"在现有匹配里已自然成立**, 无需改匹配逻辑。
- 头条 P/R/F1 完全由该函数算出。**保持 1:1 算法不动 = 满足 A3 红线。**

### A1 — 纳入负例 01/10
- 新增 `load_video_metadata(videos_csv)` → `{video: has_violation(bool)}`, 源 `datasets/gt/videos.csv`(以 `has_violation` 列为准, **不硬编码视频名**)。
- `eval_violations.py` 默认视频集改为 `sorted(metadata.keys())`(全部 11 个)。逐视频: `is_neg = not metadata[v]`; 负例 `gt_v = []` → 匹配后 `fp=len(conf)` 全部计入真误报。
- **预期头条会动(更诚实, 非削弱口径)**: 例 01 实测报 confirmed [48-60] → 现在计入 FP, 总体 P 从 0.538 降到 ~0.50, R 不变。这是 A1 的预期效果, 不是口径变化。

### A2 — FP 分类拆解(新增报告层, 不改匹配)
- 新增纯函数 `classify_false_positives(pred_events, gt_violations, is_negative, min_overlap_s)`:
  - 输入: `match_violation_events` 返回的 `fp_events`(未匹配 pred 索引) + `gt_v` + `is_negative`。
  - 逐 FP 事件判定:
    - **neg_true_fp**(真负例误报): `is_negative=True` → 直接归类。
    - **oow_true_fp**(窗外真误报): 正例视频上, 该事件与**任何** GT 窗重叠 `< min_overlap_s`(含零重叠) → 真误报。例: 09[94-96] 与 GT[11-72] 零重叠。
    - **fragment**(碎片): 正例视频上, 该事件与某 GT 窗重叠 `≥ min_overlap_s` 但未在 1:1 中配对 → 碎片(同一 GT 窗被切成多 episode)。例: 02[32-34]、07[4-13]。
  - 设计决策(待 cc 确认): 用 **`min_overlap_s`(默认 0.5s)做碎片/真误报分界**, 而非"任意 >0 重叠就算碎片"。理由: 避免 0.1s 擦边被误标碎片; 与匹配口径一致。边界 case 依赖此阈值, review 时请确认。
- 返回 `{"neg_true_fp":[idx], "oow_true_fp":[idx], "fragment":[idx]}` + 便于报告的逐条 `{video, span, light_state, gt_overlap_window}`。

### A3 — 头条 1:1 保持, 输出拆解
- `match_violation_events` **不改**。头条 P/R/F1 仍来自它。
- `eval_violations.py` 报告扩展(逐视频 + 总体):
  ```
  总体 P/R/F1(1:1)=0.500/0.778/0.636 | 真误报=N(负例M + 窗外K, 逐条video+区间+灯态) | 碎片=J(逐条video+区间+其重叠GT窗)
  ```
- **可选增强(已采纳, A 非阻塞)**: 头条 P 仍含碎片(诚实, 保留)。额外加一条诊断行:
  `P(仅真误报) = TP/(TP+真误报)` —— 即"多少次冤枉好人", 与含碎片的头条 P 并列, 一眼看清两种精度。不改头条, 纯增诊断。
- `violation_eval.py` 的 `aggregate()` 增加 `true_fp_total`(= neg+oow) 与 `fragment_total` 汇总字段(纯加法, 不改 P/R/F1 公式)。

### A 改动文件清单
- `src/redlight/evaluation/violation_eval.py`: 加 `load_video_metadata`(若放此模块) + `classify_false_positives` + `aggregate` 加字段。**不动** `match_violation_events`。
- `scripts/eval_violations.py`: 默认视频集扩到 11(读 videos.csv); 调用 `classify_false_positives`; 报告加拆解行; 支持 `--csv` 指向 videos.csv(默认 `datasets/gt/videos.csv`)。
- (可选) 把 `load_video_metadata` 放 `violation_eval.py` 还是 `eval_violations.py` 由实现定, 纯风格问题。

### A 的 TDD 用例(放 `tests/unit/test_violation_eval.py`)
1. **负例**: `gt_v=[]`, `is_negative=True`, 1 个 confirmed → `tp=0,fp=1`, `classify` → `neg_true_fp=[0]`, 头条 `P=0.0`。
2. **碎片**: `gt_v=[[21,68]]`, 2 个 confirmed `[21,68]`(TP) 与 `[30,34]`(overlap≥0.5 未配对) → `tp=1,fp=1`, `fragment=[1]`, `oow=[]`。
3. **窗外真误报**: `gt_v=[[11,72]]`, confirmed `[94,96]` 零重叠 → `tp=0,fp=1`, `oow_true_fp=[0]`, `fragment=[]`。
4. **aggregate**: 上述三者合并 → 头条 + `true_fp_total=2`(负例1+窗外1) + `fragment_total=1`。

---

## Part B — 补齐模块级评测(次优先, 需少量标注)

**总原则(写死): 不做全量学院派 GT。** 每块只测下游真正在乎的那个缺陷, 标注压到最小。wb 定 GT 格式 + 生成填空骨架(时间戳预填、框留空), **框/区间由 Jacob 填, wb 不瞎标**。

### B1 — 斑马线掩膜评测(把"11 候选缺失 62.9%"变可回归指标)
- **探测器事实**: `CrosswalkDetector.detect(frame)` 返回全宽二值掩膜, 其竖直范围为 `[cy1,cy2]`(即掩膜纹理区间)。→ 预测的"带" = 掩膜非零行区间 `[y0_pred, y1_pred]`。
- **GT 格式**(极简, 每视频一文件 `datasets/gt/crosswalk/{video}.json`):
  ```json
  {"video":"违章11",
   "frames":[{"ts":18.0,"y0":373,"y1":463,"note":"绿灯窗中段, 静止车脚下斑马线"}]}
  ```
- **关键帧选法(处理相机漂移)**: 对每个正例视频的每个 GT 违章窗(`events.csv is_violation=1`), 在窗内取 ≤3 帧(窗 10%/50%/90% 时刻); 另取 1 帧窗外中性帧查"无假带"。每视频封顶 3–5 帧。**每帧独立标, 不假设整段固定位置** → 漂移天然兼容。wb 生成骨架(只填 ts), Jacob 填 `y0/y1`。
- **指标**: 逐关键帧 `band_IoU = overlap([y0,y1],[y0_pred,y1_pred]) / union`; 汇总 per-video 均值 + 逐帧列表。直接量化"掩膜落没落在真斑马线上"。11 若真缺失 → 该帧 IoU≈0, 暴露无遗。
- **评测脚本** `scripts/eval_crosswalk_mask.py`: 对 GT 列出的每视频每关键帧, seek 到该 ts 取帧 → `CrosswalkDetector.detect(frame)`(**不带 vehicle_boxes, 纯探测器质量隔离**, 见 §待 cc 拍板) → 取掩膜竖直区间 → 算 IoU。
- **不碰管线**: 只调 `CrosswalkDetector` 公共 API, 不读/不改 `crosswalk.py` 内部。

### B2 — 跟踪评测(碎片 + 静止判定, 不做全 MOTA)
- **GT 格式**(最小, `datasets/gt/tracking/{video}.json`):
  ```json
  {"video":"违章02",
   "anchors":[{"window":[21,68],
               "frames":[{"ts":30.0,"box":[x0,y0,x1,y1]},{"ts":55.0,"box":[x0,y0,x1,y1]}]}]}
  ```
  每违章窗 1–2 个锚帧, 填违章车框即可。wb 生成骨架(填 window+ts), Jacob 填 `box`。
- **指标**(用 GT 违章车+窗做锚, 不依赖全帧标注):
  1. **ID 碎片化(归属规则写死, 已收紧)**: 统计窗内**归属该违章车的所有 track_id 之并集大小**(理想=1)。
     归属判据: 对窗内每一帧 f、每个在该帧活跃的 track t, 计算 `IoU(box_t(f), B_anchor)`(B_anchor 取时间上最近的锚帧框; 违章车静止, 框稳定, 任一锚框皆可作参考); 若 `≥ T` 则把 t 归入该车。最终 `碎片化数 = |{t : 某帧 IoU≥T}|`。
     **T 取值 = 0.5**(与 track 静止判定 IoU 量级 0.5–0.8 一致; 足够排除邻车、容得下本车道抖动)。可调, 但本次锁 0.5。
     ⚠️ 不是"取 IoU 最高的那一个"——那是单 track, 会低估碎片化。必须取并集。
  2. **静止判定准确率**: 窗内该车 `stationary=True` 的帧占比(违章车定义上应静止); 取并集内各 track 在窗内的样本。
- **如何取逐帧 track 数据(待 cc 拍板, 见下 §待 cc 拍板点 2)**: 引擎已在 `violation_engine._track_samples`(tid→[{ts,stationary,box,...}]) 累积逐帧样本, 正是所需。两种拿法:
  - (a) **首选, 加法不改逻辑**: 给 `cli.run()` 加 `return_track_samples=False` 形参, True 时把 `_track_samples` 一并返回(约 3 行, 纯加法, 不触动任何判定)。
  - (b) 备选: 独立 eval 工具 import `Tracker`+`ViolationEngine` 重跑窗内帧(代码量大、易漂移, 不推荐)。
- **评测脚本** `scripts/eval_tracking.py`: 跑完取 samples → 按锚框匹配 track → 算 ID 数 + 静止占比, 输出 per-video 首个基线。

### B3 — 不重复
灯态(`eval_light_fast.py` vs `light_states.csv`)与端到端(Part A)已有; 车牌端到端已间接 2/7, **独立车牌评测记为可选**, 不进本次。

### B 标注分工(明确哪些 Jacob 标)
- wb 交付: `datasets/gt/crosswalk/{video}.json` 与 `datasets/gt/tracking/{video}.json` 的**填空骨架**(ts/window 预填) + 一个 `scripts/gen_gt_skeleton.py` 从 `events.csv` 自动生成骨架 + 可选"dump 关键帧图"便于 Jacob 目视标框。
- Jacob 填: 每个关键帧的 `y0/y1`(B1)、每个锚帧的 `box`(B2)。**wb 不代标**。
- 最小清单: B1 正例 9 视频 × 3–5 帧 = 约 30–45 个 y-band 标注; B2 正例 9 视频 × 1–2 锚框 = 约 9–18 个框标注。量级很小。

---

## 验证(写码完成后)
- **Part A**: 跑 `python scripts/eval_violations.py`(全 11, 不加 `--reuse`)→ 新报告: 头条 1:1 + 真误报(含 01 负例逐条) + 碎片率, 逐视频列出 video+区间+灯态/重叠窗。对比 brief 基线(原 P=0.538 R=0.778, 现应 P≈0.50 且显式列出 01)。
- **Part B**: Jacob 标完后跑 `eval_crosswalk_mask.py` + `eval_tracking.py` → 首个 per-video 基线: 斑马线掩膜 IoU(尤其 11)、跟踪 ID 碎片数、静止准确率。
- 顺序: 先交 Part A(小、解锁方向); Part B 可并行但卡在标注。

---

## 红线自检(对照 brief)
- ✅ 头条 1:1 算法不动(`match_violation_events` 未改); P 下降仅因纳入负例(更诚实), 非口径削弱。
- ✅ 只改评测工具(A: `violation_eval.py`+`eval_violations.py`; B: 两个新 eval 脚本 + GT 骨架)。**管线逻辑零改动**。B2 唯一可能碰 `cli.py` 的 `return_track_samples` 是纯加法形参, 待 cc 确认是否接受。
- ✅ 负例以 `videos.csv has_violation` 为准, 无硬编码。
- ✅ Part B 不做全量 GT, 标注最小化, 框/区间由 Jacob 标, wb 只造骨架。
- ✅ 碎片只做"度量分离"(A2 单列), 不改判定行为(事件确认逻辑不变)。

## 待 cc 拍板的点(已全部裁定, 见下)
1. ✅ A2 分界阈值 = `≥ min_overlap_s`(0.5s)。理由: 与 TP 匹配阈值一致——够不到 0.5s 就配不成 TP, 对该违章不算有效检出, 归窗外真误报合理。擦边 0.3s 判真误报(非碎片)正确。
2. ✅ B2 取数 = 批准 (a) `cli.run(return_track_samples=...)` 加法形参。约束: 严格加法、默认 False、只返回已有的 `_track_samples`、零行为改动(app 层附加返回, 非管线逻辑)。
3. ✅ B1 主指标 = `detect(frame)` 不带 vehicle_boxes(本征掩膜质量)。备注: 生产路径实际带车框, 两者可能不同; 后续若要"生产态掩膜质量"再补带框次指标。本次主指标不带框。
4. ✅ v3(管线修 04/11)正式挂起, 等度量看清碎片占比/掩膜失准幅度再议。
5. 🔧 B2 归属规则已收紧(见 §B2): 取"窗内任一帧中框与锚框 IoU≥T 的所有 track_id 之并集大小"(T=0.5), 非"只取 IoU 最高那个", 否则碎片化数低估。
6. ➕ 可选增强(已采纳): 头条 P 保留含碎片; 另加诊断行 `P(仅真误报)=TP/(TP+真误报)`, 并列两种精度, 不改头条。
7. 🚦 红线复核全过: 头条 1:1 未动、只改评测工具、负例读 videos.csv、B 不做全量 GT、碎片只度量分离不改判定。
8. 🟢 放行: Part A 直接进写码(TDD 先行); Part B 收紧 B2 后即可; 标注清单最小性 OK(骨架 wb 造、框 Jacob 填)。

---
本计划已裁定放行, wb 进入写码。
