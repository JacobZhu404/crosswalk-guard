# cc → qw 任务 brief：车牌识别 — 先诊断（不写生产码）

> 署名：cc(coordinator/arbiter) — 走 [[qw-plan-execute-loop]] 双关口；本 brief 只开"诊断→自己规划"第一段，方案 gate 与效果 gate 照旧。
> 状态：**新 scope 派给 qw**（qw 此前=crosswalk，crosswalk v2 接线已收官 merge，见 [[crosswalk-v2-b1-c4-passed]]）。plate 目前无人认领，Jacob 点名让先看这块。

## 0. TL;DR
- 车牌识别**不是空地**：HyperLPR3(0.1.3,能跑) + 省份先验 + 多帧共识投票，**已优化过两轮**。任务是"量化 + 改进",不是"从头做"。低垂果实(省份、投票)已摘。
- **第一步只做诊断,不碰生产码**：跑 `eval_plate.py` 出**当前基线准确率**,再把失败**分桶归因**(没检到 / 检到但 OCR 读错 / 选错帧/共识投错 / 太小太糊)。定位真瓶颈后再出方案走 cc plan-gate。
- **相比 seg 的优势**：起步**不卡数据**——GT(events.csv 17 行带牌)、评测集、hyperlpr3 全在,立刻能测。
- **别踩的坑**：failure gallery 是 **24 incorrect / 20 missed / 6 correct**(近似五五开,**不是 missed 主导**)。**不许预设"就是检测召回问题"**,两类都得量化。

## 1. 产品定位(先摆正)
车牌是**"开给谁的罚单"**,在违章判定的**下游**。违章事件那条线 F1 已 0.889/GT 天花板 0.941(独立轴,见 [[crosswalk-v2-b1-c4-passed]])。**读牌不会提升违章 F1**,它是另一个维度:事件判对了但牌读错/读不到 → 罚单开不出。是真实执法价值,但别和违章 F1 混为一谈。

## 2. 现状 / 基础设施指针(qw 接手前先读)
- 识别核心：`src/redlight/models/plate.py`(HyperLPR3 + cv 兜底;`_apply_province_prior` 京-boost;`_filter_by_format` min_conf=0.6;`_is_valid_plate`;`_plate_color`)
- 多帧共识：`src/redlight/pipeline/plate_consensus.py`(按 track_id 投票)
- 管线集成：`src/redlight/pipeline/dag.py` n_plate 节点(plate_interval=3)→ `analysis.py`(plate_best per tid, `_find_plate_box`)
- 评测：`scripts/eval_plate.py`(GT=`datasets/gt/events.csv` violating_plates+other_plates,用 Levenshtein;GT 覆盖=**17 行带牌**)
- 失败画廊：`datasets/plate_eval_set/`(24 `_incorrect` / 20 `_missed` / 6 `_correct` + meta.csv)
- 标注闭环:annotate_plate / make_plate_gallery_v2 / feedback_to_eval_set **搭好但半成品**——当年**卡在 Jacob 没标完关键帧**。**第一步不依赖它**(events.csv GT 已够跑基线),别把它当阻塞。
- 历史:`docs/history/misc/root-handoff-plate*.md`(两轮 Beijing 优化史:京LNE560 违章02 修到 conf~1.0,过滤省份误识别)。语义锁定 pedestrian_green,dual-mode 已删。

## 3. 第一步:诊断任务(qw 执行,产出诊断报告)
**目标**:回答"当前读牌到底多准 + 错在哪一环",不写任何生产码。

1. **基线数**:跑 `PYTHONPATH=src ./.venv/bin/python scripts/eval_plate.py`,报告当前整体准确率(逐视频 + 汇总;Levenshtein 距离分布;完全正确率)。
2. **失败分桶归因**(核心,决定后续修法):对每个 GT 违章车牌,判定失败属于哪一桶并**量化占比**：
   - (a) **没检到**:该车任意帧都没产出 plate box(→ 检测/召回;可能太小/角度/遮挡)
   - (b) **检到但 OCR 读错**:有 box 但字符错(→ 识别器/省份逻辑/字符集)
   - (c) **选帧/共识错**:单帧读对过,但 `plate_consensus` 投票选错(→ 投票/关键帧策略)
   - (d) **图像质量**:box 对但车牌像素太小/糊/夜间(→ 上采样/超分/换检测层)
3. **交叉核对画廊**:24 incorrect / 20 missed 分别落在上面哪些桶,和 eval_plate 结论是否自洽(measurements-disagree → 找 bug,见 [[measurements-disagree-find-the-bug]])。
4. **结论**:指出**最高杠杆的那一桶**(占比最大且可改),给出改进方向的**候选假设**(先不实现)。

**交付** = 一份诊断报告(docs/,可复现命令 + 逐视频表 + 分桶占比)→ 提交 cc **plan-gate**。诊断过关后 qw 才出实现方案(第二段 gate),方案过关后才建 worktree 写码。

## 4. 红线 / scope(照旧)
- **不写生产码**,只诊断;诊断脚本读 GT 做 oracle 可以,**GT 永不进生产推理**。
- scoped git(**禁 `git add -A`**);署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`;实现阶段用**独立 worktree**(见 [[multi-agent-worktree-isolation]])。
- 不覆写 `ped_signal.pt` / B-only s1 缓存 / 2026-07-31 报告;模型权重 *.pt gitignored 不入库。
- 回写类脚本**先临时文件再原子替换**,别 open-truncate(见 [[no-open-truncate-rewrite]])。
- 过期文档归 `docs/history/`,别用 `archive/`(见 [[docs-archive-convention]])。

## 5. 给 qw 的开放问题(在诊断里回答)
- eval_plate 的 GT(17 行)够不够代表?哪些视频**没有** plate GT → 那些视频的读牌无法评测,要标出盲区(别静默截断,见 no-silent-caps)。
- `plate_consensus` 投票口径是否公平(按 track_id 聚合有没有把同车切成多 ID,撞 [[b2-tracking-fragmentation-blindspot]])?
- 省份先验(京-boost)在非京场景是否反而误伤?当前评测集是否全京场景?

---
*本 brief 由 cc 起草。qw 接手后先读 §2 指针 + §3 诊断任务;第一段产物(诊断报告)回 cc plan-gate,不擅启实现分支。*
