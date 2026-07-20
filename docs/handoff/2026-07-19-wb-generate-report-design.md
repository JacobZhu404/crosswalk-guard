# 可复现识别结果报告 generate_report.py — 设计 / 实现说明

> **状态**：已实现并通过单测 + 13 单测；`--fresh` 全 11 视频跑通、验收全过（`tp=8/fp=1/fn=1 F1=0.889`），已提交 `925a9c4`。**交付报告须一次性 `--fresh` 全量生成（见 §3.1），禁止旧缓存 + 新重跑混时间戳。**
> **作者**：wb（WorkBuddy）。**owner**：Jacob。**审查**：cc（Claude Code，规划+review）。

## 1. 目标与三条拍板（cc 定夺）

反复使用的报告能力，每次重大迭代后跑一次，产出**一份自包含 HTML 仪表盘**，含每视频
「是否违章 / 违章车牌 / 时间 / 小作文 / 截图 / 完整识别视频」，并与原始标注对比。

cc 已收的三条决策：
1. **单 HTML 仪表盘** —— 一个文件承载总表 + 逐视频 drill-down。
2. **GT 对比两者都要** —— ①自动判定（复用 `violation_eval.match_violation_events`，保证 ✓✗ 与 `eval_violations` 的 tp/fp/fn 一致）；②并列展示原始 free-text（`source/annotation.csv` 的 `gt_description`）+ 结构化 GT（`events.csv` 对应行）。
3. **默认读缓存 + `--fresh`** —— 默认读每视频最近一次 `run_*/` 缓存；`--fresh` 重跑全 11 视频管线；缓存缺失时明确提示「请加 --fresh」。

## 2. 前置修复（必须，否则报告不可信）

`src/redlight/pipeline/visualizer.py` 的违章红高亮在 line 49 / 89 **写死 `denom="mask"`**，
而引擎（0.889 基线）判定用 `occ_denom="box"`。→ 标注视频标红的车可能 ≠ `violations.csv`
认定的车，报告「视频看到的结果」与结论不一致。

修复（`visualizer.py` + `cli.py`）：
- `Visualizer.__init__` 增 `occ_denom` 参数，完全镜像 `BatchViolationEngine` 的口径：
  `self.denom = occ_denom if occ in (mask,box) else "mask"`；
  `self.overlap = p["box_overlap"] if self.denom=="box" else p["overlap"]`。
- line 49 / 89 改 `denom=self.denom`。
- `cli.py:80` 传 `Visualizer(cfg, preset=preset, occ_denom=occ_denom or "mask")`。

确认无其他消费者（grep 全仓仅 `cli.py` 实例化 `Visualizer`）。不动判定逻辑。

## 3. 数据源与缓存约定

- run 缓存目录（cli/run_video 产出）：`data/output/run_{video}_{preset}/`，含：
  - `annotated.mp4`（标注视频，缺 `.onnx` 模型时走 CV/color 兜底仍能生成）
  - `violations.csv`（`event_id,track_id,status,start_ts,end_ts,vehicle_class,confidence,light_state,signal_assumption,plate,evidence_image`）
  - `evidence/evXXXX_tidX_车牌.jpg`（证据截图）
  - `cot/COT_{video}.md` + `cot/analysis_{video}.json`（小作文，确定性渲染）
- `generate_report.py` 默认 `glob run_{video}*` 取 mtime 最新；任一视频缺失 → 打印
  「缓存缺失: [...] 的 run_*/ 不存在。请加 --fresh 重跑管线。」并 `sys.exit(2)`。
- `--fresh`：循环 11 视频，`cli.run(cot=True, occ_denom="box", crosswalk_detector=CrosswalkDetectorV2, preset=balanced)`，
  并保持 `cfg.output.annotated_video/csv_report/evidence_images=True`。与 0.889 基线同配置
  （仅增 cot/annotated，二者均不改 events 输出 → tp/fp/fn 不变）。

## 3.1 一致性硬规则（交付报告务必遵守）

- **交付报告必须来自一次干净的 `--fresh` 全 11 视频**, 不可把「旧缓存视频 + 新重跑视频」混进同一份报告。
- 原因: denom 修复(`visualizer` 红框改跟随 `occ_denom=box`)提交后, **修复前**渲染的 `annotated.mp4` 仍是旧 `mask-denom` 红框, 与 `box-denom` 判定不一致 → 破「视频红框 == 结论」这条核心卖点。判定数(tp/fp/fn)不受影响(引擎一直 box), 但视频一致性对一份权威、反复使用的报告是硬要求。
- 故每次 denom / 判定相关改动后, 一律 `python scripts/generate_report.py --fresh` **一次跑全 11**, 再出 HTML; 不要用「部分读缓存 + 部分重跑」凑报告。
- 调试期可用 `--videos` 分批前台跑(崩因是后台长任务被平台 SIGKILL, 非代码; `run_*` 目录逐视频落盘 = 崩了能续), 但 **FINAL 报告仍须一次全量 `--fresh`** 重渲染。

## 4. 报告结构（单 HTML）

- **顶部 summary**：生成时间、管线版本（v2 + box + balanced）、min_overlap、聚合 tp/fp/fn/F1/覆盖/车牌、真误报/碎片，并标注「基线 0.889 保留」。
- **总表（11 行）**：`[视频 / 违章Y/N / 车牌 / 时间 / GT判定✓✗ / 子标记]`，可展开。
- **逐视频 drill-down（`<details>` 折叠）**：
  - 是否违章（`violations.csv` 有无 `confirmed`）；
  - 按违章车列表：车牌 + 时间窗 + 灯态 + 车型 + 证据截图；
  - 完整识别视频：`<video>` 内嵌 `annotated.mp4`（**相对路径**）；
  - 小作文：嵌入 `COT_{video}.md`（`<pre>` 保留格式）；
  - GT 对比（两者）：自动判定结果 + 并列 free-text / 结构化 GT。
- **资产策略**：默认相对路径引用（不拷 ~1GB）；`--bundle` 复制 `annotated.mp4/evidence/cot` 到 `report_bundle/` 成可迁移归档。

## 5. GT 对比（两者都要）

- **自动判定（复用 `match_violation_events`）**：pred = `violations.csv` 的 `confirmed` 事件
  （归一化车牌后传入）；gt = `load_violation_gt(events.csv)[video]`；`has_violation` 取
  `load_video_metadata(videos.csv)`。报告层 `normalize_plate` 归一化后传入，**事件级 tp/fp/fn
  仅由时间重叠决定**（车牌只影响 `plate_hit` 子标记），故与 `eval_violations` 8tp/1fp/1fn 口径一致。
  另用 `classify_false_positives` 给出 真误报/碎片 拆解；`_flags` 生成子标记
  （`is_violation✓/✗(漏)`、`负例✓(未发现违章)/负例误报✗`、`车牌h/t`）。
- **并列展示**：free-text 来自 `source/annotation.csv` 的 `gt_description`；结构化来自
  `events.csv` 该视频全部行。既看结论又能追溯。

## 6. 红线 / 验收

- 只读、只反映，不改 `violation_engine` / GT / canonical。
- 复用现有 viz/COT/eval，不重造。
- 输出单个自包含 HTML(+资产)。scoped git、署名、trunk main。
- GT 对比匹配逻辑加单测（`tests/test_generate_report.py`：车牌归一化 / ✓✗ 汇总 / 并列加载 / 聚合一致）。
- 验收：`generate_report.py --fresh` 跑通 → 出 HTML；总表 11 行 GT ✓✗ 与 `eval_violations` 的
  8tp/1fp/1fn 对得上；负例 01/10 显示「未发现违章」；抽 1 视频人工核对视频红框 == 表格车牌。

## 7. 明确不做

- 不优化效果（判定逻辑不动）、不改 GT、不做 LLM 生成小作文（COT 是确定性渲染，保持）。

## 8. 实现文件

- `scripts/generate_report.py`（纯标准库 + 懒导入 `redlight.app.cli` 以隔离 cv2）
- `tests/test_generate_report.py`（13 单测，`.venv` 跑）
- 修复：`src/redlight/pipeline/visualizer.py`、`src/redlight/app/cli.py`
