# crosswalk-guard 脚本分类与「过程临时脚本」清理报告

> 生成：2026-10-03 ｜ 范围：`scripts/` 全部 102 个 `.py` + `src/redlight/`（生产包）
> 方法：通读 `src/` 流水线核心文件（`dag / violation_engine / decision / cli / tracker`），并抽取全部 102 个脚本的**模块 docstring + 定义 + 跨脚本 import + README/文档引用**做分类。
> 说明：**未逐行通读全部 102 个脚本**（约 1 万+ 行），但已覆盖每个脚本的用途声明、入口、耦合关系，足以支撑分类决策。

---

## 0. 结论速览

| 分层 | 文件数 | 是否最终工程需要 | 处置建议 |
|---|---|---|---|
| `src/redlight/`（生产包） | 51 个 .py | ✅ 必需（推理流水线本体） | 全部保留 |
| 生产入口 / 交付 | 4 | ✅ 必需 | 保留（`generate_report.py` 待确认冗余） |
| 核心评测框架 | 15 | ✅ 必需（质量基线） | 保留 |
| 质量门禁 / 测试 | 3 | ✅ 必需 | 保留 |
| GT 构建 / 标注闭环 | 23 | 🟡 可复现研发链（非交付物但保留） | 保留 |
| 训练 / 数据制备 | 11 | 🟡 可复现再训练链（非交付物但保留） | 保留 |
| 标定 / 环境 / 抽帧 | 5 | 🟡 一次性但可复现 | 保留 |
| 文档生成 | 2 | 🟡 交付物生成 | 保留 |
| **过程临时脚本（调查/实验/agent 验收一次性）** | **39** | ❌ 最终工程不需要 | **建议归档到 `scripts/archive/`** |

**最终工程不需要的临时脚本共 39 个**（详见 §3），绝大多数是 `diag_*` 只读诊断、`measure_*` / `sweep_*` 实验、`qw_*` agent 验收一次性件、以及两个 `_` 前缀私有辅助件。它们只为定位特定历史 bug（01FP、05/09 prior 重定位、误绿测量、可分性、track 碎片等）服务，结论已沉淀进 `docs/handoff` 与 `docs/reports`，无持续复用价值。

---

## 1. `src/redlight/` —— 生产包，全部必需

这是最终工程的本体，被 `run_video.py` / `cli.run` 调用，不应触碰：

- `infrastructure/`（config / geometry / image_utils）
- `data_pipeline/`（frame_sampler / ped_signal_dataset）
- `models/`（vehicle / crosswalk / crosswalk_v2 / traffic_light / plate / ped_light_selector / governing_disc / l3_ped_vehicle / signal_candidates / signal_state_classifier）
- `inference/engine`
- `evaluation/`（metrics / evaluator / 各 gallery / report_formatter 等）
- `pipeline/`（dag / tracker / violation_engine / plate_consensus / visualizer / temporal_fusion / decision / intermediate_state / analysis）
- `app/cli`

> 注：诊断脚本大量 `from redlight.app import cli` / `from redlight.models import governing_disc` 反向依赖生产包——说明 `src/` 是稳定基座，临时脚本依附其上，删临时脚本不影响 `src/`。

---

## 2. 最终工程需要的脚本（保留，63 个）

### 2.1 生产入口 / 交付（4）
| 脚本 | 用途 | 备注 |
|---|---|---|
| `run_video.py` | CLI 便捷入口 → `cli.run` | README 官方入口 |
| `run_diag_generic.py` | 通用灯态时间线诊断启动器 | README 官方入口 |
| `render_report.py` | **最新**交付脚本：逐视频文字报告 + 带标注结果视频 | 最近 `e7e78c5` 提交，当前交付物 |
| `generate_report.py` | 可复现识别结果报告（617 行） | ⚠️ 疑似被 `render_report.py` 取代，设计文档在 `docs/history/` 已归档；**删前需确认 `render_report` 已完全覆盖其能力（含 `tests/test_generate_report.py` 13 个单测）** |

### 2.2 核心评测框架（15）—— 能力解耦评测与端到端基线
`eval_violations.py`（e2e 事件级 P/R/F1）、`eval_light_all.py` / `eval_light_fast.py` / `eval_light_gt.py` / `eval_m1_light.py`（灯态）、`eval_plate.py`（车牌）、`eval_crosswalk_mask.py`（斑马线掩膜）、`eval_tracking.py` / `eval_tracking_ab_split.py` / `eval_tracking_gtfree.py`（跟踪）、`eval_temporal_fusion.py`（②层融合）、`eval_selection_quality.py` / `eval_selector_l2.py` / `eval_selector_l3.py` / `eval_selector_lso.py`（选灯门控）。

> ⚠️ 耦合：`eval_selection_quality.py` 第 37 行 `from scripts.train_governing_discriminator import recommend_tau`。若归档判别器线相关件，必须**成组**处理（见 §4）。

### 2.3 质量门禁 / 测试（3）
`validate_gt.py`（GT 格式一致性，红线性别）、`run_regression_test.py`、`run_tl_tests.py`。

### 2.4 GT 构建 / 标注闭环（23）—— 可复现研发链
`build_gt.py`、`gen_gt_skeleton.py`、`apply_annotation_gt.py`、`ingest_light_gt.py`、`build_light_gt_gallery.py`、`build_light_regression.py`、`build_eval_set.py`、`feedback_to_eval_set.py`、`serve_gallery.py`、`apply_classifier_retrain_feedback.py`、`apply_ped_signal_feedback.py`、`confirm_plate_labels.py`、`annotate_plate.py`、`make_light_gallery.py`、`make_light_gallery_v2.py`、`make_light_gt_template.py`、`make_ped_signal_gallery.py`、`make_plate_gallery.py`、`make_plate_gallery_v2.py`、`make_crosswalk_gallery.py`、`make_tracking_gallery.py`、`make_classifier_retrain_gallery.py`、`make_mismatch_gallery.py`。
> 这些是「人工标注 → 反馈 → 重训」闭环工具，非交付物，但保留以支撑后续模型迭代。

### 2.5 训练 / 数据制备（11）—— 可复现再训练链
`train_ped_signal.py`、`train_classifier_retrain.py`、`train_governing_discriminator.py`、`train_l3_full.py`、`build_ped_signal_crops.py`、`collect_candidates.py`、`collect_candidates_dense.py`、`build_l3_labeling.py`、`mine_classifier_retrain.py`、`mine_negatives.py`、`filter_low_signal_rows.py`。

### 2.6 标定 / 环境 / 抽帧（5）
`scan_pedestrian_signal.py`、`identify_pedestrian_signal.py`、`derive_priors.py`（生成 `light_priors.json`，已提交但保留可复现）、`download_models.py`、`extract_frames.py`。

### 2.7 文档生成（2）
`make_devdoc.py`、`make_manuals.py`。

---

## 3. 过程临时脚本（最终工程不需要，建议归档，39 个）

> 判定标准：为定位/量化某个**特定历史问题**而写的一次性只读诊断、阈值扫描、实验测量、agent 验收件。结论已写入 `docs/handoff/*` / `docs/reports/*`，脚本本体无持续复用价值。**删除前注意 §4 耦合。**

### 3.1 `diag_*` 只读诊断（24）
`diag_01fp_repro.py`（违章01 FP 根因）、`diag_05_candidate_gen.py`、`diag_05_prior_relocation.py`（05 prior 重定位）、`diag_0906_green_failure.py`、`diag_09_tp_source.py`、`diag_09_true_light.py`、`diag_base_leakage_per_video.py`⚠️（依赖 eval_selection_quality）、`diag_candidate_recall_ceiling.py`、`diag_classifier_feasibility.py`、`diag_classifier_retrain.py`、`diag_crosswalk_recall_ceiling.py`、`diag_feature_separability.py`、`diag_gt_crosswalk_ceiling.py`、`diag_plate_bucket.py`、`diag_plate_track_attr.py`、`diag_prior_motion.py`、`diag_selection_falsegreen.py`、`diag_selector_failures.py`、`diag_signal_timeline.py`、`diag_small_light_res.py`、`diag_structural_separability.py`、`diag_temporal_separability.py`、`diag_vehicle_track_fragmentation.py`、`diag_yolo_vs_gt.py`。

### 3.2 误绿测量 / 成因重算（3）
`measure_falsegreen.py`、`measure_falsegreen_canonical.py`、`recompute_cause_perframe.py`。

### 3.3 门控 / 阈值扫描 / 消融（4）
`verify_gates_01fp.py`、`sweep_box_overlap.py`、`sweep_step0.py`、`merge_sweep_step0.py`（扫频 JSON 聚合辅助）。

### 3.4 `qw_*` agent 验收一次性件（5）
`qw_b2_audit.py`、`qw_drift_probe.py`、`qw_plate_events_report.py`、`qw_wiring_ablation.py`、`qw_wiring_snapshot.py`（qw agent 在 b2/wiring 工作中的护栏与快照，属交接证据）。

### 3.5 私有辅助 / 一次性 GT 生成（3）
`_agg_neg_quality.py`（负例质量扫频聚合）、`_gen_light_states_full.py`（一次性生成 `light_states.csv`，已产出）、`analyze_signal_presence.py`（Phase B 信号占比证据统计）。

---

## 4. 删除 / 归档时的耦合与安全注意

1. **成组处理判别器线**：`train_governing_discriminator.py` ← `eval_selection_quality.py` ← `diag_base_leakage_per_video.py`。
   - `eval_selection_quality.py` import 了 `train_governing_discriminator.recommend_tau`；
   - `diag_base_leakage_per_video.py` import 了 `eval_selection_quality`。
   - 若归档 `diag_base_leakage_per_video.py`（§3.1 已标 ⚠️），不影响前两者；但若想清掉整条 HOLD 的判别器线，三者需**一起**移走，否则 `eval_selection_quality.py` 会 import 失败。当前判别器线被 Jacob 拍板 HOLD，建议保留 `train_governing_discriminator.py` + `eval_selection_quality.py`，仅归档 `diag_base_leakage_per_video.py`。

2. **`generate_report.py` 与 `render_report.py` 冗余**：先确认 `render_report.py` 已覆盖前者全部交付能力（含 `tests/test_generate_report.py` 的 13 个单测），再决定是否归档 `generate_report.py`。不要无验证删除。

3. **遵守项目红线（AGENTS.md §3）**：这是多 agent 协作仓库，临时脚本可能是某 agent 的交接证据（如 `qw_*`、`diag_*_relocation`）。**建议用 `git mv` 移到 `scripts/archive/` 而非 `rm`**，保留历史可追溯，且不破坏他人未合并分支的引用。不要对共享树 `reset --hard` / `checkout .` / `clean -fd`，归档后单独小提交并署名。

---

## 5. 建议的归档操作（待你确认后执行，本报告未改动任何文件）

```bash
# 1) 建归档目录
mkdir -p scripts/archive

# 2) 移动 39 个临时脚本（保留 git 历史）
git mv scripts/diag_*.py scripts/archive/            # 24
git mv scripts/measure_falsegreen.py scripts/measure_falsegreen_canonical.py \
       scripts/recompute_cause_perframe.py scripts/verify_gates_01fp.py scripts/archive/
git mv scripts/sweep_box_overlap.py scripts/sweep_step0.py scripts/merge_sweep_step0.py scripts/archive/
git mv scripts/qw_b2_audit.py scripts/qw_drift_probe.py scripts/qw_plate_events_report.py \
       scripts/qw_wiring_ablation.py scripts/qw_wiring_snapshot.py scripts/archive/
git mv scripts/_agg_neg_quality.py scripts/_gen_light_states_full.py \
       scripts/analyze_signal_presence.py scripts/archive/

# 3) 单独小提交 + 署名（不碰 src/ 与其他 agent 分支）
git commit -m "chore(scripts): 归档 39 个过程临时诊断/实验脚本到 scripts/archive" \
  -m "Co-Authored-By: ..."
```

> 注：本报告依据脚本 docstring 与结构分类；归档前建议对 `generate_report.py` 冗余与判别器线耦合（§4）两项做二次确认。
