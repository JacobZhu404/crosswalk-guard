# Handoff: BaseGalleryBuilder 统一 (2026-07-14)

## 目标
让 `scripts/make_light_gallery.py` 和 `scripts/make_plate_gallery.py` 统一使用 `src/redlight/evaluation/BaseGalleryBuilder` 基类，消除两份独立 HTML 生成逻辑，同时保留 plate 的手动修正输入框和智能关键帧采样。

## 已确认事实

- `BaseGalleryBuilder` 已增强：
  - 支持 `item["frame"]` 内嵌帧（实时检测模式，无需 frames_dir）
  - 支持 `_extra_feedback_inputs_html()` 子类扩展（plate 的 `corrected_plate` 输入框）
  - JS 自动收集 `.fb` 区域内所有 `input/select`（不局限于 verdict/reason/note）
  - `_load_feedback` 保留 CSV 全部字段，支持动态扩展
- `LightGalleryBuilder` 覆写 `_sample_representatives` 实现 confirmed-first 采样（与原脚本语义一致）
- `PlateGalleryBuilder` 覆写 `_sample_representatives` 为取前 N（保留脚本层优先级排序：不匹配>低置信>新车牌）
- `scripts/make_light_gallery.py` 从 414 行压缩到 ~65 行，功能无损失
- `scripts/make_plate_gallery.py` 从 407 行压缩到 ~130 行，保留实时检测 + 智能采样 + 手动修正
- 全量 pytest 205 项通过，gallery_builder 单测 19 项通过

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `src/redlight/evaluation/gallery_builder.py` | +item 内嵌 frame、+`_extra_feedback_inputs_html`、JS 自动收集、`_load_feedback` 保留全字段 |
| `src/redlight/evaluation/light_gallery.py` | +`_sample_representatives` confirmed-first |
| `src/redlight/evaluation/plate_gallery.py` | +`_sample_representatives` 取前 N、+`_extra_feedback_inputs_html` corrected_plate |
| `scripts/make_light_gallery.py` | 重写：用 `LightGalleryBuilder` |
| `scripts/make_plate_gallery.py` | 重写：用 `PlateGalleryBuilder`，保留 `analyze_video` 实时检测 |
| `tests/unit/test_gallery_builder.py` | 更新 `test_load_feedback_csv` 断言以匹配保留全字段的新行为 |

## 约束

- **零重叠**: 未碰 CC 负责的 `pipeline/models/run_video` 任何文件
- **scoped add**: 仅 add 上述 6 个文件 + handoff
- **向后兼容**: `BaseGalleryBuilder` 接口未变，子类覆写均为新增可选方法

## 下一步动作

1. 本 handoff 所在 commit 推 main 后，CC 可继续端到端冒烟
2. 未来如需新画廊类型（M1 crop / 违规帧复核），继承 `BaseGalleryBuilder` 只需实现 8 个抽象方法
3. 画廊样式统一后，如要改 CSS/JS（如增加批量导出 CSV），只需改 `gallery_builder.py` 一处

---

Co-Authored-By: Lingma <eval-agent@crosswalk-guard.agents>
