# feedback/ —— 画廊修正入库快照

> 本目录是**跟踪入库的快照**，不是画廊的工作区。

## 同步关系（重要）

- **画廊（gallery）实时写入** `data/output/annotated/`（`light_feedback.csv`、`crosswalk_feedback.csv` 等）。该目录是**工作区**，部分文件曾被 `.gitignore` 临时区吞掉（仅靠白名单 `!data/output/annotated/light_feedback.csv` 才留存；`crosswalk_feedback.csv` 无白名单、处于丢失风险）。
- **本目录 `datasets/gt/feedback/` 是受 git 跟踪的版本快照**：把画廊成果定期复制进来，作为可复现的权威修正记录。
- **Phase 1 是"复制不是搬移"**：画廊仍写 `data/output/annotated/`，本目录是其入库镜像。两者会随画廊继续演进而**漂离**，需定期重新同步（复制最新 `data/output/annotated/*_feedback.csv` 覆盖本目录快照）。
- 是否让画廊**直接写** `datasets/gt/feedback/`（消除双写与漂移）留作 **Phase 2 讨论**，本 Phase 不改动画廊写入路径。

## 文件清单

| 文件 | 行数 | 来源 | 说明 |
|------|------|------|------|
| `light_feedback.csv` | 593 | `data/output/annotated/light_feedback.csv` 快照（2026-07-15） | 灯态逐帧修正：`video,t_sec,frame_idx,pred,gt,verdict,reason,note,ts` |
| `crosswalk_feedback.csv` | 36 | `data/output/annotated/crosswalk_feedback.csv` 快照（2026-07-16） | 斑马线多边形修正：`video,t_sec,frame_idx,verdict,reason,note,annotated,imgh,imgw,poly,ts` |
| `light_feedback_2026-07-12_original.csv` | 36 | 旧 `datasets/gt/light_feedback.csv`（git mv 归档） | 早期灯态修正，已被上方 593 行版取代，作历史留存 |

## 覆盖（画廊修正覆盖的视频）

- `light_feedback.csv`：违章01,02,03,04,06,07,11（7/11）
- `crosswalk_feedback.csv`：违章02,03,04,05,06,07,08,09,11（9/11）

> 缺失视频（如 05/08/09 的灯态、01/10 的斑马线）表示画廊尚未在那里做修正——这些是后续迭代补反馈的候选。
