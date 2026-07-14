# Handoff 交接快照

## 1. 核心任务目标

优化车牌识别模块的标注评测工具，建立**标注→评测集→算法迭代**的闭环流程。用户只需标注1-2次关键帧，后续算法迭代时用评测集自动验证，无需每次重新标注。

### 验收标准
- 画廊工具支持手动填写正确车牌（corrected_plate字段）
- 智能关键帧采样策略覆盖识别错误帧、低置信度帧、首次识别帧
- 标注反馈可自动转换为评测集格式
- 画廊风格与灯态画廊保持一致

### 约束规则
- 使用便携版Git路径：`C:\Users\windows\gitportable\bin\git.exe`
- 输出目录：`data/output/plate_eval/`（画廊）、`datasets/plate_eval_set/`（评测集）
- 禁止创建多余文件，优先编辑现有文件

## 2. 已完成工作清单

### 修改文件
- [scripts/make_plate_gallery.py](file:///d:/redlight-crosswalk-violation/scripts/make_plate_gallery.py)
  - 新增 `corrected_plate` 输入框：用户可直接填写正确车牌文本
  - 实现智能关键帧采样策略（优先级排序）：
    - 识别错误帧（+3分）：detected != GT
    - 低置信度帧（+2分）：conf < 0.6
    - 首次识别帧（+1分）：首次出现的车牌
  - 标注反馈增加 `corrected_plate` 字段存储

### 新增文件
- [scripts/feedback_to_eval_set.py](file:///d:/redlight-crosswalk-violation/scripts/feedback_to_eval_set.py)
  - 将用户标注反馈（plate_feedback.csv）自动转换为评测集格式
  - 输出：`datasets/plate_eval_set/meta.csv` + `images/`
  - 字段包含：video, frame_idx, image_path, detected, corrected_plate, verdict, reason, note

### 删除文件
- `scripts/precompute_plate_diff.py`（方向偏了，用户不需要每次迭代都看diff）
- `scripts/make_plate_diff_gallery.py`
- `src/redlight/evaluation/diff_gallery_builder.py`

### 执行命令
- 生成画廊：`python scripts/make_plate_gallery.py --videos 违章01 违章02`
- Git提交：`a41b0a0 feat(plate): 优化车牌标注评测工具，增加手动修正输入框和智能关键帧采样`
- Git pull：已完成，当前分支与远程同步

## 3. 当前Git环境状态

- 分支：main
- 本地领先远程：3 commits（包含本次提交）
- 未提交改动（其他agent的工作）：
  - modified: scripts/train_ped_signal.py
  - untracked: datasets/ped_signal/
- Stash：已恢复（ddb66b20ae60a1559c854b4f27de77ce1c4a3ff2）

## 4. 中间产物

- 画廊HTML：`data/output/plate_eval/gallery.html`
- 车牌特写图：`data/output/plate_eval/crops/{video}/t{time}.jpg`
- 原始整帧图：`data/output/plate_eval/orig/{video}/t{time}.jpg`
- 标注反馈：`data/output/annotated/plate_feedback.csv`（用户标注后生成）

## 5. 当前阻塞点/未解决问题

- 用户尚未在画廊中标注关键帧（需用户手动操作）
- 评测集尚未生成（需用户标注后运行 `feedback_to_eval_set.py`）
- 未执行回归测试（需评测集生成后运行）

## 6. 下一步执行顺序（优先步骤）

1. **用户标注**：打开 `data/output/plate_eval/gallery.html`，对关键帧逐一判定并填写正确车牌
2. **生成评测集**：`python scripts/feedback_to_eval_set.py`
3. **回归测试**：`python scripts/run_regression_test.py`
4. **算法迭代**：根据评测结果优化车牌识别算法
5. **重新生成画廊**：`python scripts/make_plate_gallery.py`（验证优化效果）

## 7. 禁止重复修改/重复执行的内容红线

- 已删除的diff相关文件不要再创建
- 画廊工具已优化完成，不要再改回diff对比方向
- Git push操作需要用户确认（当前仅提交到本地）