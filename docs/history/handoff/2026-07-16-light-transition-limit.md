# Handoff — 灯态全局转换约束 enforce_transition_limit 收尾 (2026-07-16)

> 署名: Claude Code · 日期: 2026-07-16 · 接力: 未提交 WIP 的语义/测试收尾

## 目标
把 WIP 的 `enforce_transition_limit`(fuse_light 后处理: 约束红绿转换次数、高置信段兜底吸收单帧抖动碎段)收尾: 修复它默认开启后跑挂的 2 个单元测试, 且不损失精度。

## 已确认事实(eval-b, YOLO 路径, 全 11 视频 confirmed-only)
- **基线(feature OFF) 89.5%(2757/3080)** → **feature ON 92.9%(2861/3080), +3.4%, 零回退。**
- 收益来源: 违章02 84.4→87.6(+3.2)、违章03 72.0→85.2(+13.2); 其余 9 视频全部持平(06/07/11 难点不动)。
- **收益机制(实证)**: 02/03 的 observe/YOLO 原始融合输出充满 flashing/unknown 碎段(如 02: `flashing[0-25]/unknown[25-110]/flashing`)。feature 的威力=把这些碎段吸收进红绿锚点 + step1 把 unknown 兜底成相邻同色。
- 测试: `tests/unit/test_temporal_fusion.py` 21 项全过; 全 unit 套件 205 项全过。

## 已排除项(别再走)
- **豁免 unknown/flashing 不被吸收 = 灾难**: 总体崩到 68%(02→0%, 03→34.8%)。因为 02/03 精度恰恰依赖吸收 flashing/unknown 碎段。已还原。
- **step1 给 unknown 兜底加"长遮挡时长门"= 严重回退**: 长 unknown 不兜底 → 02 全判 unknown → 0%。observe 常把整段读成 unknown, 兜底成相邻同色是精度主来源, 不能门控。已还原。
- 结论: **enforce 函数保持原作者逻辑不动**; 语义正确性靠"时长≥min_seg_dur 的真实长 flashing/长遮挡天然不被吸收(step2 dur 门 + step3 转换数门)"来保证, 而非按状态豁免。

## 修改文件清单
- `src/redlight/pipeline/intermediate_state.py`: 新增 `enforce_transition_limit`(原作者逻辑, 仅 docstring 加一句"真实长 flashing/长遮挡因时长≥min_seg_dur 而保留")。
- `src/redlight/pipeline/temporal_fusion.py`: `fuse_light` 加 `max_transitions=2, transition_min_dur=3.0` 参数, 末尾调用 enforce; <=0 时旧行为。
- `tests/unit/test_temporal_fusion.py`: 修 2 个测试用真实时长(flashing `*16`=4s、unknown `*40`=3.75s, 使真实长段跨过 min_seg_dur 存活); 新增 5 个 enforce 专项测试。

## 约束条件
- 精度是唯一客观仲裁: 任何改动必须跑 `python scripts/eval_temporal_fusion.py` 验证总体不回退(YOLO 路径, 全 11 视频约 4-6 分钟, grep 会缓冲到结束才出结果)。
- 生产调用方(cli → BatchViolationEngine.fuse_kwargs)不传 max_transitions → 默认吃到约束=2(ON), 这是预期生产行为。
- 多 agent 仓库: 提交用 scoped `git add <file>`, 勿 `git add -A`; 需署名。

## 下一步动作
**尚未提交**。若认可, 提交这 3 个文件, 建议 message:
`feat(light): fuse_light 加全局转换约束 enforce_transition_limit — 总体89.5%→92.9%(02/03带动), 零回退`
