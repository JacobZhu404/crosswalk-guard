# badcases/ —— 迭代 bad case 模板

> append-only。每轮评测复盘发现算法错误，就在此目录追加一条 bad case。**不修改已有行**。

## 模板（`template.csv`，仅表头）

```
video,t_sec,window,modality,expected,actual,note,ts
```

| 列 | 说明 | 示例 |
|----|------|------|
| `video` | 视频名 | 违章04 |
| `t_sec` | 关键帧时间（秒） | 42.1 |
| `window` | 可选时间窗 `[start,end]` | [42.1,43.1] |
| `modality` | 模态 | light / crosswalk / tracking / plate / event |
| `expected` | 人工认定的真值 | red |
| `actual` | 算法输出 | unknown |
| `note` | 错误性质/原因 | 竞争水平边缘误判 |
| `ts` | 追加时间 ISO-8601 | 2026-07-19T20:00:00 |

## 约定

1. **append-only**：只在文件末尾追加，绝不改/删已有 bad case（保留完整迭代史）。
2. 每条 bad case 应可定位到具体视频+时间，便于后续复现与回归。
3. bad case 是 provenance 最高优先级层（`source=badcase`），Phase 2 的 `build_gt.py` 合并时应覆盖 feedback/source 的同位置真值。
4. 本目录 Phase 1 仅建模板；实际 bad case 在后续迭代中累积。

## 与扩充闭环的关系

每轮迭代：`评测发现 bad case → 追加到本目录 →（Phase 2）build_gt.py 据此重建 canonical → 重跑 eval 确认修复`。
