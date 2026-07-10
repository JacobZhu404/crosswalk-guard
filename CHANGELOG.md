# Changelog

## [2.0.0] - 2026-07-10 - 分层架构重构 (进行中)

### 工程化升级 (按用户 0-7 要求)
- 引入 superpowers (Spec-First TDD) + grill-me (压力测试) 方法论
- 分层架构: infrastructure / data_pipeline / models / inference / evaluation / pipeline / app
- 测试体系: pytest 单元 + 集成测试 (测试金字塔)
- Git 规范: Conventional Commits, 重大修改即提交
- 评测指标: precision/recall/F1/mAP/event-metrics/OCR-metrics
- 泛化保障: 独立评测集, 调参/评测严格分离

### 已知问题(修复中)
- E7: 静止判定 speed_thres=15 过严 + 首尾速度法对抖动敏感 -> Tracker V2 滑动窗口
- E8: 斑马线掩膜偏大 -> 掩膜质量评测 planned
- E11: config.yaml 与实现漂移 -> 已重建配置

## [1.0.0] - 2026-07-09 - 初版验证

- YOLOv8n + HyperLPR3 HIGH + CV 斑马线/红绿灯 全链路跑通
- 输出标注视频 + 违规 CSV + 证据图
- 实测: 违章01/02 违规=0 (判定阈值待调优)
