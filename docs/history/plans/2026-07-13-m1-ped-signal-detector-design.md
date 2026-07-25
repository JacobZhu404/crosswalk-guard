# M1 行人信号灯检测器重设计 — 设计规格

> 状态: brainstorming 对齐(2026-07-13, 用户拍板 3 项核心分歧), 作为 M1 重设计的权威子规格。
> 署名: **Claude Code**。上游权威: `docs/plans/2026-07-12-design-requirements-v2.md`(需求 v2) / `2026-07-12-gt-format-spec.md`(GT 格式)。
> 语义基线: 违规 = 行人绿灯/闪烁 + 车静止 + 压斑马线(E12)。本 spec **只重设计 M1(红绿灯识别)**。

---

## 1. 为什么重设计 (根因, 以代码为准)

当前 M1(`traffic_light.py` `color-v7-stable`)= **纯手工 CV**(饱和亮斑 → 信号头聚类 → 空间锚 EMA)+ **逐视频先验 `light_priors.json`**。实测全 11 视频 accuracy≈0.36 / macro-F1≈0.26, 其中 **01/07/11 接近 0**。

**根因**(读代码得出, 非"准确率低"这么笼统):
1. `light_priors.json` 只标了 02/03/04 → 01/07/11 走无先验模式 → 空间锚被反光/远树/衣服/倒计时面板劫持。**先验本质是对评测集的过拟合, 不是检测器**, 不能当泛化能力宣称。
2. "信号灯在哪"靠手调锚点/先验(定位), "什么颜色"靠 bbox 内 HSV 均值(状态)——两者都脆。**真正故障在定位**, 定位对了颜色相对好判。
3. **行人信号 ≠ 机动车信号**(handoff §8.5): 本任务判违章依赖**行人绿灯(走路小人图标)**。误锁机动车灯 → 相位反转(机动车绿=行人红) → 整段判反。

## 2. 已锁定决策 (brainstorming, 2026-07-13)

| 编号 | 决策点 | 结论 |
|------|--------|------|
| **M1-D1** | 引入多少学习成分 | **轻训练**: YOLO 出候选框 + tiny 状态分类器(walk/stand/off)。定位半学习、状态全学习。 |
| **M1-D2** | 候选来源 | **YOLO traffic-light 框 ∪ HSV 饱和亮斑候选** 并集去重(覆盖最全; 不新增检测器; HSV 本就算)。 |
| **M1-D3** | 防过拟合 / 先验处置 | **Leave-one-video-out 交叉验证** + **逐视频先验从 shipped 路径下架**(仅留标定/调试脚本)。 |
| **M1-D4** | 状态分类器形态 | **两种都保留, 实现期 LOVO 择优**: (a) tiny CNN(48×48, 导 ONNX); (b) 颜色+形状特征 + 轻量分类器(小样本更抗过拟合)。 |
| **M1-D5** | 对外接口 | **不变**: `detect(frame)->{state,confidence,...}`, 引擎/DAG/可视化零改动(drop-in)。 |
| **M1-D6** | 迁移 | 新法 `method:"ped_classifier"`(auto 默认); 旧 `color` 保留一个周期仅供 A/B 评测, 确认后删除(不留永久开关, 避免 E13 footgun)。 |

## 3. 架构与数据流

```
frame
 ├─[候选并集] YOLOv8n traffic-light 框(COCO#9, 复用 M2 同一次推理) ∪ HSV 饱和亮斑候选
 │            → 去重/合并重叠框(IoU) → 候选 ROI 列表
 ├─[状态分类] classifier 对每个 ROI → {walk(绿), stand(红), off/非信号} + conf
 ├─[选灯]     空间锚 + 持久门控(复用现有 anchor 逻辑) 锁定"那盏行人灯",
 │            偏好 classifier 高置信的 walk/stand 头
 ├─[时序平滑] 近窗多数投票(复用 global_recent) → {green,red,flashing,unknown}
 └─[输出]     与现状同结构 dict (drop-in)
```

**不变式**
- 无置信 walk/stand 头 → `unknown` → 引擎按 D1 走 review(不强判)。
- shipped 路径**不含逐视频先验**; 评测在留出视频上跑, prior-free。
- 行人信号优先: classifier 的 walk/stand 类专指行人图标态; 机动车灯若被 YOLO 检出, 靠 classifier 归为 off 或靠锚点/持久门控排除。

## 4. 组件 (单一职责, 可独立测)

| 组件 | 职责 | 接口 | 依赖 |
|------|------|------|------|
| `signal_candidates` | 产出并集候选 ROI(YOLO灯框 ∪ HSV亮斑)+ 去重 | `candidates(frame, yolo_light_boxes)->[roi...]` | cv2/numpy |
| `signal_state_classifier` | ROI→(walk/stand/off, conf), 纯推理 | `classify(roi_img)->(label,conf)` | **运行时 cv2.dnn(ONNX)**, 训练用 torch/sklearn |
| `traffic_light`(重写编排) | 候选→分类→锚选→平滑→state dict | `detect(frame)->{state,...}`(**不变**) | 上两者 |
| `scripts/build_ped_signal_crops.py` | 抽帧→跑候选→裁 ROI crop→弱标签自举 | CLI | cv2 |
| `scripts/train_ped_signal.py` | 训练分类器 + 导出 ONNX + LOVO 评测 | CLI | torch/sklearn |

**YOLO 灯框来源(零额外算力)**: `VehicleDetector` 现只留 car/bus/truck。改为在**同一次 YOLO 推理**里额外产出 traffic-light(COCO#9)框, 经 `ctx["yolo_light_boxes"]` 传给 light 节点。不新增第二次推理。detect 节点每帧跑, light 节点节流复用最近一次。

**分类器运行时选型**: 训练用 torch(或 sklearn), **导出 ONNX 经 cv2.dnn 运行** → 运行时不绑 torch, 跨 Mac/Windows, 契合现有 `models/*.onnx` 可选-权重约定。缺权重时 `auto` 回退旧 `color` 路径(过渡期)。

## 5. 训练数据 (复用已有资产)

1. 从 11 视频抽帧(已有 `datasets/frames/` 或重抽) → 跑候选并集裁 ROI crop。
2. **弱标签自举**: 用 `events.csv` 可见段灯态 + 已知先验位置自动预标 ROI, 再用**灯态画廊工具**(`make_light_gallery.py`)人工校验。
3. 负类(off) = 落选候选(反光/倒计时牌/黄按钮/树/机动车灯)。
4. 落盘 `datasets/ped_signal/{crops/, labels.csv}`; labels.csv 列: `crop_path, video, frame_ts, label(walk|stand|off), source(yolo|hsv), verified(0|1)`。

## 6. 评测规范 (M1-D3, 防再过拟合)

- **Leave-one-video-out**: 分类器 11 折, 每折在其余 10 视频的 crop 上训、在留出视频的 crop 上测, 轮换。报每折 + 汇总 walk/stand/off 的 P/R/F1。**这是"新手机视频"泛化的代理指标**。
- **端到端 M1(prior-free)**: 整检测器在留出视频上跑, per-frame state 对 `events.csv` visible-only GT(复用 `eval_light_all.py` visible 指标, `light_evidence=visible` 段才计)。
- **验收阈值**(沿用 design v2 §5.2): 逐类 P/R ≥ 0.8(green/red/flashing/unknown); `inferred`/`occluded` 段期望输出 `unknown`, 计 review 覆盖率。
- **对比基线**: 同口径下 report 旧 `color`(prior-free) vs 新 `ped_classifier`, 证明净提升后再删旧路径。

## 7. 测试

- **保留语义**: 现 `scripts/run_tl_tests.py` 的 10 条(stable green/red、flashing、moving-red-rejected、no-signal-unknown、reflections-unknown、dim-green、amber→red、low-sat-green-rejected)必须语义存活; 因内部结构变, 改造为「mock classifier 返回既定 label + 真实编排」。
- **新增单测**: 候选并集/IoU 去重; "无置信 walk/stand → unknown"; 锚点在候选间的锁定/迟滞; classifier 接口(mock ONNX)。
- 运行: 纯逻辑测试 `PYTHONPATH=src pytest`(旧 pytest 需带 PYTHONPATH); 涉 cv2 的用 `.venv`(本机已配)或 `run_tl_tests.py`。

## 8. 范围边界 (YAGNI)

**本 spec 只做 M1 检测器**(候选并集 + 状态分类 + 锚选/平滑 + 训练/评测脚手架)。**不含**: `mask_iou`/`MOTA` 指标、COT、车牌、斑马线——各自独立推进。

## 9. 风险 / 开放项

- YOLOv8n COCO traffic-light 类**可能检不到中国行人信号头**(它是机动车灯训练的) → HSV 并集兜底; 若两者都漏, 该帧 unknown(交 review, 符合 D1)。未来可微调专用信号头检测器(M1-D1 的"重训练"档, 本期不做)。
- 小样本(几百 crop)下 tiny CNN 可能过拟合 → M1-D4 保留特征分类器做 LOVO 对比。
- 弱标签自举质量依赖人工校验; 未校验 crop 标 `verified=0`, 评测只用 `verified=1`。

## 10. References (⚠️ 待联网核实; 本环境 WebSearch/WebFetch 均不可用, 凭训练知识)

- **ImVisible 数据集 + LYTNet** — 手机相机域行人信号灯识别(红/绿/倒计时/无)+ 斑马线方向, 轻量 CNN。**最对口, 验证本 spec 的 crop-分类器思路**。记忆中 GitHub: `samuelyu2002/ImVisible`, 论文 ~2019。
- 机动车灯数据集(仅用于检测预训练, 注意行人≠机动车): **LISA Traffic Light**、**Bosch Small Traffic Lights (BSTLD)**、**DriveU (DTLD)**。
- **Roboflow Universe** 搜 "pedestrian traffic light"/"walk signal": 社区数据集 + 可导出 YOLOv8 权重, 适合自举。
- 建议在有网机器跑一次 deep-research, 产出核实过、带引用、含许可证/指标的短名单, 替换本节。
</content>
