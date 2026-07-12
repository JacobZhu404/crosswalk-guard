# 外部模型与开源数据集调研报告

> 生成时间: 2026-07-12 14:14
> 背景: 本项目数据量有限(仅 11 段手机视频), 用户希望用现成预训练模型/权重或开源数据集
>        来增强 红绿灯/车牌/斑马线/跟踪/分割/识别, 减少人工调参。
> 环境约束: **Win10 CPU-only (i7-10750H 6核), 无 GPU**。torch/onnxruntime 可用。

---

## TL;DR 结论

1. **车牌** ✅ 已用 HyperLPR3, 足够, 保持即可(备选 LPRNet / Transformer 端到端 99.3%)。
2. **斑马线** ✅ 有专用数据集 **CDSet-3434** + 现成 YOLOv5/CDNet 代码与权重 → 直接替换我们脆弱的 v10 梯度法, 最易见效。
3. **交通灯检测** ⚠️ 公开数据集丰富(LISA/Bosch/TT100K/LARA), 但**多为机动车灯**; 行人灯标注极少。建议: 用 YOLOv8n 在 TT100K+LISA 训"通用灯检测"→ 稳找所有灯 → 再用我们的颜色法+行人灯位置先验选行人灯。
4. **跟踪** ✅ 用现成 **BoxMOT(封装 ByteTrack/OC-SORT/BoT-SORT)** + YOLOv8n 检测, 即插即用, 替换自研 track, 不需重写关联逻辑。
5. **分割** 🟡 斑马线用检测(CDSet)即可, 分割(BDD100K/Mapillary/Cityscapes)可暂缓, 仅在需要精确掩膜时引入。
6. **识别(车辆/人等)** ✅ YOLOv8n COCO 预训练直接覆盖, 无需额外数据。

**核心策略(回应"减少调参")**: 不要手调我们 tiny 的自定义检测器, 而是
**下载公开大数据预训练 → 冻结 backbone 只微调 head → 用我们的 11 视频做 fine-tune/评测**。
数据少时 transfer learning 远优于从零手调阈值。

---

## 分项详表

### 1. 交通灯识别 (Traffic Light Detection / State)

| 项 | 内容 |
|----|------|
| **现成模型/权重** | YOLOv8n 在 LISA 上训练(github: AbhishekSinha-git/LISA-traffic-light-dataset-yolov8-Training); Faster-RCNN 在 LISA(evanchien/LISA_TL_DETECTION); YOLOv3 在 Bosch(berktepebag/Traffic-light-detection-with-YOLOv3) |
| **开源数据集** | **TT100K**(清华-腾讯, 中文场景, 最相关) · **LISA**(4.3万图, 昼夜, go/stop/warning) · **Bosch Small TL**(5093图, 干净) · **LARA** · **GTSDB**(德系标志为主含灯) · **WPI** |
| **对咱项目建议** | 用 YOLOv8n 在 **TT100K + LISA** 训"通用交通灯检测"模型 → 替换手工 saturation 检测器(稳找所有灯, 不再靠阈值)。**行人灯状态**仍用我们的颜色法 + 位置先验(行人灯多在近端/有走路-站立人图标)选出行人灯。 |
| **行人灯专门数据集** | ⚠️ 公开极少。需从 TT100K/LISA 筛含行人灯样本, 或自标(11视频抽帧标少量)。这是"数据有限"真正难点。 |
| **CPU 可行性** | ✅ YOLOv8n(nano) CPU 可推理 |

### 2. 车牌号识别 (License Plate Recognition)

| 项 | 内容 |
|----|------|
| **现成模型/权重** | **HyperLPR3**(已在用, 北京智云视图, 支持中/新能源/警用等, CPU实时) · **LPRNet**(Intel轻量, 端到端) · **Yolov5+PaddleOCR**(12种车牌) · **Chinese-LPR-Transformer**(端到端, 普通牌 99.3% / 高难 85.7%) · PP-Vehicle(ONNX部署) |
| **开源数据集** | **CCPD**(中文车牌, 文件名存标注) · **CRPD**(高难) · EasyPR数据集 |
| **对咱项目建议** | **保持 HyperLPR3**(已集成, CPU友好)。仅当准确率不足时换 LPRNet/Transformer。车牌读取继续用"视频全局 + track id 关联"(E19)。 |
| **CPU 可行性** | ✅ HyperLPR3/LPRNet 均为轻量 CPU 实时 |

### 3. 斑马线识别 (Crosswalk / Zebra Detection)

| 项 | 内容 |
|----|------|
| **现成模型/权重** | **CDNet**(基于 YOLOv5 的实时斑马线检测, github: zhangzhengde0225/CDNet, 论文 Neural Computing 2022, 含训练权重与代码) |
| **开源数据集** | **CDSet-3434**(3434张车载斑马线图, 含白天/雨/遮挡/变形/夜/破损/炫光; 3080训练+354测试+1770有无测试; Zenodo/百度网盘) |
| **对咱项目建议** | **直接用 CDSet-3434 训 YOLOv8n/5 斑马线检测器** → 替换我们脆弱的 v10 梯度密度法(草地误检/掩膜错位 E15/E17)。这是**最易见效**的一步: 专用数据集+现成代码。 |
| **CPU 可行性** | ✅ YOLOv8n/5 nano CPU 可推理 |

### 4. 动态物体跟踪 (Multi-Object Tracking)

| 项 | 内容 |
|----|------|
| **现成模型/权重** | **BoxMOT**(封装 BoT-SORT/ByteTrack/OC-SORT/DeepOCSORT, 即插即用, 配合 YOLOv8/9/10) · **ByteTrack**(ECCV2022, MOT17 80.3 MOTA, 30fps V100) · **DeepSORT** · **PaddleDetection PP-Tracking** |
| **开源数据集** | **MOT17/MOT20**(行人车辆跟踪基准) · **BDD100K**(含跟踪标注) · CrowdHuman · Cityperson |
| **对咱项目建议** | 用 **BoxMOT + YOLOv8n 检测** 替换自研 track。关联逻辑(ByteTrack)成熟稳健, 不需重写。占用斑马线的车 → 用 track id 关联其全局车牌(E19)。 |
| **CPU 可行性** | ✅ YOLOv8n 检测 + ByteTrack 关联均为轻量, CPU 可跑(速度较慢但可接受) |

### 5. 分割 (Segmentation)

| 项 | 内容 |
|----|------|
| **现成模型/权重** | YOLOv8-seg · SegFormer · Mask2Former(均有多数据集预训练权重) |
| **开源数据集** | **BDD100K**(可行驶区域/车道线/实例分割) · **Mapillary Vistas**(66类, 含斑马线+交通灯) · **Cityscapes**(19类, 含交通灯, 但斑马线非独立类) |
| **对咱项目建议** | 🟡 **斑马线用检测(CDSet)即可, 分割暂缓**。仅当需精确斑马线掩膜/可行驶区域做占用比例(D2分母)时才引入 BDD100K/Mapillary 预训练分割模型。 |
| **CPU 可行性** | ⚠️ 分割模型较重, CPU 推理慢; 优先级低 |

### 6. 通用识别 (车辆/行人/骑行者等)

| 项 | 内容 |
|----|------|
| **现成模型/权重** | YOLOv8n COCO 预训练(ultralytics 直接 load) |
| **开源数据集** | COCO · Object365 · OpenImages |
| **对咱项目建议** | COCO 预训练 YOLOv8n 直接用作"检测所有道路参与者"的 backbone, 无需额外数据。 |
| **CPU 可行性** | ✅ YOLOv8n nano CPU 可推理 |

---

## 推荐落地路径(分阶段, 均 CPU 可行)

```
阶段 0 (立即, 低风险): 
  - 斑马线: 下 CDSet-3434, 训 YOLOv8n 斑马线检测器 → 替 v10 梯度法
  - 跟踪: 接 BoxMOT+ByteTrack → 替自研 track
  - 这两个有现成数据+代码, 不用我们数据, 直接降噪

阶段 1 (短期, 需下载公开数据 fine-tune):
  - 交通灯: 下 TT100K+LISA, 训 YOLOv8n 通用灯检测 → 替手工 saturation 找灯
  - 行人灯状态: 保留颜色法 + 加"行人灯位置/形状先验"(阶段0检测出的所有灯里筛行人灯)

阶段 2 (按需):
  - 车牌: 仅当 HyperLPR3 不够时换 LPRNet/Transformer
  - 分割: 仅当需精确斑马线掩膜时引入 BDD100K/Mapillary
```

---

## 数据量有限 → Transfer Learning 策略

| 策略 | 说明 |
|------|------|
| **冻结 backbone, 只训 head** | 公开大数据预训练好的特征提取器不动, 只用少量数据训分类/检测头 → 几十~几百张即可 |
| **混合训练提升泛化** | 公开数据(大) + 我们的 11 视频(小, 领域相关)混合 → 既学通用特征又适配手机场景 |
| **我们的 11 视频角色** | 主要做 **fine-tune + 评测闭环**(visible-only 指标), 而非从零训练 |
| **行人灯特例** | 公开缺行人灯标注 → 从 11 视频抽帧人工标少量(几十~百张) → fine-tune 行人灯分类头 |

---

## 注意事项

1. **License**: Bosch 需注册邮件获取; TT100K/LISA/CDSet 多为研究许可(非商用 OK, 商用需查)。HyperLPR3/ByteTrack/BoxMOT 开源协议宽松。
2. **中文场景适配**: 优先 TT100K(中文)、CDSet(中国道路采集)、HyperLPR3(中文车牌) —— 比纯欧美数据集更贴我们的手机视频。
3. **CPU 推理速度**: YOLOv8n + ByteTrack + HyperLPR3 在 i7-10750H 上可跑, 但逐帧检测较重(可能 5-15 fps)。我们的视频本来就是 CPU 逐帧处理, 速度可接受, 不必实时。
4. **行人灯 vs 机动车灯**: 这是本项目特有难点(见 diagnosis 报告 E20)。通用灯检测能"找到所有灯", 但必须加**行人灯判别**(位置/图标形状/相位与占道车行为一致性)才能锁定正确信号 —— 这部分仍需要我们自己的逻辑, 外部模型不能直接解决。

---

## 参考链接(节选)

- 交通灯: LISA(AbhishekSinha-git/LISA-traffic-light-dataset-yolov8-Training) · Bosch(berktepebag) · TT100K · 数据集汇总(CSDN qq_33270279)
- 车牌: HyperLPR(gitee zeusees/HyperLPR) · LPRNet(sirius-ai/LPRNet_Pytorch) · Chinese-LPR-Transformer(sosopop) · CCPD/CRPD 数据集
- 斑马线: CDSet-3434(Zenodo 10.5281/zenodo.8289874) · CDNet(zhangzhengde0225/CDNet)
- 跟踪: ByteTrack(FoundationVision/ByteTrack) · BoxMOT(mikel-brostrom/Yolov5_DeepSort_Pytorch) · PaddleDetection
- 分割/综合: BDD100K(bdd-data.berkeley.edu) · Cityscapes · Mapillary Vistas · 语义分割数据集汇总(Aliyun)

*本报告基于 2026-07-12 的 5 组 WebSearch 结果整理。下一步建议从"阶段0"(斑马线CDSet + 跟踪BoxMOT)开始, 这两个不依赖我们有限的数据即可见效。*
