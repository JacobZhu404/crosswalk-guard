# 方案:统一 canonical 灯态 GT + 预填确认式标注画廊

> 出自 cc。据 Jacob 拍板(2026-07-28):按秒抽帧画廊、框出每帧所有红绿灯+标灯色、作唯一真值;并预填"系统检测 + Jacob 旧标注",对了确认、错了才改。
> schema/密度已锁:**颜色+类型+governing**、**每 2s 一帧(~350 帧,全 11 视频)**。

## 0. 为什么要它(解决已确诊的病)
误绿调查暴露两病:①两套 GT 打架(段级 `events.csv` light_state + 稀疏 28 帧 `light_location_gt`);②"哪盏是管这条斑马线的行人灯"分不清(05:信号灯背面/透公交车窗的绿灯/远处红;06:绿树叶)。**一套稠密逐帧 box+color+type+governing GT 同时修好:去矛盾、能真评测(逐帧非稀疏)、给判别器训练负样本(干扰)。**

## 1. Canonical GT schema(v2,唯一真值)
每帧(每 2s 抽一帧,全 11 视频)一条记录:
```
{ video, source_fi, t,
  no_light: bool,                         # 整帧无任何交通灯
  boxes: [ {
     box_norm: [x1,y1,x2,y2],             # 归一化
     color:  red | green | off | countdown | unclear,
     type:   pedestrian | vehicle | distractor,   # distractor=背面/反射/树叶/其他
     governing: bool                      # 是否=管这条斑马线的那盏行人灯
  }, ... ] }
```
- **governing 约束**:每帧**至多一个** `governing=true`(且必是 `type=pedestrian`)。全帧无可见行人灯→无 governing(或 no_light)。
- **派生真值**:`gt_walk(frame) = 存在 governing 框 且 其 color==green`;governing 灯不可见/unclear → 该帧灯态 UNKNOWN(评测排除,承误绿测量三态纪律)。
- **多灯视频(02/10)**:靠 governing 标记指认唯一那盏,不再靠猜。

## 2. 预填(把"从零画"降成"点确认")
每帧预先画好三类框,Jacob 增删改:
- **系统检测框(橙)**:YOLO(cls9)∪HSV 候选。每框带**猜测** color(HSV 主色)+ **猜测** type(L3 P(ped)>阈→pedestrian,否则 vehicle/distractor)。
- **Jacob 旧标注框(蓝)**:`light_location_gt.json` 的 `true_box_norm` 贴到最近帧,预设 type=pedestrian、governing=true、color=旧标注色。
- **确认流**:每帧一个 [✓ 全确认] 按钮(采纳所有预填);或逐框改 color/type/governing 下拉、拖新框、删错框、勾"帧内无灯"。

## 3. 工具两件(新文件,不动生产)
- **A. 数据准备脚本** `scripts/build_light_gt_gallery.py`(Python):
  - 按 source fps 每 2s 抽帧 → `data/output/light_gt_gallery/frames/{video}_{fi}.jpg`。
  - 每帧跑 YOLO∪HSV 候选 + 猜 color/type(复用 `collect_candidates`/L3/`_color_state`);贴 `light_location_gt` 旧框。
  - 产出 `frames_prefill.json`(FRAMES 数组:img 路径 + 预填 boxes)。TDD:抽帧数=⌈dur/2⌉、prefill 贴框正确。
- **B. 标注画廊** `data/output/light_gt_gallery/annotate.html`(扩 `annotate_lightbox.html`):
  - 多框 canvas + 逐框下拉(color/type/governing)+ 增删框 + no_light;**localStorage 断点续存**(不怕关页);导出 canonical GT JSON。
- **C. 入库脚本** `scripts/ingest_light_gt.py`:校验(每帧≤1 governing、type/color 合法、box 范围)→ 原子写 `datasets/gt/light_canonical_gt.json`(schema 版本号)。TDD:governing 唯一性、非法值拒绝。

## 4. 取代关系(入库后)
- 误绿/定位评测**改用 `light_canonical_gt.json`**(逐帧 governing color)。
- `events.csv` light_state + 稀疏 `light_location_gt` → 降级为历史对照(不删,`measure_falsegreen.py` 改指 canonical)。
- 成因去混淆用**逐帧 governing 框**(不再静态 true_center)→ 彻底解 Diag3 混淆。

## 5. 红线 / 纪律
- 工具只读视频抽帧、只读现有模型出预填;**不接线、不动生产权重/prior**。
- 预填仅"建议",**真值以 Jacob 确认为准**(系统猜测错了他改)。
- scoped git、TDD 先(A/C 脚本)、署名、trunk main。画廊帧图/HTML 走 `data/output/`(不进 git,体积大);**canonical GT JSON 入库**。

## 6. 交付顺序
1. A 抽帧+预填(先 smoke 1-2 视频给 Jacob 看预填质量)→ 2. B 画廊(Jacob 试标 1 视频验流程)→ 3. 全量标 → 4. C 入库 → 5. 改 `measure_falsegreen.py` 指 canonical,重算误绿(真逐帧真值)。
