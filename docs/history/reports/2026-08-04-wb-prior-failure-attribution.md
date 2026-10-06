# 09/06 绿帧漏绿失败归因诊断（wb，cc plan-gate CONDITIONAL 重诊断）

> 状态：**只读诊断，未写生产码**。cc 把方案 2da20f5 打回，要我先出失败归因再放行进 code。
> 两份产物：`scripts/diag_prior_motion.py`（漂移，已修多灯 bug）、`scripts/diag_0906_green_failure.py`（五分归因）。
> 本诊断的结论**反转了原方案的核心假设**——详见 §3。

## 1. 相机运动（护栏#1，漂移脚本修复版）

cc 硬伤①：旧 `diag_prior_motion.py` 把每视频 **ALL governing 框展平取 range**，把同帧多灯间距与相机运动混一起（09 虚高 654px）。
修复：按 IoU≥0.5 跨帧关联成 **tracklet**，报每盏灯漂移；并单列**单灯帧**（该帧仅 1 个 governing 框）的漂移排除多灯干扰。

| video | tracklet数 | tracklet最大漂移 | 单灯帧 | 单灯帧漂移 | 判定 |
|---|---|---|---|---|---|
| 违章01 | 21 | 19px | 6 | 74px | 近似固定 |
| 违章02 | 99 | 52px | 18 | **527px** | 动相机✗ |
| 违章03 | 24 | 10px | 11 | **327px** | 动相机✗ |
| 违章04 | 37 | 31px | 3 | 130px | 近似固定 |
| 违章05 | 32 | 54px | 8 | **537px** | 动相机✗ |
| 违章06 | 60 | 49px | **1** | 测不出(单灯帧<2) | 近似固定 |
| 违章07 | 60 | 136px | 0 | 测不出 | 近似固定 |
| 违章08 | 40 | 4px | 5 | **281px** | 动相机✗ |
| 违章09 | 110 | 12px | 7 | **181px** | 动相机✗ |
| 违章10 | 11 | 0px | 3 | **198px** | 动相机✗ |
| 违章11 | 21 | 25px | 0 | 测不出 | 近似固定 |

**结论（钉正相机运动前提）**：
- 09 单灯帧漂移 **181px**（cc 估 ~172，同量级，**moderate 动相机**）；06 仅 1 个单灯帧 → **测不出**（cc 说"只有1个单灯帧"✓）。
- 全 11 视频是**混合运动**：02/03/05/08/09/10 真动（>150px），01/04 近似固定，06/07/11 单灯帧不足难定。
- **「11 视频全固定机位」和「全动相机 654px」都错**，cc 此项更正成立。
- 含义：对 09 这类动相机视频，固定点 re-center 仍死路（灯会移出固定锚）。但——见 §3，这已不是 09 漏绿的主因。

## 2. 09/06 绿帧漏绿五分归因（核心）

对 09(34)/06(15) 全部 gt_green 帧，同时跑两条子系统：
- **A) selection 路径**：`select_gtfree(cands, PED_PRIOR)` + `color_state(crop)` —— 即 selection-quality 报告漏绿数字的真实来源。
- **B) observe 路径**：`det.observe(frame, yolo_boxes)` 带 `signal_prior`（light_location_gt 的 prior 中心点）—— 即 prior 重定位项目要改的信号态路径。

候选池 = YOLO cls=9(conf=0.05) ∪ HSV，与 canonical 同口径。`recall` = 任一候选 IoU≥0.3 覆盖真绿框。`obs_ok` = observe 出 `green`（observe 本职是出颜色，YOLO 框比 canonical 紧框大→IoU 偏低不卡）。

**09 主因分布（34 绿帧）**：
| 类 | 帧数 | 说明 |
|---|---|---|
| ⑤ selector 选错框 | **28** | 候选池有绿框(recall OK)，`select_gtfree` 因多灯+YOLO优先挑了**非绿**候选 |
| ① 候选召回洞 | 2 | 候选池无任何框覆盖真绿（YOLO+HSV 都没检到） |
| ⑥ 已识别绿(不漏) | 4 | — |

**06 主因分布（15 绿帧）**：
| 类 | 帧数 | 说明 |
|---|---|---|
| ⑤ selector 选错框 | 5 | 同上 |
| ④ HSV 绿抑制 | 1 | 真绿区域 `color_state` 压成非绿 |
| ① 候选召回洞 | 2 | 候选池缺口 |
| ⑥ 已识别绿(不漏) | 7 | — |

> 注：本诊断 selection 用**单帧** `select_gtfree`（无 L2 时序），与报告的多帧 L2 口径略有差异，故绝对帧数不严格等于报告 09 漏绿=22；但**主因类别（⑤ 占绝对多数）稳健**。

## 3. ⚠️ 关键反转：prior 重定位修不了实测漏绿

原方案 2da20f5 假设「09 prior 偏框 → crop 错位 → 系统性选不中真绿」，故要把 `observe()` 的 crop 锚点从固定 prior 改逐帧。本诊断证明该假设**错在两个层面**：

1. **observe() 逐帧 YOLO 本就工作**。09 绿帧里 observe 几乎全出 `green`（如 fi=472 obs_iou=0.829、fi=531 obs_iou=0.993、fi=826 obs_iou=0.996…），即 cc 担心的"固定 prior crop 错位"在 observe 上**不存在**——既有逐帧路径已正确找到绿。
2. **漏绿发生在 `select_gtfree`（选择器排序），不在信号 crop**。09 有 28/34 帧：候选池里明明有 IoU=0.99 覆盖真绿的框（recall OK），但 `select_gtfree` 挑了**另一个非绿候选**（常是另一盏 red/off 的 governing 灯，或车辆灯）。`select_gtfree` 在「有 YOLO 候选时只在 YOLO 里选」+「按 L1 几何+0.15 YOLO 加成+时序」排序，遇 3 盏 governing 灯时几何分高的非绿灯胜出 → 选中框 `color_state` 非绿 → 漏绿。

**推论**：把 `observe()` 的 crop 锚点改逐帧，**对 selection-quality 报告的 09/06 漏绿数字零影响**——因为 observe 已经对了，漏绿在 selector。原方案方向（改 observe crop）与要修的指标（selector 漏绿）是**两套子系统**。

## 4. 各类真因 → 修法/验收（cc 要"定性清楚才能写码"）

| 类 | 真因 | 修法方向 | 验收（位置级 GT） |
|---|---|---|---|
| ① 召回洞 | YOLO cls=9 / HSV 都没检到该绿灯（06 fi=767/826 典型） | 提 YOLO 召回 / 补 HSV 兜底阈值 | 绿帧候选池出现 IoU≥0.3 覆盖真绿 |
| ④ HSV 抑制 | 真绿区域 `color_state` 因 sat/val 阈值压成非绿（06 fi=59 真绿3/3但1盏判非绿） | 放宽 `color_state` 阈值 / 用判别器色 | 真绿区域判为 green |
| **⑤ selector 排序** | **多灯时按几何挑了非绿灯**（09 主因 28/34） | **selector 排序引入颜色/有效灯信号**（如用 observe 逐帧灯态或 governing 判别器分参与排名，而非纯几何）；或显式"多 governing 灯时优先绿" | **选中候选=绿灯**（sel_iou≥0.3 且 color=green） |
| ②/③ prior crop | observe 实测已工作，**非主因** | 降级/暂不修（除非为 violation_engine 信号态另立指标） | — |

**⑤ 才是 09/06 漏绿的决定性真因，也是唯一能移动漏绿数字的修法杠杆。**

## 5. 对 cc 两项待拍板问题的回应

- **扩全 11 视频**：漂移脚本的多灯 bug 已修，全 11 视频漂移已重测（§1 表）。结论：混合运动，非全动/全固定。
- **branch `wb-prior-perframe`**：**按 cc 要求，诊断产物过 gate 前不建 worktree**。且本诊断表明该分支当前 scope（改 observe crop）**不对应要修的指标**——建议 cc 重 scope 为 selector 排序（⑤）后再建分支写码。

## 6. 给 cc/Jacob 的判定建议

原「09 prior 重定位是决定性关卡」的判定，本诊断**不支持**用于 selection-quality 漏绿：
- 若目标是压 selection-quality 的 09/06 漏绿 → 改修 **selector 多灯/颜色排序（⑤）**，prior 重定位无关。
- 若 prior 重定位仍想做，它的正确标的物是 **`observe()`→violation_engine 的信号态路径**（另一消费者），需另立指标（如 observe 在绿帧的 state 准确率），且 observe 当前已工作，收益待证。

建议 cc 重发 plan-gate：把 09/06 漏绿的解法从"observe crop 重定位"改为"selector 颜色/多灯感知排序"，我再出对应实现方案。

---
*署名：wb(执行诊断) / cc(复核+plan-gate)。脚本均只读、不碰生产码、权重不入库。*
