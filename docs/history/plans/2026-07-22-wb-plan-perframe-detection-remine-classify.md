# WB 实现计划: 逐帧检测 + 跟踪补缺 + 紧框裁图 + 分类

> 据 cc 方向 `docs/handoff/2026-07-22-cc-direction-perframe-detection-architecture.md` 起草。
> **状态: 计划稿, 待 cc review。cc 放行前 wb 不写任何 pipeline 代码(先计划后动红线)。**
> 前置: Diag3 v2 复核通过(cc 独立 bit-for-bit 确认); 架构已锁定。

---

## 0. 前置结论(已定, 不在本计划范围内)

- **Diag3 v2 复核通过**: 中心命中 64.3% / IoU≥0.3 57.1% / IoU≥0.5 35.7%, cc 亲自看叠框确认近中心红框=行人灯、05 漏检=YOLO 锁了车灯。
- **架构锁定**: 固定 prior=死路(相机运动); 裸 YOLO 裁框=不够(64% 命中缺口大); **定案=逐帧 YOLO 检测 + 时序跟踪补缺 + 紧框裁图 + 分类**。
- **小灯路线(本计划 §3, 代理测试已决)**: 提高有效分辨率(imgsz 1280)救回 08 类纯小灯; 05 类靠 GT 种子跟踪+模板兜底; m/x 真测留联网机复核, **不阻塞**本计划。

---

## 1. 关键代码现状盘点(重要: 非从零建)

读 `src/redlight/models/traffic_light.py` + `signal_candidates.py` 确认, **逐帧检测+分类的骨架已存在**, 本计划主要是「修/扩 + 补缺口」, 不是重写。

| 现有能力 | 位置 | 现状 | 本计划动作 |
|---|---|---|---|
| `observe(frame, yolo_light_boxes=None)` | traffic_light.py:140 | 已收 YOLO 框; 选框仍用「离 prior 最近」启发式 + prior 直采回退 | 改选灯逻辑: 用 GT 种子轨迹, 弃「离 prior 最近」 |
| `detect(frame, ..., yolo_light_boxes)` | :240 | 已调 `_update_tracks`/`_update_heads`/`_select_lit` | 复用, 接 tracked ped 框 |
| `_detect_ped` (M1 ped_classifier 路径) | :269 | **已是** `YOLO灯框 ∪ HSV亮斑 → classifier.classify(walk/stand/off) → global_recent 平滑` | 扩: 喂 tracked ped 框而非裸 YOLO |
| `build_candidates(yolo_boxes, hsv_boxes, w, h)` | signal_candidates.py:10 | YOLO∪HSV 并集去重 | 复用 |
| `_update_heads` / `_update_tracks` | :328 / :696 | 信号头/单灯轨迹雏形 | 扩为 ped 灯专用轨迹 + 补漏 |
| `_select_lit` / `_best_signal_head` | :391 / :504 | 选灯+持久门控 | 改: GT 种子优先, 弃 prior 距离 |
| `SignalStateClassifier.classify(roi)` | signal_state_classifier.py:99 | tiny CNN 3×48×48→3 | 复用(重挖后重训) |
| `datasets/light_location_gt.json` | — | 逐帧 `true_box_norm` (Jacob GT) | **作 ped 轨迹种子源**(不需新标注) |
| 公平测工具 | diag_yolo_vs_gt.py / diag_small_light_res.py | IoU vs 逐帧 GT, 去循环 | 复用为去循环评测基底 |

**缺口(本计划要补的)**:
1. 选灯靠「离 prior 最近」→ 必须换成 **GT 种子轨迹**(cc §2.1 核心)。
2. 无**跟踪补漏**(遮挡 03 / YOLO 漏帧靠外推插值)。
3. 小灯救援只 yolov8n@640 → 加 imgsz/模型档位(§3)。
4. 重挖仍走旧 prior ROI → 改 `classifier_retrain_v2/` 走 tracked 紧框。
5. 评测仍是 prior 自采样自证(坑)→ 加去循环评测(§4)。
6. 生产接线未真正喂 tracked ped 框。

---

## 1.5 命门: 生产 GT-free 选灯(cc 收口强调, M7 前必须想清楚)

> cc 一句话: "计划底子好, 但'生产怎么在没 GT 时认出行人灯'这条命门必须现在想清楚, 别等 M7 才撞墙。"

**问题**: 训练/评测态可用 GT 种子锁 ped 轨迹; 但**生产态无 GT**, 必须自动从 YOLO 多框(ped + 车灯)里认出行人灯轨迹。若不解决, M1–M6 全建在 GT 拐杖上, M7 接线撞墙(这正是"GT 种子把生产选灯糊弄过去"的风险)。

**GT 种子的两个角色必须分开**:
- **训练/评测态(合法)**: GT 种子用于 (a) 挖干净 crop、(b) 度量定位精度(IoU vs true_box_norm)、(c) 学 GT-free 判别的先验。
- **生产态(绝不能有 GT)**: 必须有一套 **GT-free 的"是不是行人灯"判别**代替种子。

**候选 GT-free 判别信号(从 GT 学先验, 生产用先验; 分层, 先便宜后昂贵)**:
- **L1 几何+外观先验(零新标注)**: 行人灯竖式长条(h/w 比高, 从 GT 的 `true_box_norm` wh 分布统计)、尺寸区间、crop 外观(分类器判 walk/stand 命中=ped 强信号; off 且非灯外观=车灯/杂)。
- **L2 轨迹时序(零新标注)**: 行人灯有红/绿切换 + 倒计时闪烁; 车灯(尾灯/信号)时序特征不同。轨迹级时序加权。
- **L3 学习式 ped-vs-vehicle 二分类头(需 Jacob 标注)**: 若 L1+L2 leave-some-out 不达标, 训轻量二分类(复用 SignalStateClassifier 骨架加 vehicle 类, 或独立头)。**这是 cc 预告的"可能需要 Jacob 再标一批 ped-vs-vehicle 样本"的触发点** — 等 L1/L2 leave-some-out 结果再定要不要上 L3。
- **位置先验(仅软特征, 不作硬锚)**: 已有 `yolo_cy_min` 顶部过滤车尾灯可留; 但单点 prior 已证死路, 位置**只能软加权、不能当硬锚**。

**护栏1 — leave-some-out 评测(防 GT 种子糊弄生产选灯, cc 硬要求)**:
- K 折留出: 每折留出若干视频**完全不给 GT 种子**, 用其余视频学的 GT-free 规则/模型选留出视频的 ped 轨迹, 度量 IoU vs `true_box_norm`。
- 这直接模拟生产 GT-free。**gate 只看 leave-some-out 分数, 不看 GT-in 分数**(GT-in 分数只作诊断哨兵)。
- M2 验收改为 **leave-some-out 分数**, 而非全量 GT-in 分数(见改后 M2)。

**护栏2 — 遮挡补位置不补状态(cc 硬要求)**:
- M3 遮挡外推**只补灯的位置**(供裁图/轨迹连续性); **状态一律判 unknown**。
- **绝不跨遮挡插值 walk/stand** — 遮挡中可能正好切换, 插值会造假信号。
- observe() 遮挡帧出 `obs=unknown/None`, 交下游时序稳定器处理(**不碰 `enforce_transition_limit`**)。

---

## 2. 模块拆分 + TDD 任务(cc review 后执行, 每阶段末 cc verify)

### M1 逐帧检测接入(修现有骨架)
- **输入**: 每帧 `frame` → YOLO(cls=9) + HSV 亮斑 → `build_candidates` 并集。
- **改动**: `observe`/`detect` 已收 `yolo_light_boxes`; 确认 `scan_video` 每帧产 YOLO 框并传入(插入点待 cc verify 时定位 `scan_video` 调用处)。
- **TDD**: `test_detect_yields_candidates` — 给定帧, 候选含 ped 灯位置(用 §4 去循环评测比对 GT)。
- **风险**: 低(骨架已在)。

### M2 GT 种子选灯 + 生产 GT-free 选灯(核心难点 #1, 含命门 §1.5)
**必须分两条路径实现, 不能只做训练态那条(否则撞 §1.5 命门)**:

**M2a 训练/评测态(GT 种子)**:
- **机制**: 新模块 `ped_track_seeder` — 读 `light_location_gt.json` 的 `true_box_norm` 作种子(02/10 多灯, Jacob 标 true 的那条即种子); 把每帧 YOLO/HSV 检测用 **IoU + 中心距** 关联到种子轨迹; 锁定 ped 轨迹与车灯分离。
- **用途**: 仅挖 crop(M5) + 度量精度(M6) + 学 M2b 先验。**不进生产。**

**M2b 生产 GT-free 选灯(命门, cc 死盯)**:
- **机制**: 新模块 `ped_selector` — 无 GT 时按 §1.5 分层判别选 ped 轨迹: L1 几何+外观先验(wh 比 + crop 外观分类) → L2 轨迹时序(切换+倒计时) → (若不达标)L3 学习式二分类。位置仅软加权。
- **用途**: **生产唯一选灯路径**(M7 接线用这条, 不是 M2a)。

**验收(护栏1, cc 硬要求)**: M2b 用 **leave-some-out** — K 折留出视频**完全不给 GT 种子**, 度量选出的 ped 轨迹 IoU vs `true_box_norm`。**gate 看 leave-some-out 分数**(§3 的 85.7% 中心命中是 GT-in 上界参考, leave-some-out 达标线 M6 首跑后与 cc 共定)。GT-in 分数仅诊断哨兵。
- **TDD**: `test_seeder_picks_ped_not_vehicle`(M2a: 05 类多框选 ped 非车灯); `test_selector_gtfree_leave_one_out`(M2b: 留出视频无 GT 种子仍选对 ped 轨迹); `test_selector_never_reads_gt_in_production`(断言生产路径不触 GT 文件)。
- **风险**: 高(命门; M2b 是架构成败关键; L1/L2 若 leave-some-out 不达标需上 L3=Jacob 标 ped-vs-vehicle 样本)。

### M3 时序跟踪补缺(核心难点 #2, 含护栏2)
- **机制**: 把 M2 关联的逐帧检测连成 ped 灯轨迹; YOLO 漏帧/遮挡(03 大巴)用**轨迹外推/插值**补**位置**; 输出每帧一个 tracked ped 紧框(供裁图连续性)。
- **护栏2(cc 硬要求, 补位置不补状态)**: 遮挡外推**只补灯的位置**, **状态一律判 `unknown`**; **绝不跨遮挡插值 walk/stand**(遮挡中可能正好切换, 插值造假)。observe() 遮挡帧出 `obs=unknown/None`, 交下游时序稳定器(**不碰 `enforce_transition_limit`**)。
- **复用**: 扩 `_update_heads`/`_update_tracks` 为 ped 专用 + 补漏逻辑; 轻量跟踪器(Bytetrack 风格或自写 Kalman-lite, 先最简线性外推, 够用再升级)。
- **TDD**: `test_track_fills_occlusion_gap_position_only` — 合成遮挡序列, 断言补漏**位置**框 IoU vs GT 达标; `test_occlusion_state_is_unknown` — 断言遮挡外推帧状态=unknown、不出 walk/stand。
- **风险**: 中(补漏位置过拟合会漂, GT 校验封顶; 状态 unknown 硬约束防造假)。

### M4 小灯救援(§3 决策落地)
- **默认**: 推理 `imgsz=1280`(cheap, +21.4pp 中心命中); 可选 `yolov8m/x`(联网机复核后升档)。
- **05 类兜底**: M2b 生产选灯 + M3 跟踪仍跟丢时, 用**模板匹配**(从命中帧建 ped 模板, 邻帧匹配)兜底; 联网复核时**优先验 m/x 能否直接救 05 车灯锁定**(cc 指定优先级)。
- **TDD**: `test_small_light_imgsz1280_recovers_08` — 08 两帧中心命中(代理已证); `test_05_template_fallback` — 05 模板兜底命中。
- **风险**: 低(分辨率是 cheap 改动; 05 模板兜底为新增但范围小)。

### M5 紧框裁图 + 重挖 `classifier_retrain_v2/`
- **裁图**: 从 M3 的 tracked ped **紧框**裁(略放大 margin 框住灯), 天然随运动、框住灯。
- **重挖**: 新目录 `datasets/classifier_retrain_v2/`, **全 11 视频**重挖(符合 Phase C 全 11); 旧 `classifier_retrain/`(prior ROI)不删, 作对照。
- **抽检**: 重挖后**必重跑画廊抽检**(Jacob 审), 沿用 `make_classifier_retrain_gallery.py` 改造输出到 v2。
- **TDD**: `test_remine_v2_coverage` — v2 crop 覆盖标注帧且 GT 一致; `test_no_prior_skew` — v2 crop 框 vs true_box_norm IoU 分布达标(去 prior skew)。
- **红线**: 绝不擅自重挖 — 仅 cc 放行后执行; 走 v2 目录。

### M6 去循环评测(硬要求, cc §2.5)
- **新脚本 `eval_location_deloop.py`**: 每视频跑 M1–M3 → 每帧 tracked ped 框; 对标注帧算 **IoU(tracked, true_box_norm) + 中心距**; 报 P(中心命中)/IoU≥0.3/≥0.5 + mean。**独立**于 prior。
- **状态评测**: tracked 紧框裁图 → `classifier.classify` → walk/stand/off vs **独立 state GT**(即 classifier_retrain 语义标签, 非 prior 自采)。
- **禁用**: "在 prior 处采样再比 GT" 的自证指标(Diag2 坑)。旧准确率仅作相对哨兵。
- **TDD**: `test_eval_deloop_matches_known` — 在 28 标注帧上复现代理测试数字(去循环口径一致); `test_no_prior_self_proof` — 断言评测不读 light_priors.json 的 prior 位置。
- **风险**: 低(工具已大半存在)。

### M7 生产接线(不回退, cc §2.6)
- **改动**: `scan_video` 每帧跑 YOLO→tracked ped 框→传 `observe/detect`; 逐帧检测定位**替代/增强**固定 prior。
- **不回退保障**: 旧 prior 路径**保留可回滚**; A/B 对照(旧 prior vs 新检测)在生产子集上比召回/误绿。
- **gate**: 不过 4 gate(见下)不接线。
- **TDD**: `test_production_no_regression` — 旧 prior 路径行为不变; `test_new_path_ab_above_baseline`。
- **风险**: 中(动生产路径, 必须 A/B + 回滚)。

### M8 工程红线(贯穿)
- scoped git、署名(cc + wb co-author)、trunk main、TDD 先、不碰 `enforce_transition_limit`、Phase C 全 11 重训、gate 不过不接线。

---

## 3. 小灯路线决策(代理测试结论, 已决)

**测试**: 沙箱下载 yolov8m/x 权重被 CDN 502 拦截; 改用 **yolov8n @ imgsz=1280** 重跑 Diag3 v2 公平测作代理(小灯漏检主因=有效分辨率不足, 提高推理分辨率代理"更大模型/更高分辨率"路线)。脚本 `diag_small_light_res.py`, 输出 `data/output/diag_small_light_imgsz1280.json` + 叠框。

| 指标 | n@640 (v2 基线) | n@1280 (代理) | Δ |
|---|---|---|---|
| 中心命中(<0.06) | 64.3% (18/28) | **85.7% (24/28)** | **+21.4pp** |
| IoU≥0.3 | 57.1% | 57.1% | 0 |
| IoU≥0.5 | 35.7% | 35.7% | 0 |
| mean IoU / ctrD | — / — | 0.353 / 0.079 | — |

**弱视频明细 (imgsz=1280)**:
- **08**(灯宽 0.016–0.018): 两帧**全救回** — fi=0 ctrD=0.013(IoU=0.30), fi=504 ctrD=0.008(**IoU=0.53**)。纯小灯、分辨率是绑定点 → 路线成立。
- **05**(灯宽 0.021): **仍锁车灯** — fi=0 出 2 框都是车灯(ctrD=0.549), 没框到 ped。分辨率不够, 需 M2 种子+M3 跟踪+模板兜底。
- **03**(大巴遮挡): 遮挡帧 fi=2793 **仍 0 框**(物理遮挡, 分辨率救不了) → 必须 M3 跟踪补漏。

**决策(写进 M4)**:
1. **默认推理 `imgsz=1280`** — cheap、零新权重、整体 +21.4pp 中心命中, 直接提升跟踪种子质量。
2. **08 类纯小灯**: imgsz 1280 已救回, 不额外处理。
3. **05 类**: 靠 **M2 GT 种子 + M3 跟踪 + 模板兜底**(YOLO 持续锁车灯, 分辨率 alone 无效)。
4. **03 类遮挡**: 靠 **M3 跟踪补漏**(外推/插值)。
5. **yolov8m/x 真测**: 留**联网机复核**(更多 capacity 可能再帮 05), **不阻塞**本计划 —— 即便 m/x 帮 05, M2/M3 架构仍必需(03 遮挡 + YOLO 漏帧)。

---

## 4. 去循环评测规格(cc §2.5, 硬要求)

- **定位准确率** = `IoU(tracked_ped_box, true_box_norm)` 对所有标注帧(位置级, 独立); 报 P(中心命中<0.06)/IoU≥0.3/≥0.5/mean。
- **状态准确率** = `classify(crop)` → walk/stand/off vs **独立 state GT**(classifier_retrain 语义标签)。
- **禁止**: 在 prior 位置采样再比 GT(自证循环)。旧 prior 口径仅作相对哨兵。
- **Gate(重定义, 待 cc 与 wb 共定阈值)**: 新定位+状态联合 gate 取代 Phase B 4 gate 的定位部分; 状态部分沿用 walk/stand/off 召回/拒识。gate 不过不接线。

---

## 5. 生产不回退 + 红线(cc §5)

- 旧 `light_priors.json` prior 路径**保留可回滚**; 新检测路径 A/B 对照。
- 不碰 `enforce_transition_limit`(在 `intermediate_state.py:29`, 全局稳定器, 与定位解耦)。
- 绝不擅自重挖(仅 cc 放行后走 `classifier_retrain_v2/`); Phase C 全 11 重训; scoped git; 署名; trunk main; TDD 先。

---

## 6. 里程碑 / 执行顺序(cc review 后; 每阶段末 cc verify)

1. **阶段1 — M1 + M2a + M2b**(检测接入 + GT 种子选灯 + **生产 GT-free 选灯, 带 leave-some-out 评测**): 先能"从 YOLO 多框里认出 ped 轨迹", **且证明无 GT 时也能认**(命门 §1.5, cc 死盯)。
2. **阶段2 — M3**(时序跟踪补缺, 护栏2: 补位置不补状态): 遮挡/漏帧补位置, 状态判 unknown。
3. **阶段3 — M4+M5**(小灯 imgsz + 重挖 `classifier_retrain_v2/`): 出干净训练集。
4. **阶段4 — M6**(去循环评测): 定位+状态联合 gate 门控。
5. **阶段5 — M7**(生产接线, A/B 不回退验证): 接线。
- 每阶段 cc verify 通过才进下一阶段; gate 不过回退到对应阶段。

---

## 7. 拍板结论(cc 2026-07-22 已答)

1. **m/x 真测不阻塞** ✅ — 代理已去险; 联网复核时**优先验 m/x 能否救 05 车灯锁定**。
2. **`classifier_retrain_v2` 全 11 视频重挖** ✅ — 旧目录留对照。
3. **GT 种子复用 `light_location_gt.json` 的 `true_box_norm`, 不需新标注** ✅ — **但受命门 §1.5 + 两护栏约束**: GT 种子仅训练/评测态(M2a); 生产走 M2b GT-free(命门); 若 M2b 的 L1/L2 leave-some-out 不达标, 才需 Jacob 补标 ped-vs-vehicle 样本上 L3。
4. **新 gate 阈值 M6 首跑后与 cc 共定** ✅ — **gate 看 M2b 的 leave-some-out 分数, 不看 GT-in**。

---

## 8. 与 git 侧任务的衔接

- `datasets/classifier_retrain_negatives/`(55 张无效负例)建议 gitignore(待 Jacob 确认, 见 2026-07-22 待办); 新 `classifier_retrain_v2/` 的 crop 同旧策略(派生产物不入库, 仅元数据入库)。
- 本计划产生的新脚本(`diag_small_light_res.py` 已落; `eval_location_deloop.py` 待写)入 git, scoped。

**下一步**: 等 cc review 补充后计划(命门 §1.5 + M2b + 两护栏) → 放行后从阶段1(M1 + M2a + M2b, 带 leave-some-out)起, 每阶段末 cc verify。cc 死盯: 生产选灯别被 GT 种子糊弄过去。
