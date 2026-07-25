# Handoff 交接快照 [PHASE-1 红绿灯识别 / 07 修复]

> 生成时间: 2026-07-12 11:56 (GMT+8)
> 项目: D:\redlight-crosswalk-violation\ (红灯压斑马线检测)
> 当前阶段: **Phase 1 仅做红绿灯识别** (违章判定/占用斑马线 延后)
> 负责人: Senior Developer (高级开发工程师)

---

## 1. 核心任务目标

**原始需求 (用户 2026-07-12 明确分阶段):**
- **第一阶段 ONLY = 红绿灯识别**: 保证能**准确找到红绿灯**、**输出红绿灯灯色**。
- 违章判定、车是否占斑马线 → 放到后面阶段, 本期不做。
- 具体触发: 用户指出 "07 前 43s 就是绿灯, 但你识别出来是红灯, 说明有很大问题" → 要求把 07 识别到的红绿灯框出来 + 识别结果 + 视频放一起, 供肉眼核对 "是红绿灯找错了还是颜色判错了"。

**验收标准 (本期):**
- 清楚可见的信号灯 → 识别正确 + 置信度高。
- 公交玻璃反射 / 被遮挡等难例 → 输出 `unknown` / 交人复核 (`review`), **不要强行判错, 更不要为这类难例调阈值 (用户原话: 优先把清楚可见的情况识别到)**。

**约束规则:**
- 不引入 VLM/LLM 做灯色判定 (符合 avoidance-of-VLM 原则, COT 留给后面阶段)。
- CPU-only (Win10, 无 GPU), 推理慢 (全帧检测 ~10-15 fps, 07 ~1350 帧 ≈ 2-3 min/遍)。
- 单元测用 `scripts/run_tl_tests.py` (pytest 因 cv2 导入慢会 hang, 勿用)。

---

## 2. 已完成工作清单

### 2.1 检测器重写 (src/redlight/models/traffic_light.py, 当前版本标注 `color-v7-head`)
- 候选提取: 饱和度主导掩膜 `S>=sat_min(130) & V>=value_floor(60)` (不依赖整体曝光, 比 v5 固定 V 鲁棒, 物理依据 E20: 信号灯 = 高饱和纯色发光体)。
- 选灯判定重写为 **信号头聚类 + 面积选灯 + 持久门控**:
  - `_cluster_heads`: 把空间邻近候选聚成"信号头", 同杆红+绿 → 一个头 (解决 v6 同杆红绿合并后红灯赢的 07 原始 bug)。
  - `_select_lit`: 取 `frames_seen>=need` 且 `total>=head_area_floor` 的持久头中 **面积最大者** 的主导色 → 亮绿灯面积 >> 同杆暗红反射 → 绿灯胜。
  - 持久门控 (`min_persist_frames`/`track_persist_min`): 移动红车/瞬态反光聚不成持久头 → 排除。
- **已废弃的错误方向 (勿复用)**: `frac_v>=0.30` "是否真点亮" 门控 —— 07 灯泡饱和但不削波到 V=255, 该门控会把正确绿灯排除 (已实证失败)。

### 2.2 07 诊断结果 (data/output/diag_07.log, 实测逐秒)
| 时段 | 判定 | 说明 |
|------|------|------|
| 0–23s | **green** | ✅ 符合 GT (visible/green) |
| 24–27s | **red** | ⚠️ 红闪 (见第 5 节阻塞点) |
| 28–42s | **green** | ✅ 符合 GT |
| 43–44s | **red** | ⚠️ 红闪 (同上) |

`state` 列 = 窗判决 (`global_recent` 最近 `window=24` 帧多数决), 不是单帧值。原始 bug (全红) 已修复, 但仍有瞬态红簇翻盘 (见 5)。

### 2.3 可视化交付物 (用户要求 "框框+识别结果+视频")
- `scripts/draw_light_boxes.py`: 画候选框(绿/红) + 持久信号头(标 G/R+frames_seen, 实环=真发射/细环=疑似反射) + 选中头黄圈 "READING" + 顶部文字 `t / state / reason / sel=color(area) / tracks`。
- 已生成: **`data/output/annotated/违章07_lightboxes.mp4`** (60 MB, 2026-07-12 11:56, 已用当前正确代码重生成) + `data/output/annotated/shots/` 关键帧。
- **注意**: 这是本次 Phase-1 的核心交付物, 待用户肉眼核对 "找对灯 + 颜色对"。

### 2.4 评测基建 (scripts/eval_light_all.py)
- 新增 **visible-only 指标** (`vis_acc` / `visF1` / `vis_n`): 仅对 `light_evidence=visible` 段算准确率是 Phase-1 主指标, 不混进难例。
- 已尊重 `light_evidence`: `inferred`/`occluded` 段 → 期望 `unknown` → 计 `review_cov` (交人复核覆盖率), 不算错。
- CSV 输出加了 visible 列。

### 2.5 测试
- `tests/unit/test_traffic_light.py`: **10/10 通过** (run_tl_tests.py)。覆盖: 稳定绿/红、闪烁=flashing、间歇绿不误判闪、移动红车被拒、暗绿灯(V=110)识别、琥珀当红、瞬态反射=unknown、无信号=unknown、低饱和绿(草地 S~75)拒绝。

### 2.6 GT 治理 (前期已完成, 本期复用)
- `datasets/gt/events.csv` (26 段, v2 格式: `video,start_s,end_s,light_state,light_evidence,violating_plates,other_plates,is_violation,note`), 07: 0-43s green/visible。
- `datasets/gt/videos.csv` (负例列表: 01,10)。
- `docs/plans/2026-07-12-gt-format-spec.md` (GT 格式权威)。
- `scripts/validate_gt.py` 校验 0 错 0 警。

### 2.7 提交记录
- **本期未提交**。最后 commit = `89ecd4c` (docs: 落盘设计需求 v2, 2026-07-12)。见第 3 节。

---

## 3. 当前 Git 环境状态

- **分支**: `master`
- **最后 commit**: `89ecd4c` docs: 落盘设计需求 v2 权威规格
- **未提交改动 (M, 16 个跟踪文件)**:
  `CHANGELOG.md, README.md, configs/config.yaml, data/input/README.txt,
  docs/plans/2026-07-10-architecture-redesign-design.md, scripts/diag_signal_timeline.py,
  scripts/download_models.py, scripts/run_video.py, src/redlight/app/cli.py,
  src/redlight/infrastructure/geometry.py, src/redlight/models/crosswalk.py,
  src/redlight/models/plate.py, src/redlight/models/traffic_light.py,
  src/redlight/pipeline/violation_engine.py, tests/unit/test_traffic_light.py,
  tests/unit/test_violation_engine_v2.py`
- **未跟踪 (??)**: `datasets/gt/`(events.csv/videos.csv/violation_events/), `docs/plans/2026-07-12-gt-format-spec.md`, `input_video/`, 以及大量 `scripts/*.py` (diag/draw/eval/analyze/debug 等诊断脚本) 与 `scripts/run_tl_tests.py`。
- **stash**: 未检查 (本次无 stash 操作)。
- **建议**: 脏树很大 (含前期跨walk/v10 等未提交改动)。提交前需与用户确认范围, 建议分次: ①检测器+单测 ②GT+文档 ③诊断脚本。用户规则: 重大修改须 commit。

---

## 4. 中间产物 (路径)

| 产物 | 路径 | 状态 |
|------|------|------|
| 07 标注视频 (核心交付) | `D:\redlight-crosswalk-violation\data\output\annotated\违章07_lightboxes.mp4` | ✅ 60MB, 已生成待核对 |
| 关键帧截图 | `D:\redlight-crosswalk-violation\data\output\annotated\shots\` | ✅ |
| 07 逐秒诊断表 | `D:\redlight-crosswalk-violation\data\output\diag_07.log` | ✅ 见第 2.2 |
| 视频生成日志 | `D:\redlight-crosswalk-violation\data\output\gen_video_07.log` | ⚠️ 仅 1 行 (空) |
| 检测器源码 | `D:\redlight-crosswalk-violation\src\redlight\models\traffic_light.py` | ✅ `color-v7-head` |
| 可视化脚本 | `D:\redlight-crosswalk-violation\scripts\draw_light_boxes.py` | ✅ |
| 诊断脚本 | `D:\redlight-crosswalk-violation\scripts\diag_07_light.py`, `dump_candidates.py` | ✅ |
| 评测脚本 | `D:\redlight-crosswalk-violation\scripts\eval_light_all.py` | ✅ (含 visible-only) |
| 单测运行器 | `D:\redlight-crosswalk-violation\scripts\run_tl_tests.py` | ✅ 10/10 |
| 配置 | `D:\redlight-crosswalk-violation\configs\config.yaml` (`traffic_light:` 段) | ✅ |

**关键调参旋钮 (configs/config.yaml 的 `traffic_light:`):** `sat_min=130, value_floor=60, min_area_px=20, max_area_ratio=0.008, max_aspect_ratio=3.5, color_s_min=22, match_radius_ratio=0.06, min_persist_frames=5, track_persist_min=0.08, signal_cy_cutoff=0.6, flicker_toggle_count=4, window=24, head_dist, head_area_floor`。

---

## 5. 当前阻塞点 / 未解决问题

### 5.1 【主要】07 在 24–27s 与 43–44s 出现红闪 (窗判决翻红)
- **现象**: 这两段 GT 是 green, 但检测器窗判决 = red。
- **根因**: `_select_lit` 当前按 **当前帧面积** `max(cands, key=lambda x: x["total"])` 选头。在 24s 附近 `r_spots` 突增到 22 (红车/公交/反射等一团红候选), 其红头 `total` 瞬时超过持久绿灯头 → 单帧 obs=red; 该红持续数帧 → 24 帧窗内 rr≥0.6 → 翻红。
- **为什么不是 "发射强度求和" 能解**: 诊断里 `r_emis`(红像素发射和) 常 >> `g_emis`, 因为 `r_spots` 含大量场景红 (车/标志/反射); 按原始 emission 求和会反过来淹没正确绿灯。所以选灯必须用 **聚类后的持久头面积**, 不能用全场景红像素和。
- **修复方向 (下一步 6.1)**: `_select_lit` 改按 **历史峰值 `max_total`** 或 **`frames_seen * total` 加权** 选头, 使一个看了 700+ 帧、历史峰值 ~740 的持久绿灯头, 不被只看 5 帧、total~237 的瞬态红簇翻盘。

### 5.2 未做: 全量校准 / 正式跑分
- `eval_light_all.py` 已就绪但 **未对 11 视频正式跑** (只手工看了 07 的 diag)。
- 未做 02/03/04 标定 T/N, 未扫全部 11 (grill-me 决议是分阶段: 02/03/04 验证 → 扫 11)。

### 5.3 用户待确认 (需看视频)
- 07 帧里有 **至少 2 个信号状物体**: Head A (~0.56,0.20 绿灯泡+暗红反射) 与 Head B (~0.83,0.47 红绿交替真发射)。当前锁 Head A 报绿。**需用户看 `违章07_lightboxes.mp4` 确认哪个是行人信号灯、颜色对不对**。

---

## 6. 下一步执行顺序 (优先步骤)

1. **【优先】修 5.1 红闪**: 改 `_select_lit` 选头加权 (用 `max_total` 或 `frames_seen*total`), 让持久绿灯头抗瞬态红簇。改后:
   - 跑 `scripts/run_tl_tests.py` 确认 10/10 仍过 (重点 `test_flickering_green_is_flashing` 不能被破坏)。
   - 重跑 `scripts/diag_07_light.py input_video/违章07.mp4 45` → 确认 0–43s 稳定 green (无 24–27/43–44 红闪)。
2. **重生成 07 标注视频** 供用户核对: `python scripts/draw_light_boxes.py input_video/违章07.mp4 --out data/output/annotated --sec 45` (用修好的代码)。
3. **跑 visible-only 评测** (Phase-1 主指标): `python scripts/eval_light_all.py` → 看 `vis_acc`/`visF1`。
4. **分阶段扩到 02/03/04** 标定, 再扫全部 11 (grill-me 决议顺序)。
5. **git commit** (按 6.1–6.4 进展, 建议分次提交: 检测器+单测 / GT+文档 / 诊断脚本)。提交前与用户确认脏树范围。

---

## 7. 禁止重复修改 / 重复执行的红线

- ❌ **勿复用 `frac_v>=0.30` "是否真点亮" 门控** —— 07 灯泡饱和但不削波到 V=255, 该门控会把正确绿灯排除 (已实证失败, 见 2.1)。
- ❌ **勿回退到 v6 "单轨迹按面积/持久度选灯"** —— 同杆红绿合并后红灯赢, 正是 07 原始 bug。
- ❌ **勿用 pytest 直接跑单测** —— cv2 导入慢导致 hang; 用 `scripts/run_tl_tests.py`。
- ❌ **勿为 `inferred`/`occluded`/公交玻璃反射 段"识别出来"而调阈值** —— 用户明确: 优先可见, 难的 → `unknown`/`review`, 禁止 overfit 到难例。
- ❌ **勿在未生成视频 + diag 表前就声称"修好了"** —— 以 `违章07_lightboxes.mp4` + `diag_07.log` 为证。
- ❌ **勿重复重写已稳定的 `_cluster_heads` 与 cx 归一化** —— `cx` 已归一化 (与 `cy` 同单位, 单位 bug 已修), 勿改回像素。
- ❌ **勿把选灯改成"全场景红/绿像素发射求和"** —— 场景红 (车/标志) 会淹没正确绿灯 (见 5.1 末段)。
- ❌ **勿用"局部对比度门控"过滤候选** —— 方向反了: 真绿灯在亮天空前 contrast≈0 被杀, 黄按钮在暗玻璃前 contrast 高被留 (8.3 实证). 排除按钮/衣服/树只用 **锚点+灯形分**。
- ❌ **勿为"让黄色按钮输出红色"而调 hue 分类** —— 按钮色相本就橙红(H≤35), 应靠锚点/灯形分不锁它; 琥珀→红单测(test_amber)仍需保持。
- ✅ **允许/应做**: 调 `head_area_floor` / `head_dist` / 选头加权公式 / `signal_cy_cutoff`; 增删诊断脚本; 跑 visible-only 评测。

---

## 8. 用户视频复核反馈 + 稳定器修复 (2026-07-12 12:19 更新)

### 8.1 用户看视频后的 3 点精确反馈
1. **黄色被误判红 + 框错区域**: 黄圈 READING 落在**倒计时数字面板**(灯在它上方), 下面被识成"红"的其实是**黄色行人按钮**. 按钮"从对比度和亮度上"不该被当信号灯抓出, 且黄色不应输出红色.
2. **24–27s ROI 跳到行人衣服** = 前后 tracking 没做好. 要求加**稳定器/动态跟踪**, 用区域+深度保持 ROI 稳定(信号位置不动).
3. **21s 后 ROI 漂到远处树木**; **43s 后真值是红灯**(那段的红是对的).

### 8.2 已实施修复 (traffic_light.py + config.yaml + 可视化)
- **形状分 `lamp_score = max_spot面积 / 候选数`** + `lamp_score_min=35`: 真灯泡=few-large(高), 倒计时数字=many-small(低) → 选灯/建锚偏好真灯泡, 不被倒计时面板劫持.
- **空间锚稳定器** `_select_lit` 重写: 前段准确帧锁定锚(anchor_radius=0.13 内迟滞); 锚位置用重 EMA(0.92/0.08)抑制漂移; 锚丢失仅当"远处且极持久(reanchor_need=150帧)且像灯泡"才重锚 → 衣服/远树/倒计时无法劫持(解决反馈②③); 锚丢失 <anchor_hold(30帧) 保持旧色, 超则 unknown(契合遮挡→复核).
- **保持 hue 分类**(琥珀 BGR(0,120,255)≈H14→红), `test_amber_treated_as_red` 仍 PASS — 黄色按钮靠"灯形分/锚点"而非色相/对比度区分.
- **可视化** `draw_light_boxes.py`: 新增**青色 ANCHOR 方框**(稳定器锁定的灯位) 区别于黄色 READING 圈, 便于肉眼确认 ROI 不漂移. `diag_07_light.py` 打 anchor(cx,cy,sel).
- 单测 **10/10 PASS** (run_tl_tests.py). **configs/config.yaml** 新增 `head_area_floor/lamp_score_min/anchor_radius/reanchor_need/anchor_hold` 并清理文档头.

### 8.3 ⚠️ 对比度门控是回归 —— 已移除 (2026-07-12 12:3x)
- **原设想**: 加 `min_contrast=40` 局部对比度门控, 候选需比背景亮 ≥40 才当灯, 以为能拒掉黄按钮(用户原话"从对比度和亮度上不该被抓出").
- **实际(实证失败)**: 方向完全反了 ——
  - 07 真绿灯在**明亮天空**前, 灯 V~200 / 天空 V~220 → `contrast≈0` 甚至负 → **真绿灯被杀** (`diag_07_anchor.log`: g_sp 全程 =0, 全 red, 锚点从未锁定 ax/ay=-1).
  - 黄按钮在**暗色公交玻璃**前, 按钮亮/玻璃暗 → `contrast` 反而**高** → 门控会**保留**按钮(与用户要求相反).
- **结论**: 对比度不能作为"是否信号灯"判据. 正确做法是靠 **锚点稳定器 + 灯形分(few-large=真灯泡)** 让真灯泡一旦锁定, 小按钮/衣服/树都无法劫持 ROI. 已在 `_candidates` 移除该门控并清理 `vh/dk/contrast` 残留变量; `__init__` 的 `min_contrast` 参数暂保留为死参数(无副作用), 后续清理.

### 8.4 待确认 (diag 重跑中: diag_07_anchor2.log)
- 预期 `data/output/diag_07_anchor2.log`: 0–43s 稳定 green (对比度门控移除后绿灯恢复检测), **无 24–27 红闪**(锚点锁绿后瞬态红车/按钮无法翻盘), anchor(cx,cy) 全程基本不动.
- 确认后**重生成 `违章07_lightboxes.mp4`**(含 ANCHOR 框) 交用户复核"框=真灯且颜色对".
- 注意 43s+ GT=红, 检测器应随之翻红(那段本来就对).

---
### 8.5 用户澄清 07 信号单元结构 (2026-07-12 12:56)
- **07 信号 = 竖向组合单元 (自上而下)**:
  - 顶部: **行人走路图标 (绿)** — 行人可通行 → 本任务"行人绿灯"
  - 中部: **倒计时数字面板** — 黄圈 READING 实际锁在此处
  - 底部: **站立小红人 + "等待"汉字 (红)** — 行人禁行 → 本任务"行人红灯"
- **用户确认**: 本场景选倒计时也可以 (倒计时颜色与行人灯态同步 → 读倒计时=读行人灯态). 青色 ANCHOR 锁在单元上部(信号灯本体/行人图标).
- **语义对齐 (关键)**: 检测器 green(0-42s)=走路图标亮=行人绿灯=**违章窗口**; red(43-44s)=站立人+"等待"=行人红灯=**安全**. 与 E12 语义反转完全一致. **07 的 GT green/red 即指行人信号态, 评测口径正确, 无需改 GT.**
- **⚠️ 扩到 02/03/04 的设计风险 (必须牢记)**: 其他视频可能**行人信号与机动车信号分离**(不同物体/机位). 本任务判违章依赖**行人绿灯(走路图标)**, 不是机动车绿灯. 若检测器误锁机动车信号 → 相位**反转**(机动车绿=行人红) → 整段误判. "选倒计时可以"仅在"倒计时色=行人态"成立时安全. **02/03/04 必须逐视频确认锁的是行人信号.**

*本快照用于上下文压缩前交接。恢复工作时先读 `src/redlight/models/traffic_light.py`(当前 `color-v7-stable` 实现: 形状分+空间锚, **已移除对比度门控**) + 本文件第 5/6/8 节。*

---

## 9. 行人信号 Prior 标定结论 (2026-07-12 晚, Phase-1 收尾)

### 9.1 标定方法(反射鲁棒, 针对 04 极小远信号)
自动聚类/面积过滤在 04 完全失效(信号灯太小太远 + 满屏绿红反射). 四层兜底扫描:
1. `scripts/scan_pedestrian_signal.py` — 持久小红/绿亮斑频率扫描, 排除偶发反射(频率低).
2. `scripts/toggle_scan.py` — 同位置"红段常红+绿段转绿"判别, 排除**静态红物体**(永红不绿).
3. `scripts/stable_lamp.py` — 红段亮斑面积稳定(cv<0.6)判别, 反射面积忽大忽小被排除.
4. `scripts/test_prior.py` — 候选 prior 做 HSV 直采时间线, 挑"红段 maxR 高 + 绿段 maxG 高".
最终用 `scripts/tl_timeline.py --prior CX CY --roi N` 确认时序切换点.

### 9.2 三视频校准结果(均用原始视频抽帧 + 检测器时间线验证)
| 视频 | prior (cx,cy) | roi_px | 时间线(检测器) | GT | 状态 |
|------|---------------|--------|----------------|-----|------|
| 违章02 | (0.78, 0.15) | 160 | 0-~20红 / 21-85绿 / 85+红 | 同 | ✅ 用户已肉眼确认(右上角走路图标绿) |
| 违章03 | (0.77, 0.05) | 160 | ≈0-73红 / 74-134绿 / 135+红 | 同 | ⚠️ 开头~10s误绿(ROI圈到绿景物), 中段/尾正确, 待用户看视频确认 |
| 违章04 | (0.69, 0.27) | 220 | 0-41.7红 / 42.4-43.1绿 | 同 | ✅ 锚点全程钉死, 完美匹配 |

**04 定位要点**: 信号灯为右侧竖排组合单元(绿走人图标 cy~0.16-0.25 在上, 红站人图标 cy~0.24-0.31 在下). roi 必须加到 220 才能同时盖住两灯, 否则只采到单色. 两清洁 prior (0.64,0.25) 与 (0.75,0.30) 均独立切换红→绿, 中心 (0.69,0.27) 覆盖整单元.

### 9.3 交付物 / 待办
- 标注视频: `data/output/annotated/违章0{2,3,4}_lightboxes.mp4` (原始1080p, 已生成, 交用户肉眼确认框/颜色)
- `configs/light_priors.json` — 02/03/04 校准 prior (eval 用)
- `scripts/eval_light_all.py` 已支持 `--priors JSON` 做公平 visible-only 评测
- ⚠️ **红线补充**: 04 必须 roi≥220; 03 开头误绿待用户确认是否要 tighter ROI; 勿为反射段调参(见 §5/§8)
- 下一步: 用户看三视频 → 反馈 → 微调 prior → git commit (当前未提交)
