# Plan v7 — 灯态误绿 / 漏短绿修复（有效性门控 → unknown + 短绿放行）

> 作者：wb。**依据**：`docs/handoff/2026-07-17-cc-direction-light-false-green.md`（cc brief）+ `docs/reports/2026-07-17-wb-diag-light-state.md`（诊断，cc 已独立复核通过）+ cc 放行留言（两条设计钢锭 + 两条改定 + 两拍板建议「是」）。
> 协作：本计划只出方案、不改码。cc 复核通过 + 两改定已写入 + 两拍板采 cc 建议「是」→ wb 直接 TDD 开工。
> 铁律：先诊断后修——诊断已批准，本计划是修复阶段。

---

## 0. 当前状态

- 诊断四根因（Q1/Q2/Q3/Q4）cc 已逐条独立验证：**诊断复核通过 ✅**。
- cc 给两条**设计钢锭**（不可违反）：
  - **①** Q1/Q3/Q4 正解 = **有效性门控 → 回 unknown**，不是「重标 01 的 prior 坐标」（per-video 打地鼠式过拟合）。
  - **②** Q2 与 Q1/Q3 **方向相反**：必须先源头门控掉环境绿，再调快翻绿。顺序倒过来违反红线。
- **cc 复核两处改定（已落实进本计划）**：
  - **改定 1（主次纠正）**：端到端违章走 `observe()`→`fuse_light`（非 `detect`）。`observe()` 门控才是修 01 端到端误绿的主路径，`detect`/`_select_lit` 是模块 eval 那条。observe 三门控须与 `_select_lit` 同等优先（非「防回归」附属）。
  - **改定 2（校准护栏）**：判别器**主要靠紧凑度/fill-ratio + few-large lamp_score**（树叶 vs 灯），**别把 area/saturation 下限设太高**——当年 `frac_v` 亮度门控就因杀了 07 暗绿灯（frac_v 仅 0.02-0.15）才被废弃（L568-575）。阈值保守偏松，靠测量护栏兜底。
- 两拍板 cc 建议均「是」（§7），wb 据此开工。

---

## 1. 根因回顾（cc 已验，不复诊）

| 缺陷 | 根因 | cc 独立验证 |
|---|---|---|
| Q1 01 误绿 | `prior[0.35,0.35]` 落在 SUV 引擎盖+树枝，采到树叶绿+车反光 | ✅ 抽帧：红框在 SUV 引擎盖+树枝，框里无信号灯 |
| Q2 04 短绿 | `prior_flip_on=5` 吃掉 42.4s 起 ~1s 短绿窗，第 5 帧(43.14)才翻，视频已结尾 | ✅ CSV：42.19s 起 g_px≥199/r_px≈0，第 5 帧才翻 |
| Q3 483 误绿 | 环境绿~200 / unknown 被迫选绿 152 / 反射~131 | 计数与 brief 一致 |
| Q4 从不 unknown | `min_frac=0.002` 太低 → env 绿使 seen 永非0 → 融合不回 unknown | GT-unknown 163→green152/red11/unknown0 一致 |

---

## 2. 修复机制

### 2.1 端到端链路（cc 已核对，主次据此定）

`dag.py:94 tl.observe(frame)` → `engine.accumulate`（`violation_engine:169 obs=light_observation.get("obs","off")`）→ `fuse_light`（`temporal_fusion:45` 把 None/off 当 off；`_window_state:20` seen==0 → `unknown`）。**`detect()` 的 `light_state` 只喂 viz/模块 eval，不进违章引擎**。`fuse_light` 对 obs=None/off → unknown（机制端到端成立 ✅）。

代码出口（门控须覆盖，主次按端到端影响排）：
- **出口 P（主路径，决定端到端 01 成败，`observe()` L140-230）**：
  - P-a YOLO 框（L194）：`_sample_box` 判色，无有效性校验。
  - P-b 先验直采（L199-203）：`_sample_prior_color` → `_sample_roi`（L626-657）按占比直采，`min_frac=0.002` 太低 → env 绿判绿。**这是 01 误绿的主出口**。
  - P-c 先验半径内候选求和降级（L207-219）：直接对 `near` 候选面积加总，**无紧凑度校验** → 树叶候选也计入绿。
- **出口 M（模块 eval 路径，`detect()` → `_select_lit` L440-452）**：最近头 `h` 落 `r*0.5` 内直接 `return h["last_dom"]`，无有效性校验；树叶/反光聚成头（cy<0.6、area≥head_area_floor）即被当灯 → 误绿。
- **改定 1（主次纠正）**：`observe()` 三出口（P-a/P-b/P-c）门控是**一等公民**（修端到端 01），与 `_select_lit`（M）**同等优先**，不是「防回归」附属。§6 有专门 TDD 钉死 01 端到端。

### 2.2 有效性门控 `_is_signal_like()`

新增共享判定，在「产生 obs 颜色」的出口强制校验：

**主判别器 = 紧凑度（fill-ratio）+ few-large（lamp_score）** —— 区分「树叶 vs 灯」的核心：
- 连通主块 fill-ratio（面积 / 外接框面积）≥ `signal_min_fill`（实心灯盘≈0.7-0.9；弥散树叶≈0.2-0.4 → 拒）。
- 头级复用既有 `lamp_score`（few-large 高、many-small 低，L66）：聚合候选「少而大」= 真灯泡，「多而小」= 树叶/倒计时数字 → 拒。
- 候选求和降级（P-c）：逐候选 fill-ratio = `area/(bw*bh)` ≥ `signal_min_fill` 才计入；弥散树叶候选（低 fill）被过滤。

**辅助（门槛刻意设低，绝不误杀暗灯）**：
- 面积：`signal_min_area` 取低（≈10px，保留远/小灯）；上限仅防「整片绿墙」(`signal_max_area_ratio`)。
- 饱和度：`signal_sat_min` 取**松**（≈70，仅剔灰白反光；**不**沿用被废弃的 `frac_v` 亮度门控——L568-575 白纸黑字：当年 `frac_v` 门控杀了 07 暗绿灯 frac_v 仅 0.02-0.15，故废弃）。**暗灯=低亮度但紧凑 → 必须放行**。

**行为（关键）**：
- 任一出口「不是信号灯样」→ 该通道返回 `None`/`"off"`（不产生绿/红 obs）。
- `obs=off` → `global_recent`/`_light_obs` 不计数绿色 → `seen` 可归 0 → `unknown`（Q4 恢复 D1 安全网）。
- 已建立状态（红/绿）下，env 绿→off 不构成「连续绿」，不误翻 → 视频保持正确红/绿（顺带把 331 个 红→绿 误判掰回红，acc 反升、误绿消除）。

**泛化性 / 不过拟合**：门槛全部复用既有特征（fill-ratio/lamp_score/area/sat），且为**全局模型参数**（写 `configs/config.yaml` 的 `traffic_light:` 段），**不新增 per-video 参数、不碰 `light_priors.json` 坐标**（尤其 01 不回标）。

### 2.3 校准护栏（改定 2：别误杀暗/小真灯）

- **阈值保守偏松**：`signal_min_fill` 初值 0.35、`signal_min_area` 10、`signal_sat_min` 70、`signal_max_area_ratio` 0.4。宁可门控漏一点（少挡几帧 env 绿），**绝不把现在能过的 8 个视频（尤其 07 暗绿灯）打回**。
- **主判别靠紧凑度，不靠亮度/面积/饱和下限**：这是当年 `frac_v` 门控翻车的反面教训（L568-575）。
- **测量护栏兜底**（不满足即回退阈值）：
  - 模块 eval 总体 acc ≥ 85.7%；
  - 端到端 8 视频（02/03/05/06/07/08/09/11 等当前能过者）不回退（tp 保持、fp 不增）；
  - **不杀暗灯**：07 暗绿灯仍判绿（端到端 07 不回退为硬指标）；
  - 负例 10 保持 0 fp、01 端到端转 0 fp。
- 若某视频因门控过严回退 → 只松该特征阈值（优先松 `signal_sat_min`/`signal_min_area`，最后才松 `signal_min_fill`），不回退到「无门控」。

### 2.4 短绿放行（Q2）—— 仅 2.2/2.3 落地且验证通过后才做

- 全局 `prior_flip_on` 从 `5` 下调到 `3`（config，全局，非 per-video）：`red→green` 只需 3 帧连续有效绿即翻，抓 04 ~1s 短绿（L776 `_trailing_run(seq,"green") >= prior_flip_on`）。
- **保持 `prior_flip_off=10` 不动**（守 02 绿灯相位内红反射抗抖；与 `enforce_transition_limit` 红线无关，但同属「不削弱迟滞/过渡吸收」精神）。
- **安全性依赖 2.2**：env 绿已门控为 off → 不会与真实短绿竞争 → 调快翻绿不再制造新误绿（呼应钢锭②）。
- 若 3→2 才能救 04 则再扫，但不过度（避免瞬态反射 2 帧翻绿）。

---

## 3. 修复顺序（依赖，呼应钢锭②——绝不倒过来）

- **Phase 1（只做 2.2 有效性门控 + 2.3 校准）**：
  - 跑模块 eval + 端到端 eval（§5）。
  - 须满足：误绿数↓、acc≥85.7%、8 视频不回退、不杀暗灯(07)、负例10 保持 0、01 端到端→0fp、04 仍 fn=1（此时未动翻绿，已知）。
- **Phase 2（仅当 Phase 1 满足 → 做 2.4 调 `prior_flip_on=3`）**：
  - 重跑双 eval。须满足：04 救回（fn→0 或绿窗命中）、且不引入新误绿（负例10 仍 0、01 仍 0 fp、8视频不回退）。
- **顺序倒过来（先调翻绿）违反红线 → 禁止。**

---

## 4. 红线映射（明示未触碰）

| 红线 | 本计划如何守 |
|---|---|
| 不削弱 `enforce_transition_limit`(dcc8dc6) | 修复在 `traffic_light` 检测/融合层，不动 `violation_engine` 过渡吸收逻辑 |
| 不加 per-video 硬编码灯参 | 门槛全为全局 config；不动 `light_priors.json` 坐标（01 不回标） |
| 模块 eval（混淆矩阵，误绿↓、acc≥85.7%） | §5 模块口径 |
| 端到端（v2+box，禁 reuse）双验证 | §5 端到端口径 |
| TDD | 先写 `_is_signal_like` 单测（合成树叶掩膜→None、合成紧凑灯块→green、低紧凑度→None、**暗但紧凑→green 不杀**、**01 环境绿帧经 observe() 门控→obs=off→fuse_light→非 green** 端到端钉死）再接生产 |

---

## 5. 验证口径（与 brief 对齐）

- **模块**：`python scripts/eval_light_fast.py --videos 违章01 违章02 ... 违章11`（全 11）→ 逐视频 acc + 混淆矩阵。判据：误绿数（红/未知→绿）↓、总体 acc ≥ 85.7%。
- **端到端**：`python scripts/eval_violations.py --detector v2 --occ-denom box`（**禁 `--reuse`**，含负例）→ 8 视频不回退、不杀暗灯(07)、负例10 保持 0 误报、01 期望 0 fp、04 期望救回。
- **回归固化**：在 `datasets/gt/light_regression.csv` 增硬用例——04 短绿窗(42.4s–结尾)期望 green、01 误绿段(46.9–50.5s)期望非 green。

---

## 6. 改动文件

- `src/redlight/models/traffic_light.py`：
  - 新增 `_is_signal_like(self, *, mask=None, spots=None, box=None)` 助手（fill-ratio / lamp_score / area / 松 sat）。
  - **observe() 三出口（P-a/P-b/P-c）一等公民门控**（改定 1）：P-a `_sample_box` 加区域有效性；P-b `_sample_roi` 加主块有效性；P-c 候选求和前按 per-spot fill-ratio 过滤，无有效候选 → obs="off"。
  - `_select_lit` 先验头路径（L440 前）按 `lamp_score`/紧凑度门控（M，模块路径）。
  - 门槛从 `cfg.traffic_light` 读取（新增属性，默认保守）。
- `configs/config.yaml` `traffic_light:` 段（L51 起）：加 `signal_min_fill` / `signal_min_area` / `signal_max_area_ratio` / `signal_sat_min` / `prior_flip_on=3`（全局，非 per-video；`prior_flip_on` 仅 Phase 2 启用）。
- `tests/unit/test_traffic_light_validity.py`：新建，TDD 先写（含 01 端到端钉死用例）。
- `docs/reports/2026-07-17-wb-fix-light-state-report.md`：写码后交付。

---

## 7. 待 Jacob 拍板（cc 建议均「是」，wb 据此开工）

- **Q1 接受「更多 unknown / 更少自动 confirmed」→ cc 建议 是**：违章系统「冤枉好人（假绿→假违章）」远重于「标记待复核」；恢复 D1（不确定→review）是正确安全侧。代价=个别真违章变 review，靠 eval 盯 recall。
- **Q2 04 极短绿救不回则接受继续漏 → cc 建议 是**：04 是 1.2s 刀尖例，边际价值；主收益在 01/误绿 + 483 帧 + D1 恢复（泛化）。
- 以上采 cc 建议视为已拍板，wb 直接 TDD 开工；若 Jacob 异议再回退。

---

## 8. 不在本计划（独立跟进）

- **crosswalk v2 running-max 过延伸护栏（Phase 1.5）**：episode 甩出 GT 窗(05/09)，泛化隐患。加衰减/上限。wb 有空时另提，不塞入本 brief。
