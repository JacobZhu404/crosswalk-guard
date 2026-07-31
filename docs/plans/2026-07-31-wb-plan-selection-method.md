# 选灯改进方法计划（governing 判别器 + 弃权机制）

> 出自 wb（执行者），据 cc 2026-07-30 复核裁定（`docs/handoff/2026-07-30-cc-verify-selection-diagnostic.md`）。
> **本文是计划，cc review 通过后再写代码 / 训模型。先计划后动。**
> **v2（2026-07-31 cc 分级复核 `2026-07-31-cc-review-selection-method-plan.md`，commit 1d1726d）**：大框架通过。§3.1 立即可动；§3.2/§3.3 已并入 **R1 判别器目标=有效行人灯vs干扰 / R2 无灯LOVO单列03·06·11 N/A / R3 τ单一全局不碰测试折 / R4 漏绿≤80硬约束**。详见 §2.1–§2.4、§3。

## 0. 已锁定的事实（方法设计的地基）

- 误绿被 canonical 逐帧 GT 干净钉死为**定位/选灯问题**：R1 误绿 5.51%(22/399)，扣 05 后 **2.19%(8/365)**；成因 **0% 状态判错，100% 选了干扰源**（反射/背面/车灯/绿树叶/绿干扰）。
- 诊断 Q2 决定性拆分（cc 独立重算逐帧对上）：8 个正常场景误绿帧 = **1 召回洞 + 2 选灯排序错 + 5 无 governing 灯（干扰自发绿）**。归并 **7/8 = 选了非 governing 干扰，1/8 = 真召回洞**。
- 当前 `select_gtfree` 结构性缺口：候选非空时**必返一个最高分候选，永不返 None**——纯相对排序，无绝对阈值。所以 5 个无灯帧必然"选个最像灯的干扰"输出绿。
- L3（`l3_ped_full.pt`）在选灯里**已被证明有害**（R2 11.78% > R1），新判别器是不同物种，取代其选灯角色，但 L3 代码保留向后兼容。

## 1. cc 两条硬要求（本计划必须落实）

1. **判别器必须能"弃权"**：最佳候选的有效行人灯置信度 < τ → `select_gtfree` 返回 `None`（该帧不输出绿）。训练必须含**无信号帧的干扰 = 负样本**，学绝对拒识，而非仅相对偏好。（判别器目标 refined→「有效行人灯 vs 干扰」，见 §2.1/R1）
2. **评分台用全 399 帧选灯质量**（别只 8 帧）：有灯帧选中框与某 governing 框 IoU≥0.3 的比例（选灯精度）；无灯帧正确弃权比例；扣 05 误绿作头条副指标。训练正样本=717 governing crop、负样本=同帧非 governing 候选（量充足）。LOVO 去循环不变。

## 2. 方法设计

### 2.1 Governing 判别器（`models/governing_discriminator.pt`）

> **R1 修订（cc 复核）**：判别器目标 = **「有效行人灯 vs 干扰」**，**不是「governing vs 非 governing」**。governing 性（管哪条斑马线）是方向/几何属性，64×64 crop 里看不见；负样本 A（同帧非 gov 候选）会混入未标注的正脸真行人灯（多 gov 视频 02/07/09），灌标签噪声。判别器只学「这是不是一盏真·处于信号态的行人灯」（拒反射/信号灯背面/车灯/绿树叶）；**governing 选择（多盏有效灯里挑管这条道的）仍交 L1 几何 + L2 时序，不压给判别器**。弃权门触发 = top 候选连「有效行人灯」都不是。

**任务**：对每个候选 crop 输出 `governing_conf ∈ [0,1]`（"这是一盏有效·信号态行人灯吗"），**校准到可用于绝对阈值 τ**。

**训练数据**（来自 `datasets/gt/light_canonical_gt.json`，399 帧）：
- 正样本：717 个 governing 框 crop（都是有效行人灯；从该帧原图按 box_norm 裁，与诊断/评测同口径）。
- **负样本 B（主力）= no_light 帧里的所有候选 crop**（干扰自发绿帧的干扰，保证是干扰，治 5/8 主导类）。
- **负样本 A（仅消融，不默认全用）= 同帧、非 governing 的候选 crop**（可能含平行斑马线正脸真灯→标签噪声风险）。**跑 LOVO 带 A vs 不带 A 各一遍，若 A 拖低泛化就丢，主力用 B。**
- countdown 属「有效行人灯」=正样本（cc 认可）；背面灯（暗/红X背）视觉可分=负样本（Jacob 点过）。
- 不混入逐帧 GT 进推理；训练只用框坐标裁图，GT 不进 `select_gtfree`。

**架构**：复用 `L3PedVehicleNet` 骨架（tiny CNN，36KB 量级），改为**单输出 sigmoid（有效行人灯 vs 干扰）**；crop 用 **64×64 RGB**；沿用 L3 数据增广（翻转/抖动）。**不堆容量**——Phase B 教训：36KB 容量非瓶颈，正则+数据广度才是，配 weight-decay + dropout + 早停。

**LOVO 训练（去循环）**：对每视频 V，用其余 10 视频的 crops 训练，在 V 上评；11 折轮完。绝不同一视频又训又评。

**校准与 τ 选择（R3 修订）**：sigmoid 输出做温度缩放，**校准集排除测试视频**。τ 用**训练折内数据（或 inner-CV）定一个全局值**，统一施于所有测试折；**报告 τ 敏感性曲线**；**禁止 per-video τ**（=per-video 硬编码，红线）；τ 选取须含 **漏绿 ≤ 80 硬约束（R4）**。

### 2.2 `select_gtfree` 加弃权门（加法式，不破坏现有行为）

新增可选参数（向后兼容，默认不传=当前行为不变）：
```
select_gtfree(candidates, ped_prior, temporal_scores=None, temporal_weight=0.4,
              l3_scores=None, l3_weight=0.3,            # 旧: 保留
              governing_scores=None, governing_weight=0.0, governing_threshold=0.5)  # 新
```
- **组合**：若 `governing_scores` 非 None，`final = (1-governing_weight)*base + governing_weight*governing_conf`（`base` = 现有 L1+YOLO+0.4·L2；**此路令 `l3_weight=0` 即不叠加有害的 L3**）。
- **弃权门（结构性修复，R1）**：选出 `best` 后，若 `governing_scores is not None and governing_conf(best) < governing_threshold` → **返回 `None`**（该帧不输出绿）。`governing_scores` 为 None 时行为完全等同今日（向后兼容、生产兜底不变）。
- 弃权触发语义 = **top 候选连「有效行人灯」都不是**（非 governing 选择）。这一门直接治 7/8：排序错帧（背面/反射）靠 conf 压低→真灯排上来；无灯帧（5/8）所有候选都不是有效灯→conf<τ 弃权，不依赖 per-video 无灯负样本充足性（R2 关键）。1/8 召回洞靠弃权顺带不输出绿（不涨误绿，仅该帧漏，可接受）。

### 2.3 评测：全 399 帧选灯质量评分台（新建 `scripts/eval_selection_quality.py`）

复用 `measure_falsegreen_canonical.py` 的候选/选灯/GT 比对骨架，主指标升级：
- **选灯精度（主指标）**：governing 存在的帧，选中框与某 governing 框 **IoU≥0.3** 的比例。
- **正确弃权率（主指标）**：no_light 帧，select 返 `None`（且不输出绿）的比例。
- **误绿率（头条副指标，扣 05）**：选中且 color=green 但帧无绿真值，扣 05 后。
- **漏绿（硬约束，R4）**：governing=green 帧但 select 返 None 或输出非绿。**基线 R1=80/154≈52%；τ 选取与 gate 把漏绿 ≤ 80 设为硬约束，漏绿 > 80 直接判不过**（误绿优先但漏绿翻倍=净回退）。
- 报 **mean ± std + worst-video**（per-video 分解）。

**无灯数据不均的评测处理（R2，据实测 99 无灯帧：61 在 03、18 在 01、06/11 为 0）**：
- **03 折单列报告**，不混进 mean（留出 03 训练丢掉 62% 无灯负样本却在「75% 无灯」的 03 上评弃权，是最难折，混进 mean 会掩盖）。
- **06 / 11 正确弃权率显式标 N/A**（零无灯帧，指标无定义），**禁用 0 或 100 充数拉平均**。
- LOVO 变体：判别器按 §2.1 折外训练后在此台评，得泛化数字（同样 03 单列、06/11 N/A）。

### 2.4 防过拟合护栏（硬，引 Phase B `negqual` 教训）

- **LOVO 去循环**：训练/评测视频严格不交叠；GT 只评测不进推理。
- **τ 单一全局（R3）**：τ 由**训练折内 / inner-CV** 定一个全局值，**统一施于所有测试折**；温度缩放校准集**排除测试视频**；报告 τ 敏感性曲线；**禁止 per-video τ**（=per-video 硬编码，红线）。
- **漏绿 ≤ 80 硬约束（R4）**：τ 选取与 gate 把漏绿 ≤ 80 写进约束（非事后附注），**漏绿 > 80 直接判不过**。
- **多 seed 报 min**：判别器训练跑 ≥5 seed，报 mean±std + **worst-seed(min)**——只在幸运 seed 过关不发货（cc 硬规）。
- **正则**：weight-decay + dropout + 早停；不堆容量。
- **TDD 先**：先写判定边界单测（IoU≥0.3、弃权门阈值、LOVO 无泄漏、分数组合向后兼容、03 单列/06·11 N/A 渲染），再写实现。
- **scoped git / 署名 / trunk main**：改动限 `ped_light_selector.py`（加法参数）+ 新 `governing_discriminator` 训练脚本 + 新评测脚本；模型权重 gitignore 不进库。

## 3. 实施步骤（分级放行，据 cc 2026-07-31 review）

- **§3.1 立即可动（已放行）**：纯管道，与判别器目标无关。TDD 先红后绿，默认不传=行为不变。
- **§3.2 训练 / §3.3 评测**：落实 R1–R4 后动手（本计划书已并入，cc 不再逐字批；交复核时 cc 按 R1–R4 对照）。

1. **§3.1（现在做）** TDD：给 `select_gtfree` 加 `governing_scores/governing_threshold` 参数 + 弃权门 + 向后兼容单测（先红后绿，不改现有行为；`governing_scores=None` 时完全等同今日）。
2. **§3.2 训练**：写 `scripts/train_governing_discriminator.py`。目标=「有效行人灯 vs 干扰」（R1）；正=717 governing crop，**负 B 主力、负 A 仅消融（带 A vs 不带 A 各跑 LOVO，拖低泛化就丢）**；tiny-CNN 二进制 + 温度缩放（校准集排除测试视频）；LOVO 训练；≥5 seed 报 worst；正则不堆容量。
3. **§3.3 评测**：写 `scripts/eval_selection_quality.py`：全 399 帧选灯精度(IoU≥0.3) + 正确弃权率 + 扣05误绿 + **漏绿硬约束(≤80, >80 判不过, R4)**；**03 折单列、06/11 弃权率 N/A（R2）**；报 mean±std + worst-video；LOVO 泛化（同样 03 单列/06·11 N/A）。
4. **τ 选取（R3）**：用训练折内/inner-CV 定**单一全局 τ**（不碰测试折），含漏绿≤80 约束，报 τ 敏感性曲线；禁止 per-video τ。
5. 跑 gate：基线（今 `select_gtfree` 无 governing）= 选灯精度 / 弃权率 0 / 误绿 2.19% / 漏绿 80。新模型须 **选灯精度↑ + 正确弃权率↑ + 扣05误绿↓ + 漏绿 ≤ 80**，LOVO worst-video（03 单列）可接受。
6. gate 不过 → **净回退不出货**（保留当前 `select_gtfree`，新模型不入生产）；gate 过 → 仍**不接线**（cc 放行才接），报告交 cc 复核 + Jacob 抽检。

## 4. 05 小灯 track（单列，不阻塞主线）

- 05 类（又小又透窗/背面）按 Jacob 拍板记为**已知局限，不投入**；主判别器大概率救不了 05。
- 评测里 05 **单独列、不计入主目标**（主标尺 = 扣 05 后的数字）。
- 真 m/x 大模型测试仍待 Jacob 下权重（CDN 502 拦）；若未来要做，归独立小灯检测 track。

## 5. 红线（不变，全守）

先计划后动 / 只读不接线 / 生产 prior·权重只读 / gate 不过不接线 / 禁 GT 进推理·LOVO 去循环 / scoped git / 署名 / trunk main / TDD 先 / 不碰 `enforce_transition_limit` / 净回退不出货 / 多 seed 报 min。

---

**交付（本轮）**：本计划 doc，交 cc review。**批准后**才进入 §3 代码+训练+评测，产出 `governing_discriminator.pt` + 评测报告交复核。不擅自改 `select_gtfree` 生产行为、不接线、不训模型，直到 cc 放行 §3。
