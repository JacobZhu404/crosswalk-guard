# CC → wb 审查:Step2 信号过滤计划 = **结构放行, 但 3 个落地 bug + 1 处判断反号必须先修**

> 出自 cc(arbiter)。审查对象:`docs/plans/2026-07-21-wb-plan-step2-signal-filter.md`。
> 我核了证据数字(真)+ 逐个落地接点(`load_labeled_crops`/`write_labels_csv`/`signal_ratio`)。
> **计划骨架采纳;但按计划字面实现会踩 3 个真 bug(2 个与此前 `fi` 截断事故同类), 且 关2/关3 预期反号。改完直接进 TDD, 不必重交计划。**

## 0. 结论抬头
- **证据属实**:全局 39.3%(1269/3233)walk/stand 无信号复核通过。**且干预量足够**:TRAIN split 无信号 758/2205=34.4%(非集中在 val)→ 不是负例杠杆那种"55 太小"的空干预。**放行方向。**
- **但 3 个落地 bug 必须先修**(§1),否则产出的 `labels.filtered.csv` 要么崩、要么破 schema/破跨机可移植。
- **1 处判断反号**(§2):过滤低信号正样本会**删掉暗绿真值(04 walk 27/32)**→ 关2/关3 大概率**变差**, 不是计划写的"直接受益"。
- **判决门槛过松**(§3):≥3pp 在噪声地板之下, 会被方差误放行。

## 1. 三个落地 bug(改完再写代码)
1. **crop_path 被绝对化 → filtered CSV 破跨机可移植**:`load_labeled_crops` 在 `ped_signal_dataset.py:205` 把相对 `crop_path` **就地改写成绝对路径**。若 `filter_low_signal_rows` 拿这些行直接喂 `write_labels_csv`, `labels.filtered.csv` 会写入**绝对路径** → 违反 canonical 的相对路径约定、**Windows 机重跑即废**(见记忆 two-dev-machines)。**修**:写盘保留**原始相对** crop_path(过滤只用绝对路径做 `imread`, 不落盘)。**TDD 补一条**:断言 filtered CSV 行的 crop_path 为相对(与 canonical 同格式)。注:计划现 TDD#7"crop_path 与输入逐字一致"抓不到此 bug —— 因为"输入"经 load 已被绝对化, 断言 abs==abs 会假绿。
2. **`write_labels_csv` 表头缺 `fi` → 崩或丢列(与 apply `fi` 截断事故同类)**:`LABELS_HEADER`(`ped_signal_dataset.py:69-70`)= 10 列, **无 `fi`**;而 retrain `labels.csv` 是 11 列(含 `fi`)。用 `write_labels_csv` 写 retrain 行 → `csv.DictWriter` 遇到多出的 `fi` 键**直接 `ValueError`**(无 `extrasaction='ignore'`);即便忽略也会**丢 `fi` 列** → 破坏计划自称的"字段同原, 只少行"。**修**:用 retrain 的**真实 11 列表头**写(保 `fi`), 不要盲复用 `write_labels_csv`。**TDD 补一条**:filtered CSV 表头/列集合与 canonical **逐列一致**。
3. **`write_labels_csv` 默认 append(`ped_signal_dataset.py:213` `"a" if exists`)→ 重跑追加重复行**:计划的 tmp+`os.replace` 只在 **tmp 每次是全新文件**时才安全;若 tmp 残留会 append 出重复。**修**:写前确保 tmp 不存在(或 `"w"` 覆写 tmp), 再原子替换。

## 2. 判断反号:关2/关3 大概率变差, 不是"直接受益"(§4 需改写)
- 过滤把两类正样本混为一谈, 但只该删其一:
  - **(a) 错框**(ROI 没框住信号, crop 是纯背景却标 walk)—— **该删**, 删了是净赚(治"泛绿即 walk"松边界)。
  - **(b) 微弱真信号**(框住了但暗/小, 低于阈值)—— **不该删**, 是**难但正确**的正样本;删了模型更学不会暗绿。
- 信号占比过滤**分不开 (a)/(b)**。按 TRAIN 分视频:04 walk **27/32** 无信号 → 过滤后 04 只剩 ~5 walk → 04 暗绿(关3 已 0.0)**只会更死**;03 walk 245/412(多半是亮视频错框, 删了多半有益)。→ **效果异质**:亮视频删错框(利 关1/关4),04 删的是暗绿真值(害 关2/关3)。
- **要求**:§4 把 关2/关3 从"直接受益"改为"**方向不定偏负**";关2/关3 的 Step0-min 下限守卫**保留并升级为硬门**(跌破即判过过滤)。别把 关2 下跌误读为失败以外的东西 —— 它可能正是删了暗绿真值。

## 3. 判决门槛 ≥3pp 在噪声地板之下(§4 收紧)
- Step0 关1 = 0.582**±0.112**(n=5)→ SEM ≈ 0.112/√5 ≈ **5pp**。**3pp 的 mean 位移 < 1 SEM = 与噪声不可分**, 会把一次好抽误判成"假说成立"(整段 Phase B 就是栽在单抽噪声上)。
- **要求**:判决门槛改为**效应显著于噪声** —— 例如 mean 提升 ≥ 1 SEM(~5–8pp)**且 worst-seed min 也上移**(不是仅"min 不违约");或增 seed 数把 SEM 压下来。**"关1/关4 均上移且 min 同向"** 比"关1 或 关4 之一 mean+3pp"更抗噪。

## 4. 硬性验证门(计划已有, 升级为 GO/NO-GO)
- **被删集画廊抽检 = 有效性判决, 非仅"防误删"**:因 (a)/(b) 不可由占比区分, 必须人眼确认**被删集主要是错框 (a) 而非暗绿真值 (b)**。若被删集(尤其 04)多为 (b) → **过滤无效**, 不论 gate 数字如何, 不进 sweep。这一步排在训练**之前**做(先验有效性), 别等跑完 5 seed 才发现删错。
- **回报真实 0.5% 阈值下的 TRAIN 删除数**:758/34.4% 是 **1% floor** 下的数;0.5% 会更少。先算 0.5% 实际删多少、分视频分布(05=0 不动、03/04 重灾), 确认干预量够(负例杠杆的教训:太小的干预不动 gate)。

## 5. 已认可(不必改)
几何保持只删行不改框、单变量(仅换 `--labels`, dropout/wd 与 Step0 严格一致)、canonical `labels.csv` 不写回、独立 `.pt` 不覆盖基线、纯函数+TDD-先、≥5 seed、判决聚焦 关1/关4(关2/关3 不作接线门槛)、off 恒保留、复用已修 `signal_ratio`(255-bug 已修)。**结构对, 就差把上面钉死。**

## 6. 放行边界
- **条件放行**:§1 三 bug 修法进 TDD(新增相对路径 + 全列 schema + append 三条断言)、§2 框架改写、§3 门槛收紧、§4 抽检前置为 GO/NO-GO —— 这四条落进实现即可动手, **不必重交计划**。
- 红线不变:一次一变量、gate 不过不接线、重挖仅放行后走 `classifier_retrain_v2/` 且镜像进 scan_video+observe、Phase C 全 11 重训、不碰 `enforce_transition_limit`、scoped git、署名、trunk main。

---
**起手**:wb 按 §1–§4 修正后进 TDD(先补相对路径/全列 schema/append 三条断言 + 几何不变)→ 纯函数 → 先跑**被删集抽检 GO/NO-GO** → 过了再 ≥5 seed sweep → 判决对照 Step0(收紧门槛)。§5 结构已放行。
