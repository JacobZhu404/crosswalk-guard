# wb → cc:Step2 信号过滤计划(几何保持 cc(B) · 纯函数+TDD · ≥5 seed mean±std+min)

> 承接 cc 裁定 `negquality-null-pivot-step2`(窄版负例杠杆无效偏有害 → 采信 wb 选项 C,
> pivot 到 Step2 正样本提纯)。本计划按 **cc(B) 几何保持原则**:只做信号存在过滤, 不动 ROI 框几何, 不重挖。
> **先计划 cc review 放行再动**。红线:不擅自改 canonical GT / 不碰 prior-ROI 几何 / 不重挖 / gate 不过不接线。

## 0. 修订(cc review 2026-07-21 条件放行, 4 条已落实现)

cc 核了证据数字(全局 39.3% 复核通过; TRAIN split 无信号 758/2205=34.4% → 干预量足够, 非"55 太小"空干预)
+ 逐个落地接点, **结构放行 + 4 条实现修正(不必重交计划)**:

1. **Bug1 相对路径**:`load_labeled_crops:205` 会把相对 crop_path 绝对化, 直接写回破跨机可移植。
   → 读盘改纯 `csv.DictReader` 保相对; 仅训练侧消费时再解析绝对。TDD `test_relative_path_preserved_bug1` 锁死。
2. **Bug2 表头缺 fi**:`LABELS_HEADER`(ped_signal_dataset.py:69)10 列无 `fi`, retrain CSV 11 列含 `fi`
   → `DictWriter` 丢列/ValueError。→ 写盘捕获输入真实 11 列表头。TDD `test_eleven_col_schema_roundtrip_bug2` 锁死。
3. **Bug3 追加重复**:`write_labels_csv` 默认 append → 重跑翻倍。→ 写盘 `open(tmp,"w")` 截断 + `os.replace` 原子。
   TDD `test_no_append_truncate_bug3` 锁死。
4. **判断反号(关2/关3)**:过滤混淆 (a) 错框[该删] 与 (b) 微弱真信号[不该删]。04 walk 27/32 无信号
   → 过滤后 04 只剩 ~5 walk → 暗绿(关3 已 0.0)只会更死。**关2/关3 大概率变差, 不是原计划的"直接受益"**。
   → 框架修正 + 被删集画廊抽检升级为**训练前 GO/NO-GO 有效性门**(确认删的是错框非暗绿真值);
   关2/关3 不作为放行门槛, 但其塌破 Step0 min 视为过过滤警讯。
5. **判决门槛过松**:≥3pp < 1 SEM(Step0 关1 std 0.112 → SEM≈5pp)。→ 收紧为 **mean 提升 ≥1 SEM 且
   worst-seed min 同向上移**, 不被单抽噪声误放行(见 §4)。

**起手顺序(cc)**:补 TDD(4 修正)→ 纯函数 → 先跑被删集抽检 GO/NO-GO → 过了再 ≥5 seed sweep → 判决对照 Step0(收紧门槛)。
**状态(2026-07-21 续)**:TDD 10 项全绿 + 真实过滤已跑(5740→4780/960), 产物待 Jacob GO/NO-GO 后再 sweep。

## 1. 目标与假说

- **根因假说(待判决性实验验证)**:39.3% 的 walk/stand 正样本 crop 里根本没有对应信号色像素
  (证据 `data/output/signal_presence_report.json`:767/1976 walk + 502/1257 stand 无信号, 全局 1269/3233=39.3%),
  集中在 07(walk 233/237 无信号!)与 04(walk 27/32)。这些"脏正样本"让模型学成"泛绿/泛红背景 = walk/stand"、
  决策边界松 → 对新假绿帧照样 fire → **关1/关4 负例拒识差**。
- **Step2 目标**:过滤掉信号占比 < 阈值的 walk/stand 正样本 → 边界收紧 →
  (a) 正样本门(关2 07val / 关3 04)直接受益;(b) **间接**改善 off 拒识(关1 拒 01/10 · 关4 泛化 01/11)
  —— 后者是**判决性实验**(见 §4)。
- **非目标**:不动 ROI 框几何 / 不重挖数据集 / 不加容量 / 不碰 loss / 不捆 Step1 降权或负例杠杆。

## 2. 硬约束(来自 cc 裁定, 逐条承接)

1. **几何保持 cc(B)**:prior-ROI 框几何不变。过滤**只删行**(丢弃低信号 crop 的标签行), 不改 crop 像素、
   不改 `crop_path`、不改框坐标。train / gate / 生产分布天然对齐, 无 skew。
2. **纯函数 + TDD**:过滤逻辑抽成纯函数, **先写测试再实现**, TDD 全绿后才写盘。
3. **≥5 seed**, 报各关 **mean±std 且 worst-seed(min)**(cc ruling `62aeaf1`)。
4. **canonical GT 不改**:`datasets/classifier_retrain/labels.csv` 原样保留(红线)。
   过滤产物写**新文件** `labels.filtered.csv` + `filter_manifest.json`, 原子替换。
5. **出版后必回测关1/关4 是否上移** —— 这是验证"脏正样本驱动负例门"的判决性实验;
   关1/关4 不动则假说被否, 走 cc 指定 fallback(选项 B 硬负例挖掘)。

## 3. 设计

### 3.1 纯函数:`filter_low_signal_rows(labels_csv, threshold, signal_fn=signal_ratio)`

- **输入**:`labels_csv`(同训练装载口径, 走 `load_labeled_crops(..., verified_only=False)`)、
  `threshold`(默认 **0.005 = 0.5%**, 即 cc(B) 指定阈值)、`signal_fn`(默认复用已 TDD 的 `signal_ratio`, 255-bug 已修)。
- **行为**(逐行):
  - `label == "off"` → **恒保留**(off 不要求有信号, 是负类)。
  - `label` 为 `delete` / 空 / None → **恒丢弃**(与 `exclude_deleted` 口径一致)。
  - `label ∈ {walk, stand}`:
    - `img = cv2.imread(crop_path)`(`load_labeled_crops` 已把 `crop_path` 解析为绝对路径)。
    - `ratio = signal_fn(img, "green" if walk else "red")`。
    - `ratio is None`(读图失败 / 空图)→ 丢弃, 记 `reason=unreadable`。
    - `ratio < threshold` → 丢弃, 记 `reason=low_signal, signal_ratio=ratio`。
    - 否则保留。
  - **返回**:`(kept_rows, dropped_rows)`;`dropped_rows` 每项含 `{crop_path, video, label, signal_ratio, reason}`。
- **纯函数特性**:不写盘、不依赖全局可变状态、可重入;复用 `signal_ratio`(`scripts/analyze_signal_presence.py`)与
  `load_labeled_crops`(`src/redlight/data_pipeline/ped_signal_dataset.py`)。

### 3.2 TDD(新增 `tests/test_filter_low_signal_rows.py`, 写盘前先绿)

用合成 crop(同 `tests/test_analyze_signal_presence.py` 的 `_make_img` 套路:灰底 + 角块纯绿/纯红)写
临时 `labels.csv` + 临时 crop 文件, 断言:

1. `walk` + 信号占比 0(全灰)> 阈值 0.005 → **丢弃**。
2. `walk` + 信号占比 0.01(>0.005)→ **保留**。
3. `stand` + 红占比 0 → 丢弃;红占比 0.05 → 保留(校验 red 分支用 `signal_ratio(img,"red")`)。
4. `off` + 任意信号占比(0 与 0.5)→ **恒保留**(负类不被过滤)。
5. `delete` / 空 `label` → **恒丢弃**(与 `exclude_deleted` 一致)。
6. 读图失败(`crop_path` 不存在)的 `walk` → 丢弃且 `reason=unreadable`。
7. **几何不变断言**:`kept_rows` 中 `crop_path` / `video` / `label` 字段与输入逐字一致(只删行, 不篡改任何值)。
8. **不改 canonical**:纯函数不写 `labels.csv`(纯内存);写盘由独立 step 负责, 且只写新文件名。
9. **阈值单调性**(snapshot):同一输入, `threshold=0.02` 丢弃数 ≥ `threshold=0.005` 丢弃数。

跑法:`PYTHONPATH=src:scripts ./.venv/bin/python -m pytest tests/test_filter_low_signal_rows.py -q`

### 3.3 写盘 step(独立, 不进纯函数):生成 `labels.filtered.csv` + `filter_manifest.json`

- 调 `filter_low_signal_rows` → **捕获输入真实 11 列表头**(含 `fi`, 不用 `write_labels_csv` 的
  `LABELS_HEADER` 10 列 —— 它缺 `fi` 会丢列; 也不走其 append 模式)→ `open(tmp,"w")` 截断 +
  `os.replace` 原子写 `datasets/classifier_retrain/labels.filtered.csv`(相对 crop_path 不变, 只少行)。
- 写 `datasets/classifier_retrain/filter_manifest.json`:`{threshold, total, kept, dropped,
  dropped_by_video:{video:{walk,stand,off,other}}, dropped_detail:[{crop_path,video,label,fi,signal_ratio,reason}]}`。
- 附 **GO/NO-GO 有效性门产物** `filter_deleted_gallery.html`:被删 crop 画廊(风险视频 04/07 全量 +
  其余随机采样 60), 每张标 signal_ratio/reason, 供 Jacob 训练前审"删的是错框非暗绿真值"。
- `canonical labels.csv` 原样保留(红线)。执行后回报各视频 dropped 计数, **核对 07/04 高浓度**(应占大头)。

### 3.4 训练集成(单变量对比, 关键:与 Step0 基线同配置)

复用 `train_classifier_retrain.py`, **仅换 `--labels`**, 其余与 Step0 完全一致 → 单变量(只加了过滤):

```bash
PYTHONPATH=src ./.venv/bin/python scripts/train_classifier_retrain.py \
  --labels datasets/classifier_retrain/labels.filtered.csv \
  --manifest datasets/classifier_retrain/manifest.json \
  --out models/ped_signal_v2_step2.pt \
  --seed $S --dropout 0.3 --weight-decay 1e-4 --epochs 30 --balanced
```

- `--dropout 0.3 --weight-decay 1e-4` **与 Step0 基线严格一致**(cc:`Step0 = 必要不充分, 保留为训练默认`)
  → 保证 Step2 vs Step0 是单变量(过滤)比较, gate 数字可比。
- 输出独立 `.pt`(`ped_signal_v2_step2.pt`), **不覆盖** `ped_signal.pt` 基线 / `ped_signal_v2.pt` / Step0 模型。

### 3.5 评估:复用 `sweep_step0.py` 聚合(mean±std+min)

- 复用 `sweep_step0.py` 的 `train_one` / `gate_one` / `_agg`, 加 `--labels` 指向 filtered, 跑 seed 0..4。
- 重点回测:**关1/关4 是否上移(判决性)**;关2 07val / 关3 04 大概率**变差**(04 只剩 ~5 walk, 暗绿更死),
  不作为放行门槛, 但塌破 Step0 min 视为过过滤警讯;median 信号占比(方向性诊断, 非硬 gate)。
- 阈值若 0.5% 方向性正但不够强, 走 §5 阈值扫描 follow-up(不捆本步)。

## 4. 判决性实验对照(Step0 基线 vs Step2, 门槛已收紧)

Step0 基线(正则化 `dropout=0.3,wd=1e-4`, 详见 `docs/reports/2026-07-21-wb-step0-regularization-result.md`):

| gate | Step0 基线 mean±std / min | Step2 预期 / 判定(判决性) |
|---|---|---|
| 关1 拒负例(01/10) | 0.582±0.112 / 0.368 | **上移 ≥1 SEM(≈5pp) 且 worst-seed min 同向上移** → 假说成立 |
| 关4 01 负例 | 0.559±0.137 / 0.296 | **上移 ≥1 SEM 且 min 同向上移** → 假说成立 |
| 关4 11 负例 | 0.800±0.196 / 0.423 | 不塌破基线 min(0.423)(守住下限) |
| 关2 07val 暗绿 | 0.451±0.247 / 0.189 | **大概率变差**(正样本被删, 非门槛) |
| 关3 04 探针 | 0.000 | **大概率更死**(04 只剩 ~5 walk, 归 Step3) |
| median 信号占比 | 1.6% | 上移(方向性, 非硬 gate) |

- **放行裁定(收紧, cc)**:关1 **与** 关4_01 任一 mean 较 Step0 有 **≥1 SEM 稳定提升(SEM=std/√5≈5pp for 关1)
  且 worst-seed(min) 同向上移** → 假说成立, Step2 放行进后续。单抽噪声(<1 SEM)不误放行。
- **守住下限**:关2/关3 不得塌破 Step0 min(否则过过滤, 真正样本被误删 → 降阈值重跑)。关2/关3 本身变差不阻断放行,
  但塌破即警讯。
- **假说被否的判定**:关1/关4 **不动**(即便关2/关3 变化)→ "脏正样本驱动负例门"不成立 → 按 cc fallback 上
  **选项 B 硬负例挖掘**(全训练集选高置信 walk/stand 但 GT=off, 去重保多样), 不接线。
- **绝不因关2/关3 变化就接线**:判决性只看关1/关4(负例门), 必须其上移才证明负例门被正样本提纯间接治好。

## 5. 阈值与范围(if 0.5% 不够, follow-up, 不捆本步)

- 0.5%(0.005)极松:48×48 ≈ 2304 像素, 0.5% 仅 ≈11 像素即过 → 可能只剔"真正空"crop, 对 0.5%–1% 弱信号保留。
  若判决性实验弱, 加扫 **1% / 2%**(复用同一纯函数只换 threshold), 各 5 seed。
  证据:1% 阈值会丢约 1269(39.3%)walk/stand, 量级足够大。
- **数据量风险**:阈值越严丢越多 → 正样本骤减可能欠拟合。`balanced=True` 兜住类平衡;
  监控 `train_acc`, 若正样本类 recall 塌 → 减阈值或补 Step3 分辨率, 不盲目加容量。
- 本步主跑**只做 0.5%**;扫描列为 follow-up, 等 0.5% 结果再决定是否触发(遵循"一次一变量")。

## 6. 红线与范围

- **几何保持 cc(B)**:不动 ROI 框几何 / 不重挖。重挖(重定心)仅放行后走新目录 `datasets/classifier_retrain_v2/`,
  且须**先镜像进 `scan_video` + Phase C `observe()`** 才上(Step2b, 本步不做)。
- **canonical GT 不改**:只产 `labels.filtered.csv` + `filter_manifest.json`;绝不写回 `labels.csv`。
- **单变量**:不捆 Step1 降权 / 负例杠杆 / 容量;独立 `.pt`, 不覆盖基线。
- **≥5 seed mean±std+min**;不凭单 seed。
- **892 盲区**:过滤后必重跑画廊抽检(Jacob 审 06/07 暗绿 + 低信号 crop), 尤其确认被删"低信号 walk"是否真无信号
  (防误删真值);抽检结论进 Step2 结果报告。
- gate 四关不过绝不接线;Phase C 全 11 重训另议。scoped git、署名、trunk main、TDD 先。

## 7. 里程碑(待 cc review 放行后执行)

1. cc review 本计划放行。
2. 写 `tests/test_filter_low_signal_rows.py`(TDD)→ 全绿。
3. 实现 `filter_low_signal_rows`(纯函数)+ 写盘 step(`labels.filtered.csv` + `filter_manifest.json`, 原子替换)。
4. 跑过滤 → 出两文件;回报各视频 dropped 数(核对 07/04 高浓度)。
5. ≥5 seed 训练(sweep, `dropout=0.3/wd=1e-4`)→ 独立 `.pt`。
6. diag 四关 + median → 聚合 mean±std+min。
7. **判决性对照 Step0**:关1/关4 是否上移 → 裁定放行 / 否 → 选项 B fallback。
8. 画廊重抽检(892 盲区)+ 误删核查。
9. 写 Step2 结果报告 → 交 cc。

## 8. cc 提示:难例可重标(Step3 准备, 不阻塞本步)

若过滤后 04(walk→stand 判反)/ 07(信号太弱)训练样本仍识别困难, 建议 Jacob 抽审这些 crop 的
真值(是否上次漏标), 作为 Step3(04 色相 + GT 审查 / 07 分辨率边界)准备的一部分。本步不阻塞。

---
**状态**:本计划为 **待 cc review 的方案文档**;cc 放行前 wb 不写任何代码 / 不跑训练。
已据 cc(B) 几何保持 + cc ruling `62aeaf1`(≥5 seed mean±std+min)+ `negquality-null-pivot-step2` 裁定逐条承接。
