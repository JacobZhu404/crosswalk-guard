# wb 计划: Phase A — 灯态判别器重训数据集构建

> 承接 `docs/handoff/2026-07-20-cc-direction-classifier-retrain-dataset.md` (238b2e3) + 诊断 `docs/reports/2026-07-20-wb-diag-classifier-feasibility.md` (c4de17d)。
> **本文件是 Phase A 的构建计划, 先交 cc review 放行, 再动手挖数据。不改生产代码、不接线。**
> 阶段纪律: 先计划后挖, 先诊断后接线, B 不过绝不硬接。
> **✅ cc 已批准(2026-07-20), 含 3 处补修(A/B/C)+ 4 决策, 见 §10 与正文标注。批准后即进挖掘+画廊+TDD。**

## 0. 目标与成功标准(锚定, 别跑偏)
- **判别器在门控里的角色**: `observe()` 产绿出口前, 裁该绿区 ROI 送判别器 → 判"这次绿读数来自**真行人信号**还是 **impostor(背心/植物/反射/车反光等非信号绿)**"; impostor → 改判 unknown(恢复 D1 安全网)。
- **Phase A 成功 = 交付一份可训练、按视频切分无泄漏、类平衡、Jacob 抽检后的数据集 + manifest**, 使 Phase B 重训后**重跑可行性诊断能过**: 拒背心绿、保 06/07 暗绿、过 04 试金石、未见视频(val)泛化。
- **Phase A 不做**: 接线、改 `traffic_light.py`、改 `enforce_transition_limit`、碰 `light_priors.json` 坐标。

## 1. 标签方案(给 cc 拍板, 这里给推荐 + 备选)

### 推荐: 复用现有 3 类 walk/stand/off(零分类器代码改动, 风险最小)
- `walk` = **真绿信号**(GT green 段 + prior ROI 抠出)
- `stand` = **真红信号**(GT red 段 + prior ROI 抠出) — 用于抓"红灯被误读成绿"→ 恢复 unknown
- `off` = **impostor(非信号绿: 背心/植物/反射/车灯/信号外强绿斑)** + 少量背景正则(见 §3 边界说明)
- **门控映射**(Phase C 接线时): `observe()` 读绿 → 裁 ROI → 判别器:
  - `walk` → 保留绿
  - `stand` → 红信号被误读成绿 → 改判 unknown
  - `off` → impostor → 改判 unknown(恢复 D1)
- **复用面**: `SignalStateClassifier`(LABELS=[walk,stand,off], 3 输出)、`train_ped_signal.py`、画廊/apply 工具**几乎零改动**。仅数据集标签含义 + 新增 impostor 挖掘源。
- 语义等价于 cc 说的"真绿/真红/impostor 三类", 仅第三类名叫 `off`(既有代码约定)。

### 备选 A(清晰命名, 略增改动)
- 重命名 LABELS 为 `real_green/real_red/impostor`, 改 `signal_state_classifier.py` + `train_ped_signal.py` + 接线处。语义同推荐, 可读性更好, 但改动面更大。

### 备选 B(二分类 real_signal vs impostor)
- 更简单, 但**丢失颜色校验**(无法抓"红灯误读成绿"), 与 cc §0 "既门控又 validate 颜色" 冲突。**不推荐**。

> **wb 推荐 = 复用 3 类 walk/stand/off**(最小爆破半径、最大复用、等价于三类语义)。最终采哪个等 cc 在 review 时定。

## 2. 挖掘源与启发式(半自动, 降 Jacob 标注成本)

复用 `src/redlight/data_pipeline/ped_signal_dataset.py` 的 `extract_crops` / `crop_box` / `light_state_to_label` 纯函数 + `gt_lookup.state_at`(段灯态) + `configs/light_priors.json`(11 视频全有 prior, roi_px 见 §4)。新增一个挖掘脚本 `scripts/mine_classifier_retrain.py`, 产出 `datasets/classifier_retrain/{<video>/crops, labels.csv, manifest.json}`。

**全新构建(cc 补修 C)**: 旧 `datasets/ped_signal/`(仅 02/03/04, 已知坏)归档到 `datasets/ped_signal_legacy_238b/`, **不覆盖合并**——旧 `off`=暗信号背景, 新 `off`=impostor, 语义不同, 混了类就脏。Phase A 一律重新挖, 旧的只留档(其 02/03/04 真绿/真红理论上可复用为正样本, 但为避免语义混淆一律重挖)。

### 2.1 真信号正样本(标签 walk/stand)
- **来源**: `datasets/gt/light_states.csv` 中 `confidence=="confirmed"` 的 green/red 段(跳过 `occluded`/`inferred`——`extract_crops` 已做)。
- **抠法**: 走现有 `prior_roi_mode`(每帧 prior ROI 直抽 1 块, 标段灯态)。
- **硬覆盖(诊断暴露的失败点, 必须进训练集)**:
  - **06 / 07 暗绿**: GT green 段 + prior ROI 抠出(暗绿 low frac_v 也不过滤亮度, 让模型学"暗也认")。
  - **04 末段暗短绿 [42–43.2s]**: 仅 ~1.2s, fps=4 → ~5 帧, 全抠, 标 walk。**04 进 train**(让它学到短暗绿; Phase B 再诊断验证试金石)。
  - 各正例视频真红段(stand)照抠, 供颜色校验。

### 2.2 impostor 负样本(标签 off)—— 现有数据集**完全缺失**的部分
三类来源, 全部**只在"引擎读绿但非真绿"处挖**, 保证 off 类是**视觉绿但非信号**:
- **(a) 负例视频假绿**: 违章01(54 帧引擎绿)/违章10(14 帧引擎绿)的 prior ROI 抠图 → 背心绿/植物/反射。GT 负例, 引擎绿必为假绿。
- **(b) 483 帧系统性误绿**: 重跑 `diag_classifier_feasibility.py` 思路的扫描——对全 11 视频用 `observe()` 复现 0.889 管线; 凡 `obs=="green"` 且 `gt_lookup.state_at(ts)` **不是 green**(confirmed red/unknown) 的帧 → 裁 prior ROI 存为 impostor。即"引擎误绿帧"。
- **(c) 信号外强绿斑**: 全图 HSV 绿斑检测, 中心距 prior > radius 的强绿斑 → impostor 候选(背心/植物/车灯/反射)。这是比 483 帧更丰富的 impostor 来源, 覆盖未见视频的泛化。

### 2.3 背景正则(标签 off, **限量**)
- 保留现有 `prior_off`(prior 外随机区域)作正则, 但**上限 ≤ off 类的 20%**。理由: 纯暗背景 off 会诱导"暗=off"偏见 → 正是 07 暗绿被屠的根因之一。门控只在绿读时触发, 背景 off 非必要, 仅少量防过拟合。

### 2.4 挖掘可靠性边界(自动标签可能错, 抽检兜底)
- **可靠**: (a) 负例视频假绿、(b) GT confirmed≠green 的引擎绿、(c) 信号外强绿斑 —— 这些在"非信号绿"上高置信。
- **可能错**:
  - 段边界帧(GT red 但实际已转绿/未转)→ 误标 impostor; **cc 决策: 段边界帧全核(最易误标 impostor 处, §2.4 自点名)。**
  - 暗绿 06/07 抠图偏暗, 自动标 walk 是对的, 但需抽检确认 ROI 没偏出信号。
  - 04 短绿仅 5 帧, 少而珍贵, 全部进抽检重点。
- 启发式不依赖任何 per-video 硬编码坐标(只用 `light_priors.json` 的通用 prior 中心, 不改它)。

## 3. 按视频切 train/val(防过拟合, 红线)
- **铁律**: 整视频进 train 或 val, **绝不按帧混切**(按帧切会泄漏、测不出泛化——现有 pt 过拟合的直接教训)。
- **提议切分(待 cc review 防泄漏)**:
  - **val = {违章01, 违章07, 违章11}**(3 视频, 留出测泛化)
    - 07 = 暗绿最难关, **训练中完全未见** → 真·泛化试金石(保暗绿)。
    - 01 = 纯 impostor 负例视频, 训练中未见 → 测 impostor 拒识泛化。
    - 11 = 不同相机/外观, 训练中未见 → 测未见视频泛化。
  - **train = {违章02,03,04,05,06,08,09,10}**(8 视频)
    - 04 进 train(学短暗绿); 06/09 进 train(学先前被屠的绿); 10 进 train(impostor 源); 02/03 进 train(既有外观基础)。
- **Phase B 再诊断跑全 11**(含 val)验证门控真泛化; train/val 仅用于训练期早停/过拟合侦测。
- `lovo_folds` 现有留一视频 CV 可复用作出训练期交叉验证(见 `train_ped_signal.py`)。

### 3.1 最终上线模型重训范围(红线补充 · cc 补修 A)
- **val={01,07,11} 是泛化"估计"**(Phase B gate 用留出集验证), **不是最终模型的训练范围**。
- **最终接进 `observe()` 的模型(Phase C), 必须在 train+val 合并的「全 11 视频」上重训**(标准 train/val → 全量重拟合), **绝不接 train-only 模型**——否则上线的是个从没见过 07/01/11 的模型。
- 流程: Phase B 用 train 训 → 在 val 验证泛化(跨过 07 暗绿 + 01 impostor + 11 未见相机)→ 估计通过后, **合并全 11 重训** → 该全量模型才进 Phase C 接线 + 端到端 eval 无回退。

## 4. 类平衡目标
- 目标 mined 计数(每视频): `walk : stand : off ≈ 1 : 1 : 1`(off 含 impostor + ≤20% 背景)。
- 各正例视频绿色段通常短于红色段 → 红多绿少; 用 **对每视频绿段过采样**(同段多抠/微抖动增强)拉平 walk/stand; impostor 用 §2.2 三源凑足。
- 训练侧 `train_ped_signal.py --balanced` 已对少数类过采样、off 下采样 → 即使 mined 略偏, 训练也重平衡。**但数据集本身需大致均衡**, 否则 val 指标失真(manifest 明示计数供 cc 核)。
- `light_priors.json` roi_px: 02/03/05/06/08/10/11=160, 04=220, 07=160, 09=160, 01=160 → 抠图尺寸统一按各视频 prior_roi_px(与现训练分布同尺度)。

## 5. Jacob 抽检工作流(半自动, 不是从零标)
- **新画廊脚本** `scripts/make_classifier_retrain_gallery.py`(复用 `make_ped_signal_gallery.py` 骨架, 支持新 3 类 + **分层抽样展示**):
  - 默认策略 `sample`: **100% impostor(off) + 100% 06/07 暗绿(walk) + 100% 04 短绿 + 100% 段边界帧 + 其余(walk/stand 主体)随机抽 20%**(cc 决策: 段边界帧全核, 最易误标 impostor)。把 Jacob 工作量压到"重点核边界 + 抽样确认"。
  - 全量模式 `--all` 供 Jacob 想全看时。
  - 支持改标 walk/stand/off + 删除(难判废图), 导出 `classifier_retrain_feedback.json`。
- **合并** 扩展 `apply_ped_signal_feedback.py`(或新 `apply_classifier_retrain_feedback.py`)读 feedback → 把对应 crop 的 `label` 改正、`verified=1` 写回 `labels.csv`。
- **训练只用 verified=1**: `train_ped_signal.py --verified-only`(默认开)。
- 抽检是**纠错兜底**: 自动标可能错(假绿帧里藏暗真绿 / 真绿 ROI 偏出信号), 抽检闭环。

## 6. 数据 manifest
- `datasets/classifier_retrain/manifest.json`(git 版本化, 小):
  ```json
  {
    "schema": "walk/stand/off",
    "split": {"train": [...], "val": [...]},
    "counts": { "违章02": {"walk":N,"stand":N,"off":N,"verified":N}, ... },
    "total": {...},
    "impostor_sources": {"neg_videos":["违章01","违章10"], "false_green_scan":N, "outside_prior_green":N}
  }
  ```
- 供 cc 核: 逐类逐视频计数 + split 无泄漏 + 类平衡。

## 7. TDD(挖矿/切分先写测, 再挖)
- `tests/test_mine_classifier_retrain.py`:
  1. **无泄漏**: 任一 video 不同时出现在 train/val(`set(train)∩set(val)==∅` 且并集=全 11)。
  2. **类平衡(豁免负例 · cc 补修 B)**: 正例视频 walk/stand/off 均 >0 且 off 占比 ∈ [0.2, 0.6](防暗背景偏见); **负例 01/10 全红无真绿 → walk=0 是正确, 必须豁免 walk>0 要求, 只要求 stand>0 + off>0**。不改这条, TDD 一跑就红。
  3. **impostor 自动标只对"非真绿"**: 抽查 impostor crop 的 ts, 断言 `gt_lookup.state_at(ts)!="green"`(负例视频全帧放开)。
  4. **硬覆盖**: manifest 含 06/07 暗绿 walk >0、04 短绿 walk >0。
  5. **ROI 尺度一致**: 抠图尺寸 = 各视频 prior_roi_px。

## 8. 交付物(Phase A 收尾)
- `datasets/classifier_retrain/{labels.csv, manifest.json, <video>/crops/*}`(**全新构建**, 旧 `datasets/ped_signal/` → 归档 `datasets/ped_signal_legacy_238b/`, 不合并, 留档)
- `scripts/mine_classifier_retrain.py`(挖掘) + `scripts/make_classifier_retrain_gallery.py` + 反馈合并脚本
- `tests/test_mine_classifier_retrain.py`(全过)
- 跑出数据集 → **Jacob 抽检**(分层抽样) → 交 cc review **split 防泄漏 / 类平衡 / 挖掘启发式可靠性** → 放行后进 Phase B 重训 + 重跑可行性诊断。

## 9. 红线(重申)
- 按视频 split 防过拟合(最重要); 类平衡; 全 11 覆盖。
- GT/eval 产物**只用于训练数据 curation, 绝不进生产推理**; 判别器不引入 per-video 硬编码。
- 不碰 `enforce_transition_limit`; 净回退不上; 不改 `traffic_light.py` / `signal_state_classifier.py`(仅数据集+挖掘+画廊脚本)。
- scoped git、署名 `Co-Authored-By: Claude Opus 4.8`、trunk main。

## 10. cc 批准记录(2026-07-20)—— 3 处补修 + 4 决策

### 3 处必须补修(已写死进正文)
- **A(§3.1)**: 最终接进 `observe()` 的模型须在 **train+val 合并全 11 重训**, 绝不接 train-only。val 仅作 Phase B 泛化估计; 估计通过即合并全量重拟合再上线。
- **B(§7 #2)**: TDD「每视频 walk>0」**豁免负例 01/10**(全红无真绿, walk=0 正确), 否则一跑就红。只对正例视频要求 walk>0。
- **C(§2/§8)**: 复用 walk/stand/off 前提 = **建全新数据集、别和旧混**。新 off=impostor, 旧 off=暗信号背景, 语义不同。**Q4 选归档旧集、全新构建**(不覆盖合并)。

### 4 个决策(已定)
1. **标签方案 → 采推荐(复用 walk/stand/off)**。最小爆破半径、零分类器改动; 附 C 约束(off=impostor+暗, 全新建)。
2. **val 切分 {01,07,11} → 批准**。06 留 train(学暗绿)、07 留 val(测暗绿泛化); 配 A(全量重拟合)兜底, 不会上线 07-盲模型。
3. **抽检采样率 → 批准**(impostor+暗绿+04 全核 + 其余 20%); **追加: 段边界帧全核**(§2.4 自点名最易误标 impostor 处)。
4. **旧数据集 → 归档** `datasets/ped_signal_legacy_238b/`, 不覆盖(留档 + 避免 off 语义冲突, 见 C)。

### 接受的风险(不阻塞)
- 07 在 val = 训练完全不见暗绿, 靠 06 泛化过去; 可能 gate 跨不过(06 暗绿与 07 不够像)。跨过=真 robust; 跨不过=暗绿多样性不足, 届时再议(更多暗绿源/更大模型), **绝不为过考题把 07 挪进 train**。
- 04 试金石可能因**融合(prior_flip 压掉末段绿)**而非分类器过不了, 按钢锭接受 04 继续漏。
