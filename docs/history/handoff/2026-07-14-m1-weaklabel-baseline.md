# Handoff: M1 行人信号灯分类器 — 弱标签基线 + .pt 主路径落地 (2026-07-14)

> 交接人: Senior Developer (senior-dev agent)
> 承接: 做 M1 评测 / 画廊校验 / 扩视频标注 的后续 agent
> 关联提交: `49db33e` (refactor: .pt 主路径, 移除 onnx 硬依赖) + 本日后续提交

## 0. TL;DR
- M1 分类器现在**直接吃 PyTorch `.pt` 权重**(零下载, 不依赖 onnx 包), 运行时已由 `SignalStateClassifier` 按扩展名自动选后端(.pt→torch / .onnx→cv2.dnn 可选)。
- 训练脚本新增**平衡采样**(`--balanced`, 默认开), 缓解弱标签下 off 占 91.7% 的失衡; 弱标签训练用 `--no-verified-only` 开启(默认仍保守只用人工校验 crop)。
- 已训出 `models/ped_signal.pt` 弱标签基线模型, 并写了端到端灯态评测脚本 `scripts/eval_m1_light.py`(color vs ped_classifier 对比 `datasets/gt/light_states.csv`)。
- **注意**: `models/ped_signal.pt` 被 `.gitignore` 的 `*.pt` 规则排除, **不入库**(仓库约定大权重走外部存储)。本提交只含代码+文档; 新环境 clone 后按 §3 命令重训导出即可得到该模型。
- **LOVO 泛化基线数字、端到端 color/ped_classifier 对比数字见文末 §5(训练/评测跑完回填)**。

## 1. 本次改了什么(代码)
### 1.1 onnx 解耦(已提交 49db33e)
- `src/redlight/models/signal_state_classifier.py`: 新增 `_build_net()` 单一真相源(训练/分类共用架构); `SignalStateClassifier.__init__` 按扩展名选后端:
  - `.pt`/`.pth` → `torch.load(weights_only=True)` + `load_state_dict`(主路径, 零额外依赖)
  - `.onnx` → `cv2.dnn.readNetFromONNX`(可选, 供无 torch 环境)
- `scripts/train_ped_signal.py`: 主产物改 `models/ped_signal.pt`(`export_torch` 存 state_dict); ONNX 降级 `--export-onnx` 可选; smoke 自检改为验证 `.pt` 契约(零下载即通过)。
- `config.yaml`: `ped_signal_onnx` → `ped_signal_model`; `traffic_light.py` 同步读取。
- 实测: `scripts/train_ped_signal.py --smoke` 零下载通过(acc=1.00, 分类 ['walk','stand','off'] 全对); 单测 3 个全绿。

### 1.2 平衡采样 + 弱标签训练开关(本日未提交/待提交)
- `train_net(...)` 新增 `balanced` 参数: 每 epoch 对少数类(walk/stand)过采样、off 下采样到均衡 maj 数, 防模型退化成"全判 off"。
- CLI: `--balanced`(默认开) / `--no-balanced`; `--no-verified-only`(关掉默认开的 `verified-only` 门槛, 用弱标签 crop 训练)。

### 1.3 端到端评测脚本(本日新增 `scripts/eval_m1_light.py`)
- 复用 `diag_signal_timeline` 的逐帧逻辑, 对 02/03/04 分别用 `traffic_light.method=color` 与 `ped_classifier` 跑检测器, 切连续段, 与 `datasets/gt/light_states.csv` 分段真值比 accuracy / macro-F1。
- 用法: `python scripts/eval_m1_light.py`(两种都跑) 或 `--method color|ped_classifier`。

## 2. 数据现状(弱标签)
- `datasets/ped_signal/labels.csv`: 10205 张 crop, **全部 `verified=0`**(弱标签), 标签分布:
  - off=9363 (91.7%), stand=517, walk=325
  - 按视频: 违章02(off1656/stand40/walk88), 违章03(off5088/stand324/walk232), 违章04(off2619/stand153/walk5)
- 弱标签来源: `extract_crops` 仅在"可见灯态段" + 先验位置附近候选标 walk/stand, 其余标 off。**标签质量依赖先验是否真指向行人信号灯**(见 §4 E20 风险)。

## 3. 怎么复现训练
```bash
# 弱标签 + 平衡采样 LOVO(留一视频交叉验证, 代理"新手机视频"泛化)
python scripts/train_ped_signal.py --no-verified-only --balanced --epochs 40
# 全量弱标签训练并导出 models/ped_signal.pt
python scripts/train_ped_signal.py --no-verified-only --balanced --epochs 40   # 默认 --out models/ped_signal.pt
# 端到端灯态对比(color vs ped_classifier)
python scripts/eval_m1_light.py
```
注意: 项目运行时是**系统 Python 3.11.9**(`C:\Users\windows\AppData\Local\Programs\Python\Python311`), 不是 WorkBuddy managed 3.13.12(无 torch)。

## 4. 已知风险 / 坑(接手前必读)
- **R1 弱标签质量**: walk/stand 由 GT 段 + 先验位置自举, 若某视频先验指错(指到机动车信号而非行人信号, 见 E20), 则该视频标签整体偏错。违章04 的 walk 仅 5 张 → 该类在该视频几乎无正样本。
- **R2 off 类语义**: off = "不在先验附近的候选", 含大量背景亮斑。训练时 off 是合理负类, 但推理时 `_detect_ped` 把所有候选(含 YOLO/HSV 亮斑)喂分类器, 与训练分布一致。
- **R3 需人工画廊校验**: 当前 crop 全 verified=0, 保守路径(`--verified-only` 默认开)会直接退出。要拿到可信模型, 需在灯态画廊里人工标 verified=1 一批 crop, 再 `--verified-only` 训练。
- **R4 视频覆盖少**: 仅 02/03/04 有 GT 灯态段 → LOVO 只有 3 折, 泛化证据弱。扩视频(05~11)标注后才能更稳。
- **R5 并行 agent 协作**: 另一 agent 在做车牌评测(根目录 `handoff.md` 是其交接); 本日改动仅限 M1 相关文件, 未碰其文件。提交走主干(用户 2026-07-13 起规定默认主干, 仅大功能拉分支)。

## 5. 结果与指标(训练/评测跑完回填)
- LOVO 平均 test_acc = ___ (各折: 02=___ 03=___ 04=___)
- 端到端灯态(02/03/04 合计): color acc=___ macro_f1=___ ; ped_classifier acc=___ macro_f1=___
- 结论: ped_classifier 相对 color ___ (改善/持平/退化), 原因 ___

## 6. 下一 agent 待办
1. 跑灯态画廊, 人工校验一批 crop 置 verified=1 → 用 `--verified-only` 重训(比弱标签更可信)。
2. 逐视频确认先验锁的是**行人信号**(走路图标/站立小人), 不是机动车信号(E20); 锁错则重标先验。
3. 扩 05~11 视频的灯态 GT 标注, 提升 LOVO 折数与泛化证据。
4. 决定 `traffic_light.method` 是否全局切 ped_classifier(目前仅验证 02/03/04 可行; 其他视频无行人信号时应回退 color, 建议按视频配置而非全局)。
