# CC 派活:①wb 跑 A 消融(负A/lit干扰负样本)②qw 诊断选灯精度天花板

> 出自 cc(arbiter/coordinator)。承 `2026-08-02-cc-verify-selection-quality-report.md` 的 gate FAIL 裁定 + Jacob "按推荐顺序、给 qw 派活"。
> 推荐顺序:①A 消融(治误绿回退/2·8 排序错)②选灯精度 40.8% 是否候选生成天花板 ③路线取舍。
> **①派 wb(主力/重训),②派 qw(只读诊断,并行,快,gate ①的解读)**。两任务写不同产物、不冲突。

---

## 红线(两人都必须遵守)
- **gate 不过不接线**:worst-seed 漏绿 ≤80 硬约束 **且** 误绿 ≤ base 2.19%(扣05);任一不满足=净回退,不接 `select_gtfree`。
- **不覆盖既有基线缓存/报告**:B-only 的 `models/governing_disc/{fold,tau,rows}_*_s1.*`(55份)与 `docs/reports/2026-07-31-wb-selection-quality.md` 锚着 FAIL 裁定的对照,**新任务一律写新路径**(见下)。别覆盖 `ped_signal.pt`。
- LOVO 去循环、GT 不进推理、多 seed 报 worst-seed(min)、无 per-video τ、权重不进库(`*.pt` 已 gitignore)。
- scoped git add(禁 `-A`)、提交签名 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`、各自 worktree 隔离、先诊断后修、原子写(tmp+os.replace)。
- **cc 会对两份产出 bit-for-bit 独立复核**(不采信聚合代码,从原始行/候选重算),再裁定。

---

## 任务 ① — wb:A 消融(引入同帧 lit 干扰负样本)

**目标**:检验加 neg_a(governing 帧中 IoU<0.3 的候选=反射/信号灯背面/平行斑马线灯等**点亮态**干扰)能否治住 B-only 的双回退——误绿 3.56%>base 2.19% + 2/8 排序错(lit 帧选错框)。这是 R1 预登记消融(`governing_disc.py:8` "负A仅消融、带A vs 不带A 各跑LOVO"),非盲改。

**改动(新增路径,不动 B-only)**:
1. [eval_selection_quality.py:245](scripts/eval_selection_quality.py#L245) 加 `--use-negative-a`(默认 False)。
2. [_load_or_train:188-199](scripts/eval_selection_quality.py#L188-L199):True 时 `pos, neg_b, neg_a = build_crop_dataset(sub, use_negative_a=True)`,`train_model(pos, neg_b + neg_a, seed=seed)`。
3. **缓存/报告新路径**(硬要求,别碰 s1):`fold_{V}_seed{seed}_A.pt`、`tau_{V}_seed{seed}_A.json`、`rows_{V}_seed{seed}_gw3_s1A.jsonl`;报告 `docs/reports/2026-08-03-wb-selection-quality-negA.md`。
4. 全 5-seed LOVO(`--seeds 0,1,2,3,4 --governing-weight 0.3`),报告主标尺同旧格式 + worst-seed。

**先诊断(全量前必做,防 neg_a 是噪声/坏框)**:
- 逐折数 `len(neg_a)`(像 neg_b 1→964 那样),确认非空、量级合理;
- **抽样存 PNG 肉眼看 neg_a**:docstring(`governing_disc.py:8` / 122-126)明警 neg_a "可能含平行斑马线**真灯**→噪声"。若 neg_a 混进真·点亮行人灯,等于教模型拒真绿 → 漏绿会**恶化**。抽 ~20 张确认是干扰非真 governing 灯;混入比例高就上报,别硬跑;
- 单折 10正10负(含 neg_a)过拟合分离 sanity(pos→正 logit / neg→负 logit 能分开),同修坐标 bug 时的 `1ef9c45` 手法。
- **诊断结论先发 cc/Jacob**,neg_a 质量过关再铺全量 5-seed(省算力,也因 qw 的天花板结果可能改写值不值得)。

**交付**:诊断小结(neg_a 计数+抽检)→ 全量报告 → 交 cc 复核。**gate 不过=不接线,如实报双回退。**

---

## 任务 ② — qw:选灯精度 40.8% 天花板诊断(只读,快,并行)

**要回答的问题**:sel_prec=40.8% 低,是**排序错**(候选池里有 IoU≥0.3 的好框但没被选中 → 判别器/A消融有 headroom)还是**候选生成天花板**(候选池根本没好框 → 判别器救不了,得回 prior偏框 [[prior-misframe-rootcause]])?缓存 rows 只存 best_cand、判不出,需重建全候选池。

**新脚本** `scripts/diag_candidate_recall_ceiling.py`(只读、无训练、model/seed 无关,一趟 399 帧):
- 逐 governing 帧(**与 [eval_video:118-124](scripts/eval_selection_quality.py#L118-L124) 同过滤**:`gov_boxes` 非空;`gcolors <= {"unclear"}` 的 UNKNOWN 帧跳过;无候选帧计入分母但天花板贡献 0),重建候选:
  ```
  res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
  yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
  hsv_px  = [s["box"] for s in det._candidates(frame)]
  cands   = build_candidates(yolo_px, hsv_px, W, H)   # 归一化同 eval_video:103
  ```
  复用 `gd._read_frames_at` / `gd._lazy_yolo` / `gd._cfg_tl` / `redlight.models.ped_light_selector.iou`,口径与 canonical 完全一致(去循环)。
- 每帧算 `ceil_iou = max over cands ( max over gov_boxes iou(cand_norm, gb) )`。
- **报告**(`docs/reports/2026-08-03-qw-candidate-recall-ceiling.md`):
  1. **候选召回天花板@IoU≥0.3** = 有候选 IoU≥0.3 的 gov 帧占比(=sel_prec 的理论上限);
  2. 逐视频分解(尤其 05 sel_prec=1.3%、01=20%、09=21.2%、10=6.7% 这几个低分视频,看是候选缺失还是排序错);
  3. 零候选 gov 帧占比;
  4. 与实测 sel_prec 40.8% 对比 → 明确裁断:**ceil≈40.8% ⇒ 候选生成是天花板(转 prior偏框);ceil≫40.8%(如≥70%)⇒ 排序问题(判别器/A消融有救)**。

**约束**:只读诊断,不接线、不改生产、不训练。写脚本 + 报告即可,scoped 提交签名。
**交付**:天花板数字 + 逐视频 → 交 cc 复核(cc 会重算对拍)。

---

## 依赖/时序
- ② 快且 model 无关,**先出**;其结论决定 ① 全量值不值得(若候选生成是天花板,A 消融最多治误绿、救不了 sel_prec)。
- ① 的诊断阶段(neg_a 抽检)与 ② 并行;① 全量铺开前等 neg_a 质量结论(可与 ② 天花板结论一起看)。
- 二者产物路径互不重叠(① 写 `_A`/`negA`,② 写新诊断脚本+qw 报告),worktree 隔离即可并行。

---
**一句话**:wb 跑 A 消融(引 lit 干扰负样本治误绿回退,新缓存路径别碰 s1,neg_a 先抽检防混真灯,gate 不过如实报);qw 只读诊断选灯精度 40.8% 是排序错还是候选生成天花板(重建全候选池算 IoU≥0.3 召回上限)。两者交 cc bit-for-bit 复核,gate 不过不接线。
