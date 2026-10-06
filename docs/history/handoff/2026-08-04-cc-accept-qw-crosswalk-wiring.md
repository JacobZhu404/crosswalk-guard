# CC 验收 qw 斑马线 v2 接线(wiring-b1, f792856+1916166)— 三护栏全部 cc 独立复现 PASS, 批准 merge; 但消融坐实 denom=box 事件级零增益(+46% 全来自 v2 时序), box vs mask 现纯语义/鲁棒性之争交 Jacob(可逆一字段)

> 出自 cc(arbiter)。qw 隔离 worktree(`crosswalk-guard-wiring`/`wiring-b1`)实施接线, 交 cc 逐项独立验收。**cc 不采信 qw 报告, 亲读 diff + 亲跑 base 路径 + 亲重算消融 + 亲跑生产入口**: (1) 接线 diff 逐行核对=纯加性/与批准方案精确一致; (2) 护栏① cc 亲 `diff -rq` → run1==run2 ∧ 回退==run1 **均 0 diff**; (3) 护栏② cc 从 4 组产物 bit-for-bit 重算事件数 = qw 表精确吻合; (4) cc 亲跑 `run_video.py`(真生产入口, 不传参)于违章11 → 确认违规=1(v11 该为0)→ **默认路径确系 v2**; (5) 护栏③ 负结果诚实, Phase 2 缺口如实记。**验收 PASS, 批准 merge。** 但消融坐实 **denom=box 在 v2 下事件级零增益**——推翻"v2+box 联合是胜因"的方案/拍板措辞: **+46% 全部来自 v2 时序**, box 保留纯为语义/可视化一致且**对 mask 偏宽脆弱**(v11+box 实证 FP 7→11)。**box vs mask 现为 Jacob 的可逆一字段决策(默认接哪个), cc 给两面+推荐不代拍。**

## 1. 接线 diff 逐行核对(cc 亲读 `git diff e72f336 f792856`)
| 文件 | cc 核实 | 结论 |
|---|---|---|
| `configs/config.yaml` | 加 `version:"v2"`+`occ_denom:"box"`, 注释声明词表 v11/v2 对齐 eval_violations + version⊥method + 回退法 | ✅ 词表对齐(cc gate #1) |
| `cli.py:66` | `_occ_denom = occ_denom if not None else cfg.crosswalk.occ_denom(默认mask)` | ✅ 显式传参优先, 加性 |
| `cli.py:74-76` | `_cw_version=getattr(...,"v11")`; `crosswalk_detector if not None else (V2 if v2 else V11)` | ✅ 显式优先, fallback v11 |
| `cli.py:85` | Visualizer 改用 `_occ_denom`(原 `occ_denom or "mask"`) | ✅ 与引擎同源; HEAD 上 mask==mask 逐字节不变 |
| `eval_tracking.py:34` / `diag_vehicle_track_fragmentation.py:174` | 显式钉死 `CrosswalkDetector(cfg)`+`occ_denom="mask"`, 注释引 cc gate | ✅ 保 b2 口径(cc gate #0) |
| 引擎/检测器/dag | 零改动 | ✅ 与方案一致 |
**结论: 接线纯加性, 与 cc 批准方案(1d2a2e8)精确一致, 无夹带。**

## 2. 护栏① bit-identical —— cc 亲 diff, PASS
cc 亲跑 `diff -rq`(非采信 qw):
- **确定性地板**: `wiring_pre/run1` vs `run2` → **0 diff, 11/11**(先排除 torch run-to-run 非确定性, cc gate #2 要求)
- **回退**: `wiring_post/post`(config 回 v11+mask)vs `run1` → **0 diff, 11/11 逐字节**
- 逐视频事件(run1)= cc 773dff4 基线 01:1/02:2/03:1/05:3/06:1/07:3/08:1/09:2 = TP7/FP7/FN2 ✅
→ **接线 diff 完全归因接线本身; 回退开关逐字节复原。加性由构造证明(diff 显示 v11 路径代码零改) + cc 实证双确认。**

## 3. 护栏② 2×2 消融 —— cc 从产物 bit-for-bit 重算, PASS + 重要更正
cc 亲数 4 组 violations.csv 逐视频事件, 独立重算(非采信 qw 表):
| 组合 | cc 数得事件(逐视频) | 总/TP/FP/FN | F1 |
|---|---|---|---|
| v11+mask | 01:1 02:2 03:1 05:3 06:1 07:3 08:1 09:2 = 14 | 14 / 7/7/2 | **0.609** |
| v2+mask | 01:1 02:1 03:1 05:1 06:1 07:1 08:1 09:1 11:1 = 9 | 9 / 8/1/1 | **0.889** |
| v11+box | 01:2 02:3 03:1 05:1 06:1 07:1 08:3 09:6 = 18 | 18 / 7/11/2 | **0.519** |
| v2+box(=e2e 默认) | 01:1 02:1 03:1 05:1 06:1 07:1 08:1 09:1 11:1 = 9 | 9 / 8/1/1 | **0.889** |
**cc 重算 4 格与 qw 报告表精确吻合。** 关键:
- **v2 时序 = +46% 的 100%**: v11+mask→v2+mask ΔF1 **+0.280**(清 6 碎片 FP 7→1 + 救回违章11 0→1)。
- **v2+box ≡ v2+mask, 全 11 视频逐视频事件数**恒等(1,1,1,0,1,1,1,1,1,0,1)→ **box 在 v2 下事件级完全中性, 零增益**。
- **v11+box 负收益**(FP 7→11, 违章09 单独 6 事件): 全宽带 mask 下 box 分母把"任何停带内的车"判压线。

## 4. cc 独立生产入口复现(闭合"信 qw 产物"缺口)
cc 亲跑 `python scripts/run_video.py input_video/违章11.mp4 ...`(**真生产入口, 不传 detector/occ_denom**):
- config 默认读到 `version=v2 / occ_denom=box`(cc 亲验 config 行);
- 输出 **确认违规=1**(track16, green, pedestrian, [18.0-28.26])。
- 违章11 在 v11 下=0(FN), v2 下=1(救回)→ **cc 独立跑出 1 = 默认路径确系 v2**, 不经 qw 任何脚本。**端到端接线闭环 cc 亲证。**
- e2e 全量(qw)8/1/1=0.889 与 cc 773dff4 逐视频吻合(02/03/05/06/07/08/09/11 TP, 01 既有 FP, 04 灯态 FN, 10 干净)。

## 5. 护栏③ 漂移探针 —— 负结果诚实, 不阻塞
- 四档(static/jitter/pan 0.5px/pan_fast 2px, cc gate #3 要求的单向 pan 已含且加码 pan_fast 累计整幅扫过)**均未现拖影膨胀**(末/首面积比 pan_fast 与 static 同量级 0.97-1.00)。机制: `_fit_trapezoid` 行 5-95 百分位裁剪结构性抑制运动弥散。
- **诚实局限如实记**: 合成仿射 ≠ 真实镜头运动; 真实运动泛化未验证 → **Phase 2 独立立项交 Jacob**; 报告**未宣称**泛化到运动镜头(守 cc gate)。固定机位生产(11 视频全监控)不触发, 接线安全。
- cc 判: 负结果对接线**有利**(running-max 拖影即便合成 pan_fast 也不发作), 不阻塞; Phase 2 缺口定位准确。

## 6. 验收裁定
**✅ 全部三护栏 cc 独立复现 PASS + 端到端默认路径 cc 亲跑确证 v2。接线实现干净、加性、可逆、与批准方案一致。批准 wiring-b1 merge 回 main。**

## 7. 但需 Jacob 拍板一件事: denom=box vs mask(消融逼出的新信息)
拍板接线时措辞是"v2 时序 + denom=box + box_overlap=0.20 **联合**是胜因"。**消融证伪了 box 那半**:
- **box 事件级零增益**(v2+box ≡ v2+mask, 全 11 视频)。+46% 全在 v2 时序。
- **box 对 mask 偏宽脆弱**(v11+box 实证 FP 7→11): box 分母 = 车足迹落斑马线比例, mask 偏宽→更多车"在斑马线上"→**更多误报**; 而 mask 分母 = 车占斑马线面积比例, mask 偏宽→分母变大→比例降→**自归一化更保守**。→ **万一 v2 在更难场景/真实运动下 mask 偏宽(护栏③只证合成不发作, 未证真实), box 是更 FP 脆弱的一侧。**
- **box 语义更正**: "车足迹压斑马线"是违章的字面定义, 且与 Visualizer 标注一致; 小车停大斑马线上 box 判得中, mask 可能判漏。

**这是 Jacob 的产品判定(D2 语义反转本就他拍), cc 给两面不代拍**:
| 选项 | 现 11 视频 | 未来/难场景风险 | 语义 | 操作 |
|---|---|---|---|---|
| **v2+box**(现接线, honoring 拍板) | 0.889(与 mask 同) | mask 偏宽时更多 FP | 车足迹压线=违章字面义 ✅ | 已实现, merge 即用 |
| **v2+mask**(回退半步) | 0.889(相同) | mask 偏宽自归一化更保守 | 占斑马线面积比(D2 原义) | config 一字段 `occ_denom:"mask"` |
- **两者可逆一字段**(cc 亲验: `violation_engine.py:144` box/mask 阈值解耦, `cli.py:66` 单点读 config, 无隐藏耦合)。改哪个都是改一行 config, 随时可翻。
- **cc 推荐**: **先 merge v2+box(不阻塞真正的胜因 v2 时序), box→mask 作为独立可逆决策交 Jacob 随时定**。纯鲁棒性看 mask 略稳, 纯语义看 box 更正; 现 11 视频零差异, 无理由为一个今日无差异的决策卡住 v2 的 +46%。

## 8. 红线
- cc 本次只读复核 + 独立复现(亲跑 run_video.py 违章11 产物在 worktree `data/output/cc_verify_11`, gitignored, 未动 qw 任何文件/产物)。
- merge 授权来自 Jacob 接线拍板 + cc 三护栏验收全过; **qw 按计划 merge wiring-b1→main**(scoped, 签名)。
- box vs mask 默认值 = Jacob 拍(可 merge 前先 flip config, 也可 merge 后随时 flip, 皆一字段可逆)。
- 真实运动镜头泛化 = Phase 2 独立立项, 接线不宣称。

---
**一句话**: qw 斑马线 v2 接线三护栏 **cc 全部独立复现 PASS**——接线 diff 逐行=纯加性/合批准方案; 护栏① cc 亲 diff run1==run2 ∧ 回退==run1 均 0-diff; 护栏② cc 从产物 bit-for-bit 重算 4 格(0.609/0.889/0.519/0.889)= qw 表精确吻合; cc 亲跑 `run_video.py` 违章11(真生产入口不传参)出确认违规=1 → 默认路径确系 v2, 端到端闭环 cc 亲证; 护栏③ 漂移负结果诚实记 Phase 2。**批准 merge wiring-b1→main**。但消融坐实 **denom=box 事件级零增益(+46% 全来自 v2 时序), box 保留纯语义/可视化且对 mask 偏宽脆弱(v11+box FP 7→11)**——推翻"v2+box 联合胜因"措辞; **box vs mask 默认值是 Jacob 可逆一字段决策**(cc 推荐先 merge v2+box 不卡胜因, box→mask 随时可 flip), cc 给两面不代拍。
