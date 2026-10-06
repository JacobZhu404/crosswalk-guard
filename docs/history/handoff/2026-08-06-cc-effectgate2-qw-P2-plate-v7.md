# cc 效果 gate 复审(第二轮):qw P2 车牌回填 v7.1(3cc0859)— **PASS(送回缺陷已修,跨线 carve-out)**

> 审核对象:`3cc0859`(plate-fix)— `_episode_plate` 加「ED≤1 变体系质心 span」空间约束 + docstring 修正 + 新验收 harness `qw_plate_events_report.py`
> 署名:cc(效果 gate/独立复核)— 遵 [[measurements-disagree-find-the-bug]]、[[qw-plan-execute-loop]] 效果 gate、[[b2-tracking-fragmentation-blindspot]]、承第一轮送回 [[qw-p2-plate-effectgate-fail]](`f26c91f`)
> 复核环境:cc 独立 detached worktree `cc-verify-plate@3cc0859`(symlink input_video/models),生产 `configs/config.yaml`(v2/box)端到端亲跑 11 视频。
> **结论:第一轮送回的核心缺陷(违章05 在真事件内回填别车牌 京N541E6 = 误罚)cc 生产 v2/box 独立复现下已修——05 现回填空串。真事件误罚=0(6 真事件全为命中或安全空串),命中 5/12 无回退,加性/回归干净,docstring 与实现一致。唯一残余误罚在违章01 负例 FP,属灯态线(wb #3 已过 plan-gate #5 授权),非车牌逻辑缺陷 → carve-out。PASS,车牌线 P1/P3 解冻。**

---

## 1. 决定性证据:生产 v2/box 逐事件全表(cc 亲跑复现)

| 视频 | hv | 回填牌 | GT violating | 判定 |
|---|---|---|---|---|
| 违章01 | **负** | 京ADF5307 | — | 误罚(负例FP,**灯态线** wb #3) |
| 违章02 | 1 | `''` | 京LNE560 | 空(安全 miss,宁缺毋滥) |
| 违章03 | 1 | 京ABV3428 | 京ABV3428 | **命中** |
| **违章05** | 1 | **`''`** | 京ADH9206 | **空 ✓ 送回条件②达成** |
| 违章06 | 1 | 京N2LE10 | 京N2LE10 | **命中** |
| 违章07 | 1 | 京Q5D2N8 | …京Q5D2N8 | **命中** |
| 违章08 | 1 | 京ACD5358 | 京ACD5358,… | **命中** |
| 违章09 | 1 | 京AC63971 | 京AC63971,… | **命中** |
| 违章11 | 1 | 京AFW1222 | (GT 盲区) | 无法判定 |

违章04/10 无 confirmed 事件(04 已知灯态 FN,v6 出局;10 真负例正确不报)。**命中 5/12;真事件(is_violation=1)误罚=0;唯一非空非命中(01)是负例 FP 的次级产物。**

## 2. 送回六条逐条核销(cc 独立核验)

| # | 送回条件 | v7.1 状态 | cc 核验 |
|---|---|---|---|
| ① | 先复现分歧(别折中) | qw 认领:harness 曾把 other_plates 并入 GT 集致漏报;修正口径后自测误罚 2(01+05) | **PASS**——qw 定位为方法学 bug(非折中),与 cc [[measurements-disagree-find-the-bug]] 一致 |
| ② | **05 变空串或京ADH9206** | 05 → 空串(京N541E6 系被 span 挡;京ADH9206 全视频 0 次可读=P3 目标) | **PASS**——cc 生产 v2/box 亲跑复现 05='' |
| ③ | 真空间隔离 or 扩宁缺毋滥兜脏袋歧义 | 加「ED≤1 变体系归属 tid 质心 span ≤ 0.25×帧宽」:05 京N541E6 系跨 7 tid span 607px>480→挡;真牌同车碎片聚集(07=69px/09=221px)保留 | **PASS**——这是真空间隔离(用 track_samples box 质心),非空谈;hits 全保留证明未过杀 |
| ④ | 生产 v2/box 逐事件报回填 vs GT 全表含负例,硬闸误罚0 | 见 §1 全表;真事件误罚=0 | **PASS(真事件口径)**——见 §3 carve-out |
| ⑤ | docstring 与实现一致,死代码注明/删 | `_episode_plate` docstring 重写为实际三重约束(弃虚假空间检查表述) | **基本 PASS**——见 §4 残留小项 |
| ⑥ | P1/P3 此前不推进 | 保持 pending | **PASS** |

## 3. 唯一残余误罚(违章01)= 跨线 carve-out,非车牌缺陷

违章01 回填 京ADF5307(与上一轮 京N8ZK53 不同 = OCR 非确定性,同为负例 FP 上的次级产物)。**因果链**:01 是真负例,其 confirmed 事件是 observe prior 直采环境绿的**灯态 FP**([[01fp-falsegreen-fix1-disproved]]);车牌层在一个(错误的)confirmed 事件上正确地回填了最高支撑的静止车牌——**给定事件,回填行为正确**。根因是事件不该存在,归灯态线。

**wb #3 时序门控方案已过 cc plan-gate #5**([[wb-01fp-temporal-plangate5-passed]],本轮同步裁定),落地后 01 FP 事件归零 → 该错罚单自动消失。这正是我第一轮就记的正向耦合。**不因灯态线的 FP 扣押车牌线**——车牌逻辑本身无缺陷。

## 4. 残留小项(不阻断,记账)

- **consensus.box 仍是死代码**:v7.1 的 span 用的是 `track_samples[tid]["box"]` 质心,**不是** `plate_consensus` records 里的 box。`plate_consensus.update(..., box=...)` 仍存 box 但无人读;其 docstring「供回填约束用」仍轻微误导。建议下轮:要么删 consensus.box,要么注明是 P3 预留。**不阻断本 gate**(真空间约束已由 track_samples 实现)。
- **违章11 京AFW1222 = GT 盲区**:GT 无牌标注,无法核验真伪 → 不可核验的自动罚单,产品层须知(潜在但不可判定)。
- **命中 5/12**:与 cc 第一轮 5/12 一致(qw 上轮报 6/12 差 1 = OCR 噪声,非本改)。

## 5. 加性 + 回归(cc 亲验)

- **加性**:`plate_consensus._recompute` 投票/去重仍只读 text/conf/ts(cc 读码坐实,line 8-26),span filter 在下游 `_episode_plate` 对候选 agg 字典过滤,**不污染 ED 投票** → 加性成立。
- **回归**:cc 亲跑 `pytest tests/` = **349 passed / 1 failed**。唯一失败 `test_mine_classifier_retrain.py::test_class_balance_exempt_neg`(违章07 off 占比 0.15∉[0.2,0.6])= **wb 灯态判别器挖矿线**([[light-classifier-retrain]]),依赖挖矿数据 artifact;**铁证:`3cc0859` 仅改 cli.py + 新 harness,不碰该测试及其数据** → 非 qw 回归,环境/wb 线问题。

## 6. 裁定

**PASS(效果 gate 第二轮)。** 第一轮送回的 05 误罚缺陷 cc 生产复现下确已修复,机制是真空间隔离(变体系质心 span)且不过杀命中;加性/回归/docstring 干净。**车牌线 P1/P3 解冻,可推进。**

**放行附条件(记账,非阻断)**:①最终「全口径误罚=0」的收口以 wb #3 落地(消 01 FP)为前置——车牌线不为此扣押,但 Jacob 排期建议 wb #3 先行;②下轮清理 consensus.box 死代码/docstring;③违章11 盲区罚单在产品层标注不可核验。

## 7. 一句话给 Jacob

qw 的车牌回填第二轮:**我上轮送回的违章05 开错罚单(京N541E6)确实修好了——生产配置下 05 现在回填空串,我亲跑 11 视频复现;5 个命中(03/06/07/08/09)一个没丢,加性和回归都干净(唯一的测试失败是 wb 灯态挖矿线的,跟车牌改动无关)**。修法是真的空间隔离(按车牌的"同一块牌 ED≤1 变体系"落在哪些车上、这些车质心横跨多宽来判——真违章车的碎片挤在一起,05 那块别车牌横跨大半个画面被挡)。**唯一还在的错罚单是违章01,但那是灯态误报事件带出来的,不是车牌逻辑的错——wb 的 #3 时序门控方案我这轮也批了,它落地后 01 这张错罚单自动消失。** 判 PASS,车牌线 P1/P3 可以继续。

---
*署名:cc(效果 gate/独立复核)。证据=cc detached worktree@3cc0859 生产 v2/box 端到端亲跑 11 视频逐事件表 + 349/1 回归 + 读码证加性。承 [[qw-p2-plate-effectgate-fail]](第一轮送回)、[[b2-tracking-fragmentation-blindspot]]、[[wb-01fp-temporal-plangate5-passed]]、[[01fp-falsegreen-fix1-disproved]]。*
