# cc 效果 gate 裁定:qw P2 事件车牌安全回填(4c23859)— **送回(FAIL)**:误罚=0 硬闸未复现,违章05 回填非 GT 牌京N541E6(生产 v2/box + v11/mask 双配置稳定复现)

> 审核对象:`4c23859`(plate-fix 分支)— `_episode_plate` 约束回填改写 + `plate_consensus.update` 加 box + `dag.py` 传 box + 3 个 P2 单测
> 署名:cc(效果 gate/独立复核) — 遵 [[measurements-disagree-find-the-bug]]、[[eval-methodology-gap-overfit]]、[[qw-plan-execute-loop]] 第二关(效果 gate)、[[b2-tracking-fragmentation-blindspot]]
> 复核环境:cc 独立 detached worktree `cc-verify-plate@4c23859`(symlink input_video/models),两套配置端到端亲跑。
> **结论:P2 的自设硬闸「误罚=0」cc 独立复现下 FAIL。违章05 回填 `京N541E6`(GT 违章牌=`京ADH9206`,京N541E6 是别车)——生产 v2/box 与 eval 默认 v11/mask 两配置、含碎片化都稳定复现。加性证明/回归/违章02 修复 PASS,但硬闸破 → 送回。**

---

## TL;DR
| 检查项 | 结果 | 判定 |
|---|---|---|
| 加性(box 不动 ED 投票) | `_recompute` 只读 text/conf/ts,从不读 box | **PASS**(读码坐实,逐 bit 不变) |
| 回归(unit+integration) | 341 通过;仅 3 个缺权重(.pt gitignored,wb 线)失败=环境非本改 | **PASS** |
| 违章02 修复(原开错罚单) | 不再回填过路车 `京A14672`,现为空串(宁缺毋滥生效) | **PASS**(记 qw 一功) |
| 设计偏离(去空间锚,改 stationary+全局帧数) | 意图正当(单锚杀二次违章车 07 / 牌全局读取语义)但**替代约束对 b2 脏袋过弱** | 部分正当,见 §3 |
| **误罚=0(P2 硬闸)** | **违章05 回填 `京N541E6` ∉ GT;违章11 回填 `京AFW1222` 而 GT=车牌看不清** | **FAIL** |
| 命中率 | 生产 v2/box **5/12**(03/06/07/08/09);qw 报 6/12 | 差 1,疑 OCR/环境噪声,非本裁定焦点 |

---

## 1. 复现方法(cc 亲跑,双配置)
- 隔离 detached worktree(不碰 qw 的 crosswalk-guard-plate,不扰 main/其它 agent),`4c23859` checkout,symlink `input_video/`+`models/`(worktree 缺 gitignored 大文件)。
- **生产配置** `--detector v2 --occ-denom box`(= `configs/config.yaml` 默认,[[crosswalk-v2-b1-c4-passed]] Jacob 拍板现状):全 11 视频端到端。
- **eval 默认配置** `--detector v11 --occ-denom mask`(`eval_violations.py` argparse 默认):交叉对照。
- 逐事件回填牌 vs `datasets/gt/events.csv` `violating_plates` 人工比对(harness 的 `plate_hits` 只算命中、**不报误罚**,须人工逐条查)。

## 2. 决定性证据:违章05 误罚(双配置稳定)
| 配置 | 05 事件 | 回填牌 | GT 违章牌 | 判定 |
|---|---|---|---|---|
| v2/box(生产) | tid=11 [0.67,64.51] | **京N541E6** | 京ADH9206 | 误罚(别车牌) |
| v11/mask(碎片) | tid=25 [26.26,46.2] | **京N541E6** | 京ADH9206 | 误罚(碎片化后仍复现) |

GT `违章05` note:「黑车占少+**主动后退不算** 白车占多静止=违章」,唯一违章牌=白车 `京ADH9206`(且 `京ADH9206` 是 qw 自己 P3 立项的「难读需 ROI 重试」牌,P2 阶段本应读不到 → 正确行为=空串)。系统却回填了另一块易读的别车牌 `京N541E6` → **开错罚单**。这正是我在方案 gate 预警、qw 自己在 v1 用「误罚=0 硬闸」否掉过的同类问题(v1 违章03 回填非 GT 京ABV200 曾被 qw 判误罚)。同一判据下 05 就是误罚。

### 2.1 根因(cc 内联探针 `/tmp/probe05.py` 亲测,已落地证据)
违章05 confirmed 事件是 **b2 过合并的 22-member 脏袋**:
```
EVENT tid=11 members=[1,2,3,9,11,7,16,17,19,25,26,29,30,33,43,45,48,49,52,57,69,82] plate='京N541E6'
  tid 17: n=342 sta_ratio=0.94 cx=1478   tid 2:  n=266 sta_ratio=0.82 cx=1369
  tid 26: n=237 sta_ratio=0.84 cx=310    tid 9:  n=85  sta_ratio=0.88 cx=1482
  tid 25: n=140 sta_ratio=0.74 cx=544    tid 11: n=91  sta_ratio=0.81 cx=410 ...
```
**约 12 个 member 通过 `sta_ratio>=0.6`,质心横跨全画面 cx=130→1773**(拥堵路口停/慢车遍地)。qw 的替代约束(窗口内静止占比 + 全局帧数)**无法在脏袋内区分违章车与其它静止车**,于是回填「全局最高读数的牌」= 别车 `京N541E6`,而非难读的真违章牌。**宁缺毋滥没有对「脏袋内多候选歧义」这一情形触发** → 破闸。

## 3. 设计偏离评估:意图正当,但落地留漏 + 文档说谎
- **正当部分**:qw 弃「牌读取时间窗」改「车在窗口内静止」是对的——违章车牌常在事件窗口外才看清(GT 03 银灰车 143s、06 黑车 46s);弃「单空间锚」也对——07 二次违章车 京Q5D2N8 会被单锚杀。这两条是 plan-execute loop 里合法的方案自我纠偏,cc 采信。
- **漏洞**:替代约束(stationary+全局帧数)**恰恰对 P2 立项要绕开的 b2 脏袋过弱**(§2.1)。P2 的立项前提(cf1d246)是「_episode_plate 能在 b2 过合并下安全回填」;05 证明当前约束不安全。b2 修根因是独立立项([[b2-tracking-fragmentation-blindspot]]),但 P2 的**安全兜底(宁缺毋滥)必须覆盖「脏袋歧义→空串」**,现未覆盖。
- **文档说谎 + 死代码**:`_episode_plate` docstring §(2) 明写「空间:plate 框与事件代表 track 车辆框时空关联(0.4s 内车辆框内)」——**实现体从不读 `rec["box"]`,无任何空间逻辑**。`box` 被加进 `plate_consensus` records、经 `dag.py` 一路传入,却**无人消费=纯死代码**。docstring 描述了一个不存在的约束,误导维护者;而真能隔离脏袋的 box 信息正躺着没用。

## 4. 附带观测(非本闸失败项,但须记)
- **违章11**:回填 `京AFW1222`,GT=「车牌看不清」(unreadable)。人工无法证真伪 → 至少「不可核验的自动读数上罚单」,潜在误罚。
- **违章01(负例)**:wb 线的 01 FP 事件现被 qw 回填 `京N8ZK53` → 一张完整的错罚单。01FP 属 [[01fp-falsegreen-fix1-disproved]] wb 线,非 qw 牌逻辑之过,但**牌回填把 FP 放大成完整错罚单**,收口 01FP 前须知。
- **命中率 5/12 vs qw 6/12**:差 1(疑 02 京LNE560 在 qw 环境命中 / OCR 非确定性)。非焦点,但提示 qw 的验收数字未在生产 v2/box 下被 cc 完全复现。

## 5. 方法学纠偏(写给 qw,防再栽)
**qw 把「miss(空串)」与「safe」混为一谈。** 非空回填 ≠ GT 违章牌 = **误罚(错罚单)**,不是无害漏检。`eval_violations.py` 的 `plate_hits/plate_total` 只报命中、把 05 记成「非命中」而非「错牌」,所以「误罚=0」很可能是**没有逐条把每个非空回填牌比对 GT**得出的。**误罚必须:对每个 confirmed 事件(含负例/FP 事件),把非空回填牌逐条比对 GT `violating_plates`,任一不匹配即误罚**,且在**生产 v2/box** 下测。

## 6. 裁定 + 送回条件(过下一次效果 gate 的硬门槛)
**送回。不 merge plate-fix。** qw 须:
1. **先复现分歧**(measurements-disagree):在生产 v2/box 下亲跑违章05,确认回填=京N541E6 并承认误罚≠0;若 qw 环境得空串/京ADH9206,须定位为何与 cc 双配置结果不同(别折中、别选边,找 bug)。
2. **堵 05 类漏**:二选一——(a) **真正实现 docstring 已承诺的空间/on-crosswalk 隔离**(用已铺好的 box 把脏袋收敛到压线违章车足迹,按 finding-1 允许多违章车),或 (b) **扩宁缺毋滥**:脏袋内出现「多个静止 member 携带不同高频牌」的歧义时回填空串。**05 必须从京N541E6 变成空串或京ADH9206。**
3. **改 docstring** 与实现一致(删掉不存在的空间约束描述);box 若保留须注明是 P3/未来空间约束的前置铺垫(当前未消费),否则删。
4. **重测误罚**:按 §5 口径,生产 v2/box,逐事件报「回填牌 vs GT」全表(含负例 01/10),硬闸=误罚 0。
5. 硬条件不变:加性(已 PASS 无须重证)、回归、F1 不回退;命中率主指标以生产 v2/box 计。

**保留的功劳**:违章02 修复(消 京A14672 开错罚单)真实有效;加性与回归干净;弃时间窗/弃单锚的设计判断正当。送回的是「误罚=0 兜底未覆盖 b2 脏袋 + 文档/死代码不一致」,不是推翻整条回填思路。

## 7. 一句话给 Jacob
qw 的 P2 车牌回填:违章02 开错罚单确实修好了(记一功),加性和回归也干净;**但 qw 自设的「误罚=0」硬闸没通过 cc 复现——违章05 回填了别车牌 `京N541E6`(真违章牌是 `京ADH9206`),生产配置和 eval 默认配置都稳定复现**。根因是 05 事件被 b2 合成 22 车脏袋、十来辆停着的车都过「静止」门槛,qw 用「静止+全局帧数」选牌选中了最容易读的别车牌,而「宁缺毋滥」没在这种歧义下兜底为空。另外 docstring 写了个根本没实现的「空间约束」(box 铺进去却没人用=死代码)。**送回让 qw 先复现这个分歧、把 05 改成空串或正确牌、再按生产配置逐条重测误罚。** 车牌线其余部分(P1/P3)在这之前先别推进。

---
*署名:cc(效果 gate/独立复核)。证据=cc detached worktree@4c23859 双配置端到端亲跑 + 内联探针 probe05 亲测脏袋 sta_ratio。承 [[b2-tracking-fragmentation-blindspot]](脏袋根因立项)、[[plate-line-diagnosis]]、[[measurements-disagree-find-the-bug]]、[[eval-methodology-gap-overfit]]。*
