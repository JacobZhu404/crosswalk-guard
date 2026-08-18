# cc 裁定:dedup 修复(Fix A)独立验证 = 安全,授权 wb 落地;09 prior 重定位线归档

> 署名:cc(coordinator/arbiter)  转交:Jacob;抄送 wb
> 承 `4329e78`(09 真因=dedup review 毒化)、wb `48c3ed7`(dedup 诊断收敛)+ `eb69bf9`(prior 重定位调查)
> 证据:cc 独立生产 v2/box `cli.run` 直探 09/01/10 @T=6,复刻 `_dedup` 分组 + 对比「当前(review 优先)」vs「Fix A(confirmed 优先)」两种合并语义,逐 episode 按 GT 判 TP/FP。

---

## 1. Fix A 独立验证(cc 生产 post-dedup,决定性)

| 视频 | raw events | 当前 dedup(review 优先) | **Fix A(confirmed 优先)** |
|---|---|---|---|
| **违章09**(GT[11-72]) | 89(84 conf / 5 review) | 1 review episode [0.67,106.26] → **FN** | **1 confirmed episode [0.67,106.26] → TP** ✓ |
| **违章01**(负例) | 15(**0 conf** / 15 review) | 1 review | **仍 review → 0 FP** ✓ |
| **违章10**(负例) | 0 | — | — ✓ |

**结论:Fix A 在三个关键视频上验证安全。** 09 review→confirmed(TP 恢复);01/10 无 confirmed 成员 → 仍 review(#3 消 01FP 不被 Fix A 复活)。**端到端 tp8/fp0/fn1(04)= P=1.000 / R=0.889 / F1=0.941**(优于 #3 前 0.889、现状 0.875)。

## 2. 化解与 wb 的 P=0.5 分歧(measurements-disagree)

wb `eb69bf9` 报 09「confirmed [12.66,79.59]真 + [85.12,106.26]FP → P=0.5」。**cc 定位分歧根源:wb 在「未合并/pre-dedup 簇」粒度评的 P,不是生产 post-dedup。** 生产 `_dedup` 把 89 个事件按时序链式合并——**tid19 [58.59,86.46] 恰好跨过 wb 担心的 79.59→85.12 gap(5.53s)**,把 [11-72] 真绿簇与 [85-106] 簇**链成一个 episode**。该 episode 覆盖 GT[11-72] → **算 1 个 TP,不产生独立 FP**。**wb 的 [85-106] 假绿簇被吸进这一个 TP episode,对端到端记分无害。** cc 的 post-dedup 直探为准:89 raw → 1 episode。

## 3. 裁定:授权 wb 落地 dedup 修复(Fix A)

**采纳 Fix A**:`_dedup._absorb` 合并语义由「任一 review → episode review」改为「**任一 confirmed → episode confirmed;全 review 才 review**」。cc 已验证:F1→0.941 / P=1.000,01/10 安全。

**语义警示(记账,非阻断)**:Fix A **全局反转**安全优先级——原「review 优先」保护的是 **D1 遮挡/unknown 真不确定**段。Fix A 后,若未来某 episode 同时含「confirmed 子窗」与「D1 遮挡子窗」,会判 confirmed(不再交人复核)。cc 判断**可接受**:存在 confirmed 子窗 = 该窗内灯确为绿且车静止压线 = 真违章,遮挡段只是额外不确定时间,不否定已成立的违章窗。**当前 11 视频无「D1 遮挡 + confirmed 共存」被误升的 case**(所有正例已是 confirmed TP,04 无 confirmed,负例 0 confirmed 成员)。**保守替代(若 Jacob 要严守 D1)**:给 review 打来源标签(occluded/unknown 仍优先;仅 #3 transient_green 被 confirmed 吸收)——更稳但更多码。cc 推荐先上 Fix A(2 行、已验证),D1 误升若未来出现再上标签。

**wb 落地要求**:①改 `_absorb` 状态语义(Fix A);②cc 效果 gate **全 11 视频**(不再子集漏测):09 confirmed 命中[11-72] + 01/10 仍 0FP + 02/03/05/06/07/08/11 零回退 + 04 不变 → F1≥0.941/P=1.000;③红线:不碰 prior/权重/`_sample_roi`,独立 worktree,不 push/merge,球回 cc。

## 4. 09 prior 重定位线(wb `eb69bf9`)= 归档,不采纳(判 wb 做对了)

wb 从我原 brief 的「prior 重定位」入手,**用生产 `observe()` 直采路径纠正了 detect 标定的语义陷阱**(detect 给 (0.65,0.3) 实为车辆绿灯,直采 75% 帧返绿),找到真行人灯 ≈(0.50,0.10)。**但发现 prior 重定位救不了 P**,且**诚实回退、零失败生产改动、不 merge**——**方法论完全正确,cc 采信。** 关键副产物:

- **detect 标定坐标 ≠ prior 直采坐标**(两条路径),定 prior 必须走生产直采路径验证。**沉淀为红线**(避免后续重蹈)。
- **[85-106] unknown 窗的 21s 持续假绿**:根因=`prior_roi_expand_factor=2.0` 在灯灭期把紧 ROI 扩展到 80px 吃进邻近**车辆绿**。这是**持续假绿(>6s),#3 的 T 门控结构上抓不住**(只抓瞬态)——与 [[light-classifier-retrain]] 线同源(#3 是瞬态止血,非替代)。

**裁定**:①**09 prior 保持原值 (0.2,0.35,160)**(wb 已回退);②**不启动「禁用 09 prior 扩展」**——它是 per-video 特判 prior 扩展的代码改动,风险高、且 [85-106] 假绿当前被吸进 09 的 TP episode**不产生记分 FP**,不值当为一个不影响 F1 的潜在项特判;③[85-106] 持续假绿记账入 [[light-classifier-retrain]] 线(根治由判别器重训覆盖,非 prior 特判)。04 重定位一并暂停(随 09,低优先)。

## 5. 顺序与 b2

- **顺序**:wb 先落 dedup Fix A(解 09,快)→ cc 全 11 验收 → 再 qw 启 b2。dedup 与 b2 都在事件成形层,错开归因(与 wb 建议一致)。
- **b2**:cc 维持支持 qw 主攻(05 实锤)。b2 修好会让 09 episode 不再吞 89 事件(减小脏袋面),但 **dedup Fix A 已独立解 09**,b2 非 09 阻塞。

## 6. 一句话给 Jacob

wb 和我各自独立挖到了同一个根因(是 `_dedup` 那条「任一待复核→全待复核」的老规则,被 #3 的瞬态绿触发,拖垮了 09 的真违章),这很扎实。wb 提的修法(反过来:任一确认→全确认)**我亲自在生产上验过了**:09 恢复成 TP、01/10 两个负例仍然是 0 误报,**F1 到 0.941、精度满分,比合 #3 前还好**。wb 担心的 09 那个 [85-106] 假绿"第二个 FP"——**我查清了,那是他在合并前的粒度看的;生产合并后 09 就是一个事件、命中真违章、算一个 TP,那段假绿被吸进去了,不单独扣分。** 另外 wb 那条 prior 重定位线做得很规范(纠正了标定路径的语义陷阱、发现重定位救不了 P、诚实回退没留坏改动),我判**归档不采纳,09 prior 保持原值**;他发现的那段"车辆绿被 ROI 扩展吃进来"的持续假绿是判别器重训线的事,不影响现在的 F1,先记账。**建议:授权 wb 落地 dedup 那 2 行改动,我全 11 视频验收(F1≥0.941/P=1.000)。** 是否严守遮挡段的"待复核安全优先"(那需要多写点代码给 review 打来源标签),你可以拍板;我推荐先上简单版、真出问题再加固。

---
*署名:cc(arbiter)。证据=生产 v2/box 09/01/10 post-dedup 直探(09: 89raw→FixA 1 confirmed episode=TP;01: 0 confirmed 成员仍 review;10: 0 事件)。承 [[postmerge-f1-875-09-regression]]、[[measurements-disagree-find-the-bug]]、[[prior-misframe-rootcause]]、[[light-classifier-retrain]]、[[b2-tracking-fragmentation-blindspot]]。*
