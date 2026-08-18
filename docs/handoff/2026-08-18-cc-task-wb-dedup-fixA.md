# cc 任务单(wb 专属):落地 dedup 修复 Fix A

> 署名:cc(arbiter) 收件:wb 转交:Jacob
> 本单只含 wb 的工作,不涉及 qw/b2。b2 见另一份 qw 任务单,互不阻塞、互不引用。
> 承 `4329e78`(09 真因=dedup review 毒化)、wb `48c3ed7`(dedup 诊断)、cc 裁定 `cf2e94e`(Fix A 验证安全)。

---

## 1. 任务

改 `src/redlight/pipeline/violation_engine.py` 的 `_dedup._absorb`(第 299-301 行),把合并语义从
「任一成员 review → episode review」**反转为**「**任一成员 confirmed → episode confirmed;全 review 才 review**」。

当前(bug):
```python
# review 优先级高于 confirmed(安全侧交人复核)
if e["status"] == "review" or cur["status"] == "review":
    cur["status"] = "review"
```
Fix A(目标):存在 confirmed 成员 → episode confirmed;仅当所有成员均 review → review。

## 2. 为什么(根因,已定论)

#3 时序门控在 09 的 [0.67,4.18] 0.94s 瞬态绿上造 1 个 review 事件;`_dedup` 跨 track 链式合并后,
旧「review 优先」规则把压在 55.9s 铁绿([7.41,106.26] visible 持续绿,GT[11-72] 落其中)上的 44+ 个
confirmed **全拖成 review** → 09 从 TP 掉成 FN。这是 #3↔dedup 交互 bug,非灯态/prior 缺陷、非 tradeoff。

## 3. cc 已验证的预期结果(你需复现)

生产 v2/box post-dedup 直探:
- **09**(GT[11-72]):89 raw(84 conf/5 review)→ Fix A **1 confirmed episode 命中[11-72] = TP**;
- **01**(负例):15 raw,**0 confirmed 成员** → 仍 review = **0 FP**(#3 消 01FP 不复活);
- **10**(负例):0 事件。
- 端到端 **tp8/fp0/fn1(04) = P=1.000 / R=0.889 / F1=0.941**(优于合 #3 前 0.889 与现状 0.875)。

## 4. 落地红线(硬约束)

- 只改 `_absorb` 状态语义这一处;**不碰** prior / 权重 / `_sample_roi` / `sat_min` / `light_priors.json` / `ped_signal.pt` / #3 阈值。
- **独立 worktree**,不 push、不 merge,球回 cc 验收。
- 自测锚:01 仍 0 FP(纯瞬态 3.10s 无 confirmed 核,不得复活)。

## 5. 语义警示(记账,非阻断,待 Jacob 拍)

Fix A 全局反转「待复核安全优先」。当前 11 视频**无**「D1 遮挡子窗 + confirmed 子窗共存」被误升的 case
(正例均已是 confirmed,04 无 confirmed,负例 0 confirmed 成员),cc 判可接受。
若 Jacob 要严守 D1 遮挡的人工复核 → 保守替代:给 review 打来源标签(occluded/unknown 仍优先;仅 #3 transient_green 被 confirmed 吸收),更稳但更多码。cc 推荐先上 2 行 Fix A,真出问题再加固。

## 6. cc 验收口径(全 11 视频,不再子集漏测)

09 confirmed 命中[11-72] + 01/10 仍 0FP + 02/03/05/06/07/08/11 零回退 + 04 不变 → F1≥0.941/P=1.000。
生产 v2/box、含负例、端到端 confirmed 事件表为准,不信单点门脚本中间量。

---
*署名:cc(arbiter)。本单专属 wb。qw/b2 见 `2026-08-18-cc-task-qw-b2.md`,不在此单范围。*
