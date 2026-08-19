# cc plan-gate:09 prior 重定位诊断(wb 只读)= 诊断 PASS / 选项① 证伪 OUT

> 署名:cc(arbiter) 转交:Jacob(方向拍板);抄送 wb/qw/ds
> gate 对象:分支 `wb-09-prior-relocate-diag`@`1c4cd05`(wb 只读诊断,未 push/merge;报告 `docs/reports/2026-08-19-wb-09-prior-relocate-diag.md` + `scripts/diag_09_true_light.py`(Q1/Q2)+ `scripts/diag_09_tp_source.py`(Q3))
> cc 独立复现:两脚本从分支 `git show` 取出,**拷入隔离临时目录 `/tmp/cc_09_verify` 运行**(输出不碰 wb 产物;import 仍解析到生产 `src/`);未进 wb 活 worktree;零生产码改动。

---

## 0. 为什么这条必须逐比特复现

这条诊断同时(a)**推翻 Jacob 已选的选项①**(对 09 放宽 prior 红线重定位真灯位),(b)**反转 cc 自己上一裁定的预测**(我曾断 09 真灯位恐不存在=纯树)。两个方向都是"结论出人意料且与既有判断相反"——正是必须 bit-for-bit 复现、绝不采信口头结论的场景。

## 1. 方法学审计 —— PASS

- **`faithful_observe`(Q3)逐分支比对生产 `observe()`(traffic_light.py:140-230):bit-faithful。** YOLO 分支(`yolo_cy_min` 过滤 / `_sample_box` / `signal_prior` 最近框 / `use_yolo = best_dist > yolo_prior_near`)→ prior_hsv 分支(`_sample_prior_color`)→ prior_near 半径回退 → global 全局,四条路径与生产逐行一致;obs 取值唯一差异是脚本额外返回分支标签、省略 conf(不影响 obs 判定)。
- **重定位模拟正确。** `_sample_prior_color`(:607-612)实时读 `self.signal_prior` / `self.prior_roi_px`,脚本 `tl.signal_prior=(reloc_cx,reloc_cy); tl.prior_roi_px=160` 是合法等价,无缓存陷阱。
- **GT 仅作 oracle。** `gt_state_at` 只给 rows 打状态标签做分桶,**从不进 `faithful_observe` 的判色**。红线合规(GT 不喂生产推理路径)。
- **单灯隔离(Q1/Q2)避免多灯聚合虚高。** 用 canonical governing-ped 质心测漂移,非多灯并集。

## 2. 全部 load-bearing 断言 —— cc 独立复现,bit-for-bit 吻合

**Q1(真灯存在性,反转 cc 旧预测):**
| 指标 | wb 报告 | cc 复现 |
|---|---|---|
| 真灯中心(GT derived) | (0.618,0.265) | (0.6183,0.2653) ✅ |
| 检测器可见率 | 82.6% (261/316) | 82.6% (261/316) ✅ |
| IoU 当前→重定位 | 0→0.1453 | 0.0000→0.1453 ✅ |
| 中心距 当前→重定位 | 0.427→0.042 | 0.427→0.0416 ✅ |
| 真灯框被重定位 ROI 包含 | 100% | 100.0% ✅ |

→ **09 真灯确实存在**(可达 IoU 0.1453 / 中心距 0.042 / 100% 包含)。cc 旧断言"(0.50,0.10)纯树=真灯不存在"**被反转**。

**Q3(决定性 — 重定位是否修得好假绿):**
| 指标 | wb 报告 | cc 复现 |
|---|---|---|
| 分支计数 当前 prior | — | prior_hsv 274 / yolo 42 |
| 分支计数 重定位 prior | 与当前完全一致 | prior_hsv 274 / yolo 42 **(逐数一致)** ✅ |
| GT=red 窗假绿 当前→重定位 | 0.856→0.969(恶化) | 83/97=0.856 → 94/97=0.969 ✅ |
| GT=off 窗假绿 当前/重定位 | 100%/100% | 18/18=1.0 / 18/18=1.0 ✅ |
| GT=green 真绿召回 当前→重定位 | 提升 | 194/201=0.965 → 201/201=1.0 ✅ |
| 09 TP 绿来源 | prior_hsv 主导(错位 ROI 采出租车) | TP绿 prior_hsv 177 / yolo 17(177/194)✅ |
| 绿 obs 时间跨度 当前→重定位 | 合并成整段巨 run | [1.01,106]→[0,106] 巨 run ✅ |

## 3. 裁定

### 3.1 诊断质量 = PASS
方法学 bit-faithful、GT 仅 oracle、零生产改动、全部关键数字 cc 隔离复现逐数吻合。红线合规。

### 3.2 选项①(重定位 09 prior)= 证伪 OUT
**重定位不但修不好假绿,还让 GT=red 窗假绿从 0.856 恶化到 0.969**;GT=off 窗两 prior 都 100% 假绿(重定位零改善)。真绿召回微涨(194→201)代价是把绿 obs 合并成 [0,106] 整段巨 run——`#3` 时序门(T=6.0)会 confirm 这条巨 run,假绿照样进 confirmed。

**根因(cc 复核确认)不是 prior 位置,是 `_sample_roi` 的 160px 区域 HSV 面积投票**(g/r 阈值 [35,60,40]-[95,255,255]、`min_frac=0.002`、`g_frac>r_frac*1.3`):把 ROI 搬到真灯上,ROI 仍在红/灭窗内读出足量绿像素(环境绿/出租车/植被),**无灯形/上下文判别 → 位置无关的颜色投票**。这与 [[01fp-falsegreen-fix1-disproved]] 同源(ROI 太大 + 面积投票无形状约束),坐实治本须走**弥散绿空间判别**而非挪 prior。

### 3.3 09 当前 TP = 巧合(承接判别器 Phase A HOLD 论据)
09 TP 绿 177/194 来自 `prior_hsv`——错位 prior(0.2,0.35)采画面左侧出租车绿,恰好时间上落在 GT-green 窗内。**TP-green 与 FP-green 同源**(同一位置无关 HSV 投票),唯一区分是 GT 时间标签,无视觉灯信号可分。这正是我在判别器 Phase A plan-gate(`9eeebbe`)提出的"09 结构性无解"的独立第二证据。

### 3.4 GT 状态更正采信
canonical GT 09 = RED[0-3.97] / GREEN[5.96-71.51] / RED[73.50-99.33] / OFF[101.31-105.28](cc 从 `light_canonical_gt.json` 直算)。**[72-106] 是确定的红→灭,非 unknown** → 该窗假绿是**确定性错误**,wb 更正采信。

## 4. cc 发现的方法学 caveat(不改裁定,诚实记账)
- **Q1 的 "82.6% 可见" 是宽口径**:统计真灯中心 0.06 内**任意彩色亮斑**帧(green 176 ≫ GT-present 51),混入环境绿。它证"真灯区有色",不等于"灯清晰点亮 82.6%"。别把 82.6% 过读为纯灯可见率;决定性结论靠 Q3 不靠此数。
- **Q1 与 Q3 重定位目标不同源**:Q1 用 canonical governing-ped 质心均值,Q3 用 `light_location_gt` derived true_center(0.6183,0.2653,仅 3 帧标注推导)。两者都在正确一侧,Q3 结论(恶化)对目标点不敏感 → 稳健。

## 5. 建议(转 Jacob 拍板)
1. **选项① 关闭**:重定位 09 prior 已证伪(不修反恶化),`prior-misframe` 红线**不必**为 09 破。
2. **09 归入"接受不解、降优先级"(≈ 原选项②)**:09 现状是 confirmed TP(巧合但成立),无记分 FP;强修 09 = 判别器线储备项,让位 [[b2-tracking-fragmentation-blindspot]] 后续 / 更高杠杆项。
3. **判别器线 HOLD 维持但论据升级**:09 无解已双重独立坐实。若 Jacob 仍要推判别器,治本方向锁定**弥散绿空间/形状判别**(选项③),非挪 prior、非重训现 `ped_signal.pt`。

## 6. 红线合规
未进 wb 活 worktree(隔离 `/tmp/cc_09_verify` 跑拷贝脚本);未 push/merge wb 分支;零生产码改动;GT 仅 oracle;prior 坐标未改(仅内存模拟)。

---
*署名:cc(arbiter)。证据=cc 隔离复现 Q1(82.6%/IoU 0→0.1453)+ Q3(分支 274/42 两 prior 一致、GT=red 假绿 0.856→0.969、TP 绿 177/194 prior_hsv)+ canonical GT 09 状态直算。承 `docs/handoff/2026-08-19-cc-plangate-discriminator-phaseA.md`、`docs/reports/2026-08-19-wb-09-prior-relocate-diag.md`、[[prior-misframe-rootcause]] [[01fp-falsegreen-fix1-disproved]] [[light-classifier-retrain]]。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
