# 统一 GT · Phase 2 设计说明（build_gt.py 重建 + 修正真 merge）

> **出自**：cc 方向 brief（Phase 1 地基已收官，commit `8efbef3`/`215d476`）。
> **本文件状态**：设计**已通过 cc review（2026-07-19）**，签字放行写码。cc 在 review 中实测验证了「feedback 与 base 在违章窗无冲突、01 全 red、06 red feedback 全落在 base 红窗内」—— 实际 GT 改动预计很小、F1 位移轻微。**下文 §2.1/§2.5/§3/§7 已据 cc 裁定更新**。
> **核心风险**：Phase 2 改 canonical = 改测量尺 → 必须可复现、可审计、旧新对比透明。
> **cc 裁定摘要**：决策 1=A（schema 保留，provenance 落 audit 不进 CSV 列）；2=**改**：Phase 2 只 commit light merge，crosswalk 仅 dry-run 报告不提交（保 F1 位移可干净归因到灯段）；3=否（events/videos 保留 base 不退化成文本解析）；4=本地快照 + git tag。写码前补两点：①伪代码对齐「gt==base 不改写」；②加冲突守卫（green↔red 翻转影响违章资格的高亮待签）+ 负例不翻断言。
> **红线**：每处变更可追溯到源行；旧 canonical 留快照；旧/新 F1 对比讲清；输入只读；scoped add + 署名 + trunk main + TDD。

---

## 0. 输入实测结构（已核对，非假设）

| 文件 | 形态 | 关键列 | 规模 |
|---|---|---|---|
| `datasets/gt/light_states.csv` | **段级 canonical**（被重建） | `video, start_s, end_s, state, confidence, note` | 11 视频 / 24 段；state∈{green×9, red×13, unknown×2}，confidence∈{confirmed×21, occluded×1, tentative×2} |
| `datasets/gt/feedback/light_feedback.csv` | **帧级人工修正**（输入） | `video, t_sec, frame_idx, pred, gt, verdict, reason, note, ts` | 593 行（592 数据）；verdict∈{algo_wrong×486, other×104, both_wrong×2}；**gt 仅 green×269 / red×323**（无空 gt）；覆盖 7/11 视频（01/02/03/04/06/07/11，缺 05/08/09/10） |
| `datasets/gt/feedback/crosswalk_feedback.csv` | **逐帧 poly 修正**（输入） | `video, t_sec, frame_idx, verdict, reason, note, annotated, imgh, imgw, poly, ts` | 36 行，9/11 视频；verdict 均 `labeled`，poly 为 `x,y;x,y;...` |
| `datasets/gt/source/annotation.csv` | **自由文本原始描述**（输入） | `input_video, gt_description, result` | 11 视频，描述为非结构化中文 |
| `datasets/gt/events.csv` / `videos.csv` | **结构化 canonical**（被重建/保留） | events: 违章窗+车牌；videos: 清单+has_violation | 已 9 处/多处引用 |
| `datasets/gt/tracking/*.json` | **骨架 canonical** | `box: null` 未标 | 9 视频，本 Phase 不动 |

**关键事实（决定 merge 规则）**：
1. `light_feedback` 的 `gt` 列**全为非空**（green/red），即 592 行每一行都是"该帧真实灯态=green/red"的有效修正信号 → 均应作为覆盖点。
2. `verdict=other`（104 行）与 `both_wrong`（2 行）仍带有效 `gt`，与 `algo_wrong` 同等处理（均覆盖）。
3. `pred` 含 flashing/unknown（即算法把灯判成非 green/red），human 纠正回 green/red。
4. `light_states` base 已含 `unknown` 段；feedback 只产出 green/red 覆盖（无人标 unknown 修正）。
5. `source/annotation.csv` 是**自由文本**，无法可靠反向重建结构化 events。

---

## 1. 总体管线（build_gt.py）

```
只读输入:  source/annotation.csv  (原始描述, 自由文本)
           feedback/*.csv         (画廊修正, 帧级)
           badcases/*.csv         (迭代修正, 当前空, 最高优先级)
        + 当前 canonical 作为 base 种子 (light_states / crosswalk/* / events / videos)

重建输出:  light_states.csv  (feedback/badcase 覆盖 base)  ← Phase 2 唯一提交的 canonical
           crosswalk/*.json  ← Phase 2 不提交，仅 dry-run 报告(见 §3)
           events.csv / videos.csv (保留 base, 见 §4 决策3)
           + provenance 落 note/confidence(A 方案) + build_audit.json

审计产物:  _snapshot_pre_build/  (重建前 canonical 快照, gitignore + tag gt-pre-phase2)
           build_audit.json + build_report.md (每处变更→源行 + 高风险待签清单)
           eval 旧GT-F1 vs 新GT-F1 对比 (仅 light 改动, 可干净归因)
           冲突守卫: green↔red 翻转高亮待签(§2.5) + 负例(01/10)不翻断言(§2.5)
```

**优先级（合并冲突时）**：`badcase > feedback > base(canonical/派生)`。
实现即：以 base 段为种子 → 应用 feedback 覆盖 → 应用 badcase 覆盖（顶替）。每层覆盖在 provenance 里记下来源。

---

## 2. light 帧级→段级 merge 规则（核心，待 review）

### 2.1 单视频算法（伪代码）

```
def build_light_video(video, base_segs, overrides, badcases, merge_gap_tol=0.5):
    # overrides: 该视频 feedback 生成的 [(t_sec, gt, src_ref), ...]  已按 t 排序
    # 1) 合并相邻同 gt 覆盖点 → 覆盖区间 [t0,t1]=gt
    ivs = merge_intervals(overrides, gap_tol=merge_gap_tol)   # 同 gt 且间隔<=tol 合并
    # 2) 用 base 段 + 覆盖区间切分时间轴
    cuts = sorted(set(base 边界) ∪ set(ivs 边界))
    out = []
    for (a,b) in consecutive(cuts):            # 每个子区间
        base_state = state_of(base_segs, mid(a,b))
        ov = iv_covering(ivs, mid(a,b))         # 该中点命中的覆盖区间(若有)
        # 仅当覆盖区间存在且 gt≠base_state 才改写段；gt==base 时回退 base，
        # 保持 diff 干净（与 §2.2 一致，不因「gt==base 也标 src=feedback」而污染 note）
        if ov and ov.gt != base_state:  state, src = ov.gt, ov.src_ref
        else:                           state, src = base_state, "base"
        out.append((a,b,state,src))
    # 3) 合并相邻同 (state, src) 子区间
    out = merge_adjacent(out, key=(state,src))
    return out
```

### 2.2 覆盖点生成（feedback → overrides）

- 每一行 feedback（gt 非空）→ 一个覆盖点 `(t_sec, gt, src_ref)`，`src_ref = "feedback:违章02@11.08 gt=red reason=reading_point"`。
- **仅在 `gt != base_state(mid)` 时改写段**；若 `gt == base_state`，**不改动该段**（避免无谓重写 note、保持 diff 干净）。这样违章02 那些 "flashing→red 但 base 本就是 red" 的确认行不产生变更。
- `both_wrong`/`other` 同 `algo_wrong` 处理（gt 有效即覆盖）。

### 2.3 区间合并与防碎裂

- `merge_gap_tol`（默认 0.5s）：两相邻同 gt 覆盖点间隔 ≤ tol → 合成一段连续覆盖，避免帧采样抖动造成密集碎段。例：feedback 在 30.0/30.3/30.6 都标 green，合成 [29.9,30.7]=green 一段。
- **密度置信度（审计信号，可选）**：连续覆盖区间若由 ≥K 个采样点支撑 → `confidence="feedback"`；若仅孤立 1–2 点 → `confidence="feedback-tentative"`，提示该段可能欠采样、需人工复核。K 与 tol 为可调超参，设计阶段定默认值（建议 K=3, tol=0.5）。

### 2.4 provenance 落地（**决策 1，需 cc 拍板**）

`light_states.csv` 现列：`video, start_s, end_s, state, confidence, note`。两种方案：

- **方案 A（推荐·schema 保留）**：不改列结构。`confidence` 置 `"feedback"`/`"feedback-tentative"`；`note` 追加溯源，形如：
  `merge(feedback): 违章02@11.08 gt=red reason=reading_point | base: 旧GT肉眼确认`。
  → 现有 9 处消费者按列名读取，零破坏；审计靠 note 文本仍可逐行溯源。**推荐**。
- **方案 B（按 SCHEMA.md 加列）**：新增 `source` / `ts` 列，严格落 provenance。`light_states.csv` 表头变更 → 需确认 9 处引用不会因列数/位置解析而崩。风险较高。

> 我倾向 **方案 A**（安全、可审计等价），但 SCHEMA.md 写了 source/confidence/ts/note，若 cc 坚持严格 schema 则走 B 并先验证消费者。请 cc 定。

### 2.5 冲突守卫 + 负例不翻断言（cc review 补强，必实现）

即便实测「feedback 与 base 在违章窗无冲突」，**build 仍必须内建两道防御**，防未来 feedback/badcase 引入隐患、且绝不静默提交高危变更：

1. **green↔red 翻转守卫（高风险高亮 + 人工签字）**
   - merge 后若某段 `state` 相对 base 发生了 **green↔red**（或 green/red↔unknown 且 unknown 意味着「无灯态信号」使其可能撑起/撤销违章资格）的翻转，且该视频在 `videos.csv` 中 `has_violation=True`（即翻转"可能"改变 is_violation 资格），则：
     - 在 `build_report.md` 单列一节 **「⚠️ 高风险变更（需人工签字）」**，逐条列出 `(video, [a,b], base_state→new_state, src_refs)`；
     - `build_gt.py` 默认 **dry-run 不写 canonical**（exit code 非 0 + 打印待签清单），除非显式传 `--accept-high-risk`（要求运营/人工在报告上签字后再跑一次）。
   - 仅当翻转落在 `has_violation=False` 的视频（如 01/10 的负例，或 base 本就 unknown 且无人标 green 撑起违章）时，属低危，正常入 audit 不阻塞。
   - **设计意图**：F1 位移必须可归因到具体灯段，绝不能让「灯 GT 被翻」悄悄改变评测结果而不留痕。

2. **负例不翻断言（hard assert）**
   - 对 `videos.csv` 中 `has_violation=False` 的视频（01、10），断言 merge 后**仍无任何 `state=green` 段**（`red`/`unknown` 允许）。
   - 若断言失败（如某 badcase/feedback 给负例标了 green）→ **立即 abort**，不产出任何 canonical，并打印触发源行。这是「负例不污染」的硬边界，永不退化。

> cc review 实测佐证（增强信心，非替代守卫）：01 feedback 全 red（强化负例，安全）；10 无 feedback；06 的 red feedback 全部落在 base 红窗 [29.09,45.41] 内，未触碰 base 绿窗 [0,29]，故 06 不会从 TP 翻 FN。当前数据下两守卫均不触发，但它们为后续迭代（尤其是 badcase 累积）兜住底线。

---

## 3. crosswalk 重建（决策 2 已改：Phase 2 仅 dry-run，不提交）

> **cc 裁定（决策 2 改动）**：Phase 2 的**核心价值是测量尺变更透明可归因**。若同时改 light + crosswalk 两个 canonical，F1 一旦位移就无法干净归因是灯还是斑马线导致的。故：
> - **Phase 2 只 commit `light_states.csv` 的 merge**（单一模态，F1 位移可逐事件归因到灯段）。
> - **crosswalk 在 `build_gt.py` 里只跑 dry-run 报告**（输出 base vs feedback poly 的 IoU/差异），**不提交** `crosswalk/*.json`。预计 `crosswalk/*.json` 本就接近 human（base≈human，近乎 no-op），dry-run 会证实。若 delta 微小 → 下个 follow-up 顺手 fold；若 delta 大 → 单独审。

- **base** = 当前 `crosswalk/*.json`（每视频 1 个 polygon canonical）。
- **feedback** = `crosswalk_feedback.csv` 逐帧 `poly`（labeled）。
- **dry-run 策略（仅报告，不写）**：
  - 若视频**无** crosswalk_feedback → 标记 `no-feedback`（无动作）。
  - 若有 → 做**一致性校验**：feedback 多帧 poly 与 base poly 的 IoU/顶点差。一致则报告 `consistent`；**不一致**则报告 `divergent` 并列出 human poly 代表帧（按形状聚类取最大簇代表）。
  - 报告进 `build_report.md` 的「crosswalk dry-run」专节 + `build_audit.json` 的 `crosswalk_dryrun` 字段；**绝不写 `crosswalk/*.json`**。

> 注意：crosswalk 重建是"对齐/采纳"，改动面预计很小（base 本就接近 human）。dry-run 会如实反映"一致/不一致"数量，供后续 follow-up 决策。

---

## 4. events / videos 处理（**决策 3，需 cc 拍板**）

- `events.csv`（违章窗+车牌）、`videos.csv`（清单+has_violation）已是**最高质量结构化 canonical**，被 9+ 处引用。
- `source/annotation.csv` 是**自由文本**，`gt_description` 无法可靠解析回结构化字段（如 violating_plates、精确窗）。
- **建议（决策 3）**：Phase 2 **不从自由文本再生 events/videos**；二者作为**稳定 base 保留**，仅当有 badcase/feedback 针对事件时才覆盖（目前 events 无 feedback/badcase）。避免把最好的 GT 退化成文本解析产物。
- 若 cc 坚持"events 也要从 source 重建"，需另写文本→结构解析器且人工抽检，风险/收益不匹配，我反对。

---

## 5. 审计 / 透明 / 红线落实

1. **旧快照**：重建前 `cp -r datasets/gt/{light_states.csv,crosswalk/,events.csv,videos.csv,tracking/} datasets/gt/_snapshot_pre_build/`，并打 git tag `gt-pre-phase2`（不可变，供 diff）。快照目录加 `.gitignore`（不污染仓库）或按需提交——**决策 4**（建议：目录本地留 + tag 永久）。
2. **变更溯源**：`build_audit.json` 逐视频逐段列出
   `old: (state, note) → new: (state, note, src_refs)`，src_refs 指向具体 feedback/badcase 行。不可解释的变更 = bug（红灯）。
3. **测量尺变更透明**：重建后跑
   `eval_violations --detector v2 --occ-denom box`（禁 reuse）→ 报告 **旧GT-F1=0.889 vs 新GT-F1**，并对每个 tp/fp/fn 变化的事件，归因到该视频中**被改写的 light_states 段**（哪些灯 GT 更正导致判定翻转）。**绝不静默把 baseline 从 0.889 改成新数**——报告同时给两个数。
4. **输入只读**：脚本只读 `feedback/ source/ badcases/`，绝不回写。
5. **提交纪律**：scoped `git add`（禁 `-A`）；`Co-Authored-By: Claude Opus 4.8`；trunk main。

---

## 6. TDD 计划（写码前先写单测）

`tests/test_build_gt.py`：

| 用例 | 验证 |
|---|---|
| 优先级-1 | badcase 同位置覆盖 feedback 覆盖 base：结果取 badcase |
| 优先级-2 | feedback 覆盖 base：结果取 feedback |
| 优先级-3 | 无 feedback/badcase：输出==base（幂等） |
| 聚合-1 | base [0,20]=red，feedback 在 5.0/5.3/5.6 标 green(间隔<tol 0.5，合并为 [4.9,5.7]) → 输出 [0,4.9]=red,[4.9,5.7]=green,[5.7,20]=red |
| 聚合-2 | feedback 在孤立单点 10 标 green、base red、左右 base red → 输出绿段为孤立窄段（tentative 标记） |
| 聚合-3 | feedback gt==base → **不产生变更**（note 不重写） |
| provenance | 变更段 note 含 `merge(feedback):` + 源 video@t_sec + gt |
| 审计 | build_audit 每行 old→new 均可映射到至少一条 input 行 |
| 守卫-1 负例不翻 | `has_violation=False` 的视频（如 01）经 merge 后仍无 `state=green`；若某 input 给负例标 green → assert abort 且零产出 |
| 守卫-2 翻转高亮 | `has_violation=True` 视频在 merge 后发生 green↔red 翻转 → 入 `build_report.md` 高风险待签清单 + 默认 dry-run 不写 canonical（`--accept-high-risk` 才放行） |
| 守卫-3 翻转低危 | `has_violation=False` 视频发生 **red↔unknown** 翻转（不产生 green）→ 负例断言通过，正常入 audit，不进入高危清单（green↔red 在负例视频会被负例断言直接 abort） |
| 改写对齐 | feedback.gt==base_state 时，输出段 `src="base"`、note 不被 `merge(feedback)` 污染（对应 §2.1 伪代码修正） |

单测用构造的微型输入（不依赖 593 行真实数据），真实数据仅作集成验证（跑通后产出 audit/diff）。

---

## 7. 4 个决策（cc 已裁定，2026-07-19 review）

| # | 决策点 | cc 裁定 | 落实位置 |
|---|---|---|---|
| 1 | provenance 落点：schema 保留(方案A) vs 加 source/ts 列(方案B) | **A 同意**：schema 保留。`build_audit.json` 作机器可读 provenance，`note` 作人读溯源。SCHEMA.md 的 source/ts 落进 audit，**不进 CSV 列**（免破坏 9 处消费者） | §2.4 |
| 2 | crosswalk poly 采纳规则 | **改**：Phase 2 **只 commit light merge**；crosswalk 仅 dry-run 报告、**不提交**（保 F1 位移可干净归因到灯段） | §3 |
| 3 | events/videos 是否从自由文本重建 | **否，强烈同意**：events 是最高质量结构化 GT，不退化成文本解析；保留为稳定 base | §4 |
| 4 | 旧快照存放 | 本地 `_snapshot_pre_build/`(gitignore) + git tag `gt-pre-phase2`，同意 | §5.1 |

**cc 放行写码（2026-07-19）**：设计通过，进入 Task #28（写 build_gt.py + TDD）→ Task #29（重建+快照diff+旧新F1对比）。写码前已将「①伪代码对齐 gt==base 不改写 ②冲突守卫+负例不翻断言」补入 §2.1/§2.5。

> **实测利好（cc review 验证，增强信心）**：feedback 与 base 在违章窗内无冲突；01 feedback 全 red（不翻负例）；10 无 feedback；06 red feedback 全落在 base 红窗 [29.09,45.41] 内，未碰绿窗 [0,29]。预计 GT 改动小、F1 位移轻微，两守卫本批数据均不触发。
