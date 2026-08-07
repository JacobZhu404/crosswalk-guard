# cc 最终效果 gate:wb #3 时序门控实现(19a4b58)— **窄送回:机制/代码/G1 全 PASS,但 T=10 须重调(违章11 真绿仅 2.7% margin)+ 修 config 假注释**

> 审核对象:`19a4b58`(branch `wb-01fp-temporal-gate`)— fuse_light 标注 max_raw_green_run_s + decision.py 门控 + engine/cli 接线 + config + 验证脚本 + 报告
> 署名:cc(效果 gate/独立复核)— 遵 [[measurements-disagree-find-the-bug]]、承 [[wb-01fp-temporal-plangate5-passed]](plan-gate #5 六硬条件,尤其 C1 n=1 过拟合)
> 复核环境:cc 独立 detached worktree `cc-verify-gate@19a4b58`(symlink input_video/models),`verify_gates_01fp.py` 亲跑 baseline(T=0)vs fixed(T=10)于 01/10/11/06/07。
> **结论:机制正确、实现干净、红线合规、G1 消 01 误绿 cc 亲验成立、真绿不回退——但 T=10 落在真实 GAP(3.10, 10.27)的顶端,违章11 真绿段仅 10.266s = **距阈值 2.7%**,而其余真绿 margin ≥146%。授权我出方案时基于的 8.6× margin 在全 11 视频集上是假的(实际 1.03×)——这正是 C1 要防的 n=1/子集过拟合在边界应验。config 注释仍写「真绿最短 26.8s」= 假。窄送回:重调 T 到 GAP 中部 + 修注释 + 重跑 G1/G2/G4,其余全部保留。**

---

## 1. cc 亲跑复现(bit-for-bit,GT 独立)

| 视频 | baseline confirmed | fixed(T=10) confirmed | fixed review | green 段 max_run | 距 T=10 margin |
|---|---|---|---|---|---|
| 违章01 | 1 | **0** | 1 | 3.098s | −69%(降级 ✓) |
| 违章10 | 0 | 0 | 0 | — | — |
| **违章11** | 1 | **1** | 0 | **10.266s** | **+2.7%** ⚠️ |
| 违章06(石) | 1 | 1 | 0 | 31.22s | +212% |
| 违章07 | 1 | 1 | 0 | 24.645s | +146% |

**G1 cc 亲验成立**:01 confirmed 1→0(误绿段 [48.36,61.29] max_run=3.098s 降级 review);10 保持 0。**真绿不回退**:06/07/11 全保 confirmed。数字与 wb 报告一致。

## 2. 决定性问题:T=10 在全集上不满足 C1(过拟合防护)

- **授权前提被证伪**:我在 plan-gate #5 复现的可分性表只含 6 视频(01/05/06/07/08/09),真绿最短 =26.80s(07)→ 8.6× margin。**但那个子集不含违章11**。wb 跑全 11 视频后,**违章11 的真绿是一段 genuinely 短绿(单段 raw run 10.266s)** —— 全集真绿地板从 26.8s 塌到 10.27s。**真实 GAP = (3.10, 10.27),T=10 坐在其 97% 处**,离真绿地板只有 0.266s。
- **fragile**:违章11 是硬救回的 TP([[crosswalk-ceiling-finding]] 救回11)。其**全部 confirmed 就压在这一段 10.266s 上**。检测/采样的 run-to-run 抖动只要把它压到 <10.0s,11 就从 confirmed 翻 review = 丢掉这个 TP(F1 回退)。2.7% margin 不是「safe」,是「勉强过」。
- **wb 优化了错的一侧**:报告 §6 自陈「取 10s 留足缓冲防单点过拟合」——但那个 buffer 是对**假绿侧(01=3.10s)**的,代价是**真绿侧(11=10.27s)几乎零 margin**。wb 也写了「T 下探到 ~5s 仍可分」——即 wb 已知中部 T 可用。01 在任何 T>3.10 都被消除,把 T 顶到 10 对消 01 零增益,只让 11 变脆。

## 3. config 注释造假(须修)

`configs/config.yaml` 新增项注释写:「落在 01 假绿(3.10s)与真绿最短(**26.8s**)两群 GAP」。**26.8s 是 6 视频子集数,全集真绿最短是 10.27s(违章11)**。注释把真实 margin(1.03×)误述成子集 margin(8.6×),误导维护者以为 T=10 很安全。报告 §1 自己写的是 10.27s——**config 注释与报告自相矛盾**。

## 4. PASS 的部分(全部保留,cc 采信)

- **机制正确**:`_raw_green_runs`/`_max_run_in_segment` 口径与横测一致(run 时长=末帧-首帧);fuse_light 标注纯加性(只加字段,不改既有段判定);decision.py 门控用 `s.get("max_raw_green_run_s",1e9)` 向后兼容;瞬态绿→review 非硬杀(C1 兜底)。
- **接线加性**:cli.run/engine 的 `min_persistent_green_run_s` 形参 config 默认路径行为可控;显式传参优先(诊断注入)。**改动面比方案说的「两处」多了 engine/cli/config 接线**(§2),但那是必要 plumbing 非 scope creep,cc 采信。
- **红线合规**:未碰 `_sample_roi`/`sat_min`/`light_priors.json`/`ped_signal.pt`/`enforce_transition_limit`;禁 select_gtfree;GT 仅段级对齐当裁判不进推理;scoped 提交。
- **验证方法学干净**:`verify_gates_01fp.py` 复用 cli.run + monkeypatch 抓段,GT 只用于真/假绿分类(裁判),无 oracle 注入。
- **G1 成立**(cc 亲验)。**额外收益 09 假绿降级**(0.943s 落 GT red 区间)方向合理,cc 未单独跑 09 但采信机制(与 01 同类)。

## 5. 送回条件(窄,快)

**不 merge。** wb 须:
1. **T 重调到真实 GAP(3.10, 10.27)中部**,给两侧都留健康 margin(任一侧不得 <~30%)。cc 建议 **T=6.0**(01 距阈 −48%、11 距阈 +71%;几何中点 √(3.10×10.27)=5.64)。区间 [5.5, 7] 均可——须 above 全部已观测假绿(01=3.10 / 09=0.94)且 below 全部真绿(min 10.27)。
2. **修 config.yaml 注释**:真绿最短改为 10.27s(违章11),margin 据新 T 如实写;删「26.8s」。
3. **新 T 下重跑 G1/G2/G4**(脚本已就绪,把两轮的 fixed min_run 从硬编码 10.0 改为受验 T):证 01/09 仍降级 review、违章11 仍 confirmed 且 margin 健康、其余真绿不回退、视频10 无新增 FP。
4. 报告与 config 注释口径统一到全 11 视频真实数字。

**保留**:机制、代码结构、接线、G1 事实、验证脚本——**只调一个数 + 修注释 + 重验**,不推翻实现。

## 6. 一句话给 Jacob

wb 的 #3 时序门控**机制是对的、代码干净、违章01 的误报我亲测确实消掉了**。但有一个**边界隐患**必须先修再合:授权时我复核的可分性只测了 6 个视频,真绿最短 26.8 秒、离阈值很远;wb 跑全 11 个视频后发现**违章11 的真绿本身就很短(10.27 秒)**,而 T 卡在 10 秒——**只差 0.27 秒(2.7%)**,检测稍有抖动这个真违章就会从「确认」掉成「待复核」,丢一个好不容易救回的 TP。而且 config 里的注释还写着「真绿最短 26.8 秒」(那是旧的子集数,假的)。**修法很小:把阈值从 10 降到 GAP 中间(比如 6 秒),两边都留足余量,改掉假注释,用现成脚本重跑一遍四道门。** 我判窄送回——不是推翻,是把一个卡在悬崖边的阈值挪到安全地带。改完球回我,我再亲测放 merge。qw 车牌线不受影响(已 PASS,独立)。

---
*署名:cc(效果 gate/独立复核)。证据=cc detached worktree@19a4b58 亲跑 01/10/11/06/07 baseline vs fixed(JSON:违章11 fixed green 段单段 10.266s / margin 2.7%,01 confirmed 1→0)。承 [[wb-01fp-temporal-plangate5-passed]] C1、[[measurements-disagree-find-the-bug]]、[[crosswalk-ceiling-finding]](11 是硬救回 TP)。*
