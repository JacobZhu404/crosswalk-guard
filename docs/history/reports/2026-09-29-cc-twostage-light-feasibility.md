# 两段式灯态(先定位 panel 再判色)决定性可行性测试 — cc 2026-09-29

> 署名:cc(arbiter,Jacob 离机自闭环) 承 Jacob 拍板"可以先识别交通灯 panel 再识别颜色不,两段式"
> 数据:`datasets/gt/light_canonical_gt.json`(399 帧 / 717 governing box / 全 11 视频,Jacob 手标真色+框+governing)
> 三个决定性实验,全部我独立真跑。诊断脚本 `/tmp/cc_twostage_*.py`(只读,零生产改动)。

---

## 0. 结论(三条,先给答案)

1. **色元 stage 本身成立**:给定**紧 panel 框**,生产同款 HSV 投票 → 真绿召回 95.5%,假绿在 6/9 视频归 0%。**证 Jacob 直觉对:多数视频的假绿是"定位"问题(ROI 太大采到环境绿),紧框能治。**
2. **但定位 stage 是瓶颈,且卡在老根因**:①YOLO-COCO(class9)对 governing **行人**灯中心命中仅 69%,05 塌到 10%(COCO 找车灯不找行人灯);②生产可行版(prior ROI 内 blob 定位,无 GT)只把假绿 30.1%→27.3%、真绿召回 81.2%→85.7% —— **残余假绿全在偏框 prior 视频(05=100%,09=82%)**,ROI 本身就指着环境绿,ROI 内再定位也没用。承 [[prior-misframe-rootcause]]。
3. **两段式≠记分收益(是加固,像 b2)**:①它救不了 04(见 §3);②它压的假绿(05/09)已被 #3 时序门(6.0s)在记分层中和,当前 P=1.000 里那些假绿根本没记分。**现在接线 → F1 零变动 + 冒破 P 风险。**

## 1. 实验A:紧 GT panel 框 + 生产 HSV 投票(隔离色元 vs 定位)

n=297 帧。混淆矩阵:green→green 147/154(95.5% 召回);red/off→green **18/143=12.6% 假绿**。
逐视频假绿率:01/03/04/07/10/11 = **0%**;02=8%;09=18%;06=30%;**05=67%**。
→ 紧框把假绿从生产的 30%+ 压到多数视频 0%,**但 05/06/09 紧框也治不好**(色元问题,非定位)。

## 2. 实验B:YOLO 能否定位 governing panel(Stage-1 生产可行性)

YOLO(class9,conf0.05,imgsz1280)vs GT governing box,中心命中率:04=100/07=93/06=92/02=80/01=81,但 **05=10% / 03=41% / 08=52%**,ALL=69%。meanBestIoU=0.601。
→ COCO traffic-light 类偏车辆灯,对行人 governing 灯不稳,05 几乎全灭 = 与判别器线同一堵墙。

## 3. 实验C(关键):两段式能否解锁 04?

假设:若两段式把 01 假绿 run 压到 04 真绿(1.2s)以下,就能降 #3 门恢复 04。
实测 8fps 逐帧 max green run:
- **01(负例)**:full=2.00s / blob=2.25s
- **04(真违章)**:full=0.75s / blob=1.00s
→ **04 真绿 run 反而比 01 假绿 run 短**,blob 投票把两者都拉长。**无任何时序阈值能分开 04 与 01。04 结构性无解被第三次独立坐实**(承判别器 Phase A、09 prior 重定位两条)。

## 4. 裁定 + 建议

- **两段式方向对(Jacob 直觉正确),但当前接线是加固非修复**:F1 已 0.941/P=1.000,两段式压的假绿不在记分路径上,接了不涨分还冒破 P 险 —— 与 b2 同性质(零输出改动的基础设施)。
- **真正需要两段式的是"偏框 prior 三视频(05/06/09)+ 判别器线"**:但那需要①prior 重定位(05/06/09,IoU=0,曾被 Jacob 判红线)②或行人灯专用检测器(非 COCO)。都属被 HOLD/未拍的战略线。
- **建议**:两段式作为**判别器线解冻后的 Stage-1 组件**储备(已证色元在紧框上 95.5% 召回,直接可用);**现在不单独接线**(无记分收益)。cc 把可行性数据钉在此,供 Jacob 决定何时连同判别器线一起启动。

---
*署名:cc(arbiter)。证据=canonical GT 397帧三实验(紧框假绿12.6%/YOLO中心命中69%/01-04 green run 2.0s vs 1.0s)。承 [[accuracy-ceiling-blocked-on-gates]] [[prior-misframe-rootcause]] [[light-classifier-retrain]] [[canonical-light-gt]]。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
