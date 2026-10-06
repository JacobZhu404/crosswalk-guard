# CC Fork A:公开行人灯数据侦察结果 + 给 wb 的"量误绿一次"brief + A/B 天平更新

> 出自 cc(arbiter)。Jacob 拍板走 **fork A**(打补丁前行、先量误绿定标尺,并行侦察公开数据决定要不要 B)。
> 本文=A 的两个 cc 交付:①公开数据侦察(cc 已做);②给 wb 的误绿测量 brief(待 Jacob relay)。

## 0. 一句话
**公开行人灯数据比 07-12 调研估计的多得多,且带倒计时类别 + 部分中国样本 + MIT 许可**——这既降 B 的成本/风险,又给 A 一个白赚的杠杆(状态分类器可用公开数据预训练,绕开被毒化的挖矿 crop)。

## 1. 侦察结果(更新 `docs/diag/model-dataset-survey-2026-07-12.md` 的"行人灯极稀缺"结论)
07-12 调研只看了机动车灯数据集(LISA/Bosch/TT100K),结论"行人灯标注极少"**不准确**。聚焦搜索后,行人灯专门数据集有三类:

| 数据集 | 规模 | 标注形态 | 类别 | 许可 | 对我们 |
|---|---|---|---|---|---|
| **ImVisible / PTL**(samuelyu2002) | **5,059 图**(含 4032×3024 原图) | **整图分类** + 斑马线中线坐标(非灯框) | Red / Green / **Countdown-Green / Countdown-Blank** / None | ✅ **MIT(商用可,已核实)** | ⭐ 状态分类器预训练金矿;**含倒计时类别=贴合北京行人灯**;egocentric(盲人手机视角) |
| **ronaldosm / PTL-Crosswalk** | **4,801 图**(巴西/法/**中国**/意/德) | 整图分类 + 斑马线坐标 + IScrosswalk | 0 红 / 1 绿 / 2 无 | ❌ **无许可(已核实 license=null)** | **禁直接用**=默认 all-rights-reserved,且聚合了 Cheng(中)/Roters(德)第三方集;要用须联系作者或改用其底层源集各自许可 |
| **Roboflow Universe**(ono-gedd7/pedestrian-traffic-light-puf4a) | **926 图**(单项目) | **框级检测**(YOLO/COCO 就绪) | 6 类(green/red/pedestrian TL/signal-light/traffic_light/trafficlight,**混行人+车灯,taxonomy 乱**) | ✅ **CC BY 4.0(已核实)** | 唯一**框级**来源但**小且脏**;仅够预训练/增广,不足以单独训稳检测器;西方视角 |

**关键区分 + 许可核实(2026-07-26 cc)**:ImVisible / ronaldosm 是**整图分类 + 斑马线线**(不是逐灯 bbox)→ 只喂**状态分类器**;**框级**只有 Roboflow(小/脏)。许可:**ImVisible=MIT ✅**、**Roboflow ono=CC BY 4.0 ✅**、**ronaldosm=无许可 ❌(禁用)**。

## 1b. 核查后的净结论
- **状态分类器预训练 = 放行(强)**:ImVisible(MIT/5059/带倒计时)干净可商用,立即能预训练 walk/stand/off,绕开毒化 crop。**这是不管 A/B 都该做的稳赚一步。**
- **ronaldosm = 弃用**:无许可默认保留全部权利 + 第三方聚合,法务不干净;且 ImVisible 已覆盖状态分类需求,不值得为它冒险/联系作者。
- **检测器(B 的地基)= 公开数据帮助很小**:框级只有 Roboflow(~900 图/项目、6 类混杂、西方视角),**不足以单独训稳行人灯检测器**,顶多预训练 + 重度 fine-tune 我们 11 视频。→ **B 的核心成本(密集标注我们自己的视频)并未因公开数据显著下降。**

## 2. 侦察对 A/B 的影响
1. **状态分类器(误绿真正的病灶)可被公开数据预训练**——这是独立于 A/B 的新杠杆:
   - 现状病根:挖矿 crop 被偏框毒化(prior-misframe),分类器从没见过真行人灯正样本([[prior-misframe-rootcause]] / [[light-classifier-retrain]])。
   - 现在:~1 万张 MIT/公开的**已标行人灯状态图**(含倒计时、含中国),可直接预训练/混训 walk/stand/off 头,再用我们干净 crop fine-tune。**绕开毒化数据。**
2. **检测器(B 的地基)成本下降但未清零**:框级公开数据仅 Roboflow(西方 egocentric 为主),规模/许可待核;真要建行人灯检测器,公开数据可预训练 backbone,仍需我们 11 视频 fine-tune 补域差(视角:我们是旁观者拍路口,公开多是行人自拍视角)。
3. **天平(核查后修正)**:我原先"偏 B"的倾向**被侦察结果往回拨了**——
   - 状态分类器有 ImVisible(MIT)可预训练=**A 变强**(补丁路线现在能配一个像样的状态分类器,而非只靠毒化 crop);
   - 检测器的公开框数据小/脏/无干净许可=**B 未变便宜**(仍要密集标注我们 11 视频)。
   - 合起来:**先做 A(ImVisible 预训练状态分类器 + 量误绿),只有当误绿仍高且成因主要是"定位错(选错灯/框偏)"时才认 B** —— 因为定位错既非状态分类器能救、也非公开框数据能便宜救,那才轮到"密集标注→检测器"这笔贵投资。

## 3. 给 wb 的 brief:量"误绿一次"(A 的核心产出,待 Jacob relay)
**目标**:把"当前管线到底误绿多严重"从猜变成一个数字——这是决定"要不要 B/检测器"的标尺。**只测量,不改生产、不接线、不重训。**

- **定义**:误绿 = 真值为"非 walk"(stand/off/红/无灯)的帧/事件,被判成 walk(绿)→ 可能触发假违章。以 `datasets/light_location_gt.json` + 现有 state GT 为真值口径。
- **口径去循环(硬要求)**:禁用"在 prior 处采样再比同一 GT"的自证指标(Diag2 坑,[[eval-methodology-gap-overfit]])。定位用逐帧 YOLO∪HSV 候选 + 当前最好选灯(L1/L2/L3,GT-free 路径,**不得用 GT 播种**),状态用现有分类器;真值只做比对,不进推理。
- **输出**:①总体误绿率(帧级 + 事件级);②按 11 视频分解(定位哪些视频/场景在误绿);③误绿样本的成因归类(候选选错=车灯被当行人灯 / crop 偏框 / 状态判错 dark-green 等),各占比。
- **交付**:一份 report(`docs/reports/`)+ 误绿样本画廊供 Jacob 抽检。**先出方法(cc review 去循环口径)再跑。**

## 4. cc 建议的推进顺序
1. **wb**:先出 §3 误绿测量方法(cc review 口径)→ 跑 → 出数字 + 画廊。
2. **cc(已完成本轮核查)**:许可核实完毕(ImVisible=MIT ✅ / Roboflow=CC BY ✅ 但小脏 / ronaldosm=无许可 ❌)。**结论:B 的检测器公开数据帮助有限,状态分类器有 ImVisible 强 bootstrap。**
3. **拿到误绿数字后**:误绿可接受→A 收工(接 ImVisible 预训练的状态分类器即可);误绿仍高**且成因主要是定位错(选错灯/框偏)**→才认 B 建检测器(公开框数据只够预训练,主力靠 11 视频密集标注 fine-tune)。

## 5. 红线(不变)
只测量不接线、生产 prior/权重不动、gate 不过不接线、禁 GT 进推理、去循环口径 cc 先审、scoped git、署名、trunk main、TDD 先。**公开数据许可逐个核实**(ImVisible=MIT 可商用;ronaldosm/Roboflow 未定)。

---
**Sources(侦察)**:
- ImVisible/PTL:https://github.com/samuelyu2002/ImVisible (MIT, 5059 图, Red/Green/Countdown-Green/Countdown-Blank/None)
- ronaldosm/PTL-Crosswalk:https://github.com/ronaldosm/PedestrianTrafficLightsAndCrosswalkDetection (4801 图, 含中国, 整图分类)
- Roboflow Universe:https://universe.roboflow.com/search?q=class:%22traffic+light%22 (框级 Red/Green PTL, 逐项目待核)
