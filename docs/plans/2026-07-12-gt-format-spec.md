# GT 标注格式规范 v1 (事件级)

> 依据 2026-07-12 grill-me 对齐结论 (4 项主干决策 + 细粒度/迁移/负例清单 3 项叶子决策)。
> 本文档是 `datasets/gt/events.csv` 与 `datasets/gt/videos.csv` 的**唯一权威格式定义**。
> 配套校验: `python scripts/validate_gt.py` (ERROR 必须清零后再跑评测)。

---

## 1. 为什么改格式 (背景)

旧 `events.csv` 把"推断绿"(反光/没拍到灯, 由车占道反推) 与"直接看到的绿"都标成 `light_state=green`,
导致评测逼着检测器去"看见"不存在的灯, 准召被污染段扭曲。
本规范按**设计文档 D1** 的精神拆分真相与可见性:
**信号灯是否直接入镜(`light_evidence`)** 与 **实际灯态(`light_state`)** 解耦。

---

## 2. events.csv (事件级真值)

一行 = 一个"情况均匀"的时间段。时间段内部灯态/占道行为应一致; 不一致就拆行。

### 列定义 (顺序固定, 9 列)

| # | 列名 | 取值 | 说明 |
|---|------|------|------|
| 1 | `video` | `违章NN` | 与 `input_video/违章NN.mp4` 对应 |
| 2 | `start_s` | float 秒 (含) | 段起点 |
| 3 | `end_s` | float 秒 (不含) | 段终点, 须 `> start_s` |
| 4 | `light_state` | `green`\|`red`\|`flashing`\|`unknown` | 实际灯态真相 |
| 5 | `light_evidence` | `visible`\|`inferred`\|`occluded` | **灯是否直接入镜**; `light_state=unknown` 时**必须留空** |
| 6 | `is_violation` | `0`\|`1` | 法律真相: 该段是否构成"占道阻碍行人"违章 |
| 7 | `violating_plates` | `;`分隔 | 本段**真正违章**的车牌; 特殊标记见 §3; 无则空 |
| 8 | `other_plates` | `;`分隔 | 同框占道但**非违章**的车 (如后退车/后车); 无则空 |
| 9 | `note` | 自由文本 | 读牌时刻/上下文/疑难说明 |

### `light_evidence` 语义 (评测据此定"期望输出")

- `visible` → 信号灯**直接入镜**, 检测器**期望输出字面 `light_state`** (green/red/flashing)。
- `inferred` → 灯**不在画面**, 由上下文(车静止占道/相位逻辑)推断灯态。
  检测器**期望输出 `unknown`** → COT 推断绿灯 → `verdict=review` (对应 D1)。
- `occluded` → 灯**存在但被遮挡**(如大车挡住)。同 `inferred`, 期望 `unknown` → review。
- `light_state=unknown` 的段 (无灯态信息) → `light_evidence` **留空**, 期望 `unknown`。

### 验收分两层 (呼应设计文档 §7)

1. **visible 违章段**: 检测器须自动检出灯态 + 占道 → 计入端到端 recall (目标 ≥0.9)。
2. **inferred/occluded 违章段**: 检测器须输出 `unknown` 进入 review 队列 (不漏、不误清为通过)。
   用 `review 覆盖率` = 推断段里 pred==unknown 的比例 衡量。

---

## 3. 车牌字段规则

- 分隔符: 英文分号 `;`, 无空格。例: `京LNE560;无牌`
- 每个 token 三选一:
  - **真实车牌**: `京LNE560` 等 (字母/数字/汉字, ≥4 字符)
  - `无牌`: 该车无车牌
  - `?`: 看不清/无法读取
- `is_violation=1` → `violating_plates` **必非空**。
- **读牌时刻可不在违规窗口内** (E19: 车牌须视频全局读取, 按 track 关联占道车)。
  车牌列按"该车出现在哪段"填写, 具体读清时刻写进 `note`。
- 多车段: 违章车进 `violating_plates`, 占道但非违章车进 `other_plates`。
  例 05: `violating_plates=京ADH9206` (白车占多), `other_plates=京N541E6` (黑车后退不算)。

---

## 4. 时间区间规则

- 同一视频内**段不可重叠** (校验脚本会查)。
- 段不必覆盖整段视频; **间隙帧**默认期望 `unknown`、`is_violation` 隐含 0。
- 细粒度: 若一个 green 段内部夹杂"没拍到灯"的间隙 (如 08),
  **拆成 visible 子段 + inferred 间隙子段** (见 §6 工作流)。

---

## 5. videos.csv (视频级负例清单)

| 列 | 取值 | 说明 |
|----|------|------|
| `video` | `违章NN` | |
| `has_violation` | `0`\|`1` | 该视频是否含任意违章事件 (= 任意 events 行 is_violation=1) |
| `notes` | 自由文本 | |

作用: 显式声明 `违章01`/`违章10` 为**纯负例**, 使"01/10 零误报"验收有标尺。

---

## 6. 标注工作流 (推荐)

1. 看视频, 对每个"情况均匀"的时段写一行。
2. 灯态按**实际真相**填 `light_state`; 同时判断灯是否直接入镜填 `light_evidence`。
3. 占道违章车填 `violating_plates`, 非违章占道车填 `other_plates`。
4. 段内灯可见性不一致 → 拆行 (visible / inferred 子段)。
5. 跑 `python scripts/validate_gt.py` 清零 ERROR。
6. 跑 `python scripts/eval_light_all.py --out data/output/light_eval.csv` 看灯态准召 + review 覆盖率。

---

## 7. 示例

```csv
video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note
违章02,0,21,red,visible,0,京LNE560,,白车占道但红灯, 不算违章
违章02,21,68,green,visible,1,京LNE560,,白车静止占道+绿灯=违章
违章03,74,134,green,inferred,1,京ABV3428;无牌,,绿灯仅能从公交车玻璃反光看到(难例)
违章08,0,12,green,visible,1,京ACD5358;京ACW6553,,灯可见
违章08,12,30,green,inferred,1,京ACD5358;京ACW6553,,中段手机未拍到灯, 应填充绿灯
违章08,30,50.8,green,visible,1,京ACD5358;京ACW6553,,灯重新可见
```
