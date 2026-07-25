# wb 计划 v2:修 04/11 漏检(v11 斑马线条带跳变 ≠ 空掩膜)

> 作者:wb。状态:**v2 — 按 cc review(F1–F7)修订,待 cc 轻量复核后放行写码**。
> 上游 brief:`docs/handoff/2026-07-16-cc-direction-for-wb-crosswalk-flicker.md`
> 根因(已诊断):`_cv_v11` 每帧独立 `argmax` 无时间状态 → 04/11 竞争水平边缘(建筑线/路沿/阴影)偶超真斑马线 → 条带逐帧跳 → 错位帧 overlap 全 0 → 漏检。

---

## 0. 分工与口径(确认)
- 本文件 = **计划**,不含代码改动。CC 按 F1–F7 + 验证口径复核后放行,wb 再写码。
- 唯一标尺:`python scripts/eval_violations.py`(9 视频 balanced,事件级 1:1 重叠匹配)。
  基线 **P=0.538 R=0.778 F1=0.636 覆盖=0.426 车牌=2/7**。每改一处重跑,不回退。
- eval 跑 6-9 分钟、结束才出 stdout → 必须 `> file 2>&1` 后等完;**禁 `--reuse`**,须 fresh。
- scoped `git add <file>`,禁 `-A`;commit 末尾 `Co-Authored-By`。
- TDD:`pytest tests/unit tests/integration -q` 全绿,再跑 eval。

---

## 1. 生命周期审计结论(坑1 查证点 — cc 已实地确认属实)
| 路径 | 实例化点 | 跨视频复用? |
|---|---|---|
| 生产 `cli.py:62-64` | `comp["crosswalk"] = CrosswalkDetector(cfg)` 在 `run()` 内 | ❌ 每视频 `run()` 新建 → **不污染**(cc 确认) |
| `eval_violations.py:48-56` | `_run_pipeline` 调 `cli.run(...)` | ❌ 同上,逐视频新建 |
| `diag_mask_04_11.py:65` | 建一次后 `for v in [04,11]` | ⚠️ 复用 → 会污染(throwaway 诊断) |
| `diag_signal_timeline.py:44` | 同上 | ⚠️ 复用(throwaway) |

**处置**:加 `reset()` + `cli.run()` 顶部调用(防御性零成本);诊断脚本两视频间补 `reset()`。

---

## 2. cc review 响应摘要(F1–F7 → 处置)

| # | 严重度 | cc 指出 | v2 处置 |
|---|---|---|---|
| F1 | 🔴 | `band_max_drift_frac=0.15`(108px)废掉 11 修复:11 错位仅 55–89px<108px → 走 EMA 跟随 → flicker 没治 | **改为 0.05(=36px)**,卡在"11 最小错位 55px"之下、"真条带厚度 33px"之上(§3 阈值论证) |
| F2 | 🔴 | 跨帧 `band_reanchor_ratio` 不成立:score 帧间不可比(`mean_val/std_val` 每帧每带自适应 + 车辆 ×3.5) | **废弃 score ratio**,重锚改"候选同新位置持续 ≥K 帧"(照 `traffic_light.py:495-501` 的 `frames_seen>=reanchor_need` + 空间距离) |
| F3 | 🔴 | `_best_streak` 没定义清(best is argmax 不是真代码)且方向错:只要"best 是远的"就 +1,两个瞬态错跳也攒够 streak | **streak 锚在位置一致性**:本帧 best 距上帧 best < ~条带厚度(~33px)才 +1,否则清零;存 `(_last_best_cy, _cand_streak)` |
| F4 | 🔴 | `_mask_anchor` 用 `±band_h/2=±108px` → 216px 掩膜,真条带仅 ~33px(6–7× 过厚)→ 7 好视频 FP 回退 + 虚高覆盖率 | 建锚时存真实半厚 `(cy2-cy1)/2`,`_mask_anchor` 用它重建(不再用 band_h) |
| F5 | 🟡 | 首帧无条件硬锁无 warmup:frame-1 噪声 argmax 锁死后难纠 | 加 `band_warmup`(3–5 帧)裸 argmax,warmup 结束且位置稳定后才硬锁(复用 F3 streak) |
| F6 | 🟡 | 远距惩罚 ×0.4 改动全部 9 视频 score,可能把 7 好视频带回退 | 惩罚**仅在首锁前(warmup)生效**(`vehicle_penalty_scope=warmup_only`);锁定后靠位置 streak,不再动 score。仍须全量 eval 确认 7 好视频不回退 |
| F7 | 🟡 | 测试打在真实几何上;hold 阈值用秒/ticks 表达别用裸帧数 | 新增"竞争边距真条带 ~60px(11 真实失败几何)"用例 + "drift=40px 合理平移要跟随"边界用例;`band_hold_sec` 按 `inference.fps` 折算帧 |

**两个阈值的正式答复(cc 裁定)**:`band_max_drift_frac → 0.05`(非 0.15);`band_reanchor_ratio → 废弃`,改用位置持久帧数(`band_reanchor_frames`,照红绿灯 `reanchor_need`)。

---

## 3. 阈值论证(实测依据,非估计)
cc 实测(来自 `违章11_timeline.csv` + `违章04`):
- 帧高 **H = 720**(CSV `mask_cx2=1279` → 宽 1280 → 720p)。
- 视频 **11**:条带在两簇间跳 — 正确簇 `cy≈345`、错误簇 `cy≈256–293` → **错位幅度 55–89px**;真条带厚度 `cy2-cy1 ≈ 33px`。
- 视频 **04**:错位幅度 **>200px**(易排除);厚度 ~31px。
- 生命周期:cli.py:64 每视频新建,当前无污染。

推导:
- `band_max_drift_frac`:要挡住 11 的最小错位 55px,同时允许真条带逐帧小幅漂移(≤ 真厚度 33px)。取 **0.05×720 = 36px**:`55 > 36`(错误簇落窗外→不跟随)且 `33 ≈ 36`(真条带抖动在窗内→跟随)。上限别超 0.07(50px)。
- `band_streak_px`(位置一致阈值):≈ 真条带厚度 → **0.045×720 ≈ 32px**。
- `band_reanchor_frames`(远锚重锚需持久帧数):照红绿灯 `reanchor_need`,取 **15 帧**(≈0.5s @30fps)。flicker 的瞬态错跳达不到 15 帧连续一致,故不夺锚;真场景切换(持续新位置)能正确重锚。
- `band_warmup`:**4 帧**裸 argmax,足以让位置稳定后再硬锁(F5)。
- `band_hold_sec`:无候选时维持旧锚 **1.0s** → 折算 `int(1.0 × inference.fps)` 帧。

---

## 4. 改动清单(文件级,精确到函数)
1. **`src/redlight/models/crosswalk.py`**
   - `__init__`:加锚状态属性 + 读 cfg 参数(见 §5)。存 `self._cfg = cfg` 供 fps 折算。
   - 新增 `reset(self)`:清全部锚状态。
   - `detect`:`self._fi += 1`。
   - `_cv_v11`:保留行 135-151 的 score 计算(仅作**帧内** argmax 选键,**不跨帧比**);行 140-147 车辆乘子改为 `vehicle_penalty_scope == "warmup_only"` 且锚未锁时才施加远距惩罚;**替换行 153-169 的裸 argmax+掩膜**为锚定逻辑(§5)。
   - 新增 `_mask_anchor(h, w)`:用 `_anchor_cy ± _anchor_half` 重建(≠ band_h)。
   - 新增 `_lock(best)`:锁 `_anchor_cy/_anchor_half/_anchor_last_seen`。
2. **`src/redlight/app/cli.py`** `run()`:`comp` 建好后、帧循环前加 `comp["crosswalk"].reset()`。
3. **`configs/config.yaml`** `crosswalk:` 段加参数(§6)。
4. **`scripts/diag_mask_04_11.py`**:两视频间加 `cw.reset()`。
5. **新增 `tests/unit/test_crosswalk_v11_anchor.py`**(TDD)。

---

## 5. 算法(替换 `_cv_v11` 行 153-169)

**新增状态(`__init__`)**
```python
self._cfg = cfg
self._fi = 0
self._anchor_cy = None        # 锁定 run 中心 Y(abs px)
self._anchor_half = None      # 锁定 run 半厚(abs px),掩膜重建用(F4)
self._anchor_last_seen = -10**9
self._last_best_cy = None     # 位置一致性 streak 用(F3)
self._cand_streak = 0
self.band_warmup        = int(getattr(cfg.crosswalk, "band_warmup", 4))
self.band_reanchor_frames = int(getattr(cfg.crosswalk, "band_reanchor_frames", 15))
self.band_streak_px     = float(getattr(cfg.crosswalk, "band_streak_px", 0.045))  # ×帧高≈32px
self.band_max_drift_frac= float(getattr(cfg.crosswalk, "band_max_drift_frac", 0.05))  # ×帧高=36px(F1)
self.band_hold_sec      = float(getattr(cfg.crosswalk, "band_hold_sec", 1.0))     # F7:秒
self.vehicle_anchor_radius = float(getattr(cfg.crosswalk, "vehicle_anchor_radius", 0.15))
self.vehicle_far_penalty   = float(getattr(cfg.crosswalk, "vehicle_far_penalty", 0.4))
self.vehicle_penalty_scope = getattr(cfg.crosswalk, "vehicle_penalty_scope", "warmup_only")  # F6
```

**车辆门控(替换行 140-147,修正 F6)**
```python
# 远距惩罚仅在"锚未建立"时生效,锁定后靠位置 streak,不碰 score
apply_penalty = (self.vehicle_penalty_scope == "warmup_only" and self._anchor_cy is None)
if vehicle_boxes:
    cy_center_abs = y0 + (best_run[0] + best_run[1]) / 2.0
    nearest = min(abs(cy_center_abs - (vb[1]+vb[3])/2.0)/float(h)
                  for vb in vehicle_boxes if len(vb) >= 4)
    if nearest < self.vehicle_anchor_radius:
        score *= (1.5 + 2.0*(self.vehicle_anchor_radius - nearest))  # 近距加分(保留)
    elif apply_penalty and nearest > self.vehicle_anchor_radius*2:
        score *= self.vehicle_far_penalty                              # 远距惩罚(仅 warmup)
```
> 近距加分保留(位置先验、低风险);远距惩罚仅在首锁前生效 → 不污染已稳定帧,也不动 7 好视频的锁定后 score。

**锚定逻辑(替换行 153-169)**
```python
if not candidates:
    if self._anchor_cy is not None and (self._fi - self._anchor_last_seen) <= self._hold_frames():
        return self._mask_anchor(h, w)      # 维持旧锚
    return np.zeros((h, w), dtype=np.uint8)

candidates.sort(key=lambda c: c[0], reverse=True)   # 帧内 argmax(选键,不跨帧比)
best = candidates[0]
best_cy = (best[1] + best[2]) / 2.0

# --- F3 位置一致性 streak ---
consist = max(8, self.band_streak_px * h)           # ≈32px @720
if self._last_best_cy is not None and abs(best_cy - self._last_best_cy) <= consist:
    self._cand_streak += 1
else:
    self._cand_streak = 1
self._last_best_cy = best_cy

# --- F5 首锁 / warmup ---
if self._anchor_cy is None:
    if self._fi < self.band_warmup:
        self._anchor_cy = best_cy                  # warmup:裸 argmax,不硬锁
        self._anchor_half = (best[2]-best[1])/2.0
        self._anchor_last_seen = self._fi
        return self._mask_of(best, h, w)
    self._lock(best, h, w); return self._mask_anchor(h, w)   # warmup 过+位置稳→硬锁

# --- 已锁:位置判定(F1/F2/F3 合流) ---
drift = abs(best_cy - self._anchor_cy)
if drift <= self.band_max_drift_frac * h:
    # 同位置(≤36px):直接采 best 跟随(小漂移,不会跨到 55–89px 错簇)
    self._anchor_cy = best_cy
    self._anchor_last_seen = self._fi
    return self._mask_anchor(h, w)                 # 用 _anchor_half 重建(F4)
# 远锚:不 EMA,仅当"同新位置持续 ≥ band_reanchor_frames"才重锚(照红绿灯 reanchor_need)
if self._cand_streak >= self.band_reanchor_frames:
    self._lock(best, h, w); return self._mask_anchor(h, w)
if (self._fi - self._anchor_last_seen) <= self._hold_frames():
    return self._mask_anchor(h, w)                 # 维持旧锚(治 flicker 核心)
return np.zeros((h, w), dtype=np.uint8)            # 锚丢失超时
```

**掩膜重建(F4)**
```python
def _mask_anchor(self, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = max(0, int(self._anchor_cy - self._anchor_half))
    y2 = min(h, int(self._anchor_cy + self._anchor_half))
    cv2.rectangle(mask, (0, y1), (w, y2), 255, cv2.FILLED)
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 8))
    return cv2.dilate(mask, k_d, iterations=1)      # ~真条带厚度 + 轻膨胀,非 216px

def _mask_of(self, best, h, w):   # warmup 用:同现有行 161-167
    ...

def _lock(self, best, h, w):      # F4:存真实半厚
    self._anchor_cy = (best[1]+best[2])/2.0
    self._anchor_half = (best[2]-best[1])/2.0
    self._anchor_last_seen = self._fi

def _hold_frames(self):           # F7:秒→帧
    fps = getattr(getattr(self._cfg, "inference", None), "fps", 30) or 30
    return int(self.band_hold_sec * fps)
def reset(self):
    self._fi = 0; self._anchor_cy = None; self._anchor_half = None
    self._anchor_last_seen = -10**9; self._last_best_cy = None; self._cand_streak = 0
```

**保守性(坑2 不回退论证)**:首锁经 warmup 过滤噪声;真斑马线迟现/小抖动走"同位置跟随"(≤36px);场景切换靠"15 帧连续一致的新位置"才重锚,瞬态 flicker 错跳(55–89px、非连续)永远攒不够 streak → 不夺锚;掩膜缩到真厚度(F4)反而减少 7 好视频 FP。验证必须全量 9 视频 eval 逐视频核对。

---

## 6. config 参数(替换原 4 参数)
```yaml
crosswalk:
  method: auto
  band_warmup: 4              # F5:首锁前裸 argmax 帧数
  band_reanchor_frames: 15    # F2/F3:远锚重锚需持久帧数(照 traffic_light reanchor_need)
  band_streak_px: 0.045       # F3:位置一致阈值(×帧高≈32px≈真条带厚)
  band_max_drift_frac: 0.05   # F1:同位置跟随窗(×帧高=36px)
  band_hold_sec: 1.0          # F7:无候选维持旧锚时长(折算帧)
  vehicle_anchor_radius: 0.15
  vehicle_far_penalty: 0.4
  vehicle_penalty_scope: warmup_only  # F6:远距惩罚仅首锁前生效
```

---

## 7. 4 个坑逐条回应(更新)
- **坑1 跨视频污染** → §1。cc 已确认生产每视频新建无污染;仍加 `reset()` + `run()` 顶部调用 + 诊断脚本补 `reset()`。
- **坑2 不回退 7 好视频** → §5 保守性 + F4 缩掩膜减 FP;**全量 9 视频 eval** 逐视频核对;阈值经 §3 实测推导。
- **坑3 静止约束错层** → §4/§5 不 plumb 静态态,用"车框距离门控"代理;远距惩罚仅在 warmup 生效(降副作用)。
- **坑4 修 flickering ≠ 04 转 TP** → 中间标尺看 **occupancy/覆盖率 delta**(掩膜对位后静止车 overlap 应不再恒 0);04 灯态短绿漏判/11 绿灯迟 5s 为单独议题(本次不做),报告单列避免误判。

---

## 8. 测试用例(TDD,真实几何)
合成帧构造器:上半部强水平梯度"假边"(建筑线),下半部交替明暗"真斑马条纹";支持真条纹缓慢平移或跳到别处。
1. `test_anchor_rejects_competing_edge`:真条纹在底部、**竞争假边距真条带 ~60px**(11 真实失败几何,非顶部远边)→ 连跑 N 帧断言掩膜 Y 不跳到假边(F1 直接验)。
2. `test_anchor_follows_slow_drift`:真条纹**总平移 40px**(每帧 <drift 窗)→ 断言锚逐步跟随、不丢(边界用例卡 F1 阈值)。
3. `test_anchor_reanchors_on_persistent`:真条纹跳到远处并**连续 ≥ band_reanchor_frames 帧一致** → 断言锚迁移到新位(F2/F3)。
4. `test_flicker_no_steal`:best 在 345↔256(89px)间逐帧跳 → 断言 streak 每帧清零、锚不丢到错簇(核心 flicker 用例)。
5. `test_reset_clears_state`:视频A(真条纹在顶)→ `reset()` → 视频B(在底)首帧不被 A 偏(坑1)。
6. `test_warmup_no_hardlock_noise`:前 `band_warmup` 帧注入噪声 → 断言 warmup 期内不硬锁错误位置(F5)。
7. `test_mask_thickness_real`:锁定时存真实半厚 → 断言 `_mask_anchor` 高度 ≈ 真条带厚(非 band_h)(F4)。
8. `test_vehicle_penalty_warmup_only`:远假边无车近 → warmup 期不夺锚;锁定后即使远边出现也不动 score(F6)。

---

## 9. 验证口径(CC review 重点)
1. **TDD 门禁**:`pytest tests/unit/test_crosswalk_v11_anchor.py tests/unit tests/integration -q` 全绿。
2. **全量 eval(禁 --reuse)**:`python scripts/eval_violations.py > /tmp/eval_wb_flicker.txt 2>&1`(6-9 分钟等完)。对比基线,逐视频核对 **7 好视频不回退**(F6 重点);报 **04/11 occupancy/覆盖 delta**。
3. **中间指标(坑4)**:复用 `diag_mask_04_11.py`(已补 reset)跑前后对比,看 `max_overlap` 是否仍恒 0(应不再恒 0)。
4. **不回退承诺**:F1 总体 ≥ 0.636;7 好视频各自 P/R/F1 不劣于当前;04/11 occupancy 上升。
5. **报告含**:9 视频逐行 P/R/F1/覆盖 + 04/11 occupancy delta + F1–F7 各项验证结论。

---

## 10. 不做项(与 brief 对齐)
- ❌ 方案3(条纹纹理 FFT/方差)、方案4(换 YOLO seg)→ 记长期。
- ❌ 不削弱 eval 口径;不恢复 `_dedup`;不动灯态 `enforce_transition_limit`;车牌漏牌另案。
- ❌ 不 plumb 静止态进 `detect()`(坑3 处置);04 灯态短绿漏判不在此次范围。

---

## 11. 实施顺序(放行后)
1. TDD:写 `test_crosswalk_v11_anchor.py`(合成帧构造器)→ 此时必败。
2. `crosswalk.py`:状态/`reset()`/`detect` 计数/车辆门控/`_cv_v11` 锚定 + `_mask_anchor`/`_lock`/`_hold_frames`。
3. `cli.py`:`run()` 顶部 `reset()`;`config.yaml`:§6 参数。
4. `pytest` 全绿。
5. 全量 eval(禁 reuse),对比基线,写报告。
6. scoped `git add` + 署名 commit;push 前等 CC 确认。

---

/cc: 计划 v2 已按 F1–F7 修订完毕。重点请复核:(a) **F2/F3** 新重锚逻辑(`_cand_streak` 位置一致性 + `band_reanchor_frames` 持久帧数,废弃 score ratio);(b) **F1/F4** 取值(`band_max_drift_frac=0.05`=36px 卡在 11 错位 55px 下;`_mask_anchor` 改真实半厚);(c) **F6** 远距惩罚仅 warmup 生效。通过后即放行写码。
