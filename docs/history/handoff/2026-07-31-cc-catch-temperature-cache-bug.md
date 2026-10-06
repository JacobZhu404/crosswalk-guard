# CC 独立复核 ef84c3c(S1+多seed):S1/聚合正确,但揪出**续跑路径静默 bug**——温度不入 state_dict

> 出自 cc(arbiter)。逐文件核 `ef84c3c` + 实机验证。**S1 与多seed聚合正确;发现一个缓存/续跑正确性 bug,趁全量后台跑(task qhOAwE)未完先拦。**

## 0. 核过通过项
- **S1 正确**:`eval_video` 传 `governing_threshold=0.0`(内部不弃权),照记 `best_cand`+`best_conf`,弃权门交 `metrics_for_tau` 按每个扫描 τ 施加(`best_conf>=tau`)。τ 敏感性曲线全段诚实,headline 不变。✅
- **多seed聚合方向正确**:worst-seed 取 `max(fg_rate)`/`max(miss)`(误绿/漏绿越高越差=最差种子),gate 用 `max(misses)<=80`(报最差不报幸运),符合"多seed报min"。mean±std / worst-video / 11视频分解 / 03单列 / 06·11 N/A / τ敏感性(pool)均对。✅
- **cache 有 `_s1` 后缀**给 rows 做 cache-busting,避免复用 pre-S1 旧行——细节到位。✅
- **wb 补 worst-seed(min) 是真发现**:原 main 只用 seeds[0],worst-seed 聚合缺失。这是 cc 裁定漏揪、wb 抓对的,记功。

## 1. 🔴 续跑路径静默 bug:`GoverningDiscNet.temperature` 不在 state_dict
`self.temperature`(A3 温度缩放校准值)是普通 float 属性,**不是 buffer/参数,不进 `state_dict()`**。实机验证:
```
temperature in state_dict? False
reloaded temperature = 1.0   # 校准值 2.7 存盘再 load 后丢失
```
后果(正是 cache 要防的 SIGKILL 续跑场景):
- `_load_or_train` 命中 `.pt` 缓存 → `load_state_dict` 只恢复权重,**temperature 退回 1.0(未校准)**;
- 而 `_tau_for_fold`/`_eval_cached` 的 τ、rows 可能是**上次进程用校准态模型算的**(已落盘)→ 续跑后 **T=1.0 模型 与 校准态 τ/rows 静默不一致**;
- **A3 的立意(温度缩放使 τ 跨折可比)在任何续跑后失效**——所有 reload 折都变 T=1.0。

**严重度**:单进程一次跑完 = 正确(内存模型带校准 T,不走 reload);**一旦被 SIGKILL 续跑 = 报告静默错**。wb 恰恰为 macOS SIGKILL 建了这套缓存 → 续跑路径大概率被触发。

## 2. 处置
### 对在跑的全量(task qhOAwE)
- **仅当它单进程一次性跑完(无 kill/resume)才可信**。跑完先查日志:若中途重启过(cache 命中 reload 训练折),**该报告作废重跑**。
- 现进程内存态模型是校准的,故不中断即正确;03/seed0 复用的是冒烟已缓存的 τ+rows(校准态)+ reload 模型未被使用,巧合无害。

### 代码修(wb,不接线前)
- **持久化 temperature**:`_load_or_train` 存/取时带上 T。两种干净法:
  1. 存 `{"state": state_dict(), "temperature": float(model.temperature)}`,load 后 `model.temperature = ckpt["temperature"]`;或
  2. 加旁挂 json(同 `tau_*.json` 风格)。
- 修后**跑一个"训练→存→reload→score 对拍"单测**(reload 前后 `score_crop` 必须逐位相等),防回归。这类"存盘即截断/丢状态"正是要 arbiter 落地查([[no-open-truncate-rewrite]] 同源教训:别取信"已实现",实机验证)。

## 3. 放行边界(不变)
- 温度持久化修 + reload 对拍单测过之前,**续跑产出的报告不采信**。
- 其余红线不变:gate 不过净回退不接线、GT 不进推理、LOVO 去循环、权重不进库、报告交 cc 复核 + Jacob 抽检才谈接线。

---
**一句话给 wb**:S1 与多seed聚合都对(worst-seed 方向正确)。但揪出续跑 bug:`temperature` 不入 state_dict,SIGKILL 续跑后 reload 模型退回 T=1.0,与已缓存的校准态 τ/rows 静默打架,A3 温度缩放形同虚设。**在跑的全量仅当一次性跑完才可信(跑完查有无 resume);修法=`_load_or_train` 持久化 T + reload 前后 score 对拍单测。**
