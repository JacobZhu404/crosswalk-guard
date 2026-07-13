# datasets/ 数据约定 (所有 agent + 两台机遵守)

本工程跨 **两台机**(Mac 公司 / Windows 家里, 很少同时在线)+ **多 agent** 协作。
数据按"**能否再生 / 大小**"分层, 避免把大二进制推进 git(会永久撑爆历史)。

## 分层规则

| 数据 | 位置 | 进 git? | 说明 |
|------|------|---------|------|
| 代码 / 配置 | 仓库 | ✅ | 正常版本化 |
| **GT 真值** `datasets/gt/` | 仓库 | ✅ | 小、需版本化(events/light_states/videos/light_state/…) |
| **训练 crop** `datasets/ped_signal/` | 仓库 | ✅ | 小(48×48 JPEG,几 MB),是真训练数据;跨机 `git pull` 复用 |
| **模型权重** `models/*.onnx` | 仓库 | ✅(小的) | `ped_signal.onnx`~34KB 可进 git;`yolov8n.pt`(~6MB)让 ultralytics 自动下,别提交 |
| **抽帧缓存** `datasets/frames/` | 各机本地 | ❌ gitignore | **可再生**:各机 `python scripts/extract_frames.py --all --fps 4` 重建,不同步 |
| **源视频** `input_video/*.mp4` | git 之外 | ❌ gitignore | 真源数据但大;见下"同步" |
| **运行产物** `data/output/` | 各机本地 | ❌ gitignore | 标注视频/评测画廊等,可再生;**勿 `git add -f` 硬塞进库** |

## 为什么不把大文件推 git
git 把二进制**永久写进历史**:每次 clone 拖全量,想删要 `git filter-repo`/BFG 重写历史。二进制也无法 diff/合并。→ **代码进 git,大数据进别处,git 只存小数据/指针。**

## 源视频跨机同步 (待定, 2026-07 用户回家解决)
- 两台机很少同时在线 → **P2P(Syncthing/LocalSend)不适用**(需两端同时开机)。
- 公司网络**封了国产网盘**(坚果云/百度网盘等)→ 暂不可用。
- 目标方案:**永远在线的 hub** + 异步上传/下载。候选:阿里云 OSS / 腾讯云 COS(S3 兼容)+ `rclone`/`ossutil`,或 **DVC-on-OSS**(数据版本跟 git commit)。凭证用 **RAM 子账号最小权限 + 环境变量**,勿进 git。
- **关键**:源视频只需在有它的那台机 **跑一次** `build_ped_signal_crops.py` 抽出 crop → crop 进 git → 另一台机只靠 crop 训练,**日常无需视频跨机**。

## M1 行人信号分类器数据流 (Phase2)
```
源视频/帧 → scripts/build_ped_signal_crops.py → datasets/ped_signal/{crops, labels.csv}
          → 灯态画廊人工校验 verified=1
          → scripts/train_ped_signal.py → models/ped_signal.onnx (LOVO 评测)
          → configs/config.yaml: traffic_light.method=ped_classifier 启用
```
`labels.csv` 列: `crop_path, video, frame_ts, x1,y1,x2,y2, source, label(walk|stand|off), verified(0|1)`。
</content>
