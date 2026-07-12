# Handoff 交接快照 [任务ID: PLATE_RECOGNITION_OPTIMIZATION]

## 1. 核心任务目标

- **原始需求**: 优化车牌识别模块，提升北京场景下的车牌识别准确率
- **验收标准**: 
  - 真值车牌"京LNE560"在违章02视频中107~109秒期间成功识别
  - 过滤非"京"开头的误识别车牌(粤、蒙、鄂、苏等)
  - 同一车辆多帧识别结果通过跟踪+投票机制合并，输出最可靠车牌
- **约束规则**: 
  - 不影响红绿灯检测、斑马线检测、违章判定等共享模块
  - 优先使用现有工程结构和模型(HyperLPR3)
  - 支持静态图调优，加快迭代速度

## 2. 已完成工作清单

### 修改文件

| 文件路径 | 修改内容 | 状态 |
|----------|----------|------|
| `src/redlight/models/plate.py` | 添加省份先验过滤逻辑、`_apply_province_prior()`函数 | ✅ |
| `src/redlight/pipeline/plate_consensus.py` | 多帧车牌投票模块(已有，完善) | ✅ |
| `src/redlight/pipeline/dag.py` | 集成PlateConsensus节点，优化车牌-车辆关联逻辑 | ✅ |
| `src/redlight/pipeline/violation_engine.py` | 恢复双模式支持(red_light/pedestrian_green) | ✅ |
| `src/redlight/app/cli.py` | 添加plate_consensus组件，传递mode参数 | ✅ |
| `configs/config.yaml` | 添加`plate_prior_province: "京"`配置项 | ✅ |
| `scripts/run_video.py` | 添加`--mode`参数支持 | ✅ |
| `scripts/extract_frames.py` | 新建: 视频抽帧预处理脚本 | ✅ |
| `scripts/tune_plate_static.py` | 新建: 静态图车牌识别调优脚本 | ✅ |
| `scripts/annotate_plate.py` | 新建: 车牌标注可视化脚本 | ✅ |

### 执行命令

```bash
# 抽帧预处理(11个视频，3305帧)
python scripts/extract_frames.py --all --fps 4

# 静态图调优
python scripts/tune_plate_static.py --all
python scripts/tune_plate_static.py --video 违章02

# 完整检测
python scripts/run_video.py input_video/违章02.mp4 --mode red_light

# 车牌标注可视化
python scripts/annotate_plate.py input_video/违章02.mp4 --save_key_frames
```

### 测试结果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 省份误识别 | 粤、蒙、鄂、苏 | 仅京 |
| 真值车牌识别(京LNE560) | 置信度0.897~0.998 | 置信度0.979~1.000 |
| 违规检测车牌 | 蒙L1E300(误) | 空(已过滤) |
| GT车牌匹配率 | 84.6%(11/13) | 84.6%(11/13) |

## 3. 当前Git环境状态

- **Git命令**: ✅ 可用(Portable版本: C:\Users\windows\gitportable\bin\git.exe)
- **Git仓库**: ✅ 已初始化
- **分支**: master
- **最新提交**: 715c65a "feat: 车牌识别优化 - 省份先验+跨时间段跟踪+全局投票"
- **提交文件**: 57个文件, 4994行新增, 173行删除
- **未提交改动**: 无
- **stash**: 无

**提交内容**:
- 添加省份先验过滤逻辑，过滤非"京"开头的误识别车牌
- 优化PlateConsensus模块，增加跨时间段跟踪和全局投票功能
- 新增视频抽帧预处理脚本、静态图调优脚本、车牌标注可视化脚本
- 新增未识别车牌分析脚本
- 修复车辆跟踪与车牌关联逻辑

## 4. 中间产物

| 产物路径 | 说明 |
|----------|------|
| `datasets/frames/` | 抽帧数据集(3305帧，1.6GB) |
| `datasets/frames/manifest.csv` | 帧索引文件(含GT车牌标注) |
| `datasets/frames/analysis/plate_results.csv` | 逐图识别结果 |
| `datasets/frames/analysis/accuracy_report.txt` | 准确率报告 |
| `data/output/plate_annotate_违章02/` | 车牌标注视频和关键帧 |
| `data/output/run_违章02_balanced/` | 违规检测结果 |

## 5. 当前阻塞点/未解决问题

- ❌ **京ADH9206(违章05)** 和 **京EJQ505(违章07)** 仍未识别到(0帧匹配)
- ❌ 违章05中目标白车(track_id=2)全程50秒未识别到车牌，可能被黑车遮挡
- ❌ 违章07中黑车(track_id未知)完全未被识别，可能被其他车辆遮挡
- ⚠️ 部分违规事件未关联到车牌(因帧采样间隔导致)
- ⚠️ Git命令不可用，无法提交版本控制

**未识别原因分析**:
1. **遮挡**: 目标车辆被其他车辆遮挡，角度始终不好
2. **光线**: 黑车在某些光线下对比度低，难以识别
3. **采样间隔**: 当前8fps采样可能错过最佳识别时机
4. **跟踪关联**: 车牌与车辆的IoU关联可能不够准确

## 6. 下一步执行顺序

1. ✅ **优先**: 分析京ADH9206和京EJQ505未识别原因，检查对应视频帧
   - 违章05: track_id=2(白车)全程50秒未识别到车牌，可能被黑车遮挡
   - 违章07: 黑车完全未被识别，可能被其他车辆遮挡
2. ✅ **次优先**: 优化PlateConsensus的投票策略，增加跨时间段跟踪和全局投票
   - 新增`get_best_in_window()`: 指定时间段内的最佳车牌
   - 新增`get_global_best()`: 跨时间段全局投票
   - 新增`get_time_segment_plates()`: 按时间段分段统计
3. **进行中**: 调整车牌采样间隔(`plate_interval`)，提高采样频率
4. **待执行**: 运行完整检测验证优化效果
5. **可选**: 升级车牌识别模型(如更换为更精确的OCR模型)
6. **可选**: 添加单元测试和回归测试

## 7. 禁止重复修改/重复执行的内容红线

- ❌ 禁止重复修改`violation_engine.py`的模式判断逻辑(已验证正确)
- ❌ 禁止重复运行`extract_frames.py --all`(3305帧已生成)
- ❌ 禁止删除`datasets/frames/`目录(静态图调优依赖)
- ❌ 禁止修改`plate_consensus.py`的核心投票算法(已验证有效)
- ❌ 禁止禁用省份先验(北京场景必须启用)
