"""L2 数据管道: 抽帧策略 (纯函数, 易测)。

将原始视频按目标帧率降采样, 降低 CPU 推理开销。
默认 fps=8 在 CPU 上足以抓静止车辆。
"""


def sampling_interval(src_fps, target_fps):
    """返回每多少原始帧抽一帧的间隔 (>=1)。"""
    if target_fps is None or target_fps <= 0:
        return 1
    if src_fps is None or src_fps <= 0:
        return 1
    return max(1, int(round(src_fps / target_fps)))


def should_sample(frame_idx, interval):
    """frame_idx 是否落在采样点上。interval=1 表示全采。"""
    if interval <= 1:
        return True
    return frame_idx % interval == 0


def sample_indices(total_frames, interval):
    """生成所有采样帧的序号列表。"""
    if interval <= 1:
        return list(range(total_frames))
    return list(range(0, total_frames, interval))


def sample_at_times(total_frames, src_fps, target_times):
    """按指定时间点(秒)返回最近帧序号 (用于关键帧抽取/诊断)。"""
    out = []
    for t in target_times:
        idx = int(round(t * src_fps))
        if 0 <= idx < total_frames:
            out.append(idx)
    return out
