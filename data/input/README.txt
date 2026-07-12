把待检测的视频放在本目录（实际工作目录为工程根下的 input_video/，放这里也可被读取）。

然后在工程根目录运行:

  python -m redlight.app.cli --video <本目录>/sample.mp4 --output data/output/run_sample --preset balanced

或便捷脚本:

  python scripts/run_video.py <本目录>/sample.mp4 data/output/run_sample --preset balanced

preset 可选: strict / balanced / loose / very_loose（默认 balanced）。
