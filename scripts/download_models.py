"""模型准备脚本

- 车辆检测: yolo11n.pt 由 Ultralytics 首次调用自动下载, 这里触发一次。
- 斑马线分割 / 红绿灯状态: 暂无统一公开权重, 提供两种获取方式:
    1) 直接下载: 把权重 URL 填入下方常量或环境变量 CROSSWALK_SEG_URL / TRAFFIC_LIGHT_URL
    2) 自行训练: 用 CDSet-3434(斑马线) / LISA、GTSDB(红绿灯) 训练 YOLOv11

运行: python scripts/download_models.py
"""
import os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CROSSWALK_SEG_URL = os.environ.get("CROSSWALK_SEG_URL", "")
TRAFFIC_LIGHT_URL = os.environ.get("TRAFFIC_LIGHT_URL", "")


def download_vehicle():
    print("[1/3] 下载/校验车辆检测模型 yolo11n.pt ...")
    YOLO("yolo11n.pt")  # 自动下载到 Ultralytics 默认缓存
    print("    完成 (缓存于 Ultralytics 默认目录)")


def try_download(url, name):
    if not url:
        return False
    import urllib.request
    dst = os.path.join(MODELS_DIR, name)
    print(f"    下载 {name} <- {url}")
    urllib.request.urlretrieve(url, dst)
    return True


def main():
    download_vehicle()
    print("[2/3] 斑马线分割模型 crosswalk_seg.pt ...")
    if not try_download(CROSSWALK_SEG_URL, "crosswalk_seg.pt"):
        print("    未配置 URL -> 将使用经典 CV 兜底 (无需权重)。")
        print("    如需更高精度: 用 CDSet-3434 / Roboflow crosswalk_seg 训练 YOLOv11-seg。")
    print("[3/3] 红绿灯状态模型 traffic_light.pt ...")
    if not try_download(TRAFFIC_LIGHT_URL, "traffic_light.pt"):
        print("    未配置 URL -> 将使用颜色兜底 (无需权重)。")
        print("    如需更高精度: 用 LISA / GTSDB 训练 YOLOv11 红绿灯状态分类。")
    print("模型准备完成。")


if __name__ == "__main__":
    main()
