"""分析未识别到的车牌"""
import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    plate = PlateRecognizer(cfg, verbose=False)

    missed_cases = [
        {"video": "违章05", "plate": "京ADH9206", "note": "白车占道违章"},
        {"video": "违章07", "plate": "京EJQ505", "note": "黑车占道"},
    ]

    for case in missed_cases:
        video_path = os.path.join(ROOT, "input_video", f"{case['video']}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        interval = max(1, int(round(fps / 8)))

        print(f"\n{'='*60}")
        print(f"分析: {case['video']} - 未识别车牌: {case['plate']}")
        print(f"说明: {case['note']}")
        print(f"{'='*60}")

        results = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                ts = frame_idx / fps
                plates = plate.detect(frame)
                for p in plates:
                    txt = p.get("text", "")
                    conf = p.get("conf", 0.0)
                    if conf > 0.5:
                        x1, y1, x2, y2 = p.get("xyxy", [0, 0, 0, 0])
                        w = int(x2 - x1)
                        results.append({
                            "time": round(ts, 2), "plate": txt, "conf": round(conf, 4),
                            "w": w, "pos": (int(x1), int(y1))
                        })
            frame_idx += 1
        cap.release()

        if not results:
            print("  未识别到任何车牌")
            continue

        print(f"  识别到的车牌:")
        for r in results[:30]:
            print(f"    @{r['time']:.2f}s {r['plate']:12s} conf={r['conf']:.3f} w={r['w']} pos={r['pos']}")

        print(f"\n  共识别到 {len(results)} 个车牌结果")


if __name__ == "__main__":
    main()
