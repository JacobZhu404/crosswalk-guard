import sys, cv2
src = sys.argv[1] if len(sys.argv) > 1 else r"D:\redlight-crosswalk-violation\data\output\run10b\annotated.mp4"
outdir = sys.argv[2] if len(sys.argv) > 2 else r"D:\redlight-crosswalk-violation\data\output\run10b\snapshots"
import os; os.makedirs(outdir, exist_ok=True)
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
# 抽取指定秒数附近的帧
targets = [float(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [1.3, 8.1, 12.0]
for t in targets:
    fi = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if ok:
        p = os.path.join(outdir, f"t{t:.1f}s.jpg")
        cv2.imwrite(p, frame)
        print("saved", p)
    else:
        print("miss", t)
cap.release()
