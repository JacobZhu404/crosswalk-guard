"""生成全11视频行人灯态 GT -> datasets/gt/light_states.csv。

策略: 02/03/04 保留旧 GT(用户肉眼确认, 更准; label_result_01.csv 描述不完整会漏段),
其余 01/05/06/07/08/09/10/11 从 label_result_01.csv 解析(identify parse_desc + 手动补)。

label_result_01.csv (GBK) 解析漏的 4 视频手动补:
  07: "00:00-00:43 绿灯, 44秒以后红灯" (中文冒号致正则失败)
  09: "前7s红灯, 11s到1分12s绿灯" (格式特殊)
  10: "全程红色" (措辞"红色"非"红灯")
  11: "15s之前红色, 15s之后绿灯"
"""
import csv
import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

spec = importlib.util.spec_from_file_location("ids", os.path.join(ROOT, "scripts", "identify_pedestrian_signal.py"))
ids = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ids)

# 02/03/04 保留旧GT(肉眼确认, label描述不完整会漏85s+红灯等段)
OLD_GT_KEEP = {"违章02", "违章03", "违章04"}
# 02/03/04 旧GT(从git历史9b8f16c~1恢复, 用户肉眼确认)
OLD_GT = {
    "违章02": [(0, 20.9, "red", "confirmed"), (20.9, 85, "green", "confirmed"), (85, 999, "red", "confirmed")],
    "违章03": [(0, 73, "red", "confirmed"), (74, 134, "green", "confirmed"), (135, 999, "red", "confirmed")],
    "违章04": [(0, 41.7, "red", "confirmed"), (42.4, 999, "green", "confirmed")],
}
# 手动补 parse_desc 漏的 4 视频
MANUAL = {
    "违章07": [(0, 43, "green", "confirmed"), (44, 999, "red", "confirmed")],
    "违章09": [(0, 7, "red", "confirmed"), (7, 11, "unknown", "tentative"), (11, 72, "green", "confirmed"), (72, 999, "unknown", "tentative")],
    "违章10": [(0, 999, "red", "confirmed")],
    "违章11": [(0, 15, "red", "confirmed"), (15, 999, "green", "confirmed")],
}


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "input_video", "label_result_01.csv"), encoding="gbk", errors="ignore")))
    out = []
    for r in rows:
        v = r.get("input_video", "").strip()
        if v in OLD_GT_KEEP:
            segs = OLD_GT[v]
        elif v in MANUAL:
            segs = MANUAL[v]
        else:
            s = ids.parse_desc(r.get("gt_description", ""))
            segs = [(a, b if b < 99999 else 999, c, "confirmed") for a, b, c in s]
        for a, b, st, conf in segs:
            out.append({"video": v, "start_s": a, "end_s": b,
                        "state": st, "confidence": conf,
                        "note": "旧GT肉眼确认" if v in OLD_GT_KEEP else "从label_result_01.csv解析"})

    out_path = os.path.join(ROOT, "datasets", "gt", "light_states.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "start_s", "end_s", "state", "confidence", "note"])
        w.writeheader()
        w.writerows(out)

    print(f"已生成 {out_path} (全{len(set(r['video'] for r in out))}视频, {len(out)}段)")
    for v in sorted(set(r["video"] for r in out)):
        segs = [r for r in out if r["video"] == v]
        s = " ".join(f"{x['state']}[{x['start_s']}-{x['end_s'] if int(x['end_s']) < 999 else 'end'}]" for x in segs)
        print(f"  {v}: {s}")


if __name__ == "__main__":
    main()

