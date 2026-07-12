"""校验 datasets/gt/events.csv + videos.csv 的格式与一致性。

检查项:
  1. 表头必须是规定 9 列
  2. 时间: start_s < end_s, 可解析为 float
  3. light_state ∈ {green,red,flashing,unknown}
  4. light_evidence: unknown 段必须为空; 非 unknown 段必须为 visible|inferred|occluded
  5. is_violation ∈ {0,1}
  6. violating_plates: is_violation=1 时必非空; 所有 plate token 合法
  7. plate token: 车牌 / 无牌 / ?(看不清); 其余报 WARNING
  8. 同一视频内时间区间不重叠
  9. (若有 videos.csv) has_violation 与事件行一致: 存在 is_violation=1 -> has_violation=1

用法:
    python scripts/validate_gt.py
    python scripts/validate_gt.py --events datasets/gt/events.csv --videos datasets/gt/videos.csv
"""
import sys, os, csv, argparse, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_COLS = ["video", "start_s", "end_s", "light_state",
                 "light_evidence", "is_violation",
                 "violating_plates", "other_plates", "note"]
STATES = {"green", "red", "flashing", "unknown"}
EVID = {"visible", "inferred", "occluded"}
PLATE_RE = re.compile(r"^[A-Za-z0-9一-鿿]+$")  # 粗略: 字母数字汉字
SPECIAL = {"无牌", "?"}

errors, warnings = [], []


def err(msg): errors.append(msg)
def warn(msg): warnings.append(msg)


def parse_plates(s):
    """分号分隔 -> token 列表(去空去空格)。"""
    return [t.strip() for t in (s or "").split(";") if t.strip()]


def check_plate_token(tok, where):
    if tok in SPECIAL:
        return
    if PLATE_RE.match(tok) and len(tok) >= 4:
        return
    warn(f"{where}: 车牌 token '{tok}' 非常规(非车牌/非 无牌/非 ?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--videos", default=os.path.join(ROOT, "datasets", "gt", "videos.csv"))
    args = ap.parse_args()

    if not os.path.exists(args.events):
        err(f"找不到 events.csv: {args.events}")
        _finish()
        return

    with open(args.events, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows:
        err("events.csv 为空")
    if fieldnames != EXPECTED_COLS:
        err(f"表头不符. 期望 {EXPECTED_COLS}; 实际 {fieldnames}")

    by_video = {}
    for i, r in enumerate(rows):
        ln = f"events.csv 行{i+2}({r.get('video','?')} {r.get('start_s','?')}-{r.get('end_s','?')})"
        # 时间
        try:
            a = float(r["start_s"]); b = float(r["end_s"])
            if not (a < b):
                err(f"{ln}: start_s>=end_s")
        except Exception as e:
            err(f"{ln}: 时间解析失败 {e}"); a = b = None
        # 状态
        st = (r.get("light_state") or "").strip()
        if st not in STATES:
            err(f"{ln}: light_state='{st}' 非法")
        # evidence
        ev = (r.get("light_evidence") or "").strip()
        if st == "unknown":
            if ev != "":
                warn(f"{ln}: light_state=unknown 但 light_evidence='{ev}' (应留空)")
        else:
            if ev not in EVID:
                err(f"{ln}: 非 unknown 段 light_evidence='{ev}' 非法(应 visible/inferred/occluded)")
        # is_violation
        iv = (r.get("is_violation") or "").strip()
        if iv not in ("0", "1"):
            err(f"{ln}: is_violation='{iv}' 非法")
        # plates
        for col, role in (("violating_plates", "violating"), ("other_plates", "other")):
            for tok in parse_plates(r.get(col, "")):
                check_plate_token(tok, f"{ln}.{col}")
        if iv == "1" and not parse_plates(r.get("violating_plates", "")):
            err(f"{ln}: is_violation=1 但 violating_plates 为空")
        # 重叠
        if a is not None and b is not None:
            by_video.setdefault(r["video"], []).append((a, b, i))

    for v, segs in by_video.items():
        segs.sort()
        for k in range(1, len(segs)):
            if segs[k][0] < segs[k-1][1] - 1e-6:
                err(f"视频 {v}: 行{segs[k-1][2]+2} 与 行{segs[k][2]+2} 时间区间重叠 "
                    f"({segs[k-1][0]}-{segs[k-1][1]} vs {segs[k][0]}-{segs[k][1]})")

    # videos.csv 一致性
    if os.path.exists(args.videos):
        vrows = list(csv.DictReader(open(args.videos, encoding="utf-8")))
        vmap = {r["video"]: r for r in vrows}
        for v, segs in by_video.items():
            has_any = any(rows[i]["is_violation"] == "1" for _, _, i in segs)
            vr = vmap.get(v)
            if vr is None:
                warn(f"videos.csv 缺少视频 {v}")
                continue
            hv = (vr.get("has_violation") or "").strip()
            if has_any and hv != "1":
                err(f"视频 {v}: 存在 is_violation=1 事件, 但 videos.csv has_violation={hv}")
            if not has_any and hv == "1":
                warn(f"视频 {v}: videos.csv 标 has_violation=1, 但事件表无 is_violation=1")
    else:
        warn(f"找不到 videos.csv: {args.videos} (跳过一致性检查)")

    _finish()


def _finish():
    print(f"\n=== GT 校验结果 ===")
    print(f"ERROR:   {len(errors)}")
    print(f"WARNING: {len(warnings)}")
    for e in errors:
        print(f"  [E] {e}")
    for w in warnings:
        print(f"  [W] {w}")
    if errors:
        print("\n存在 ERROR, 请修正后再跑评测。")
        sys.exit(1)
    else:
        print("\n✅ GT 格式与一致性检查通过。")


if __name__ == "__main__":
    main()
