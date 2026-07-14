import csv, collections
p='data/output/annotated/light_feedback.csv'
rows=list(csv.DictReader(open(p,encoding='utf-8-sig')))

# A. red->green 212帧 拆视频 + 备注样例
print("=== pred=red 但 gt=green 的 212 帧: 按视频 + 备注聚类 ===")
rg=[r for r in rows if r['pred']=='red' and r['gt']=='green']
print("  总数:", len(rg))
print("  视频分布:", dict(collections.Counter(r['video'] for r in rg)))
# 备注中前40字符去重计数
note_cnt=collections.Counter((r['note'] or '').strip()[:40] for r in rg)
print("  备注Top:")
for n,c in note_cnt.most_common(8): print(f"    [{c:3}] {n}")

# B. reading_point 67帧 备注
print("\n=== reading_point(黄圈偏) 67帧: 备注样例 ===")
rp=[r for r in rows if r['reason']=='reading_point']
for r in rp[:12]:
    print(f"  [{r['video']} {r['t_sec']}s] pred={r['pred']:7} gt={r['gt']:7} | {(r['note'] or '')[:80]}")

# C. search_area 38帧
print("\n=== search_area(蓝框没罩住) 38帧: 备注样例 ===")
sa=[r for r in rows if r['reason']=='search_area']
for r in sa[:8]:
    print(f"  [{r['video']} {r['t_sec']}s] pred={r['pred']:7} gt={r['gt']:7} | {(r['note'] or '')[:80]}")

# D. 各视频 混淆矩阵
print("\n=== 各视频 混淆矩阵(pred x gt, 仅 algo/both) ===")
for v in sorted(set(r['video'] for r in rows)):
    vr=[r for r in rows if r['video']==v and r['verdict'] in ('algo_wrong','both_wrong')]
    if not vr: continue
    conf=collections.Counter((r['pred'],r['gt']) for r in vr)
    gts=sorted(set(r['gt'] for r in vr))
    print(f"\n  -- {v} ({len(vr)} 帧) --")
    print("  "+' '.join(f"{g:9}" for g in gts))
    for pr in sorted(set(r['pred'] for r in vr)):
        print(f"  {pr:9}"+' '.join(f"{conf.get((pr,g),0):9}" for g in gts))

# E. "unknown"应出但未出的帧(备注含unknown)
print("\n=== 备注提示'应输出unknown'的帧 ===")
unk=[r for r in rows if 'unknown' in (r['note'] or '').lower() or '遮挡' in (r['note'] or '') or '没有红绿灯' in (r['note'] or '')]
print("  总数:", len(unk))
print("  视频分布:", dict(collections.Counter(r['video'] for r in unk)))
print("  这些帧的 pred 分布:", dict(collections.Counter(r['pred'] for r in unk)))
print("  这些帧的 verdict:", dict(collections.Counter(r['verdict'] for r in unk)))
