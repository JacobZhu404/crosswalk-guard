import csv, collections
p='data/output/annotated/light_feedback.csv'
rows=list(csv.DictReader(open(p,encoding='utf-8-sig')))

print("=== 'other' 原因帧的备注样例(前 30) ===")
others=[r for r in rows if r['reason']=='other']
for r in others[:30]:
    note=(r['note'] or '').strip()
    print(f"  [{r['video']} {r['t_sec']}s] pred={r['pred']:7} gt={r['gt']:7} | {note[:90]}")

print("\n=== 备注关键词频率(全量) ===")
text=' '.join((r['note'] or '') for r in rows)
for kw in ['遮挡','prior','蓝框','黄圈','搜索区','反光','白边','读错','灯','位置','闪烁','flashing','红','绿','黄','未','看不清','偏','移位','整','颜色','区间','段','边界','错']:
    c=text.count(kw)
    if c: print(f"  {kw:6} {c}")

print("\n=== 各视频 x reason ===")
vids=sorted(set(r['video'] for r in rows))
reasons=['search_area','reading_point','color','gt_flipped','other','']
header='video   '+' '.join(f"{x[:8]:9}" for x in reasons)
print(header)
for v in vids:
    vr=[r for r in rows if r['video']==v]
    cc=collections.Counter(r['reason'] for r in vr)
    print(f"{v:8}"+' '.join(f"{cc.get(x,0):9}" for x in reasons))

print("\n=== 混淆矩阵(pred行 x gt列, 仅 algo/both) ===")
conf=collections.Counter()
for r in rows:
    if r['verdict'] in ('algo_wrong','both_wrong'):
        conf[(r['pred'],r['gt'])]+=1
gts=sorted(set(r['gt'] for r in rows))
print(f"{'pred_x_gt':10}"+' '.join(f"{g:9}" for g in gts))
for pr in sorted(set(r['pred'] for r in rows)):
    print(f"{pr:10}"+' '.join(f"{conf.get((pr,g),0):9}" for g in gts))

# 分类: 算法真错(需修) vs GT错(标反) vs 其他
print("\n=== 可行动分类 ===")
fixable=[r for r in rows if r['verdict'] in ('algo_wrong','both_wrong')]
print(f"  算法真错(可指导修复): {len(fixable)}")
print(f"    - search_area(蓝框没罩住真信号): {sum(1 for r in fixable if r['reason']=='search_area')}")
print(f"    - reading_point(黄圈偏):         {sum(1 for r in fixable if r['reason']=='reading_point')}")
print(f"    - other(看备注):                 {sum(1 for r in fixable if r['reason']=='other')}")
print(f"  other判定(需人工复核是否真错): {sum(1 for r in rows if r['verdict']=='other')}")
print(f"  both_wrong: {sum(1 for r in rows if r['verdict']=='both_wrong')}")
print(f"  label_wrong(GT错,算法对): {sum(1 for r in rows if r['verdict']=='label_wrong')}")
