#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成两份程序说明书 Word 版:
  v1 = 专业准确版(工程/技术读者)
  v2 = 科普版(小学生 / 家长 / 科技竞赛老师)

用法: python scripts/make_manuals.py [--out docs/manuals]
依赖: python-docx
"""
import os
import argparse
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ---------- 共用小工具 ----------
def _set_base_font(doc, font="微软雅黑", size=11):
    style = doc.styles["Normal"]
    style.font.name = font
    style.font.size = Pt(size)
    # 中文字体需单独设置 eastAsia
    from docx.oxml.ns import qn
    style.element.rPr.rFonts.set(qn("w:eastAsia"), font)


def _title(doc, text, subtitle=None):
    h = doc.add_heading(text, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if subtitle:
        p = doc.add_paragraph(subtitle)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(10)
        p.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def _p(doc, text, bold=False, size=None, color=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    return p


def _bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25 + 0.25 * level)
    return p


def _num(doc, text):
    return doc.add_paragraph(text, style="List Number")


def _table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(10)
    return t


# ======================================================================
# v1 — 专业准确版
# ======================================================================
def build_v1(path):
    doc = Document()
    _set_base_font(doc)

    _title(doc, "斑马线行人绿灯违章占道检测系统",
           "程序说明书 v1（专业版） · 系统版本 2.3.0 · 生成于 2026-09-30")

    doc.add_heading("1. 系统概述", level=1)
    _p(doc, "本系统面向手机录制的路口视频，自动检测“斑马线行人绿灯（或闪烁清空相位）时，"
            "车辆仍静止压在斑马线上阻碍行人通行”的违章行为，识别涉事车牌，并输出标注视频、"
            "违章事件表与证据截图。")
    _p(doc, "设计约束：纯计算机视觉 + 轻量模型，不依赖大语言/视觉大模型（VLM）；面向本地 "
            "Windows / CPU 环境批量处理；机位不固定（手持拍摄）。判定为确定性规则，结果可复现、可解释。")

    _p(doc, "违章语义（唯一权威定义，E12 反转后）：", bold=True)
    _bullet(doc, "违章 = 行人绿灯/闪烁清空相位 ∧ 车辆静止 ∧ 车辆压在斑马线上，且持续足够时长。")
    _bullet(doc, "红灯 = 车辆可通行，不算违章。")
    _bullet(doc, "未拍到灯（灯态 unknown）默认不判违章；仅当斑马线疑似被遮挡时降为“待复核”，绝不自动判违章。")

    doc.add_heading("2. 技术架构", level=1)
    _p(doc, "系统采用分层管线（DAG 编排），逐帧流式采集观测，视频结束后批处理产出违章事件。"
            "当前默认配置：斑马线检测器 v2（时序聚合）、占道分母 box（车辆足迹压线比例）、采样 8fps、preset=balanced。")
    _p(doc, "2.1 逐帧处理链（DAG 节点顺序）", bold=True)
    _p(doc, "detect → track → crosswalk → light → plate →（consensus）→ accumulate → visualize")
    _table(doc,
           ["模块", "方法", "职责"],
           [
               ["车辆检测", "YOLOv8n（COCO car/bus/truck）+ IoU 跟踪", "逐帧检测车辆并分配 track_id"],
               ["斑马线检测", "CrosswalkDetectorV2（四边形探针 + 时序聚合 running-max）", "输出斑马线掩膜/区域"],
               ["红绿灯检测", "颜色法 v7-stable（HSV 亮斑 + 区域投票）+ 逐视频位置先验", "单帧灯态观测 green/red/off"],
               ["车牌识别", "HyperLPR3（HIGH）+ 按 track_id 多帧加权投票", "全局聚合每车最佳车牌"],
               ["静止判定", "TrackStateManagerV2（滑窗速度，抗抖动）", "判断车辆是否静止"],
               ["可视化", "Visualizer（与判定阈值同源）", "实时标注帧写入 annotated.mp4"],
           ])

    _p(doc, "2.2 时序融合与判定（视频结束后批处理）", bold=True)
    _p(doc, "② 时序层（temporal_fusion.fuse_light）：把逐帧灯态观测融合成灯态区间段（滑窗多数表决 + 迟滞 + "
            "转换次数约束），并为每个绿/闪烁段标注“段内最长连续 raw 绿 run 时长”。车辆静止/压线各自聚合为时间区间。")
    _p(doc, "③ 判定层（decision.decide_violations）：对 绿段 ∩ 静止区间 ∩ 压线区间（overlap ≥ preset 阈值）做区间交集，"
            "交集时长达标即成事件。#3 时序门控：绿段内最长 raw 绿 run ≥ 6.0s 判 confirmed；不足（瞬态绿，多为过路车/"
            "反光误绿）降级 review（不静默丢弃）。")
    _p(doc, "事件后处理：_dedup（跨 track 时序合并为违章窗）→ _narrow_members（b2 车组归组，收窄成员用于碎片指标）；"
            "车牌回填走原始全成员集 member_tracks_all，并施加三重约束（见 §3）。")

    doc.add_heading("3. 车牌回填的三重约束（防误罚核心）", level=1)
    _p(doc, "违章车牌由事件成员车的多帧 OCR 投票回填。为避免把“搭车/过路/别的车”的车牌错安到违章车上（误罚），"
            "回填施加三重约束：")
    _num(doc, "违章车组约束：候选归属车在事件窗内静止占比 ≥ 0.6（挡过路/移动车）。")
    _num(doc, "全局真实性约束：候选车牌全视频被读到的帧数 ≥ 5（挡孤证/幻觉牌）。")
    _num(doc, "空间聚集约束：候选车牌“编辑距离≤1 变体系”所归属车辆的质心横向跨度 ≤ 0.25×帧宽"
              "（挡跨车关联污染，如白车碎片与黑车被误并时的错牌）。")
    _p(doc, "上述约束经单元测试与变异测试守护（禁用空间约束会立刻复现跨车误罚）。")

    doc.add_heading("4. 灵敏度预设", level=1)
    _table(doc,
           ["预设", "speed(px/s)", "duration(帧)", "overlap", "说明"],
           [
               ["strict", "15", "8", "0.30", "最严，少误报"],
               ["balanced（默认）", "30", "5", "0.20", "平衡，出厂默认"],
               ["loose", "50", "3", "0.15", "更灵敏"],
               ["very_loose", "80", "2", "0.10", "最灵敏，多召回"],
           ])
    _p(doc, "阈值定义于 pipeline/tracker.py 的 SENSITIVITY_PRESETS，经 --preset 选择。")

    doc.add_heading("5. 输入与输出", level=1)
    _p(doc, "输入：input_video/ 下的 .mp4（手机录制）。", bold=True)
    _p(doc, "输出（--output 目录）：", bold=True)
    _bullet(doc, "annotated.mp4：实时标注视频（斑马线青色填充、车辆框+ID、静止标黄、违章标红、灯态 HUD、车牌框）。")
    _bullet(doc, "violations.csv：事件表，字段含 event_id/track_id/status/start_ts/end_ts/vehicle_class/"
                 "confidence/light_state/plate/plates/evidence_image。")
    _bullet(doc, "evidence/：每起事件的证据截图（文件名含 event_id/track_id/车牌）。")
    _p(doc, "事件状态：confirmed = 三条件同时成立且持久绿达标；review = 压线+静止但灯态未知且斑马线疑似遮挡，交人工复核。")

    doc.add_heading("6. 评测口径与当前指标", level=1)
    _p(doc, "评测在全 11 段真实路口视频（含 2 段负例）上进行，检测器 v2 / 占道分母 box / preset=balanced。")
    _p(doc, "6.1 窗级（每过街窗 1 票）", bold=True)
    _table(doc, ["指标", "数值"],
           [["Precision", "1.000"], ["Recall", "0.889"], ["F1", "0.941"],
            ["TP / FP / FN", "8 / 0 / 1"], ["真误报", "0"]])
    _p(doc, "6.2 车级（每违章车 1 票，具名车牌口径）", bold=True)
    _table(doc, ["指标", "数值"],
           [["具名违章车召回", "0.583（命中 7 / 漏 5）"],
            ["具名精度", "1.000"], ["误罚（预测到已知非违章车）", "0"],
            ["车牌命中（TP 事件内）", "5 / 7"]])
    _p(doc, "口径说明：车级召回分母为“具名违章车”，匿名（车牌不可读，GT 记为 ?）不计，故为召回下界。"
            "误罚指预测到 GT 已知非违章车（如某视频黑车），硬约束必须为 0。", size=10, color=(0x66, 0x66, 0x66))

    doc.add_heading("7. 已知局限", level=1)
    _bullet(doc, "个别真绿窗极短（约 1.2s）且落在视频结尾，被时序门控/融合过滤，形成漏检（结构性难例）。")
    _bullet(doc, "部分违章车牌受成像质量限制，OCR 无法读出（宁缺毋滥，留空而非误填）。")
    _bullet(doc, "同一过街窗内多辆违章车的“逐车独立开单”，需车级评测配合轨迹标注进一步支撑。")
    _bullet(doc, "红绿灯位置先验偏框的少数视频，环境绿仍可能造成灯态判别压力（弥散绿空间判别为后续方向）。")

    doc.add_heading("8. 运行方式", level=1)
    _p(doc, "单视频：", bold=True)
    _p(doc, "python -m redlight.app.cli --video input_video/违章02.mp4 --output data/output/run_02 --preset balanced")
    _p(doc, "批量评测（窗级+车级）：", bold=True)
    _p(doc, "python scripts/eval_violations.py --detector v2 --occ-denom box")
    _p(doc, "生成逐视频报告 + 标注结果视频：", bold=True)
    _p(doc, "python scripts/render_report.py")

    doc.save(path)
    return path


# ======================================================================
# v2 — 科普版(小学生 / 家长 / 竞赛老师)
# ======================================================================
def build_v2(path):
    doc = Document()
    _set_base_font(doc, font="微软雅黑", size=12)

    _title(doc, "会“抓拍”不文明车辆的智能小卫士",
           "给同学、家长和老师看的科普说明书 · v2")

    doc.add_heading("这个程序是做什么的？", level=1)
    _p(doc, "过马路的时候，行人绿灯亮了，斑马线本来应该是我们行人先走。可有的汽车会一直停在斑马线上不动，"
            "挡住我们过马路——这是很不文明、也不安全的行为。")
    _p(doc, "我们做的这个程序，就像一位不知疲倦的“交通小卫士”。给它一段用手机拍的路口视频，"
            "它就能自己看懂视频，把“行人绿灯时还赖在斑马线上不走的汽车”找出来，还能认出它的车牌号，"
            "最后给出一段画好标记的视频和一张“违章名单”。")

    doc.add_heading("它是怎么“看懂”视频的？（四个步骤）", level=1)
    _p(doc, "小卫士看视频，其实是把一件复杂的事拆成四个简单的小问题，一个一个回答：", bold=True)

    _p(doc, "第 1 步：找汽车在哪儿", bold=True)
    _bullet(doc, "它先在每一帧画面里把所有汽车圈出来，并给每辆车一个专属编号，这样就能一路盯着同一辆车。")

    _p(doc, "第 2 步：找斑马线在哪儿", bold=True)
    _bullet(doc, "它会认出画面里的斑马线（那些白色的条纹），知道“行人该走的地方”在哪里。")

    _p(doc, "第 3 步：看红绿灯是什么颜色", bold=True)
    _bullet(doc, "它盯着行人信号灯，判断现在是绿灯（该我们走）还是红灯（该车走）。")

    _p(doc, "第 4 步：认车牌号", bold=True)
    _bullet(doc, "对停着的车，它会努力看清车牌，而且是看很多帧、投票选出最靠谱的号码，避免看错。")

    _p(doc, "最后做判断：", bold=True)
    _p(doc, "只有当【行人是绿灯】而且【汽车停着不动】而且【汽车压在斑马线上】这三件事同时发生、"
            "并且持续了一小会儿，小卫士才会判定：这辆车违章了！")
    _p(doc, "如果是红灯，车本来就可以停，那不算违章；如果视频里根本没拍到红绿灯，它会很谨慎，"
            "不乱冤枉好车，而是标成“请人再看一眼”。", size=11, color=(0x66, 0x66, 0x66))

    doc.add_heading("为什么它很“聪明”也很“公正”？", level=1)
    _bullet(doc, "不冤枉好车：拿不准的时候，它宁可不判，也不乱贴罚单。在测试里，它一次都没有认错车牌去冤枉守规矩的车。")
    _bullet(doc, "会“盯人”：就算一辆车被拍得忽远忽近、编号变来变去，它也能想办法认出“还是刚才那辆车”。")
    _bullet(doc, "看很多遍再决定：车牌它要看很多帧一起投票，红绿灯也要看一小段时间，不会因为一帧看花眼就下结论。")

    doc.add_heading("它最后会给我们什么？", level=1)
    _bullet(doc, "一段“标记视频”：画面上会把汽车框出来，停着的车标黄色，正在违章的车标红色，"
                 "还会在上面写出车牌和“违章确认”。")
    _bullet(doc, "一张“违章名单”：写清楚哪辆车、在第几秒、车牌是多少。")
    _bullet(doc, "几张“证据照片”：把违章那一刻拍下来存好。")

    doc.add_heading("它现在做得怎么样？", level=1)
    _p(doc, "我们用了 11 段真实路口视频来考它（其中 2 段是“没有违章”的，专门看它会不会乱报）。结果是：")
    _bullet(doc, "该报违章的场景，它基本都抓到了；不该报的，它一个都没乱报。")
    _bullet(doc, "认车牌这件事最难（有些车拍得太模糊，人也不一定看得清），所以偶尔会有车牌读不出——"
                 "这时它选择“留空”，绝不瞎猜。")
    _p(doc, "一句话：在“判断有没有违章”这件事上它已经很稳、很少出错；还在继续变强的地方，"
            "主要是把模糊的车牌看得更清楚。")

    doc.add_heading("给老师和家长的话", level=1)
    _p(doc, "这个项目最棒的地方，是它没有用“黑箱”式的大模型去“猜”，而是把问题拆成一个个"
            "看得见、讲得清的小步骤：找车 → 找斑马线 → 看灯 → 认牌 → 按规则判断。")
    _p(doc, "每一步都可以单独检查对不对，最后的结论也能说清“为什么判它违章”。这正是科学做事的方式——"
            "把复杂问题拆开、每一步都能验证、结论可以复现。对同学来说，这也是理解“人工智能到底在做什么”"
            "的一个很好的例子：它不是魔法，而是一步步严谨的判断。")

    doc.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "docs", "manuals"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    v1 = build_v1(os.path.join(args.out, "程序说明书_v1_专业版.docx"))
    v2 = build_v2(os.path.join(args.out, "程序说明书_v2_科普版.docx"))
    print("[生成]", v1)
    print("[生成]", v2)


if __name__ == "__main__":
    main()
