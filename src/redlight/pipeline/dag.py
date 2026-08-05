"""L5 任务编排: 轻量 Pipeline DAG。

每个节点是一个接收 ctx(dict) 的可调用对象; 节点间通过拓扑排序顺序执行。
优势:
  - 可单独 mock 测试单个节点 (如给定 states 测试 evaluate)
  - 节点可替换 / 增删, 不影响主循环
  - 无第三方 DAG 依赖 (自制轻量版, 契合 YAGNI)
"""
import os
import cv2
from collections import deque


class PipelineDAG:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, func):
        self.nodes[name] = func
        self.edges.setdefault(name, [])

    def add_edge(self, a, b):
        if a in self.edges:
            self.edges[a].append(b)

    def topological_sort(self):
        indeg = {n: 0 for n in self.nodes}
        for a in self.edges:
            for b in self.edges[a]:
                indeg[b] += 1
        q = deque([n for n in self.nodes if indeg[n] == 0])
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in self.edges[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)
        if len(order) != len(self.nodes):
            raise RuntimeError("DAG 中存在环, 无法拓扑排序")
        return order

    def run(self, ctx):
        for name in self.topological_sort():
            self.nodes[name](ctx)
        return ctx


def build_default_dag(cfg, comp):
    """构建默认视频处理 DAG。

    comp 须包含: vehicle, crosswalk, light, plate, trackstate, engine, viz
    可选: plate_consensus
    节点按线性依赖连接: detect -> track -> crosswalk -> light -> plate -> consensus -> evaluate -> visualize -> output
    crosswalk/light/plate 按配置间隔节流 (复用最近一次结果)。
    """
    dag = PipelineDAG()
    det = comp["vehicle"]
    cw = comp["crosswalk"]
    tl = comp["light"]
    plate = comp["plate"]
    tracker = comp["trackstate"]
    engine = comp["engine"]
    viz = comp["viz"]
    consensus = comp.get("plate_consensus")

    cw_int = getattr(cfg.inference, "crosswalk_interval", 4)
    light_int = getattr(cfg.inference, "light_interval", 8)
    plate_int = getattr(cfg.inference, "plate_interval", 3)

    def n_detect(ctx):
        ctx["dets"] = det.detect(ctx["frame"])

    def n_track(ctx):
        ctx["states"] = tracker.update(ctx["dets"], ctx["ts"])

    def n_crosswalk(ctx):
        if ctx["proc"] % cw_int == 0:
            # 传入车辆框以启用 v11 车辆锚定加分(斑马线通常在静止车附近, 修复 E17 泛化);
            # detect 在 track 之后运行, ctx["dets"] 已就绪。
            vb = [d["xyxy"] for d in ctx.get("dets", [])]
            ctx["mask"] = cw.detect(ctx["frame"], vb)

    def n_light(ctx):
        if ctx["proc"] % light_int == 0:
            # M1: 把 vehicle 检测器同一次推理暴露的 traffic-light 框传给灯检测器(候选并集)
            boxes = getattr(det, "last_light_boxes", None)
            res = tl.detect(ctx["frame"], yolo_light_boxes=boxes)
            ctx["light"] = res
            ctx["light_state"] = res.get("state", "unknown") if isinstance(res, dict) else res
            # ②③ 接线: 同时收集单帧观测供批处理决策
            ctx["light_observation"] = tl.observe(ctx["frame"], yolo_light_boxes=boxes)

    def n_plate(ctx):
        if ctx["proc"] % plate_int == 0:
            vb = [d["xyxy"] for d in ctx["dets"]]
            ctx["plates"] = plate.detect(ctx["frame"], vb)

    def n_consensus(ctx):
        if consensus and ctx.get("plates"):
            states = ctx.get("states", {})
            for p in ctx["plates"]:
                if not p.get("text"):
                    continue
                best_tid = None
                best_iou = 0.0
                px1, py1, px2, py2 = p.get("xyxy", [0, 0, 0, 0])
                p_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                p_area = (px2 - px1) * (py2 - py1)
                if p_area <= 0:
                    continue
                for tid, st in states.items():
                    if st.get("active") and st.get("box"):
                        bx1, by1, bx2, by2 = st["box"]
                        ix1 = max(px1, bx1)
                        iy1 = max(py1, by1)
                        ix2 = min(px2, bx2)
                        iy2 = min(py2, by2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        b_area = (bx2 - bx1) * (by2 - by1)
                        iou = inter / min(p_area, b_area) if min(p_area, b_area) > 0 else 0.0
                        if iou > best_iou and iou >= 0.2:
                            best_iou = iou
                            best_tid = tid
                if best_tid is not None:
                    # box 随读取记录(纯加性, 供回填时间/空间约束; 投票逻辑不变)
                    consensus.update(best_tid, p["text"], p.get("conf", 0.0), ctx["ts"],
                                     p.get("xyxy"))
                else:
                    for tid, st in states.items():
                        if st.get("active") and st.get("box"):
                            bx1, by1, bx2, by2 = st["box"]
                            if bx1 <= p_center[0] <= bx2 and by1 <= p_center[1] <= by2:
                                consensus.update(tid, p["text"], p.get("conf", 0.0), ctx["ts"],
                                                 p.get("xyxy"))
                                break
            ctx["consensus_plates"] = consensus.get_all()

    def n_accumulate(ctx):
        # ②③ 接线: 逐帧收集原始观测与跟踪状态, 供视频结束后批处理决策
        engine.accumulate(
            ctx["states"], ctx.get("mask"),
            ctx.get("light_observation", {}), ctx["ts"]
        )

    def n_visualize(ctx):
        # light_boxes: YOLO 信号灯框(供可视化画红框+识别结果)
        lb = getattr(det, "last_light_boxes", None) or []
        ctx["disp"] = viz.draw(ctx["frame"], ctx["dets"], ctx["states"],
                               ctx["mask"], ctx["light_state"], ctx["plates"],
                               light_boxes=lb)

    dag.add_node("detect", n_detect)
    dag.add_node("track", n_track)
    dag.add_node("crosswalk", n_crosswalk)
    dag.add_node("light", n_light)
    dag.add_node("plate", n_plate)
    dag.add_node("accumulate", n_accumulate)
    dag.add_node("visualize", n_visualize)
    if consensus:
        dag.add_node("consensus", n_consensus)

    dag.add_edge("detect", "track")
    dag.add_edge("track", "crosswalk")
    dag.add_edge("crosswalk", "light")
    dag.add_edge("light", "plate")
    if consensus:
        dag.add_edge("plate", "consensus")
        dag.add_edge("consensus", "accumulate")
    else:
        dag.add_edge("plate", "accumulate")
    dag.add_edge("accumulate", "visualize")
    return dag
