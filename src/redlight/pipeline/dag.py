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
            ctx["mask"] = cw.detect(ctx["frame"])

    def n_light(ctx):
        if ctx["proc"] % light_int == 0:
            res = tl.detect(ctx["frame"])
            ctx["light"] = res
            ctx["light_state"] = res.get("state", "unknown") if isinstance(res, dict) else res

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
                        if iou > best_iou and iou >= 0.3:
                            best_iou = iou
                            best_tid = tid
                if best_tid is not None:
                    consensus.update(best_tid, p["text"], p.get("conf", 0.0), ctx["ts"])
            ctx["consensus_plates"] = consensus.get_all()

    def n_evaluate(ctx):
        ctx["new_events"] = engine.evaluate(ctx["states"], ctx["mask"], ctx["light_state"], ctx["ts"])

    def n_visualize(ctx):
        ctx["disp"] = viz.draw(ctx["frame"], ctx["dets"], ctx["states"],
                               ctx["mask"], ctx["light_state"], ctx["plates"])

    def n_output(ctx):
        for ev in ctx["new_events"]:
            plate_text = ""
            consensus_plates = ctx.get("consensus_plates", {})
            if ev["track_id"] in consensus_plates:
                plate_text = consensus_plates[ev["track_id"]]["text"]
            if not plate_text:
                for p in ctx.get("plates", []):
                    if p.get("text"):
                        plate_text = p["text"]
                        break
            if ctx["cfg"].output.evidence_images:
                fname = f"ev{ev['event_id']:04d}_tid{ev['track_id']}"
                if plate_text:
                    fname += f"_{plate_text}"
                fname += ".jpg"
                fpath = os.path.join(ctx["evidence_dir"], fname)
                cv2.imwrite(fpath, ctx["frame"])
                ev["evidence_image"] = fpath
            ctx["csv_rows"].append({
                "event_id": ev["event_id"], "track_id": ev["track_id"],
                "status": ev["status"], "start_ts": ev["start_ts"],
                "end_ts": ev["end_ts"], "vehicle_class": ev["vehicle_class"],
                "confidence": ev["confidence"], "light_state": ev["light_state"],
                "signal_assumption": ctx["cfg"].output.signal_assumption,
                "plate": plate_text, "evidence_image": ev.get("evidence_image", ""),
            })
            ev["plate"] = plate_text

    dag.add_node("detect", n_detect)
    dag.add_node("track", n_track)
    dag.add_node("crosswalk", n_crosswalk)
    dag.add_node("light", n_light)
    dag.add_node("plate", n_plate)
    dag.add_node("evaluate", n_evaluate)
    dag.add_node("visualize", n_visualize)
    dag.add_node("output", n_output)
    if consensus:
        dag.add_node("consensus", n_consensus)

    dag.add_edge("detect", "track")
    dag.add_edge("track", "crosswalk")
    dag.add_edge("crosswalk", "light")
    dag.add_edge("light", "plate")
    if consensus:
        dag.add_edge("plate", "consensus")
        dag.add_edge("consensus", "evaluate")
    else:
        dag.add_edge("plate", "evaluate")
    dag.add_edge("evaluate", "visualize")
    dag.add_edge("visualize", "output")
    return dag
