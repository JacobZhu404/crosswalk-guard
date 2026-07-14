"""L3 红绿灯状态检测 v7 (信号头聚类 + 面积选灯 + 持久门控, 修复 E16/E20, 重写选灯逻辑):

设计原则 (不过拟合单视频, 见 02/07/11 对比诊断):
  1. **饱和度主导亮斑**: lit = S>=sat_min & V>=value_floor. 信号灯是高饱和纯色发光体,
     不依赖整体曝光(比 v5 的 V>=200 / v6 早期自适应V 都鲁棒).
  2. **宽颜色分类**: 绿[h∈40..100, s>=22]; 红[h<=35 或 h>=150, s>=22]; 橙/黄归红侧; 灰白/青忽略.
  3. **信号头聚类 (核心, v7)**: 同屏位置邻近(归一距离 D~0.15)的红绿亮斑聚成一个"信号头"
     (行人信号 = 绿灯泡在上 + 红灯泡在下, 竖向间距~0.1). 一个信号头每帧只亮一盏灯.
  4. **面积选灯 (E20 修正)**: 用**灯炮物理面积**衡量每盏灯, 而非"高亮像素占比 frac_v".
     实测 07 的信号灯虽可见但相机未把像素顶到 V=255(frac_v 仅 0.02-0.15), 故 frac_v 门控会
     误杀真绿灯; 而亮绿灯面积(220-740)远大于同杆暗红反射(~300) -> 面积选灯绿灯胜出(匹配用户"07是绿灯").
     闪烁清空相位: 红绿逐帧交替亮 -> 每帧 obs 在红/绿间跳 -> 全局检测为 flashing.
  5. **持久门控 (Q3)**: 只有"持久(固定位置)信号头"才参与判定, 移动红车/瞬态反光聚不成持久头 -> 拒绝.
  6. **软位置先验** signal_cy_cutoff(默认0.6): cy>=此值的候选不参与判定.
信号类型假设: 斑马线行人信号灯 (见设计文档 §5.3.1 / Q5).
"""
import collections
import os
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo


class TrafficLightDetector(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        tl = getattr(cfg, "traffic_light", None)
        # 时序窗口(近窗颜色统计长度, 单位=detect调用次数)
        self.window = int(getattr(tl, "smoothing_window", 24))
        # 迟滞阈值(0-1): 已建立的状态需被反色占比超过此值才翻转 -> 抑制抖动/瞬态误读,
        # 呼应"红绿灯状态连续多秒". 不翻转时仍返回真实占比作为 confidence, 供融合层判断低置信.
        self.hysteresis = float(getattr(tl, "hysteresis", 0.68))
        self._last_state = None  # 已提交状态(迟滞用)
        # 1) 自适应亮斑参数
        self.value_floor = int(getattr(tl, "value_floor", 60))   # 亮度下限(避免接住近黑)
        self.sat_min = int(getattr(tl, "sat_min", 35))            # 亮斑饱和度下限(排除灰白反光)
        # 2) 候选形态过滤
        self.min_area_px = int(getattr(tl, "min_area_px", 20))
        self.max_area_ratio = float(getattr(tl, "max_area_ratio", 0.008))
        self.max_aspect = float(getattr(tl, "max_aspect_ratio", 3.5))
        # 3) 颜色分类
        self.color_s_min = int(getattr(tl, "color_s_min", 22))
        # 4) 信号轨迹/头跟踪
        self.match_radius_ratio = float(getattr(tl, "match_radius_ratio", 0.06))
        self.head_dist = float(getattr(tl, "head_dist", 0.15))    # 信号头聚类半径(归一)
        self.min_persist_frames = int(getattr(tl, "min_persist_frames", 5))
        self.track_persist_min = float(getattr(tl, "track_persist_min", 0.08))
        # 信号灯上部区域先验(软): cy>=此值(路面/车灯区)的候选不参与信号判定
        self.signal_cy_cutoff = float(getattr(tl, "signal_cy_cutoff", 0.6))
        # 信号头面积门限: 该头"总面积"需 >= 此值才算有效信号头(排除极小噪点)
        self.head_area_floor = float(getattr(tl, "head_area_floor", 80.0))
        # 信号灯形状分: max_spot面积 / 候选数; 真灯泡=few-large(高), 倒计时数字=many-small(低)
        self.lamp_score_min = float(getattr(tl, "lamp_score_min", 35.0))
        # 空间锚定(稳定器): 信号灯静止不动 -> 锁定后 ROI 不再漂移(防衣服/远树/倒计时面板劫持)
        self.anchor_radius = float(getattr(tl, "anchor_radius", 0.13))
        self.reanchor_need = int(getattr(tl, "reanchor_need", 150))
        self.anchor_hold = int(getattr(tl, "anchor_hold", 30))  # 锚丢失后保持旧色的最大帧数(超则unknown)
        # 行人信号位置先验(标定用, 非人工猜测): 给定后仅在先验附近选灯, 且永不重锚到远处
        #   (修复 02/03/04 多信号灯横跳: 检测器在多个灯间跳, 没锁住行人信号)
        #   先验由 scripts/identify_pedestrian_signal.py 用 GT 行人灯态匹配得出
        sp = getattr(tl, "signal_prior", None)
        self.signal_prior = tuple(sp) if isinstance(sp, (list, tuple)) and len(sp) == 2 else None
        self.prior_search_radius = float(getattr(tl, "prior_search_radius", 0.13))
        self.prior_hold = int(getattr(tl, "prior_hold", 90))  # 先验模式下暗灯间歇丢失时保持旧色的最大帧数(~3s)
        self.prior_roi_px = int(getattr(tl, "prior_roi_px", 160))  # HSV直采ROI边长(px), 吸收手持漂移; 小灯视频可加大
        # 闪烁判定
        self.flicker_toggle = int(getattr(tl, "flicker_toggle_count", 4))
        # 状态
        self._fi = 0
        self.tracks = []   # 单灯轨迹(可视化/COT 定位用)
        self.heads = []    # 信号头轨迹(选灯判定: 同杆红绿聚成一个头)
        self.anchor = None  # 空间锚(稳定器): 锁定静态信号灯位置 {cx,cy,last_dom,last_seen,...}
        self.global_recent = collections.deque(maxlen=self.window)  # 全局灯态时间线
        self._last_sample = (0, 0)  # HSV直采最近一次(g_px, r_px), 供调参观察
        self._loaded = True
        self._vb = verbose
        # M1 重设计(spec M1-D2/D4/D6): ped_classifier 路径。缺权重时 available=False -> 回退 color。
        self.method = str(getattr(tl, "method", "color"))
        models = getattr(cfg, "models", None)
        clf_path = getattr(models, "ped_signal_model", None) if models is not None else None
        from .signal_state_classifier import SignalStateClassifier
        self.classifier = SignalStateClassifier(clf_path, verbose=verbose)

    def load(self, weights_path=None):
        self._loaded = True

    def set_video_prior(self, video_name, priors_path=None):
        """按视频名从 light_priors.json 加载 per-video 行人信号位置先验。

        light_priors.json: {video: [cx, cy, roi_px]} (cx/cy 归一化, roi_px HSV直采边长)。
        命中则设 signal_prior + prior_roi_px (observe()/detect() 的 prior 直采路径生效);
        未命中则清空 signal_prior (回退全局亮斑路径)。无该视频先验返回 False。
        生产路径 (cli/run_video) 跑每个视频前调用, 让 per-video prior 自动接入 observe。
        """
        import json
        if priors_path is None:
            from ..infrastructure.config import project_root
            priors_path = os.path.join(project_root(), "configs", "light_priors.json")
        try:
            with open(priors_path, encoding="utf-8") as f:
                priors = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.signal_prior = None
            return False
        p = priors.get(video_name)
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            self.signal_prior = None
            return False
        self.signal_prior = (float(p[0]), float(p[1]))
        self.prior_roi_px = int(p[2]) if len(p) > 2 else self.prior_roi_px
        return True

    def observe(self, frame):
        """单帧灯态观测(供 TemporalFusion 消费): 只出这帧看到什么, 不做跨帧时序。

        obs ∈ 'green'|'red'|'off'|None。用单帧候选主色, 不 append global_recent、
        不跑 _state_from_global(那是②的活)。无内部状态改变 -> 同帧多次调用一致。

        若配置了 signal_prior, 使用先验 ROI 直采(_sample_prior_color)排除全局环境干扰;
        直采失败则回退到先验搜索半径内 candidates 面积加总。
        """
        self._last_frame = frame
        spots = self._candidates(frame)

        # 先验模式: 直采 ROI 颜色, 排除全局环境绿/树叶干扰
        if self.signal_prior is not None and frame is not None:
            sampled = self._sample_prior_color()
            if sampled is not None:
                g_n, r_n = self._last_sample
                total_colored = g_n + r_n
                conf = round(max(g_n, r_n) / total_colored, 3) if total_colored > 0 else 0.0
                return {"obs": sampled, "conf": conf, "candidates": spots}
            # 直采失败: 回退到先验半径内 candidates 面积加总
            px, py = self.signal_prior
            r = self.prior_search_radius
            near = [s for s in spots
                    if ((s["cx"] - px) ** 2 + (s["cy"] - py) ** 2) ** 0.5 <= r
                    and s["color"] in ("green", "red")]
            if near:
                greens = sum(s["area"] for s in near if s["color"] == "green")
                reds = sum(s["area"] for s in near if s["color"] == "red")
                if greens == 0 and reds == 0:
                    obs, conf = "off", 0.0
                elif greens >= reds:
                    obs, conf = "green", round(greens / (greens + reds), 3)
                else:
                    obs, conf = "red", round(reds / (greens + reds), 3)
                return {"obs": obs, "conf": conf, "candidates": spots}

        # 无先验: 保持原行为(全局亮斑面积加总)
        greens = sum(s["area"] for s in spots if s.get("color") == "green")
        reds = sum(s["area"] for s in spots if s.get("color") == "red")
        if greens == 0 and reds == 0:
            obs, conf = "off", 0.0
        elif greens >= reds:
            obs, conf = "green", round(greens / (greens + reds), 3)
        else:
            obs, conf = "red", round(reds / (greens + reds), 3)
        return {"obs": obs, "conf": conf, "candidates": spots}

    def get_info(self):
        return ModelInfo(name="TrafficLightDetector", version="color-v7-stable",
                         classes=["red", "green", "flashing", "unknown"], input_size=(0, 0))

    def infer(self, frame, crosswalk_mask=None):
        return self.detect(frame, crosswalk_mask)

    # ---------- 主入口 ----------
    def detect(self, frame, crosswalk_mask=None, yolo_light_boxes=None):
        self._fi += 1
        if frame is not None:
            self._last_w = frame.shape[1]
        # M1 ped_classifier 路径(分类器可用时); 否则回退到下方 color 路径
        if self.method == "ped_classifier" and getattr(self.classifier, "available", False):
            return self._detect_ped(frame, yolo_light_boxes or [])
        spots = self._candidates(frame)
        self._update_tracks(spots)       # 单灯轨迹(可视化/COT 定位)
        self._update_heads(spots)        # 信号头轨迹(选灯判定)
        self._last_frame = frame          # 供 prior 模式直采降级用
        obs, sel_head = self._select_lit()
        self.global_recent.append(obs)
        state, reason, conf = self._state_from_global()
        best = self._best_signal_head(sel_head)
        return {
            "state": state,
            "confidence": conf,
            "stable": best is not None,
            "is_flashing": state == "flashing",
            "reason": reason,
            "candidates": spots,          # 当前帧亮斑(供 COT/截图)
            "track": best,
            "anchor": self.anchor,       # 空间锚(稳定器): 锁定静态信号灯位置
            "g_px": self._last_sample[0] if hasattr(self, "_last_sample") else 0,
            "r_px": self._last_sample[1] if hasattr(self, "_last_sample") else 0,
        }

    # ---------- M1 ped_classifier 路径: 候选并集 + 状态分类 + 现有时序平滑 ----------
    def _detect_ped(self, frame, yolo_light_boxes):
        """候选=YOLO灯框∪HSV亮斑 -> 分类器判 walk/stand/off -> 复用 global_recent 平滑。"""
        from .signal_candidates import build_candidates
        h, w = (frame.shape[0], frame.shape[1]) if frame is not None else (1, 1)
        hsv_boxes = [s["box"] for s in self._candidates(frame)]   # 复用 HSV 亮斑框
        cands = build_candidates(yolo_light_boxes, hsv_boxes, w, h)
        obs, best_conf = None, 0.0
        for c in cands:
            x1, y1, x2, y2 = c["box"]
            roi = frame[max(0, y1):y2, max(0, x1):x2] if frame is not None else None
            label, conf = self.classifier.classify(roi)
            if label == "off":
                continue
            if conf > best_conf:
                best_conf = conf
                obs = "green" if label == "walk" else "red"
        self.global_recent.append(obs)
        state, reason, conf = self._state_from_global()
        return {"state": state, "confidence": conf, "stable": obs is not None,
                "is_flashing": state == "flashing", "reason": "ped_" + reason,
                "candidates": cands, "track": None, "anchor": None, "g_px": 0, "r_px": 0}

    # ---------- 选灯判定 (v7: 信号头聚类 + 面积选灯 + 持久门控) ----------
    def _cluster_heads(self, spots):
        """把候选按空间邻近聚成信号头; 返回每头的 {cx,cy, dom, total, cur_g, cur_r}。"""
        D = self.head_dist
        heads = []
        # 先放大块以便稳定聚类中心
        for s in sorted(spots, key=lambda x: -x["area"]):
            placed = False
            for h in heads:
                d = ((s["cx"] - h["cx"]) ** 2 + (s["cy"] - h["cy"]) ** 2) ** 0.5
                if d < D:
                    h["spots"].append(s)
                    placed = True
                    break
            if not placed:
                heads.append({"cx": s["cx"], "cy": s["cy"], "spots": [s]})
        for h in heads:
            cg = cr = 0.0
            for s in h["spots"]:
                if s["color"] == "green":
                    cg += s["area"]
                elif s["color"] == "red":
                    cr += s["area"]
            h["cur_g"] = cg
            h["cur_r"] = cr
            h["total"] = cg + cr
            h["dom"] = "green" if cg >= cr else "red"
            nsp = len(h["spots"])
            h["n_spots"] = nsp
            h["max_spot"] = float(max((s["area"] for s in h["spots"]), default=0.0))
            # 形状分: 真灯泡=few-large(高); 倒计时数字=many-small(低) -> 选灯/建锚时偏好高者
            h["lamp_score"] = h["max_spot"] / max(1, nsp)
            # 稳定中心 = 头内亮斑均值
            h["cx"] = float(np.mean([s["cx"] for s in h["spots"]]))
            h["cy"] = float(np.mean([s["cy"] for s in h["spots"]]))
        return heads

    def _update_heads(self, spots):
        D = self.head_dist
        cur = self._cluster_heads(
            [s for s in spots if s["cy"] < self.signal_cy_cutoff
             and s["color"] in ("green", "red")])
        # 先把所有持久头的"本帧发射"清零(本帧未出现的头 cur=0)
        for h in self.heads:
            h["cur_g"] = 0.0
            h["cur_r"] = 0.0
            h["total"] = 0.0
        for ch in cur:
            best, bd = None, D
            for i, ht in enumerate(self.heads):
                d = ((ch["cx"] - ht["cx"]) ** 2 + (ch["cy"] - ht["cy"]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, i
            if best is not None:
                ht = self.heads[best]
                ht["cx"] = 0.7 * ht["cx"] + 0.3 * ch["cx"]
                ht["cy"] = 0.7 * ht["cy"] + 0.3 * ch["cy"]
                ht["frames_seen"] += 1
                ht["last_seen"] = self._fi
                ht["cur_g"] = ch["cur_g"]
                ht["cur_r"] = ch["cur_r"]
                ht["total"] = ch["total"]
                ht["last_dom"] = ch["dom"]
                ht["max_total"] = max(ht.get("max_total", 0.0), ch["total"])
                # 每帧刷新灯形分(勿冻结): 否则建锚时 lamp_score 永远停在创建帧的偏低值 -> 锚点永不锁定
                ht["lamp_score"] = ch["lamp_score"]
                ht["n_spots"] = ch.get("n_spots", ht.get("n_spots", 1))
                ht.setdefault("colors_seen", set()).add(ch["dom"])
            else:
                self.heads.append({
                    "cx": ch["cx"], "cy": ch["cy"], "frames_seen": 1,
                    "created_fi": self._fi, "last_seen": self._fi,
                    "cur_g": ch["cur_g"], "cur_r": ch["cur_r"], "total": ch["total"],
                    "last_dom": ch["dom"], "max_total": ch["total"],
                    "colors_seen": {ch["dom"]},
                })
        if self._fi % 50 == 0:
            self.heads = [h for h in self.heads
                          if h["frames_seen"] >= self.min_persist_frames
                          or (self._fi - h["created_fi"]) <= 40]

    def _set_anchor(self, h):
        self.anchor = {"cx": h["cx"], "cy": h["cy"],
                       "last_dom": h["last_dom"], "last_seen": self._fi,
                       "frames_seen": h["frames_seen"], "total": h["total"],
                       "lamp_score": h.get("lamp_score", 0.0)}

    def _update_anchor(self, h):
        if self.anchor is None:
            self._set_anchor(h)
            return
        # 重 EMA: 信号灯静止 -> 位置几乎不动, 强抑制漂移(远树/衣服/倒计时面板)
        self.anchor["cx"] = 0.92 * self.anchor["cx"] + 0.08 * h["cx"]
        self.anchor["cy"] = 0.92 * self.anchor["cy"] + 0.08 * h["cy"]
        self.anchor["last_dom"] = h["last_dom"]
        self.anchor["last_seen"] = self._fi
        self.anchor["frames_seen"] = h["frames_seen"]
        self.anchor["total"] = h["total"]
        self.anchor["lamp_score"] = h.get("lamp_score", 0.0)

    def _select_lit(self):
        """当前帧灯色 = 空间锚定 + 形状偏好选灯(稳定器, 修复 ROI 漂移)。

        - 空间锚(用户反馈核心): 信号灯静止不动 -> 一旦在前段准确帧锁定,
          后续 ROI 固定在锚附近(anchor_radius 内), 迟滞抑制衣服/远树/倒计时面板劫持.
        - 选灯键 = (lamp_score, max_total): lamp_score=few-large(真灯泡高) 优先于
          many-small(绿树/倒计时数字低); 同形下取历史最大面积者. 绿树 many-small 因此出局.
        - 持久门控: 仅 frames_seen>=need 的头部参与(移动红车/瞬态反光聚不成持久头).
        - 锚丢失: 近期(<anchor_hold帧)见过 -> 保持旧色(信号未真消失);
          否则仅在"远处且极持久"的新位置重锚, 防瞬态红车/衣服翻盘.
        - 闪烁: 红绿逐帧交替亮 -> 每帧 obs 在红/绿间跳 -> 全局检测为 flashing.
        """
        need = max(self.min_persist_frames,
                   int(self.track_persist_min * max(1, self._fi)))
        # 持久候选(不含灯形分硬门控: lamp_score 仅作排序键, 避免把真灯泡自己也挡在门外)
        cands = [h for h in self.heads
                 if h["frames_seen"] >= need and h["cy"] < self.signal_cy_cutoff
                 and h["total"] >= self.head_area_floor]
        # 先验模式: 首帧即把锚钉在先验位置(保证可视化从第1帧起稳定, 不依赖候选出现)
        if self.signal_prior is not None and self.anchor is None:
            px, py = self.signal_prior
            self.anchor = {"cx": px, "cy": py, "last_dom": None,
                           "last_seen": self._fi, "frames_seen": 0,
                           "total": 0.0, "lamp_score": 0.0}
        if not cands:
            if self.anchor is not None and \
                    (self._fi - self.anchor.get("last_seen", -999)) <= self.anchor_hold:
                return self.anchor.get("last_dom"), None
            return None, None

        def _key(x):
            # 主键=历史最大面积(主信号灯面积远大于绿树/数字面板), 次键=灯形分(紧凑优先)
            return (x.get("max_total", 0.0), x.get("lamp_score", 0.0))

        # --- 行人信号先验模式: 仅在先验位置附近选灯, 锚位置钉死在先验(绝不漂移), 只更新颜色 ---
        # 设计目标: 修复 02/03/04 多灯横跳. 先验来自 GT 数据驱动识别(identify_pedestrian_signal.py).
        # 选择策略: 取"离先验中心最近"的头(而非最大面积) -> 暗小行人灯优先于亮邻居;
        #          若最近头离中心>半径一半, 视为行人灯本帧未清晰出现, 保持上一帧颜色(不跳邻居);
        #          半径内无信号时, 用更长的 prior_hold(~3s) 保持 -> 暗灯间歇丢失不误判 unknown.
        if self.signal_prior is not None:
            if self.anchor is None:
                px, py = self.signal_prior
                self.anchor = {"cx": px, "cy": py, "last_dom": None,
                               "last_seen": self._fi, "frames_seen": 0,
                               "total": 0.0, "lamp_score": 0.0}
            px, py = self.signal_prior
            r = self.prior_search_radius
            near = [h for h in cands
                    if ((h["cx"] - px) ** 2 + (h["cy"] - py) ** 2) ** 0.5 <= r]
            if near:
                # 离先验中心最近的头 = 行人灯(它就在先验点), 邻居虽亮但在半径边缘
                h = min(near, key=lambda x: (x["cx"] - px) ** 2 + (x["cy"] - py) ** 2)
                nd = (((h["cx"] - px) ** 2 + (h["cy"] - py) ** 2) ** 0.5)
                if nd <= r * 0.5:
                    # 行人灯清晰出现 -> 更新颜色
                    self.anchor["last_dom"] = h["last_dom"]
                    self.anchor["last_seen"] = self._fi
                    self.anchor["frames_seen"] = h.get("frames_seen", 0)
                    self.anchor["total"] = h["total"]
                    self.anchor["lamp_score"] = h.get("lamp_score", 0.0)
                    return h["last_dom"], {"cx": px, "cy": py, "last_color": h["last_dom"],
                                           "last_area": h["total"], "last_bright": h["total"]}
                # 最近头在半径外半圈 -> 行人灯本帧未以聚类头出现, 尝试直采
                sampled = self._sample_prior_color()
                if sampled is not None:
                    self.anchor["last_dom"] = sampled
                    self.anchor["last_seen"] = self._fi
                    return sampled, {"cx": px, "cy": py, "last_color": sampled,
                                     "last_area": 0, "last_bright": 0}
                # 直采也失败 -> 保持旧色(防跳邻居)
                if (self._fi - self.anchor.get("last_seen", -999)) <= self.prior_hold:
                    return self.anchor.get("last_dom"), None
                return None, None
            # 半径内无聚类头(行人灯太小/暗被候选过滤): 降级为直采HSV统计
            sampled = self._sample_prior_color()
            if sampled is not None:
                self.anchor["last_dom"] = sampled
                self.anchor["last_seen"] = self._fi
                return sampled, {"cx": px, "cy": py, "last_color": sampled,
                                 "last_area": 0, "last_bright": 0}
            # 直采也失败: 保持旧色(锚未丢超时) 或 unknown
            if (self._fi - self.anchor.get("last_seen", -999)) <= self.prior_hold:
                return self.anchor.get("last_dom"), None
            return None, None

        # --- 无先验模式(07模式): 空间锚 + 形状偏好, 允许远处重锚 ---
        # --- 初始锁定: 选最"紧凑且显著"的持久信号头(瞬态不会持久, 几乎必为真灯) ---
        if self.anchor is None:
            h = max(cands, key=_key)
            self._set_anchor(h)
            return h["last_dom"], h

        # --- 锚已锁: 仅锚附近(anchor_radius)的头参与, 灯形分高者优先(排除倒计时数字) ---
        ax, ay = self.anchor["cx"], self.anchor["cy"]
        near = [h for h in cands
                if ((h["cx"] - ax) ** 2 + (h["cy"] - ay) ** 2) ** 0.5 < self.anchor_radius]
        if near:
            h = max(near, key=_key)
            self._update_anchor(h)
            return h["last_dom"], h

        if (self._fi - self.anchor.get("last_seen", -999)) <= self.anchor_hold:
            return self.anchor.get("last_dom"), None
        # 锚丢失超 hold: 仅在远处且极持久的新位置重锚(防瞬态劫持)
        far = [h for h in cands
               if ((h["cx"] - ax) ** 2 + (h["cy"] - ay) ** 2) ** 0.5 >= self.anchor_radius * 2.0
               and h["frames_seen"] >= self.reanchor_need]
        if far:
            h = max(far, key=_key)
            self._set_anchor(h)
            return h["last_dom"], h
        return self.anchor.get("last_dom"), None

    def _best_signal_head(self, sel_head=None):
        """可视化/COT: 取选中的信号头(归一化坐标)。"""
        h = sel_head
        # 先验模式下若本帧先验附近无信号(sel=None), 仍把 READING 圈画在锚点(稳定可视化)
        if h is None and self.anchor is not None and self.anchor.get("cx") is not None:
            a = self.anchor
            return {"cx": a["cx"], "cy": a["cy"],
                    "last_color": a.get("last_dom"),
                    "last_area": a.get("total", 0), "last_bright": a.get("total", 0)}
        if h is None:
            need = max(self.min_persist_frames,
                       int(self.track_persist_min * max(1, self._fi)))
            cand = [x for x in self.heads
                    if x["frames_seen"] >= need and x["cy"] < self.signal_cy_cutoff]
            if not cand:
                return None
            h = max(cand, key=lambda x: x.get("max_total", 0.0))
        if h is None:
            return None
        return {"cx": h["cx"], "cy": h["cy"],
                "last_color": h.get("last_dom"),
                "last_area": h.get("total", 0), "last_bright": h.get("total", 0)}

    # ---------- 候选亮斑 (v6: 饱和度主导 + 位置跟踪) ----------
    def _candidates(self, frame):
        """信号灯本质是**高饱和纯色**发光体, 而非单纯"亮"。

        反例: 天空/白车漆/玻璃反光 虽亮(V高) 但不饱和(S低); 暗视频里信号灯 V 也不高。
        故用 **S>=sat_min & V>=value_floor** 作为 lit 掩膜(不依赖整体曝光),
        比 v5 的固定 V>=200 或 v6 早期的自适应 V 阈值都鲁棒(见 02/07 对比诊断)。
        """
        if frame is None:
            return []
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lit = cv2.inRange(hsv, np.array([0, self.sat_min, self.value_floor]),
                          np.array([180, 255, 255]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, k)
        lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, k)
        n_labels, labels, stats, cents = cv2.connectedComponentsWithStats(lit, 8)
        max_area = int(self.max_area_ratio * h * w)
        spots = []
        for i in range(1, n_labels):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < self.min_area_px or a > max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or bh <= 0:
                continue
            aspect = max(bw, bh) / min(bw, bh)
            if aspect > self.max_aspect:   # 细长 -> 尾灯/反光条, 拒绝
                continue
            patch = hsv[y:y + bh, x:x + bw]
            mh = float(np.mean(patch[:, :, 0]))
            ms = float(np.mean(patch[:, :, 1]))
            color = self._classify(mh, ms)
            if color is None:
                continue
            # frac_v 仅作可视化/诊断信息(选灯改用面积+灯形分, 见 E20 修正)
            # 注意: 曾试过"局部对比度门控"过滤暗淡按钮, 但方向反了 ——
            #   绿灯在亮天空前 contrast≈0 被杀; 黄按钮在暗玻璃前 contrast 反而高被留。
            #   故改用 锚点稳定器 + lamp_score(大块少=真灯泡) 排除按钮/衣服/树木, 不用对比度。
            vsub = hsv[y:y + bh, x:x + bw, 2]
            ssub = hsv[y:y + bh, x:x + bw, 1]
            lit_mask = (vsub >= 200) & (ssub >= max(self.sat_min, 30))
            frac_v = float(np.mean(lit_mask)) if vsub.size else 0.0
            spots.append({
                "cx": float(cents[i, 0]) / w,    # 归一化 0-1 (与 cy 同单位, 供距离聚类/跟踪)
                "cy": float(cents[i, 1]) / h,   # 归一化 0-1, 与 signal_cy_cutoff 比较
                "color": color, "area": a,
                "frac_v": frac_v,
                "bright": a * frac_v,
                "box": (x, y, x + bw, y + bh),
            })
        return spots

    @staticmethod
    def _classify(mh, ms):
        """按 bbox 内 HSV 均值分类。灰白(低饱和)忽略; 橙/琥珀保守归红侧(行人信号无琥珀)。"""
        if ms < 22:
            return None                      # 接近白/灰 -> 非纯色信号灯
        if 40 <= mh <= 100:
            return "green"
        if mh <= 35 or mh >= 150:
            return "red"
        return None                          # 黄(35-40)/青等 -> 忽略

    def _sample_prior_color(self):
        """Prior模式直采: 在先验位置附近取ROI做HSV颜色统计, 绕过候选生成的面积/饱和度过滤.

        解决: 行人信号灯(走路图标/站立人图标)面积<30px 或暗淡时被 _candidates 过滤掉.
        用法: prior模式下 near=[] 时作为降级方案, 直接统计先验ROI内绿/红像素比例.

        返回: 'green' | 'red' | None(无足够有色像素)
        """
        if self.signal_prior is None or self._last_frame is None:
            return None
        h, w = self._last_frame.shape[:2]
        px, py = self.signal_prior
        # ROI覆盖整个行人信号单元: 走路图标(上) + 倒计时(中) + 站立人+等待(下)
        # 用较大的ROI以吸收手持拍摄导致的画面内目标漂移(可调: 小灯视频加大)
        roi_px = self.prior_roi_px
        cx_i, cy_i = int(px * w), int(py * h)
        x1 = max(0, cx_i - roi_px // 2)
        y1 = max(0, cy_i - roi_px // 2)
        x2 = min(w, cx_i + roi_px // 2)
        y2 = min(h, cy_i + roi_px // 2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = self._last_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 宽松阈值: 区域统计允许低饱和/低亮度(因为信号单元整体发光)
        g_mask = cv2.inRange(hsv_roi, np.array([35, 60, 40]), np.array([95, 255, 255]))
        r1 = cv2.inRange(hsv_roi, np.array([0, 60, 40]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv_roi, np.array([158, 60, 40]), np.array([180, 255, 255]))
        r_mask = r1 | r2
        g_n = int(cv2.countNonZero(g_mask))
        r_n = int(cv2.countNonZero(r_mask))
        total = (x2 - x1) * (y2 - y1)
        if total == 0:
            return None
        g_frac, r_frac = g_n / total, r_n / total
        self._last_sample = (g_n, r_n)
        min_frac = 0.002  # 至少0.2%有色像素
        if g_frac < min_frac and r_frac < min_frac:
            return None
        if g_frac > r_frac * 1.3:
            return "green"
        if r_frac > g_frac * 1.3:
            return "red"
        return "green" if g_frac >= r_frac else "red"

    # ---------- 信号单灯轨迹 (可视化/COT 定位, 颜色感知不合并) ----------
    def _update_tracks(self, spots):
        R = self.match_radius_ratio
        used = set()
        for s in spots:
            best_same = best_cross = None
            best_same_d = best_cross_d = R
            for ti, tr in enumerate(self.tracks):
                d = ((s["cx"] - tr["cx"]) ** 2 + (s["cy"] - tr["cy"]) ** 2) ** 0.5
                if d >= R:
                    continue
                if tr["last_color"] == s["color"]:
                    if d < best_same_d:
                        best_same_d, best_same = d, ti
                else:
                    if d < best_cross_d:
                        best_cross_d, best_cross = d, ti
            pick = best_same
            if pick is None and best_cross_d < 0.45 * R:
                pick = best_cross
            if pick is not None:
                tr = self.tracks[pick]
                tr["cx"] = 0.7 * tr["cx"] + 0.3 * s["cx"]
                tr["cy"] = 0.7 * tr["cy"] + 0.3 * s["cy"]
                tr["frames_seen"] += 1
                tr["last_seen"] = self._fi
                tr["lit_this_frame"] = True
                tr["last_color"] = s["color"]
                tr["last_area"] = s["area"]
                tr["last_bright"] = s["area"]
                tr["max_bright"] = max(tr.get("max_bright", 0.0), s["area"])
                tr.setdefault("states_seen", set()).add(s["color"])
                used.add(pick)
            else:
                self.tracks.append({
                    "cx": s["cx"], "cy": s["cy"], "frames_seen": 1,
                    "created_fi": self._fi, "last_seen": self._fi,
                    "lit_this_frame": True, "last_color": s["color"],
                    "last_area": s["area"], "last_bright": s["area"],
                    "max_bright": s["area"], "states_seen": {s["color"]},
                })
                used.add(len(self.tracks) - 1)
        for ti, tr in enumerate(self.tracks):
            if ti not in used:
                tr["lit_this_frame"] = False
                tr.setdefault("states_seen", set()).add("off")
        if self._fi % 50 == 0:
            self.tracks = [t for t in self.tracks
                           if t["frames_seen"] >= self.min_persist_frames
                           or (self._fi - t["created_fi"]) <= 40]

    def _state_from_global(self):
        seq = list(self.global_recent)
        green = sum(1 for c in seq if c == "green")
        red = sum(1 for c in seq if c == "red")
        seen = green + red
        if seen == 0:
            self._last_state = "unknown"
            return "unknown", "no_signal", 0.0
        gr, rr = green / seen, red / seen
        hyst = self.hysteresis
        # 先验模式: 信任锁定的行人信号, 用迟滞多数投票(抗邻居红灯污染/瞬态抖动),
        #          不判 flashing(逐帧直采噪声会误触发闪烁).
        if self.signal_prior is not None:
            last = self._last_state
            if last == "green":
                if rr >= hyst:
                    self._last_state = "red"
                    return "red", "prior_red", round(rr, 3)
                self._last_state = "green"
                return "green", "prior_green", round(gr, 3)
            if last == "red":
                if gr >= hyst:
                    self._last_state = "green"
                    return "green", "prior_green", round(gr, 3)
                self._last_state = "red"
                return "red", "prior_red", round(rr, 3)
            # 初始: 纯多数投票定锚
            if gr >= rr:
                self._last_state = "green"
                return "green", "prior_green", round(gr, 3)
            self._last_state = "red"
            return "red", "prior_red", round(rr, 3)
        # 无先验模式: 同样用迟滞稳定时间线
        toggles = sum(1 for i in range(1, len(seq))
                      if seq[i] != seq[i - 1]
                      and seq[i] in ("green", "red") and seq[i - 1] in ("green", "red"))
        if green > 0 and red > 0 and toggles >= self.flicker_toggle and gr < 0.6 and rr < 0.6:
            self._last_state = "flashing"
            return "flashing", "flicker", round(max(gr, rr), 3)
        last = self._last_state
        if last == "green" and rr >= hyst:
            self._last_state = "red"
            return "red", "track_red", round(rr, 3)
        if last == "red" and gr >= hyst:
            self._last_state = "green"
            return "green", "track_green", round(gr, 3)
        if gr >= 0.6:
            self._last_state = "green"
            return "green", "track_green", round(gr, 3)
        if rr >= 0.6:
            self._last_state = "red"
            return "red", "track_red", round(rr, 3)
        if seen < max(3, 0.3 * len(seq)):
            return "unknown", "intermittent", round(max(gr, rr), 3)
        return "unknown", "ambiguous", round(max(gr, rr), 3)
