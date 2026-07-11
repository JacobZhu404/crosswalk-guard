"""L4a 推理引擎: 统一封装各检测/识别模型, 提供逐帧推理接口。

把模型层 (L3) 的四个检测器聚合成一个推理入口, 便于 DAG 节点与评测调用。
"""


class InferenceEngine:
    def __init__(self, cfg, verbose=True):
        from ..models.vehicle import VehicleDetector
        from ..models.crosswalk import CrosswalkDetector
        from ..models.traffic_light import TrafficLightDetector
        from ..models.plate import PlateRecognizer
        self.vehicle = VehicleDetector(cfg, verbose=verbose)
        self.crosswalk = CrosswalkDetector(cfg, verbose=verbose)
        self.light = TrafficLightDetector(cfg, verbose=verbose)
        self.plate = PlateRecognizer(cfg, verbose=verbose)

    def detect_vehicles(self, frame):
        return self.vehicle.detect(frame)

    def detect_crosswalk(self, frame):
        return self.crosswalk.detect(frame)

    def detect_light(self, frame):
        return self.light.detect(frame)

    def detect_plates(self, frame, vehicle_boxes):
        return self.plate.detect(frame, vehicle_boxes)

    def get_info(self):
        return {
            "vehicle": self.vehicle.get_info(),
            "crosswalk": self.crosswalk.get_info(),
            "light": self.light.get_info(),
            "plate": self.plate.get_info(),
        }
