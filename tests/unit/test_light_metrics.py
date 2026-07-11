"""信号灯状态分类指标单元测试 (要求#6)。"""
from redlight.evaluation.metrics import light_state_metrics


def test_perfect():
    pred = ["red", "green", "flashing", "unknown"] * 3
    gt = list(pred)
    r = light_state_metrics(pred, gt)
    assert r["accuracy"] == 1.0
    assert r["per_class"]["red"]["recall"] == 1.0
    assert r["per_class"]["green"]["precision"] == 1.0
    assert r["n"] == 12


def test_partial():
    # idx: 0 红对红(TP); 1 预测红但GT绿(FP+FN); 2 绿对绿(TP); 3 预测unknown但GT红(FN)
    pred = ["red", "red", "green", "unknown"]
    gt = ["red", "green", "green", "red"]
    r = light_state_metrics(pred, gt)
    assert abs(r["per_class"]["red"]["precision"] - 0.5) < 1e-9
    assert abs(r["per_class"]["red"]["recall"] - 0.5) < 1e-9
    assert r["accuracy"] == 0.5
    # green: tp=1(idx2), fn=1(idx1 预测红但GT绿) -> recall=0.5; fp=0 -> precision=1.0
    assert abs(r["per_class"]["green"]["recall"] - 0.5) < 1e-9
    assert r["per_class"]["green"]["precision"] == 1.0


def test_empty():
    r = light_state_metrics([], [])
    assert r["n"] == 0
    assert r["accuracy"] == 0.0
