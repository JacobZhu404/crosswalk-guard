import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.signal_state_classifier import SignalStateClassifier, LABELS


def test_unavailable_when_no_weights():
    clf = SignalStateClassifier(model_path=None)
    assert clf.available is False
    assert clf.classify(np.zeros((10, 10, 3), np.uint8)) == ("off", 0.0)


def test_argmax_maps_to_label(monkeypatch):
    clf = SignalStateClassifier(model_path=None)
    clf.available = True
    # LABELS = ["walk","stand","off"]; 让 stand(idx1) 概率最高
    monkeypatch.setattr(clf, "_infer", lambda blob: np.array([0.1, 0.7, 0.2]))
    label, conf = clf.classify(np.zeros((48, 48, 3), np.uint8))
    assert label == "stand"
    assert abs(conf - 0.7) < 1e-6


def test_preprocess_shape():
    clf = SignalStateClassifier(model_path=None)
    blob = clf._preprocess(np.zeros((20, 60, 3), np.uint8))
    assert blob.shape == (1, 3, 48, 48)   # NCHW, 48x48
