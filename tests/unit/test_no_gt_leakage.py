"""红线单测(cc 复核重点①): GT 绝不进生产检测器。

断言:
  - 生产检测器 crosswalk_v2.py 真实代码(剥离 docstring/注释后)不含 datasets/gt /
    GtCrosswalkDetector / crosswalk/*.json 引用。
    (docstring 里的"禁用说明"是手册, 不算泄漏; 故先剥离再扫。)
  - cli.py 生产路径不实例化 GtCrosswalkDetector(仅 diag 脚本可用)。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

SRC_V2 = os.path.join(ROOT, "src", "redlight", "models", "crosswalk_v2.py")
SRC_CLI = os.path.join(ROOT, "src", "redlight", "app", "cli.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _strip_docstrings(src):
    src = re.sub(r'""".*?"""', '', src, flags=re.DOTALL)
    src = re.sub(r"'''.*?'''", '', src, flags=re.DOTALL)
    src = re.sub(r'#.*$', '', src, flags=re.MULTILINE)
    return src


def test_v2_has_no_gt_leakage():
    code = _strip_docstrings(_read(SRC_V2))
    assert "datasets/gt" not in code, "crosswalk_v2 真实代码不得引用 datasets/gt"
    assert "GtCrosswalkDetector" not in code, "crosswalk_v2 不得用 GtCrosswalkDetector"
    assert "crosswalk/*.json" not in code, "crosswalk_v2 不得读标注 json 作生产输入"


def test_cli_does_not_instantiate_gt_detector_in_production():
    src = _read(SRC_CLI)
    assert "GtCrosswalkDetector" not in src, \
        "cli.py 生产路径不得实例化 GtCrosswalkDetector(仅 diag 脚本可用)"


if __name__ == "__main__":
    test_v2_has_no_gt_leakage()
    test_cli_does_not_instantiate_gt_detector_in_production()
    print("ALL no-gt-leakage tests passed")
