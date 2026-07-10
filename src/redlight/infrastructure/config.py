"""L1 基础设施: 配置加载。

集中管理所有运行时参数。改 configs/config.yaml 即可调参, 无需动代码。
"""
import os
from types import SimpleNamespace
import yaml


def dict_to_ns(d):
    """递归把 dict 转成可用属性访问的 SimpleNamespace。"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict_to_ns(data)


def project_root():
    """返回工程根目录 (src/redlight/infrastructure -> 上溯两级)。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
