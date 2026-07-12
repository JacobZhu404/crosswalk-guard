"""直接运行 traffic_light 单元测试(绕过 pytest, 打印耗时, 便于定位卡顿)。"""
import sys, os, time, importlib.util

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests", "unit"))

spec = importlib.util.spec_from_file_location("tl_tests", "tests/unit/test_traffic_light.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

names = [n for n in dir(mod) if n.startswith("test_")]
ok = 0
for n in names:
    fn = getattr(mod, n)
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"PASS {n:42s} {dt:6.2f}s")
        ok += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"FAIL {n:42s} {dt:6.2f}s  -> {type(e).__name__}: {e}")
print(f"\n{ok}/{len(names)} passed")
