"""Dependency-free runner (pytest preferred: `pytest -q`)."""
import sys, traceback, importlib; sys.path.insert(0, ".")
fails = total = 0
for mod in ("tests.test_core", "tests.test_agents"):
    t = importlib.import_module(mod)
    for n in [n for n in dir(t) if n.startswith("test_")]:
        total += 1
        try: getattr(t, n)(); print("PASS", n)
        except Exception: fails += 1; print("FAIL", n); traceback.print_exc()
print(f"{total-fails}/{total} passed"); sys.exit(fails > 0)
