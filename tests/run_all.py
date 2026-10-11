"""Dependency-free runner (pytest also works: `pytest -q`)."""
import importlib, pkgutil, sys, traceback
sys.path.insert(0, ".")
failed = passed = 0
for m in pkgutil.iter_modules(["tests"]):
    if not m.name.startswith("test_"):
        continue
    mod = importlib.import_module(f"tests.{m.name}")
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)(); passed += 1; print("PASS", m.name, name)
            except Exception:
                failed += 1; print("FAIL", m.name, name); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed"); sys.exit(1 if failed else 0)
