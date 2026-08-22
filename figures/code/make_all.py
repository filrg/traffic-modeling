from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as C  # noqa: E402

# render order: the arrival side, then the size side, then the tables
MODULES = ("arrivals", "excursion", "gaps", "generator", "jobsize", "jobsize_regimes", "studies")


def load(name: str):
    return importlib.import_module(name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render every figure and its csv twin.")
    ap.add_argument(
        "--module", action="append", choices=MODULES, help="restrict to one module; repeatable"
    )
    ap.add_argument("--list", action="store_true", help="list modules and views, then exit")
    ap.add_argument(
        "--check", action="store_true", help="report which inputs each module is missing, then exit"
    )
    args = ap.parse_args()

    chosen = tuple(args.module) if args.module else MODULES

    if args.list:
        for name in chosen:
            mod = load(name)
            print(f"{name}: {', '.join(mod.VIEWS)}")
        return

    if args.check:
        ok = True
        for name in chosen:
            mod = load(name)
            print(f"{name}:")
            ok &= C.check_inputs(getattr(mod, "INPUTS", ()))
        raise SystemExit(0 if ok else 1)

    failed = []
    for name in chosen:
        mod = load(name)
        missing = [k for k in getattr(mod, "INPUTS", ()) if not C.ARTIFACTS[k][0].exists()]
        if missing:
            print(f"=== {name}: SKIPPED, missing {', '.join(missing)}")
            failed.append(name)
            continue
        for view, fn in mod.VIEWS.items():
            print(f"=== {name}.{view} ===")
            try:
                fn()
            except Exception:
                traceback.print_exc()
                failed.append(f"{name}.{view}")
    if failed:
        print("\nfailed or skipped:", ", ".join(failed))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
