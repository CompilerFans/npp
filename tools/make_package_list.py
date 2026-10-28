#!/usr/bin/env python3
"""Generate per-package function lists from a package definition.

Definition file (api_analysis/packages/packages.json):

{
  "p1.1a": {
    "title": "NPPS binary arithmetic",
    "module": "npps_arithmetic_and_logical_operations.h",
    "families": ["nppsAdd", "nppsSub"]
  }
}

Usage:
  python3 tools/make_package_list.py --update
  python3 tools/make_package_list.py --package p1.1a
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
PKG_DIR = ROOT / "api_analysis" / "packages"
DEFS = PKG_DIR / "packages.json"

FIELDS = ["function_name", "family", "status", "ctx", "dtype", "params", "signature"]


def load_catalog() -> list[dict]:
    with open(CATALOG, newline="") as fh:
        return list(csv.DictReader(fh))


def select(rows: list[dict], spec: dict) -> list[dict]:
    families = set(spec.get("families", []))
    exact = set(spec.get("functions", []))
    modules = set(spec.get("modules", [])) or {spec["module"]}
    out = []
    for r in rows:
        if r["module"] not in modules:
            continue
        if exact:
            if r["function_name"] not in exact:
                continue
        elif r["family"] not in families:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["family"], r["function_name"]))
    return out


def write_package(pkg: str, spec: dict, rows: list[dict]) -> None:
    selected = select(rows, spec)
    path = PKG_DIR / f"{pkg}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)
    missing = sum(1 for r in selected if r["status"] != "real")
    by_family: dict[str, list[int]] = {}
    for r in selected:
        stats = by_family.setdefault(r["family"], [0, 0, 0])
        stats[0] += 1
        if r["status"] == "real":
            stats[1] += 1
        else:
            stats[2] += 1
    print(f"{pkg}: {len(selected)} functions, {missing} remaining -> {path.relative_to(ROOT)}")
    for fam, (total, real, rem) in sorted(by_family.items(), key=lambda kv: -kv[1][2]):
        print(f"  {fam:28s} total={total:4d} real={real:4d} remaining={rem:4d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="write every package list")
    ap.add_argument("--package", help="write one package list")
    args = ap.parse_args()

    defs = json.loads(DEFS.read_text())
    rows = load_catalog()
    names = [args.package] if args.package else list(defs)
    if args.package and args.package not in defs:
        raise SystemExit(f"unknown package {args.package}; known: {', '.join(defs)}")
    for name in names:
        write_package(name, defs[name], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
