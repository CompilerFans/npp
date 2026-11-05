#!/usr/bin/env python3
"""Generate CPU-golden tests for the nppi morphological operations.

The plain Erode/Dilate entry points sample outside the ROI at the borders;
the NPP documentation declares that undefined, so their drivers compare only
interior pixels.  The Border/composite families use replicate sampling and are
compared over the full ROI.  nppiMorphGradientBorder is measured to produce
close - src, matching BlackHat.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_ref_tests as G

CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
CTYPE = {"8u": "Npp8u", "16u": "Npp16u", "32f": "Npp32f", "16s": "Npp16s", "64f": "Npp64f"}
LAYOUT = {"C1": (1, 1), "C3": (3, 3), "C4": (4, 4), "AC4": (4, 3)}


def emit(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    m = re.search(r"_(AC4|C1|C3|C4)R_Ctx$", fn)
    if m is None:
        return None
    tok = m.group(1)
    if tok not in LAYOUT:
        return None
    C, A = LAYOUT[tok]
    dtype = row["dtype"]
    if dtype not in CTYPE:
        return None
    S = CTYPE[dtype]
    fam = row["family"]

    if fam in ("nppiDilate3x3", "nppiErode3x3"):
        dilate = fam == "nppiDilate3x3"
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)c; return {fn}(a, sa, d, ds, r, c); }}")
        prefix = f"num::run_morph_plain3<{S}, {C}, {A}, {str(dilate).lower()}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")

    if fam in ("nppiDilate", "nppiErode"):
        dilate = fam == "nppiDilate"
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, const Npp8u *m, "
               f"NppiSize ms, NppiPoint an, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)m; (void)ms; (void)an; (void)c; "
               f"return {fn}(a, sa, d, ds, r, m, ms, an, c); }}")
        prefix = f"num::run_morph_plain_mask<{S}, {C}, {A}, {str(dilate).lower()}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")

    if fam in ("nppiDilate3x3Border", "nppiErode3x3Border"):
        dilate = fam == "nppiDilate3x3Border"
        lam = (f"[](const {S} *a, int sa, NppiSize ss, NppiPoint so, {S} *d, int ds, "
               f"NppiSize r, NppiBorderType bt, NppStreamContext c) "
               f"{{ (void)sa; (void)ss; (void)so; (void)ds; (void)r; (void)bt; (void)c; "
               f"return {fn}(a, sa, ss, so, d, ds, r, bt, c); }}")
        prefix = f"num::run_morph_border3<{S}, {C}, {A}, {str(dilate).lower()}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")

    if fam in ("nppiDilateBorder", "nppiErodeBorder"):
        dilate = fam == "nppiDilateBorder"
        lam = (f"[](const {S} *a, int sa, NppiSize ss, NppiPoint so, {S} *d, int ds, "
               f"NppiSize r, const Npp8u *m, NppiSize ms, NppiPoint an, NppiBorderType bt, "
               f"NppStreamContext c) "
               f"{{ (void)sa; (void)ss; (void)so; (void)ds; (void)r; (void)m; (void)ms; "
               f"(void)an; (void)bt; (void)c; return {fn}(a, sa, ss, so, d, ds, r, m, ms, an, bt, c); }}")
        prefix = f"num::run_morph_border_mask<{S}, {C}, {A}, {str(dilate).lower()}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")

    combos = {
        "nppiMorphOpenBorder": 0,
        "nppiMorphCloseBorder": 1,
        "nppiMorphGradientBorder": 2,
        "nppiMorphTopHatBorder": 3,
        "nppiMorphBlackHatBorder": 4,
    }
    if fam in combos:
        kind = combos[fam]
        lam = (f"[](const {S} *a, int sa, NppiSize ss, NppiPoint so, {S} *d, int ds, "
               f"NppiSize r, const Npp8u *m, NppiSize ms, NppiPoint an, Npp8u *buf, "
               f"NppiBorderType bt, NppStreamContext c) "
               f"{{ (void)sa; (void)ss; (void)so; (void)ds; (void)r; (void)m; (void)ms; "
               f"(void)an; (void)buf; (void)bt; (void)c; "
               f"return {fn}(a, sa, ss, so, d, ds, r, m, ms, an, buf, bt, c); }}")
        prefix = f"num::run_morph_combo<{S}, {C}, {A}, {kind}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")

    if fam in ("nppiGrayDilateBorder", "nppiGrayErodeBorder"):
        if C != 1:
            return None
        dilate = fam == "nppiGrayDilateBorder"
        M = "Npp32s" if dtype == "8u" else "Npp32f"
        lam = (f"[](const {S} *a, int sa, NppiSize ss, NppiPoint so, {S} *d, int ds, "
               f"NppiSize r, const {M} *m, NppiSize ms, NppiPoint an, NppiBorderType bt, "
               f"NppStreamContext c) "
               f"{{ (void)sa; (void)ss; (void)so; (void)ds; (void)r; (void)m; (void)ms; "
               f"(void)an; (void)bt; (void)c; return {fn}(a, sa, ss, so, d, ds, r, m, ms, an, bt, c); }}")
        prefix = f"num::run_morph_gray<{S}, {M}, {str(dilate).lower()}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG))
            if r["module"] == "nppi_morphological_operations.h" and r["ctx"] == "1"]
    out, skipped = [], []
    for row in rows:
        text = emit(row)
        if text is None:
            skipped.append(dict(function=row["function_name"], family=row["family"],
                                reason="unsupported signature/type"))
            continue
        out.append(text)
    header = ("// GENERATED by tools/gen_nppi_morph_tests.py -- DO NOT EDIT.\n"
              "// module: nppi_morphological_operations.h\n"
              "#include \"num_nppi_morph.h\"\n\n")
    dest = Path(args.out) if args.out else ROOT / "test" / "unit" / "num" / "test_gen_nppi_morphology.cpp"
    if args.emit:
        dest.write_text(header + "\n".join(out))
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev["nppi_morphological_operations.h"] = dict(emitted=len(out), skipped=skipped)
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests to {dest}; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
        for s in skipped[:20]:
            print("  skip", s["function"], s["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
