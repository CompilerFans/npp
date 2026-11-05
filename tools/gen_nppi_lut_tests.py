#!/usr/bin/env python3
"""Generate CPU-golden tests for the nppi point/linear/cubic LUT families.

Covers nppiLUT (point), nppiLUT_Linear and nppiLUT_Cubic for
8u/16u/16s/32f x C1/C3/C4/AC4 x R/IR.  The reference lives in
test/unit/num/num_nppi_lut.h and follows the measured NPP 13.4 formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional, List

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
CTYPE = {"8u": "Npp8u", "16u": "Npp16u", "16s": "Npp16s", "32f": "Npp32f"}
MODE = {"nppiLUT": "Point", "nppiLUT_Linear": "Linear", "nppiLUT_Cubic": "Cubic"}
LAYOUT = {"C1": (1, 1), "C3": (3, 3), "C4": (4, 4), "AC4": (4, 3)}


def emit_lut(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in MODE:
        return None
    m = re.search(r"_(AC4|C1|C3|C4)(I?)R_Ctx$", fn)
    if m is None:
        return None
    tok, inplace = m.group(1), m.group(2) == "I"
    S = CTYPE.get(row["dtype"])
    if tok not in LAYOUT or S is None:
        return None
    C, A = LAYOUT[tok]
    mode = MODE[fam]
    V = "Npp32f" if S == "Npp32f" else "Npp32s"
    name = re.sub(r"\W", "_", fn)
    prefix = None
    if tok == "C1":
        if inplace:
            lam = (f"[]({S} *a, int st, NppiSize r, const {V} *vals, const {V} *lev, int n, "
                   f"NppStreamContext c) "
                   f"{{ (void)st; (void)r; (void)vals; (void)lev; (void)n; (void)c; "
                   f"return {fn}(a, st, r, vals, lev, n, c); }}")
            prefix = f"num::lut::run_lut_c1_ir<{S}, num::lut::LutMode::{mode}>"
        else:
            lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, const {V} *vals, "
                   f"const {V} *lev, int n, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)vals; (void)lev; (void)n; (void)c; "
                   f"return {fn}(a, sa, d, ds, r, vals, lev, n, c); }}")
            prefix = f"num::lut::run_lut_c1<{S}, num::lut::LutMode::{mode}>"
    else:
        if inplace:
            lam = (f"[]({S} *a, int st, NppiSize r, const {V} **vals, const {V} **lev, int *n, "
                   f"NppStreamContext c) "
                   f"{{ (void)st; (void)r; (void)vals; (void)lev; (void)n; (void)c; "
                   f"return {fn}(a, st, r, vals, lev, n, c); }}")
            prefix = f"num::lut::run_lut_multi_ir<{S}, {C}, {A}, num::lut::LutMode::{mode}>"
        else:
            lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, const {V} **vals, "
                   f"const {V} **lev, int *n, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)vals; (void)lev; (void)n; (void)c; "
                   f"return {fn}(a, sa, d, ds, r, vals, lev, n, c); }}")
            prefix = f"num::lut::run_lut_multi<{S}, {C}, {A}, num::lut::LutMode::{mode}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



PAL_C1 = {"8u": ("Npp8u", "Npp8u", "Npp8u"), "16u": ("Npp16u", "Npp16u", "Npp16u"),
          "8u32u": ("Npp8u", "Npp32u", "Npp32u"), "16u8u": ("Npp16u", "Npp8u", "Npp8u"),
          "16u32u": ("Npp16u", "Npp32u", "Npp32u")}


def emit_lut_palette(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    m = re.match(r"nppiLUTPalette_(\w+?)_C1R_Ctx$", fn)
    if m:
        key = m.group(1)
        if key in ("8u24u", "16u24u"):
            S = "Npp8u" if key == "8u24u" else "Npp16u"
            lam = (f"[](const {S} *a, int sa, Npp8u *d, int ds, NppiSize r, const Npp8u *tab, "
                   f"int bits, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)tab; (void)bits; (void)c; "
                   f"return {fn}(a, sa, d, ds, r, tab, bits, c); }}")
            return (f"TEST(NumGen, {name}) {{\n  num::lut::run_lut_palette24<{S}>(\n      \"{fn}\",\n"
                    f"      {lam});\n}}\n")
        if key in PAL_C1:
            S, T, D = PAL_C1[key]
            lam = (f"[](const {S} *a, int sa, {D} *d, int ds, NppiSize r, const {T} *tab, "
                   f"int bits, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)tab; (void)bits; (void)c; "
                   f"return {fn}(a, sa, d, ds, r, tab, bits, c); }}")
            return (f"TEST(NumGen, {name}) {{\n  num::lut::run_lut_palette_c1<{S}, {T}, {D}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")
        return None
    m = re.match(r"nppiLUTPalette_(8u|16u)_(C3R|C4R|AC4R)_Ctx$", fn)
    if m:
        S = "Npp8u" if m.group(1) == "8u" else "Npp16u"
        C, A = LAYOUT[m.group(2)[:-1]]
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, const {S} **tabs, int bits, "
               f"NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)tabs; (void)bits; (void)c; "
               f"return {fn}(a, sa, d, ds, r, tabs, bits, c); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::lut::run_lut_palette_multi<{S}, {C}, {A}>(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")
    m = re.match(r"nppiLUTPaletteSwap_(8u|16u)_C3A0C4R_Ctx$", fn)
    if m:
        S = "Npp8u" if m.group(1) == "8u" else "Npp16u"
        lam = (f"[](const {S} *a, int sa, int alpha, {S} *d, int ds, NppiSize r, const {S} **tabs, "
               f"int bits, NppStreamContext c) "
               f"{{ (void)sa; (void)alpha; (void)ds; (void)r; (void)tabs; (void)bits; (void)c; "
               f"return {fn}(a, sa, alpha, d, ds, r, tabs, bits, c); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::lut::run_lut_palette_swap<{S}>(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    m = re.match(r"nppiLUT_Trilinear_8u_(C4R|AC4R|AC4IR)_Ctx$", fn)
    if m:
        kind = m.group(1)
        if kind == "AC4IR":
            lam = (f"[](const Npp8u *a, int sa, Npp8u *d, int ds, NppiSize r, Npp32u *vals, "
                   f"Npp8u **lev, int *nl, NppStreamContext c) "
                   f"{{ (void)d; (void)ds; (void)r; (void)vals; (void)lev; (void)nl; (void)c; "
                   f"return {fn}(const_cast<Npp8u *>(a), sa, r, vals, lev, nl, c); }}")
            prefix = "num::lut::run_lut_trilinear<false, true>"
        else:
            lam = (f"[](const Npp8u *a, int sa, Npp8u *d, int ds, NppiSize r, Npp32u *vals, "
                   f"Npp8u **lev, int *nl, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)vals; (void)lev; (void)nl; (void)c; "
                   f"return {fn}(a, sa, d, ds, r, vals, lev, nl, c); }}")
            prefix = "num::lut::run_lut_trilinear<false, false>" if kind == "AC4R" \
                else "num::lut::run_lut_trilinear<true, false>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG))
            if r["module"] == "nppi_color_conversion.h" and r["ctx"] == "1"]
    out: List[str] = []
    skipped: List[dict] = []
    for row in rows:
        fam = row["family"]
        text = emit_lut(row) if fam in MODE else emit_lut_palette(row)
        if text is None:
            skipped.append(dict(function=row["function_name"], family=fam,
                                reason="later color batch / unsupported"))
            continue
        out.append(text)
    header = ("// GENERATED by tools/gen_nppi_lut_tests.py -- DO NOT EDIT.\n"
              "// module: nppi_color_conversion.h (point/linear/cubic LUT)\n"
              "#include \"num_nppi_lut.h\"\n\n")
    dest = Path(args.out) if args.out else ROOT / "test" / "unit" / "num" / "test_gen_nppi_lut.cpp"
    if args.emit:
        dest.write_text(header + "\n".join(out))
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev["nppi_color_conversion.h"] = dict(emitted=len(out), skipped=skipped)
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests to {dest}; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
