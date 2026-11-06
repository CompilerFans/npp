#!/usr/bin/env python3
"""Generate CPU-golden tests for nppi_filtering_functions.h.

Each supported family is routed to a driver in test/unit/num/num_nppi_filter_*.h
that computes the expected result on the host and compares it with the device
output (checked against NVIDIA NPP on the RTX 5090 and against the in-tree
implementation on CUDA and MACA).
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
DEST = ROOT / "test" / "unit" / "num" / "test_gen_nppi_filter.cpp"
REPORT = ROOT / "test" / "unit" / "num" / "generated_report.json"

DT = {"8u": ("Npp8u", "Npp8u"), "16u": ("Npp16u", "Npp16u"),
      "16s": ("Npp16s", "Npp16s"), "32f": ("Npp32f", "Npp32f"),
      "8u16s": ("Npp8u", "Npp16s"), "8s16s": ("Npp8s", "Npp16s"),
      "64f": ("Npp64f", "Npp64f"), "32s": ("Npp32s", "Npp32s"),
      "8s": ("Npp8s", "Npp8s")}
LAYOUT = {"C1R": (1, 1), "C2R": (2, 2), "C3R": (3, 3), "C4R": (4, 4), "AC4R": (4, 3)}

FIXED_OPS = {"HighPass": "kHighPass", "LowPass": "kLowPass",
             "Sharpen": "kSharpen", "Laplace": "kLaplace"}


def emit_fixed(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiFilter(HighPass|LowPass|Sharpen|Laplace)(Border)?_(\w+)_(C1R|C3R|C4R|AC4R)_Ctx$", fn)
    if not m:
        return None
    op, border, dt, lay = m.group(1), m.group(2) is not None, m.group(3), m.group(4)
    if dt not in DT or lay not in LAYOUT:
        return None
    src, dst = DT[dt]
    c, a = LAYOUT[lay]
    name = re.sub(r"\W", "_", fn)
    if border:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, NPP_BORDER_REPLICATE, ctx);"
                if op == "Sharpen" else
                f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, (NppiMaskSize)mask, "
                f"NPP_BORDER_REPLICATE, ctx);")
    else:
        body = (f"return {fn}(sp, ss, dp, ds, roi, ctx);" if op == "Sharpen" else
                f"return {fn}(sp, ss, dp, ds, roi, (NppiMaskSize)mask, ctx);")
    lam = (f"[](const {src} *sp, int ss, {dst} *dp, int ds, "
           f"NppiSize roi, NppiSize srcSize, NppiPoint off, int mask, NppStreamContext ctx) "
           f"{{ (void)srcSize; (void)off; (void)mask; {body} }}")
    prefix = (f"num::filtfixed::run_fixed<{src}, {dst}, {c}, {a}, "
              f"num::filtfixed::{FIXED_OPS[op]}, {'true' if border else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


DERIV_RE = (r"nppiFilter(PrewittHoriz|PrewittVert|RobertsDown|RobertsUp|SobelHorizSecond|"
            r"SobelVertSecond|SobelHorizMask|SobelVertMask|SobelCross|SobelHoriz|SobelVert|"
            r"ScharrHoriz|ScharrVert)(Border)?_(\w+)_(C1R|C3R|C4R|AC4R)_Ctx$")


def emit_deriv(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(DERIV_RE, fn)
    if not m:
        return None
    op, border, dt, lay = m.group(1), m.group(2) is not None, m.group(3), m.group(4)
    if dt not in DT or lay not in LAYOUT:
        return None
    src, dst = DT[dt]
    c, a = LAYOUT[lay]
    masked = ("Second" in op or op == "SobelCross" or "Mask" in op or
              (dt in ("8u16s", "8s16s") and op in ("SobelHoriz", "SobelVert")))
    base = op.replace("Mask", "")
    name = re.sub(r"\W", "_", fn)
    if border:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, (NppiMaskSize)mask, "
                f"NPP_BORDER_REPLICATE, ctx);" if masked else
                f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, NPP_BORDER_REPLICATE, ctx);")
    else:
        body = (f"return {fn}(sp, ss, dp, ds, roi, (NppiMaskSize)mask, ctx);" if masked else
                f"return {fn}(sp, ss, dp, ds, roi, ctx);")
    lam = (f"[](const {src} *sp, int ss, {dst} *dp, int ds, NppiSize roi, NppiSize srcSize, "
           f"NppiPoint off, int mask, NppStreamContext ctx) "
           f"{{ (void)srcSize; (void)off; (void)mask; {body} }}")
    prefix = (f"num::filtderiv::run_deriv<{src}, {dst}, {c}, {a}, num::filtderiv::k{base}, "
              f"{'true' if masked else 'false'}, {'true' if border else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


GAUSS_RE = r"nppiFilter(GaussAdvancedBorder|GaussAdvanced|GaussBorder|Gauss)_(\w+)_(C1R|C3R|C4R|AC4R)_Ctx$"


def emit_gauss(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(GAUSS_RE, fn)
    if not m:
        return None
    fam, dt, lay = m.group(1), m.group(2), m.group(3)
    if dt not in DT or lay not in LAYOUT:
        return None
    src, dst = DT[dt]
    c, a = LAYOUT[lay]
    adv = fam.startswith("GaussAdvanced")
    border = "Border" in fam
    name = re.sub(r"\W", "_", fn)
    if adv:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, taps, ker, "
                f"NPP_BORDER_REPLICATE, ctx);" if border else
                f"return {fn}(sp, ss, dp, ds, roi, taps, ker, ctx);")
    else:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, (NppiMaskSize)mask, "
                f"NPP_BORDER_REPLICATE, ctx);" if border else
                f"return {fn}(sp, ss, dp, ds, roi, (NppiMaskSize)mask, ctx);")
    lam = (f"[](const {src} *sp, int ss, {dst} *dp, int ds, NppiSize roi, NppiSize srcSize, "
           f"NppiPoint off, int mask, int taps, const Npp32f *ker, NppStreamContext ctx) "
           f"{{ (void)srcSize; (void)off; (void)mask; (void)taps; (void)ker; {body} }}")
    prefix = (f"num::filtgauss::run_gauss<{src}, {dst}, {c}, {a}, "
              f"{'true' if adv else 'false'}, {'true' if border else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


WIN_RE = (r"nppiFilter(BoxBorderAdvanced|MedianBorder|Median|MinBorder|Min|MaxBorder|Max|"
          r"BoxBorder|Box)_(\w+)_(C1R|C3R|C4R|AC4R)_Ctx$")
WIN_OP = {"Box": "kBox", "BoxBorder": "kBox", "BoxBorderAdvanced": "kBox",
          "Min": "kMin", "MinBorder": "kMin", "Max": "kMax", "MaxBorder": "kMax",
          "Median": "kMedian", "MedianBorder": "kMedian"}


def emit_window(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(WIN_RE, fn)
    if not m:
        return None
    fam, dt, lay = m.group(1), m.group(2), m.group(3)
    if dt not in DT or lay not in LAYOUT:
        return None
    src, dst = DT[dt]
    c, a = LAYOUT[lay]
    op = WIN_OP[fam]
    adv = fam == "BoxBorderAdvanced"
    border = fam.endswith("Border")
    med = op == "kMedian"
    name = re.sub(r"\W", "_", fn)
    if adv:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, mask, anchor, "
                f"(NppiBorderType)mode, pbuf, ctx);")
    elif border:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, mask, anchor, "
                + ("pbuf, " if med else "")
                + f"(NppiBorderType)mode, ctx);")
    else:
        body = f"return {fn}(sp, ss, dp, ds, roi, mask, anchor, " + ("pbuf, " if med else "") + "ctx);"
    lam = (f"[](const {src} *sp, int ss, {dst} *dp, int ds, NppiSize roi, NppiSize srcSize, "
           f"NppiPoint off, NppiSize mask, NppiPoint anchor, int mode, Npp8u *pbuf, "
           f"NppStreamContext ctx) {{ (void)srcSize; (void)off; (void)mode; (void)pbuf; {body} }}")
    prefix = (f"num::filtwin::run_window<{src}, {c}, {a}, num::filtwin::{op}, "
              f"{'true' if border else 'false'}, {'true' if adv else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


GEN_RE = re.compile(r"nppiFilter(Border32f|Border|RowBorder32f|RowBorder|ColumnBorder32f|"
                    r"ColumnBorder|Row32f|Row|Column32f|Column|32f)?_(\w+)_"
                    r"(C1R|C2R|C3R|C4R|AC4R)_Ctx$")


def emit_generic(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = GEN_RE.match(fn)
    if not m:
        return None
    fam, dt, lay = m.group(1), m.group(2), m.group(3)
    if dt not in DT or lay not in LAYOUT:
        return None
    src, dst = DT[dt]
    c, a = LAYOUT[lay]
    border = bool(fam and "Border" in fam)
    g = 1 if (fam and "Row" in fam) else (2 if (fam and "Column" in fam) else 0)
    if dt == "64f" and fam in (None, "Row", "Column"):
        k = 2
        assert not border
    elif (fam and "32f" in fam) or dt == "32f":
        # 32f sources of the integer families also take a float kernel and no
        # divisor (measured against the NPP 13.4 headers).
        k = 1
    else:
        k = 0
    clamp = (not border) and dt == "32f" and lay == "C1R"
    kt = {0: "Npp32s", 1: "Npp32f", 2: "Npp64f"}[k]
    name = re.sub(r"\W", "_", fn)
    if g == 0:
        argmid = f"roi, (const {kt} *)ker, ksize, anchor"
    else:
        argmid = f"roi, (const {kt} *)ker, mask, anchor.{'x' if g == 1 else 'y'}"
    if k == 0:
        argmid += ", divisor"
    if border:
        call = f"return {fn}(sp, ss, srcSize, off, dp, ds, {argmid}, NPP_BORDER_REPLICATE, ctx);"
    else:
        call = f"return {fn}(sp, ss, dp, ds, {argmid}, ctx);"
    lam = (f"[](const {src} *sp, int ss, {dst} *dp, int ds, NppiSize roi, NppiSize srcSize, "
           f"NppiPoint off, const void *ker, NppiSize ksize, NppiPoint anchor, int mask, "
           f"int divisor, int border, NppStreamContext ctx) "
           f"{{ (void)srcSize; (void)off; (void)ksize; (void)anchor; (void)mask; (void)divisor; "
           f"(void)border; {call} }}")
    prefix = (f"num::filtgen::run_generic<{src}, {dst}, {c}, {a}, {g}, {k}, "
              f"{'true' if border else 'false'}, {'true' if clamp else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG))
            if r["module"] == "nppi_filtering_functions.h" and r["ctx"] == "1"]
    out: List[str] = []
    skipped: List[dict] = []
    for row in rows:
        text = (emit_fixed(row) or emit_deriv(row) or emit_gauss(row) or emit_window(row)
                or emit_generic(row))
        if text is None:
            skipped.append(dict(function=row["function_name"], family=row["family"],
                                reason="later filtering batch / unsupported"))
            continue
        out.append(text)
    header = ("// GENERATED by tools/gen_nppi_filter_tests.py -- DO NOT EDIT.\n"
              "// module: nppi_filtering_functions.h\n"
              "#include \"num_nppi_filter_fixed.h\"\n\n"
              "#include \"num_nppi_filter_deriv.h\"\n\n"
              "#include \"num_nppi_filter_gauss.h\"\n\n"
              "#include \"num_nppi_filter_window.h\"\n\n"
              "#include \"num_nppi_filter_generic.h\"\n\n")
    if args.emit:
        DEST.write_text(header + "\n".join(out))
        prev = {}
        if REPORT.exists():
            try:
                prev = json.loads(REPORT.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev["nppi_filtering_functions.h"] = dict(emitted=len(out), skipped=skipped)
        REPORT.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests to {DEST}; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
