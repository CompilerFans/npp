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
      "8s": ("Npp8s", "Npp8s"), "16f": ("Npp16f", "Npp16f")}
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


SUMW_RE = r"nppiSumWindow(Row|Column)(Border)?_(8u|16u|16s)32f_(C1R|C3R|C4R)_Ctx$"
SUMW_SRC = {"8u": "Npp8u", "16u": "Npp16u", "16s": "Npp16s"}


def emit_sumwindow(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(SUMW_RE, fn)
    if not m:
        return None
    orient, border, dt, lay = m.group(1), m.group(2) is not None, m.group(3), m.group(4)
    src = SUMW_SRC[dt]
    c = LAYOUT[lay][0]
    name = re.sub(r"\W", "_", fn)
    if border:
        body = (f"return {fn}(sp, ss, srcSize, off, dp, ds, roi, mask, anchor, "
                f"NPP_BORDER_REPLICATE, ctx);")
    else:
        body = f"return {fn}(sp, ss, dp, ds, roi, mask, anchor, ctx);"
    lam = (f"[](const {src} *sp, int ss, Npp32f *dp, int ds, NppiSize roi, NppiSize srcSize, "
           f"NppiPoint off, int mask, int anchor, int border, NppStreamContext ctx) "
           f"{{ (void)srcSize; (void)off; (void)border; {body} }}")
    prefix = (f"num::filtsum::run_sumwindow<{src}, {c}, {'true' if orient == 'Row' else 'false'}, "
              f"{'true' if border else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


GRAD_RE = r"nppiGradientVector(Sobel|Prewitt|Scharr)Border_(16s32f|16u32f|32f|8u16s)_(C1R|C3C1R)_Ctx$"
GRAD_OP = {"Sobel": "kSobel", "Prewitt": "kPrewitt", "Scharr": "kScharr"}
GRAD_DT = {"16s32f": ("Npp16s", "Npp32f"), "16u32f": ("Npp16u", "Npp32f"),
           "32f": ("Npp32f", "Npp32f"), "8u16s": ("Npp8u", "Npp16s")}


def emit_gradient(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(GRAD_RE, fn)
    if not m:
        return None
    op, dt, lay = m.group(1), m.group(2), m.group(3)
    src, dst = GRAD_DT[dt]
    srcc = 1 if lay == "C1R" else 3
    name = re.sub(r"\W", "_", fn)
    body = (f"return {fn}(sp, ss, srcSize, off, dX, sX, dY, sY, dM, sM, dA, sA, roi, "
            f"(NppiMaskSize)mask, (NppiNorm)norm, NPP_BORDER_REPLICATE, ctx);")
    lam = (f"[](const {src} *sp, int ss, NppiSize srcSize, NppiPoint off, {dst} *dX, int sX, "
           f"{dst} *dY, int sY, {dst} *dM, int sM, Npp32f *dA, int sA, NppiSize roi, int mask, "
           f"int norm, int border, NppStreamContext ctx) {{ (void)border; {body} }}")
    prefix = (f"num::filtgrad::run_gradient<{src}, {dst}, {srcc}, num::filtgrad::{GRAD_OP[op]}, "
              f"{'true' if op == 'Sobel' else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


TAIL_DT = {"8u": "Npp8u", "16u": "Npp16u", "32f": "Npp32f"}


def emit_tail(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    m = re.match(r"nppiFilterBilateralGaussBorder_(8u|16u|32f)_(C1R|C3R)_Ctx$", fn)
    if m:
        src = TAIL_DT[m.group(1)]
        c = 1 if m.group(2) == "C1R" else 3
        lam = (f"[](const {src} *sp, int ss, NppiSize srcSize, NppiPoint off, {src} *dp, int ds, "
               f"NppiSize roi, int radius, int step, float vs, float ps, int border, "
               f"NppStreamContext ctx) {{ return {fn}(sp, ss, srcSize, off, dp, ds, roi, radius, "
               f"step, vs, ps, (NppiBorderType)border, ctx); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::filttail::run_bilateral<{src}, {src}, {c}>("
                f"\n      \"{fn}\",\n      {lam});\n}}\n")
    if fn == "nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx":
        lam = (f"[](const Npp8u *sp, int ss, NppiSize srcSize, NppiPoint off, Npp8u *dp, int ds, "
               f"NppiSize roi, NppiSize mask, float delta, int gt, int le, int border, "
               f"NppStreamContext ctx) {{ return {fn}(sp, ss, srcSize, off, dp, ds, roi, mask, "
               f"delta, (Npp8u)gt, (Npp8u)le, (NppiBorderType)border, ctx); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::filttail::run_threshold_adaptive("
                f"\n      \"{fn}\",\n      {lam});\n}}\n")
    if fn == "nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx":
        lam = (f"[](const Npp8u *sp, int ss, NppiSize srcSize, NppiPoint off, Npp32f *dp, int ds, "
               f"NppiSize roi, int ft, int mask, int avg, float k, float scale, int border, "
               f"Npp8u *pbuf, NppStreamContext ctx) {{ return {fn}(sp, ss, srcSize, off, dp, ds, "
               f"roi, (NppiDifferentialKernel)ft, (NppiMaskSize)mask, (NppiMaskSize)avg, k, scale, "
               f"(NppiBorderType)border, pbuf, ctx); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::filttail::run_harris("
                f"\n      \"{fn}\",\n      {lam});\n}}\n")
    return None


def emit_wiener(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiFilterWienerBorder_(8u|16s|32f)_(C1R|C3R|C4R|AC4R)_Ctx$", fn)
    if not m:
        return None
    src = {"8u": "Npp8u", "16s": "Npp16s", "32f": "Npp32f"}[m.group(1)]
    c, a = LAYOUT[m.group(2)]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {src} *sp, int ss, NppiSize srcSize, NppiPoint off, {src} *dp, int ds, "
           f"NppiSize roi, NppiSize mask, NppiPoint anchor, const Npp32f *noise, int border, "
           f"NppStreamContext ctx) {{ return {fn}(sp, ss, srcSize, off, dp, ds, roi, mask, "
           f"anchor, const_cast<Npp32f *>(noise), (NppiBorderType)border, ctx); }}")
    prefix = f"num::filtwien::run_wiener<{src}, {c}, {a}>"
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


UNSHARP_DT = {"8u": ("Npp8u", 0), "16u": ("Npp16u", 1), "16s": ("Npp16s", 2), "32f": ("Npp32f", 3)}


def emit_unsharp(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiFilterUnsharpBorder_(8u|16u|16s|32f)_(C1R|C3R|C4R|AC4R)_Ctx$", fn)
    if not m:
        return None
    src, dtcode = UNSHARP_DT[m.group(1)]
    c, a = LAYOUT[m.group(2)]
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {src} *sp, int ss, NppiPoint off, {src} *dp, int ds, NppiSize roi, float radius, "
           f"float sigma, float weight, float threshold, int border, Npp8u *pbuf, NppStreamContext ctx) "
           f"{{ return {fn}(sp, ss, off, dp, ds, roi, radius, sigma, weight, threshold, "
           f"(NppiBorderType)border, pbuf, ctx); }}")
    prefix = f"num::filtunsharp::run_unsharp<{src}, {src}, {c}, {a}, {dtcode}>"
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"


def emit_pyramid(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiFilterGaussPyramidLayer(Down|Up)Border_(8u|16u|32f)_(C1R|C3R)_Ctx$", fn)
    if not m:
        return None
    down, dt, lay = m.group(1) == "Down", m.group(2), m.group(3)
    src = {"8u": "Npp8u", "16u": "Npp16u", "32f": "Npp32f"}[dt]
    c = 1 if lay == "C1R" else 3
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {src} *sp, int ss, NppiSize srcSize, NppiPoint off, {src} *dp, int ds, "
           f"NppiSize roi, float rate, int taps, const Npp32f *ker, int border, NppStreamContext ctx) "
           f"{{ return {fn}(sp, ss, srcSize, off, dp, ds, roi, rate, taps, ker, "
           f"(NppiBorderType)border, ctx); }}")
    prefix = f"num::filtpyr::run_pyramid<{src}, {src}, {c}, {'true' if down else 'false'}>"
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"



def emit_hog(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiHistogramOfGradientsBorder_(8u32f|16u32f|16s32f|32f)_(C1R|C3R)_Ctx$", fn)
    if not m:
        return None
    src = {"8u32f": "Npp8u", "16u32f": "Npp16u", "16s32f": "Npp16s", "32f": "Npp32f"}[m.group(1)]
    c = 1 if m.group(2) == "C1R" else 3
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {src} *sp, int ss, NppiSize srcSize, NppiPoint off, const NppiPoint *locs, "
           f"int nLoc, Npp32f *dp, NppiSize roi, NppiHOGConfig cfg, Npp8u *scratch, int border, "
           f"NppStreamContext ctx) {{ return {fn}(sp, ss, srcSize, off, locs, nLoc, dp, roi, cfg, "
           f"scratch, (NppiBorderType)border, ctx); }}")
    prefix = f"num::filthog::run_hog<{src}, {c}>"
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"



MBUF_DT = {"8u": 0, "16u": 1, "16s": 2, "32f": 3}


def emit_median_bufsize(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiFilterMedian(Border)?GetBufferSize_(8u|16u|16s|32f)_(C1R|C3R|C4R|AC4R)_Ctx$", fn)
    if not m:
        return None
    border = m.group(1) is not None
    dt = MBUF_DT[m.group(2)]
    c, a = LAYOUT[m.group(3)]
    name = re.sub(r"\W", "_", fn)
    if border:
        body = (f"return {fn}(roi, mask, size, (NppiBorderType)border, ctx);")
    else:
        body = f"(void)border; return {fn}(roi, mask, size, ctx);"
    lam = (f"[](NppiSize roi, NppiSize mask, Npp32u *size, int border, NppStreamContext ctx) "
           f"{{ {body} }}")
    prefix = (f"num::mediabufdrv::run_median_bufsize<{dt}, {c}, {a}, "
              f"{'true' if border else 'false'}>")
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"



PBA_DT = {"8u": "Npp8u", "8s": "Npp8s", "16u": "Npp16u", "16s": "Npp16s", "32f": "Npp32f", "64f": "Npp64f"}


def emit_distance_transform(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiDistanceTransform(Abs)?PBA_(8u|8s|16u|16s|32f|64f)(16u|32f|64f)?_C1R_Ctx$", fn)
    if not m:
        return None
    src = PBA_DT[m.group(2)]
    dst = PBA_DT[m.group(3) or m.group(2)]
    name = re.sub(r"\W", "_", fn)
    has_abuf = fn.endswith("64f_C1R_Ctx")
    if has_abuf:
        body = (f"return {fn}(sp, ss, minSite, maxSite, nullptr, 0, nullptr, 0, nullptr, 0, dt, dts, roi, "
                f"buf, abuf, ctx);")
    else:
        body = (f"return {fn}(sp, ss, minSite, maxSite, nullptr, 0, nullptr, 0, nullptr, 0, dt, dts, roi, "
                f"buf, ctx);")
    lam = (f"[]({src} *sp, int ss, {src} minSite, {src} maxSite, Npp16s *dv, int dsv, Npp16s *di, "
           f"int dsi, Npp16s *dm, int dsm, {dst} *dt, int dts, NppiSize roi, Npp8u *buf, "
           f"Npp8u *abuf, NppStreamContext ctx) {{ (void)abuf; {body} }}")
    prefix = f"num::distxform::run_pba<{src}, {dst}>"
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"



FLOOD_DT = {"8u": "Npp8u", "16u": "Npp16u", "32u": "Npp32u"}
FLOOD_VAR = {"": "kPlain", "Boundary": "kBoundary", "Range": "kRange",
             "Gradient": "kGradient", "RangeBoundary": "kRangeBoundary",
             "GradientBoundary": "kGradientBoundary"}


def emit_floodfill(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiFloodFill(Boundary|GradientBoundary|RangeBoundary|Gradient|Range)?_"
                 r"(8u|16u|32u)_(C1IR|C3IR)_Ctx$", fn)
    if not m:
        return None
    var = FLOOD_VAR[m.group(1) or ""]
    src = FLOOD_DT[m.group(2)]
    c = 1 if m.group(3) == "C1IR" else 3
    name = re.sub(r"\W", "_", fn)
    scalar = c == 1
    a = (lambda n: f"{n}[0]") if scalar else (lambda n: f"const_cast<{src} *>({n})")
    kind = var.replace("k", "")
    if kind in ("Plain", "Boundary"):
        args = f"seed, {a('nv')}"
        if kind == "Boundary":
            args += f", {a('bv')}"
        args += ", (NppiNorm)norm, roi, reg, buf, ctx"
    else:
        args = f"seed, {a('mn')}, {a('mx')}, {a('nv')}"
        if kind.endswith("Boundary"):
            args += f", {a('bv')}"
        args += ", (NppiNorm)norm, roi, reg, buf, ctx"
    body = f"return {fn}(p, step, {args});"
    lam = (f"[]({src} *p, int step, NppiPoint seed, const {src} *nv, const {src} *bv, "
           f"const {src} *mn, const {src} *mx, int norm, NppiSize roi, NppiConnectedRegion *reg, "
           f"Npp8u *buf, NppStreamContext ctx) {{ {body} }}")
    prefix = f"num::floodfill::run_flood<{src}, {c}, num::floodfill::{var}>"
    return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"



def emit_label_markers(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiLabelMarkersUF_(8u32u|16u32u|32u)_C1R_Ctx$", fn)
    if m:
        src = {"8u32u": "Npp8u", "16u32u": "Npp16u", "32u": "Npp32u"}[m.group(1)]
        name = re.sub(r"\W", "_", fn)
        lam = (f"[]({src} *sp, int ss, Npp32u *dp, int ds, NppiSize roi, int norm, Npp8u *buf, "
               f"NppStreamContext ctx) {{ return {fn}(sp, ss, dp, ds, roi, (NppiNorm)norm, buf, ctx); }}")
        prefix = f"num::labelmarkers::run_label<{src}>"
        return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"
    m = re.match(r"nppiLabelMarkersUFBatch_(8u32u|16u32u|32u)_C1R(_Advanced)?_Ctx$", fn)
    if m:
        src = {"8u32u": "Npp8u", "16u32u": "Npp16u", "32u": "Npp32u"}[m.group(1)]
        advanced = m.group(2) is not None
        name = re.sub(r"\W", "_", fn)
        lam = (f"[](const NppiImageDescriptor *sp, NppiImageDescriptor *dp, int nb, NppiSize roi, "
               f"int norm, NppStreamContext ctx) {{ return {fn}(sp, dp, nb, roi, (NppiNorm)norm, ctx); }}")
        adv = "true" if advanced else "false"
        prefix = f"num::labelmarkers::run_label_batch<{src}, {adv}>"
        return f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n"
    return None


def emit_hough(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    if fn == "nppiFilterHoughLine_8u32f_C1R_Ctx":
        lam = (f"[](const Npp8u *sp, int ss, NppiSize roi, NppPointPolar delta, int thr, "
               f"NppPointPolar *lines, int maxLines, int *count, Npp8u *buf, NppStreamContext ctx) {{ "
               f"return {fn}(sp, ss, roi, delta, thr, lines, maxLines, count, buf, ctx); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::hough::run_hough(\n      \"{fn}\",\n      {lam});\n}}\n")
    if fn == "nppiFilterHoughLineRegion_8u32f_C1R_Ctx":
        lam = (f"[](const Npp8u *sp, int ss, NppiSize roi, NppPointPolar delta, int thr, "
               f"NppPointPolar *lines, NppPointPolar *dstRoi, int maxLines, int *count, Npp8u *buf, "
               f"NppStreamContext ctx) {{ return {fn}(sp, ss, roi, delta, thr, lines, dstRoi, maxLines, "
               f"count, buf, ctx); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::hough::run_hough_region(\n      \"{fn}\",\n      {lam});\n}}\n")
    return None


def emit_canny(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if fn != "nppiFilterCannyBorder_8u_C1R_Ctx":
        return None
    lam = (f"[](const Npp8u *sp, int ss, NppiSize ssSize, NppiPoint off, Npp8u *dp, int ds, NppiSize roi, "
           f"int ftype, int mask, Npp16s lo, Npp16s hi, int norm, int border, Npp8u *buf, "
           f"NppStreamContext ctx) {{ return {fn}(sp, ss, ssSize, off, dp, ds, roi, "
           f"(NppiDifferentialKernel)ftype, (NppiMaskSize)mask, lo, hi, (NppiNorm)norm, "
           f"(NppiBorderType)border, buf, ctx); }}")
    name = re.sub(r"\W", "_", fn)
    return f"TEST(NumGen, {name}) {{\n  num::canny::run_canny(\n      \"{fn}\",\n      {lam});\n}}\n"


def emit_compress_labels(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if fn == "nppiCompressMarkerLabelsUF_32u_C1IR_Ctx":
        lam = (f"[](Npp32u *p, int step, NppiSize roi, int start, int *newNumber, Npp8u *buf, "
               f"NppStreamContext ctx) {{ return {fn}(p, step, roi, start, newNumber, buf, ctx); }}")
        return ("TEST(NumGen, " + re.sub(r"\W", "_", fn) + ") {\n  num::compresslabels::run_compress(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")
    m = re.match(r"nppiCompressMarkerLabelsUFBatch_32u_C1IR(_Advanced)?_Ctx$", fn)
    if m:
        advanced = m.group(1) is not None
        adv = "true" if advanced else "false"
        lam = (f"[](NppiImageDescriptor *p, NppiBufferDescriptor *b, unsigned int *n, int nb, "
               f"NppiSize roi, int bufsz, NppStreamContext ctx) {{ return {fn}(p, b, n, nb, roi, bufsz, ctx); }}")
        return ("TEST(NumGen, " + re.sub(r"\W", "_", fn) + ") {\n"
                f"  num::compresslabels::run_compress_batch<{adv}>(\n      \"{fn}\",\n      {lam});\n}}\n")
    return None


def emit_signed_dt(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppiSignedDistanceTransform(Abs)?PBA_(32f64f|32f|64f)_C1R_Ctx$", fn)
    if not m:
        return None
    absf = m.group(1) is not None
    dt = m.group(2)
    src = "Npp32f" if dt.startswith("32f") else "Npp64f"
    out = "Npp32f" if dt == "32f" else "Npp64f"
    name = re.sub(r"\W", "_", fn)
    has_abuf = dt != "32f"
    abuf_arg = ", abuf" if has_abuf else ""
    man = "(Npp16u *)man" if absf else "(Npp16s *)man"
    lam = (f"[]({src} *sp, int ss, {src} cutoff, {out} sx, {out} sy, Npp16s *vor, int vs, "
           f"Npp16s *idx, int ix, void *man, int ms, {out} *dt, int dts, "
           f"NppiSize roi, Npp8u *buf, Npp8u *abuf, NppStreamContext ctx) {{ (void)abuf; "
           f"return {fn}(sp, ss, cutoff, sx, sy, vor, vs, idx, ix, {man}, ms, dt, dts, roi, "
           f"buf{abuf_arg}, ctx); }}")
    # Value model per entry point (see the header's Model enum).  Only the Abs
    # 32f64f entry point receives the antialiasing workspace (it is mandatory
    # there and its value rule is reproducible for closed regions); the other
    # flavours are exercised in their documented non-AA mode, where the
    # reference emits the folded signed / folded magnitude transforms.
    if absf and dt == "32f64f":
        model = "num::signeddt::kAaAbs"
    elif absf and dt == "32f":
        model = "num::signeddt::kAbsFolded"
    else:
        model = "num::signeddt::kSignedFolded"
    prefix = f"num::signeddt::run_signed_dt<{src}, {out}, {model}>"
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
                or emit_generic(row) or emit_sumwindow(row) or emit_gradient(row)
                or emit_tail(row) or emit_wiener(row) or emit_unsharp(row)
                or emit_pyramid(row) or emit_hog(row) or emit_median_bufsize(row)
                or emit_distance_transform(row) or emit_floodfill(row)
                or emit_signed_dt(row) or emit_label_markers(row) or emit_compress_labels(row)
                or emit_canny(row) or emit_hough(row))
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
              "#include \"num_nppi_filter_generic.h\"\n\n"
              "#include \"num_nppi_filter_sumwindow.h\"\n\n"
              "#include \"num_nppi_filter_gradient.h\"\n\n"
              "#include \"num_nppi_filter_tail.h\"\n\n"
              "#include \"num_nppi_filter_wiener.h\"\n\n"
              "#include \"num_nppi_filter_unsharp.h\"\n\n"
              "#include \"num_nppi_filter_pyramid.h\"\n\n"
              "#include \"num_nppi_filter_hog.h\"\n\n"
              "#include \"num_nppi_median_bufsize.h\"\n\n"
              "#include \"num_nppi_distance_transform.h\"\n\n"
              "#include \"num_nppi_floodfill.h\"\n\n"
              "#include \"num_nppi_signed_distance_transform.h\"\n\n"
              "#include \"num_nppi_label_markers.h\"\n\n"
              "#include \"num_nppi_compress_labels.h\"\n\n"
              "#include \"num_nppi_canny.h\"\n\n"
              "#include \"num_nppi_hough.h\"\n\n")
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
