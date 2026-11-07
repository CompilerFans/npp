#!/usr/bin/env python3
"""Generate CPU-golden tests for the nppi color module.

Batches: point/linear/cubic LUT, palette and trilinear LUT, and the
single-image ColorTwist families.  References live in
num_nppi_lut.h and num_nppi_color_twist.h.
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



CTW = {"8u": "Npp8u", "8s": "Npp8s", "16u": "Npp16u", "16s": "Npp16s",
       "16f": "Npp16f", "32f": "Npp32f"}
TW_LAYOUT = {"C1": (1, 1), "C2": (2, 2), "C3": (3, 3), "C4": (4, 4), "AC4": (4, 3)}


def emit_twist(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if "Batch" in fn or not fn.startswith("nppiColorTwist"):
        return None
    name = re.sub(r"\W", "_", fn)
    m = re.search(r"_(8u|8s|16u|16s|16f|32f)C?_", fn)
    if m is None:
        return None
    S = CTW.get(m.group(1))
    if S is None:
        return None
    if "32fC" in fn:
        mm = re.search(r"_(C4)(I?)R_Ctx$", fn)
        if mm is None:
            return None
        inplace = mm.group(2) == "I"
        if inplace:
            lam = (f"[]({S} *a, int sa, NppiSize r, const float *tw, const float *ac, "
                   f"NppStreamContext c) "
                   f"{{ (void)sa; (void)r; (void)tw; (void)ac; (void)c; "
                   f"return {fn}(a, sa, r, (const Npp32f (*)[4])tw, ac, c); }}")
            prefix = f"num::twist::run_twist_c<{S}, true>"
        else:
            lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, const float *tw, "
                   f"const float *ac, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)tw; (void)ac; (void)c; "
                   f"return {fn}(a, sa, d, ds, r, (const Npp32f (*)[4])tw, ac, c); }}")
            prefix = f"num::twist::run_twist_c<{S}, false>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    mm = re.search(r"_(C1|C2|C3|C4|AC4|P3|IP3)(I?)R_Ctx$", fn)
    if mm is None:
        return None
    tok, inplace = mm.group(1), mm.group(2) == "I"
    if tok in ("P3", "IP3"):
        ip = tok == "IP3" or inplace
        if ip:
            lam = (f"[]({S} **sp, int ss, NppiSize r, const float *tw, NppStreamContext c) "
                   f"{{ (void)ss; (void)r; (void)tw; (void)c; "
                   f"return {fn}(sp, ss, r, (const Npp32f (*)[4])tw, c); }}")
            prefix = f"num::twist::run_twist_planar<{S}, true>"
        else:
            lam = (f"[](const {S} **sp, int ss, {S} **dp, int ds, NppiSize r, const float *tw, "
                   f"NppStreamContext c) "
                   f"{{ (void)ss; (void)ds; (void)r; (void)tw; (void)c; "
                   f"return {fn}(sp, ss, dp, ds, r, (const Npp32f (*)[4])tw, c); }}")
            prefix = f"num::twist::run_twist_planar<{S}, false>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    if tok not in TW_LAYOUT:
        return None
    C, A = TW_LAYOUT[tok]
    if inplace:
        lam = (f"[]({S} *a, int sa, NppiSize r, const float *tw, NppStreamContext c) "
               f"{{ (void)sa; (void)r; (void)tw; (void)c; "
               f"return {fn}(a, sa, r, (const Npp32f (*)[4])tw, c); }}")
        prefix = f"num::twist::run_twist<{S}, {C}, {A}, true>"
    else:
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, const float *tw, "
               f"NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)tw; (void)c; "
               f"return {fn}(a, sa, d, ds, r, (const Npp32f (*)[4])tw, c); }}")
        prefix = f"num::twist::run_twist<{S}, {C}, {A}, false>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



def emit_gamma(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in ("nppiGammaFwd", "nppiGammaInv"):
        return None
    fwd = fam == "nppiGammaFwd"
    name = re.sub(r"\W", "_", fn)
    m = re.match(r"nppiGamma(?:Fwd|Inv)_8u_(C3|AC4|P3|IP3)(I?)R_Ctx$", fn)
    if not m:
        return None
    tok, ip = m.group(1), m.group(2) == "I"
    inplace = ip or tok == "IP3"
    if tok in ("P3", "IP3"):
        layout = 2
        if inplace:
            lam = (f"[](Npp8u **sp, int ss, NppiSize r, NppStreamContext c) "
                   f"{{ (void)ss; (void)r; (void)c; return {fn}(sp, ss, r, c); }}")
        else:
            lam = (f"[](const Npp8u **sp, int ss, Npp8u **dp, int ds, NppiSize r, NppStreamContext c) "
                   f"{{ (void)ss; (void)ds; (void)r; (void)c; return {fn}(sp, ss, dp, ds, r, c); }}")
    else:
        layout = 0 if tok == "C3" else 1
        if inplace:
            lam = (f"[](Npp8u *a, int st, NppiSize r, NppStreamContext c) "
                   f"{{ (void)st; (void)r; (void)c; return {fn}(a, st, r, c); }}")
        else:
            lam = (f"[](const Npp8u *a, int sa, Npp8u *d, int ds, NppiSize r, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)c; return {fn}(a, sa, d, ds, r, c); }}")
    prefix = f"num::cvt::run_gamma<{layout}, {str(inplace).lower()}, {str(fwd).lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_gray(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in ("nppiRGBToGray", "nppiColorToGray"):
        return None
    m = re.match(r"nppi(?:RGBToGray|ColorToGray)_(8u|16u|16s|32f)_(C3|C4|AC4)C1R_Ctx$", fn)
    if not m:
        return None
    S = CTW.get(m.group(1))
    tok = m.group(2)
    layout = {"C3": 0, "C4": 1, "AC4": 2}[tok]
    name = re.sub(r"\W", "_", fn)
    if fam == "nppiRGBToGray":
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, float *co, int n, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)co; (void)n; (void)c; "
               f"return {fn}(a, sa, d, ds, r, c); }}")
        ncoef = 3
    else:
        lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, float *co, int n, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)n; (void)c; "
               f"return {fn}(a, sa, d, ds, r, (const Npp32f *)co, c); }}")
        ncoef = 3 if tok in ("C3", "AC4") else 4
    prefix = f"num::cvt::run_color_to_gray<{S}, {layout}, {str(fam == 'nppiRGBToGray').lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam}, {ncoef});\n}}\n")


def emit_gradient(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if row["family"] != "nppiGradientColorToGray":
        return None
    m = re.match(r"nppiGradientColorToGray_(8u|16u|16s|32f)_C3C1R_Ctx$", fn)
    if not m:
        return None
    S = CTW.get(m.group(1))
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const {S} *a, int sa, {S} *d, int ds, NppiSize r, int norm, NppStreamContext c) "
           f"{{ (void)sa; (void)ds; (void)r; (void)norm; (void)c; "
           f"return {fn}(a, sa, d, ds, r, (NppiNorm)norm, c); }}")
    return (f"TEST(NumGen, {name}) {{\n  num::cvt::run_gradient_gray<{S}>(\n      \"{fn}\",\n"
            f"      {lam});\n}}\n")



def emit_twist_batch(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if not fn.startswith("nppiColorTwistBatch"):
        return None
    m = re.search(r"_(8u|16f|32f)C?_", fn)
    if not m:
        return None
    S = CTW.get(m.group(1))
    if S is None:
        return None
    mm = re.search(r"_(C1|C3|C4|AC4)(I?)R_Ctx$", fn)
    if not mm:
        return None
    tok, inplace = mm.group(1), mm.group(2) == "I"
    C, A = TW_LAYOUT[tok]
    consts = "32fC" in fn
    name = re.sub(r"\W", "_", fn)
    lam = ("[](Npp32f mn, Npp32f mx, NppiSize r, NppiColorTwistBatchCXR *l, int n, NppStreamContext c) "
           f"{{ (void)mn; (void)mx; (void)r; (void)l; (void)n; (void)c; "
           f"return {fn}(mn, mx, r, l, n, c); }}")
    prefix = (f"num::twist::run_twist_batch<{S}, {C}, {A}, {str(inplace).lower()}, "
              f"{str(consts).lower()}>")
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



HLS_KIND = {"nppiRGBToHLS": "RGB2HLS", "nppiHLSToRGB": "HLS2RGB",
            "nppiRGBToHSV": "RGB2HSV", "nppiHSVToRGB": "HSV2RGB"}


def emit_hls(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in HLS_KIND:
        return None
    m = re.match(r"nppi(?:RGBToHLS|HLSToRGB|RGBToHSV|HSVToRGB)_8u_(C3|AC4)R_Ctx$", fn)
    if not m:
        return None
    ac4 = m.group(1) == "AC4"
    name = re.sub(r"\W", "_", fn)
    lam = (f"[](const Npp8u *a, int sa, Npp8u *d, int ds, NppiSize r, NppStreamContext c) "
           f"{{ (void)sa; (void)ds; (void)r; (void)c; return {fn}(a, sa, d, ds, r, c); }}")
    prefix = f"num::hls::run_hls<num::hls::Kind::{HLS_KIND[fam]}, {str(ac4).lower()}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



YUV_CONV = {"nppiRGBToYUV": ("RGB2YUV", False), "nppiBGRToYUV": ("RGB2YUV", True),
            "nppiYUVToRGB": ("YUV2RGB", False), "nppiYUVToBGR": ("YUV2RGB", True),
            "nppiRGBToYCbCr": ("RGB2YCbCr", False), "nppiBGRToYCbCr": ("RGB2YCbCr", True),
            "nppiYCbCrToRGB": ("YCbCr2RGB", False), "nppiYCbCrToBGR": ("YCbCr2RGB", True)}
YUV_SL = {"C3R": "C3", "AC4R": "AC4", "P3R": "P3"}
YUV_DL = {"C3R": "C3", "AC4R": "AC4", "P3R": "P3", "P4R": "P4", "P3C3R": "C3",
          "C3P3R": "P3", "AC4P4R": "P4", "AC4P3R": "P3", "P3C4R": "C4A"}


def _yuv_layouts(fn: str):
    # Return (src_layout, dst_layout) tokens, or None.
    for tok, dl in (("AC4P4R", ("AC4", "P4")), ("AC4P3R", ("AC4", "P3")),
                    ("C3P3R", ("C3", "P3")), ("P3C4R", ("P3", "C4A")),
                    ("P3C3R", ("P3", "C3")), ("P3R", ("P3", "P3")),
                    ("AC4R", ("AC4", "AC4")), ("C3R", ("C3", "C3"))):
        if fn.endswith("_" + tok + "_Ctx"):
            return dl
    return None


def emit_yuv(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in YUV_CONV:
        return None
    lay = _yuv_layouts(fn)
    if lay is None:
        return None
    sl, dl = lay
    conv, bgr = YUV_CONV[fam]
    name = re.sub(r"\W", "_", fn)
    lam = ("[](const Npp8u *const *sp, const Npp8u *ps, int ss, Npp8u *const *dp, Npp8u *pd, int ds, "
           "NppiSize r, int alpha, NppStreamContext c) "
           "{ (void)sp; (void)ps; (void)ss; (void)dp; (void)pd; (void)ds; (void)r; (void)alpha; (void)c; "
           f"return {fn}(")
    if sl == "P3":
        if dl in ("P3", "P4"):
            lam += "sp, ss, const_cast<Npp8u **>(dp), ds, r, c); }"
        elif dl == "C4A":
            lam += "sp, ss, pd, ds, r, (Npp8u)alpha, c); }"
        else:
            lam += "sp, ss, pd, ds, r, c); }"
    else:
        if dl in ("P3", "P4"):
            lam += "ps, ss, const_cast<Npp8u **>(dp), ds, r, c); }"
        else:
            lam += "ps, ss, pd, ds, r, c); }"
    prefix = (f"num::yuv::run_yuv<num::yuv::Conv::{conv}, num::yuv::SL::{sl}, "
              f"num::yuv::DL::{dl}, {str(bgr).lower()}>")
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



HLS_LAYOUTS = {
    "AC4P4R": (False, 4, True, 4), "AC4R": (False, 4, False, 4),
    "AP4C4R": (True, 4, False, 4), "AP4R": (True, 4, True, 4),
    "C3P3R": (False, 3, True, 3), "P3C3R": (True, 3, False, 3),
    "P3R": (True, 3, True, 3),
}


def emit_hls_layout(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in ("nppiBGRToHLS", "nppiHLSToBGR"):
        return None
    m = re.match(r"nppi(?:BGRToHLS|HLSToBGR)_8u_(\w+?)_Ctx$", fn)
    if not m:
        return None
    tok = m.group(1)
    if tok not in HLS_LAYOUTS:
        return None
    spl, sc, dpl, dc = HLS_LAYOUTS[tok]
    kind = "BGR2HLS" if fam == "nppiBGRToHLS" else "HLS2BGR"
    name = re.sub(r"\W", "_", fn)
    lam = ("[](const Npp8u *const *sp, const Npp8u *ps, int ss, Npp8u *const *dp, Npp8u *pd, int ds, "
           "NppiSize r, NppStreamContext c) "
           "{ (void)sp; (void)ps; (void)ss; (void)dp; (void)pd; (void)ds; (void)r; (void)c; "
           f"return {fn}(")
    if spl and dpl:
        lam += "sp, ss, const_cast<Npp8u **>(dp), ds, r, c); }"
    elif spl:
        lam += "sp, ss, pd, ds, r, c); }"
    elif dpl:
        lam += "ps, ss, const_cast<Npp8u **>(dp), ds, r, c); }"
    else:
        lam += "ps, ss, pd, ds, r, c); }"
    prefix = (f"num::hls::run_hls_layout<num::hls::HKind::{kind}, {str(spl).lower()}, {sc}, "
              f"{str(dpl).lower()}, {dc}>")
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



CS_KIND = {"nppiRGBToXYZ": "RGB2XYZ", "nppiXYZToRGB": "XYZ2RGB",
           "nppiRGBToLUV": "RGB2LUV", "nppiLUVToRGB": "LUV2RGB",
           "nppiRGBToYCC": "RGB2YCC", "nppiYCCToRGB": "YCC2RGB"}


def emit_cs(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam in CS_KIND:
        m = re.search(r"_(C3|AC4)R_Ctx$", fn)
        if not m:
            return None
        ac4 = m.group(1) == "AC4"
        name = re.sub(r"\W", "_", fn)
        lam = (f"[](const Npp8u *a, int sa, Npp8u *d, int ds, NppiSize r, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)c; return {fn}(a, sa, d, ds, r, c); }}")
        prefix = f"num::cs::run_cs<num::cs::CS::{CS_KIND[fam]}, {str(ac4).lower()}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    if fam in ("nppiBGRToLab", "nppiLabToBGR"):
        m = re.search(r"_8u_C3R_Ctx$", fn)
        if not m:
            return None
        kind = "BGR2Lab" if fam == "nppiBGRToLab" else "Lab2BGR"
        name = re.sub(r"\W", "_", fn)
        lam = (f"[](const Npp8u *a, int sa, Npp8u *d, int ds, NppiSize r, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)c; return {fn}(a, sa, d, ds, r, c); }}")
        prefix = f"num::cs::run_lab<num::cs::LabDir::{kind}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    return None



def emit_cfa(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in ("nppiCFAToRGB", "nppiCFAToRGBA"):
        return None
    m = re.match(r"nppiCFAToRGBA?_(8u|16u|32u)_C1(C3|AC4)R_Ctx$", fn)
    if not m:
        return None
    S = {"8u": "Npp8u", "16u": "Npp16u", "32u": "Npp32u"}[m.group(1)]
    out_c = 3 if m.group(2) == "C3" else 4
    name = re.sub(r"\W", "_", fn)
    if fam == "nppiCFAToRGB":
        lam = (f"[](const {S} *a, int ss, NppiSize sz, NppiRect roi, {S} *d, int ds, int grid, "
               f"unsigned alpha, NppStreamContext c) "
               f"{{ (void)alpha; (void)ss; (void)sz; (void)roi; (void)ds; (void)grid; (void)c; "
               f"return {fn}(a, ss, sz, roi, d, ds, (NppiBayerGridPosition)grid, "
               f"(NppiInterpolationMode)NPPI_INTER_UNDEFINED, c); }}")
    else:
        lam = (f"[](const {S} *a, int ss, NppiSize sz, NppiRect roi, {S} *d, int ds, int grid, "
               f"unsigned alpha, NppStreamContext c) "
               f"{{ (void)ss; (void)sz; (void)roi; (void)ds; (void)grid; (void)c; "
               f"return {fn}(a, ss, sz, roi, d, ds, (NppiBayerGridPosition)grid, "
               f"(NppiInterpolationMode)NPPI_INTER_UNDEFINED, ({S})alpha, c); }}")
    prefix = f"num::cfa::run_cfa<{S}, {out_c}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")



YUV_FAMILIES = {"nppiYUV420ToRGB", "nppiYUV420ToBGR", "nppiYUV422ToRGB",
                "nppiRGBToYUV420", "nppiRGBToYUV422", "nppiBGRToYUV420"}


YCBCR420_TABLE = {
    # function: (kind, srcPp/dstPp, bgr, swap, aval)
    "nppiRGBToYCbCr420_8u_C3P3R_Ctx": ("Fwd420", 3, 0, 0, 0),
    "nppiRGBToYCrCb420_8u_AC4P3R_Ctx": ("Fwd420", 4, 0, 1, 0),
    "nppiBGRToYCrCb420_8u_C3P3R_Ctx": ("Fwd420", 3, 1, 1, 0),
    "nppiBGRToYCrCb420_8u_AC4P3R_Ctx": ("Fwd420", 4, 1, 1, 0),
    "nppiRGBToYCbCr411_8u_C3P3R_Ctx": ("Fwd411", 3, 0, 0, 0),
    "nppiRGBToYCbCr411_8u_AC4P3R_Ctx": ("Fwd411", 4, 0, 0, 0),
    "nppiBGRToYCbCr411_8u_C3P3R_Ctx": ("Fwd411", 3, 1, 0, 0),
    "nppiBGRToYCbCr411_8u_AC4P3R_Ctx": ("Fwd411", 4, 1, 0, 0),
    "nppiYCbCr420ToRGB_8u_P3C3R_Ctx": ("Inv420", 3, 0, 0, 0),
    "nppiYCbCr420ToBGR_8u_P3C3R_Ctx": ("Inv420", 3, 1, 0, 0),
    "nppiYCbCr420ToBGR_8u_P3C4R_Ctx": ("Inv420", 4, 1, 0, 0x37),
    "nppiYCrCb420ToRGB_8u_P3C4R_Ctx": ("Inv420", 4, 0, 1, 0x37),
    "nppiYCbCr411ToRGB_8u_P3C3R_Ctx": ("Inv411", 3, 0, 0, 0),
    "nppiYCbCr411ToRGB_8u_P3C4R_Ctx": ("Inv411", 4, 0, 0, 0x37),
    "nppiYCbCr411ToBGR_8u_P3C3R_Ctx": ("Inv411", 3, 1, 0, 0),
    "nppiYCbCr411ToBGR_8u_P3C4R_Ctx": ("Inv411", 4, 1, 0, 0x37),
    "nppiBGRToYCbCr420_8u_C3P3R_Ctx": ("Fwd420", 3, 1, 0, 0),
    "nppiBGRToYCbCr420_8u_AC4P3R_Ctx": ("Fwd420", 4, 1, 0, 0),
    "nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx": ("Fwd420m1", 3, 1, 0, 0),
    "nppiBGRToYCbCr420_709CSC_8u_AC4P3R_Ctx": ("Fwd420m1", 4, 1, 0, 0),
    "nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx": ("Fwd420m2", 4, 1, 0, 0),
    "nppiBGRToYCrCb420_709CSC_8u_C3P3R_Ctx": ("Fwd420m1", 3, 1, 1, 0),
    "nppiBGRToYCrCb420_709CSC_8u_AC4P3R_Ctx": ("Fwd420m1", 4, 1, 1, 0),
    "nppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx": ("Inv420m1", 3, 1, 0, 0),
    "nppiYCbCr420ToBGR_709HDTV_8u_P3C4R_Ctx": ("Inv420m2", 4, 1, 0, 0x37),
    "nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx": ("Inv444m1", 3, 1, 0, 0),
    "nppiYCbCrToBGR_709CSC_8u_P3C4R_Ctx": ("Inv444m1", 4, 1, 0, 0x37),
}


def emit_ycbcr420(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if fn not in YCBCR420_TABLE:
        return None
    kind, pp, bgr, swap, aval = YCBCR420_TABLE[fn]
    mat = {"Fwd420m1": 1, "Fwd420m2": 2, "Inv420m1": 1, "Inv420m2": 2, "Inv444m1": 1}.get(kind, 0)
    kind = kind.replace("m1", "").replace("m2", "")
    name = re.sub(r"\W", "_", fn)
    sig = ("[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
           "int ds, int *dsa, NppiSize r, NppStreamContext c) ")
    voids = "(void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)c; "
    fwd = kind.startswith("Fwd")
    step = "ssa[0]" if str(kind).startswith("Inv444") else "ssa"
    if fwd:
        body = f"return {fn}(ps, ss, const_cast<Npp8u **>(dp), dsa, r, c);"
        kind_arg = f"num::ycbcr420::Kind::{kind}, {pp}, 0"
    else:
        if aval:
            body = f"return {fn}(sp, {step}, pd, ds, r, (Npp8u)0x37, c);"
        else:
            body = f"return {fn}(sp, {step}, pd, ds, r, c);"
        kind_arg = f"num::ycbcr420::Kind::{kind}, 0, {pp}"
    lam = f"{sig}{{ {voids}{body} }}"
    return (f"TEST(NumGen, {name}) {{\n"
            f"  num::ycbcr420::run_ycbcr420(\n      \"{fn}\", {kind_arg}, {bgr}, {swap}, {aval}, {mat},\n"
            f"      {lam});\n}}\n")


C3CONV422_KIND = {
    "rgbToYCbCr422C3C2": ("FwdC3C2", 3, "0"),
    "rgbToYCbCr422C3P3": ("FwdC3P3", 2, "0"),
    "rgbToYCbCr422P3C2": ("FwdP3C2", 0, "0"),
    "yCbCr422ToRgbC2": ("InvC2C", 4, "0"),
    "yCbCr422ToRgbC2P3": ("InvC2P3", 1, "0"),
    "yCbCr422ToRgbP3C3": ("InvP3C3", 1, "0"),
    "yCbCr422ToCbYCr422": ("Reorder", 0, "0"),
    "rgbToCbYCr422GammaC3C2": ("GammaErr", 0, "0"),
}
_YCBCR422_FILES: dict = {}
YCBCR422_FAMILIES = {
    "nppiRGBToYCbCr422", "nppiBGRToYCbCr422", "nppiYCbCr422ToRGB", "nppiYCbCr422ToBGR",
    "nppiYCbCr422ToCbYCr422", "nppiRGBToCbYCr422Gamma",
    "nppiRGBToCbYCr422", "nppiBGRToCbYCr422", "nppiCbYCr422ToRGB", "nppiCbYCr422ToBGR",
}


CMYK_TABLE = {
    "nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx": ("C3", 0, "c3"),
    "nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R_Ctx": ("P3", 0, "p3"),
    "nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R_Ctx": ("C3", 1, "c3"),
    "nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R_Ctx": ("P3", 1, "p3"),
}


def emit_cmyk(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    sig = ("[](const Npp8u *const *sp, int ss, Npp8u *const *dp, Npp8u *pd, int ds, int *dsa, "
           "NppiSize r, NppStreamContext c) ")
    voids = "(void)sp; (void)ss; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)c; "
    if fn in CMYK_TABLE:
        out, bgr, form = CMYK_TABLE[fn]
        body = (f"return {fn}(const_cast<const Npp8u **>(sp), ss, pd, ds, r, c);" if form == "c3"
                else f"return {fn}(const_cast<const Npp8u **>(sp), ss, const_cast<Npp8u **>(dp), ds, r, c);")
        lam = f"{sig}{{ {voids}{body} }}"
        return (f"TEST(NumGen, {name}) {{\n"
                f"  num::cmyk::run_cmyk_to_rgb<num::cmyk::Out::{out}, {'true' if bgr else 'false'}>(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")
    if fn == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx":
        lam = ("[](const Npp8u *const *sp, int ss, Npp8u *const *dp, int ds, NppiSize r, NppStreamContext c) "
               "{ return nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx(const_cast<const Npp8u **>(sp), ss, const_cast<Npp8u **>(dp), ds, r, c); }")
        return (f"TEST(NumGen, {name}) {{\n  num::cmyk::run_ycck_to_cmyk(\n      \"{fn}\",\n      {lam});\n}}\n")
    return None


COLOR_KEY = "0x20, 0x40, 0x60, 0x80"


def emit_color_key(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    if fn == "nppiCompColorKey_8u_C1R_Ctx":
        lam = ("[](const Npp8u *s1, int s1s, const Npp8u *s2, int s2s, Npp8u *d, int ds, NppiSize r, "
               "const Npp8u *key, NppStreamContext c) "
               "{ return nppiCompColorKey_8u_C1R_Ctx(s1, s1s, s2, s2s, d, ds, r, key[0], c); }")
        return (f"TEST(NumGen, {name}) {{\n  num::colorkey::run_comp_key<1>(\n      \"{fn}\", kColorKey, {lam});\n}}\n")
    if fn in ("nppiCompColorKey_8u_C3R_Ctx", "nppiCompColorKey_8u_C4R_Ctx"):
        ch = 3 if "C3" in fn else 4
        lam = (f"[](const Npp8u *s1, int s1s, const Npp8u *s2, int s2s, Npp8u *d, int ds, NppiSize r, "
               f"const Npp8u *key, NppStreamContext c) "
               f"{{ return {fn}(s1, s1s, s2, s2s, d, ds, r, const_cast<Npp8u *>(key), c); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::colorkey::run_comp_key<{ch}>(\n      \"{fn}\", kColorKey, {lam});\n}}\n")
    if fn == "nppiAlphaCompColorKey_8u_AC4R_Ctx":
        lam = ("[](const Npp8u *s1, int s1s, Npp8u a1, const Npp8u *s2, int s2s, Npp8u a2, Npp8u *d, int ds, "
               "NppiSize r, const Npp8u *key, NppiAlphaOp op, NppStreamContext c) "
               "{ return nppiAlphaCompColorKey_8u_AC4R_Ctx(s1, s1s, a1, s2, s2s, a2, d, ds, r, "
               "const_cast<Npp8u *>(key), op, c); }")
        return (f"TEST(NumGen, {name}) {{\n  num::colorkey::run_alpha_comp_key(\n      \"{fn}\", kColorKey, {lam});\n}}\n")
    return None


TWIST_MATRIX = "0.5f, 0.3f, 0.2f, 10.0f, -0.2f, 0.7f, 0.1f, 20.0f, 0.1f, -0.3f, 0.9f, 5.0f"

YUV_TWIST_TABLE = {
    # function: (T, kind, srcPp, dstPp, alpha, alphaParam, form)
    "nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx": ("Npp8u", "Inv420C", 0, 3, 0, 0, "inv"),
    "nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx": ("Npp8u", "Inv420C", 0, 4, 255, 0, "inv"),
    "nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx": ("Npp8u", "Inv420C", 0, 4, 0x37, 1, "inva"),
    "nppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx": ("Npp8u", "Inv420P3", 0, 0, 0, 0, "invp3"),
    "nppiYUV420ToRGB_16u_ColorTwist32f_P3C3R_Ctx": ("Npp16u", "Inv420C", 0, 3, 0, 0, "inv"),
    "nppiYUV420ToRGB_16u_ColorTwist32f_P3C4R_Ctx": ("Npp16u", "Inv420C", 0, 4, 65535, 0, "inv"),
    "nppiYUV420ToRGB_16u_ColorTwist32f_P3AC4R_Ctx": ("Npp16u", "Inv420C", 0, 4, 0x1234, 1, "inva"),
    "nppiYUV420ToRGB_16u_ColorTwist32f_P3R_Ctx": ("Npp16u", "Inv420P3", 0, 0, 0, 0, "invp3"),
    "nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx": ("Npp8u", "Inv422C2", 2, 3, 0, 0, "invc2"),
    "nppiYUV422ToRGB_8u_ColorTwist32f_P3C3R_Ctx": ("Npp8u", "Inv422C", 0, 3, 0, 0, "inv"),
    "nppiYUV422ToRGB_8u_ColorTwist32f_P3C4R_Ctx": ("Npp8u", "Inv422C", 0, 4, 255, 0, "inv"),
    "nppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx": ("Npp8u", "Inv422P3", 0, 0, 0, 0, "invp3"),
    "nppiYUV422ToRGB_16u_ColorTwist32f_C2C3R_Ctx": ("Npp16u", "Inv422C2", 2, 3, 0, 0, "invc2"),
    "nppiYUV422ToRGB_16u_ColorTwist32f_P3C3R_Ctx": ("Npp16u", "Inv422C", 0, 3, 0, 0, "inv"),
    "nppiYUV422ToRGB_16u_ColorTwist32f_P3C4R_Ctx": ("Npp16u", "Inv422C", 0, 4, 65535, 0, "inv"),
    "nppiYUV422ToRGB_16u_ColorTwist32f_P3R_Ctx": ("Npp16u", "Inv422P3", 0, 0, 0, 0, "invp3"),
    "nppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R_Ctx": ("Npp8u", "Inv422C", 0, 4, 0x37, 1, "inva"),
    "nppiYUV422ToRGB_16u_ColorTwist32f_P3AC4R_Ctx": ("Npp16u", "Inv422C", 0, 4, 0x1234, 1, "inva"),
    "nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx": ("Npp8u", "Fwd420P3", 3, 0, 0, 0, "fwdp3"),
    "nppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx": ("Npp8u", "Fwd420P3P3", 0, 0, 0, 0, "fwdp3p3"),
    "nppiRGBToYUV420_16u_ColorTwist32f_C3P3R_Ctx": ("Npp16u", "Fwd420P3", 3, 0, 0, 0, "fwdp3"),
    "nppiRGBToYUV420_16u_ColorTwist32f_P3R_Ctx": ("Npp16u", "Fwd420P3P3", 0, 0, 0, 0, "fwdp3p3"),
    "nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx": ("Npp8u", "Fwd422C2", 3, 0, 0, 0, "fwdc2"),
    "nppiRGBToYUV422_8u_ColorTwist32f_C3P3R_Ctx": ("Npp8u", "Fwd422P3", 3, 0, 0, 0, "fwdp3"),
    "nppiRGBToYUV422_8u_ColorTwist32f_P3R_Ctx": ("Npp8u", "Fwd422P3P3", 0, 0, 0, 0, "fwdp3p3"),
    "nppiRGBToYUV422_16u_ColorTwist32f_C3C2R_Ctx": ("Npp16u", "Fwd422C2", 3, 0, 0, 0, "fwdc2"),
    "nppiRGBToYUV422_16u_ColorTwist32f_C3P3R_Ctx": ("Npp16u", "Fwd422P3", 3, 0, 0, 0, "fwdp3"),
    "nppiRGBToYUV422_16u_ColorTwist32f_P3R_Ctx": ("Npp16u", "Fwd422P3P3", 0, 0, 0, 0, "fwdp3p3"),
}


def emit_yuv_twist(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if fn not in YUV_TWIST_TABLE:
        return None
    T, kind, spp, dpp, alpha, has_alpha, form = YUV_TWIST_TABLE[fn]
    name = re.sub(r"\W", "_", fn)
    sig = (f"[]({'const ' + T} *const *sp, const {T} *ps, int ss, int *ssa, {T} *const *dp, {T} *pd, "
           f"int ds, int *dsa, NppiSize r, const float *tw, NppStreamContext c) ")
    voids = "(void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)tw; (void)c; "
    cast = "const Npp32f (*)[4]"
    if form == "inv":
        body = f"return {fn}(sp, ssa, pd, ds, r, ({cast})tw, c);"
    elif form == "inva":
        body = f"return {fn}(sp, ssa, pd, ds, r, ({cast})tw, ({T})0x{'37' if T == 'Npp8u' else '1234'}, c);"
    elif form == "invp3":
        body = f"return {fn}(sp, ssa, const_cast<{T} **>(dp), dsa, r, ({cast})tw, c);"
    elif form == "invc2":
        body = f"return {fn}(ps, ss, pd, ds, r, ({cast})tw, c);"
    elif form == "fwdp3":
        body = f"return {fn}(ps, ss, const_cast<{T} **>(dp), dsa, r, ({cast})tw, c);"
    elif form == "fwdp3p3":
        body = f"return {fn}(sp, ssa, const_cast<{T} **>(dp), dsa, r, ({cast})tw, c);"
    else:  # fwdc2
        body = f"return {fn}(ps, ss, pd, ds, r, ({cast})tw, c);"
    lam = f"{sig}{{ {voids}{body} }}"
    return (f"TEST(NumGen, {name}) {{\n"
            f"  num::yuvtwist::run_twist<{T}>(\n      \"{fn}\", num::yuvtwist::Kind::{kind}, {spp}, {dpp}, "
            f"({T}){alpha},\n      kTwist, {lam});\n}}\n")


YCBCR420_LAYOUT = {
    # function: (kind, a0, lambda-form)
    "nppiYCbCr420_8u_P3P2R_Ctx": ("P3P2_420", 0, "p3p2"),
    "nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx": ("P3P2_420", 1, "p3p2"),
    "nppiYCbCr420_8u_P2P3R_Ctx": ("P2P3_420", 0, "p2p3"),
    "nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx": ("P2P3_420", 1, "p2p3"),
    "nppiYCbCr420ToYCbCr422_8u_P3R_Ctx": ("P3P3_422", 0, "p3p3"),
    "nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx": ("P2P3_422", 0, "p2p3"),
    "nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx": ("P2C2_422", 0, "p2c2"),
    "nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx": ("P2C2_422", 1, "p2c2"),
    "nppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx": ("YCrCb2CbYCr", 0, "p3c2"),
    "nppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx": ("P3P2_411", 1, "p3p2"),
    "nppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx": ("P3P2_411", 0, "p3p2"),
    "nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx": ("P2P3_411", 0, "p2p3"),
}


def emit_ycbcr420_layout(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if fn not in YCBCR420_LAYOUT:
        return None
    kind, a0, form = YCBCR420_LAYOUT[fn]
    name = re.sub(r"\W", "_", fn)
    sig = ("[](const Npp8u *const *sp, int ss, int *ssa, Npp8u *const *dp, int ds, int *dsa, "
           "NppiSize r, NppStreamContext c) ")
    voids = "(void)sp; (void)ss; (void)ssa; (void)dp; (void)ds; (void)dsa; (void)r; (void)c; "
    if form == "p3p2":
        body = f"return {fn}(sp, ssa, dp[0], dsa[0], dp[1], dsa[1], r, c);"
    elif form == "p2p3":
        body = f"return {fn}(sp[0], ssa[0], sp[1], ssa[1], const_cast<Npp8u **>(dp), dsa, r, c);"
    elif form == "p3p3":
        body = f"return {fn}(sp, ssa, const_cast<Npp8u **>(dp), dsa, r, c);"
    elif form == "p2c2":
        body = f"return {fn}(sp[0], ssa[0], sp[1], ssa[1], dp[0], dsa[0], r, c);"
    else:
        body = f"return {fn}(sp, ssa, dp[0], dsa[0], r, c);"
    lam = f"{sig}{{ {voids}{body} }}"
    return (f"TEST(NumGen, {name}) {{\n"
            f"  num::ycbcr420layout::run_layout420(\n      \"{fn}\", "
            f"num::ycbcr420layout::Kind::{kind}, {a0},\n      {lam});\n}}\n")


YCBCR422_SPECIAL = {
    # function: (kind, a0, a1, a2, a3, matrix, invStyle, body-form)
    "nppiYCrCb422ToRGB_8u_C2C3R_Ctx": ("YCrCbInvC2", 3, 0, 0, 0, "mat601v", 0, "c2"),
    "nppiYCrCb422ToRGB_8u_C2P3R_Ctx": ("YCrCbInvP3", 0, 0, 0, 0, "mat601v", 0, "c2p3"),
    "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx": ("CbYCr709InvC2", 3, 0, 0, 0, "mat709f", 2, "c2"),
    "nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx": ("CbYCr709InvC2", 4, 0, 0, 0x37, "mat709f", 2, "c2a"),
    "nppiBGRToCbYCr422_709HDTV_8u_C3C2R_Ctx": ("BgrFwd709C2", 3, 1, 0, 0, "mat709f", 0, "c2"),
    "nppiBGRToCbYCr422_709HDTV_8u_AC4C2R_Ctx": ("BgrFwd709C2", 4, 1, 0, 0, "mat709f", 0, "c2"),
    "nppiRGBToYCrCb422_8u_C3C2R_Ctx": ("YCrCbFwdC2", 3, 0, 0, 0, "mat601v", 0, "c2"),
    "nppiRGBToYCrCb422_8u_P3C2R_Ctx": ("YCrCbFwdP3C2", 0, 0, 0, 0, "mat601v", 0, "p3c2"),
}


def emit_ycbcr422(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fn in YCBCR422_SPECIAL:
        kind, a0, a1, a2, a3, mat, style, form = YCBCR422_SPECIAL[fn]
        name = re.sub(r"\W", "_", fn)
        sig = ("[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
               "int ds, int *dsa, NppiSize r, NppStreamContext c) ")
        voids = ("(void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; "
                 "(void)r; (void)c; ")
        if form == "c2":
            body = f"return {fn}(ps, ss, pd, ds, r, c);"
        elif form == "p3c2":
            body = f"return {fn}(sp, ssa[0], pd, ds, r, c);"
        elif form == "c2a":
            body = f"return {fn}(ps, ss, pd, ds, r, (Npp8u)0x37, c);"
        else:
            body = f"return {fn}(ps, ss, const_cast<Npp8u **>(dp), ds, r, c);"
        lam = f"{sig}{{ {voids}{body} }}"
        return (f"TEST(NumGen, {name}) {{\n"
                f"  num::ycbcr422::run_ycbcr422(\n      \"{fn}\", num::ycbcr422::Kind::{kind}, "
                f"&num::ycbcr422::{mat}(), {style},\n      {a0}, {a1}, {a2}, {a3},\n"
                f"      {lam});\n}}\n")
    if fam not in YCBCR422_FAMILIES:
        return None
    path = ROOT / "src" / "nppi" / "nppi_color_conversion_operations" / f"nppi_c3_{fam}_generated.cpp"
    if not path.exists():
        return None
    text = _YCBCR422_FILES.setdefault(str(path), path.read_text())
    m = re.search(re.escape(fn) + r"\s*\(", text)
    if not m:
        return None
    m2 = re.search(r"return nppi::c3conv::(\w+)\(([^;]*)\);", text[m.end():])
    if not m2 or m2.group(1) not in C3CONV422_KIND:
        return None
    exec_name = m2.group(1)
    kind, nargs, _ = C3CONV422_KIND[exec_name]
    tail = m2.group(2).split("nppStreamCtx.hStream")[-1]
    ints = [int(x) for x in re.findall(r"-?\d+", tail)]
    vals = (ints + [0, 0, 0, 0])[:4]
    if kind == "InvC2C" and vals[0] == 4:
        vals[3] = 0x37  # constant alpha used by the emitted lambda
    name = re.sub(r"\W", "_", fn)
    sig = ("[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
           "int ds, int *dsa, NppiSize r, NppStreamContext c) ")
    voids = "(void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)c; "
    if kind == "FwdC3C2" or kind == "Reorder" or kind == "GammaErr":
        body = f"return {fn}(ps, ss, pd, ds, r, c);"
    elif kind == "FwdC3P3":
        body = f"return {fn}(ps, ss, const_cast<Npp8u **>(dp), dsa, r, c);"
    elif kind == "InvC2P3":
        body = f"return {fn}(ps, ss, const_cast<Npp8u **>(dp), ds, r, c);"
    elif kind == "FwdP3C2":
        body = f"return {fn}(sp, ssa[0], pd, ds, r, c);"
    elif kind == "InvP3C3":
        body = f"return {fn}(sp, ssa, pd, ds, r, c);"
    else:  # InvC2C
        if vals[0] == 4:
            body = f"return {fn}(ps, ss, pd, ds, r, (Npp8u)0x37, c);"
        else:
            body = f"return {fn}(ps, ss, pd, ds, r, c);"
    lam = f"{sig}{{ {voids}{body} }}"
    return (f"TEST(NumGen, {name}) {{\n"
            f"  num::ycbcr422::run_ycbcr422(\n      \"{fn}\", num::ycbcr422::Kind::{kind}, "
            f"&num::ycbcr422::mat601v(), 0,\n      {vals[0]}, {vals[1]}, {vals[2]}, {vals[3]},\n"
            f"      {lam});\n}}\n")


LAYOUT_KIND = {
    "c2ToP3": ("C2P3", 4), "c2ToP2": ("C2P2", 3), "p3ToC2": ("P3C2", 3),
    "p2ToC2_411": ("P2C2_411", 1), "p2ToP3_411": ("P2P3_411", 0),
    "p2ToP3_422": ("P2P3_422", 0), "p2ToP3_420": ("P2P3_420", 1),
    "c2Reorder": ("C2REORDER", 2), "p3Resample": ("P3RESAMPLE", 3),
    "p3ToP2": ("P3P2", 2),
}
_LAYOUT_FILES: dict = {}


def _layout_call(fn: str, fam: str):
    path = ROOT / "src" / "nppi" / "nppi_color_conversion_operations" / f"nppi_c3_{fam}_generated.cpp"
    if not path.exists():
        return None
    text = _LAYOUT_FILES.setdefault(str(path), path.read_text())
    m = re.search(re.escape(fn) + r"\s*\(", text)
    if not m:
        return None
    m2 = re.search(r"return nppi::c3yuv::(\w+)\(([^;]*)\);", text[m.end():])
    if not m2 or m2.group(1) not in LAYOUT_KIND:
        return None
    tail = m2.group(2).split("nppStreamCtx.hStream")[-1]
    ints = [int(x) for x in re.findall(r"-?\d+", tail)]
    return m2.group(1), ints


def emit_yuv_layout(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    call = _layout_call(fn, fam)
    if call is None:
        return None
    exec_name, ints = call
    kind, nargs = LAYOUT_KIND[exec_name]
    args = (ints + [0, 0, 0, 0])[:4]
    name = re.sub(r"\W", "_", fn)
    sig = ("[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
           "int ds, int *dsa, NppiSize r, NppStreamContext c) ")
    if kind == "C2P3":
        body = f"return {fn}(ps, ss, const_cast<Npp8u **>(dp), dsa, r, c);"
    elif kind == "C2P2":
        body = f"return {fn}(ps, ss, dp[0], dsa[0], dp[1], dsa[1], r, c);"
    elif kind == "P3C2":
        body = f"return {fn}(sp, ssa, pd, ds, r, c);"
    elif kind == "P2C2_411":
        body = f"return {fn}(sp[0], ssa[0], sp[1], ssa[1], pd, ds, r, c);"
    elif kind in ("P2P3_411", "P2P3_422", "P2P3_420"):
        body = f"return {fn}(sp[0], ssa[0], sp[1], ssa[1], const_cast<Npp8u **>(dp), dsa, r, c);"
    elif kind == "C2REORDER":
        body = f"return {fn}(ps, ss, pd, ds, r, c);"
    elif kind == "P3RESAMPLE":
        body = f"return {fn}(sp, ssa, const_cast<Npp8u **>(dp), dsa, r, c);"
    else:  # P3P2
        body = f"return {fn}(sp, ssa, dp[0], dsa[0], dp[1], dsa[1], r, c);"
    lam = f"{sig}{{ (void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)c; {body} }}"
    return (f"TEST(NumGen, {name}) {{\n"
            f"  num::yuvlayout::run_layout(\n      \"{fn}\", num::yuvlayout::Kind::{kind}, "
            f"{args[0]}, {args[1]}, {args[2]}, {args[3]},\n      {lam});\n}}\n")


def emit_yuv_planar(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if fam not in YUV_FAMILIES or "ColorTwist" in fn:
        return None
    m = re.search(r"_(8u)_(P3C3R|P3C4R|P3AC4R|AC4P3R|P3R|C2C3R|C3P3R|C3C2R)_Ctx$", fn)
    if not m:
        return None
    suffix = m.group(2)
    sub = "S420" if "420" in fn else "S422"
    bgr = "BGR" in fn
    name = re.sub(r"\W", "_", fn)
    voids = "(void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)c; "
    if fam.startswith("nppiYUV"):
        if suffix == "P3C4R":
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(sp, ssa, pd, ds, r, c); }}")
            prefix = f"num::yuvplanar::run_yuv_c4_alpha<num::yuvplanar::Sub::{sub}, {'true' if bgr else 'false'}>"
        elif suffix == "C2C3R":
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(ps, ss, pd, ds, r, c); }}")
            prefix = f"num::yuvplanar::run_yuv_inv<num::yuvplanar::Sub::{sub}, num::yuvplanar::In::C2, num::yuvplanar::Out::C3, {'true' if bgr else 'false'}>"
        elif suffix == "P3R":
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(sp, ssa, const_cast<Npp8u **>(dp), ds, r, c); }}")
            prefix = f"num::yuvplanar::run_yuv_inv<num::yuvplanar::Sub::{sub}, num::yuvplanar::In::P3, num::yuvplanar::Out::P3, {'true' if bgr else 'false'}>"
        else:
            out = "AC4" if suffix == "P3AC4R" else "C3"
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(sp, ssa, pd, ds, r, c); }}")
            prefix = f"num::yuvplanar::run_yuv_inv<num::yuvplanar::Sub::{sub}, num::yuvplanar::In::P3, num::yuvplanar::Out::{out}, {'true' if bgr else 'false'}>"
    else:  # forward RGB -> YUV
        if suffix == "C3C2R":
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(ps, ss, pd, ds, r, c); }}")
            prefix = f"num::yuvplanar::run_yuv_fwd<num::yuvplanar::Sub::{sub}, true, true>"
        elif suffix in ("C3P3R", "AC4P3R", "P3R"):
            src = "sp" if suffix == "P3R" else "ps"
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}({src}, ss, const_cast<Npp8u **>(dp), dsa, r, c); }}")
            bgr_flag = "true" if "BGR" in fn else "false"
            pp = 4 if suffix == "AC4P3R" else 3
            prefix = (f"num::yuvplanar::run_yuv_fwd<num::yuvplanar::Sub::{sub}, "
                      f"{'false' if suffix == 'P3R' else 'true'}, false, {bgr_flag}, {pp}>")
        else:
            return None
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


YCBCR_SUB = {"444": "S444", "420": "S420", "422": "S422", "411": "S411"}


def emit_ycbcr_jpeg(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if "_JPEG_" not in fn:
        return None
    m = re.search(r"(444|420|422|411)\w*_JPEG_8u_(C3P3R|P3R|P3C3R)_Ctx$", fn)
    if not m:
        return None
    sub = YCBCR_SUB[m.group(1)]
    suffix = m.group(2)
    bgr = "true" if "BGR" in fn else "false"
    arr = sub != "S444"
    name = re.sub(r"\W", "_", fn)
    voids = "(void)sp; (void)ps; (void)ss; (void)ssa; (void)dp; (void)pd; (void)ds; (void)dsa; (void)r; (void)c; "
    if "ToYCbCr" in fn:
        src = "ps" if suffix == "C3P3R" else "sp"
        dstep = "dsa" if arr else "ds"
        lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
               f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
               f"{{ {voids}return {fn}({src}, ss, const_cast<Npp8u **>(dp), {dstep}, r, c); }}")
        prefix = (f"num::ycbcr::run_ycbcr_jpeg<num::ycbcr::Sub::{sub}, true, "
                  f"{'true' if suffix == 'C3P3R' else 'false'}, false, {bgr}>")
    else:
        sstep = "ssa" if arr else "ss"
        if suffix == "P3C3R":
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(sp, {sstep}, pd, ds, r, c); }}")
        else:
            lam = (f"[](const Npp8u *const *sp, const Npp8u *ps, int ss, int *ssa, Npp8u *const *dp, Npp8u *pd, "
                   f"int ds, int *dsa, NppiSize r, NppStreamContext c) "
                   f"{{ {voids}return {fn}(sp, {sstep}, const_cast<Npp8u **>(dp), ds, r, c); }}")
        prefix = (f"num::ycbcr::run_ycbcr_jpeg<num::ycbcr::Sub::{sub}, false, false, "
                  f"{'true' if suffix == 'P3C3R' else 'false'}, {bgr}>")
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


def emit_nv12(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    name = re.sub(r"\W", "_", fn)
    if fam in ("nppiNV12ToRGB_601", "nppiNV12ToBGR_601", "nppiNV12ToRGB_BT2020",
               "nppiNV12ToBGR_BT2020"):
        if not re.search(r"_8u_P2C3R_Ctx$", fn):
            return None
        order = "BGR" if "BGR" in fn else "RGB"
        coef = "0" if "601" in fn else "2"
        lam = ("[](const Npp8u *const *sp, int ss, Npp8u *d, int ds, NppiSize r, NppStreamContext c) "
               "{ (void)ss; (void)d; (void)ds; (void)r; (void)c; "
               f"return {fn}(sp, ss, d, ds, r, c); }}")
        prefix = f"num::nv12::run_nv12_ext_c3<{coef}, num::nv12::OutOrder::{order}, 3>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    if fam in ("nppiNV12ToRGB", "nppiNV12ToBGR") and "_P2C4R_Ctx" in fn and "NV21" not in fn:
        order = "BGR" if "BGR" in fn else "RGB"
        lam = ("[](const Npp8u *const *sp, int ss, Npp8u *d, int ds, NppiSize r, NppiColorSpace cs, "
               "Npp8u a, NppStreamContext c) "
               "{ (void)ss; (void)d; (void)ds; (void)r; (void)cs; (void)a; (void)c; "
               f"return {fn}(sp, ss, d, ds, r, cs, a, c); }}")
        out = []
        for cs, tag in (("NPP_BT601JPEG", "601JPEG"), ("NPP_BT709HDTV", "709HDTV"),
                        ("NPP_BT2020HDTV", "2020HDTV")):
            out.append(f"TEST(NumGen, {name}_{tag}) {{\n"
                       f"  num::nv12::run_nv12_ext_c4<num::nv12::OutOrder::{order}>(\n"
                       f"      \"{fn}\",\n      {cs}, 0x5A,\n      {lam});\n}}\n")
        return "".join(out)
    if fam == "nppiRGBToNV12":
        m = re.match(r"nppiRGBToNV12_8u_(C3|P3)P2R_Ctx$", fn)
        if not m:
            return None
        planar = m.group(1) == "P3"
        if planar:
            lam = ("[](const Npp8u *const *sp, int ss, Npp8u *dy, Npp8u *duv, int ds, NppiSize r, "
                   "NppiColorSpace cs, NppStreamContext c) "
                   "{ (void)ss; (void)dy; (void)duv; (void)ds; (void)r; (void)cs; (void)c; "
                   f"Npp8u *dp[2] = {{dy, duv}}; return {fn}(sp, ss, dp, ds, r, cs, c); }}")
            prefix = "num::nv12::run_rgb_to_nv12_ext<true>"
        else:
            lam = ("[](const Npp8u *sp, int ss, Npp8u *dy, Npp8u *duv, int ds, NppiSize r, "
                   "NppiColorSpace cs, NppStreamContext c) "
                   "{ (void)ss; (void)dy; (void)duv; (void)ds; (void)r; (void)cs; (void)c; "
                   f"Npp8u *dp[2] = {{dy, duv}}; return {fn}(sp, ss, dp, ds, r, cs, c); }}")
            prefix = "num::nv12::run_rgb_to_nv12_ext<false>"
        out = []
        for cs, tag in (("NPP_BT601JPEG", "601JPEG"), ("NPP_BT709HDTV", "709HDTV"),
                        ("NPP_BT2020HDTV", "2020HDTV")):
            out.append(f"TEST(NumGen, {name}_{tag}) {{\n"
                       f"  {prefix}(\n      \"{fn}\",\n      {cs},\n      {lam});\n}}\n")
        return "".join(out)
    if fam == "nppiYUV420ToNV12":
        if not re.search(r"_8u_P3P2R_Ctx$", fn):
            return None
        lam = ("[](const Npp8u *const *sp, const int *ss, Npp8u *dy, Npp8u *duv, int ds, NppiSize r, "
               "NppStreamContext c) "
               "{ (void)ss; (void)dy; (void)duv; (void)ds; (void)r; (void)c; "
               f"Npp8u *dp[2] = {{dy, duv}}; return {fn}(sp, ss, dp, ds, r, c); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::nv12::run_yuv420_to_nv12(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    if fam in ("nppiNV12ToRGB", "nppiNV12ToBGR", "nppiNV12ToRGB_709CSC",
               "nppiNV12ToRGB_709HDTV", "nppiNV12ToBGR_709CSC", "nppiNV12ToBGR_709HDTV",
               "nppiNV21ToRGB", "nppiNV21ToBGR"):
        if "_ColorTwist32f_" in fn:
            return None
        m = re.search(r"_(8u|16u)_P2C(3|4)R_Ctx$", fn)
        if not m:
            return None
        if "NV21" in fn:
            var = "NV21_FIXED"
        elif "709CSC" in fn:
            var = "NV12_709_CSC"
        elif "709" in fn:
            var = "NV12_709"
        else:
            var = "NV12_601"
        order = "BGR" if "BGR" in fn else "RGB"
        dstc = int(m.group(2))
        lam = ("[](const Npp8u *const *sp, int ss, Npp8u *d, int ds, NppiSize r, NppStreamContext c) "
               "{ (void)ss; (void)ds; (void)r; (void)c; "
               f"return {fn}(sp, ss, d, ds, r, c); }}")
        prefix = f"num::nv12::run_nv12_to_rgb<num::nv12::Var::{var}, num::nv12::OutOrder::{order}, {dstc}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    if fam == "nppiNV12ToYUV420":
        m = re.search(r"_8u_P2P3R_Ctx$", fn)
        if not m:
            return None
        lam = ("[](const Npp8u *const *sp, int ss, Npp8u *const *dp, int *dss, NppiSize r, "
               "NppStreamContext c) "
               "{ (void)ss; (void)dss; (void)r; (void)c; "
               f"return {fn}(sp, ss, const_cast<Npp8u **>(dp), dss, r, c); }}")
        return (f"TEST(NumGen, {name}) {{\n  num::nv12::run_nv12_to_yuv420(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    return None



def emit_uyvy(row: dict) -> Optional[str]:
    fn = row["function_name"]
    name = re.sub(r"\W", "_", fn)
    CS = (("NPP_BT601JPEG", "601JPEG"), ("NPP_BT709HDTV", "709HDTV"), ("NPP_BT2020HDTV", "2020HDTV"))
    # UYVY(16) -> RGB(16)
    m = re.match(r"nppiUYVY(16)?_(8u|16u)_ToRGB_(8u|16u)_C3(C3|AC4|P3)R_Ctx$", fn)
    if m:
        S = "Npp16u" if m.group(2) == "16u" else "Npp8u"
        mode = {"C3": 0, "AC4": 1, "P3": 2}[m.group(4)]
        alpha = "12345" if S == "Npp16u" else "0x5A"
        out = []
        for cs, tag in CS:
            if mode == 1:
                lam = (f"[]({S} *s, int ss, NppiPoint off, {S} *d, int ds, NppiSize r, NppiColorSpace cs, "
                       f"{S} a, NppStreamContext c) "
                       f"{{ (void)ss; (void)off; (void)d; (void)ds; (void)r; (void)cs; (void)a; (void)c; "
                       f"return {fn}(s, ss, off, d, ds, r, cs, a, c); }}")
                call = f"num::uyvy::run_uyvy_dec<{S}, 1>(\"{fn}\", {cs}, ({S}){alpha}, {lam})"
            elif mode == 2:
                lam = (f"[]({S} *s, int ss, NppiPoint off, {S} **dp, int ds, NppiSize r, NppiColorSpace cs, "
                       f"NppStreamContext c) "
                       f"{{ (void)ss; (void)off; (void)dp; (void)ds; (void)r; (void)cs; (void)c; "
                       f"return {fn}(s, ss, off, dp, ds, r, cs, c); }}")
                call = f"num::uyvy::run_uyvy_dec<{S}, 2>(\"{fn}\", {cs}, ({S})0, {lam})"
            else:
                lam = (f"[]({S} *s, int ss, NppiPoint off, {S} *d, int ds, NppiSize r, NppiColorSpace cs, "
                       f"NppStreamContext c) "
                       f"{{ (void)ss; (void)off; (void)d; (void)ds; (void)r; (void)cs; (void)c; "
                       f"return {fn}(s, ss, off, d, ds, r, cs, c); }}")
                call = f"num::uyvy::run_uyvy_dec<{S}, 0>(\"{fn}\", {cs}, ({S})0, {lam})"
            out.append(f"TEST(NumGen, {name}_{tag}) {{\n  {call};\n}}\n")
        return "".join(out)
    # RGB(A)(16) -> UYVY(16)
    m = re.match(r"nppiRGBA?(16)?_(8u|16u)_ToUYVY(16)?_(8u|16u)_(C3|P3|AC4)(C3R|P3R|C3R)_Ctx$", fn)
    if m:
        S = "Npp16u" if m.group(2) == "16u" else "Npp8u"
        tok = m.group(5)
        out = []
        for cs, tag in CS:
            if tok == "P3":
                lam = (f"[](const {S} *const *sp, int ss, NppiPoint off, {S} *d, int ds, NppiSize r, "
                       f"NppiColorSpace cs, NppStreamContext c) "
                       f"{{ (void)ss; (void)off; (void)d; (void)ds; (void)r; (void)cs; (void)c; "
                       f"return {fn}(sp, ss, off, d, ds, r, cs, c); }}")
                call = f"num::uyvy::run_rgb_enc<{S}, 1>(\"{fn}\", {cs}, {lam})"
            else:
                mode = 2 if tok == "AC4" else 0
                lam = (f"[]({S} *s, int ss, NppiPoint off, {S} *d, int ds, NppiSize r, NppiColorSpace cs, "
                       f"NppStreamContext c) "
                       f"{{ (void)ss; (void)off; (void)d; (void)ds; (void)r; (void)cs; (void)c; "
                       f"return {fn}(s, ss, off, d, ds, r, cs, c); }}")
                call = f"num::uyvy::run_rgb_enc<{S}, {mode}>(\"{fn}\", {cs}, {lam})"
            out.append(f"TEST(NumGen, {name}_{tag}) {{\n  {call};\n}}\n")
        return "".join(out)
    return None


def emit_nv12_twist(row: dict) -> Optional[str]:
    fn = row["function_name"]
    if "ColorTwist32f" not in fn:
        return None
    m = re.search(r"_(8u|16u)_ColorTwist32f_", fn)
    if not m:
        return None
    S = {"8u": "Npp8u", "16u": "Npp16u"}[m.group(1)]
    name = re.sub(r"\W", "_", fn)
    if "RGBToNV12" in fn:
        planar = "P3P2R" in fn
        if planar:
            lam = (f"[](const {S} *const *sp, int *ss, {S} *const *dp, int *ds, NppiSize r, const float *tw, "
                   f"NppStreamContext c) "
                   f"{{ (void)ss; (void)ds; (void)r; (void)tw; (void)c; "
                   f"return {fn}(sp, ss, const_cast<{S} **>(dp), ds, r, (const Npp32f (*)[4])tw, c); }}")
            prefix = f"num::nv12::run_rgb_twist_to_nv12<{S}, true>"
        else:
            lam = (f"[](const {S} *sp, int ss, {S} *const *dp, int *ds, NppiSize r, const float *tw, "
                   f"NppStreamContext c) "
                   f"{{ (void)ss; (void)ds; (void)r; (void)tw; (void)c; "
                   f"return {fn}(sp, ss, const_cast<{S} **>(dp), ds, r, (const Npp32f (*)[4])tw, c); }}")
            prefix = f"num::nv12::run_rgb_twist_to_nv12<{S}, false>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    if "NV12ToRGB" in fn:
        lam = (f"[](const {S} *const *sp, int *ss, {S} *d, int ds, NppiSize r, const float *tw, "
               f"NppStreamContext c) "
               f"{{ (void)ss; (void)ds; (void)r; (void)tw; (void)c; "
               f"return {fn}(sp, ss, d, ds, r, (const Npp32f (*)[4])tw, c); }}")
        prefix = f"num::nv12::run_nv12_twist_to_rgb<{S}>"
        return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    return None


def emit_color_batch(row: dict) -> Optional[str]:
    fn = row["function_name"]
    m = re.match(r"nppi(YCbCr|YUV)(420|422)?To(RGB|BGR)Batch_8u_(P3C3R|C3R)(_Advanced)?_Ctx$", fn)
    if not m:
        return None
    yuv = m.group(1) == "YUV"
    sub = m.group(2) or "444"
    form = m.group(4)
    if sub != "444" and form != "P3C3R":
        return None
    kind = f"k{yuv and 'YUV' or 'YCbCr'}{sub}{'P3' if form == 'P3C3R' else 'C3'}"
    bgr = "true" if m.group(3) == "BGR" else "false"
    adv = "true" if m.group(5) else "false"
    name = re.sub(r"\W", "_", fn)
    lam = ("[](const NppiImageDescriptor *const *sp, NppiImageDescriptor *pd, int n, NppiSize r, "
           "NppStreamContext c) { "
           + (f"return {fn}(sp, pd, n, r, c);" if form == "P3C3R"
              else f"return {fn}(sp[0], pd, n, r, c);")
           + " }")
    prefix = f"num::colbatch::run_batch<num::colbatch::{kind}, {bgr}, {adv}>"
    return (f"TEST(NumGen, {name}) {{\n  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")


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
        text = (emit_lut(row) or emit_lut_palette(row) or emit_twist(row) or emit_twist_batch(row)
                or emit_gamma(row) or emit_gray(row) or emit_gradient(row) or emit_hls(row)
                or emit_hls_layout(row) or emit_yuv(row) or emit_cs(row) or emit_cfa(row)
                or emit_ycbcr_jpeg(row) or emit_yuv_planar(row) or emit_yuv_layout(row)
                or emit_ycbcr422(row) or emit_ycbcr420(row) or emit_ycbcr420_layout(row)
                or emit_yuv_twist(row) or emit_color_key(row) or emit_cmyk(row) or emit_nv12(row)
                or emit_nv12_twist(row) or emit_uyvy(row) or emit_color_batch(row))
        if text is None:
            skipped.append(dict(function=row["function_name"], family=fam,
                                reason="later color batch / unsupported"))
            continue
        out.append(text)
    header = ("// GENERATED by tools/gen_nppi_color_tests.py -- DO NOT EDIT.\n"
              "// module: nppi_color_conversion.h (LUT + ColorTwist + gray/gamma)\n"
              "#include \"num_nppi_lut.h\"\n"
              "#include \"num_nppi_color_twist.h\"\n"
              "#include \"num_nppi_color_cvt.h\"\n"
              "#include \"num_nppi_hls.h\"\n"
              "#include \"num_nppi_yuv.h\"\n"
              "#include \"num_nppi_yuv420.h\"\n"
              "#include \"num_nppi_yuv_layout.h\"\n"
              "#include \"num_nppi_ycbcr422.h\"\n"
              "#include \"num_nppi_ycbcr420.h\"\n"
              "#include \"num_nppi_ycbcr420_layout.h\"\n"
              "#include \"num_nppi_yuv_twist.h\"\n"
              "#include \"num_nppi_color_key.h\"\n"
              "#include \"num_nppi_cmyk.h\"\n"
              "#include \"num_nppi_colorspace.h\"\n"
              "#include \"num_nppi_cfa.h\"\n"
              "#include \"num_nppi_ycbcr.h\"\n"
              "#include \"num_nppi_yuv.h\"\n"
              "#include \"num_nppi_nv12.h\"\n"
              "#include \"num_nppi_uyvy.h\"\n"
              "#include \"num_nppi_color_batch.h\"\n"
              "\n"
              "static const float kTwist[12] = {" + TWIST_MATRIX + "};\n"
              "static const Npp8u kColorKey[4] = {" + COLOR_KEY + "};\n")
    dest = Path(args.out) if args.out else ROOT / "test" / "unit" / "num" / "test_gen_nppi_color.cpp"
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
