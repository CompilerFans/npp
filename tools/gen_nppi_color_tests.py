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
    m = re.search(r"_(8u|8s|16u|16s|16f|32f)_", fn)
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
        text = (emit_lut(row) if fam in MODE
                else emit_lut_palette(row) or emit_twist(row) or emit_twist_batch(row)
                or emit_gamma(row) or emit_gray(row) or emit_gradient(row) or emit_hls(row)
                or emit_yuv(row))
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
              "#include \"num_nppi_yuv.h\"\n\n")
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
