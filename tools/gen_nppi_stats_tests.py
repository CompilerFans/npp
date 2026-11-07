#!/usr/bin/env python3
"""Generate CPU-golden tests for the nppi statistics reduction families.

First batch: Sum/Mean/Mean_StdDev, Norm/NormDiff/NormRel, error metrics,
DotProd, Min/Max/Indx/MinMax(+Indx), MinEvery/MaxEvery and GetBufferHostSize
smoke tests.  Histogram / cross-correlation / integral / quality / batch
metrics are left for later batches and listed in generated_report.json.
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
COMPLEX = {"Npp16sc", "Npp32sc", "Npp32fc", "Npp64sc", "Npp64fc"}

RED1 = {
    "nppiSum_": "Sum",
    "nppiMean_": "Mean",
    "nppiNorm_L1_": "NormL1",
    "nppiNorm_L2_": "NormL2",
    "nppiNorm_Inf_": "NormInf",
}
RED2 = {
    "nppiNormDiff_L1_": "NormDiffL1",
    "nppiNormDiff_L2_": "NormDiffL2",
    "nppiNormDiff_Inf_": "NormDiffInf",
    "nppiNormRel_L1_": "NormRelL1",
    "nppiNormRel_L2_": "NormRelL2",
    "nppiNormRel_Inf_": "NormRelInf",
}
ERR = {
    "nppiMaximumError_": "MaxAbsErr",
    "nppiAverageError_": "AvgAbsErr",
    "nppiMaximumRelativeError_": "MaxRelErr",
    "nppiAverageRelativeError_": "AvgRelErr",
}
OUT_NAMES = {"pSum", "aSum", "pMean", "aMean", "pNorm", "aNorm", "pNormDiff", "aNormDiff",
             "pNormRel", "aNormRel", "pError", "pDp", "aDp"}


def ctype(params, name):
    for p in params:
        if p.name == name:
            return p.ctype
    return None


def out_ctype(params):
    for p in params:
        if p.name in OUT_NAMES:
            return p.ctype
    return None


def suffix(fn: str):
    m = re.search(r"_(AC4|C1|C2|C3|C4)([A-Za-z]*)_Ctx$", fn)
    if not m:
        return None
    return m.group(1), m.group(2)


def chan_count(tok: str) -> int:
    return {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "AC4": 4}[tok]


def skip_reason(params, suf, family: str) -> Optional[str]:
    if any(p.ctype in COMPLEX for p in params):
        return "complex"
    if suf is None:
        return "unsupported signature"
    tok, rest = suf
    if "M" in rest:
        return "masked"
    return None


def args_for(params, mapping):
    out = []
    for p in params:
        if p.name == "nppStreamContext":
            continue
        out.append(mapping[p.name])
    return out


def emit(row: dict) -> Optional[str]:
    fn = row["function_name"]
    params = G.parse_params(row["params"])
    if params is None:
        return None
    name = re.sub(r"\W", "_", fn)
    # --- circular radial profile ----------------------------------------
    if row["family"] == "nppiCircularRadialProfile":
        if row["dtype"] != "32f":
            return None
        lam = ("[](const Npp32f *p, int st, NppiSize roi, NppiPoint32f ctr, "
               "NppiProfileData *out, Npp16u samples, NppStreamContext c) { "
               "(void)st; (void)roi; (void)ctr; (void)out; (void)samples; (void)c; return "
               + fn + "(p, st, roi, ctr, out, samples, c); }")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_circular_radial_profile(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    # --- elliptical radial profile --------------------------------------
    if row["family"] == "nppiEllipticalRadialProfile":
        if row["dtype"] != "32f":
            return None
        lam = ("[](const Npp32f *p, int st, NppiSize roi, NppiPoint32f ctr, Npp32f ratio, "
               "Npp32f orient, NppiProfileData *out, Npp16u samples, NppStreamContext c) { "
               "(void)st; (void)roi; (void)ctr; (void)ratio; (void)orient; (void)out; "
               "(void)samples; (void)c; return " + fn +
               "(p, st, roi, ctr, ratio, orient, out, samples, c); }")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_elliptical_radial_profile(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    # --- even histogram levels (host-side) -------------------------------
    if row["family"] == "nppiEvenLevelsHost":
        plain = re.sub(r"_Ctx$", "", fn)
        lam = ("[](Npp32s *lv, int n, Npp32s lo, Npp32s hi, NppStreamContext c) {\n"
               "#ifdef USE_NVIDIA_NPP_TESTS\n"
               "        (void)c;\n"
               f"        return {plain}(lv, n, lo, hi);\n"
               "#else\n"
               f"        return {fn}(lv, n, lo, hi, c);\n"
               "#endif\n"
               "      }")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_even_levels_host(\n      \"{fn}\",\n      {lam});\n}}\n")
    # --- batch image quality metrics ------------------------------------
    if row["family"] in ("nppiMSEBatch", "nppiPSNRBatch"):
        m = re.search(r"_8u_C([13])R(_Advanced)?_Ctx$", fn)
        if m is None:
            return None
        C = int(m.group(1))
        adv = m.group(2) is not None
        psnr = row["family"] == "nppiPSNRBatch"
        lam = ("[](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int nb, "
               "NppiSize r, Npp32f *o, NppiBufferDescriptor *buf, NppStreamContext c) "
               "{ (void)r; (void)buf; (void)c; return " + fn + "(a, b, nb, r, o, buf, c); }")
        prefix = (f"num::run_nppi_metric_batch<{C}, {str(psnr).lower()}, "
                  f"{str(adv).lower()}>")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  {prefix}(\n      \"{fn}\",\n      {lam});\n}}\n")
    # --- SSIM / WMSSSIM / MSSSIM -----------------------------------------
    if row["family"] in ("nppiSSIM", "nppiWMSSSIM", "nppiMSSSIM",
                         "nppiSSIMBatch", "nppiWMSSSIMBatch"):
        m = re.search(r"_8u_C([1234])R", fn)
        if m is None:
            return None
        C = int(m.group(1))
        if row["family"] == "nppiMSSSIM":
            if C != 1:
                return None
            lam = ("[](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize r, "
                   "Npp32f *o, Npp8u *buf, NppStreamContext c) { (void)sa; (void)sb; "
                   "(void)r; (void)o; (void)buf; (void)c; return " + fn +
                   "(a, sa, b, sb, r, o, buf, c); }")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_msssim(\n      \"{fn}\",\n      {lam});\n}}\n")
        if row["family"] in ("nppiSSIMBatch", "nppiWMSSSIMBatch"):
            if C not in (1, 3, 4):
                return None
            wm = row["family"] == "nppiWMSSSIMBatch"
            adv = "_Advanced" in fn
            if adv and not wm:
                return None
            lam = ("[](const NppiImageDescriptor *a, const NppiImageDescriptor *b, "
                   "int nb, NppiSize r, Npp32f *o, NppiBufferDescriptor *buf, "
                   "NppStreamContext c) { (void)r; (void)buf; (void)c; return " +
                   fn + "(a, b, nb, r, o, buf, c); }")
            prefix = (f"num::run_nppi_ssim_batch<{C}, {str(wm).lower()}, "
                      f"{str(adv).lower()}>")
            return (f"TEST(NumGenNppiStats, {name}) {{\n  {prefix}(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")
        if C not in (1, 3, 4):
            return None
        wm = row["family"] == "nppiWMSSSIM"
        lam = ("[](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize r, "
               "Npp32f *o, Npp8u *buf, NppStreamContext c) { (void)sa; (void)sb; "
               "(void)r; (void)o; (void)buf; (void)c; return " + fn +
               "(a, sa, b, sb, r, o, buf, c); }")
        prefix = f"num::run_nppi_ssim_single<{C}, {str(wm).lower()}>"
        return (f"TEST(NumGenNppiStats, {name}) {{\n  {prefix}(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")

    # --- GetBufferHostSize smoke test ------------------------------------
    if "GetBufferHostSize" in fn:
        size_t_sig = any(p.name == "hpBufferSize" and p.ctype == "size_t" for p in params)
        if size_t_sig:
            # NPP 13.4-style declaration: the host size is size_t on both the
            # reference and the in-tree library.
            lam = (
                "[](NppiSize roi, int *sz, NppStreamContext c) {\n"
                "        size_t s = 0;\n"
                f"        NppStatus st = {fn}(roi, &s, c);\n"
                "        if (st == NPP_NO_ERROR) *sz = (int)(s > 2147483647u ? 2147483647u : s);\n"
                "        return st;\n"
                "      }")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_buffer_size(\n      \"{fn}\",\n      {lam});\n}}\n")
        lam = (
            "[](NppiSize roi, int *sz, NppStreamContext c) {\n"
            "#if defined(USE_NVIDIA_NPP_TESTS) || NPP_BUFFER_SIZE_IS_SIZE_T
"
            "        size_t s = 0;\n"
            f"        NppStatus st = {fn}(roi, &s, c);\n"
            "        if (st == NPP_NO_ERROR) *sz = (int)(s > 2147483647u ? 2147483647u : s);\n"
            "        return st;\n"
            "#else\n"
            f"        return {fn}(roi, sz, c);\n"
            "#endif\n"
            "      }")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_buffer_size(\n      \"{fn}\",\n      {lam});\n}}\n")

    # --- Histogram GetBufferSize smoke test -----------------------------
    if "Histogram" in fn and "GetBufferSize" in fn:
        m = re.search(r"_(AC4|C1|C2|C3|C4)M?([A-Za-z]*)_Ctx$", fn)
        per_ch = bool(m and m.group(1) in ("AC4", "C3", "C4"))
        nl_arg = "nl" if per_ch else "nl[0]"
        lam = (
            "[](NppiSize roi, int *nl, int *sz, NppStreamContext c) {\n"
            "#if defined(USE_NVIDIA_NPP_TESTS) || NPP_BUFFER_SIZE_IS_SIZE_T
"
            "        size_t s = 0;\n"
            f"        NppStatus st = {fn}(roi, {nl_arg}, &s, c);\n"
            "        if (st == NPP_NO_ERROR) *sz = (int)(s > 2147483647u ? 2147483647u : s);\n"
            "        return st;\n"
            "#else\n"
            f"        return {fn}(roi, {nl_arg}, sz, c);\n"
            "#endif\n"
            "      }")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_histogram_buffer_size(\n      \"{fn}\",\n      {lam});\n}}\n")

    suf = suffix(fn)
    if suf is not None:
        tok, rest = suf
        C = chan_count(tok)
        A = C - 1 if tok == "AC4" else C
    else:
        tok = rest = None
    # --- complex error metrics -------------------------------------------
    if any(p.ctype in COMPLEX for p in params):
        kinds = {"nppiMaximumError": "MaxAbsErr", "nppiAverageError": "AvgAbsErr",
                 "nppiMaximumRelativeError": "MaxRelErr",
                 "nppiAverageRelativeError": "AvgRelErr"}
        kind = None
        for k, v in kinds.items():
            if row["family"] == k:
                kind = v
        if kind is not None:
            S = ctype(params, "pSrc1")
            if S is None or suf is None:
                return None
            A = C - 1 if tok == "AC4" else C
            amap = {"pSrc1": "a", "nSrc1Step": "sa", "pSrc2": "b", "nSrc2Step": "sb",
                    "oSizeROI": "r", "pError": "o", "pDeviceBuffer": "buf",
                    "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            lam = (f"[](const {S} *a, int sa, const {S} *b, int sb, NppiSize r, Npp64f *o, "
                   f"Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)sb; (void)r; (void)buf; (void)c; return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_error_complex<{S}, {C}, {A}, "
                    f"num::NppiStatKind::{kind}>(\n      \"{fn}\",\n      {lam});\n}}\n")

    # --- CrossCorr / SqrDistance -----------------------------------------
    if "CrossCorr" in fn or "SqrDistance" in fn:
        if fn.startswith("nppiCrossCorrFull") or fn.startswith("nppiSqrDistanceFull"):
            mode = "Full"
        elif fn.startswith("nppiCrossCorrValid") or fn.startswith("nppiSqrDistanceValid"):
            mode = "Valid"
        elif fn.startswith("nppiCrossCorrSame") or fn.startswith("nppiSqrDistanceSame"):
            mode = "Same"
        else:
            return None
        if "NormLevelAdvanced" in fn:
            op = "NormLevelAdvanced"
        elif "NormLevel" in fn:
            op = "NormLevel"
        elif "SqrDistance" in fn:
            op = "SqrDistance"
        elif "Norm" in fn:
            op = "Norm"
        else:
            op = "Plain"
        S = ctype(params, "pSrc")
        D = ctype(params, "pDst")
        if S is None or D is None or suf is None:
            return None
        A = C - 1 if tok == "AC4" else C
        has_sf = any(p.name == "nScaleFactor" for p in params)
        has_buf = any(p.name == "pDeviceBuffer" for p in params)
        advanced = any(p.name == "pAdvancedScratchBuffer" for p in params)
        amap = {"pSrc": "a", "nSrcStep": "sa", "oSrcRoiSize": "sr", "pTpl": "t",
                "nTplStep": "ts", "oTplRoiSize": "tr", "pDst": "d", "nDstStep": "ds",
                "nScaleFactor": "sf", "pDeviceBuffer": "buf",
                "pAdvancedScratchBuffer": "abuf", "nppStreamContext": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {S} *a, int sa, NppiSize sr, const {S} *t, int ts, NppiSize tr, "
               f"{D} *d, int ds, int sf, Npp8u *buf, Npp8u *abuf, NppStreamContext c) "
               f"{{ (void)sa; (void)sr; (void)ts; (void)tr; (void)ds; (void)sf; "
               f"(void)buf; (void)abuf; (void)c; return {call}; }}")
        prefix = (f"num::run_nppi_crosscorr<{S}, {D}, {C}, {A}, num::XCorrMode::{mode}, "
                  f"num::XCorrOp::{op}, {str(has_sf).lower()}, {str(has_buf).lower()}, "
                  f"{str(advanced).lower()}>")
        return (f"TEST(NumGenNppiStats, {name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if rest is not None and "M" in rest:
        fam = row["family"]
        use_coi = rest.startswith("C")
        if fam == "nppiMean_StdDev":
            S = ctype(params, "pSrc")
            if S is None:
                return None
            amap = {"pSrc": "a", "nSrcStep": "sa", "pMask": "mk", "nMaskStep": "ms",
                    "oSizeROI": "r", "nCOI": "coi", "pDeviceBuffer": "buf",
                    "pMean": "mo", "pStdDev": "so", "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            coi_p = ", int coi" if use_coi else ""
            coi_v = "(void)coi; " if use_coi else ""
            lam = (f"[](const {S} *a, int sa, const Npp8u *mk, int ms, NppiSize r{coi_p}, "
                   f"Npp64f *mo, Npp64f *so, Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)ms; (void)r; {coi_v}(void)buf; (void)c; "
                   f"return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_meanstd_masked<{S}, {C}, {str(use_coi).lower()}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")
        if fam == "nppiMinMaxIndx":
            S = ctype(params, "pSrc")
            if S is None:
                return None
            amap = {"pSrc": "a", "nSrcStep": "sa", "pMask": "mk", "nMaskStep": "ms",
                    "oSizeROI": "r", "nCOI": "coi", "pMinValue": "mn", "pMaxValue": "mx",
                    "pMinIndex": "pi", "pMaxIndex": "qi", "pDeviceBuffer": "buf",
                    "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            coi_p = ", int coi" if use_coi else ""
            coi_v = "(void)coi; " if use_coi else ""
            lam = (f"[](const {S} *a, int sa, const Npp8u *mk, int ms, NppiSize r{coi_p}, "
                   f"{S} *mn, {S} *mx, NppiPoint *pi, NppiPoint *qi, Npp8u *buf, "
                   f"NppStreamContext c) {{ (void)sa; (void)ms; (void)r; {coi_v}"
                   f"(void)buf; (void)c; return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_minmaxindx_masked<{S}, {C}, {str(use_coi).lower()}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")
        kinds1 = {"nppiMean": "Mean", "nppiNorm_L1": "NormL1",
                  "nppiNorm_L2": "NormL2", "nppiNorm_Inf": "NormInf"}
        kinds2 = {"nppiNormDiff_L1": "NormDiffL1", "nppiNormDiff_L2": "NormDiffL2",
                  "nppiNormDiff_Inf": "NormDiffInf", "nppiNormRel_L1": "NormRelL1",
                  "nppiNormRel_L2": "NormRelL2", "nppiNormRel_Inf": "NormRelInf"}
        if fam in kinds1:
            kind = kinds1[fam]
            S = ctype(params, "pSrc")
            out = out_ctype(params)
            if S is None or out is None:
                return None
            amap = {"pSrc": "a", "nSrcStep": "sa", "pMask": "mk", "nMaskStep": "ms",
                    "oSizeROI": "r", "nCOI": "coi", "pDeviceBuffer": "buf",
                    "pMean": "o", "aMean": "o", "pNorm": "o", "aNorm": "o",
                    "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            coi_p = ", int coi" if use_coi else ""
            coi_v = "(void)coi; " if use_coi else ""
            lam = (f"[](const {S} *a, int sa, const Npp8u *mk, int ms, NppiSize r{coi_p}, "
                   f"{out} *o, Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)ms; (void)r; {coi_v}(void)buf; (void)c; "
                   f"return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_reduce1_masked<{S}, {C}, {A}, "
                    f"num::NppiStatKind::{kind}, {out}, {str(use_coi).lower()}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")
        if fam in kinds2:
            S = ctype(params, "pSrc1")
            if S is None:
                return None
            kind = kinds2[fam]
            amap = {"pSrc1": "a", "nSrc1Step": "sa", "pSrc2": "b", "nSrc2Step": "sb",
                    "pMask": "mk", "nMaskStep": "ms", "oSizeROI": "r", "nCOI": "coi",
                    "pDeviceBuffer": "buf", "pNormDiff": "o", "aNormDiff": "o",
                    "pNormRel": "o", "aNormRel": "o", "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            coi_p = ", int coi" if use_coi else ""
            coi_v = "(void)coi; " if use_coi else ""
            lam = (f"[](const {S} *a, int sa, const {S} *b, int sb, const Npp8u *mk, "
                   f"int ms, NppiSize r{coi_p}, Npp64f *o, Npp8u *buf, "
                   f"NppStreamContext c) {{ (void)sa; (void)sb; (void)ms; (void)r; "
                   f"{coi_v}(void)buf; (void)c; return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_reduce2_masked<{S}, {C}, {A}, "
                    f"num::NppiStatKind::{kind}, {str(use_coi).lower()}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")
        return None

    reason = skip_reason(params, suf, row["family"])
    if reason:
        return None
    tok, rest = suf
    C = chan_count(tok)
    A = C - 1 if tok == "AC4" else C
    S = ctype(params, "pSrc") or ctype(params, "pSrc1")
    if S is None:
        return None

    # --- Histograms -------------------------------------------------------
    fam = row["family"]
    if fam in ("nppiHistogramEven", "nppiHistogramRange"):
        S = ctype(params, "pSrc")
        if S is None:
            return None
        A = C - 1 if tok == "AC4" else C
        lv = "Npp32f" if row["dtype"] == "32f" else "Npp32s"
        per_ch = A > 1
        hs = "hist[0]" if not per_ch else "hist"
        if fam == "nppiHistogramEven":
            amap = {"pSrc": "a", "nSrcStep": "sa", "oSizeROI": "r",
                    "pHist": hs, "nLevels": ("nl" if per_ch else "nl[0]"),
                    "nLowerLevel": ("lo" if per_ch else "lo[0]"),
                    "nUpperLevel": ("hi" if per_ch else "hi[0]"),
                    "pBuffer": "buf", "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            lam = (f"[](const {S} *a, int sa, NppiSize r, Npp32s **hist, int *nl, "
                   f"{lv} *lo, {lv} *hi, Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)r; (void)nl; (void)lo; (void)hi; (void)buf; (void)c; "
                   f"return {call}; }}")
            prefix = (f"num::run_nppi_histogram<{S}, {lv}, {C}, {A}, "
                      f"num::NppiHistKind::Even>")
        else:
            amap = {"pSrc": "a", "nSrcStep": "sa", "oSizeROI": "r",
                    "pHist": hs, "pLevels": ("lv" if per_ch else "lv[0]"),
                    "nLevels": ("nl" if per_ch else "nl[0]"),
                    "pBuffer": "buf", "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            lam = (f"[](const {S} *a, int sa, NppiSize r, Npp32s **hist, const {lv} **lv, "
                   f"int *nl, Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)r; (void)nl; (void)buf; (void)c; return {call}; }}")
            prefix = (f"num::run_nppi_histogram<{S}, {lv}, {C}, {A}, "
                      f"num::NppiHistKind::Range>")
        return (f"TEST(NumGenNppiStats, {name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    # --- Integral images --------------------------------------------------
    if fam == "nppiIntegral":
        S = ctype(params, "pSrc")
        D = ctype(params, "pDst")
        if S is None or D is None:
            return None
        amap = {"pSrc": "a", "nSrcStep": "sa", "pDst": "d", "nDstStep": "sd",
                "oROI": "r", "nVal": "nval", "nppStreamContext": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {S} *a, int sa, {D} *d, int sd, NppiSize r, {D} nval, "
               f"NppStreamContext c) {{ (void)sa; (void)sd; (void)r; (void)nval; (void)c; "
               f"return {call}; }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_integral<{S}, {D}>(\n      \"{fn}\",\n      {lam});\n}}\n")
    if fam == "nppiSqrIntegral":
        S = ctype(params, "pSrc")
        DS = ctype(params, "pDst")
        DQ = ctype(params, "pSqr")
        if S is None or DS is None or DQ is None:
            return None
        amap = {"pSrc": "a", "nSrcStep": "sa", "pDst": "d", "nDstStep": "sd",
                "pSqr": "q", "nSqrStep": "sq", "oSrcROI": "r", "nVal": "nval",
                "nValSqr": "nvalsqr", "nppStreamContext": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {S} *a, int sa, {DS} *d, int sd, {DQ} *q, int sq, NppiSize r, "
               f"{DS} nval, {DQ} nvalsqr, NppStreamContext c) "
               f"{{ (void)sa; (void)sd; (void)sq; (void)r; (void)nval; (void)nvalsqr; "
               f"(void)c; return {call}; }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_sqr_integral<{S}, {DS}, {DQ}>(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    # --- CountInRange (all variants; per-channel bounds for C3/AC4) -------
    if fam == "nppiCountInRange":
        S = ctype(params, "pSrc")
        if S is None:
            return None
        A = C - 1 if tok == "AC4" else C
        per_ch = ctype(params, "aCounts") is not None
        if per_ch:
            amap = {"pSrc": "a", "nSrcStep": "sa", "oSizeROI": "r",
                    "aCounts": "o", "aLowerBound": "lo", "aUpperBound": "hi",
                    "pDeviceBuffer": "buf", "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            lam = (f"[](const {S} *a, int sa, NppiSize r, int *o, {S} *lo, {S} *hi, "
                   f"Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)r; (void)buf; (void)c; return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_count_range_multi<{S}, {C}, {A}>(\n      \"{fn}\",\n"
                    f"      {lam});\n}}\n")
        else:
            amap = {"pSrc": "a", "nSrcStep": "sa", "oSizeROI": "r",
                    "pCounts": "o", "nLowerBound": "lo[0]", "nUpperBound": "hi[0]",
                    "pDeviceBuffer": "buf", "nppStreamContext": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            lam = (f"[](const {S} *a, int sa, NppiSize r, int *o, {S} *lo, {S} *hi, "
                   f"Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)sa; (void)r; (void)buf; (void)c; return {call}; }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_count_range_multi<{S}, {C}, {A}>(\n      \"{fn}\",\n"
                    f"      {lam});\n}}\n")

    # --- Quality metrics (QualityIndex / MSE / PSNR / RectStdDev) --------
    if fam == "nppiQualityIndex":
        S = ctype(params, "pSrc1")
        if S is None:
            return None
        A = C - 1 if tok == "AC4" else C
        amap = {"pSrc1": "a", "nSrc1Step": "sa", "pSrc2": "b", "nSrc2Step": "sb",
                "oRoiSize": "r", "pDst": "o", "pDeviceBuffer": "buf",
                "nppStreamContext": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {S} *a, int sa, const {S} *b, int sb, NppiSize r, Npp32f *o, "
               f"Npp8u *buf, NppStreamContext c) "
               f"{{ (void)sa; (void)sb; (void)r; (void)buf; (void)c; return {call}; }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_quality_index<{S}, {C}, {A}>(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if fam in ("nppiMSE", "nppiPSNR"):
        S = ctype(params, "pSrc1")
        if S is None:
            return None
        A = C - 1 if tok == "AC4" else C
        psnr = fam == "nppiPSNR"
        amap = {"pSrc1": "a", "nSrc1Step": "sa", "pSrc2": "b", "nSrc2Step": "sb",
                "oSizeROI": "r", "pMSE": "o", "pPSNR": "o", "pDeviceBuffer": "buf",
                "nppStreamContext": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {S} *a, int sa, const {S} *b, int sb, NppiSize r, Npp32f *o, "
               f"Npp8u *buf, NppStreamContext c) "
               f"{{ (void)sa; (void)sb; (void)r; (void)buf; (void)c; return {call}; }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_mse<{S}, {C}, {A}, {str(psnr).lower()}>(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if fam == "nppiRectStdDev":
        S = ctype(params, "pSrc")
        DS = ctype(params, "pDst")
        DQ = ctype(params, "pSqr")
        if S is None or DS is None or DQ is None:
            return None
        has_sf = any(p.name == "nScaleFactor" for p in params)
        amap = {"pSrc": "a", "nSrcStep": "sa", "pSqr": "b", "nSqrStep": "sb",
                "pDst": "d", "nDstStep": "sd", "oSizeROI": "r", "oRect": "rect",
                "nScaleFactor": "sf", "nppStreamContext": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        if has_sf:
            lam = (f"[](const {S} *a, int sa, const {DQ} *b, int sb, {DS} *d, int sd, "
                   f"NppiSize r, NppiRect rect, int sf, NppStreamContext c) "
                   f"{{ (void)sa; (void)sb; (void)sd; (void)r; (void)rect; (void)sf; (void)c; "
                   f"return {call}; }}")
        else:
            lam = (f"[](const {S} *a, int sa, const {DQ} *b, int sb, {DS} *d, int sd, "
                   f"NppiSize r, NppiRect rect, NppStreamContext c) "
                   f"{{ (void)sa; (void)sb; (void)sd; (void)r; (void)rect; (void)c; "
                   f"return {call}; }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_rect_stddev<{S}, {DS}, {DQ}, {str(has_sf).lower()}>(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")

    if fam == "nppiEvenLevelsHost":
        lam = (f"[](Npp32s *lv, int nl, Npp32s lo, Npp32s hi, NppStreamContext c) "
               f"{{ (void)lv; (void)nl; (void)lo; (void)hi; (void)c; "
               f"return {fn}(lv, nl, lo, hi, c); }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_even_levels(\n      \"{fn}\",\n      {lam});\n}}\n")

    # --- Mean_StdDev ------------------------------------------------------
    if fn.startswith("nppiMean_StdDev_"):
        if rest not in ("R", "CR"):
            return None
        if rest == "CR":
            if tok != "C3":
                return None
            lam = (f"[](const {S} *p, int step, NppiSize roi, int coi, Npp64f *m, Npp64f *s, "
                   f"Npp8u *buf, NppStreamContext c) {{ (void)coi; (void)buf; (void)c; "
                   f"return {fn}(p, step, roi, coi, buf, m, s, c); }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_meanstd<{S}, {C}>(\n      \"{fn}\",\n      {lam},\n      1);\n}}\n")
        lam = (f"[](const {S} *p, int step, NppiSize roi, Npp64f *m, Npp64f *s, Npp8u *buf, "
               f"NppStreamContext c) {{ (void)buf; (void)c; "
               f"return {fn}(p, step, roi, buf, m, s, c); }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_meanstd<{S}, {C}>(\n      \"{fn}\",\n      {lam});\n}}\n")

    # --- MinMaxIndx -------------------------------------------------------
    if fn.startswith("nppiMinMaxIndx_"):
        if any(p.name == "nCOI" for p in params):
            lam = (f"[](const {S} *p, int step, NppiSize roi, int coi, {S} *mn, {S} *mx, "
                   f"NppiPoint *pi, NppiPoint *qi, Npp8u *buf, NppStreamContext c) "
                   f"{{ (void)buf; (void)c; return {fn}(p, step, roi, coi, mn, mx, pi, qi, buf, c); }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_minmaxindx_coi<{S}, {C}>(\n      \"{fn}\",\n"
                    f"      {lam});\n}}\n")
        if rest != "R":
            return None
        lam = (f"[](const {S} *p, int step, NppiSize roi, {S} *mn, {S} *mx, NppiPoint *pi, "
               f"NppiPoint *qi, Npp8u *buf, NppStreamContext c) {{ (void)buf; (void)c; "
               f"return {fn}(p, step, roi, mn, mx, pi, qi, buf, c); }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_minmaxindx<{S}, {C}>(\n      \"{fn}\",\n      {lam});\n}}\n")

    # --- extrema ----------------------------------------------------------
    ext_map = [
        ("nppiMinIndx_", True, False, True, False),
        ("nppiMaxIndx_", False, True, False, True),
        ("nppiMinMax_", True, True, False, False),
        ("nppiMin_", True, False, False, False),
        ("nppiMax_", False, True, False, False),
    ]
    for prefix, wmin, wmax, wminidx, wmaxidx in ext_map:
        if fn.startswith(prefix):
            mapping = {"pSrc": "p", "nSrcStep": "step", "oSizeROI": "roi",
                       "pDeviceBuffer": "buf", "pMin": "mn", "pMax": "mx",
                       "aMin": "mn", "aMax": "mx"}
            if "MinIndx" in fn:
                mapping["pIndexX"] = "mnx"
                mapping["pIndexY"] = "mny"
                mapping["aIndexX"] = "mnx"
                mapping["aIndexY"] = "mny"
            elif "MaxIndx" in fn:
                mapping["pIndexX"] = "mxx"
                mapping["pIndexY"] = "mxy"
                mapping["aIndexX"] = "mxx"
                mapping["aIndexY"] = "mxy"
            args = args_for(params, mapping)
            lam = (f"[](const {S} *p, int step, NppiSize roi, Npp8u *buf, {S} *mn, {S} *mx, "
                   f"int *mnx, int *mny, int *mxx, int *mxy, NppStreamContext c) "
                   f"{{ (void)buf; (void)c; return {fn}({', '.join(args + ['c'])}); }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_extrema<{S}, {C}, {A}, {str(wmin).lower()}, "
                    f"{str(wmax).lower()}, {str(wminidx).lower()}, {str(wmaxidx).lower()}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")

    # --- MinEvery / MaxEvery ---------------------------------------------
    if fn.startswith("nppiMinEvery_") or fn.startswith("nppiMaxEvery_"):
        is_max = "MaxEvery" in fn
        args = args_for(params, {"pSrc": "p", "nSrcStep": "step", "pSrcDst": "d",
                                 "nSrcDstStep": "dstep", "oSizeROI": "roi"})
        lam = (f"[](const {S} *p, int step, {S} *d, int dstep, NppiSize roi, NppStreamContext c) "
               f"{{ (void)c; return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_every<{S}, {C}, {A}, {str(is_max).lower()}>(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")

    # --- error metrics ----------------------------------------------------
    for prefix, kind in ERR.items():
        if fn.startswith(prefix):
            args = args_for(params, {"pSrc1": "p", "nSrc1Step": "step", "pSrc2": "p2",
                                     "nSrc2Step": "step2", "oSizeROI": "roi",
                                     "pDeviceBuffer": "buf", "pError": "o"})
            lam = (f"[](const {S} *p, int step, const {S} *p2, int step2, NppiSize roi, "
                   f"Npp64f *o, Npp8u *buf, NppStreamContext c) {{ (void)buf; (void)c; "
                   f"return {fn}({', '.join(args + ['c'])}); }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_error<{S}, {C}, {A}, num::NppiStatKind::{kind}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")

    # --- DotProd ----------------------------------------------------------
    if fn.startswith("nppiDotProd_"):
        S2 = ctype(params, "pSrc2")
        out = out_ctype(params)
        if S2 != S or out is None:
            return None
        args = args_for(params, {"pSrc1": "p", "nSrc1Step": "step", "pSrc2": "p2",
                                 "nSrc2Step": "step2", "oSizeROI": "roi",
                                 "pDeviceBuffer": "buf", "pDp": "o", "aDp": "o"})
        lam = (f"[](const {S} *p, int step, const {S} *p2, int step2, NppiSize roi, "
               f"{out} *o, Npp8u *buf, NppStreamContext c) {{ (void)buf; (void)c; "
               f"return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppiStats, {name}) {{\n"
                f"  num::run_nppi_dotprod<{S}, {C}, {A}, {out}>(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    # --- NormDiff / NormRel ----------------------------------------------
    for prefix, kind in RED2.items():
        if fn.startswith(prefix):
            args = args_for(params, {"pSrc1": "p", "nSrc1Step": "step", "pSrc2": "p2",
                                     "nSrc2Step": "step2", "oSizeROI": "roi",
                                     "pDeviceBuffer": "buf", "pNormDiff": "o",
                                     "aNormDiff": "o", "pNormRel": "o", "aNormRel": "o"})
            lam = (f"[](const {S} *p, int step, const {S} *p2, int step2, NppiSize roi, "
                   f"Npp64f *o, Npp8u *buf, NppStreamContext c) {{ (void)buf; (void)c; "
                   f"return {fn}({', '.join(args + ['c'])}); }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_reduce2<{S}, {C}, {A}, num::NppiStatKind::{kind}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")

    # --- Sum / Mean / Norm ------------------------------------------------
    for prefix, kind in RED1.items():
        if fn.startswith(prefix):
            out = out_ctype(params)
            if out is None:
                return None
            args = args_for(params, {"pSrc": "p", "nSrcStep": "step", "oSizeROI": "roi",
                                     "pDeviceBuffer": "buf", "pSum": "o", "aSum": "o",
                                     "pMean": "o", "aMean": "o", "pNorm": "o", "aNorm": "o"})
            lam = (f"[](const {S} *p, int step, NppiSize roi, {out} *o, Npp8u *buf, "
                   f"NppStreamContext c) {{ (void)buf; (void)c; "
                   f"return {fn}({', '.join(args + ['c'])}); }}")
            return (f"TEST(NumGenNppiStats, {name}) {{\n"
                    f"  num::run_nppi_reduce1<{S}, {C}, {A}, num::NppiStatKind::{kind}, {out}>(\n"
                    f"      \"{fn}\",\n      {lam});\n}}\n")

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG))
            if r["module"] == "nppi_statistics_functions.h" and r["ctx"] == "1"]
    out, skipped = [], []
    for row in rows:
        text = emit(row)
        if text is None:
            skipped.append(row["function_name"])
        else:
            out.append(text)
    header = ("// GENERATED by tools/gen_nppi_stats_tests.py -- DO NOT EDIT.\n"
              "// module: nppi_statistics_functions.h (reduction batch)\n"
              "#include \"num_nppi_stats.h\"\n\n")
    if args.emit:
        Path(args.out).write_text(header + "\n".join(out))
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev["nppi_statistics_functions.h"] = dict(
            emitted=len(out),
            skipped=[{"function": s, "reason": "later batch / masked / complex"}
                     for s in skipped])
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
