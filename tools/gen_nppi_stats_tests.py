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
    m = re.search(r"_(AC4|C1|C3|C4)([A-Z]*)_Ctx$", fn)
    if not m:
        return None
    return m.group(1), m.group(2)


def chan_count(tok: str) -> int:
    return {"C1": 1, "C3": 3, "C4": 4, "AC4": 4}[tok]


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
    suf = suffix(fn)
    reason = skip_reason(params, suf, row["family"])
    if reason:
        return None
    tok, rest = suf
    C = chan_count(tok)
    A = C - 1 if tok == "AC4" else C
    S = ctype(params, "pSrc") or ctype(params, "pSrc1")
    if S is None:
        return None

    # --- GetBufferHostSize smoke test ------------------------------------
    if "GetBufferHostSize" in fn:
        lam = (
            "[](NppiSize roi, int *sz, NppStreamContext c) {\n"
            "#ifdef USE_NVIDIA_NPP_TESTS\n"
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
