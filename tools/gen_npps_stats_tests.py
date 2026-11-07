#!/usr/bin/env python3
"""Generate CPU-golden tests for the npps statistics module.

Real-valued reduction, extrema, error-metric, every, count and zero-crossing
APIs are covered by the drivers in test/unit/num/num_npps.h.  Complex-valued
variants are skipped for now (tracked in generated_report.json), as are the
non-Ctx entry points that NPP 13 removed.  GetBufferSize entry points get a
small smoke test because their byte count is implementation-defined.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_ref_tests as G

CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
COMPLEX = {"Npp8uc", "Npp16uc", "Npp16sc", "Npp32uc", "Npp32sc",
           "Npp32fc", "Npp64sc", "Npp64fc"}

STAT1 = {
    "nppsSum_": "Sum",
    "nppsMean_": "Mean",
    "nppsStdDev_": "StdDev",
    "nppsNorm_L1_": "NormL1",
    "nppsNorm_L2Sqr_": "NormL2Sqr",
    "nppsNorm_L2_": "NormL2",
    "nppsNorm_Inf_": "NormInf",
}
STAT2 = {
    "nppsNormDiff_L1_": "NormL1",
    "nppsNormDiff_L2Sqr_": "NormL2Sqr",
    "nppsNormDiff_L2_": "NormL2",
    "nppsNormDiff_Inf_": "NormInf",
    "nppsMaximumError_": "MaxAbsErr",
    "nppsAverageError_": "AvgAbsErr",
    "nppsMaximumRelativeError_": "MaxRelErr",
    "nppsAverageRelativeError_": "AvgRelErr",
    "nppsDotProd_": "DotProd",
}
EXTREMA = ("nppsMin_", "nppsMax_", "nppsMinIndx_", "nppsMaxIndx_",
           "nppsMinMax_", "nppsMinMaxIndx_", "nppsMinAbs_", "nppsMaxAbs_",
           "nppsMinAbsIndx_", "nppsMaxAbsIndx_")


def ctype(params, name):
    for p in params:
        if p.name == name:
            return p.ctype
    return None


def output_ctype(params):
    for p in params:
        if p.name in ("pDst", "pSum", "pMean", "pStdDev", "pNorm", "pDp"):
            return p.ctype
    return None


def sf_list(params):
    return "{0, 1, 2}" if any(p.name == "nScaleFactor" for p in params) else "{0}"


def has_complex(params) -> bool:
    return any(p.ctype in COMPLEX for p in params)


def emit(row: dict) -> Optional[str]:
    fn = row["function_name"]
    params = G.parse_params(row["params"])
    if params is None:
        return None
    name = re.sub(r"\W", "_", fn)
    lam_head = "      "

    if "GetBufferSize" in fn:
        args = []
        for p in params:
            if p.name == "nLength":
                args.append("n")
            elif p.name == "hpBufferSize":
                args.append("sz")
            elif p.name == "nppStreamContext":
                args.append("c")
            else:
                raise KeyError(p.name)
        lam = (
            "[](int n, int *sz, NppStreamContext c) {\n"
            "#if defined(USE_NVIDIA_NPP_TESTS) || NPP_BUFFER_SIZE_IS_SIZE_T
"
            f"        size_t s = 0; NppStatus st = {fn}((size_t)n, &s, c);\n"
            "        if (st == NPP_NO_ERROR) *sz = (int)(s > 2147483647u ? 2147483647u : s);\n"
            "        return st;\n"
            "#else\n"
            f"        return {fn}(n, sz, c);\n"
            "#endif\n"
            "      }")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_buffer_size(\n      \"{fn}\",\n      {lam});\n}}\n")

    if fn.startswith("nppsMeanStdDev_"):
        s = ctype(params, "pSrc")
        om = ctype(params, "pMean")
        os_ = ctype(params, "pStdDev")
        if not (s and om and os_):
            return None
        args = []
        for p in params:
            args.append({"pSrc": "a", "pMean": "m", "pStdDev": "s",
                         "nLength": "n", "nScaleFactor": "sf",
                         "pDeviceBuffer": "buf", "nppStreamContext": "c"}[p.name])
        lam = (f"[](const {s} *a, {om} *m, {os_} *s, int n, int sf, Npp8u *buf, "
               f"NppStreamContext c) {{ (void)n; (void)sf; (void)buf; (void)c; "
               f"return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_meanstd<{s}, {om}, {os_}>(\n      \"{fn}\",\n"
                f"{lam_head}{lam},\n      {sf_list(params)});\n}}\n")

    if fn.startswith("nppsCountInRange_"):
        s = ctype(params, "pSrc")
        args = []
        for p in params:
            args.append({"pSrc": "a", "pCounts": "o", "nLength": "n",
                         "nLowerBound": "lo", "nUpperBound": "hi",
                         "pDeviceBuffer": "buf", "nppStreamContext": "c"}[p.name])
        lam = (f"[](const {s} *a, int *o, int n, Npp32s lo, Npp32s hi, Npp8u *buf, "
               f"NppStreamContext c) {{ (void)n; (void)lo; (void)hi; (void)buf; (void)c; "
               f"return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_count_range(\n      \"{fn}\",\n{lam_head}{lam});\n}}\n")

    if fn.startswith("nppsZeroCrossing_"):
        s = ctype(params, "pSrc")
        args = []
        for p in params:
            args.append({"pSrc": "a", "pValZC": "o", "nLength": "n",
                         "tZCType": "mode", "pDeviceBuffer": "buf",
                         "nppStreamContext": "c"}[p.name])
        lam = (f"[](const {s} *a, Npp32f *o, int n, NppsZCType mode, Npp8u *buf, "
               f"NppStreamContext c) {{ (void)n; (void)mode; (void)buf; (void)c; "
               f"return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_zero_crossing<{s}>(\n      \"{fn}\",\n{lam_head}{lam});\n}}\n")

    if fn.startswith("nppsMinEvery_") or fn.startswith("nppsMaxEvery_"):
        s = ctype(params, "pSrc")
        is_max = "MaxEvery" in fn
        args = []
        for p in params:
            args.append({"pSrc": "a", "pSrcDst": "d", "nLength": "n",
                         "nppStreamContext": "c"}[p.name])
        lam = (f"[](const {s} *a, {s} *d, int n, NppStreamContext c) "
               f"{{ (void)n; (void)c; return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_every<{s}, {str(is_max).lower()}>(\n      \"{fn}\",\n"
                f"{lam_head}{lam});\n}}\n")

    if any(fn.startswith(p) for p in EXTREMA):
        s = ctype(params, "pSrc")
        abs_only = "Abs" in fn
        want_min = ("Min" in fn)
        want_max = ("Max" in fn)
        want_min_idx = ("MinIndx" in fn) or ("MinMaxIndx" in fn) or ("MinAbsIndx" in fn)
        want_max_idx = ("MaxIndx" in fn) or ("MinMaxIndx" in fn) or ("MaxAbsIndx" in fn)
        args = []
        for p in params:
            if p.name == "pSrc":
                args.append("a")
            elif p.name in ("pMin", "pMinAbs"):
                args.append("mn")
            elif p.name in ("pMax", "pMaxAbs"):
                args.append("mx")
            elif p.name == "pIndx":
                args.append("mi" if "Min" in fn else "xi")
            elif p.name == "pMinIndx":
                args.append("mi")
            elif p.name == "pMaxIndx":
                args.append("xi")
            elif p.name == "nLength":
                args.append("n")
            elif p.name == "pDeviceBuffer":
                args.append("buf")
            elif p.name == "nppStreamContext":
                args.append("c")
            else:
                raise KeyError(p.name)
        lam = (f"[](const {s} *a, {s} *mn, int *mi, {s} *mx, int *xi, int n, "
               f"Npp8u *buf, NppStreamContext c) {{ (void)n; (void)buf; (void)c; "
               f"return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_extrema<{s}, {str(abs_only).lower()}, "
                f"{str(want_min).lower()}, {str(want_max).lower()}, "
                f"{str(want_min_idx).lower()}, {str(want_max_idx).lower()}>(\n"
                f"      \"{fn}\",\n{lam_head}{lam});\n}}\n")

    kind = None
    two = False
    for pref, k in STAT1.items():
        if fn.startswith(pref):
            kind = k
            break
    if kind is None:
        for pref, k in STAT2.items():
            if fn.startswith(pref):
                kind = k
                two = True
                break
    if kind is not None:
        s = ctype(params, "pSrc") or ctype(params, "pSrc1")
        out = output_ctype(params)
        s2 = ctype(params, "pSrc2")
        if s is None or out is None:
            return None
        args = []
        for p in params:
            mp = {"pSrc": "a", "pSrc1": "a", "pSrc2": "b",
                  "pDst": "o", "pSum": "o", "pMean": "o", "pStdDev": "o",
                  "pNorm": "o", "pDp": "o", "nLength": "n",
                  "nScaleFactor": "sf", "pDeviceBuffer": "buf",
                  "nppStreamContext": "c"}
            args.append(mp[p.name])
        if two:
            lam = (f"[](const {s} *a, const {s2} *b, {out} *o, int n, int sf, "
                   f"Npp8u *buf, NppStreamContext c) {{ (void)n; (void)sf; (void)buf; "
                   f"(void)c; return {fn}({', '.join(args + ['c'])}); }}")
            return (f"TEST(NumGenNppsStats, {name}) {{\n"
                    f"  num::run_npps_stat2<{s}, {s2}, {out}, num::StatKind::{kind}>(\n"
                    f"      \"{fn}\",\n{lam_head}{lam},\n      {sf_list(params)});\n}}\n")
        lam = (f"[](const {s} *a, {out} *o, int n, int sf, Npp8u *buf, NppStreamContext c) "
               f"{{ (void)n; (void)sf; (void)buf; (void)c; "
               f"return {fn}({', '.join(args + ['c'])}); }}")
        return (f"TEST(NumGenNppsStats, {name}) {{\n"
                f"  num::run_npps_stat1<{s}, {out}, num::StatKind::{kind}>(\n"
                f"      \"{fn}\",\n{lam_head}{lam},\n      {sf_list(params)});\n}}\n")

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG))
            if r["module"] == "npps_statistics_functions.h" and r["ctx"] == "1"]
    out, skipped = [], []
    for row in rows:
        text = emit(row)
        if text is None:
            skipped.append(row["function_name"])
        else:
            out.append(text)
    header = ("// GENERATED by tools/gen_npps_stats_tests.py -- DO NOT EDIT.\n"
              "// module: npps_statistics_functions.h\n#include \"num_npps.h\"\n\n")
    if args.emit:
        Path(args.out).write_text(header + "\n".join(out))
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev["npps_statistics_functions.h"] = dict(
            emitted=len(out),
            skipped=[{"function": s, "reason": "complex or unsupported"}
                     for s in skipped])
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
