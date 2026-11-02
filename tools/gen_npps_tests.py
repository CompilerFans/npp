#!/usr/bin/env python3
"""Generate 1D CPU-golden tests for the npps modules.

The drivers live in test/unit/num/num_npps.h.  This generator covers the
element-wise families first (copy/set/zero/convert/threshold); reduction and
signal-processing families are added as their drivers land.
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

THR = {
    "nppsThreshold": "Threshold",
    "nppsThreshold_Val": "ThresholdVal",
    "nppsThreshold_GT": "GT",
    "nppsThreshold_LT": "LT",
    "nppsThreshold_GTVal": "GTVal",
    "nppsThreshold_LTVal": "LTVal",
}
COMPLEX_SHORT = {"Npp16sc", "Npp32sc", "Npp64sc"}

ARITH = {
    "nppsAdd": ("binary", "Add"),
    "nppsSub": ("binary", "Sub"),
    "nppsMul": ("binary", "Mul"),
    "nppsDiv": ("binary", "Div"),
    "nppsAnd": ("binary", "And"),
    "nppsOr": ("binary", "Or"),
    "nppsXor": ("binary", "Xor"),
    "nppsAddC": ("const", "AddC"),
    "nppsSubC": ("const", "SubC"),
    "nppsSubCRev": ("const", "SubCRev"),
    "nppsMulC": ("const", "MulC"),
    "nppsDivC": ("const", "DivC"),
    "nppsDivCRev": ("const", "DivCRev"),
    "nppsAndC": ("const", "AndC"),
    "nppsOrC": ("const", "OrC"),
    "nppsXorC": ("const", "XorC"),
    "nppsLShiftC": ("const", "LShiftC"),
    "nppsRShiftC": ("const", "RShiftC"),
    "nppsSqr": ("unary", "Sqr"),
    "nppsSqrt": ("unary", "Sqrt"),
    "nppsAbs": ("unary", "Abs"),
    "nppsLn": ("unary", "Ln"),
    "nppsExp": ("unary", "Exp"),
    "nppsNot": ("unary", "Not"),
    "npps10Log10": ("unary", "Log10"),
    "nppsCubrt": ("unary", "Cubrt"),
    "nppsArctan": ("unary", "Arctan"),
}


def ctype_of(params, name):
    for p in params:
        if p.name == name:
            return p.ctype
    return None


def emit(row: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    params = G.parse_params(row["params"])
    if params is None:
        return None
    # parse_params drops the stream context.
    def lam_call(argmap):
        args = [argmap(p) for p in params] + ["c"]
        return f"{fn}({', '.join(args)})"

    def arg_by_name(p):
        n = p.name
        if n in ("pSrc", "pSrc1"):
            return "s"
        if n in ("pSrcDst", "pDst", "pDst1"):
            return "d"
        if n == "nLength":
            return "n"
        if n == "nppStreamCtx":
            return "c"
        if n == "nValue":
            return "value"
        if n == "nLevel":
            return "level"
        if n == "nRelOp":
            return "static_cast<NppCmpOp>(op)"
        if n == "eRoundMode":
            return "static_cast<NppRoundMode>(mode)"
        if n == "nScaleFactor":
            return "sf"
        raise KeyError(n)

    if fam == "nppsCopy":
        t = ctype_of(params, "pSrc")
        if t is None:
            return None
        call = lam_call(arg_by_name)
        lam = (f"[](const {t} *s, {t} *d, int n, NppStreamContext c) "
               f"{{ (void)n; (void)c; return {call}; }}")
        body = f"num::run_npps_copy<{t}>"
        args = f"      {lam});"
    elif fam == "nppsSet":
        t = ctype_of(params, "pDst")
        if t is None:
            return None
        call = lam_call(arg_by_name)
        lam = (f"[]({t} value, {t} *d, int n, NppStreamContext c) "
               f"{{ (void)n; (void)c; return {call}; }}")
        body = f"num::run_npps_set<{t}>"
        args = f"      {lam});"
    elif fam == "nppsZero":
        t = ctype_of(params, "pDst")
        if t is None:
            return None
        call = lam_call(arg_by_name)
        lam = (f"[]({t} *d, int n, NppStreamContext c) "
               f"{{ (void)n; (void)c; return {call}; }}")
        body = f"num::run_npps_zero<{t}>"
        args = f"      {lam});"
    elif fam == "nppsConvert":
        sct = ctype_of(params, "pSrc")
        dct = ctype_of(params, "pDst")
        if sct is None or dct is None:
            return None
        sfs = "_Sfs" in fn or any(p.name == "nScaleFactor" for p in params)
        rnd = any(p.name == "eRoundMode" for p in params)
        call = lam_call(arg_by_name)
        lam = (f"[](const {sct} *s, {dct} *d, int n, int mode, int sf, NppStreamContext c) "
               f"{{ (void)n; (void)mode; (void)sf; (void)c; return {call}; }}")
        body = f"num::run_npps_convert<{sct}, {dct}>"
        fin_half_up = "64s32s" in fn
        args = (f"      {lam},\n      {str(sfs).lower()}, {str(rnd).lower()}, "
                f"{str(fin_half_up).lower()});")
    elif fam == "nppsMulC" and fn.startswith("nppsMulC_64f64s"):
        dct = ctype_of(params, "pDst")
        vct = ctype_of(params, "nValue")
        if dct is None or vct is None:
            return None
        amap = {"nValue": "value", "pDst": "d", "nLength": "n",
                "nScaleFactor": "sf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[]({vct} value, {dct} *d, int n, int sf, NppStreamContext c) "
               f"{{ (void)n; (void)sf; (void)c; return {call}; }}")
        body = f"num::run_npps_mulc64i<{dct}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGenNpps, {test_name}) {{\n  {body}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    elif fam in ARITH:
        kind, op = ARITH[fam]
        sct = ctype_of(params, "pSrc1") or ctype_of(params, "pSrc") or ctype_of(params, "pSrcDst")
        s2ct = ctype_of(params, "pSrc2") or sct
        dct = ctype_of(params, "pDst") or ctype_of(params, "pSrcDst")
        if sct is None or dct is None:
            return None
        complex_types = COMPLEX_SHORT | {"Npp32fc", "Npp64fc"}
        is_complex = sct in complex_types or s2ct in complex_types or dct in complex_types
        if is_complex and kind == "binary" and not (sct == s2ct == dct):
            # Mixed real/complex multiplies (32f*32fc, 32s*32sc).
            if op != "Mul":
                return None
            inplace = any(p.name == "pSrcDst" for p in params)
            bct = dct if inplace else s2ct
            sfs = "_Sfs" in fn or any(p.name == "nScaleFactor" for p in params)
            runp = "{0, 1, 2}" if sfs else "{0}"
            amap = {"pSrc1": "a", "pSrc": "a", "pSrc2": "b", "pSrcDst": "d",
                    "pDst": "d", "nLength": "n", "nScaleFactor": "sf",
                    "nppStreamCtx": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            if inplace:
                lam = (f"[](const {sct} *a, {dct} *d, int n, int sf, NppStreamContext c) "
                       f"{{ (void)n; (void)sf; (void)c; return {call}; }}")
            else:
                lam = (f"[](const {sct} *a, const {bct} *b, {dct} *d, int n, int sf, "
                       f"NppStreamContext c) {{ (void)n; (void)sf; (void)c; return {call}; }}")
            body = f"num::run_npps_mixed_mul<{str(inplace).lower()}, {sct}, {bct}, {dct}>"
            test_name = re.sub(r"\W", "_", fn)
            return (f"TEST(NumGenNpps, {test_name}) {{\n  {body}(\n      \"{fn}\",\n"
                    f"      {lam});\n}}\n")
        if is_complex:
            allowed = op in {"Add", "Sub", "Mul", "Div", "AddC", "SubC", "SubCRev",
                             "MulC", "DivC", "Sqr", "Sqrt"}
            if not allowed:
                return None
            if kind == "binary" and not (sct == s2ct == dct):
                return None
            if kind == "unary" and not (sct == dct):
                return None
        inplace = any(p.name == "pSrcDst" for p in params)
        # NPP's mixed-type Div kernel (16s/32s) produces operand-dependent
        # garbage on zero divisors; keep it out of the generated matrix.
        if op == "Div" and sct != s2ct:
            return None
        sfs = "_Sfs" in fn or any(p.name == "nScaleFactor" for p in params)
        cst = ctype_of(params, "nValue") or sct
        def ar_arg(p):
            n = p.name
            if n in ("pSrc1", "pSrc"):
                return "a"
            if n == "pSrc2":
                return "b"
            if n in ("pSrcDst", "pDst"):
                return "d"
            if n == "nLength":
                return "n"
            if n == "nValue":
                return "cv"
            if n == "nScaleFactor":
                return "sf"
            if n == "nppStreamCtx":
                return "c"
            raise KeyError(n)
        call = f"{fn}({', '.join([ar_arg(p) for p in params] + ['c'])})"
        runp = "{0, 1, 2}" if sfs else "{0}"
        if kind == "binary":
            lam = (f"[](const {sct} *a, const {s2ct} *b, {dct} *d, int n, int sf, "
                   f"NppStreamContext c) "
                   f"{{ (void)n; (void)sf; (void)c; return {call}; }}")
            body = f"num::run_npps_binary<{sct}, {s2ct}, {dct}, num::Op::{op}>"
        elif kind == "unary":
            lam = (f"[](const {sct} *a, {dct} *d, int n, int sf, NppStreamContext c) "
                   f"{{ (void)n; (void)sf; (void)c; return {call}; }}")
            body = f"num::run_npps_unary<{sct}, {dct}, num::Op::{op}>"
        else:
            lam = (f"[](const {sct} *a, {cst} cv, {dct} *d, int n, int sf, NppStreamContext c) "
                   f"{{ (void)n; (void)sf; (void)c; return {call}; }}")
            body = f"num::run_npps_const<{sct}, {dct}, {cst}, num::Op::{op}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGenNpps, {test_name}) {{\n  {body}(\n      \"{fn}\",\n"
                f"      {lam},\n      {runp}, {str(inplace).lower()});\n}}\n")
    elif fam in THR:
        t = ctype_of(params, "pSrc")
        if t is None:
            t = ctype_of(params, "pSrcDst")
        if t is None:
            return None
        inplace = any(p.name == "pSrcDst" for p in params)
        call = lam_call(arg_by_name)
        level_ct = {"Npp16sc": "Npp16s", "Npp32fc": "Npp32f",
                    "Npp64fc": "Npp64f"}.get(t, t)
        lam = (f"[](const {t} *s, {t} *d, int n, {level_ct} level, {t} value, int op, "
               f"NppStreamContext c) "
               f"{{ (void)n; (void)level; (void)value; (void)op; (void)c; return {call}; }}")
        body = f"num::run_npps_threshold<{t}, {level_ct}, num::ThrKind::{THR[fam]}>"
        args = f"      {lam},\n      {str(inplace).lower()});"
    elif fam == "nppsAddProduct":
        s1 = ctype_of(params, "pSrc1")
        s2 = ctype_of(params, "pSrc2")
        dct = ctype_of(params, "pDst")
        if s1 is None or s2 is None or dct is None:
            return None
        amap = {"pSrc1": "a", "pSrc2": "b", "pDst": "d", "nLength": "n",
                "nScaleFactor": "sf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {s1} *a, const {s2} *b, {dct} *d, int n, int sf, "
               f"NppStreamContext c) {{ (void)n; (void)sf; (void)c; return {call}; }}")
        sfs = any(p.name == "nScaleFactor" for p in params)
        runp = "{0, 1, 2}" if sfs else "{0}"
        body = f"num::run_npps_accum<{s1}, {s2}, {dct}>"
        args = f"      {lam},\n      {runp});"
    elif fam == "nppsAddProductC":
        sct = ctype_of(params, "pSrc")
        vct = ctype_of(params, "nValue")
        dct = ctype_of(params, "pDst")
        if sct is None or vct is None or dct is None:
            return None
        amap = {"pSrc": "a", "nValue": "value", "pDst": "d", "nLength": "n",
                "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {sct} *a, {vct} value, {dct} *d, int n, "
               f"NppStreamContext c) {{ (void)n; (void)c; return {call}; }}")
        body = f"num::run_npps_accum_c<{sct}, {vct}, {dct}>"
        args = f"      {lam});"
    elif fam == "nppsDiv_Round":
        s1 = ctype_of(params, "pSrc1") or ctype_of(params, "pSrc")
        s2 = ctype_of(params, "pSrc2")
        dct = ctype_of(params, "pDst") or ctype_of(params, "pSrcDst")
        if s1 is None or dct is None:
            return None
        inplace = any(p.name == "pSrcDst" for p in params)
        amap = {"pSrc": "a", "pSrc1": "a", "pSrc2": "b", "pDst": "d",
                "pSrcDst": "d", "nLength": "n",
                "nRndMode": "static_cast<NppRoundMode>(mode)",
                "nScaleFactor": "sf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        if inplace:
            lam = (f"[](const {s1} *a, {dct} *d, int n, int mode, int sf, "
                   f"NppStreamContext c) {{ (void)n; (void)mode; (void)sf; (void)c; "
                   f"return {call}; }}")
            body = f"num::run_npps_div_round<true, {s1}, {dct}, {dct}>"
        else:
            lam = (f"[](const {s1} *a, const {s2} *b, {dct} *d, int n, int mode, int sf, "
                   f"NppStreamContext c) {{ (void)n; (void)mode; (void)sf; (void)c; "
                   f"return {call}; }}")
            body = f"num::run_npps_div_round<false, {s1}, {s2}, {dct}>"
        args = f"      {lam});"
    elif fam == "nppsNormalize":
        sct = ctype_of(params, "pSrc")
        dct = ctype_of(params, "pDst")
        sub_ct = ctype_of(params, "vSub")
        div_ct = ctype_of(params, "vDiv")
        if sct is None or dct is None or sub_ct is None or div_ct is None:
            return None
        amap = {"pSrc": "a", "pDst": "d", "nLength": "n", "vSub": "sub",
                "vDiv": "div", "nScaleFactor": "sf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {sct} *a, {dct} *d, int n, {sub_ct} sub, {div_ct} div, "
               f"int sf, NppStreamContext c) {{ (void)n; (void)sub; (void)div; "
               f"(void)sf; (void)c; return {call}; }}")
        body = f"num::run_npps_normalize<{sct}, {div_ct}>"
        sfs = any(p.name == "nScaleFactor" for p in params)
        runp = "{0, 1, 2}" if sfs else "{0}"
        args = f"      {lam},\n      {runp});"
    elif fam == "nppsMul_Low":
        s1 = ctype_of(params, "pSrc1")
        s2 = ctype_of(params, "pSrc2")
        dct = ctype_of(params, "pDst")
        if s1 is None or s2 is None or dct is None:
            return None
        amap = {"pSrc1": "a", "pSrc2": "b", "pDst": "d", "nLength": "n",
                "nScaleFactor": "sf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {s1} *a, const {s2} *b, {dct} *d, int n, int sf, "
               f"NppStreamContext c) {{ (void)n; (void)sf; (void)c; return {call}; }}")
        body = f"num::run_npps_mullow<{s1}, {s2}, {dct}>"
        args = f"      {lam});"
    elif fam == "nppsMulC_Low":
        sct = ctype_of(params, "pSrc")
        vct = ctype_of(params, "nValue")
        dct = ctype_of(params, "pDst")
        if sct is None or vct is None or dct is None:
            return None
        amap = {"pSrc": "a", "nValue": "value", "pDst": "d", "nLength": "n",
                "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {sct} *a, {vct} value, {dct} *d, int n, "
               f"NppStreamContext c) {{ (void)n; (void)c; return {call}; }}")
        body = f"num::run_npps_mulclow<{sct}, {vct}, {dct}>"
        args = f"      {lam});"
    elif fam == "nppsMulC_64f64s":
        dct = ctype_of(params, "pDst")
        vct = ctype_of(params, "nValue")
        if dct is None or vct is None:
            return None
        amap = {"nValue": "value", "pDst": "d", "nLength": "n",
                "nScaleFactor": "sf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[]({vct} value, {dct} *d, int n, int sf, NppStreamContext c) "
               f"{{ (void)n; (void)sf; (void)c; return {call}; }}")
        body = f"num::run_npps_mulc64i<{dct}>"
        args = f"      {lam});"
    elif fam == "nppsSumLn":
        sct = ctype_of(params, "pSrc")
        dct = ctype_of(params, "pDst")
        if sct is None or dct is None:
            return None
        amap = {"pSrc": "a", "pDst": "d", "nLength": "n",
                "pDeviceBuffer": "buf", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {sct} *a, {dct} *d, int n, Npp8u *buf, NppStreamContext c) "
               f"{{ (void)n; (void)buf; (void)c; return {call}; }}")
        body = f"num::run_npps_sumln<{sct}, {dct}>"
        args = f"      {lam});"
    elif fam == "nppsSumLnGetBufferSize":
        lam = ("[](int n, int *sz, NppStreamContext c) {\n"
               "#ifdef USE_NVIDIA_NPP_TESTS\n"
               f"        size_t s = 0; NppStatus st = {fn}((size_t)n, &s, c);\n"
               "        if (st == NPP_NO_ERROR) *sz = (int)(s > 2147483647u ? 2147483647u : s);\n"
               "        return st;\n"
               "#else\n"
               f"        return {fn}(n, sz, c);\n"
               "#endif\n"
               "      }")
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGenNpps, {test_name}) {{\n"
                f"  num::run_npps_buffer_size(\n      \"{fn}\",\n      {lam});\n}}\n")
    elif fam in ("nppsCauchy", "nppsCauchyD", "nppsCauchyDD2"):
        t = ctype_of(params, "pSrcDst")
        if t is None:
            return None
        amap = {"pSrcDst": "d", "pD2FVal": "d2", "nLength": "n",
                "nParam": "param", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        if fam == "nppsCauchyDD2":
            lam = (f"[]({t} *d, {t} *d2, int n, {t} param, NppStreamContext c) "
                   f"{{ (void)n; (void)param; (void)c; return {call}; }}")
            body = f"num::run_npps_cauchy_dd2<{t}>"
        elif fam == "nppsCauchyD":
            lam = (f"[]({t} *d, int n, {t} param, NppStreamContext c) "
                   f"{{ (void)n; (void)param; (void)c; return {call}; }}")
            body = f"num::run_npps_cauchy_d<{t}>"
        else:
            lam = (f"[]({t} *d, int n, {t} param, NppStreamContext c) "
                   f"{{ (void)n; (void)param; (void)c; return {call}; }}")
            body = f"num::run_npps_cauchy<{t}>"
        args = f"      {lam});"
    else:
        return None

    test_name = re.sub(r"\W", "_", fn)
    return (f"TEST(NumGenNpps, {test_name}) {{\n  {body}(\n      \"{fn}\",\n"
            f"{args}\n}}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CATALOG)) if r["module"] == args.module and r["ctx"] == "1"]
    out, skipped = [], []
    for row in rows:
        text = emit(row)
        if text is None:
            skipped.append(row["function_name"])
        else:
            out.append(text)
    header = ("// GENERATED by tools/gen_npps_tests.py -- DO NOT EDIT.\n"
              f"// module: {args.module}\n#include \"num_npps.h\"\n\n")
    if args.emit:
        Path(args.out).write_text(header + "\n".join(out))
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev[args.module] = dict(emitted=len(out), skipped=[{"function": s, "reason": "unsupported"} for s in skipped])
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests to {args.out}; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
