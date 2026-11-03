#!/usr/bin/env python3
"""Generate CPU-golden numerical conformance tests from the API catalog.

Every generated TEST drives one ``_Ctx`` entry point through the full
small/medium/large geometry matrix in ``test/unit/num/num_suite.h``.  The
expected values are computed by the reference formulas in that header (no
golden recordings).  A family is emitted only when its semantics are known;
everything else is listed in the JSON report so the coverage gap stays
visible.

Usage:
  python3 tools/gen_num_tests.py --module nppi_arithmetic_and_logical_operations.h
  python3 tools/gen_num_tests.py --module ... --emit
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_ref_tests as G  # signature parsing / catalog helpers

CATALOG = ROOT / "api_analysis" / "api_catalog.csv"

DTYPE_CTYPE = {
    "8u": "Npp8u", "8s": "Npp8s", "16u": "Npp16u", "16s": "Npp16s",
    "16f": "Npp16f", "32u": "Npp32u", "32s": "Npp32s", "32f": "Npp32f",
    "64f": "Npp64f", "64u": "Npp64u", "64s": "Npp64s",
    "16sc": "Npp16sc", "32sc": "Npp32sc", "32fc": "Npp32fc", "64fc": "Npp64fc",
    "8uc": "Npp8uc", "8sc": "Npp8sc", "16uc": "Npp16uc", "32uc": "Npp32uc",
}

# family -> (runner kind, reference Op, scale-factor driven)
# Entry points that the NPP 13.4 headers declare but its libraries do not
# export.  They are exercised against the in-tree MPP build only.
NPP_MISSING = {
    "nppiSub_32s_C4RSfs_Ctx",
    "nppiSub_32s_C4IRSfs_Ctx",
    # NPP 13.4's 8u C4-in-place MulDeviceCScale entry point segfaults inside
    # libnppial itself (reproduced with a standalone probe), so only the
    # in-tree build exercises it.
    "nppiMulDeviceCScale_8u_C4IR_Ctx",
}

SPECS: Dict[str, dict] = {
    "nppiAdd": dict(kind="binary", op="Add", sfs=True),
    "nppiSub": dict(kind="binary", op="Sub", sfs=True),
    "nppiMul": dict(kind="binary", op="Mul", sfs=True),
    "nppiDiv": dict(kind="binary", op="Div", sfs=True),
    "nppiAnd": dict(kind="binary", op="And", sfs=False),
    "nppiOr": dict(kind="binary", op="Or", sfs=False),
    "nppiXor": dict(kind="binary", op="Xor", sfs=False),
    "nppiAbsDiff": dict(kind="binary", op="AbsDiff", sfs=False),
    "nppiSqr": dict(kind="unary", op="Sqr", sfs=True),
    "nppiSqrt": dict(kind="unary", op="Sqrt", sfs=True),
    "nppiLn": dict(kind="unary", op="Ln", sfs=True),
    "nppiExp": dict(kind="unary", op="Exp", sfs=True),
    "nppiAbs": dict(kind="unary", op="Abs", sfs=False),
    "nppiNot": dict(kind="unary", op="Not", sfs=False),
    "nppiAddC": dict(kind="const", op="AddC", sfs=True),
    "nppiSubC": dict(kind="const", op="SubC", sfs=True),
    "nppiMulC": dict(kind="const", op="MulC", sfs=True),
    "nppiDivC": dict(kind="const", op="DivC", sfs=True),
    "nppiAndC": dict(kind="const", op="AndC", sfs=False),
    "nppiOrC": dict(kind="const", op="OrC", sfs=False),
    "nppiXorC": dict(kind="const", op="XorC", sfs=False),
    "nppiLShiftC": dict(kind="const", op="LShiftC", sfs=False),
    "nppiRShiftC": dict(kind="const", op="RShiftC", sfs=False),
    "nppiAbsDiffC": dict(kind="const", op="AbsDiffC", sfs=False),
    "nppiThreshold": dict(kind="threshold", thr="Threshold", op_params=True),
    "nppiThreshold_Val": dict(kind="threshold", thr="ThresholdVal", op_params=True),
    "nppiThreshold_GT": dict(kind="threshold", thr="GT", op_params=False),
    "nppiThreshold_LT": dict(kind="threshold", thr="LT", op_params=False),
    "nppiThreshold_GTVal": dict(kind="threshold", thr="GTVal", op_params=False),
    "nppiThreshold_LTVal": dict(kind="threshold", thr="LTVal", op_params=False),
    "nppiThreshold_LTValGTVal": dict(kind="threshold", thr="LTValGTVal", op_params=False),
    "nppiCompare": dict(kind="compare"),
    "nppiCompareC": dict(kind="compare_const"),
    "nppiSet": dict(kind="set"),
    "nppiCopy": dict(kind="copy"),
    "nppiTranspose": dict(kind="transpose"),
    "nppiDup": dict(kind="dup"),
    "nppiSwapChannels": dict(kind="swap"),
    "nppiCopyConstBorder": dict(kind="border", border="Const"),
    "nppiCopyReplicateBorder": dict(kind="border", border="Replicate"),
    "nppiCopyWrapBorder": dict(kind="border", border="Wrap"),
    "nppiCopySubpix": dict(kind="subpix"),
    "nppiConvert": dict(kind="convert"),
    "nppiScale": dict(kind="scale"),
    "nppiCompareEqualEps": dict(kind="compare_eps"),
    "nppiCompareEqualEpsC": dict(kind="compare_eps_const"),
    "nppiAddDeviceC": dict(kind="devicec", op="AddC", sfs=True),
    "nppiSubDeviceC": dict(kind="devicec", op="SubC", sfs=True),
    "nppiMulDeviceC": dict(kind="devicec", op="MulC", sfs=True),
    "nppiDivDeviceC": dict(kind="devicec", op="DivC", sfs=True),
    "nppiAbsDiffDeviceC": dict(kind="devicec", op="AbsDiffC", sfs=False),
    "nppiAddProduct": dict(kind="addproduct"),
    "nppiAddSquare": dict(kind="addsquare"),
    "nppiAddWeighted": dict(kind="addweighted"),
    "nppiDiv_Round": dict(kind="divround"),
    "nppiMulScale": dict(kind="mulscale"),
    "nppiMulCScale": dict(kind="mulcscale"),
    "nppiMulDeviceCScale": dict(kind="mulcscale"),
    "nppiAlphaCompC": dict(kind="alphacomp_c"),
    "nppiAlphaComp": dict(kind="alphacomp"),
    "nppiAlphaPremulC": dict(kind="alphapremul_c"),
    "nppiAlphaPremul": dict(kind="alphapremul"),
    "nppiMagnitude": dict(kind="magnitude", sqr=False),
    "nppiMagnitudeSqr": dict(kind="magnitude", sqr=True),
}

# Alpha composition layouts: (channels per pixel, colour channels, has alpha).
ALPHA_LAYOUT = {
    "AC4": (4, 3, True),
    "AC1": (2, 1, True),
    "C1": (1, 1, False),
    "C3": (3, 3, False),
    "C4": (4, 4, False),
}


def emit_alpha(row: dict, spec: dict) -> Optional[str]:
    """AlphaComp / AlphaPremul entry points (measured fixed-point contract)."""
    fn = row["function_name"]
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_(AC4I|AC1I|C1I|C3I|C4I|AC4|AC1|C1|C3|C4)R$", base)
    if m is None:
        return None
    tok = m.group(1)
    inplace = tok.endswith("I")
    layout_tok = tok[:-1] if inplace else tok
    if layout_tok not in ALPHA_LAYOUT:
        return None
    C, A, ac = ALPHA_LAYOUT[layout_tok]
    dtype = row["dtype"]
    if dtype not in ("8u", "16u", "32f"):
        return None
    S = DTYPE_CTYPE[dtype]
    kind = spec["kind"]
    if kind == "alphacomp_c" and layout_tok not in ("C1", "C3", "C4", "AC4"):
        return None
    if kind == "alphacomp" and layout_tok not in ("AC1", "AC4"):
        return None
    if kind == "alphapremul_c" and layout_tok not in ("C1", "C3", "C4", "AC4"):
        return None
    if kind == "alphapremul" and layout_tok not in ("AC4",):
        return None
    const = kind.endswith("_c")
    test_name = re.sub(r"\W", "_", fn)
    if kind == "alphacomp_c":
        lam = (f"[](const {S} *s1, int ss1, {S} a1, const {S} *s2, int ss2, {S} a2, "
               f"{S} *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) "
               f"{{ return {fn}(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); }}")
        prefix = f"num::run_alphacomp_c<{S}, {C}, {A}, {str(ac).lower()}>"
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    if kind == "alphacomp":
        lam = (f"[](const {S} *s1, int ss1, const {S} *s2, int ss2, {S} *d, int ds, "
               f"NppiSize r, NppiAlphaOp op, NppStreamContext c) "
               f"{{ return {fn}(s1, ss1, s2, ss2, d, ds, r, op, c); }}")
        prefix = f"num::run_alphacomp_pixel<{S}, {C}, {A}>"
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")
    # AlphaPremul families
    const_tok = "true" if const else "false"
    inplace_tok = "true" if inplace else "false"
    prefix = (f"num::run_alphapremul<{S}, {C}, {A}, {const_tok}, {str(ac).lower()}, "
              f"{inplace_tok}>")
    if const:
        if inplace:
            lam = (f"[]({S} a, {S} *sd, int st, NppiSize r, NppStreamContext c) "
                   f"{{ return {fn}(a, sd, st, r, c); }}")
        else:
            lam = (f"[](const {S} *s, int ss, {S} a, {S} *d, int ds, NppiSize r, "
                   f"NppStreamContext c) {{ return {fn}(s, ss, a, d, ds, r, c); }}")
    else:
        if inplace:
            lam = (f"[]({S}, {S} *sd, int st, NppiSize r, NppStreamContext c) "
                   f"{{ return {fn}(sd, st, r, c); }}")
        else:
            lam = (f"[](const {S} *s, int ss, {S}, {S} *d, int ds, NppiSize r, "
                   f"NppStreamContext c) {{ return {fn}(s, ss, d, ds, r, c); }}")
    return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
            f"      {lam});\n}}\n")

THR_TWO_INTERVALS = {"nppiThreshold_LTValGTVal"}


def ctype_of(params, name):
    for p in params:
        if p.name == name:
            return p.ctype
    return None


def thr_arg(p: G.Param, fn: str, two: bool) -> str:
    """Normalized threshold-call argument for one declared parameter."""
    n = p.name
    if n == "pSrc":
        return "s"
    if n in ("pDst", "pSrcDst"):
        return "d"
    if n == "nSrcStep":
        return "ss"
    if n in ("nDstStep", "nSrcDstStep"):
        return "ds"
    if n == "oSizeROI":
        return "r"
    if n == "eComparisonOperation":
        return "static_cast<NppCmpOp>(op)"
    if n == "nEpsilon":
        return "epsf"
    if n == "nppStreamCtx":
        return "c"
    if "hreshold" in n:
        second = two and n.endswith("GT")
        base = "thr2" if second else "thr"
        return base if p.is_array else f"{base}[0]"
    if "alue" in n:
        second = two and n.endswith("GT")
        base = "val2" if second else "val"
        return base if p.is_array else f"{base}[0]"
    raise KeyError(n)


def cmp_arg(p: G.Param, fn: str, is_const: bool) -> str:
    n = p.name
    if n in ("pSrc1", "pSrc"):
        return "a"
    if n == "pSrc2":
        return "b"
    if n in ("pDst", "pSrcDst"):
        return "d"
    if n in ("nSrc1Step", "nSrcStep"):
        return "sa"
    if n == "nSrc2Step":
        return "sb"
    if n == "nDstStep":
        return "ds"
    if n == "oSizeROI":
        return "r"
    if n == "eComparisonOperation":
        return "static_cast<NppCmpOp>(op)"
    if n == "nEpsilon":
        return "epsf"
    if n == "nppStreamCtx":
        return "c"
    if is_const and n in ("pConstants", "nConstant"):
        return "cv" if n == "pConstants" else "cv[0]"
    raise KeyError(n)


def layout(fn: str):
    """Return (channels, inplace) from an nppi layout suffix, or None."""
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_(AC|C)(\d)(I?)(M?)(?:R|CR)", base)
    if not m:
        return None
    ch = int(m.group(2))
    return ch, m.group(3) == "I"


def tol_for(dtype: str):
    if dtype.endswith("f"):
        if dtype == "16f":
            return 2e-3, 2e-3
        if dtype == "64f":
            return 1e-9, 1e-12
        return 1e-5, 1e-6
    return 0.0, 0.0


def arg_expr(p: G.Param, ctx: dict) -> Optional[str]:
    n = p.name
    if n in ("pSrc1", "pSrc"):
        return "a"
    if n == "pSrc2":
        return "b"
    if n == "pSrcDst":
        return "d"
    if n == "pDst":
        return "d"
    if n in ("nSrc1Step", "nSrcStep"):
        return "sa"
    if n == "nSrc2Step":
        return "sb"
    if n in ("nDstStep", "nSrcDstStep"):
        return "sd"
    if n == "oSizeROI":
        return "r"
    if n == "nScaleFactor":
        return "p"
    if n == "nppStreamCtx":
        return "c"
    if n == "nConstant":
        return "scalar"
    if n == "aConstants":
        return "arr"
    return None


def copy_layout(fn: str):
    """Return (SCH, DCH, NCOPY, masked) for a packed nppiCopy variant."""
    base = re.sub(r"_Ctx$", "", fn)
    tok = None
    for t in base.split("_"):
        if re.match(r"^(AC|C|P)\d", t):
            tok = t
            break
    if tok is None or "P" in tok:
        return None
    ms = list(re.finditer(r"(AC|C)(\d)", tok))
    if not ms:
        return None
    pairs = [(m.group(1), int(m.group(2))) for m in ms]
    tail = tok[ms[-1].end():]
    masked = "M" in tail
    selected = tail.startswith("CR") or tail.startswith("CR")
    if len(pairs) == 1:
        sch = dch = pairs[0][1]
        ncopy = 1 if selected else (3 if pairs[0][0] == "AC" else sch)
    else:
        sch, dch = pairs[0][1], pairs[1][1]
        ncopy = 1 if selected else min(sch, dch)
    return sch, dch, ncopy, masked


def dup_layout(fn: str):
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_C1(C3|C4|AC4)R$", base)
    if not m:
        return None
    kind = m.group(1)
    stride = 3 if kind == "C3" else 4
    fill = 3 if kind == "AC4" else stride
    return stride, fill


def swap_layout(fn: str):
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_((?:AC|C)\d(?:C\d)?)(I?)R$", base)
    if not m:
        return None
    tok, inplace = m.group(1), bool(m.group(2))
    pairs = [(a, int(b)) for a, b in re.findall(r"(AC|C)(\d)", tok)]
    c3c4 = tok.startswith("C3C4")
    if len(pairs) == 1:
        sc = dc = pairs[0][1]
        ncopy = 3 if pairs[0][0] == "AC" else dc
    else:
        sc, dc = pairs[0][1], pairs[1][1]
        ncopy = min(sc, dc)
    return sc, dc, ncopy, c3c4, inplace


def emit_test(row: dict, spec: dict) -> Optional[str]:
    fn = row["function_name"]
    fam = row["family"]
    if spec["kind"].startswith("alpha"):
        return emit_alpha(row, spec)
    if spec["kind"] == "magnitude":
        params = G.parse_params(row["params"])
        if params is None:
            return None
        Si = ctype_of(params, "pSrc")
        Di = ctype_of(params, "pDst")
        if Si is None or Di is None:
            return None
        lam = (f"[](const {Si} *a, int sa, {Di} *d, int ds, NppiSize r, "
               f"NppStreamContext c) {{ (void)sa; (void)ds; (void)r; (void)c; "
               f"return {fn}(a, sa, d, ds, r, c); }}")
        test_name = re.sub(r"\W", "_", fn)
        sqr = str(spec["sqr"]).lower()
        return (f"TEST(NumGen, {test_name}) {{\n  num::run_magnitude<{sqr}>(\n"
                f"      \"{fn}\",\n      {lam});\n}}\n")
    par = layout(fn)
    if par is None:
        if spec["kind"] in ("copy", "set", "transpose", "dup", "swap"):
            C, inplace = 1, False
        else:
            return None
    else:
        C, inplace = par
    dtype = row["dtype"]
    ctype = DTYPE_CTYPE.get(dtype)
    if ctype is None:
        return None
    params = G.parse_params(row["params"])
    if params is None:
        return None
    A = 3 if "_AC" in fn else C
    sfs = "Sfs" in fn or any(p.name == "nScaleFactor" for p in params)
    kind = spec["kind"]

    # Constant type comes from the declared parameter, which may differ from
    # the image type (16f APIs take Npp32f constants).
    cst = None
    per_channel = False
    if kind == "const":
        for p in params:
            if p.name in ("nConstant", "aConstants"):
                cst = p.ctype
                per_channel = p.name == "aConstants"
        if cst is None:
            return None

    # Threshold and compare families use dedicated drivers and a normalized
    # call shape rather than the elementwise generic path.
    if spec["kind"] == "copy":
        mp = re.search(r"_(AC|C|P)(\d)(AC|C|P)(\d)I?R$", re.sub(r"_Ctx$", "", fn))
        if mp is not None and ("P" in (mp.group(1), mp.group(3))):
            src_tok, dst_tok = mp.group(1), mp.group(3)
            ch = int(mp.group(2))
            if ch != int(mp.group(4)):
                return None
            S = ctype_of(params, "pSrc") or ctype_of(params, "aSrc")
            if S is None:
                return None
            to_planar = src_tok != "P"
            # The lambda always takes (pack, packedStep, planes, planeStep),
            # matching run_copy_planar's call.  The NPP signature orders the
            # source pair before the destination pair, so the step names swap
            # for packed->planar vs planar->packed.
            amap = {"pSrc": "pack", "nSrcStep": "ps" if to_planar else "plstep",
                    "aDst": "planes", "aSrc": "planes", "pDst": "pack",
                    "nDstStep": "plstep" if to_planar else "ps",
                    "oSizeROI": "r", "nppStreamCtx": "c"}
            call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
            if to_planar:
                lam = (f"[](const {S} *pack, int ps, {S} *const *planes, int plstep, "
                       f"NppiSize r, NppStreamContext c) "
                       f"{{ (void)ps; (void)plstep; (void)r; (void)c; return {call}; }}")
            else:
                lam = (f"[]({S} *pack, int ps, const {S} *const *planes, int plstep, "
                       f"NppiSize r, NppStreamContext c) "
                       f"{{ (void)ps; (void)plstep; (void)r; (void)c; return {call}; }}")
            prefix = f"num::run_copy_planar<{S}, {ch}, {str(to_planar).lower()}>"
            test_name = re.sub(r"\W", "_", fn)
            return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                    f"      {lam});\n}}\n")
        lay = copy_layout(fn)
        if lay is None:
            return None
        sch, dch, ncopy, masked = lay
        def copy_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n == "pDst": return "d"
            if n == "nSrcStep": return "ss"
            if n == "nDstStep": return "ds"
            if n == "oSizeROI": return "r"
            if n == "pMask": return "mask"
            if n == "nMaskStep": return "ms"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [copy_arg(p) for p in params]
        args.append("c")
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {ctype} *s, int ss, {ctype} *d, int ds, NppiSize r, "
               f"const Npp8u *mask, int ms, NppStreamContext c) "
               f"{{ (void)ss; (void)ds; (void)r; (void)mask; (void)ms; (void)c; return {call}; }}")
        prefix = f"num::run_copy<{ctype}, {sch}, {dch}, {ncopy}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n"
                f"  {prefix}(\n"
                f"      \"{fn}\",\n"
                f"      {lam},\n"
                f"      {str(masked).lower()});\n"
                f"}}\n")

    if spec["kind"] == "set":
        fill_all = "C3CR" in fn or "C4CR" in fn
        masked = "MR" in fn
        vparam = None
        for p in params:
            if p.name in ("nValue", "aValue", "aValues", "oValue"):
                vparam = p
        if vparam is None:
            return None
        vt = vparam.ctype
        per_channel = vparam.name in ("aValue", "aValues")
        def set_arg(p):
            n = p.name
            if n == "pDst": return "d"
            if n == "nDstStep": return "ds"
            if n == "oSizeROI": return "r"
            if n == "nppStreamCtx": return "c"
            if n in ("aValue", "aValues"): return "vals"
            if n in ("nValue", "oValue"): return "scalar"
            if n == "pMask": return "mask"
            if n == "nMaskStep": return "ms"
            raise KeyError(n)
        args = [set_arg(p) for p in params]
        args.append("c")
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {vt} *vals, {vt} scalar, {ctype} *d, int ds, NppiSize r, "
               f"const Npp8u *mask, int ms, NppStreamContext c) "
               f"{{ (void)vals; (void)scalar; (void)ds; (void)r; (void)mask; (void)ms; (void)c; "
               f"return {call}; }}")
        prefix = f"num::run_set<{ctype}, {vt}, {C}, {A}>"
        test_name = re.sub(r"\W", "_", fn)
        # NPP 13.4's 16f Set returns an error and writes nothing; keep those
        # variants for the in-tree CUDA/MACA builds only.
        guard = "#ifndef USE_NVIDIA_NPP_TESTS\n" if dtype == "16f" else ""
        unguard = "#endif\n" if dtype == "16f" else ""
        return (f"{guard}TEST(NumGen, {test_name}) {{\n"
                f"  {prefix}(\n"
                f"      \"{fn}\",\n"
                f"      {lam},\n"
                f"      {str(per_channel).lower()}, {str(fill_all).lower()}, "
                f"{str(masked).lower()});\n"
                f"}}\n{unguard}")

    if spec["kind"] == "transpose":
        lay = layout(fn)
        if lay is None:
            return None
        C, inplace = lay
        if inplace:
            return None
        def tr_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n == "pDst": return "d"
            if n == "nSrcStep": return "ss"
            if n == "nDstStep": return "ds"
            if n == "oSrcROI": return "r"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [tr_arg(p) for p in params] + ["c"]
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {ctype} *s, int ss, {ctype} *d, int ds, NppiSize r, "
               f"NppStreamContext c) {{ (void)ss; (void)ds; (void)r; (void)c; return {call}; }}")
        prefix = f"num::run_transpose<{ctype}, {C}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if spec["kind"] == "dup":
        lay = dup_layout(fn)
        if lay is None:
            return None
        stride, fill = lay
        def dup_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n == "pDst": return "d"
            if n == "nSrcStep": return "ss"
            if n == "nDstStep": return "ds"
            if n == "oDstSizeROI": return "r"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [dup_arg(p) for p in params] + ["c"]
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {ctype} *s, int ss, {ctype} *d, int ds, NppiSize r, "
               f"NppStreamContext c) {{ (void)ss; (void)ds; (void)r; (void)c; return {call}; }}")
        prefix = f"num::run_dup<{ctype}, {stride}, {fill}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if spec["kind"] == "swap":
        lay = swap_layout(fn)
        if lay is None:
            return None
        sc, dc, ncopy, c3c4, inplace = lay
        def sw_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n in ("pDst", "pSrcDst"): return "d"
            if n == "nSrcStep": return "ss"
            if n in ("nDstStep", "nSrcDstStep"): return "ds"
            if n == "oSizeROI": return "r"
            if n == "aDstOrder": return "ord"
            if n == "nValue": return "value"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [sw_arg(p) for p in params] + ["c"]
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {ctype} *s, int ss, {ctype} *d, int ds, NppiSize r, const int *ord, "
               f"{ctype} value, NppStreamContext c) "
               f"{{ (void)ss; (void)ds; (void)r; (void)ord; (void)value; (void)c; return {call}; }}")
        prefix = f"num::run_swap<{ctype}, {sc}, {dc}, {ncopy}, {str(c3c4).lower()}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam},\n      {str(inplace).lower()});\n}}\n")

    if spec["kind"] == "convert":
        base = re.sub(r"_Ctx$", "", fn)
        m = re.search(r"_(AC|C)(\d)", base)
        if m is None:
            return None
        C = int(m.group(2))
        A = 3 if m.group(1) == "AC" else C
        sfs = "Sfs" in fn or any(p.name == "nScaleFactor" for p in params)
        round_mode = any(p.name == "eRoundMode" for p in params)
        sct = dct = None
        for p in params:
            if p.name == "pSrc": sct = p.ctype
            if p.name == "pDst": dct = p.ctype
        if sct is None or dct is None:
            return None
        def cv_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n == "nSrcStep": return "ss"
            if n == "pDst": return "d"
            if n == "nDstStep": return "ds"
            if n == "oSizeROI": return "r"
            if n == "eRoundMode": return "static_cast<NppRoundMode>(mode)"
            if n == "nScaleFactor": return "sf"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [cv_arg(p) for p in params] + ["c"]
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {sct} *s, int ss, {dct} *d, int ds, NppiSize r, int mode, int sf, "
               f"NppStreamContext c) "
               f"{{ (void)ss; (void)ds; (void)r; (void)mode; (void)sf; (void)c; return {call}; }}")
        prefix = f"num::run_convert<{sct}, {dct}, {C}, {A}>"
        test_name = re.sub(r"\W", "_", fn)
        # NPP 13.4 does not implement 32f->16f Convert (error -1027 with a raw
        # context, unstable values otherwise); exercise it on MPP only.
        broken = "Convert_32f16f" in fn
        guard = "#ifndef USE_NVIDIA_NPP_TESTS\n" if broken else ""
        unguard = "#endif\n" if broken else ""
        fin_away = "Convert_16s8s" in fn
        return (f"{guard}TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam},\n      {str(sfs).lower()}, {str(round_mode).lower()}, "
                f"{str(fin_away).lower()});\n}}\n{unguard}")

    if spec["kind"] == "scale":
        sct = ctype_of(params, "pSrc")
        dct = ctype_of(params, "pDst")
        if sct is None or dct is None:
            return None
        A = 3 if "_AC4" in fn else C
        has_hint = any(p.name == "nHint" for p in params)
        minmax = any(p.name == "nMin" for p in params)
        amap = {"pSrc": "a", "nSrcStep": "sa", "pDst": "d", "nDstStep": "ds",
                "oSizeROI": "r", "nHint": "static_cast<NppHintAlgorithm>(hint)",
                "hint": "static_cast<NppHintAlgorithm>(hint)", "nMin": "mn",
                "nMax": "mx", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        lam = (f"[](const {sct} *a, int sa, {dct} *d, int ds, NppiSize r, "
               f"float mn, float mx, int hint, NppStreamContext c) "
               f"{{ (void)sa; (void)ds; (void)r; (void)mn; (void)mx; (void)hint; (void)c; "
               f"return {call}; }}")
        prefix = (f"num::run_scale<{sct}, {dct}, {C}, {A}, {str(minmax).lower()}, "
                  f"{str(has_hint).lower()}>")
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if spec["kind"] == "border":
        lay = layout(fn)
        if lay is None:
            return None
        C, inplace = lay
        if inplace:
            return None
        A = 3 if "_AC4" in fn else C
        vparam = None
        for p in params:
            if p.name in ("nValue", "aValue"):
                vparam = p
        vt = vparam.ctype if vparam else ctype
        per_channel = bool(vparam and vparam.name == "aValue")
        def bd_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n == "nSrcStep": return "ss"
            if n == "oSrcSizeROI": return "ssize"
            if n == "pDst": return "d"
            if n == "nDstStep": return "ds"
            if n == "oDstSizeROI": return "dsize"
            if n == "nTopBorderHeight": return "top"
            if n == "nLeftBorderWidth": return "left"
            if n == "aValue": return "vals"
            if n == "nValue": return "scalar"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [bd_arg(p) for p in params] + ["c"]
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {ctype} *s, int ss, NppiSize ssize, {ctype} *d, int ds, NppiSize dsize, "
               f"int top, int left, const {vt} *vals, {vt} scalar, NppStreamContext c) "
               f"{{ (void)ss; (void)ssize; (void)ds; (void)dsize; (void)top; (void)left; "
               f"(void)vals; (void)scalar; (void)c; return {call}; }}")
        prefix = (f"num::run_border<{ctype}, {vt}, {C}, {A}, "
                  f"num::BorderKind::{spec['border']}>")
        test_name = re.sub(r"\W", "_", fn)
        guard = "#ifndef USE_NVIDIA_NPP_TESTS\n" if False else ""
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam},\n      {str(per_channel).lower()});\n}}\n")

    if spec["kind"] == "subpix":
        lay = layout(fn)
        if lay is None:
            return None
        C, inplace = lay
        if inplace:
            return None
        A = 3 if "_AC4" in fn else C
        def sp_arg(p):
            n = p.name
            if n == "pSrc": return "s"
            if n == "nSrcStep": return "ss"
            if n == "pDst": return "d"
            if n == "nDstStep": return "ds"
            if n == "oDstSizeROI": return "r"
            if n == "nDx": return "dx"
            if n == "nDy": return "dy"
            if n == "nppStreamCtx": return "c"
            raise KeyError(n)
        args = [sp_arg(p) for p in params] + ["c"]
        call = f"{fn}({', '.join(args)})"
        lam = (f"[](const {ctype} *s, int ss, {ctype} *d, int ds, NppiSize r, float dx, float dy, "
               f"NppStreamContext c) "
               f"{{ (void)ss; (void)ds; (void)r; (void)dx; (void)dy; (void)c; return {call}; }}")
        prefix = f"num::run_subpix<{ctype}, {C}, {A}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if spec["kind"] in ("threshold", "compare", "compare_const", "compare_eps", "compare_eps_const"):
        if spec["kind"] == "threshold":
            two = fam in THR_TWO_INTERVALS
            args = [thr_arg(p, fn, two) for p in params]
            args.append("c")
            call = f"{fn}({', '.join(args)})"
            per_channel = any(p.is_array for p in params if "hreshold" in p.name)
            if inplace:
                lam = (f"[](const {ctype} *, int, {ctype} *d, int ds, NppiSize r, "
                       f"const {ctype} *thr, const {ctype} *val, const {ctype} *thr2, "
                       f"const {ctype} *val2, int op, NppStreamContext c) "
                       f"{{ (void)val; (void)val2; (void)thr2; (void)op; return {call}; }}")
            else:
                lam = (f"[](const {ctype} *s, int ss, {ctype} *d, int ds, NppiSize r, "
                       f"const {ctype} *thr, const {ctype} *val, const {ctype} *thr2, "
                       f"const {ctype} *val2, int op, NppStreamContext c) "
                       f"{{ (void)val; (void)val2; (void)thr2; (void)op; return {call}; }}")
            prefix = f"num::run_threshold<{ctype}, {C}, {A}, num::ThrKind::{spec['thr']}>"
            run_params = ("{NPP_CMP_LESS, NPP_CMP_GREATER}" if spec["op_params"] else "{0}")
            test_name = re.sub(r"\W", "_", fn)
            return (f"TEST(NumGen, {test_name}) {{\n"
                    f"  {prefix}(\n"
                    f"      \"{fn}\",\n"
                    f"      {lam},\n"
                    f"      {run_params}, {str(per_channel).lower()}, {str(inplace).lower()});\n"
                    f"}}\n")
        eps_kind = spec["kind"] in ("compare_eps", "compare_eps_const")
        is_const = spec["kind"] in ("compare_const", "compare_eps_const")
        args = [cmp_arg(p, fn, is_const) for p in params]
        args.append("c")
        call = f"{fn}({', '.join(args)})"
        if eps_kind:
            if is_const:
                per_channel = any(p.name == "pConstants" for p in params)
                lam = (f"[](const {ctype} *a, int sa, const {ctype} *cv, Npp8u *d, int ds, "
                       f"NppiSize r, float epsf, NppStreamContext c) "
                       f"{{ (void)sa; (void)ds; (void)r; (void)c; return {call}; }}")
                prefix = f"num::run_compare_eps_const<{ctype}, {C}, {A}>"
                suffix = f", {str(per_channel).lower()}"
            else:
                lam = (f"[](const {ctype} *a, int sa, const {ctype} *b, int sb, Npp8u *d, int ds, "
                       f"NppiSize r, float epsf, NppStreamContext c) "
                       f"{{ (void)sa; (void)sb; (void)ds; (void)r; (void)c; return {call}; }}")
                prefix = f"num::run_compare_eps<{ctype}, {C}, {A}>"
                suffix = ""
            test_name = re.sub(r"\W", "_", fn)
            return (f"TEST(NumGen, {test_name}) {{\n"
                    f"  {prefix}(\n"
                    f"      \"{fn}\",\n"
                    f"      {lam},\n"
                    f"      {{0.0, 0.001, 0.25, 1.0}}{suffix});\n"
                    f"}}\n")
        if is_const:
            per_channel = any(p.name == "pConstants" for p in params)
            lam = (f"[](const {ctype} *a, int sa, const {ctype} *cv, Npp8u *d, int ds, "
                   f"NppiSize r, int op, NppStreamContext c) "
                   f"{{ (void)sa; (void)ds; (void)r; (void)op; (void)c; return {call}; }}")
            prefix = f"num::run_compare_const<{ctype}, {C}, {A}>"
            extra = f", {str(per_channel).lower()}"
        else:
            per_channel = False
            lam = (f"[](const {ctype} *a, int sa, const {ctype} *b, int sb, Npp8u *d, int ds, "
                   f"NppiSize r, int op, NppStreamContext c) "
                   f"{{ (void)sa; (void)sb; (void)ds; (void)r; (void)op; (void)c; return {call}; }}")
            prefix = f"num::run_compare<{ctype}, {C}, {A}>"
            extra = ""
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n"
                f"  {prefix}(\n"
                f"      \"{fn}\",\n"
                f"      {lam},\n"
                f"      {{NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ, NPP_CMP_GREATER_EQ, "
                f"NPP_CMP_GREATER}}{extra});\n"
                f"}}\n")

    if spec["kind"] == "devicec":
        sct = (ctype_of(params, "pSrc1") or ctype_of(params, "pSrc")
               or ctype_of(params, "pSrcDst"))
        dct = ctype_of(params, "pDst") or ctype_of(params, "pSrcDst")
        cct = ctype_of(params, "pConstant") or ctype_of(params, "pConstants")
        if sct is None or dct is None or cct is None:
            return None
        amap = {"pSrc1": "a", "pSrc": "a", "nSrc1Step": "sa", "nSrcStep": "sa",
                "pConstant": "cv", "pConstants": "cv", "pDst": "d", "pSrcDst": "d",
                "nDstStep": "sd", "nSrcDstStep": "sd", "oSizeROI": "r",
                "nScaleFactor": "p", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        rtol, atol = tol_for(dtype)
        # 32s Sfs division rounding is not reproducible across NPP runs; keep
        # the 32s division variants at sf == 0 like the other Div families.
        is_div32s = spec.get("op") == "DivC" and sct == "Npp32s"
        run_params = ("{0, 1, 2}" if spec.get("sfs") and sfs and not is_div32s
                      else "{0}")
        if inplace:
            lam = (f"[]({cct} *cv, {sct} *d, int sd, NppiSize r, int p, NppStreamContext c) "
                   f"{{ (void)sd; (void)r; (void)p; (void)c; return {call}; }}")
            prefix = f"num::run_devicec_inplace<{sct}, {cct}, {C}, {A}, num::Op::{spec['op']}>"
        else:
            lam = (f"[](const {sct} *a, int sa, {cct} *cv, {sct} *d, int sd, NppiSize r, "
                   f"int p, NppStreamContext c) "
                   f"{{ (void)sa; (void)sd; (void)r; (void)p; (void)c; return {call}; }}")
            prefix = f"num::run_devicec<{sct}, {cct}, {C}, {A}, num::Op::{spec['op']}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam},\n      {run_params}, {rtol!r}, {atol!r});\n}}\n")

    if spec["kind"] in ("addproduct", "addsquare", "addweighted"):
        masked = "M" in fn.split("_C1")[-1][:4] or any(p.name == "pMask" for p in params)
        s1 = ctype_of(params, "pSrc1") or ctype_of(params, "pSrc")
        s2 = ctype_of(params, "pSrc2")
        dct = ctype_of(params, "pSrcDst")
        if s1 is None or dct is None or (spec["kind"] == "addproduct" and s2 is None):
            return None
        amap = {"pSrc1": "a", "pSrc": "a", "nSrc1Step": "sa", "nSrcStep": "sa",
                "pSrc2": "b", "nSrc2Step": "sb", "pMask": "m", "nMaskStep": "ms",
                "pSrcDst": "d", "nSrcDstStep": "sd", "oSizeROI": "r",
                "nAlpha": "alpha", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        rtol, atol = (2e-3, 2e-3) if dct == "Npp16f" else (1e-5, 1e-4)
        if spec["kind"] == "addproduct":
            if masked:
                lam = (f"[](const {s1} *a, int sa, const {s2} *b, int sb, const Npp8u *m, int ms, "
                       f"{dct} *d, int sd, NppiSize r, NppStreamContext c) "
                       f"{{ (void)sa; (void)sb; (void)ms; (void)sd; (void)r; (void)c; "
                       f"return {call}; }}")
            else:
                lam = (f"[](const {s1} *a, int sa, const {s2} *b, int sb, {dct} *d, int sd, "
                       f"NppiSize r, NppStreamContext c) "
                       f"{{ (void)sa; (void)sb; (void)sd; (void)r; (void)c; return {call}; }}")
            prefix = (f"num::run_add_product<{s1}, {s2}, {dct}, {C}, {A}, "
                      f"{str(masked).lower()}>")
        elif spec["kind"] == "addsquare":
            if masked:
                lam = (f"[](const {s1} *a, int sa, const Npp8u *m, int ms, {dct} *d, int sd, "
                       f"NppiSize r, NppStreamContext c) "
                       f"{{ (void)sa; (void)ms; (void)sd; (void)r; (void)c; return {call}; }}")
            else:
                lam = (f"[](const {s1} *a, int sa, {dct} *d, int sd, NppiSize r, "
                       f"NppStreamContext c) {{ (void)sa; (void)sd; (void)r; (void)c; "
                       f"return {call}; }}")
            prefix = f"num::run_add_square<{s1}, {dct}, {C}, {A}, {str(masked).lower()}>"
        else:
            if masked:
                lam = (f"[](const {s1} *a, int sa, const Npp8u *m, int ms, {dct} *d, int sd, "
                       f"NppiSize r, float alpha, NppStreamContext c) "
                       f"{{ (void)sa; (void)ms; (void)sd; (void)r; (void)alpha; (void)c; "
                       f"return {call}; }}")
            else:
                lam = (f"[](const {s1} *a, int sa, {dct} *d, int sd, NppiSize r, "
                       f"float alpha, NppStreamContext c) "
                       f"{{ (void)sa; (void)sd; (void)r; (void)alpha; (void)c; return {call}; }}")
            prefix = f"num::run_add_weighted<{s1}, {dct}, {C}, {A}, {str(masked).lower()}>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam},\n      {{0}}, {rtol!r}, {atol!r});\n}}\n")

    if spec["kind"] == "divround":
        sct = ctype_of(params, "pSrc1") or ctype_of(params, "pSrc")
        dct = ctype_of(params, "pDst") or ctype_of(params, "pSrcDst")
        if sct is None or dct is None:
            return None
        amap = {"pSrc1": "a", "pSrc": "a", "nSrc1Step": "sa", "nSrcStep": "sa",
                "pSrc2": "b", "nSrc2Step": "sb", "pDst": "d", "pSrcDst": "d",
                "nDstStep": "sd", "nSrcDstStep": "sd", "oSizeROI": "r",
                "rndMode": "static_cast<NppRoundMode>(mode)",
                "nRndMode": "static_cast<NppRoundMode>(mode)",
                "eRoundMode": "static_cast<NppRoundMode>(mode)",
                "nScaleFactor": "param", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        if inplace:
            lam = (f"[](const {sct} *a, int sa, {sct} *d, int sd, NppiSize r, int mode, "
                   f"int param, NppStreamContext c) "
                   f"{{ (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; "
                   f"return {call}; }}")
            prefix = f"num::run_div_round2d<{sct}, {C}, {A}, true>"
        else:
            lam = (f"[](const {sct} *a, int sa, const {sct} *b, int sb, {sct} *d, int sd, "
                   f"NppiSize r, int mode, int param, NppStreamContext c) "
                   f"{{ (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; "
                   f"(void)c; return {call}; }}")
            prefix = f"num::run_div_round2d<{sct}, {C}, {A}, false>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if spec["kind"] == "mulscale":
        sct = ctype_of(params, "pSrc1") or ctype_of(params, "pSrc")
        dct = ctype_of(params, "pDst") or ctype_of(params, "pSrcDst")
        if sct is None or dct is None:
            return None
        amap = {"pSrc1": "a", "pSrc": "a", "nSrc1Step": "sa", "nSrcStep": "sa",
                "pSrc2": "b", "nSrc2Step": "sb", "pDst": "d", "pSrcDst": "d",
                "nDstStep": "sd", "nSrcDstStep": "sd", "oSizeROI": "r",
                "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        if inplace:
            lam = (f"[](const {sct} *a, int sa, {sct} *d, int sd, NppiSize r, "
                   f"NppStreamContext c) {{ (void)sa; (void)sd; (void)r; (void)c; "
                   f"return {call}; }}")
            prefix = f"num::run_mul_scale<{sct}, {C}, {A}, true>"
        else:
            lam = (f"[](const {sct} *a, int sa, const {sct} *b, int sb, {sct} *d, int sd, "
                   f"NppiSize r, NppStreamContext c) "
                   f"{{ (void)sa; (void)sb; (void)sd; (void)r; (void)c; return {call}; }}")
            prefix = f"num::run_mul_scale<{sct}, {C}, {A}, false>"
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    if spec["kind"] == "mulcscale":
        sct = (ctype_of(params, "pSrc1") or ctype_of(params, "pSrc")
               or ctype_of(params, "pSrcDst"))
        dct = ctype_of(params, "pDst") or ctype_of(params, "pSrcDst")
        if sct is None or dct is None:
            return None
        devicec = "DeviceC" in fn
        amap = {"pSrc1": "a", "pSrc": "a", "nSrc1Step": "sa", "nSrcStep": "sa",
                "nConstant": "cv[0]", "aConstants": "cv", "pConstant": "cv",
                "pConstants": "cv",
                "pDst": "d", "pSrcDst": "d", "nDstStep": "sd", "nSrcDstStep": "sd",
                "oSizeROI": "r", "nppStreamCtx": "c"}
        call = f"{fn}({', '.join([amap[p.name] for p in params] + ['c'])})"
        if inplace:
            lam = (f"[](const {sct} *cv, {sct} *d, int sd, NppiSize r, NppStreamContext c) "
                   f"{{ (void)sd; (void)r; (void)c; return {call}; }}")
            prefix = (f"num::run_mul_cscale<{sct}, {C}, {A}, true, "
                      f"{str(devicec).lower()}>")
        else:
            lam = (f"[](const {sct} *a, int sa, const {sct} *cv, {sct} *d, int sd, "
                   f"NppiSize r, NppStreamContext c) "
                   f"{{ (void)sa; (void)sd; (void)r; (void)c; return {call}; }}")
            prefix = (f"num::run_mul_cscale<{sct}, {C}, {A}, false, "
                      f"{str(devicec).lower()}>")
        test_name = re.sub(r"\W", "_", fn)
        return (f"TEST(NumGen, {test_name}) {{\n  {prefix}(\n      \"{fn}\",\n"
                f"      {lam});\n}}\n")

    args: List[str] = []
    for p in params:
        e = arg_expr(p, {})
        if e is None:
            return None
        args.append(e)
    # parse_params drops the stream context; append it explicitly so the
    # generated call always targets the Ctx entry point (NPP 13 removed the
    # non-Ctx ABI).
    args.append("c")
    call = f"{fn}({', '.join(args)})"

    fam = row["family"]
    is_div = fam in ("nppiDiv", "nppiDivC")
    # NPP's 32s Sfs division rounding is not reproducible (sf>0 results differ
    # between otherwise identical inputs), so 32s is exercised at sf=0 only.
    run_params = "{0, 1, 2}" if spec.get("sfs") and sfs and not (is_div and dtype == "32s") else "{0}"
    rtol, atol = tol_for(dtype)
    op = spec["op"]

    if kind == "binary" and not inplace:
        lam = (f"[](const {ctype} *a, int sa, const {ctype} *b, int sb, {ctype} *d, int sd, "
               f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        prefix = f"num::run_binary<{ctype}, {ctype}, {C}, {A}, num::Op::{op}>"
    elif kind == "binary" and inplace:
        lam = (f"[](const {ctype} *a, int sa, {ctype} *d, int sd, "
               f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        prefix = f"num::run_binary_inplace<{ctype}, {C}, {A}, num::Op::{op}>"
    elif kind == "unary" and not inplace:
        lam = (f"[](const {ctype} *a, int sa, {ctype} *d, int sd, "
               f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        prefix = f"num::run_unary<{ctype}, {ctype}, {C}, {A}, num::Op::{op}>"
    elif kind == "unary" and inplace:
        lam = (f"[]({ctype} *d, int sd, NppiSize r, int p, NppStreamContext c) "
               f"{{ (void)p; return {call}; }}")
        prefix = f"num::run_unary_inplace<{ctype}, {C}, {A}, num::Op::{op}>"
    elif kind == "const" and not inplace:
        if per_channel:
            lam = (f"[](const {ctype} *a, int sa, const {cst} *arr, {cst}, {ctype} *d, int sd, "
                   f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        else:
            lam = (f"[](const {ctype} *a, int sa, const {cst} *, {cst} scalar, {ctype} *d, int sd, "
                   f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        prefix = (f"num::run_const<{ctype}, {ctype}, {cst}, {C}, {A}, num::Op::{op}>")
    elif kind == "const" and inplace:
        if per_channel:
            lam = (f"[](const {cst} *arr, {cst}, {ctype} *d, int sd, "
                   f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        else:
            lam = (f"[](const {cst} *, {cst} scalar, {ctype} *d, int sd, "
                   f"NppiSize r, int p, NppStreamContext c) {{ (void)p; return {call}; }}")
        prefix = f"num::run_const_inplace<{ctype}, {cst}, {C}, {A}, num::Op::{op}>"
    else:
        return None

    test_name = re.sub(r"\W", "_", fn)
    nonneg = spec["op"] == "RShiftC" and dtype == "8s" and C > 1
    if kind == "const":
        extra = f", {str(per_channel).lower()}, {str(nonneg).lower()}, {str(is_div).lower()}"
    elif kind == "binary":
        extra = f", {str(is_div).lower()}"
    else:
        extra = ""
    guard = "#ifndef USE_NVIDIA_NPP_TESTS\n" if fn in NPP_MISSING else ""
    unguard = "#endif\n" if fn in NPP_MISSING else ""
    return (
        f"{guard}TEST(NumGen, {test_name}) {{\n"
        f"  {prefix}(\n"
        f"      \"{fn}\",\n"
        f"      {lam},\n"
        f"      {run_params}, {rtol!r}, {atol!r}{extra});\n"
        f"}}\n{unguard}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="nppi_arithmetic_and_logical_operations.h")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(CATALOG)) if r["module"] == args.module and r["ctx"] == "1"]
    out: List[str] = []
    skipped: List[dict] = []
    for row in rows:
        fam = row["family"]
        spec = SPECS.get(fam)
        if spec is None:
            skipped.append(dict(function=row["function_name"], family=fam, reason="no family spec"))
            continue
        text = emit_test(row, spec)
        if text is None:
            skipped.append(dict(function=row["function_name"], family=fam, reason="unsupported signature/type"))
            continue
        if row["function_name"] in NPP_MISSING and "#ifndef USE_NVIDIA_NPP_TESTS" not in text:
            text = "#ifndef USE_NVIDIA_NPP_TESTS\n" + text + "#endif\n"
        out.append(text)

    header = (
        "// GENERATED by tools/gen_num_tests.py -- DO NOT EDIT.\n"
        f"// module: {args.module}\n"
        "#include \"num_suite.h\"\n\n"
    )
    body = header + "\n".join(out)
    if args.emit:
        dest = Path(args.out) if args.out else (
            ROOT / "test" / "unit" / "num" / ("test_gen_" + re.sub(r"\W", "_", args.module) + ".cpp"))
        dest.write_text(body)
        report = ROOT / "test" / "unit" / "num" / "generated_report.json"
        prev = {}
        if report.exists():
            try:
                prev = json.loads(report.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev[args.module] = dict(emitted=len(out), skipped=skipped)
        report.write_text(json.dumps(prev, indent=1))
        print(f"emitted {len(out)} tests to {dest}; skipped {len(skipped)}")
    else:
        print(f"would emit {len(out)} tests; skipped {len(skipped)}")
        for s in skipped[:20]:
            print("  skip", s["function"], s["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
