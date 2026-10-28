#!/usr/bin/env python3
"""Generate Ctx + non-Ctx wrappers for the npps scalar / bitwise / shift /
reduction arithmetic families (api_analysis/packages/p1.1c.csv).  Every `_Ctx`
entry dispatches to a single npps::unary executor; the plain entry forwards to
it via nppGetStreamContext().

Usage: python3 mpp_scripts/gen_unary_wrappers.py [--emit]
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "api_analysis" / "packages" / "p1.1c.csv"
OUT = ROOT / "src" / "npps" / "npps_arithmetic_operations" / "npps_unary_generated.cpp"

CTY = {"8u":"Npp8u","16u":"Npp16u","32u":"Npp32u","16s":"Npp16s","32s":"Npp32s",
       "16sc":"Npp16sc","32sc":"Npp32sc","32f":"Npp32f","64f":"Npp64f",
       "32fc":"Npp32fc","64fc":"Npp64fc","64s":"Npp64s"}

SCALAR_OP = {
    "nppsSqrt":"SqrtOp","nppsSqr":"SqrOp","nppsExp":"ExpOp","nppsLn":"LnOp",
    "npps10Log10":"Log10Op","nppsAbs":"AbsOp","nppsArctan":"ArctanOp","nppsCubrt":"CubrtOp",
}
DT_TOKENS = ("16sc","32sc","32fc","64fc","32s","16s","32u","16u","64f","64s","32f","8u")

def ctype(dt):
    return CTY[dt]

def parse_types(base, family):
    # strip family prefix, then parse leading dtype token(s) and a trailing suffix
    rest = base[len(family) + 1:] if base.startswith(family) else base
    src = None
    for t in DT_TOKENS:
        if rest.startswith(t):
            src = t; rest = rest[len(t):]; break
    for suf in ("_Sfs","_ISfs","_Ctx","_I"):
        if rest.endswith(suf):
            rest = rest[: -len(suf)]
    dst = rest if rest else src
    return src, dst

def scalar_types(base, family):
    return parse_types(base, family)

def sumln_dts(base):
    src = None; rest = base.split("_", 1)[1] if "_" in base else base
    for t in ("16s","32f","64f","32s"):
        if rest.startswith(t):
            src = t; rest = rest[len(t):]; break
    dst = rest if rest else src
    return src, dst

def param_names(params):
    out = []
    for x in params.split(","):
        n = x.replace("*", " ").strip().split()
        out.append(n[-1] if n else "")
    return out

def executor_call(r):
    name = r["function_name"]
    family = r["family"]
    p = r["params"]
    stream = "nppStreamCtx.hStream"
    base = name[:-4] if name.endswith("_Ctx") else name
    if family in SCALAR_OP:
        op = SCALAR_OP[family]
        inplace = "pSrcDst" in p
        sf = "nScaleFactor" in p
        src, dst = scalar_types(base, family)
        # nppsAbs has no scale factor; integer abs still uses the scaled path with sf=0.
        int_abs = (family == "nppsAbs" and src in ("16s","32s"))
        use_sfs = sf or int_abs
        if inplace:
            sfs_args = f", {('nScaleFactor' if sf else '0')}" if use_sfs else ""
            return f"npps::unary::run{'SfsI' if use_sfs else 'FI'}<npps::unary::{op},{ctype(src)}>(pSrcDst, nLength{sfs_args}, {stream})"
        if use_sfs:
            return f"npps::unary::runSfs<npps::unary::{op},{ctype(src)},{ctype(dst)}>(pSrc, pDst, nLength, {'nScaleFactor' if sf else '0'}, {stream})"
        return f"npps::unary::runF<npps::unary::{op},{ctype(src)},{ctype(dst)}>(pSrc, pDst, nLength, {stream})"
    if family in ("nppsAnd","nppsOr","nppsXor"):
        a = ctype(r["dtype"]); o = family[4:]
        fn = "bitwise" + o
        if "pSrcDst" in p:
            return f"npps::unary::{fn}I<{a}>(pSrc, pSrcDst, nLength, {stream})"
        return f"npps::unary::{fn}<{a}>(pSrc1, pSrc2, pDst, nLength, {stream})"
    if family == "nppsNot":
        a = ctype(r["dtype"])
        if "pSrcDst" in p:
            return f"npps::unary::notOpI<{a}>(pSrcDst, nLength, {stream})"
        return f"npps::unary::notOp<{a}>(pSrc, pDst, nLength, {stream})"
    if family in ("nppsLShiftC","nppsRShiftC"):
        a = ctype(r["dtype"]); op = "lShiftC" if family == "nppsLShiftC" else "rShiftC"
        if "pSrcDst" in p:
            return f"npps::unary::{op}I<{a}>(nValue, pSrcDst, nLength, {stream})"
        return f"npps::unary::{op}<{a}>(pSrc, nValue, pDst, nLength, {stream})"
    if family == "nppsMul_Low":
        return "npps::unary::mulLow(pSrc1, pSrc2, pDst, nLength, nScaleFactor, " + stream + ")"
    if family == "nppsSumLn":
        src, dst = sumln_dts(base)
        return f"npps::unary::sumLn<{ctype(src)},{ctype(dst)}>(pSrc, nLength, pDst, pDeviceBuffer, {stream})"
    if family == "nppsSumLnGetBufferSize":
        src, _ = sumln_dts(base)
        return f"npps::unary::sumLnGetBufferSize<{ctype(src)}>(nLength, hpBufferSize)"
    if family == "nppsCauchy":
        return "npps::unary::cauchy(pSrcDst, nLength, nParam, " + stream + ")"
    if family == "nppsCauchyD":
        return "npps::unary::cauchyD(pSrcDst, nLength, nParam, " + stream + ")"
    if family == "nppsCauchyDD2":
        return "npps::unary::cauchyDD2(pSrcDst, pD2FVal, nLength, nParam, " + stream + ")"
    raise ValueError("unhandled " + name)

def main():
    emit = "--emit" in sys.argv
    rows = list(csv.DictReader(open(CSV)))
    W = []
    for r in rows:
        if int(r["ctx"]) != 1:
            continue
        fn = r["function_name"]
        plain = fn[:-4] if fn.endswith("_Ctx") else fn
        sig = r["signature"]  # Ctx signature
        call = executor_call(r)
        W.append(sig + " {")
        W.append("  return " + call + ";")
        W.append("}")
        W.append("")
        p2 = r["params"]
        stripped = p2[:-len(", NppStreamContext nppStreamCtx")] if p2.endswith(", NppStreamContext nppStreamCtx") else p2
        psig = f"NppStatus {plain}({stripped})"
        pnames = ", ".join(param_names(stripped))
        W.append(psig + " {")
        W.append("  NppStreamContext nppStreamCtx;")
        W.append("  NppStatus status = nppGetStreamContext(&nppStreamCtx);")
        W.append("  if (status != NPP_NO_ERROR) { return status; }")
        W.append(f"  return {fn}({pnames}, nppStreamCtx);")
        W.append("}")
        W.append("")
    header = [
        "// GENERATED by mpp_scripts/gen_unary_wrappers.py -- DO NOT EDIT.",
        "// Unary scalar / bitwise / shift / reduction families (p1.1c).",
        "",
        '#include "npp.h"',
        '#include "npps_unary_executor.h"',
        "",
    ]
    text = "\n".join(header) + "\n" + "\n".join(W)
    if emit:
        OUT.write_text(text)
        print("wrote %s (%d bytes)" % (OUT, len(text)))
    else:
        print(text)

if __name__ == "__main__":
    main()
