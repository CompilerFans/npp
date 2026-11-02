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
}


def layout(fn: str):
    """Return (channels, inplace) from an nppi layout suffix, or None."""
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_(AC|C)(\d)(I?)[RP]", base)
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


def emit_test(row: dict, spec: dict) -> Optional[str]:
    fn = row["function_name"]
    par = layout(fn)
    if par is None:
        return None
    C, inplace = par
    dtype = row["dtype"]
    ctype = DTYPE_CTYPE.get(dtype)
    if ctype is None:
        return None
    # Complex-short types need a comparator that does not exist in the
    # harness ValueKind set; they are added separately later.
    if dtype in ("16sc", "32sc"):
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
