#!/usr/bin/env python3
"""Generate reference unit tests for the public NPP API surface.

Each generated test calls the `_Ctx` entry point (NPP 13 removed the non-Ctx
ABI), records the status and output buffers through `npp_diff::Case`, and is
meant to run against official NVIDIA NPP on the 5090 first (`record`), then
against mcMPP (`verify`).

The generator synthesizes arguments conservatively from parameter names and
emits a skip manifest for functions it cannot construct safely.  Irregular
families are expected to be refined by module owners with per-module
overrides/skips; the generator core stays shared.

Usage:
  python3 tools/gen_ref_tests.py --module npps_arithmetic_and_logical_operations.h
  python3 tools/gen_ref_tests.py --all --chunk-size 250
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "API"
OUT_DIR = ROOT / "test" / "unit" / "ref"
CATALOG = ROOT / "api_analysis" / "api_catalog.csv"

SCALAR = {
    "Npp8u": ("U8", 1), "Npp8s": ("I8", 1), "Npp16u": ("U16", 2),
    "Npp16s": ("I16", 2), "Npp16f": ("F16", 2), "Npp32u": ("U32", 4),
    "Npp32s": ("I32", 4), "Npp32f": ("F32", 4), "Npp64u": ("U64", 8),
    "Npp64s": ("I64", 8), "Npp64f": ("F64", 8), "Npp32fc": ("CF32", 8),
    # Fixed-point complex: the ValueKind (CF32) is 8 bytes wide but the C
    # type can be narrower; steps/counts must use sizeof(T) or NPP reads
    # past the allocation.  ref_capture_bytes captures exact bytes anyway.
    "Npp64fc": ("CF64", 16), "Npp16sc": ("CF32", 4), "Npp32sc": ("CF32", 8),
    "Npp8uc": ("CF32", 2), "Npp8sc": ("CF32", 2), "Npp16uc": ("CF32", 4),
    "Npp32uc": ("CF32", 8),
    "int": ("I32", 4),
    "unsigned int": ("U32", 4),
    "size_t": ("U64", 8),
    "float": ("F32", 4),
    "double": ("F64", 8),
}

ENUMS = {
    "NppiBorderType": "NPP_BORDER_REPLICATE",
    "NppiInterpolationMode": "NPPI_INTER_LINEAR",
    "NppCmpOp": "NPP_CMP_LESS",
    "NppRoundMode": "NPP_RND_NEAR",
    "NppiMaskSize": "NPP_MASK_SIZE_3_X_3",
    "NppHintAlgorithm": "NPP_ALG_HINT_NONE",
    "NppiAxis": "NPP_HORIZONTAL_AXIS",
    "NppiDifferentialKernel": "NPP_FILTER_SOBEL",
    "NppiAlphaOp": "NPPI_OP_ALPHA_PLUS",
}

SCRATCH_NAMES = {"pDeviceBuffer", "pBuffer", "pScratch", "pScratchBuffer",
                 "pDeviceScratch", "pDeviceScratchBuffer", "pBufferDevice"}
HOST_NAMES = {"hpBufferSize", "hpBufferSizeInBytes", "hpSize", "hpBufferSizeInBytes"}
OUTPUT_SCALAR = {"pSum", "pMean", "pMin", "pMax", "pNorm", "pDotProd", "pError",
                 "pAvg", "pStdDev", "pMinIndex", "pMaxIndex", "pIndex", "pCount",
                 "pHist", "pHistogram", "pResult", "pVal", "pValuesOut"}
INPUT_PREFIXES = ("pSrc", "pMask", "pKernel", "pTpl", "pTemplate", "pConstant",
                  "pConstants", "pLevels", "pValues", "pPalette", "pLookupTable",
                  "pTable")
OUTPUT_PREFIXES = ("pDst", "pSrcDst")
SKIP_PARAM_TYPES = {"NppStreamContext"}


@dataclass
class Param:
    ctype: str
    name: str
    is_pointer: bool
    is_const: bool
    is_array: bool
    array_len: int


def split_params(params: str) -> List[str]:
    out, depth, start = [], 0, 0
    for i, ch in enumerate(params):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(params[start:i].strip())
            start = i + 1
    tail = params[start:].strip()
    if tail:
        out.append(tail)
    return out


PARAM_RE = re.compile(
    r"^(?P<pre>const\s+)?(?P<type>[A-Za-z_]\w*(?:\s*\*)?(?:\s+const)?)\s*"
    r"(?P<stars>\**)\s*(?P<name>[A-Za-z_]\w*)\s*(?P<arr>\[[^\]]*\])?$"
)


def parse_param(raw: str) -> Optional[Param]:
    raw = raw.replace("*const", "* const").strip()
    if "(*" in raw:
        return None
    m = PARAM_RE.match(raw)
    if not m:
        return None
    pre = bool(m.group("pre"))
    base = m.group("type").strip()
    stars = m.group("stars") or ""
    name = m.group("name")
    arr = m.group("arr") or ""
    is_const = pre or "const" in base
    base = base.replace("const", "").strip()
    if base.endswith("*"):
        base = base[:-1].strip()
        stars += "*"
    is_pointer = bool(stars) or bool(arr)
    array_len = 0
    if arr:
        try:
            array_len = int(arr.strip("[]").strip())
        except ValueError:
            array_len = 3
    return Param(base, name, is_pointer, is_const, bool(arr), array_len)


def parse_params(params: str) -> Optional[List[Param]]:
    out = []
    for raw in split_params(params):
        p = parse_param(raw)
        if p is None:
            return None
        if p.ctype == "NppStreamContext":
            continue
        out.append(p)
    return out


def strip_ctx(fn: str) -> str:
    """Drop the _Ctx suffix; channel/planar markers precede it and the raw
    markers never match a `[A-Z0-9_]*$` tail containing the mixed-case Ctx."""
    return re.sub(r"_Ctx$", "", fn)


def channels(fn: str) -> int:
    fn = strip_ctx(fn)
    # Tail is mixed case: R / IR / RSfs / IRSfs / M ... follow the channel
    # marker (e.g. nppiAddC_8u_C3RSfs), so [A-Z0-9_]*$ would never match.
    m = re.search(r"_(AC|C)(\d)[A-Za-z0-9_]*$", fn)
    if m:
        n = int(m.group(2))
        # NPP AC1 images carry one color channel plus a separate alpha
        # channel (2 values per pixel); AC4 packs alpha into the 4th slot,
        # so only AC1 needs widening.
        return 2 if m.group(1) == "AC" and n == 1 else n
    m = re.search(r"_P(\d)[A-Za-z0-9_]*$", fn)
    return int(m.group(1)) if m else 1


def is_planar(fn: str) -> bool:
    return bool(re.search(r"_P\d[A-Za-z0-9_]*$", strip_ctx(fn)))


def pointee(p: Param) -> Optional[str]:
    return p.ctype if p.ctype in SCALAR else None


def pointer_role(p: Param) -> Optional[str]:
    n = p.name
    if n in SCRATCH_NAMES:
        return "scratch"
    if n in HOST_NAMES:
        return "host"
    if n in OUTPUT_SCALAR:
        return "scalar_out"
    if n in ("pSrcDst", "pSrcDst1", "pSrcDst2"):
        return "inplace"
    if n.startswith("pDst"):
        return "out"
    if n.startswith(INPUT_PREFIXES):
        return "in"
    if n.startswith("pHint") or n.startswith("pDevice"):
        return "scratch"
    return None


def scalar_value(p: Param) -> Optional[str]:
    t = p.ctype
    if t in ENUMS:
        return ENUMS[t]
    if t == "NppiSize":
        return None  # handled by call site
    if t == "NppiPoint":
        return "{1, 1}"
    if t == "NppiRect":
        return "{0, 0, 17, 13}"
    if t in ("Npp32f", "Npp64f", "float", "double"):
        return "0.5"
    if t == "Npp16f":
        return "float_to_npp16f_host(0.5f)"
    if t in ("Npp16sc", "Npp32sc", "Npp32fc", "Npp64fc", "Npp8uc", "Npp16uc", "Npp32uc"):
        return f"{t}{{2, 2}}"
    if t in SCALAR:
        return "2"
    return None


def fn_kind(fn: str, params: Sequence[Param]) -> str:
    if "GetBuffer" in fn or "GetGeometry" in fn or "GetBufferSize" in fn:
        return "helper"
    if any(p.name == "nLength" for p in params):
        return "signal"
    return "image"


def skip_reason(row: dict, params: Optional[Sequence[Param]]) -> Optional[str]:
    if row["ctx"] != "1":
        return "non-ctx (NPP 13 removed the ABI)"
    if params is None:
        return "unsupported parameter syntax"
    if not params:
        return "no parameters"
    if is_planar(row["function_name"]):
        return "planar pointer-array layout (manual follow-up)"
    for p in params:
        if p.is_array:
            # C array parameter: caller-allocated host memory passed by value
            # (e.g. `const Npp8u aConstants[3]`), not a device pointer.
            if p.ctype not in SCALAR:
                return f"unsupported array element type {p.ctype} ({p.name})"
            if not p.is_const:
                return f"output host array unsupported ({p.name})"
            continue
        if p.is_pointer:
            if p.ctype not in SCALAR:
                return f"unsupported pointer type {p.ctype} ({p.name})"
            if pointer_role(p) is None:
                return f"unknown pointer role {p.name}"
        else:
            if p.ctype in SCALAR or p.ctype in ENUMS or p.ctype in ("NppiSize", "NppiPoint", "NppiRect"):
                continue
            return f"unsupported scalar type {p.ctype} ({p.name})"
    return None


def align_step(width: int, count: int, esize: int) -> str:
    return f"align16({width} * {count} * {esize})"


def generate(row: dict, ov: Optional[dict] = None) -> Tuple[Optional[str], Optional[str]]:
    fn = row["function_name"]
    ov = ov or {}
    for exact in ov.get("skip_exact", []):
        if fn == exact:
            return None, "module override: skip_exact"
    for needle in ov.get("skip_contains", []):
        if needle in fn:
            return None, f"module override: skip_contains {needle}"
    arg_ov = (ov.get("arg_overrides") or {}).get(fn, {})
    params = parse_params(row["params"])
    reason = skip_reason(row, params)
    if reason:
        return None, reason
    assert params is not None
    ch = channels(fn)
    kind = fn_kind(fn, params)
    W, H, N = 17, 13, 257
    # Border-style functions declare a source/destination image of
    # {W+4, H+4} (oSrcSize / oSrcSizeROI / oDstSizeROI) while the ROI stays
    # {W, H}; image buffers must cover every row the call can touch at the
    # emitted pitch.
    has_full_size = any(p.ctype == "NppiSize"
                        and p.name in ("oSrcSize", "oSrcSizeROI", "oDstSizeROI")
                        for p in params)
    rows = H + 4 if has_full_size else H
    cols = W + 4 if has_full_size else W

    def img_count(t: str) -> int:
        es = SCALAR[t][1]
        pitch = (W * ch * es + 15) & ~15
        return ((rows - 1) * pitch + cols * ch * es) // es + 8

    def mask_count(t: str) -> int:
        es = SCALAR[t][1]
        pitch = (3 * es + 15) & ~15
        return ((3 - 1) * pitch + 3 * es) // es + 4

    # Discover the image element type (first image pointer) for step sizing.
    img_t = "Npp8u"
    for p in params:
        if p.is_pointer and p.ctype in SCALAR:
            img_t = p.ctype
            break
    esize = SCALAR[img_t][1]
    step = f"align16({W} * {ch} * {esize})"

    decls: List[str] = [f"  NppStreamContext ctx = ref_current_context();"]
    src_counts: Dict[str, int] = {}
    dst_counts: Dict[str, int] = {}

    def count_for(p: Param) -> int:
        role = pointer_role(p)
        if role == "scratch":
            return 1024 * 1024
        if role == "host":
            return 1
        if role == "scalar_out":
            return max(ch, 1)
        if p.name in ("pConstant", "pConstants"):
            return max(ch, 1)
        if role in ("in", "out", "inplace"):
            if p.name in ("pMask", "pKernel", "pTpl", "pTemplate"):
                return mask_count(p.ctype)
            if kind == "signal":
                return N * max(ch, 1)
            return img_count(p.ctype)
        return max(ch, 1)

    allocs: List[str] = []
    args: List[str] = []
    captures: List[str] = []
    names: Dict[str, str] = {}

    # Steps are referenced in the argument list; build them first.  Each
    # step uses the element size of its associated pointer (mixed-type
    # functions have different src/dst element sizes).
    ptr_by_name = {p.name: p for p in params if p.is_pointer}
    steps: Dict[str, str] = {}
    for p in params:
        if not p.name.endswith("Step"):
            continue
        base_ptr = re.sub(r"^n", "p", re.sub(r"Step$", "", p.name))
        q = ptr_by_name.get(base_ptr)
        if q is not None and q.ctype in SCALAR:
            es = SCALAR[q.ctype][1]
            if base_ptr in ("pMask", "pKernel", "pTpl"):
                steps[p.name] = f"align16(3 * {es})"
            else:
                steps[p.name] = f"align16({W} * {ch} * {es})"
        else:
            steps[p.name] = step

    for p in params:
        if p.is_array:
            # C array parameter: by-value host memory (e.g. aConstants[3]).
            n = p.array_len or max(ch, 1)
            val = scalar_value(p)
            decls.append(f"  const {p.ctype} {p.name}[{n}] = {{{', '.join([val] * n)}}};")
            args.append(p.name)
            continue
        role = pointer_role(p) if p.is_pointer else None
        if p.is_pointer and role == "host":
            decls.append(f"  SIZE_TYPE {p.name} = 0;")
            args.append(f"&{p.name}")
            continue
        if p.is_pointer and role:
            if p.ctype not in SCALAR:
                return None, f"unsupported pointer type {p.ctype}"
            n = p.name
            count = count_for(p)
            names[n] = n
            decls.append(f"  auto {n} = ref_device<{p.ctype}>({count}, 0x{int(hashlib.sha256((fn + n).encode()).hexdigest()[:8], 16):08x});")
            args.append(f"{n}->get()")
            if role in ("out", "inplace", "scalar_out"):
                # Captured by exact byte size: the ValueKind element size used by
                # ref_capture<T> can exceed sizeof(T) for fixed-point complex types
                # (e.g. Npp16sc is 4 bytes but maps to the 8-byte CF32 kind), which
                # makes the D2H copy read past the allocation for scalar outputs.
                captures.append(f'  ref_capture_bytes(c, {n}->get(), {count} * sizeof({p.ctype}));')
            continue
        if p.is_pointer:
            return None, f"unsupported pointer role {p.name}"
        # scalar / enum / struct
        if p.name in arg_ov:
            args.append(str(arg_ov[p.name]))
            continue
        if p.name in steps:
            # Per-pointer step (mixed-type functions have different src/dst
            # element sizes; the steps dict already resolved the right one).
            args.append(steps[p.name])
            continue
        if p.ctype == "NppiSize":
            if "Mask" in p.name or "Kernel" in p.name or "Tpl" in p.name:
                args.append("{3, 3}")
            elif p.name in ("oSrcSize", "oSrcSizeROI", "oDstSizeROI"):
                args.append(f"{{{W + 4}, {H + 4}}}")
            else:
                args.append(f"{{{W}, {H}}}")
            continue
        if p.name == "nLength":
            args.append("static_cast<NppSignalLength>(257)")
            continue
        if p.name in ("nSrcStep", "nDstStep", "nSrc1Step", "nSrc2Step", "nSrcDstStep",
                      "nMaskStep", "nTplStep", "nBufferStep"):
            args.append(step)
            continue
        if p.ctype == "size_t" and "Length" in p.name:
            args.append("static_cast<size_t>(257)")
            continue
        val = scalar_value(p)
        if val is None:
            return None, f"unsupported scalar {p.ctype} {p.name}"
        args.append(val)

    call = f"{fn}({', '.join(args)}, ctx)"
    body = [
        f"TEST(RefSuite, {fn}) {{",
        *decls,
        "  npp_diff::clear_cuda_error();",
        f"  NppStatus st = {call};",
        f'  npp_diff::Case c("ref/{fn}", "{fn}");',
        "  c.status(st);",
        *captures,
        "  c.finish();",
        "}",
    ]
    return "\n".join(body), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", help="header module to generate")
    ap.add_argument("--all", action="store_true", help="generate every module")
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.module and not args.all:
        ap.error("pass --module or --all")

    rows = list(csv.DictReader(open(args.catalog)))
    by_mod: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        if args.module and r["module"] != args.module:
            continue
        by_mod.setdefault(r["module"], []).append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_ok = total_skip = 0
    for module, mrows in by_mod.items():
        stem0 = re.sub(r"[^A-Za-z0-9]+", "_", module.replace(".h", ""))
        ov_path = args.out_dir / "overrides" / f"{stem0}.json"
        ov = json.loads(ov_path.read_text()) if ov_path.is_file() else {}
        ok, skipped = [], []
        for r in mrows:
            text, reason = generate(r, ov)
            if text is None:
                skipped.append({"function": r["function_name"], "reason": reason})
            else:
                ok.append((r["function_name"], text))
        total_ok += len(ok)
        total_skip += len(skipped)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", module.replace(".h", ""))
        for idx in range(0, len(ok), args.chunk_size):
            chunk = ok[idx : idx + args.chunk_size]
            path = args.out_dir / f"test_{stem}_{idx // args.chunk_size}_generated.cpp"
            body = [
                "// GENERATED by tools/gen_ref_tests.py -- DO NOT EDIT.",
                f"// module: {module}",
                '#include "npp_diff_harness.h"',
                '#include "npp_test_base.h"',
                '#include "ref_test_utils.h"',
                "",
                "using namespace npp_functional_test;",
                "using namespace npp_diff;",
                "using namespace ref;",
                "",
                "\n\n".join(t for _, t in chunk),
                "",
            ]
            if not args.dry_run:
                path.write_text("\n".join(body))
            print(f"{path.name}: {len(chunk)} tests")
        skip_path = args.out_dir / f"skipped_{stem}.json"
        if not args.dry_run:
            skip_path.write_text(json.dumps(skipped, indent=2) + "\n")
        print(f"  skipped {len(skipped)} -> {skip_path.name}")
    print(f"total: {total_ok} tests, {total_skip} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
