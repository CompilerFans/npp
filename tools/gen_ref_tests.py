#!/usr/bin/env python3
"""Generate reference unit tests for the public NPP API surface.

Each generated test calls the `_Ctx` entry point (NPP 13 removed the non-Ctx
ABI), records the status and output buffers through `npp_diff::Case`, and is
meant to run against official NVIDIA NPP on the 5090 first (`record`), then
against mcMPP (`verify`).

The generator synthesizes arguments conservatively from parameter names and
emits a skip manifest for functions it cannot construct safely.  Image
buffers are sized for the padded destination geometry so no family can run
past its allocation, and fixed-size host constant arrays (channel orders,
warp coefficients, per-channel values) are initialized to identity/degenerate
values.  Irregular families are expected to be refined by module owners with
per-module overrides/skips; the generator core stays shared.

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
                  "pTable", "pXMap", "pYMap")
OUTPUT_PREFIXES = ("pDst", "pSrcDst")
SKIP_PARAM_TYPES = {"NppStreamContext"}

# Fixed-size host constant arrays are passed by value-ish semantics (channel
# orders, warp coefficient matrices, ROI quadrangles, per-channel fill values).
HOST_ARRAY_PREFIXES = ("a",)
KERNELISH = ("pMask", "pKernel", "pTpl", "pTemplate", "pXMap", "pYMap")
# Scalar parameters whose safe value depends on the name, not the type.
SCALAR_NAME_DEFAULTS = {"nMin": "0.5", "nMax": "100.5",
                        "eInterpolation": "NPPI_INTER_LINEAR"}


@dataclass
class Param:
    ctype: str
    name: str
    is_pointer: bool
    is_const: bool
    is_array: bool
    array_len: int
    arr_dims: tuple = ()  # declarator dims, e.g. (2, 3) for "aCoeffs[2][3]"
    is_ptr_array: bool = False  # element is itself a pointer, e.g. "pDst[3]"


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
    r"^(?P<pre>const\s+)?(?P<type>(?:unsigned\s+|signed\s+|long\s+|short\s+)*"
    r"[A-Za-z_]\w*(?:\s*\*)?(?:\s+const)?)\s*"
    r"(?P<stars>\**)\s*(?P<name>[A-Za-z_]\w*)\s*(?P<arr>(?:\[[^\]]*\])+)?$"
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
    dims = []
    for d in re.findall(r"\[([^\]]*)\]", arr):
        try:
            dims.append(int(d.strip()))
        except ValueError:
            dims.append(3)
    total = 1
    for d in dims:
        total *= d
    return Param(base, name, is_pointer, is_const, bool(arr),
                 total if dims else 0, tuple(dims), bool(stars and dims))


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


def channel_tokens(fn: str) -> List[int]:
    """All channel counts in the layout suffix, in order of appearance.

    ``nppiSwapChannels_8u_C3C4R_Ctx`` -> [3, 4] (src 3 channels, dst 4),
    ``nppiConvert_16f32f_AC4R_Ctx`` -> [4].  The NPP 13 ABI names all carry a
    trailing ``_Ctx``, which the historic single-token regex never matched.
    """
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_(?:AC|C|P)\d", base)
    tail = base[m.start() + 1:] if m else ""
    toks = []
    for m in re.finditer(r"(AC|C|P)(\d+)", tail):
        n = int(m.group(2))
        # NPP AC1 images carry one color channel plus a separate alpha
        # channel (2 values per pixel); AC4 packs alpha into the 4th slot.
        toks.append(2 if m.group(1) == "AC" and n == 1 else n)
    return toks or [1]


def channels(fn: str) -> int:
    return channel_tokens(fn)[0]


def is_planar(fn: str) -> bool:
    base = re.sub(r"_Ctx$", "", fn)
    return bool(re.search(r"_P\d", base))


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
    if p.name in SCALAR_NAME_DEFAULTS:
        t = p.ctype
        # Name-based defaults only apply to the types where they make sense.
        if p.name == "eInterpolation" and t in SCALAR:
            return SCALAR_NAME_DEFAULTS[p.name]
        if p.name in ("nMin", "nMax") and t in ("Npp32f", "Npp64f", "float", "double"):
            return SCALAR_NAME_DEFAULTS[p.name]
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


def host_array_init(p: Param, toks: Sequence[int], W: int, H: int,
                    WMAX: int, HMAX: int) -> Optional[str]:
    """Initializer for a fixed-size host constant array.

    Channel orders become the identity permutation (clamped to the source
    channel count), warp coefficient matrices become the identity transform
    and quadrangles map the ROI corners, so every family degenerates to a
    safe, deterministic copy-like operation.
    """
    dims = p.arr_dims or ((p.array_len,) if p.array_len else (1,))
    if p.name == "aDstOrder":
        src_ch = toks[0] if toks else 1
        return "{%s}" % ", ".join(str(min(i, src_ch - 1)) for i in range(dims[0]))
    if "Coeffs" in p.name and len(dims) == 2:
        rows, cols = dims
        return "{%s}" % ", ".join(
            "{%s}" % ", ".join("1" if c == r else "0" for c in range(cols))
            for r in range(rows))
    if "Quad" in p.name and dims == (4, 2):
        wmax, hmax = (WMAX - 1, HMAX - 1) if "Dst" in p.name else (W - 1, H - 1)
        return "{%s}" % ", ".join("{%d, %d}" % (x, y) for x, y in
                                  ((0, 0), (wmax, 0), (wmax, hmax), (0, hmax)))
    val = scalar_value(p)
    if val is None:
        return None
    total = 1
    for d in dims:
        total *= d
    return "{%s}" % ", ".join([val] * total)


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
        if p.is_ptr_array:
            return f"array-of-pointers parameter {p.name}"
        if p.is_array:
            # Fixed-size host constant array (channel orders, warp
            # coefficients, per-channel fill values, ...).
            if p.ctype not in SCALAR:
                return f"unsupported host array type {p.ctype} ({p.name})"
            if not (p.is_const or p.name.startswith(HOST_ARRAY_PREFIXES)):
                return f"unsupported host output array {p.name}"
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
    toks = channel_tokens(fn)
    ch = toks[0]
    kind = fn_kind(fn, params)
    W, H, N = 17, 13, 257
    # Destination geometry (border copies, sub-pixel copies, padded dst ROIs)
    # is W+4 x H+4; every image buffer is sized for it so no family can run
    # past its allocation regardless of which size parameter claims what.
    WMAX, HMAX = W + 4, H + 4

    # Discover the image element type (first image pointer) for the fallback step.
    img_t = "Npp8u"
    for p in params:
        if p.is_pointer and not p.is_array and p.ctype in SCALAR:
            img_t = p.ctype
            break
    esize = SCALAR[img_t][1]
    step = f"align16({WMAX} * {ch} * {esize})"

    decls: List[str] = [f"  NppStreamContext ctx = ref_current_context();"]

    # Per-pointer channel counts: the channel tokens in the layout suffix map,
    # in order, onto the image pointers that carry pixel data (C3C4R -> src 3,
    # dst 4).  Mask/kernel/template and coordinate-map pointers stay single
    # channel; a single token applies to every image pointer (AC4R -> 4).
    pch: Dict[str, int] = {}
    img_idx = 0
    for p in params:
        if p.is_array or not p.is_pointer:
            continue
        if p.name in KERNELISH:
            pch[p.name] = 1
            continue
        if pointer_role(p) in ("in", "out", "inplace"):
            pch[p.name] = toks[img_idx] if img_idx < len(toks) else toks[-1]
            img_idx += 1

    def ch_of(p: Param) -> int:
        return pch.get(p.name, ch)

    def count_for(p: Param) -> int:
        role = pointer_role(p)
        if role == "scratch":
            return 1024 * 1024
        if role == "host":
            return 1
        if role == "scalar_out":
            return max(ch, 1)
        if role in ("in", "out", "inplace"):
            if p.name in ("pMask", "pKernel", "pTpl", "pTemplate"):
                return 3 * (3 + 16)
            if kind == "signal":
                return N * max(ch_of(p), 1)
            # HMAX rows of (WMAX * ch + 16 slack) elements: covers any
            # align16 step rounding and every size parameter value below.
            return HMAX * (WMAX * ch_of(p) + 16)
        return max(ch, 1)

    args: List[str] = []
    captures: List[str] = []

    # Steps are referenced in the argument list; build them first.  Each step
    # uses the element size of its associated pointer (mixed-type functions
    # have different src/dst element sizes).
    ptr_by_name = {p.name: p for p in params if p.is_pointer}
    steps: Dict[str, str] = {}
    for p in params:
        if not p.name.endswith("Step"):
            continue
        base_ptr = re.sub(r"^n", "p", re.sub(r"Step$", "", p.name))
        q = ptr_by_name.get(base_ptr)
        if q is not None and q.ctype in SCALAR and not q.is_array:
            es = SCALAR[q.ctype][1]
            if base_ptr in ("pMask", "pKernel", "pTpl"):
                steps[p.name] = f"align16(3 * {es})"
            else:
                steps[p.name] = f"align16({WMAX} * {ch_of(q)} * {es})"
        else:
            steps[p.name] = step

    for p in params:
        role = pointer_role(p) if p.is_pointer else None
        # Fixed-size host constant arrays (channel orders, coefficients, ...).
        if p.is_array:
            init = host_array_init(p, toks, W, H, WMAX, HMAX)
            if init is None:
                return None, f"unsupported host array {p.ctype} {p.name}"
            dims = "".join(f"[{d}]" for d in (p.arr_dims or (p.array_len,)))
            const = "const " if p.is_const else ""
            decls.append(f"  {const}{p.ctype} {p.name}{dims} = {init};")
            args.append(p.name)
            continue
        if p.is_pointer and role == "host":
            decls.append(f"  SIZE_TYPE {p.name} = 0;")
            args.append(f"&{p.name}")
            continue
        if p.is_pointer and role:
            if p.ctype not in SCALAR:
                return None, f"unsupported pointer type {p.ctype}"
            n = p.name
            count = count_for(p)
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
            elif p.name in ("oDstSize", "oDstSizeROI"):
                # Padded destination geometry (border copies, sub-pixel copies).
                args.append(f"{{{WMAX}, {HMAX}}}")
            else:
                args.append(f"{{{W}, {H}}}")
            continue
        if p.name == "nLength":
            args.append("static_cast<NppSignalLength>(257)")
            continue
        if p.ctype == "NppiRect":
            if p.name in ("oDstROI", "oDstRectROI"):
                args.append(f"{{0, 0, {WMAX}, {HMAX}}}")
            else:
                args.append(f"{{0, 0, {W}, {H}}}")
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
