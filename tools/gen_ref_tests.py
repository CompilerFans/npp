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

Extended layouts are generated too: planar pointer arrays (``pSrc[3]``), other
arrays of device pointers (per-channel LUT/histogram tables) and batch
descriptor lists (``NppiImageDescriptor``/``Nppi*BatchCXR``).  Per-plane
buffers use a uniform two-channel-pitch step so every planar/semi-planar
geometry stays in bounds, and batch lists are built host-side then uploaded.
Functions using these layouts are emitted after the base-layout functions of
their module (in separate chunk files) so already-recorded chunks stay
byte-identical when the generator gains support for new layouts.  A third
"misc" stream holds functions the legacy rules skipped but the widened
misc-mode rules (extra scalar/struct types, enum defaults, host-visible
outputs, optional-output nulls) can construct; it is emitted after the
extended-layout chunks for the same reason.

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
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
                 "pHist", "pHistogram", "pResult", "pVal", "pValuesOut",
                 # Batched quality metrics: one scalar per image in the batch.
                 "pMSE", "pPSNR", "pSSIM", "pWMSSSIM", "pNewMaxLabelIDList"}
# Output scalar names only recognized in batched entry points; their non-batch
# counterparts stay skipped exactly as before (no base-test churn).
BATCH_ONLY_SCALARS = {"pMSE", "pPSNR", "pSSIM", "pWMSSSIM", "pNewMaxLabelIDList"}
# Per-channel input count/bound arrays that NPP declares without a const.
KNOWN_INPUT_ARRAYS = {"nLevels", "nLowerLevel", "nUpperLevel"}
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
                        "eInterpolation": "NPPI_INTER_LINEAR",
                        "nAstigmatismRatio": "2.0f",
                        "nOrientationRadians": "0.0f",
                        # PBA site range: the PRNG float sources cover [-1, 1),
                        # so [0.5, 1.0] makes ~25% of the pixels sites and the
                        # distance transform has real work to do.  Integral
                        # site values use the traditional 0 (see scalar_value;
                        # non-zero integral ranges are nondeterministic in
                        # NPP 13.4).
                        "nMinSiteValue": "0.5", "nMaxSiteValue": "1.0"}

# --- Misc-mode extension (REF_MISC) -----------------------------------------
# Everything below widens the rule sets for functions the legacy rules above
# skipped.  Those functions are emitted into a third "misc" chunk stream (see
# main()); the legacy snapshot classifies rows exactly as the pre-misc
# generator did, so the two already-recorded streams stay byte-identical.
LEGACY_SCALAR = dict(SCALAR)
LEGACY_ENUMS = dict(ENUMS)
LEGACY_SCRATCH_NAMES = set(SCRATCH_NAMES)
LEGACY_HOST_NAMES = set(HOST_NAMES)
LEGACY_OUTPUT_SCALAR = set(OUTPUT_SCALAR)
LEGACY_KNOWN_INPUT_ARRAYS = set(KNOWN_INPUT_ARRAYS)
LEGACY_BATCH_ONLY_SCALARS = set(BATCH_ONLY_SCALARS)

# Extra scalar C types: the Npp64sc signal type and the small geometry
# structs used as device outputs (NppiPoint indices, NppPointPolar Hough
# lines, NppiProfileData radial-profile samples).  "Bytes" kinds capture the
# raw struct bytes; the size column is sizeof(T).
SCALAR.update({
    "Npp64sc": ("CF64", 16),
    "NppPointPolar": ("Bytes", 8),
    "NppiPoint": ("Bytes", 8),
    "NppiProfileData": ("Bytes", 12),
})
# Extra enums with a safe default constant (NPP 13.4 nppdefs.h names).
ENUMS.update({
    "NppiNorm": "nppiNormL1",
    "NppiBayerGridPosition": "NPPI_BAYER_BGGR",
    "NppDataType": "NPP_8U",
    "NppiChannels": "NPP_CH_1",
    "NppsZCType": "nppZCR",
    "NppiWatershedSegmentBoundaryType": "NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE",
})
# Extra scratch / working-buffer pointer names.
SCRATCH_NAMES |= {"pAdvancedScratchBuffer", "pAntialiasingDeviceBuffer", "pBuffer64"}
# Extra host-visible output scalars: NPP writes them through a host pointer;
# the typed host variable is captured after the call.
HOST_NAMES |= {"nBufferSize"}
# Extra single-value device output pointers.
OUTPUT_SCALAR |= {"pNormDiff", "pNormRel", "pDp", "pMinValue", "pMaxValue",
                  "pIndexX", "pIndexY", "pIndx", "pMinIndx", "pMaxIndx",
                  "pMaxAbs", "pMinAbs", "pCounts", "pValZC", "pMSSSIM",
                  "pDeviceLineCount"}
# Fixed-count device output arrays (element count; 257 == signal length N).
OUTPUT_ARRAYS = {"pDeviceLines": 8, "pProfileData": 2, "pD2FVal": 257}
# Host input helpers passed by address.  The FusedAbsDiff thresholds are
# void* in the ABI; the element type follows the NPP_8U/NPP_CH_1 default.
HOST_IN_SCALARS = {"pThreshold": "Npp8u", "pvalue": "Npp8u"}
# Host input arrays: name -> (element type, count, brace initializer).
# The two HOG descriptor windows are stacked vertically: NPP requires
# nSrcStep >= 4 * (union x extent of the windows), which the natural
# 8u C1 pitch (align16(21) == 32) only meets when the windows share x.
HOST_IN_ARRAYS = {"hpLocations": ("NppiPoint", 2, "{0, 0}, {0, 4}")}
# Host output structs written through a host pointer: name -> C type.
HOST_STRUCT_OUT = {"pConnectedRegion": "NppiConnectedRegion"}
# Optional device outputs declared "or NULL" in the headers (PBA voronoi
# planes): pass nullptr so only the transform output is produced.
NULL_PTR_NAMES = {"pDstVoronoi", "pDstVoronoiIndices",
                  "pDstVoronoiAbsoluteManhattanDistances",
                  "pDstVoronoiRelativeManhattanDistances"}
# Image-shaped device outputs that do not follow the pDst naming.
OUT_IMAGE_NAMES = {"pMarkerLabels"}
# Extra per-channel input count arrays declared without a const.
# oDstROI is the polar search-region input of FilterHoughLineRegion (the
# header declares it without a const).
KNOWN_INPUT_ARRAYS |= {"nColorKeyConst", "oDstROI"}

# --- Extended layouts -------------------------------------------------------
# Batched entry points take nBatchSize > 1; two images keep every family
# meaningful while staying cheap to build and record.
NBATCH = 2
# Per-plane pitch used for planar pointer arrays: two channels cover the
# interleaved UV plane of semi-planar layouts (NV12/NV21), and every other
# plane geometry is narrower, so a single uniform step stays in bounds.
PLANE_CH = 2
# Batch descriptor structs (NPP batch geometry structures, nppdefs.h).
BATCH_STRUCTS = ("NppiImageDescriptor", "NppiBufferDescriptor",
                 "NppiResizeBatchROI_Advanced", "NppiColorTwistBatchCXR",
                 "NppiMirrorBatchCXR", "NppiResizeBatchCXR",
                 "NppiWarpAffineBatchCXR", "NppiWarpPerspectiveBatchCXR")
BATCH_DTYPE = {"8u": "Npp8u", "8s": "Npp8s", "16u": "Npp16u", "16s": "Npp16s",
               "16f": "Npp16f", "32u": "Npp32u", "32s": "Npp32s",
               "32f": "Npp32f", "64f": "Npp64f"}
IDENTITY_2X3 = "1, 0, 0, 0, 1, 0"
IDENTITY_3X3 = "1, 0, 0, 0, 1, 0, 0, 0, 1"
IDENTITY_4X5 = "1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0"


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
    const_pointee: bool = False  # leading const, e.g. "const Npp8u * pSrc[3]"


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
                 total if dims else 0, tuple(dims), bool(stars and dims), pre)


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
    Layout tokens are C (packed), P (planar), AC (alpha packed), AP (alpha
    planar, e.g. nppiBGRToHLS_8u_AP4C4R) and IP (in-place planar, e.g.
    nppiColorTwist_32f_IP3R); A/I must stay glued to their C/P so the first
    token anchor still lands on the layout suffix.
    """
    base = re.sub(r"_Ctx$", "", fn)
    m = re.search(r"_(?:[AI]?[CP])\d", base)
    tail = base[m.start() + 1:] if m else ""
    toks = []
    for m in re.finditer(r"([AI]?[CP])(\d+)", tail):
        n = int(m.group(2))
        # NPP AC1 images carry one color channel plus a separate alpha
        # channel (2 values per pixel); AC4 packs alpha into the 4th slot.
        toks.append(2 if m.group(1) == "AC" and n == 1 else n)
    if len(toks) == 1 and re.search(r"(?:8|16|32|64)u24u_", base):
        # A 24-bit packed destination dtype (nppiLUTPalette_*24u): NPP 13.4
        # declares the destination as Npp8u * but writes 3 bytes per pixel,
        # so the second image pointer needs a 3-wide token or its step
        # under-covers the row (NPP writes overlapping rows -- a data race
        # -- instead of rejecting the step).
        toks = [toks[0], 3]
    return toks or [1]


def channels(fn: str) -> int:
    return channel_tokens(fn)[0]


def is_planar(fn: str) -> bool:
    base = re.sub(r"_Ctx$", "", fn)
    return bool(re.search(r"_P\d", base))


def pointee(p: Param) -> Optional[str]:
    return p.ctype if p.ctype in SCALAR else None


def pointer_role(p: Param, legacy: bool = False) -> Optional[str]:
    n = p.name
    if n in (LEGACY_SCRATCH_NAMES if legacy else SCRATCH_NAMES):
        return "scratch"
    if n in (LEGACY_HOST_NAMES if legacy else HOST_NAMES):
        return "host"
    if n in (LEGACY_OUTPUT_SCALAR if legacy else OUTPUT_SCALAR):
        return "scalar_out"
    if not legacy:
        if n in OUTPUT_ARRAYS:
            return "array_out"
        if n in HOST_IN_SCALARS:
            return "host_in_scalar"
        if n in HOST_IN_ARRAYS:
            return "host_in_array"
        if n in HOST_STRUCT_OUT:
            return "host_out_struct"
        if n in NULL_PTR_NAMES:
            return "null"
        if n == "pSqr":
            return "in" if p.is_const else "out"
        if n in OUT_IMAGE_NAMES:
            return "out"
    if n in ("pSrcDst", "pSrcDst1", "pSrcDst2"):
        return "inplace"
    if n.startswith("pDst"):
        return "out"
    if n.startswith(INPUT_PREFIXES):
        return "in"
    if n.startswith("pHint") or n.startswith("pDevice"):
        return "scratch"
    return None


def ptr_array_role(p: Param) -> Optional[str]:
    """Role of a ``T * p[N]`` parameter: a host array of N device pointers.

    Planar image planes, per-channel LUT tables and per-plane histogram bins
    all use this shape; NppiImageDescriptor arrays are per-plane batch lists.
    """
    n = p.name
    if p.ctype == "NppiImageDescriptor":
        return "batch_planes"  # one descriptor list per plane of the batch
    if n.startswith("pSrcDst"):
        return "inplace"
    if n.startswith(OUTPUT_PREFIXES) or n.startswith("aDst"):
        return "out"
    if n.startswith(("pSrc", "aSrc", "pMask", "pTpl", "pTemplate", "pKernel")):
        return "in"
    if n in OUTPUT_SCALAR and n == "pHist":
        return "scalar_out"  # nLevels - 1 == 1 bins per plane
    if n.startswith(("pValues", "pLevels", "pTables", "pPalette", "pTable")):
        return "lut_in"
    return None


def batch_role(p: Param) -> Optional[str]:
    """Role of a batch descriptor list parameter (plain struct pointer)."""
    if p.ctype not in BATCH_STRUCTS:
        return None
    if "Buffer" in p.name:
        return "scratch"
    if p.ctype == "NppiResizeBatchROI_Advanced":
        return "roi"
    if p.ctype in ("NppiColorTwistBatchCXR", "NppiMirrorBatchCXR",
                   "NppiResizeBatchCXR", "NppiWarpAffineBatchCXR",
                   "NppiWarpPerspectiveBatchCXR"):
        return "cxr"
    n = p.name
    if n.startswith("pSrcDst"):
        return "inplace"
    if n.startswith("pSrc") or "BatchSrc" in n:
        return "in"
    if n.startswith("pDst") or "BatchDst" in n:
        return "out"
    return None


def needs_ext_layout(fn: str, params: Optional[Sequence[Param]]) -> bool:
    """True when the function needs the extended layout support.

    These functions used to be blanket-skipped; generating them now must not
    re-chunk the already-recorded base-layout tests of the module.
    """
    if is_planar(fn):
        return True
    for p in params or []:
        if p.is_ptr_array:
            return True
        if p.ctype in BATCH_STRUCTS:
            return True
        if p.is_array and p.ctype == "int" and p.name.endswith("Step"):
            return True
    return False


def scalar_value(p: Param, misc: bool = False) -> Optional[str]:
    if p.name in SCALAR_NAME_DEFAULTS:
        t = p.ctype
        # Name-based defaults only apply to the types where they make sense.
        if p.name == "eInterpolation" and t in SCALAR:
            return SCALAR_NAME_DEFAULTS[p.name]
        if p.name in ("nMin", "nMax", "nAstigmatismRatio",
                      "nOrientationRadians") \
                and t in ("Npp32f", "Npp64f", "float", "double"):
            return SCALAR_NAME_DEFAULTS[p.name]
        if misc and p.name in ("nMinSiteValue", "nMaxSiteValue"):
            # PBA site range.  Integral site extraction is only
            # run-to-run deterministic for the traditional all-zeros range
            # (probe: 8/8 stable at [0,0] vs 5/6 unstable at [2,2]/[1,4]/
            # [1,8] on RTX 5090, NPP 13.4); PRNG integral data covers
            # 0..15, so [0,0] still selects ~1/16 of the pixels as sites.
            # Float sources cover [-1, 1), so [0.5, 1.0] selects ~25%.
            if t in ("Npp32f", "Npp64f", "float", "double"):
                return SCALAR_NAME_DEFAULTS[p.name]
            if t in SCALAR and t != "Npp16f":
                return "0"
    t = p.ctype
    if t in ENUMS:
        return ENUMS[t]
    if t == "NppiSize":
        return None  # handled by call site
    if t == "NppiPoint":
        return "{1, 1}"
    if t == "NppiRect":
        return "{0, 0, 17, 13}"
    if t == "NppiPoint32f":
        return "{8.5f, 6.5f}"  # inside the default ROI (radial profiles)
    if t == "NppPointPolar":
        return "{1.0f, 1.0f}"  # Hough-line discretization step
    if t == "NppiHOGConfig":
        # Smallest valid HOG configuration (cell 4 <= block 8 <= window 8,
        # 4 bins), probe-verified against the NPP 13.4 mode checks.
        return "{4, 8, 4, {8, 8}}"
    if t in ("Npp32f", "Npp64f", "float", "double"):
        return "0.5"
    if t == "Npp16f":
        return "float_to_npp16f_host(0.5f)"
    if t in ("Npp16sc", "Npp32sc", "Npp32fc", "Npp64fc", "Npp64sc",
             "Npp8uc", "Npp16uc", "Npp32uc"):
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
    if p.name == "oDstROI":
        # Hough-line region filter: two polar points (rho, theta) bounding
        # the search wedge; probe-verified on the 5090.
        return "{{0.0f, 0.0f}, {21.0f, 6.3f}}"
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


def skip_reason(row: dict, params: Optional[Sequence[Param]],
                legacy: bool = False) -> Optional[str]:
    """Reason a function cannot be constructed safely, or None.

    ``legacy=True`` evaluates the pre-misc rule set; it classifies rows into
    the recorded streams exactly as before (see main()).
    """
    if row["ctx"] != "1":
        return "non-ctx (NPP 13 removed the ABI)"
    if params is None:
        return "unsupported parameter syntax"
    if not params:
        return "no parameters"
    batch = any(p.ctype in BATCH_STRUCTS for p in params)
    has_ptr_array = any(p.is_ptr_array for p in params)
    scalar_t = LEGACY_SCALAR if legacy else SCALAR
    enums = LEGACY_ENUMS if legacy else ENUMS
    known_in = LEGACY_KNOWN_INPUT_ARRAYS if legacy else KNOWN_INPUT_ARRAYS
    # Non-batch metric outputs (pMSE/...) are only constructed by the misc
    # rules; the legacy classification keeps skipping them.
    batch_only = LEGACY_BATCH_ONLY_SCALARS if legacy else frozenset()
    for p in params:
        if p.is_ptr_array:
            if (p.ctype == "NppiImageDescriptor" or p.ctype in SCALAR) \
                    and ptr_array_role(p) is not None:
                continue
            return f"unsupported array-of-pointers parameter {p.name}"
        if p.is_array:
            # Per-plane step array (e.g. int rSrcStep[3]) of a planar entry
            # point: one step per plane, filled with the plane pitch.
            if p.ctype == "int" and p.name.endswith("Step"):
                continue
            # Fixed-size host constant array (channel orders, warp
            # coefficients, per-channel fill values, ...).  Per-channel LUT
            # level counts ride along with the pointer-array tables they
            # describe (ext-layout functions only in the legacy rules).
            if p.ctype not in scalar_t:
                return f"unsupported host array type {p.ctype} ({p.name})"
            if not (p.is_const or p.name.startswith(HOST_ARRAY_PREFIXES)
                    or (p.name in known_in and (has_ptr_array or not legacy))):
                return f"unsupported host output array {p.name}"
            continue
        if p.is_pointer:
            if p.ctype in BATCH_STRUCTS:
                if batch_role(p) is None:
                    return f"unknown batch list role {p.name}"
                continue
            role = pointer_role(p, legacy=legacy)
            # Roles whose emission supplies its own type (host helpers and
            # scratch buffers) are accepted regardless of the catalog type;
            # the legacy rules matched the original ctype check only.
            hostish = not legacy and role in (
                "host", "host_in_scalar", "host_in_array", "host_out_struct",
                "scratch")
            if p.ctype not in scalar_t and not hostish \
                    and not (not legacy and p.ctype == "void"
                             and role in ("in", "out", "inplace")):
                return f"unsupported pointer type {p.ctype} ({p.name})"
            if role is None or (p.name in batch_only and not batch):
                return f"unknown pointer role {p.name}"
        else:
            if batch and p.ctype == "NppiNorm":
                continue  # batched label entry points take an inf-norm default
            structish = ("NppiSize", "NppiPoint", "NppiRect")
            if not legacy:
                # Small geometry structs the misc rules can default safely.
                structish += ("NppiPoint32f", "NppPointPolar", "NppiHOGConfig")
            if p.ctype in scalar_t or p.ctype in enums or p.ctype in structish:
                continue
            return f"unsupported scalar type {p.ctype} ({p.name})"
    return None


def align_step(width: int, count: int, esize: int) -> str:
    return f"align16({width} * {count} * {esize})"


def generate(row: dict, ov: Optional[dict] = None,
             misc: bool = False) -> Tuple[Optional[str], Optional[str]]:
    fn = row["function_name"]
    ov = ov or {}
    for exact in ov.get("skip_exact", []):
        # Entries are either plain function names or objects carrying a
        # precise, verifiable skip reason for the manifest.
        if isinstance(exact, dict):
            if fn == exact.get("function"):
                return None, exact.get("reason") or "module override: skip_exact"
        elif fn == exact:
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
    # Semi-planar 4:2:0 layouts (NV12/NV21 in the NPP naming scheme) need an
    # even ROI in both dimensions: NPP 13.4 hard-rejects odd sizes
    # (NPP_SIZE_ERROR, nothing executed) for the semi-planar -> packed paths.
    # Buffers and steps are still sized from WMAX/HMAX, so they stay
    # oversized for the rounded-down ROI.
    if "NV12" in fn or "NV21" in fn:
        W, H = W - 1, H - 1
    elif "MirrorBatch" in fn and re.search(r"_A?C\d+IR", fn):
        # In-place mirror batches likewise hard-require an even ROI in both
        # dimensions (NPP_SIZE_ERROR, nothing executed), while the
        # out-of-place batch and the single-image mirror variants accept the
        # odd default geometry.  The in-place layout token is CnIR (and
        # ACnIR for the alpha-packed variant).
        W, H = W - 1, H - 1
    elif "WMSSSIMBatch_" in fn:
        # The WMSSSIM batch metric is documented (header note) to process
        # only ROIs of 16x16 pixels and above; smaller ROIs are undefined.
        # Buffers are sized from HMAX so the taller ROI still fits.
        H = max(H, 16)
    elif misc and "MSSSIM_" in fn:
        # The single-image (W)MSSSIM metrics hard-reject small ROIs (probe:
        # NPP_SIZE_ERROR at 17x13; 256x256 returns NPP_NO_ERROR).  WMAX/HMAX
        # follow the enlarged ROI so steps and buffer counts stay consistent
        # (a 256-wide ROI with a WMAX-sized pitch reads past the buffers).
        W, H = 256, 256
        WMAX, HMAX = W + 4, H + 4
    elif misc and "CFAToRGB" in fn:
        # Bayer CFA demosaic needs an even ROI in both dimensions (probe:
        # 17x13 -> NPP_SIZE_ERROR, 16x12 -> NPP_NO_ERROR), the same even-ROI
        # rule the semi-planar paths above already follow.  Buffers and steps
        # stay sized from WMAX/HMAX and remain oversized.
        W, H = W - 1, H - 1
    elif misc and "DistanceTransform" in fn and "PBA" in fn:
        # The PBA distance transforms hard-require an ROI of at least 64x64
        # in both dimensions (probe: 63x63 and 64x17 -> NPP_SIZE_ERROR,
        # 64x64 and 65x65 -> NPP_NO_ERROR).  WMAX/HMAX follow the enlarged
        # ROI so the transform pitch clears the NPP step checks
        # (nDstTransformStep >= ROI width * 8 bytes).
        W, H = 64, 64
        WMAX, HMAX = W + 4, H + 4

    # Discover the image element type (first image pointer) for the fallback step.
    img_t = "Npp8u"
    for p in params:
        if p.is_pointer and not p.is_array and p.ctype in SCALAR:
            img_t = p.ctype
            break
    esize = SCALAR[img_t][1]
    step = f"align16({WMAX} * {ch} * {esize})"

    is_batch = any(p.ctype in BATCH_STRUCTS for p in params)
    # Batch image element type(s): the dtype segment of the batch entry point
    # name ("nppiMirrorBatch_32f_C3R" -> 32f; "nppiLabelMarkersUFBatch_
    # 16u32u_C1R" -> 16-bit input, 32-bit labels).  BatchInit entry points
    # carry no dtype segment; they only transform coefficients, so the buffers
    # they point at are never dereferenced and the default is safe.
    batch_types = ["Npp8u"]
    if is_batch:
        parts = re.sub(r"_Ctx$", "", fn).split("_")
        if len(parts) > 1:
            tocs = [BATCH_DTYPE[t] for t in
                    re.findall(r"(?:8|16|32|64)(?:u|s|f)", parts[1]) if t in BATCH_DTYPE]
            if tocs:
                batch_types = tocs

    def batch_t(role: str) -> str:
        if len(batch_types) > 1 and role in ("out", "inplace"):
            return batch_types[-1]
        return batch_types[0]

    def seed_for(key: str) -> str:
        return f"0x{int(hashlib.sha256((fn + key).encode()).hexdigest()[:8], 16):08x}"

    # Planar pointer-array geometry: a uniform two-channel pitch covers every
    # plane layout (including the interleaved UV plane of semi-planar formats)
    # and the buffers hold HMAX rows of it plus slack.
    def plane_step(ctype: str) -> str:
        return f"align16({WMAX} * {PLANE_CH} * {SCALAR[ctype][1]})"

    plane_cnt = HMAX * (WMAX * PLANE_CH + 16)

    decls: List[str] = [f"  NppStreamContext ctx = ref_current_context();"]

    def pr(p: Param) -> Optional[str]:
        # Legacy rows resolve pointer roles with the legacy name sets so
        # their emission stays byte-identical; misc rows use the new sets.
        return pointer_role(p, legacy=not misc)

    # Per-pointer channel counts: the channel tokens in the layout suffix map,
    # in order, onto the image pointers that carry pixel data (C3C4R -> src 3,
    # dst 4).  Mask/kernel/template and coordinate-map pointers stay single
    # channel; a single token applies to every image pointer (AC4R -> 4).
    # Pointer arrays consume one token per array (they are a set of
    # single-channel planes or per-plane batch lists), LUT/histogram arrays
    # consume none.
    pch: Dict[str, int] = {}
    img_idx = 0
    for p in params:
        if p.is_ptr_array:
            if ptr_array_role(p) in ("in", "out", "inplace", "batch_planes"):
                pch[p.name] = toks[img_idx] if img_idx < len(toks) else toks[-1]
                img_idx += 1
            continue
        if p.is_array or not p.is_pointer:
            continue
        if p.name in KERNELISH:
            pch[p.name] = 1
            continue
        if (batch_role(p) or pr(p)) in ("in", "out", "inplace"):
            pch[p.name] = toks[img_idx] if img_idx < len(toks) else toks[-1]
            img_idx += 1

    def ch_of(p: Param) -> int:
        return pch.get(p.name, ch)

    def count_for(p: Param) -> int:
        role = pr(p)
        if role == "scratch":
            # The misc (W)MSSSIM metrics need 1.7-5.6 MB of scratch for
            # their 256x256 ROI (per the GetBufferHostSize entry points);
            # every other family stays far below 1 MB at the default
            # geometry.
            if misc and "MSSSIM" in fn:
                return 8 * 1024 * 1024
            return 1024 * 1024
        if role == "host":
            return 1
        if role == "array_out":
            # Fixed-count device output arrays (Hough lines, profile data).
            return OUTPUT_ARRAYS[p.name]
        if role == "scalar_out":
            # Batched metric outputs carry one value per image in the batch,
            # per channel (NPP docs size them nBatchSize * channels).
            return NBATCH * max(ch, 1) if is_batch else max(ch, 1)
        if role in ("in", "out", "inplace"):
            if misc and p.name == "pMask":
                # NPP masks are ROI-sized channel-blind bitmaps: the legacy
                # 3x3 sizing is narrower than the default ROI, so the misc
                # stream pitches the mask like a single-channel image
                # (nMaskStep < ROI width -> NPP_STEP_ERROR).
                return HMAX * (WMAX + 16)
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
    # have different src/dst element sizes).  A step parameter whose base
    # pointer is an array of planes (shared per-plane steps like nSrcStep of
    # P3R entry points, or step arrays like rSrcStep[3]) resolves to the
    # uniform plane pitch.
    ptr_by_name = {p.name: p for p in params if p.is_pointer}
    steps: Dict[str, str] = {}
    # Plane arrays addressed by a shared k-scaled scalar step (JPEG family):
    # their buffers must hold k-pitch rows, not the uniform plane pitch.
    wide_planes: Set[str] = set()
    for p in params:
        if not p.name.endswith("Step"):
            continue
        stem = re.sub(r"Step$", "", p.name)
        cands = [re.sub(r"^n", "p", stem), stem]
        if stem[:1] in "nra" and len(stem) > 1:
            cands += ["p" + stem[1:], "a" + stem[1:]]
        matched, q = None, None
        for cand in cands:
            if cand in ptr_by_name:
                matched, q = cand, ptr_by_name[cand]
                break
        if q is not None and q.ctype in SCALAR and not q.is_array:
            es = SCALAR[q.ctype][1]
            if matched in ("pMask", "pKernel", "pTpl"):
                if misc and matched == "pMask":
                    # ROI-sized channel-blind mask pitch (see count_for).
                    steps[p.name] = f"align16({WMAX} * {es})"
                elif misc:
                    # Correlation templates and filter kernels are
                    # channel-packed: a 3-pixel row needs 3 * channels *
                    # esize bytes, or NPP rejects the call with
                    # NPP_STEP_ERROR (probed on 16u32f/32f/64f C3/C4
                    # templates carrying a C1-sized pitch).  Legacy rows
                    # keep the historic channel-blind pitch.
                    steps[p.name] = f"align16(3 * {ch} * {es})"
                else:
                    steps[p.name] = f"align16(3 * {es})"
            else:
                steps[p.name] = f"align16({WMAX} * {ch_of(q)} * {es})"
        elif q is not None and q.is_ptr_array and q.ctype in SCALAR:
            if "JPEG" in fn and not p.is_array:
                # The JPEG PnR entry points host-validate a shared scalar
                # plane-array step against the interleaved row size
                # (width * plane count) for their destinations, so pitch the
                # family k-scaled; per-plane step arrays keep the per-plane
                # pitch, which their own checks accept.
                k = q.array_len or 3
                steps[p.name] = f"align16({WMAX} * {k} * {SCALAR[q.ctype][1]})"
                wide_planes.add(q.name)
            else:
                steps[p.name] = plane_step(q.ctype)
        elif is_planar(fn):
            steps[p.name] = f"align16({WMAX} * {PLANE_CH} * {esize})"
        else:
            steps[p.name] = step

    prelude: List[str] = []

    # CompressMarkerLabelsUFBatch sizes its per-image working buffers through
    # the documented GetBufferSize helper: the batch argument must be exactly
    # the returned value (worst case: one label per ROI pixel).
    batch_buf_size = None
    if is_batch:
        for p in params:
            if p.name in ("nPerImageBufferSize", "nLargestPerImageBufferSize"):
                decls.append("  int hpCompressBufferSize = 0;")
                decls.append(f"  nppiCompressMarkerLabelsGetBufferSize_32u_C1R({W} * {H}, &hpCompressBufferSize);")
                decls.append(f"  const int {p.name} = hpCompressBufferSize > 0 ? hpCompressBufferSize : {1024 * 1024};")
                batch_buf_size = p.name
                break

    def emit_ptr_array(p: Param) -> Optional[str]:
        """Emit a host array of device pointers plus its backing buffers."""
        role = ptr_array_role(p)
        n, k = p.name, (p.array_len or 3)
        c = "const " if p.const_pointee else ""
        if role == "batch_planes":
            # One descriptor list per plane; each list covers NBATCH images
            # of a single-channel plane (planes use the uniform plane pitch).
            t = batch_t("in")
            stp = plane_step(t)
            lists = []
            for i in range(k):
                for b in range(NBATCH):
                    decls.append(f"  auto {n}_{i}_{b} = ref_device<{t}>({plane_cnt}, {seed_for(f'{n}_{i}_{b}')});")
                decls.append(f"  std::vector<NppiImageDescriptor> {n}_host{i}({NBATCH});")
                for b in range(NBATCH):
                    decls.append(f"  {n}_host{i}[{b}].pData = {n}_{i}_{b}->get();")
                    decls.append(f"  {n}_host{i}[{b}].nStep = {stp};")
                    decls.append(f"  {n}_host{i}[{b}].oSize = {{{W}, {H}}};")
                decls.append(f"  auto {n}_dev{i} = std::make_unique<DeviceMemory<NppiImageDescriptor>>({NBATCH});")
                decls.append(f"  {n}_dev{i}->copyFromHost({n}_host{i});")
                lists.append(f"{n}_dev{i}->get()")
            decls.append(f"  {c}NppiImageDescriptor * {n}[{k}] = {{{', '.join(lists)}}};")
            args.append(n)
            return None
        if p.ctype not in SCALAR:
            return f"unsupported array-of-pointers type {p.ctype} ({n})"
        if role in ("in", "out", "inplace"):
            # A set of single-channel planes sharing the uniform plane pitch,
            # or k-pitch rows when a shared scalar step of the family demands
            # interleaved-sized rows (JPEG PnR).
            cnt = HMAX * (WMAX * k + 16) if n in wide_planes else plane_cnt
            for i in range(k):
                decls.append(f"  auto {n}_{i} = ref_device<{p.ctype}>({cnt}, {seed_for(f'{n}_{i}')});")
            decls.append(f"  {c}{p.ctype} * {n}[{k}] = {{{', '.join(f'{n}_{i}->get()' for i in range(k))}}};")
            args.append(n)
            if role in ("out", "inplace"):
                for i in range(k):
                    captures.append(f"  ref_capture_bytes(c, {n}_{i}->get(), {cnt} * sizeof({p.ctype}));")
            return None
        if role == "lut_in":
            if n == "pLevels" and any(q.name == "aLevels" for q in params):
                # Trilinear LUT: pLevels[i] are HOST pointers to the
                # aLevels[i] (= 2) input level values of each cube edge.
                for i in range(k):
                    decls.append(f"  {p.ctype} {n}_h{i}[2] = {{0, 255}};")
                decls.append(f"  {c}{p.ctype} * {n}[{k}] = {{{', '.join(f'{n}_h{i}' for i in range(k))}}};")
                args.append(n)
                return None
            # Per-channel device tables, sized like the single-channel case.
            cnt = HMAX * (WMAX + 16)
            for i in range(k):
                decls.append(f"  auto {n}_{i} = ref_device<{p.ctype}>({cnt}, {seed_for(f'{n}_{i}')});")
            decls.append(f"  {c}{p.ctype} * {n}[{k}] = {{{', '.join(f'{n}_{i}->get()' for i in range(k))}}};")
            args.append(n)
            return None
        if role == "scalar_out":
            # Per-plane histogram bins: nLevels == 2, so one bin per plane.
            for i in range(k):
                decls.append(f"  auto {n}_{i} = ref_device<{p.ctype}>(1, {seed_for(f'{n}_{i}')});")
            decls.append(f"  {p.ctype} * {n}[{k}] = {{{', '.join(f'{n}_{i}->get()' for i in range(k))}}};")
            args.append(n)
            for i in range(k):
                captures.append(f"  ref_capture_bytes(c, {n}_{i}->get(), sizeof({p.ctype}));")
            return None
        return f"unsupported array-of-pointers parameter {n}"

    def emit_batch_list(p: Param) -> Optional[str]:
        """Emit a device batch descriptor list built from host-side structs."""
        role, n, T = batch_role(p), p.name, p.ctype
        if role in ("in", "out", "inplace"):
            t = batch_t(role)
            c = pch.get(n, ch)
            cnt = HMAX * (WMAX * c + 16)
            # The label compressor requires the exact pitch of an unpitched
            # cudaMalloc allocation (oSize.width * sizeof(element)).
            stp = f"{W} * sizeof({t})" if "CompressMarkerLabels" in fn \
                else f"align16({WMAX} * {c} * {SCALAR[t][1]})"
            for b in range(NBATCH):
                decls.append(f"  auto {n}_img{b} = ref_device<{t}>({cnt}, {seed_for(f'{n}_{b}')});")
            decls.append(f"  std::vector<{T}> {n}_host({NBATCH});")
            for b in range(NBATCH):
                decls.append(f"  {n}_host[{b}].pData = {n}_img{b}->get();")
                decls.append(f"  {n}_host[{b}].nStep = {stp};")
                decls.append(f"  {n}_host[{b}].oSize = {{{W}, {H}}};")
            decls.append(f"  auto {n} = std::make_unique<DeviceMemory<{T}>>({NBATCH});")
            decls.append(f"  {n}->copyFromHost({n}_host);")
            args.append(f"{n}->get()")
            if role in ("out", "inplace"):
                for b in range(NBATCH):
                    captures.append(f"  ref_capture_bytes(c, {n}_img{b}->get(), {cnt} * sizeof({t}));")
            return None
        if role == "scratch":
            for b in range(NBATCH):
                decls.append(f"  auto {n}_buf{b} = ref_device<Npp8u>({batch_buf_size or 1024 * 1024}, {seed_for(f'{n}_{b}')});")
            decls.append(f"  std::vector<{T}> {n}_host({NBATCH});")
            for b in range(NBATCH):
                decls.append(f"  {n}_host[{b}].pData = {n}_buf{b}->get();")
                decls.append(f"  {n}_host[{b}].nBufferSize = {batch_buf_size or 1024 * 1024};")
            decls.append(f"  auto {n} = std::make_unique<DeviceMemory<{T}>>({NBATCH});")
            decls.append(f"  {n}->copyFromHost({n}_host);")
            args.append(f"{n}->get()")
            return None
        if role == "roi":
            decls.append(f"  std::vector<{T}> {n}_host({NBATCH});")
            for b in range(NBATCH):
                decls.append(f"  {n}_host[{b}].oSrcRectROI = {{0, 0, {W}, {H}}};")
                decls.append(f"  {n}_host[{b}].oDstRectROI = {{0, 0, {WMAX}, {HMAX}}};")
            decls.append(f"  auto {n} = std::make_unique<DeviceMemory<{T}>>({NBATCH});")
            decls.append(f"  {n}->copyFromHost({n}_host);")
            args.append(f"{n}->get()")
            return None
        if role == "cxr":
            t = batch_t("cxr")
            c = pch.get(n, ch)
            cnt = HMAX * (WMAX * c + 16)
            stp = f"align16({WMAX} * {c} * {SCALAR[t][1]})"
            is_twist = T == "NppiColorTwistBatchCXR"
            is_warp = T in ("NppiWarpAffineBatchCXR", "NppiWarpPerspectiveBatchCXR")
            for b in range(NBATCH):
                decls.append(f"  auto {n}_src{b} = ref_device<{t}>({cnt}, {seed_for(f'{n}_src{b}')});")
                decls.append(f"  auto {n}_dst{b} = ref_device<{t}>({cnt}, {seed_for(f'{n}_dst{b}')});")
                if is_twist:
                    # "32fC" variants twist with a 4x5 matrix plus constant
                    # vector; the others use the plain 3x3 matrix.
                    m = 20 if "32fC" in fn else 9
                    ident = IDENTITY_4X5 if m == 20 else IDENTITY_3X3
                    decls.append(f"  auto {n}_coef{b} = std::make_unique<DeviceMemory<Npp32f>>({m});")
                    decls.append(f"  {n}_coef{b}->copyFromHost({{{ident}}});")
                elif is_warp:
                    m = 6 if T == "NppiWarpAffineBatchCXR" else 9
                    ident = IDENTITY_2X3 if m == 6 else IDENTITY_3X3
                    decls.append(f"  auto {n}_coef{b} = std::make_unique<DeviceMemory<Npp64f>>({m});")
                    decls.append(f"  {n}_coef{b}->copyFromHost({{{ident}}});")
            decls.append(f"  std::vector<{T}> {n}_host({NBATCH});")
            for b in range(NBATCH):
                decls.append(f"  {n}_host[{b}].pSrc = {n}_src{b}->get();")
                decls.append(f"  {n}_host[{b}].nSrcStep = {stp};")
                decls.append(f"  {n}_host[{b}].pDst = {n}_dst{b}->get();")
                decls.append(f"  {n}_host[{b}].nDstStep = {stp};")
                if is_twist:
                    decls.append(f"  {n}_host[{b}].pTwist = {n}_coef{b}->get();")
                elif is_warp:
                    decls.append(f"  {n}_host[{b}].pCoeffs = {n}_coef{b}->get();")
            decls.append(f"  auto {n} = std::make_unique<DeviceMemory<{T}>>({NBATCH});")
            decls.append(f"  {n}->copyFromHost({n}_host);")
            args.append(f"{n}->get()")
            if fn.endswith("BatchInit_Ctx"):
                # The Init entry point's output is the transformed-coefficient
                # block it writes inside each device-side struct; the struct
                # also embeds device pointers, so capture through the masking
                # helper to keep the golden reproducible across runs.
                captures.append(f"  ref_capture_batch_structs(c, {n}->get(), {NBATCH});")
            else:
                for b in range(NBATCH):
                    captures.append(f"  ref_capture_bytes(c, {n}_dst{b}->get(), {cnt} * sizeof({t}));")
                # Warp batch entry points require the matching Init call to
                # precompute the per-image transformed coefficients.
                if T == "NppiWarpAffineBatchCXR":
                    prelude.append(f"  NppStatus stInit = nppiWarpAffineBatchInit_Ctx({n}->get(), {NBATCH}, ctx);")
                    prelude.append("  (void)stInit;")
                elif T == "NppiWarpPerspectiveBatchCXR":
                    prelude.append(f"  NppStatus stInit = nppiWarpPerspectiveBatchInit_Ctx({n}->get(), {NBATCH}, ctx);")
                    prelude.append("  (void)stInit;")
            return None
        return f"unknown batch list role {n}"

    for p in params:
        role = pr(p) if p.is_pointer else None
        # Arrays of device pointers: planar planes, per-channel LUT tables,
        # per-plane histograms, per-plane batch lists.
        if p.is_ptr_array:
            err = emit_ptr_array(p)
            if err:
                return None, err
            continue
        # Batch descriptor lists (device-side struct arrays).
        if p.is_pointer and p.ctype in BATCH_STRUCTS:
            err = emit_batch_list(p)
            if err:
                return None, err
            continue
        # Fixed-size host constant arrays (channel orders, coefficients, ...)
        # and per-plane step arrays.
        if p.is_array:
            if p.ctype == "int" and p.name.endswith("Step"):
                k = p.array_len or 3
                sv = steps.get(p.name, step)
                if not sv.lstrip().startswith("static_cast<int>"):
                    # align16() yields size_t; the brace initializer of an int
                    # array would otherwise warn about narrowing.
                    sv = f"static_cast<int>({sv})"
                dims = "".join(f"[{d}]" for d in (p.arr_dims or (k,)))
                decls.append(f"  int {p.name}{dims} = {{{', '.join([sv] * k)}}};")
                args.append(p.name)
                continue
            init = host_array_init(p, toks, W, H, WMAX, HMAX)
            if init is None:
                return None, f"unsupported host array {p.ctype} {p.name}"
            dims = "".join(f"[{d}]" for d in (p.arr_dims or (p.array_len,)))
            const = "const " if p.is_const else ""
            decls.append(f"  {const}{p.ctype} {p.name}{dims} = {init};")
            args.append(p.name)
            continue
        if p.is_pointer and role == "host":
            if misc:
                # Host-side outputs are captured after the call.  NPP 13.4
                # writes the FilterMedian buffer size as an Npp32u; the other
                # host outputs (hpBufferSize family) are size_t == SIZE_TYPE.
                if p.name == "nBufferSize":
                    decls.append(f"  Npp32u {p.name} = 0;")
                    args.append(f"&{p.name}")
                    captures.append(f"  ref_capture_host_bytes(c, &{p.name}, sizeof(Npp32u));")
                else:
                    decls.append(f"  SIZE_TYPE {p.name} = 0;")
                    args.append(f"&{p.name}")
                    captures.append(f"  ref_capture_host_bytes(c, &{p.name}, sizeof(SIZE_TYPE));")
            else:
                decls.append(f"  SIZE_TYPE {p.name} = 0;")
                args.append(f"&{p.name}")
            continue
        if p.is_pointer and role == "host_in_scalar":
            t = HOST_IN_SCALARS[p.name]
            decls.append(f"  {t} {p.name} = 2;")
            args.append(f"&{p.name}")
            continue
        if p.is_pointer and role == "host_in_array":
            t, k, init = HOST_IN_ARRAYS[p.name]
            decls.append(f"  {t} {p.name}[{k}] = {{{init}}};")
            args.append(p.name)
            continue
        if p.is_pointer and role == "host_out_struct":
            t = HOST_STRUCT_OUT[p.name]
            decls.append(f"  {t} {p.name};")
            decls.append(f"  std::memset(&{p.name}, 0xAB, sizeof({t}));")
            args.append(f"&{p.name}")
            captures.append(f"  ref_capture_host_bytes(c, &{p.name}, sizeof({t}));")
            continue
        if p.is_pointer and role == "null":
            # Optional output the header declares "or NULL" (PBA voronoi).
            args.append("nullptr")
            continue
        if p.is_pointer and role:
            if p.ctype not in SCALAR and p.ctype != "void":
                return None, f"unsupported pointer type {p.ctype}"
            # void* image pointers (FusedAbsDiff family) use the discovered
            # image element type, which defaults to Npp8u.
            t = p.ctype if p.ctype in SCALAR else img_t
            n = p.name
            count = count_for(p)
            decls.append(f"  auto {n} = ref_device<{t}>({count}, {seed_for(n)});")
            args.append(f"{n}->get()")
            if role in ("out", "inplace", "scalar_out", "array_out"):
                # Captured by exact byte size: the ValueKind element size used by
                # ref_capture<T> can exceed sizeof(T) for fixed-point complex types
                # (e.g. Npp16sc is 4 bytes but maps to the 8-byte CF32 kind), which
                # makes the D2H copy read past the allocation for scalar outputs.
                captures.append(f'  ref_capture_bytes(c, {n}->get(), {count} * sizeof({t}));')
            continue
        if p.is_pointer:
            return None, f"unsupported pointer role {p.name}"
        # scalar / enum / struct
        if p.name in arg_ov:
            args.append(str(arg_ov[p.name]))
            continue
        if is_batch and p.name in ("nPerImageBufferSize", "nLargestPerImageBufferSize"):
            args.append(p.name)  # declared from GetBufferSize above
            continue
        if is_batch and p.name in ("nMaxWidth", "nMaxHeight"):
            # Largest image dimension in the batch; buffers are WMAX x HMAX.
            args.append(str(WMAX if p.name == "nMaxWidth" else HMAX))
            continue
        if p.ctype == "NppiNorm":
            # Batched label entry points keep the inf-norm default; the misc
            # stream uses the general enum default (nppiNormL1).
            args.append("nppiNormInf" if is_batch else ENUMS["NppiNorm"])
            continue
        if p.name in steps:
            # Per-pointer step (mixed-type functions have different src/dst
            # element sizes; the steps dict already resolved the right one).
            args.append(steps[p.name])
            continue
        if p.ctype == "NppiSize":
            if "Mask" in p.name or "Kernel" in p.name or "Tpl" in p.name:
                args.append("{3, 3}")
            elif p.name in ("oDstSize", "oDstSizeROI") or (is_batch and p.name == "oSmallestDstSize"):
                # Padded destination geometry (border copies, sub-pixel copies).
                args.append(f"{{{WMAX}, {HMAX}}}")
            elif misc and p.name == "oSrcSize" \
                    and any(q.name == "oSrcOffset" for q in params):
                # Border entry points new to the misc stream: the padded
                # source geometry keeps oSrcOffset + oSizeROI inside the
                # source image (buffers are sized from WMAX/HMAX anyway).
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
            elif p.name == "oRect":
                # RectStdDev reads pSrc/pSqr over (oSizeROI + oRect) in both
                # dimensions (header contract); the default full-ROI rect
                # doubles the required pitch, so use a small interior rect.
                args.append("{0, 0, 3, 3}")
            else:
                args.append(f"{{0, 0, {W}, {H}}}")
            continue
        if p.ctype == "size_t" and "Length" in p.name:
            args.append("static_cast<size_t>(257)")
            continue
        val = scalar_value(p, misc)
        if val is None:
            return None, f"unsupported scalar {p.ctype} {p.name}"
        args.append(val)

    call = f"{fn}({', '.join(args)}, ctx)"
    body = [
        f"TEST(RefSuite, {fn}) {{",
        *decls,
        "  npp_diff::clear_cuda_error();",
        *prelude,
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
        ok, skipped, ext, misc = [], [], [], []
        for r in mrows:
            params = parse_params(r["params"])
            # Functions the legacy rules skipped are emitted by the misc rules
            # into a third chunk stream, so both already-recorded streams
            # chunk exactly as before.
            legacy_reason = skip_reason(r, params, legacy=True)
            text, reason = generate(r, ov, misc=legacy_reason is not None)
            if text is None:
                skipped.append({"function": r["function_name"], "reason": reason})
            elif legacy_reason is not None:
                misc.append((r["function_name"], text))
            elif needs_ext_layout(r["function_name"], params):
                ext.append((r["function_name"], text))
            else:
                ok.append((r["function_name"], text))
        total_ok += len(ok) + len(ext) + len(misc)
        total_skip += len(skipped)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", module.replace(".h", ""))
        # Three streams keep already-recorded chunks byte-identical: base-layout
        # functions chunk exactly as before from index 0, extended-layout
        # functions (planar, pointer arrays, batch lists) start at the first
        # free chunk index of their module, and misc functions (newly
        # constructible under the widened rules) start after both.
        base_chunks = -(-len(ok) // args.chunk_size)
        ext_chunks = -(-len(ext) // args.chunk_size)
        for stream, start in ((ok, 0), (ext, base_chunks),
                              (misc, base_chunks + ext_chunks)):
            for idx in range(0, len(stream), args.chunk_size):
                chunk = stream[idx : idx + args.chunk_size]
                path = args.out_dir / f"test_{stem}_{start + idx // args.chunk_size}_generated.cpp"
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
