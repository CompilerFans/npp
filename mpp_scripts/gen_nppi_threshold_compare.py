#!/usr/bin/env python3
"""Generator for package p2.3: NPPI threshold and compare operations.

Reads api_analysis/packages/p2.3.csv (functions still marked `missing` in
nppi_threshold_and_compare_operations.h) and emits, from the canonical
signatures given there:

  * src/nppi/nppi_threshold_and_compare_operations/nppi_threshold_compare_generated.cu
      templated 2D kernels + `*_Ctx_impl` entry points (correctness baseline
      NVIDIA NPP 13.4 on RTX 5090),
  * src/nppi/nppi_threshold_and_compare_operations/nppi_threshold_compare_generated.cpp
      public `_Ctx` + non-`_Ctx` wrappers (validation + stream forwarding),
  * test/unit/diff/test_nppi_threshold_compare_generated.cpp
      differential tests (suite NppiThresholdCompareDiff).

Only functions whose CSV status is `missing` are emitted, so the generated
files never collide with the hand-written nppi_threshold.cpp / nppi_compare.cpp
wrappers that already exist for the `real` subset.

Semantics (NVIDIA NPP 13.4):
  Threshold (base/GT/LT/GTVal/LTVal/Val/LTValGTVal)
    * base `nppiThreshold`:  if comp(src, thr) -> thr else src.
    * `_GT`/`_LT`:           if src >/< thr  -> thr else src.
    * `_GTVal`/`_LTVal`:     if src >/< thr  -> val else src.
    * `_Val`:                if comp(src,thr) -> val else src.
    * `_LTValGTVal`:         src<thrLT->valLT; src>thrGT->valGT; else src.
    * C1 takes a scalar threshold/value; C3 and AC4 take 3-element arrays.
    * AC4 (alpha-preserving) applies the operation to RGB only; the alpha
      channel is passed through untouched.
  Compare (nppiCompare / CompareC / CompareEqualEps / CompareEqualEpsC)
    * output is a single Npp8u mask per pixel: 255 when the per-channel
      comparison succeeds for every channel, 0 otherwise.
    * CompareC / CompareEqualEpsC compare against per-channel constants.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "api_analysis/packages/p2.3.csv"
D = ROOT / "src/nppi/nppi_threshold_and_compare_operations"
T = ROOT / "test/unit/diff"

CT = {"8u": "Npp8u", "16u": "Npp16u", "16s": "Npp16s", "32f": "Npp32f"}
ES = {"8u": 1, "16u": 2, "16s": 2, "32f": 4}
KT = {"8u": "U8", "16u": "U16", "16s": "I16", "32f": "F32"}

# layout -> (physical channels, inplace, keep alpha, threshold-array size)
LAY = {
    "C1R": (1, False, False, 1),
    "C1IR": (1, True, False, 1),
    "C3R": (3, False, False, 3),
    "C3IR": (3, True, False, 3),
    "C4R": (4, False, False, 4),
    "AC4R": (4, False, True, 3),
    "AC4IR": (4, True, True, 3),
}

TH_FAM_MODE = {
    "nppiThreshold": "thBase",
    "nppiThreshold_GT": "thGT",
    "nppiThreshold_LT": "thLT",
    "nppiThreshold_GTVal": "thGTV",
    "nppiThreshold_LTVal": "thLTV",
    "nppiThreshold_Val": "thVal",
    "nppiThreshold_LTValGTVal": "thLTVGT",
}
CMP_FAM_MODE = {
    "nppiCompare": 0,
    "nppiCompareC": 1,
    "nppiCompareEqualEps": 2,
    "nppiCompareEqualEpsC": 3,
}

ALLOWED_FAMS = set(TH_FAM_MODE) | set(CMP_FAM_MODE)
LAYOUT_RE = re.compile(r"_(C1R|C1IR|C3R|C3IR|C4R|AC4R|AC4IR)(_Ctx)?$")


def load_missing():
    out = []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            fam = row["family"]
            if fam in ("family",) or row["status"] != "missing":
                continue
            if fam not in ALLOWED_FAMS:
                continue
            m = LAYOUT_RE.search(row["function_name"])
            if not m:
                continue  # nppiFusedAbsDiff handled as a manual special case
            out.append({
                "fn": row["function_name"],
                "fam": fam,
                "ctx": int(row["ctx"]),
                "dt": row["dtype"],
                "params": row["params"],
                "signature": row["signature"],
                "layout": m.group(1),
            })
    return out


def parse_params(p):
    """Return list of (type, name, array_size) parsed from a param-list string."""
    out = []
    for tok in p.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(.*?)\b([A-Za-z_]\w*)\s*(\[\s*(\d+)\s*\])?$", tok)
        if not m:
            out.append((tok, "", ""))
            continue
        out.append((m.group(1).strip(), m.group(2), m.group(4) or ""))
    return out


def cut_ctx(params):
    """Return (args_without_ctx, with_ctx_tuple) for a _Ctx signature params."""
    idx = params.rfind("NppStreamContext")
    if idx < 0:
        return params, params
    head = params[:idx].rstrip(" ,")
    return head, params


def strip_ctx_param(params):
    head, _ = cut_ctx(params)
    return head


def locals_of(params):
    """Return dict name->(type, arraysize) for a param-list (ctx stripped)."""
    body = strip_ctx_param(params)
    loc = {}
    for ty, name, size in parse_params(body):
        loc[name] = (ty, size)
    return loc

# scalar/array name -> ThreshData field
SCALAR_FIELD = {
    "nThreshold": "t", "nValue": "v",
    "nThresholdLT": "lt", "nValueLT": "vlt",
    "nThresholdGT": "gt", "nValueGT": "vgt",
}
ARRAY_FIELD = {
    "rThresholds": "t", "rValues": "v",
    "rThresholdsLT": "lt", "rValuesLT": "vlt",
    "rThresholdsGT": "gt", "rValuesGT": "vgt",
}

# ---------------------------------------------------------------------------
# .cu : executor kernels + *_Ctx_impl entry points
# ---------------------------------------------------------------------------
CU_HEADER = r'''// GENERATED by mpp_scripts/gen_nppi_threshold_compare.py -- DO NOT EDIT.
// NPPI threshold + compare GPU kernels and *_Ctx_impl entry points (package p2.3).
// Semantics baseline: NVIDIA NPP 13.4 on RTX 5090 (see docs/agent_packages/p2.3.md).
#include "npp.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

enum Thode { thBase, thGT, thLT, thGTV, thLTV, thVal, thLTVGT };

template <typename T> struct ThreshData {
  T t[4], v[4], lt[4], vlt[4], gt[4], vgt[4];
  int mode;
  int op;
};

template <typename T> struct CmpData {
  T cst[4];
  float eps;
  int mode;
  int op;
};

inline NppStatus tcKernelStatus() {
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T> __device__ inline bool tcComp(T a, T b, int op) {
  switch (op) {
    case NPP_CMP_LESS:       return a < b;
    case NPP_CMP_LESS_EQ:    return a <= b;
    case NPP_CMP_EQ:         return a == b;
    case NPP_CMP_GREATER_EQ: return a >= b;
    case NPP_CMP_GREATER:    return a > b;
    default: return false;
  }
}

// ---------------------------------------------------------------------------
// Threshold kernel
// ---------------------------------------------------------------------------
template <typename T, int CH, bool INPL, bool KEEP>
__global__ void kThresh(T* pDst, int nDstStep, const T* pSrc, int nSrcStep, int w, int h,
                        ThreshData<T> p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  T* dr = (T*)((char*)pDst + (size_t)y * nDstStep);
  const T* sr = (const T*)((const char*)pSrc + (size_t)y * nSrcStep);
  for (int c = 0; c < CH; ++c) {
    const int si = x * CH + c;
    if (KEEP && c == 3) { dr[si] = sr[si]; continue; }
    const int ti = (CH == 1) ? 0 : c;
    const T sv = INPL ? dr[si] : sr[si];
    T out = sv;
    switch (p.mode) {
      case thBase: if (tcComp(sv, p.t[ti], p.op)) out = p.t[ti]; break;
      case thGT:   if (sv >  p.t[ti]) out = p.t[ti]; break;
      case thLT:   if (sv <  p.t[ti]) out = p.t[ti]; break;
      case thGTV:  if (sv >  p.t[ti]) out = p.v[ti]; break;
      case thLTV:  if (sv <  p.t[ti]) out = p.v[ti]; break;
      case thVal:  if (tcComp(sv, p.t[ti], p.op)) out = p.v[ti]; break;
      case thLTVGT:
        if (sv < p.lt[ti]) out = p.vlt[ti];
        else if (sv > p.gt[ti]) out = p.vgt[ti];
        break;
    }
    dr[si] = out;
  }
}

template <typename T, int CH, bool INPL, bool KEEP>
inline NppStatus thRun(void* pDst, int nDstStep, const void* pSrc, int nSrcStep, int w, int h,
                       ThreshData<T> p, cudaStream_t st) {
  if (w <= 0 || h <= 0) return NPP_NO_ERROR;
  dim3 b(16, 16), g((unsigned)((w + 15) / 16), (unsigned)((h + 15) / 16));
  (void)cudaGetLastError();
  kThresh<T, CH, INPL, KEEP><<<g, b, 0, st>>>((T*)pDst, nDstStep, (const T*)pSrc, nSrcStep, w, h, p);
  return tcKernelStatus();
}

// ---------------------------------------------------------------------------
// Compare kernels (output is a single Npp8u mask per pixel)
// ---------------------------------------------------------------------------
template <typename T, int CH, int MODE>
__global__ void kCompare(const T* p1, int s1, const T* p2, int s2, Npp8u* pDst, int dStep,
                         int w, int h, CmpData<T> p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const T* a = (const T*)((const char*)p1 + (size_t)y * s1);
  const T* b = (const T*)((const char*)p2 + (size_t)y * s2);
  Npp8u* dr = (Npp8u*)((char*)pDst + (size_t)y * dStep);
  bool all = true;
  for (int c = 0; c < CH; ++c) {
    const T av = a[x * CH + c];
    const T bv = (MODE == 0 || MODE == 2) ? b[x * CH + c] : p.cst[c];
    const bool hit = (MODE == 2 || MODE == 3)
                         ? (fabsf((float)av - (float)bv) <= p.eps)
                         : tcComp(av, bv, p.op);
    if (!hit) { all = false; break; }
  }
  dr[x] = all ? (Npp8u)0xFF : (Npp8u)0;
}

template <typename T, int CH, int MODE>
inline NppStatus cmpRun(const void* p1, int s1, const void* p2, int s2, void* pDst, int dStep,
                        int w, int h, CmpData<T> p, cudaStream_t st) {
  if (w <= 0 || h <= 0) return NPP_NO_ERROR;
  dim3 b(16, 16), g((unsigned)((w + 15) / 16), (unsigned)((h + 15) / 16));
  (void)cudaGetLastError();
  kCompare<T, CH, MODE><<<g, b, 0, st>>>((const T*)p1, s1, (const T*)p2, s2, (Npp8u*)pDst, dStep, w, h, p);
  return tcKernelStatus();
}

}  // namespace
'''

def thresh_ctx_info(rows):
    """Yield info dicts for threshold _Ctx_impl functions to emit."""
    for r in rows:
        if r["fam"] not in TH_FAM_MODE or r["ctx"] != 1:
            continue
        fam, dt, lay = r["fam"], r["dt"], r["layout"]
        ch, inpl, keep, tsize = LAY[lay]
        T = CT[dt]
        loc = locals_of(r["params"])
        src = "pSrcDst" if "pSrcDst" in loc else "pSrc"
        dst = "pDst" if "pDst" in loc else src
        sstep = "nSrcDstStep" if "nSrcDstStep" in loc else ("nSrcStep" if "nSrcStep" in loc else None)
        dstep = "nDstStep" if "nDstStep" in loc else sstep
        roi = "oSizeROI" if "oSizeROI" in loc else "oi"
        op = "eComparisonOperation" if "eComparisonOperation" in loc else None
        fields = {}
        for scalar, field in SCALAR_FIELD.items():
            if scalar in loc:
                fields[field] = (scalar, False)
        for arr, field in ARRAY_FIELD.items():
            if arr in loc:
                fields[field] = (arr, True)
        head, _ = cut_ctx(r["params"])
        yield {
            "fn": r["fn"], "mode": TH_FAM_MODE[fam], "T": T, "dt": dt,
            "ch": ch, "inpl": inpl, "keep": keep, "tsize": tsize,
            "src": src, "dst": dst, "sstep": sstep, "dstep": dstep, "roi": roi,
            "op": op, "fields": fields, "params_head": head,
        }


def emit_thresh_impl(i):
    T = i["T"]
    body = []
    body.append("  ThreshData<%s> p{};" % T)
    body.append("  p.mode = %s;" % i["mode"])
    if i["op"]:
        body.append("  p.op = %s;" % i["op"])
    for field, (ident, isarr) in i["fields"].items():
        if isarr:
            for c in range(i["tsize"]):
                body.append("  p.%s[%d] = %s[%d];" % (field, c, ident, c))
        else:
            body.append("  p.%s[0] = %s;" % (field, ident))
    run = ("thRun<%s, %d, %s, %s>((void*)%s, %s, (const void*)%s, %s, %s.width, %s.height, p, nppStreamCtx.hStream)"
           % (T, i["ch"], "true" if i["inpl"] else "false", "true" if i["keep"] else "false",
              i["dst"], i["dstep"], i["src"], i["sstep"], i["roi"], i["roi"]))
    body.append("  return %s;" % run)
    im = impl_name(i["fn"])
    sig = "NppStatus %s(%s, NppStreamContext nppStreamCtx)" % (im, i["params_head"])
    proto = "extern \"C\" NppStatus %s(%s, NppStreamContext);" % (im, i["params_head"])
    return proto + "\n" + sig + " {\n" + "\n".join(body) + "\n}"


def compare_ctx_info(rows):
    for r in rows:
        if r["fam"] not in CMP_FAM_MODE or r["ctx"] != 1:
            continue
        fam, dt, lay = r["fam"], r["dt"], r["layout"]
        ch, _, _, _ = LAY[lay]
        T = CT[dt]
        mode = CMP_FAM_MODE[fam]
        loc = locals_of(r["params"])
        p1 = "pSrc1" if "pSrc1" in loc else ("pSrc" if "pSrc" in loc else None)
        p2 = "pSrc2" if "pSrc2" in loc else p1
        s1 = "nSrc1Step" if "nSrc1Step" in loc else ("nSrcStep" if "nSrcStep" in loc else None)
        s2 = "nSrc2Step" if "nSrc2Step" in loc else s1
        dst = "pDst"
        dstep = "nDstStep" if "nDstStep" in loc else None
        roi = "oSizeROI" if "oSizeROI" in loc else "oi"
        op = "eComparisonOperation" if "eComparisonOperation" in loc else None
        eps = "nEpsilon" if "nEpsilon" in loc else None
        cst = None
        if mode in (1, 3):
            cst = "pConstants" if "pConstants" in loc else ("nConstant" if "nConstant" in loc else None)
        head, _ = cut_ctx(r["params"])
        yield {
            "fn": r["fn"], "T": T, "dt": dt, "ch": ch, "mode": mode,
            "p1": p1, "p2": p2, "s1": s1, "s2": s2,
            "dst": dst, "dstep": dstep, "roi": roi,
            "op": op, "eps": eps, "cst": cst, "params_head": head,
        }


def emit_compare_impl(i):
    T = i["T"]
    body = []
    body.append("  CmpData<%s> p{};" % T)
    body.append("  p.mode = %d;" % i["mode"])
    if i["op"]:
        body.append("  p.op = %s;" % i["op"])
    if i["eps"]:
        body.append("  p.eps = %s;" % i["eps"])
    if i["cst"]:
        if i["ch"] == 1:
            body.append("  p.cst[0] = %s;" % i["cst"])
        else:
            for c in range(i["ch"]):
                body.append("  p.cst[%d] = %s[%d];" % (c, i["cst"], c))
    run = ("cmpRun<%s, %d, %d>((const void*)%s, %s, (const void*)%s, %s, (void*)%s, %s, "
           "%s.width, %s.height, p, nppStreamCtx.hStream)"
           % (T, i["ch"], i["mode"], i["p1"], i["s1"], i["p2"], i["s2"], i["dst"], i["dstep"],
              i["roi"], i["roi"]))
    body.append("  return %s;" % run)
    im = impl_name(i["fn"])
    sig = "NppStatus %s(%s, NppStreamContext nppStreamCtx)" % (im, i["params_head"])
    proto = "extern \"C\" NppStatus %s(%s, NppStreamContext);" % (im, i["params_head"])
    return proto + "\n" + sig + " {\n" + "\n".join(body) + "\n}"


def impl_name(fn):
    return (fn[:-4] + "_impl") if fn.endswith("_Ctx") else (fn + "_impl")


def cu_text(rows):
    L = [CU_HEADER.rstrip(), "", "extern \"C\" {"]
    parts = []
    for i in thresh_ctx_info(rows):
        parts.append(emit_thresh_impl(i))
    for i in compare_ctx_info(rows):
        parts.append(emit_compare_impl(i))
    L.append("\n\n".join(parts))
    L.append("}")
    L.append("")
    return "\n".join(L)

# ---------------------------------------------------------------------------
# .cpp : public wrappers
# ---------------------------------------------------------------------------
CPP_HEADER = '''// GENERATED by mpp_scripts/gen_nppi_threshold_compare.py -- DO NOT EDIT.
// NPPI threshold + compare public wrappers (package p2.3).
#include "npp.h"

static inline NppStatus tcNull(const void *p) { return p ? NPP_SUCCESS : NPP_NULL_POINTER_ERROR; }
static inline NppStatus tcStep(int s) { return (s <= 0) ? NPP_STEP_ERROR : NPP_SUCCESS; }
static inline NppStatus tcStepMin(int s, int need) { return (s < need) ? NPP_STEP_ERROR : NPP_SUCCESS; }
static inline bool tcEmpty(NppiSize o) { return o.width <= 0 || o.height <= 0; }
static inline NppStatus tcBadOp(int op) {
  return (op == NPP_CMP_LESS || op == NPP_CMP_LESS_EQ || op == NPP_CMP_EQ ||
          op == NPP_CMP_GREATER_EQ || op == NPP_CMP_GREATER)
             ? NPP_NO_ERROR : NPP_NOT_SUPPORTED_MODE_ERROR;
}
'''


def arg_names(params):
    body = strip_ctx_param(params)
    return ", ".join(n for n in (p[1] for p in parse_params(body)) if n)


def validations_thresh(i, es):
    ch = i["ch"]
    need_s = "%s.width * %d * %d" % (i["roi"], ch, es)
    need_d = need_s
    L = []
    L.append("  NppStatus s = tcNull((const void*)%s); if (s != NPP_SUCCESS) return s;" % i["src"])
    if i["dst"] != i["src"]:
        L.append("  if ((s = tcNull((const void*)%s)) != NPP_SUCCESS) return s;" % i["dst"])
    L.append("  if ((s = tcStep(%s)) != NPP_SUCCESS) return s;" % i["sstep"])
    if i["dstep"] != i["sstep"]:
        L.append("  if ((s = tcStep(%s)) != NPP_SUCCESS) return s;" % i["dstep"])
    L.append("  if ((s = tcStepMin(%s, %s)) != NPP_SUCCESS) return s;" % (i["sstep"], need_s))
    if i["dstep"] != i["sstep"]:
        L.append("  if ((s = tcStepMin(%s, %s)) != NPP_SUCCESS) return s;" % (i["dstep"], need_d))
    L.append("  if (tcEmpty(%s)) return NPP_NO_ERROR;" % i["roi"])
    if i["op"]:
        L.append("  if ((s = tcBadOp(%s)) != NPP_SUCCESS) return s;" % i["op"])
    return L


def validations_compare(i, es):
    need = "%s.width * %d * %d" % (i["roi"], i["ch"], es)
    need_d = "%s.width" % i["roi"]
    L = []
    L.append("  NppStatus s = tcNull((const void*)%s); if (s != NPP_SUCCESS) return s;" % i["p1"])
    if i["p2"] != i["p1"]:
        L.append("  if ((s = tcNull((const void*)%s)) != NPP_SUCCESS) return s;" % i["p2"])
    L.append("  if ((s = tcNull((const void*)%s)) != NPP_SUCCESS) return s;" % i["dst"])
    L.append("  if ((s = tcStep(%s)) != NPP_SUCCESS) return s;" % i["s1"])
    if i["s2"] != i["s1"]:
        L.append("  if ((s = tcStep(%s)) != NPP_SUCCESS) return s;" % i["s2"])
    L.append("  if ((s = tcStep(%s)) != NPP_SUCCESS) return s;" % i["dstep"])
    L.append("  if ((s = tcStepMin(%s, %s)) != NPP_SUCCESS) return s;" % (i["s1"], need))
    if i["s2"] != i["s1"]:
        L.append("  if ((s = tcStepMin(%s, %s)) != NPP_SUCCESS) return s;" % (i["s2"], need))
    L.append("  if ((s = tcStepMin(%s, %s)) != NPP_SUCCESS) return s;" % (i["dstep"], need_d))
    L.append("  if (tcEmpty(%s)) return NPP_NO_ERROR;" % i["roi"])
    if i["op"]:
        L.append("  if ((s = tcBadOp(%s)) != NPP_SUCCESS) return s;" % i["op"])
    return L


def cpp_ctx_wrapper(i, es, validators):
    im = impl_name(i["fn"])
    proto = "extern \"C\" NppStatus %s(%s, NppStreamContext);" % (im, i["params_head"])
    L = [proto, ""]
    L.append("NppStatus %s(%s, NppStreamContext nppStreamCtx) {" % (i["fn"], i["params_head"]))
    L.extend(validators(i, es))
    callargs = arg_names(i["params_head"])
    L.append("  return %s(%s, nppStreamCtx);" % (im, callargs))
    L.append("}")
    return "\n".join(L)


def cpp_nonctx_wrapper(r):
    # non-Ctx: params identical to the _Ctx head; forward to _Ctx.
    ctxfn = r["fn"] + "_Ctx"
    args = arg_names(r["params"])
    return (
        "NppStatus %s(%s) {\n"
        "  NppStreamContext c; NppStatus st = nppGetStreamContext(&c);\n"
        "  if (st != NPP_SUCCESS) return st;\n"
        "  return %s(%s, c);\n"
        "}" % (r["fn"], r["params"], ctxfn, args)
    )


def cpp_text(rows):
    L = [CPP_HEADER, ""]
    seen = set()
    for i in thresh_ctx_info(rows):
        L.append(cpp_ctx_wrapper(i, ES[i["dt"]], validations_thresh))
        L.append("")
    for i in compare_ctx_info(rows):
        L.append(cpp_ctx_wrapper(i, ES[i["dt"]], validations_compare))
        L.append("")
    for r in rows:
        if r["ctx"] != 0 or r["fn"] in seen:
            continue
        seen.add(r["fn"])
        L.append(cpp_nonctx_wrapper(r))
        L.append("")
    return "\n".join(L)

# ---------------------------------------------------------------------------
# test/unit/diff translation unit
# ---------------------------------------------------------------------------
TEST_HEADER = r'''// Differential tests for package p2.3: NPPI threshold and compare operations.
//   Suite: NppiThresholdCompareDiff
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiThresholdCompareDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiThresholdCompareDiff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

// Threshold, out of place.  The functor binds the family-specific scalar/array params.
template <typename T, typename F>
void thresh_run(const char *fname, const std::string &key, int ch, int width, int height, int pad,
                uint64_t seed, ValueKind kind, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const int dStep = align_step(width * px + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiThresh/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, kind, 0.0, 0.0);
  c.finish();
}

// Threshold, in place.
template <typename T, typename F>
void thresh_run_i(const char *fname, const std::string &key, int ch, int width, int height, int pad,
                  uint64_t seed, ValueKind kind, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const size_t se = (size_t)height * sStep / es;
  std::vector<T> hv = make_uniform<T>(se, seed + 11);
  DeviceMemory<T> &d = arena<T>(se);
  d.copyFromHost(hv);
  clear_cuda_error();
  NppStatus st = f(d.get(), sStep, NppiSize{width, height}, current_ctx());
  Case c("nppiThresh/" + key, fname);
  c.input(hv.data(), hv.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), se, kind, 0.0, 0.0);
  c.finish();
}

// Compare mask output.
template <typename T, typename F>
void cmp_run(const char *fname, const std::string &key, int ch, int width, int height, int pad,
             uint64_t seed, F &&f) {
  const int es = (int)sizeof(T);
  const int px = ch * es;
  const int sStep = align_step(width * px + pad, px);
  const int dStep = align_step(width + pad, 1);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep;
  std::vector<T> h1 = make_uniform<T>(se, seed + 11);
  std::vector<T> h2 = make_uniform<T>(se, seed + 29);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(de, seed + 37);
  DeviceMemory<T> &a1 = arena<T>(se);
  DeviceMemory<T> &a2 = arena<T>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  a1.copyFromHost(h1); a2.copyFromHost(h2); d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(a1.get(), sStep, a2.get(), sStep, d.get(), dStep, NppiSize{width, height}, current_ctx());
  Case c("nppiCmp/" + key, fname);
  c.input(h1.data(), h1.size() * sizeof(T));
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 0.0, 0.0);
  c.finish();
}

}  // namespace

class NppiThresholdCompareDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};
'''


def thresh_extra(fam):
    """Return (op, scalar_extra, array_extra_typed) rendering helpers for a threshold family.
    'scalar_extra' is the C expression for extra positional args using scalar
    identifiers; 'array_extra' uses array identifiers."""
    base = "base" in fam  # nppiThreshold
    if fam == "nppiThreshold":
        return True, "TH, eop", "ARR, eop"
    if fam == "nppiThreshold_GT":
        return False, "TH", "ARR"
    if fam == "nppiThreshold_LT":
        return False, "TH", "ARR"
    if fam == "nppiThreshold_GTVal":
        return False, "TH, VA", "ARR, VAR"
    if fam == "nppiThreshold_LTVal":
        return False, "TH, VA", "ARR, VAR"
    if fam == "nppiThreshold_Val":
        return True, "TH, VA, eop", "ARR, VAR, eop"
    if fam == "nppiThreshold_LTValGTVal":
        return False, "TL, VL, TG, VG", "ARRL, VARL, ARRG, VARG"
    return False, "TH", "ARR"


def test_thresh_body():
    L = []
    # (family, tag, opname for base/val)
    FAM = [
        ("nppiThreshold", "Thr", "NPP_CMP_LESS"),
        ("nppiThreshold_GT", "ThrGt", None),
        ("nppiThreshold_LT", "ThrLt", None),
        ("nppiThreshold_GTVal", "ThrGtVal", None),
        ("nppiThreshold_LTVal", "ThrLtVal", None),
        ("nppiThreshold_Val", "ThrVal", "NPP_CMP_GREATER"),
        ("nppiThreshold_LTValGTVal", "ThrLtValGtVal", None),
    ]
    seed = 2000
    for fam, tag, opname in FAM:
        _, sextra, aextra = thresh_extra(fam)
        L.append("// ===== %s =======" % fam)
        L.append("TEST_F(NppiThresholdCompareDiff, Thresh_%s) {" % tag)
        # C1 8u scalar
        opcall = "NPP_CMP_GREATER" if opname else None
        L.append("  { // 8u C1 17x13")
        texpr = sextra.replace("TH", "100").replace("VA", "200")
        texpr = texpr.replace("TL", "50").replace("VL", "90").replace("TG", "150").replace("VG", "180")
        if opcall:
            texpr = texpr.replace("eop", opcall)
        L.append("    thresh_run<Npp8u>(\"%s_8u_C1R\", \"%s/8u/C1/17x13\", 1, 17, 13, 16, %d, ValueKind::U8," % (fam, tag, seed))
        L.append("      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_8u_C1R_Ctx(s, ss, d, ds, o, %s, c); });" % (fam, texpr))
        L.append("  }")
        # C3 16s array
        L.append("  { // 16s C3 64x32")
        aexpr = aextra.replace("ARR", "thr").replace("VAR", "val")
        aexpr = aexpr.replace("ARRL", "tlt").replace("VARL", "vlt").replace("ARRG", "tgt").replace("VARG", "vgt")
        if opcall:
            aexpr = aexpr.replace("eop", opcall)
        L.append("    static const Npp16s thr[3] = {1000, 3000, -2000};")
        if "val" in aexpr or "vlt" in aexpr or "VAR" in aexpr:
            L.append("    static const Npp16s val[3] = {7, 250, -100};")
        if "tlt" in aexpr:
            L.append("    static const Npp16s tlt[3] = {100, 2000, 4000};")
            L.append("    static const Npp16s vlt[3] = {1, 2, 3};")
            L.append("    static const Npp16s tgt[3] = {8000, 12000, 16000};")
            L.append("    static const Npp16s vgt[3] = {9, 8, 7};")
        L.append("    thresh_run<Npp16s>(\"%s_16s_C3R\", \"%s/16s/C3/64x32\", 3, 64, 32, 32, %d, ValueKind::I16," % (fam, tag, seed + 1))
        L.append("      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_16s_C3R_Ctx(s, ss, d, ds, o, %s, c); });" % (fam, aexpr))
        L.append("  }")
        # AC4 32f array, alpha preserved
        L.append("  { // 32f AC4 1x1 alpha")
        aexpr = aextra.replace("ARR", "thrf").replace("VAR", "valf")
        aexpr = aexpr.replace("ARRL", "tlf").replace("VARL", "vlf").replace("ARG", "tgf").replace("VARG", "vgf").replace("ARRG", "tgf")
        if opcall:
            aexpr = aexpr.replace("eop", opcall)
        L.append("    static const Npp32f thrf[3] = {-0.5f, 0.0f, 0.5f};")
        if "val" in aexpr or "vf" in aexpr:
            L.append("    static const Npp32f valf[3] = {0.25f, -0.25f, 1.0f};")
        if "tlf" in aexpr:
            L.append("    static const Npp32f tlf[3] = {-1.0f, 0.0f, 1.0f}; static const Npp32f vlf[3] = {0.1f,0.2f,0.3f};")
            L.append("    static const Npp32f tgf[3] = {2.0f, 3.0f, 4.0f}; static const Npp32f vgf[3] = {0.4f,0.5f,0.6f};")
        L.append("    thresh_run<Npp32f>(\"%s_32f_AC4R\", \"%s/32f/AC4/1x1\", 4, 1, 1, 16, %d, ValueKind::F32," % (fam, tag, seed + 2))
        L.append("      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_32f_AC4R_Ctx(s, ss, d, ds, o, %s, c); });" % (fam, aexpr))
        L.append("  }")
        # AC4IR 16u in place
        L.append("  { // 16u AC4 in place (IR)")
        aexpr = aextra.replace("ARR", "thru").replace("VAR", "valu")
        aexpr = aexpr.replace("ARRL", "tlu").replace("VARL", "vlu").replace("ARRG", "tgu").replace("VARG", "vgu")
        if opcall:
            aexpr = aexpr.replace("eop", opcall)
        L.append("    static const Npp16u thru[3] = {100, 20000, 40000};")
        if "valu" in aexpr or "vlu" in aexpr:
            L.append("    static const Npp16u valu[3] = {7, 8, 9};")
        if "tlu" in aexpr:
            L.append("    static const Npp16u tlu[3] = {100, 20000, 40000}; static const Npp16u vlu[3] = {1,2,3};")
            L.append("    static const Npp16u tgu[3] = {50000, 60000, 65535}; static const Npp16u vgu[3] = {5,6,7};")
        L.append("    thresh_run_i<Npp16u>(\"%s_16u_AC4IR\", \"%s/16u/AC4IR/64x32\", 4, 64, 32, 32, %d, ValueKind::U16," % (fam, tag, seed + 3))
        L.append("      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_16u_AC4IR_Ctx(d, ds, o, %s, c); });" % (fam, aexpr))
        L.append("  }")
        L.append("}")
        seed += 10
    return L

def _arr(aexpr, which):
    """Ordered array-placeholder substitution (longest first to avoid partial overlap)."""
    for k, name in (("ARRG", "tgt"), ("ARRL", "tlt"), ("VARG", "vgt"), ("VARL", "vlt"),
                    ("ARR", "thr"), ("VAR", "val")):
        aexpr = aexpr.replace(k, name)
    return aexpr


def test_thresh_body_corrected():
    L = []
    FAM = [
        ("nppiThreshold", "Thr", "NPP_CMP_GREATER"),
        ("nppiThreshold_GT", "ThrGt", None),
        ("nppiThreshold_LT", "ThrLt", None),
        ("nppiThreshold_GTVal", "ThrGtVal", None),
        ("nppiThreshold_LTVal", "ThrLtVal", None),
        ("nppiThreshold_Val", "ThrVal", "NPP_CMP_LESS"),
        ("nppiThreshold_LTValGTVal", "ThrLtValGtVal", None),
    ]
    seed = 2000
    for fam, tag, opname in FAM:
        _, sextra, aextra = thresh_extra(fam)
        L.append("// ===== %s =======" % fam)
        L.append("TEST_F(NppiThresholdCompareDiff, Thresh_%s) {" % tag)
        # --- C1 8u scalar, out of place ---
        texpr = sextra.replace("TH", "100").replace("VA", "200")
        texpr = texpr.replace("TL", "50").replace("VL", "90").replace("TG", "150").replace("VG", "180")
        if opname:
            texpr = texpr.replace("eop", opname)
        L.append("  { // 8u C1 17x13")
        L.append("    thresh_run<Npp8u>(\"%s_8u_C1R\", \"%s/8u/C1/17x13\", 1, 17, 13, 16, %d, ValueKind::U8," % (fam, tag, seed))
        L.append("      [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_8u_C1R_Ctx(s, ss, d, ds, o, %s, c); });" % (fam, texpr))
        L.append("  }")
        # --- C3 16s array ---
        aexpr = _arr(aextra, "c3")
        if opname:
            aexpr = aexpr.replace("eop", opname)
        L.append("  { // 16s C3 64x32")
        if "thr" in aexpr:
            L.append("    static const Npp16s thr[3] = {1000, 3000, -2000};")
        if "val" in aexpr:
            L.append("    static const Npp16s val[3] = {7, 250, -100};")
        if "tlt" in aexpr:
            L.append("    static const Npp16s tlt[3] = {100, 2000, 4000};")
            L.append("    static const Npp16s vlt[3] = {1, 2, 3};")
            L.append("    static const Npp16s tgt[3] = {8000, 12000, 16000};")
            L.append("    static const Npp16s vgt[3] = {9, 8, 7};")
        L.append("    thresh_run<Npp16s>(\"%s_16s_C3R\", \"%s/16s/C3/64x32\", 3, 64, 32, 32, %d, ValueKind::I16," % (fam, tag, seed + 1))
        L.append("      [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_16s_C3R_Ctx(s, ss, d, ds, o, %s, c); });" % (fam, aexpr))
        L.append("  }")
        # --- AC4 32f array (alpha preserved) ---
        aexpr = _arr(aextra, "ac4")
        if opname:
            aexpr = aexpr.replace("eop", opname)
        L.append("  { // 32f AC4 1x1 alpha")
        if "thr" in aexpr:
            L.append("    static const Npp32f thr[3] = {-0.5f, 0.0f, 0.5f};")
        if "vgt" in aexpr or "vlt" in aexpr:
            L.append("    static const Npp32f vgt[3] = {0.1f, 0.2f, 0.3f}; static const Npp32f vlt[3] = {0.4f, 0.5f, 0.6f};")
        if "val" in aexpr:
            L.append("    static const Npp32f val[3] = {0.25f, -0.25f, 1.0f};")
        if "tlt" in aexpr:
            L.append("    static const Npp32f tlt[3] = {-1.0f, 0.0f, 1.0f};")
            L.append("    static const Npp32f tgt[3] = {2.0f, 3.0f, 4.0f};")
        L.append("    thresh_run<Npp32f>(\"%s_32f_AC4R\", \"%s/32f/AC4/1x1\", 4, 1, 1, 16, %d, ValueKind::F32," % (fam, tag, seed + 2))
        L.append("      [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_32f_AC4R_Ctx(s, ss, d, ds, o, %s, c); });" % (fam, aexpr))
        L.append("  }")
        # --- AC4IR 16u in place ---
        aexpr = _arr(aextra, "ir")
        aexpr2 = aextra.replace("eop", "NPP_CMP_GREATER") if (fam == "nppiThreshold_Val") else aexpr
        if opname:
            aexpr = aexpr.replace("eop", opname)
        L.append("  { // 16u AC4 in place (IR)")
        if "thr" in aexpr:
            L.append("    static const Npp16u thr[3] = {100, 20000, 40000};")
        if "val" in aexpr:
            L.append("    static const Npp16u val[3] = {7, 8, 9};")
        if "tlt" in aexpr:
            L.append("    static const Npp16u tlt[3] = {100, 20000, 40000};")
            L.append("    static const Npp16u vlt[3] = {1, 2, 3};")
            L.append("    static const Npp16u tgt[3] = {50000, 60000, 65535};")
            L.append("    static const Npp16u vgt[3] = {5, 6, 7};")
        L.append("    thresh_run_i<Npp16u>(\"%s_16u_AC4IR\", \"%s/16u/AC4IR/64x32\", 4, 64, 32, 32, %d, ValueKind::U16," % (fam, tag, seed + 3))
        L.append("      [](Npp16u *d, int ds, NppiSize o, NppStreamContext c) {")
        L.append("        return %s_16u_AC4IR_Ctx(d, ds, o, %s, c); });" % (fam, aexpr))
        L.append("  }")
        L.append("}")
        seed += 10
    return L

OPS_ALL = ["NPP_CMP_LESS", "NPP_CMP_LESS_EQ", "NPP_CMP_EQ", "NPP_CMP_GREATER_EQ", "NPP_CMP_GREATER"]


def test_compare_body():
    L = []
    L.append("// ===== nppiCompare =======")
    L.append("TEST_F(NppiThresholdCompareDiff, Cmp_Compare) {")
    for idx, op in enumerate(OPS_ALL):
        L.append("  { // 8u C1 17x13 compare %s" % op)
        L.append("    cmp_run<Npp8u>(\"nppiCompare_8u_C1R\", \"cmp/%s/C1/17x13\", 1, 17, 13, 16, %d," % (op, 4000 + idx))
        L.append("      [](const Npp8u *a, int as, const Npp8u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
        L.append("        return nppiCompare_8u_C1R_Ctx(a, as, b, bs, d, ds, o, %s, c); });" % op)
        L.append("  }")
    L.append("  { // 16u C3 64x32 EQ")
    L.append("    cmp_run<Npp16u>(\"nppiCompare_16u_C3R\", \"cmp/EQ/C3/64x32\", 3, 64, 32, 32, 4010,")
    L.append("      [](const Npp16u *a, int as, const Npp16u *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompare_16u_C3R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_EQ, c); });")
    L.append("  }")
    L.append("  { // 16s C4 17x13 GE")
    L.append("    cmp_run<Npp16s>(\"nppiCompare_16s_C4R\", \"cmp/GE/C4/17x13\", 4, 17, 13, 16, 4011,")
    L.append("      [](const Npp16s *a, int as, const Npp16s *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompare_16s_C4R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_GREATER_EQ, c); });")
    L.append("  }")
    L.append("  { // 32f AC4 1x1 LE")
    L.append("    cmp_run<Npp32f>(\"nppiCompare_32f_AC4R\", \"cmp/LE/AC4/1x1\", 4, 1, 1, 16, 4012,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompare_32f_AC4R_Ctx(a, as, b, bs, d, ds, o, NPP_CMP_LESS_EQ, c); });")
    L.append("  }")
    L.append("}")
    L.append("// ===== nppiCompareC =======")
    L.append("TEST_F(NppiThresholdCompareDiff, Cmp_CompareC) {")
    L.append("  { // 8u C1 const 17x13")
    L.append("    cmp_run<Npp8u>(\"nppiCompareC_8u_C1R\", \"cc/GT/C1/17x13\", 1, 17, 13, 16, 4020,")
    L.append("      [](const Npp8u *a, int as, const Npp8u *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareC_8u_C1R_Ctx(a, as, 100, d, ds, o, NPP_CMP_GREATER, c); });")
    L.append("  }")
    L.append("  { // 16s C3 const array 64x32")
    L.append("    static const Npp16s cst[3] = {-2000, 5000, 100};")
    L.append("    cmp_run<Npp16s>(\"nppiCompareC_16s_C3R\", \"cc/LT/C3/64x32\", 3, 64, 32, 32, 4021,")
    L.append("      [](const Npp16s *a, int as, const Npp16s *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareC_16s_C3R_Ctx(a, as, cst, d, ds, o, NPP_CMP_LESS, c); });")
    L.append("  }")
    L.append("  { // 16u C4 const array 1x1")
    L.append("    static const Npp16u cst[4] = {100, 200, 300, 400};")
    L.append("    cmp_run<Npp16u>(\"nppiCompareC_16u_C4R\", \"cc/EQ/C4/1x1\", 4, 1, 1, 16, 4022,")
    L.append("      [](const Npp16u *a, int as, const Npp16u *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareC_16u_C4R_Ctx(a, as, cst, d, ds, o, NPP_CMP_EQ, c); });")
    L.append("  }")
    L.append("  { // 32f AC4 const array 17x13")
    L.append("    static const Npp32f cst[4] = {0.5f, -0.5f, 1.5f, 2.0f};")
    L.append("    cmp_run<Npp32f>(\"nppiCompareC_32f_AC4R\", \"cc/GE/AC4/17x13\", 4, 17, 13, 16, 4023,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareC_32f_AC4R_Ctx(a, as, cst, d, ds, o, NPP_CMP_GREATER_EQ, c); });")
    L.append("  }")
    L.append("}")
    L.append("// ===== nppiCompareEqualEps / EqualEpsC =======")
    L.append("TEST_F(NppiThresholdCompareDiff, Cmp_EqualEps) {")
    L.append("  { // 32f C1 eps 0.01 17x13")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEps_32f_C1R\", \"eps/0.01/C1/17x13\", 1, 17, 13, 16, 4030,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEps_32f_C1R_Ctx(a, as, b, bs, d, ds, o, 0.01f, c); });")
    L.append("  }")
    L.append("  { // 32f C3 eps 0.0 64x32")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEps_32f_C3R\", \"eps/0/C3/64x32\", 3, 64, 32, 32, 4031,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEps_32f_C3R_Ctx(a, as, b, bs, d, ds, o, 0.0f, c); });")
    L.append("  }")
    L.append("  { // 32f AC4 eps 1.0 17x13")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEps_32f_AC4R\", \"eps/1/AC4/17x13\", 4, 17, 13, 16, 4032,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEps_32f_AC4R_Ctx(a, as, b, bs, d, ds, o, 1.0f, c); });")
    L.append("  }")
    L.append("  { // 32f C4 eps 0.0 1x1")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEps_32f_C4R\", \"eps/0/C4/1x1\", 4, 1, 1, 16, 4033,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *b, int bs, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEps_32f_C4R_Ctx(a, as, b, bs, d, ds, o, 0.0f, c); });")
    L.append("  }")
    L.append("  { // 32f C1 EqualEpsC const 0.5")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEpsC_32f_C1R\", \"epsc/0.5/C1/17x13\", 1, 17, 13, 16, 4034,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEpsC_32f_C1R_Ctx(a, as, 0.25f, d, ds, o, 0.5f, c); });")
    L.append("  }")
    L.append("  { // 32f C3 EqualEpsC const array 2.0 64x32")
    L.append("    static const Npp32f cst[3] = {0.0f, 1.0f, -1.0f};")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEpsC_32f_C3R\", \"epsc/2/C3/64x32\", 3, 64, 32, 32, 4035,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEpsC_32f_C3R_Ctx(a, as, cst, d, ds, o, 2.0f, c); });")
    L.append("  }")
    L.append("  { // 32f AC4 EqualEpsC const 64x32")
    L.append("    static const Npp32f cst[4] = {0.1f, -0.1f, 1.0f, 0.5f};")
    L.append("    cmp_run<Npp32f>(\"nppiCompareEqualEpsC_32f_AC4R\", \"epsc/0/C4/64x32\", 4, 64, 32, 32, 4036,")
    L.append("      [](const Npp32f *a, int as, const Npp32f *, int, Npp8u *d, int ds, NppiSize o, NppStreamContext c){")
    L.append("        return nppiCompareEqualEpsC_32f_AC4R_Ctx(a, as, cst, d, ds, o, 0.0f, c); });")
    L.append("  }")
    L.append("}")
    return L


def test_contract_body():
    L = []
    L.append("// ===== Contract / status semantics =======")
    L.append("TEST_F(NppiThresholdCompareDiff, Contract) {")
    L.append("  int seed = 4100;")
    L.append("  { // threshold GT: null pointer")
    L.append("    std::vector<Npp8u> hd = make_uniform<Npp8u>(64, seed + 1);")
    L.append("    DeviceMemory<Npp8u> &d = arena<Npp8u>(64); d.copyFromHost(hd);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(nullptr, 8, d.get(), 8, NppiSize{4, 4}, 100, current_ctx());")
    L.append("    Case c(\"nppiThresh/nullPtr\", \"nppiThreshold_GT_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // threshold GT: negative size")
    L.append("    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 11); std::vector<Npp8u> hd = make_uniform<Npp8u>(64, seed + 21);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); DeviceMemory<Npp8u> &d = arena<Npp8u>(64);")
    L.append("    s.copyFromHost(hs); d.copyFromHost(hd);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{-4, 4}, 100, current_ctx());")
    L.append("    Case c(\"nppiThresh/negSize\", \"nppiThreshold_GT_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // threshold GT: empty ROI (no-op)")
    L.append("    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 31); std::vector<Npp8u> hd = make_uniform<Npp8u>(64, seed + 41);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); DeviceMemory<Npp8u> &d = arena<Npp8u>(64);")
    L.append("    s.copyFromHost(hs); d.copyFromHost(hd);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{0, 4}, 100, current_ctx());")
    L.append("    Case c(\"nppiThresh/emptyRoi\", \"nppiThreshold_GT_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // threshold GT: zero step")
    L.append("    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 51); std::vector<Npp8u> hd = make_uniform<Npp8u>(64, seed + 61);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); DeviceMemory<Npp8u> &d = arena<Npp8u>(64);")
    L.append("    s.copyFromHost(hs); d.copyFromHost(hd);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 0, d.get(), 8, NppiSize{4, 4}, 100, current_ctx());")
    L.append("    Case c(\"nppiThresh/zeroStep\", \"nppiThreshold_GT_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // threshold GT: insufficient step")
    L.append("    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 71); std::vector<Npp8u> hd = make_uniform<Npp8u>(64, seed + 81);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); DeviceMemory<Npp8u> &d = arena<Npp8u>(64);")
    L.append("    s.copyFromHost(hs); d.copyFromHost(hd);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiThreshold_GT_8u_C1R_Ctx(s.get(), 4, d.get(), 8, NppiSize{8, 4}, 100, current_ctx());")
    L.append("    Case c(\"nppiThresh/shortStep\", \"nppiThreshold_GT_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // threshold base: illegal comparison operation")
    L.append("    std::vector<Npp8u> hs = make_uniform<Npp8u>(64, seed + 91); std::vector<Npp8u> hd = make_uniform<Npp8u>(64, seed + 101);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); DeviceMemory<Npp8u> &d = arena<Npp8u>(64);")
    L.append("    s.copyFromHost(hs); d.copyFromHost(hd);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiThreshold_16u_C1R_Ctx(s.get(), 8, d.get(), 8, NppiSize{4, 4}, 100, (NppCmpOp)99, current_ctx());")
    L.append("    Case c(\"nppiThresh/badOp\", \"nppiThreshold_16u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // compare: null dst")
    L.append("    std::vector<Npp8u> h = make_uniform<Npp8u>(64, seed + 111);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); s.copyFromHost(h);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiCompare_8u_C1R_Ctx(s.get(), 8, s.get(), 8, nullptr, 8, NppiSize{4, 4}, NPP_CMP_LESS, current_ctx());")
    L.append("    Case c(\"nppiCmp/nullDst\", \"nppiCompare_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("  { // compare: invalid comparison op")
    L.append("    std::vector<Npp8u> h = make_uniform<Npp8u>(64, seed + 121);")
    L.append("    DeviceMemory<Npp8u> &s = arena<Npp8u>(64); s.copyFromHost(h);")
    L.append("    clear_cuda_error();")
    L.append("    NppStatus st = nppiCompare_8u_C1R_Ctx(s.get(), 8, s.get(), 8, nullptr, 8, NppiSize{4, 4}, (NppCmpOp)99, current_ctx());")
    L.append("    Case c(\"nppiCmp/badOp\", \"nppiCompare_8u_C1R\"); c.status(st); c.finish();")
    L.append("  }")
    L.append("}")
    return L


def test_text(rows):
    L = [TEST_HEADER]
    L.extend(test_thresh_body_corrected())
    L.extend(test_compare_body())
    L.extend(test_contract_body())
    return "\n".join(L)


def main():
    rows = load_missing()
    th = len(list(thresh_ctx_info(rows)))
    cp = len(list(compare_ctx_info(rows)))
    print("load_missing: %d funcs (threshold impls %d, compare impls %d)" % (len(rows), th, cp))
    (D / "nppi_threshold_compare_generated.cu").write_text(cu_text(rows))
    (D / "nppi_threshold_compare_generated.cpp").write_text(cpp_text(rows))
    (T / "test_nppi_threshold_compare_generated.cpp").write_text(test_text(rows))
    print("wrote 3 generated files")


if __name__ == "__main__":
    main()
