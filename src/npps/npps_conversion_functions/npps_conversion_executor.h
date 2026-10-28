#pragma once

// Shared 1D executor for the npps conversion + threshold families
// (nppsConvert_* and nppsThreshold_* / _GT / _LT / _GTVal / _LTVal).
//
// Host-visible surface (safe for plain .cpp translation units): executor
// entry-point templates are declared here and defined + explicitly
// instantiated in npps_conversion_ops.cu / npps_threshold_ops.cu.
//
// Convert semantics (baseline NVIDIA NPP 13.4):
//   * plain   : dst = cast(src), integer dests saturate.
//   * scaled  : dst = src / 2^sf;  integer dest uses round-half-to-even.
//   * rounded : dst = round(src / 2^sf, eRoundMode), then saturate.
//   (Differential tests validate the exact rounding / saturation edge cases.)
//
// Threshold semantics (mirrors the nppi threshold contract in
// nppi_threshold_and_compare_operations.h):
//   * nppsThreshold_<T>(level, op): dst = (src OP level) ? level : src
//   * nppsThreshold_GT : dst = (src >  level) ? level : src
//   * nppsThreshold_LT : dst = (src <  level) ? level : src
//   * nppsThreshold_GTVal: dst = (src > level) ? value : src
//   * nppsThreshold_LTVal: dst = (src < level) ? value : src
//   For complex sample types the compare uses the real part only and the
//   "set" value is zero (the level is a real scalar).

#include "npp.h"
#include "npp_version_compat.h"
#include "../npps_arithmetic_operations/npps_elementwise_executor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace npps {
namespace convert {

// dst = convert(src), integer saturation for narrowing.
template <typename A, typename D>
NppStatus convert(const A* src, D* dst, NppSignalLength nLength, cudaStream_t stream);

// dst = convert(src) / 2^sf ; int dest round-half-to-even + saturate.
template <typename A, typename D>
NppStatus convertScale(const A* src, D* dst, NppSignalLength nLength, int nScaleFactor,
                       cudaStream_t stream);

// dst = round(src / 2^sf, mode) saturated to D.
template <typename A, typename D>
NppStatus convertRound(const A* src, D* dst, NppSignalLength nLength, NppRoundMode mode,
                       int nScaleFactor, cudaStream_t stream);

}  // namespace convert

namespace threshold {

// dst = (src OP level) ? level : src
template <typename T, typename L>
NppStatus thresholdOp(const T* src, T* dst, NppSignalLength nLength, L nLevel, NppCmpOp relOp,
                      cudaStream_t stream);

// dst = (src > level) ? level : src
template <typename T, typename L>
NppStatus thresholdGT(const T* src, T* dst, NppSignalLength nLength, L nLevel, cudaStream_t stream);

// dst = (src < level) ? level : src
template <typename T, typename L>
NppStatus thresholdLT(const T* src, T* dst, NppSignalLength nLength, L nLevel, cudaStream_t stream);

// dst = (src > level) ? value : src
template <typename T, typename L>
NppStatus thresholdGTVal(const T* src, T* dst, NppSignalLength nLength, L nLevel, T nValue,
                         cudaStream_t stream);

// dst = (src < level) ? value : src
template <typename T, typename L>
NppStatus thresholdLTVal(const T* src, T* dst, NppSignalLength nLength, L nLevel, T nValue,
                         cudaStream_t stream);

}  // namespace threshold
}  // namespace npps
