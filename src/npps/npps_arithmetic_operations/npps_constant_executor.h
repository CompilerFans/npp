#pragma once

// Shared 1D executor for the npps constant (scalar) arithmetic families:
//   nppsAddC / nppsSubC / nppsSubCRev / nppsMulC / nppsDivC / nppsDivCRev
//   / nppsAndC / nppsOrC / nppsXorC / nppsAddProductC / nppsMulC_Low
//   / nppsNormalize
//
// Host-visible surface (safe for plain .cpp translation units): the executor
// entry-point templates are declared here and defined + explicitly
// instantiated in npps_constant_ops.cu.
//
// Semantics (validated against NVIDIA NPP 13.4 by differential tests):
//   * constant binary  : pDst = op(pSrc, nValue)
//   * in-place (_I/_ISfs): pSrcDst = op(pSrcDst, nValue)
//   * nppsAddProductC  : pDst = pDst + pSrc * nValue      (accumulate)
//   * nppsSubCRev      : pDst = nValue - pSrc
//   * nppsDivCRev      : pDst = nValue / pSrc
//   * nppsMulC_Low     : low part of (pSrc * nValue) written to the output
//   * nppsNormalize    : pDst = (pSrc - vSub) / vDiv
//
// Integer results are computed in a 64-bit accumulator and saturated to the
// destination type.  nScaleFactor is a right shift ("integer result scaling")
// applied with round-half-to-even, matching the element-wise executor.

#include "npp.h"
#include "npp_version_compat.h"
#include "npps_elementwise_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace npps {
namespace constant {

namespace ed = npps::elementwise;
namespace ek = npps::elementwise::detail;

// ---------------------------------------------------------------------------
// Entry points (declared; defined in npps_constant_ops.cu)
//   Same-type constant binary (S == value type == dest type), scaled.
//   In place callers pass pSrcDst as both src and dst.
// ---------------------------------------------------------------------------
template <typename A>
NppStatus addC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename A>
NppStatus subC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename A>
NppStatus subCRev(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename A>
NppStatus mulC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename A>
NppStatus divC(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

template <typename A>
NppStatus divCRev(const A* src, A nValue, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

// Bitwise constant ops: integer only, no scaling.
template <typename A>
NppStatus andC(const A* src, A nValue, A* dst, NppSignalLength nLength, cudaStream_t stream);

template <typename A>
NppStatus orC(const A* src, A nValue, A* dst, NppSignalLength nLength, cudaStream_t stream);

template <typename A>
NppStatus xorC(const A* src, A nValue, A* dst, NppSignalLength nLength, cudaStream_t stream);

// Irregular / mixed-type entry points.
NppStatus mulCFloatToI16(const Npp32f* src, Npp32f nValue, Npp16s* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);
NppStatus mulCInplaceFloatToI64(Npp64f nValue, Npp64s* pSrcDst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);
NppStatus mulCLow(const Npp32f* src, Npp32f nValue, Npp16s* dst, NppSignalLength nLength, cudaStream_t stream);
NppStatus addProductC(const Npp32f* src, Npp32f nValue, Npp32f* dst, NppSignalLength nLength, cudaStream_t stream);

// nppsNormalize: pDst = (pSrc - vSub) / vDiv.  Type V is the offset scalar type
// (same as the sample type), DV the real divisor (real float for complex).
template <typename A, typename DV>
NppStatus normalize(const A* src, A vSub, DV vDiv, A* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream);

}  // namespace constant
}  // namespace npps
