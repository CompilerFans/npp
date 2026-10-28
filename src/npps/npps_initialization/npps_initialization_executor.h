#pragma once

// Host-visible surface for the npps initialization families:
//   nppsSet / nppsCopy / nppsZero
//
// These are 1-D, element-wise, same-type operations with no workspace:
//   * nppsSet   : dst[i] = value
//   * nppsCopy  : dst[i] = src[i]
//   * nppsZero  : dst[i] = 0  (implemented with cudaMemsetAsync)
//
// The executor entry-point templates are declared here and defined + explicitly
// instantiated in npps_init_ops.cu so plain .cpp wrappers can dispatch to them.
//
// Error contract (kept aligned with the pre-existing npps_set.cpp hand-written
// entries and re-validated against NVIDIA NPP 13.4 by differential tests):
//   * null destination/source -> NPP_NULL_POINTER_ERROR
//   * nLength == 0            -> NPP_SIZE_ERROR
//   * nLength <  0            -> NPP_SIZE_ERROR

#include "npp.h"
#include "npp_version_compat.h"
#include <cuda_runtime.h>

namespace npps {
namespace init {

template <typename T>
NppStatus set(T value, T *pDst, NppSignalLength nLength, cudaStream_t stream);

template <typename T>
NppStatus copy(const T *pSrc, T *pDst, NppSignalLength nLength, cudaStream_t stream);

template <typename T>
NppStatus zero(T *pDst, NppSignalLength nLength, cudaStream_t stream);

}  // namespace init
}  // namespace npps
