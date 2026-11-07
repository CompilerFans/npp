#pragma once

// Planar (P3R/IP3R) gamma executors for the nppiGammaFwd / nppiGammaInv
// families.
//
// NVIDIA NPP 13.4 semantics, measured in the reference implementation:
//   * Every byte of each of the three planes inside the ROI is mapped through
//     the family LUT (the tables in nppi_gamma.cu, verified byte-exact).
//     Row padding past the ROI is untouched.
//   * Host validation: negative width/height -> NPP_SIZE_ERROR; empty ROI ->
//     NPP_NO_ERROR (short-circuits the step checks); step == 0 or
//     (unsigned)step < width -> NPP_STEP_ERROR per buffer; steps have no
//     alignment requirement (8u planar).
//   * NVIDIA only null-checks plane 0 (pSrc[0]/pDst[0]/pSrcDst[0]); a null
//     plane 1/2 pointer returns NPP_NO_ERROR and then the kernel faults the
//     CUDA context asynchronously.  mcMPP deliberately validates all planes
//     (documented deviation: a crash cannot be a semantic).
//   * NVIDIA returns NPP_NO_ERROR for negative steps and launches a kernel
//     that walks out-of-bounds rows (nondeterministic output); mcMPP returns
//     NPP_NO_ERROR without launching.  Negative steps are not covered by
//     reference outputs.

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace gamma {

// direction: 0 = forward LUT, 1 = inverse LUT.
NppStatus gammaPlanar(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                      int direction, cudaStream_t stream);

// direction: 0 = forward LUT, 1 = inverse LUT.
NppStatus gammaPlanarInplace(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, int direction,
                             cudaStream_t stream);

}  // namespace gamma
}  // namespace nppi
