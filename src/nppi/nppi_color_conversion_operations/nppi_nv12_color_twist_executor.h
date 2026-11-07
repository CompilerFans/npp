#pragma once

// NV12 <-> packed/planar RGB ColorTwist executors for nppi_color_conversion.h:
//   nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx
//   nppiRGBToNV12_{8u,16u}_ColorTwist32f_{C3P2R,P3P2R}_Ctx
//
// Host-visible surface: the entry points are declared here and defined +
// explicitly instantiated in nppi_nv12_color_twist_executor.cu.  The public
// wrappers are generated.
//
// Semantics (baseline NVIDIA NPP 13.4, measured in the reference implementation):
//   * ColorTwist applies the caller's 3x4 matrix to (Y,U,V,1) for
//     NV12ToRGB and to (R,G,B,1) for RGBToNV12.
//   * Each conversion result is rounded half-up (floor(x + 0.5)).
//   * NV12ToRGB clamps each rounded channel to [0, max] of the element type
//     and writes RGB per pixel; chroma is taken from the top-left sample of
//     each 2x2 block (UV plane row y/2, byte/element offset (x/2)*2 and +1).
//     An odd ROI height is processed as ceil(height/2)*2 rows (the extra row
//     belongs to the last 2x2 chroma group); an odd ROI width returns
//     NPP_DOUBLE_SIZE_WARNING and writes nothing.  A height of 1 makes NVIDIA
//     fail its zero-sized chroma launch (NPP_CUDA_KERNEL_EXECUTION_ERROR) and
//     write nothing; mcMPP returns NPP_NO_ERROR without writing.
//   * RGBToNV12 writes the Y plane per pixel (rounded + clamped).  Chroma is
//     rounded per pixel (without clamping), averaged over the 2x2 block with
//     round-half-up ((sum + 2) / 4), then clamped to [0, max] and written as
//     interleaved U,V at (x/2, y/2).  An odd ROI width or height returns
//     NPP_DOUBLE_SIZE_WARNING and writes nothing.
//   * Validation order: null pointers -> negative width/height
//     (NPP_SIZE_ERROR) -> empty ROI (NPP_NO_ERROR, no write) -> odd size
//     (NPP_DOUBLE_SIZE_WARNING, no write) -> step checks.
//   * Deliberate deviations from NVIDIA (documented in the module):
//     NVIDIA dereferences a null plane / null aTwist / null step array and
//     faults asynchronously, and it launches kernels with zero or too-small
//     steps; mcMPP validates every pointer and requires the minimum row
//     steps (NPP_STEP_ERROR).  NVIDIA reports NPP_CUDA_KERNEL_EXECUTION_ERROR
//     for an empty ROI (zero-sized launch); mcMPP returns NPP_NO_ERROR
//     without launching.  These cases are not covered by reference outputs.

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace nv12 {

// T is Npp8u or Npp16u.
template <typename T>
NppStatus nv12ToRgbColorTwist(const T *const pSrc[2], const int aSrcStep[2], T *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp32f aTwist[3][4], cudaStream_t stream);

// Packed 3-channel RGB source.
template <typename T>
NppStatus rgbToNv12ColorTwist(const T *pSrc, int nSrcStep, T *pDst[2], const int aDstStep[2], NppiSize oSizeROI,
                              const Npp32f aTwist[3][4], cudaStream_t stream);

// Planar 3-channel RGB source (pSrc[0..2] = R, G, B planes).
template <typename T>
NppStatus rgbToNv12ColorTwistPlanar(const T *const pSrc[3], const int aSrcStep[3], T *pDst[2],
                                    const int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                    cudaStream_t stream);

}  // namespace nv12
}  // namespace nppi
