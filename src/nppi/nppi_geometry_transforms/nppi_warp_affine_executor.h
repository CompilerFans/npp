// WarpAffine family executors (nppiWarpAffine / Back / Quad / Batch /
// BatchInit).
//
// Host-visible surface (safe for plain.cpp translation units): the entry-point
// templates are declared here and defined + explicitly instantiated in
// nppi_warp_affine_executor.cu.  The public wrappers are generated.
//
// Semantics (NPP 13.4 header contract, pinned by the reference comparisons
// reference comparisons against the reference outputs):
//   * The single-image / quad families map *absolute* (full image)
//     coordinates: the affine coefficients are applied to the destination
//     pixel position to obtain the source sample position.  The batch family
//     maps the ROI-relative index instead (see below).
//   * nppiWarpAffine:       aCoeffs maps source -> destination and is inverted
//                           internally (NPP_COEFFICIENT_ERROR when singular).
//   * nppiWarpAffineBack:   aCoeffs already is the inverse (dst -> src) map.
//   * nppiWarpAffineQuad:   the affine transform is derived from the first
//                           three src/dst quad vertices; a mismatching fourth
//                           vertex yields NPP_AFFINE_QUAD_INCORRECT_WARNING.
//   * nppiWarpAffineBatch:  one shared src/dst ROI for every image in the
//                           batch; the per-image descriptors live in device
//                           memory and must have been initialized by
//                           nppiWarpAffineBatchInit (which pre-inverts the
//                           per-image coefficient matrices).
//   * Samples whose neighbourhood crosses the source image edge replicate the
//     edge pixel (the reference clamps the taps to [0, size-1]).  Destination
//     pixels whose mapped point falls outside the clipped source ROI are left
//     untouched; the gate accepts [roi.x, roi.x + roi.width - 1] inclusive.
//   * nppiWarpAffineBatch maps the destination ROI-relative index and writes
//     at absolute coordinates, gates with the same rule, drops an odd row when
//     its even partner is outside the gate and reuses the even row's x tap
//     block when both rows share a tap base row.
//   * AC4 variants warp channels 0..2 and leave the alpha channel untouched.
//   * Interpolation modes: NPPI_INTER_NN (round half up), NPPI_INTER_LINEAR
//     (bilinear) and NPPI_INTER_CUBIC (Catmull-Rom).
//   * Error paths: null pointers, degenerate image size, empty / disjoint
//     source ROI (NPP_WRONG_INTERSECTION_ROI_ERROR), <= 1 px source ROI
//     intersection (NPP_RECTANGLE_ERROR), degenerate destination ROI, illegal
//     interpolation mode and singular coefficients.  The batch family differs:
//     nBatchSize < 2 is NPP_BAD_ARGUMENT_ERROR, every degenerate ROI is
//     NPP_RECTANGLE_ERROR, per-image steps are not validated and
//     BatchInit(0) reports NPP_CUDA_KERNEL_EXECUTION_ERROR.
//   * Quad maps: the derived affine is rounded to float before inversion (the
//     reference feeds the kernel float coefficients).
#ifndef NPPI_WARP_AFFINE_EXECUTOR_H
#define NPPI_WARP_AFFINE_EXECUTOR_H

#include "npp.h"

#include <cuda_runtime.h>

namespace nppi {
namespace geometry {

// ---------------------------------------------------------------------------
// nppiWarpAffine (forward affine transform, matrix inverted internally)
// ---------------------------------------------------------------------------

// Packed pixel layouts (C1R / C3R / C4R / AC4R).
template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffine(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                     NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, cudaStream_t stream);

// Planar pixel layouts (P3R / P4R).
template <typename T, int Channels>
NppStatus warpAffinePlanar(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                           T *const *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, cudaStream_t stream);

// ---------------------------------------------------------------------------
// nppiWarpAffineBack (coefficients already are the dst -> src map)
// ---------------------------------------------------------------------------

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffineBack(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                         NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, cudaStream_t stream);

template <typename T, int Channels>
NppStatus warpAffineBackPlanar(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               T *const *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                               int eInterpolation, cudaStream_t stream);

// ---------------------------------------------------------------------------
// nppiWarpAffineQuad (affine transform derived from two quads)
// ---------------------------------------------------------------------------

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffineQuad(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                         const double aSrcQuad[4][2], T *pDst, int nDstStep, NppiRect oDstROI,
                         const double aDstQuad[4][2], int eInterpolation, cudaStream_t stream);

template <typename T, int Channels>
NppStatus warpAffineQuadPlanar(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               const double aSrcQuad[4][2], T *const *pDst, int nDstStep, NppiRect oDstROI,
                               const double aDstQuad[4][2], int eInterpolation, cudaStream_t stream);

// ---------------------------------------------------------------------------
// nppiWarpAffineBatch / nppiWarpAffineBatchInit
// ---------------------------------------------------------------------------

// Descriptor list must live in device memory and must have been pre-inverted by
// warpAffineBatchInit.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpAffineBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation,
                          NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream);

// Pre-invert every per-image coefficient matrix (coefficients <- pCoeffs device
// pointer, aTransformedCoeffs <- inverse) of the device-resident batch list.
NppStatus warpAffineBatchInit(NppiWarpAffineBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream);

}  // namespace geometry
}  // namespace nppi

#endif  // NPPI_WARP_AFFINE_EXECUTOR_H
