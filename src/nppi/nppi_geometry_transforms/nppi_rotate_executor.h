// nppiRotate family executor (module: the AC4 variants plus the corrected
// C1R/C3R/C4R entry points).
//
// Semantics measured in the reference implementation, NVIDIA NPP 13.4 (reference outputs of
// reference comparisons
//   * Destination to source mapping is done in *float32*:
//       fx = ca*(x - nShiftX) - sa*(y - nShiftY)
//       fy = sa*(x - nShiftX) + ca*(y - nShiftY)
//     with ca/sa = cos/sin(nAngle * pi / 180) rounded to float32.  The row-11
//     "exactly 1.5" measurements only reproduce with single precision.
//   * A destination pixel is written when
//       srcROI.x - 0.5 <= fx <= srcROI.x + width - 1   (likewise in y).
//     The asymmetric lower edge is the measured behaviour: pixels whose source
//     position is up to half a pixel left/above the ROI are still sampled.
//     Pixels that fall outside stay untouched (the rotate kernel does not fill).
//   * Interpolation: NN round-half-up, bilinear, and the 4-tap cubic Lagrange
//     kernel described in nppi_warp_sampling.cuh; taps clamp to the image bounds.
//   * AC4 destinations keep their alpha channel untouched.
//   * Contract (contract reference outputs): null pointer -> NPP_NULL_POINTER_ERROR;
//     step <= 0 -> NPP_STEP_ERROR; a step that is not a multiple of the largest
//     power of two dividing channels*sizeof(T) -> NPP_NOT_EVEN_STEP_ERROR (the
//     32f AC4 measurements use 392-byte steps and record exactly that); negative ROI
//     origin -> NPP_SIZE_ERROR; empty/negative ROI ->
//     NPP_WRONG_INTERSECTION_QUAD_WARNING; invalid interpolation ->
//     NPP_INTERPOLATION_ERROR; an empty source image or an empty destination
//     ROI rectangle is a no-op (NPP_NO_ERROR).
#ifndef NPPI_ROTATE_EXECUTOR_H
#define NPPI_ROTATE_EXECUTOR_H

#include "npp.h"
#include "nppi_geometry_transforms.h"
#include <cuda_runtime.h>

namespace nppi {
namespace rotate {

// Packed (interleaved) layouts; AlphaPreserve keeps the destination alpha of
// AC4 layouts untouched.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus rotate(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                 NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation,
                 cudaStream_t stream);

} // namespace rotate
} // namespace nppi

#endif // NPPI_ROTATE_EXECUTOR_H
