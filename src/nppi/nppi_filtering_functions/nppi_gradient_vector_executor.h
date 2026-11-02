// GradientVector executor: nppiGradientVector{Sobel,Prewitt,Scharr}Border.
//
// Semantics measured against NVIDIA NPP 13.4 (RTX 5090), see
// tools/probes/f13_probe{4..10}.cu:
//   * X output = -(Vert response) for Sobel/Prewitt but +(Vert response) for
//     Scharr; Y output = +(Horiz response) for all three.  The 3x3 tables are
//     the same taps as the derivative-filter executor; Sobel additionally
//     supports the 5x5 mask (Prewitt/Scharr reject it with -33).
//   * Magnitude: L1 = |X|+|Y|, L2 = sqrt(X^2+Y^2); a 16s destination is
//     truncated toward zero, a 32f destination keeps the float value.
//     Angle = atan2(Y, X) in radians (NPP evaluates it in double precision;
//     the difference to atan2f is <= 1 ulp, far below the differential-test
//     tolerance).
//   * Outputs are individually optional; a null output is not written.  All
//     four null -> NPP_NULL_POINTER_ERROR.
//   * C3C1R: gradients are computed per source channel and the channel with
//     the largest L2 (X^2+Y^2, first channel wins ties) provides ALL outputs.
//   * Validation order: null src -> all outputs null -> negative ROI -> zero
//     ROI (returns) -> step of the FIRST non-null output (min size, then
//     element alignment) -> srcSize -> offset (must be inside the source)
//     -> border type (REPLICATE only) -> norm (L1/L2 only) -> mask size.
//     Only the first non-null output's step is validated; the source step is
//     never validated.  The ROI may overhang the source image (reads clamp).
#pragma once

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace gv {

enum GvOp {
  kSobel,
  kPrewitt,
  kScharr,
};

template <typename Src, typename Dst, int SrcChannels, GvOp Op>
NppStatus gradientVectorBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                               Dst *pDstX, int nDstXStep, Dst *pDstY, int nDstYStep, Dst *pDstMag,
                               int nDstMagStep, Npp32f *pDstAngle, int nDstAngleStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType,
                               cudaStream_t stream);

}  // namespace gv
}  // namespace nppi
