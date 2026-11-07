// Filtering long-tail executor: nppiFilterBilateralGaussBorder,
// nppiFilterThresholdAdaptiveBoxBorder and nppiFilterHarrisCornersBorder.
//
// These three families share the Border validation skeleton of the filtering
// module but each has its own arithmetic; the common contract lives here.
//
// Measured semantics (reference, NVIDIA NPP 13.4, measurements):
//
// Bilateral (documented formula, measured for step 1; step > 1 is
// implemented with the literal `x + w*step` sampling and is NOT bit-exact
// against NPP, which snaps the kernel centre to a grid - see f13_notes.md):
//   dst = sum(W1*W2*S) / sum(W1*W2) over the (2r+1)^2 kernel,
//   W1 = exp(-(S - S_centre)^2 / (2*nValSquareSigma)),
//   W2 = exp(-(h^2 + w^2) / (2*nPosSquareSigma)),
//   REPLICATE and MIRROR borders are accepted; 8u/16u truncate toward zero.
//
// ThresholdAdaptiveBox:
//   avg = mean of the oMaskSize window (REPLICATE), dst = nValGT when
//   src > avg - nDelta else nValLE.  The mask must be odd and square
//   (even / non-square -> NPP_MASK_SIZE_ERROR); only REPLICATE is supported.
//
// HarrisCorners:
//   gradients with the Sobel (3x3/5x5) or Scharr (3x3) mask, products of the
//   gradients summed over the eAvgWindowSize window (3x3/5x5, REPLICATE), and
//   H = ((XX*YY - XY*XY) - nK*(XX+YY)^2) * nScale.
//   The device buffer is required (null -> NPP_NULL_POINTER_ERROR); a null
//   destination is accepted and produces no output (NPP 13.4 quirk).
#ifndef NPPI_FILTER_TAIL_EXECUTOR_H
#define NPPI_FILTER_TAIL_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace filter {

template <typename Src, typename Dst, int Channels>
NppStatus bilateralGaussBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                               Dst *pDst, int nDstStep, NppiSize oSizeROI, int nRadius,
                               int nStepBetweenSrcPixels, Npp32f nValSquareSigma,
                               Npp32f nPosSquareSigma, NppiBorderType eBorderType,
                               cudaStream_t stream);

NppStatus thresholdAdaptiveBoxBorder(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                     NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, Npp32f nDelta,
                                     Npp8u nValGT, Npp8u nValLE, NppiBorderType eBorderType,
                                     cudaStream_t stream);

NppStatus harrisCornersBorder(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                              NppiPoint oSrcOffset, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                              NppiMaskSize eAvgWindowSize, Npp32f nK, Npp32f nScale,
                              NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                              cudaStream_t stream);

}  // namespace filter
}  // namespace nppi

#endif  // NPPI_FILTER_TAIL_EXECUTOR_H
