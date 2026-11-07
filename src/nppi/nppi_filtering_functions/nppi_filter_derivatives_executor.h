// Derivative-filter family executor: nppiFilterPrewitt*/Roberts*/Sobel*/Scharr*
// (first and second derivative, plain / Border / Mask / MaskBorder variants).
//
// Host-visible surface (safe for plain.cpp translation units): the entry-point
// templates are declared here and defined + explicitly instantiated in
// nppi_filter_derivatives_executor.cu.  The public wrappers are generated.
//
// Semantics (baseline NVIDIA NPP 13.4 in the reference implementation, measured):
//   * Kernels are convolutions measured from NVIDIA NPP; several differ from
//     the NPP documentation in sign/orientation (see nppi_filter_derivatives_executor.cu).
//   * 8u destinations saturate to [0,255]; 16s destinations to [-32768,32767];
//     32f passes through; integer accumulation is exact.
//   * plain variants read raw pixels around pSrc (no clamping); out-of-buffer
//     reads are caller-visible, exactly like NVIDIA NPP.
//   * Border variants support NPP_BORDER_REPLICATE only (any other value
//     returns NPP_NOT_SUPPORTED_MODE_ERROR).  Reads clamp to the source image
//     bounds: pSrc points at oSrcOffset inside the oSrcSize image, and read
//     coordinates are clamped to [-oSrcOffset, oSrcSize-1-oSrcOffset] in
//     pSrc-relative terms.
//   * AC4 (AlphaPreserve=true) computes channels 0..2 and leaves the alpha
//     channel of pDst untouched; C4 computes all four channels.
//   * Validation order (measured): null pointers -> negative ROI
//     (NPP_SIZE_ERROR) -> zero-size ROI (NPP_NO_ERROR, no writes) ->
//     [border only] bad oSrcSize / negative oSrcOffset
//     (NPP_OUT_OFF_RANGE_ERROR) -> unsupported border type
//     (NPP_NOT_SUPPORTED_MODE_ERROR) -> [mask variants] bad eMaskSize
//     (NPP_MASK_SIZE_ERROR) -> dst step < ROI row bytes (NPP_STEP_ERROR) ->
//     dst step not a multiple of the alignment (NPP_NOT_EVEN_STEP_ERROR;
//     element size for C1/C3, pixel size for C4/AC4).  The source step is
//     never validated.
//   * Legacy validation (PrewittHoriz/VertBorder, RobertsDown/UpBorder only,
//     legacyValidation=true): null -> oSrcSize <= 0 (NPP_SIZE_ERROR) ->
//     negative ROI (NPP_SIZE_ERROR) -> negative oSrcOffset
//     (NPP_OUT_OFF_RANGE_ERROR) -> unsupported border type
//     (NPP_NOT_SUPPORTED_MODE_ERROR) -> zero-size ROI (NPP_NO_ERROR, skips
//     the step check) -> step checks.
#ifndef NPPI_FILTER_DERIVATIVES_EXECUTOR_H
#define NPPI_FILTER_DERIVATIVES_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace filter {

enum DerivOp {
  kPrewittHoriz,
  kPrewittVert,
  kRobertsDown,
  kRobertsUp,
  kSobelHoriz,
  kSobelVert,
  kSobelCross,
  kSobelHorizSecond,
  kSobelVertSecond,
  kScharrHoriz,
  kScharrVert,
};

// Fixed 3x3 filter (Prewitt / Roberts / Sobel first derivative; C1/C3/C4/AC4).
// dst(j,i) = saturate(sum_{a,b} K[a][b] * pSrc[j+a-1][i+b-1]).
template <typename Src, typename Dst, int Channels, bool AlphaPreserve, DerivOp Op>
NppStatus filterDeriv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                      cudaStream_t stream);

// Border variant of the fixed 3x3 filter (NPP_BORDER_REPLICATE only).
// legacyValidation selects the Prewitt/Roberts Border validation order (see
// the header comment); it defaults to the modern (Sobel) order.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve, DerivOp Op>
NppStatus filterDerivBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                            int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, cudaStream_t stream,
                            bool legacyValidation = false);

// Masked 3x3/5x5 filter (single channel): SobelHoriz/SobelVert conversion
// variants, SobelCross, Sobel*Second and the SobelHoriz/VertMask families.
template <typename Src, typename Dst, DerivOp Op>
NppStatus filterDerivMask(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize, cudaStream_t stream);

// Border variant of the masked 3x3/5x5 filter.
template <typename Src, typename Dst, DerivOp Op>
NppStatus filterDerivMaskBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Dst *pDst,
                                int nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiBorderType eBorderType,
                                cudaStream_t stream);

}  // namespace filter
}  // namespace nppi

#endif  // NPPI_FILTER_DERIVATIVES_EXECUTOR_H
