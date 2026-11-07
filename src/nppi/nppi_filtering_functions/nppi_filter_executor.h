// nppiFilter32f / nppiFilterBorder(32f) / nppiFilterRow(32f) / nppiFilterColumn(32f)
// family executor (float and integer kernels, optional replicate border).
//
// Host-visible surface (safe for plain.cpp translation units): the entry-point
// templates are declared here and defined + explicitly instantiated in
// nppi_filter_executor.cu.  The public wrappers are generated.
//
// Semantics (NVIDIA NPP 13.4 in the reference implementation, validated by measurements/reference comparisons):
//   * kernel/mask arrays follow the NPP contract: HOST pointers, staged to the
//     device internally (cudaMemcpyDefault also accepts device pointers, which
//     the test harness uses).
//   * generic filter: dst(x,y)[c] = sum_{i,j} src(x+ax-i, y+ay-j)[c] * k[j*kw+i]
//     (row: src(x+ax-i, y);  column: src(x, y+ay-i)).
//   * float kernels accumulate in float; integer destinations truncate the sum
//     toward zero and then saturate (0.75->0, -1.5->-1, -30000*2 -> -32768).
//   * integer kernels (nppiFilterBorder_<int>) accumulate with int32 wrap-around,
//     divide by nDivisor with C truncation toward zero, then saturate.
//     Exception (measured): the 8u C4R/AC4R ROW/COLUMN 1D families accumulate
//     in unsigned 32-bit SATURATING arithmetic and divide unsigned, so a
//     negative signed sum lands high and saturates to 255.  The 2D 8u
//     families are signed.
//   * AC4 filters the first three channels only; the alpha channel is preserved
//     (also by the empty-kernel/zero-mask fill).
//   * non-border variants read raw (no bounds handling); border variants accept
//     only NPP_BORDER_REPLICATE and clamp image coordinates to [0, oSrcSize-1]
//     in BOTH axes (also for the row/column border variants).
//   * exception (measured): the 32f C1R family clamps taps into the ROI for odd
//     square kernels >= 3 (2D) and odd masks >= 3 (row/column); every other
//     32f layout/shape and all integer families read raw.
//   * 64f row/column never validate the anchor; nMaskSize <= 0 is a single
//     k[0] tap at the (signed) anchor.  64f empty 2D kernels zero the whole ROI.
//   * empty/zero/negative kernel or mask: the destination is zero-filled over
//     limit(dim, k) = clamp(dim - 1 + k, 0, dim) on each axis (so a size-0 axis
//     leaves the last line untouched and a negative size shrinks the fill).
//   * border source image origin is pSrc - oSrcOffset; dst(x,y) filters around
//     image pixel (x + oSrcOffset.x, y + oSrcOffset.y).
//   * validation order (all families): null pointers first, then negative sizes
//     (NPP_SIZE_ERROR), then family-specific checks:
//       - 2D non-border anchor: 1-roiDim <= anchor <= kernelDim (OUT_OFF_RANGE)
//       - row/column integer anchor: 1-roiDim <= anchor <= mask; 64f never checks
//       - border: unsupported mode -> NPP_NOT_SUPPORTED_MODE_ERROR,
//         divisor 0 -> NPP_DIVISOR_ERROR, srcOffset outside [0, oSrcSize) ->
//         NPP_OUT_OFF_RANGE_ERROR; anchor/mask are NOT validated.
//       - dst step below the ROI row size (unsigned compare, checked only when
//         the ROI height is > 0) -> NPP_STEP_ERROR.  The src step is not checked.
//   * empty ROIs (width or height 0) succeed without touching the destination.
//   * 16f variants convert through float (NPP 13.4 accepts them on sm_120).
#ifndef NPPI_FILTER_EXECUTOR_H
#define NPPI_FILTER_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace filter {

// nppiFilter32f_<dt>_<lay>R: general 2D filter, 32f kernel, no border.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filter32f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                    cudaStream_t stream);

// nppiFilterBorder32f_<dt>_<lay>R and nppiFilterBorder_32f_<lay>R:
// 2D filter with border handling, 32f kernel.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterBorder32f(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f *pKernel,
                          NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                          cudaStream_t stream);

// nppiFilterBorder_<dt>_<lay>R (integer source): 2D filter with border handling,
// 32s kernel + divisor.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterBorderDiv(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32s *pKernel,
                          NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                          NppiBorderType eBorderType, cudaStream_t stream);

// nppiFilterRow32f_<dt>_<lay>R / nppiFilterColumn32f_<dt>_<lay>R:
// separable 1D filters, 32f kernel, no border.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterRow32f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                       const Npp32f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream);
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterColumn32f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                          const Npp32f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream);

// nppiFilterRowBorder32f_<dt>_<lay>R / nppiFilterColumnBorder32f_<dt>_<lay>R:
// separable 1D filters with border handling, 32f kernel.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterRowBorder32f(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f *pKernel,
                             int nMaskSize, int nAnchor, NppiBorderType eBorderType,
                             cudaStream_t stream);
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterColumnBorder32f(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f *pKernel,
                                int nMaskSize, int nAnchor, NppiBorderType eBorderType,
                                cudaStream_t stream);

// nppiFilter_<dt>_<lay>R (integer source): general 2D filter, 32s kernel +
// divisor, no border.  Accumulation wraps at 32 bits, division truncates
// toward zero, the quotient saturates (same ring arithmetic as filterBorderDiv).
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterDiv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                    cudaStream_t stream);

// nppiFilter_64f_C1R: general 2D filter, 64f kernel, no border.  The
// accumulation is done in double; only Npp64f destinations are instantiated.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filter64f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp64f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                    cudaStream_t stream);

// nppiFilterRow_<dt>_<lay>R / nppiFilterColumn_<dt>_<lay>R (integer source):
// separable 1D filters, 32s kernel + divisor, no border.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterRowDiv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                       const Npp32s *pKernel, int nMaskSize, int nAnchor, Npp32s nDivisor,
                       cudaStream_t stream);
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterColumnDiv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                          const Npp32s *pKernel, int nMaskSize, int nAnchor, Npp32s nDivisor,
                          cudaStream_t stream);

// nppiFilterRow_64f_C1R / nppiFilterColumn_64f_C1R: separable 1D filters, 64f
// kernel; accumulation in double, Npp64f destinations only.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterRow64f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                       const Npp64f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream);
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterColumn64f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                          const Npp64f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream);

// nppiFilterRowBorder_<dt>_<lay>R / nppiFilterColumnBorder_<dt>_<lay>R
// (integer source): separable 1D filters with border handling, 32s kernel +
// divisor.
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterRowBorderDiv(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32s *pKernel,
                             int nMaskSize, int nAnchor, Npp32s nDivisor,
                             NppiBorderType eBorderType, cudaStream_t stream);
template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterColumnBorderDiv(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32s *pKernel,
                                int nMaskSize, int nAnchor, Npp32s nDivisor,
                                NppiBorderType eBorderType, cudaStream_t stream);

// nppiSumWindowRow_<dt>32f_<lay>R / nppiSumWindowColumn_<dt>32f_<lay>R:
// 1D running-window sum (unit weights) into an Npp32f destination, no border.
// The window is [p-a, p-a+mask-1] (anchor sign inverted w.r.t. the filters);
// taps are read raw from the surrounding buffer.  Measured quirk: the row
// kernel clamps the tap coordinate to W+mask-1 (the column kernel does not).
template <typename Src, int Channels>
NppStatus sumWindowRow(const Src *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                       int nMaskSize, int nAnchor, cudaStream_t stream);
template <typename Src, int Channels>
NppStatus sumWindowColumn(const Src *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                          int nMaskSize, int nAnchor, cudaStream_t stream);

// nppiSumWindowRowBorder_<dt>32f_<lay>R / nppiSumWindowColumnBorder_<dt>32f_<lay>R:
// 1D running-window sum with replicate border handling (full window sum; the
// non-filtered axis is replicate-clamped as well).
template <typename Src, int Channels>
NppStatus sumWindowRowBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Npp32f *pDst, int nDstStep, NppiSize oROI, int nMaskSize, int nAnchor,
                             NppiBorderType eBorderType, cudaStream_t stream);
template <typename Src, int Channels>
NppStatus sumWindowColumnBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                Npp32f *pDst, int nDstStep, NppiSize oROI, int nMaskSize, int nAnchor,
                                NppiBorderType eBorderType, cudaStream_t stream);

// 16f variants use the generic templates: samples are converted through float
// (NPP 13.4 accepts them in the reference implementation; the reference outputs are NO_ERROR).

}  // namespace filter
}  // namespace nppi

#endif  // NPPI_FILTER_EXECUTOR_H
