// nppiFilterBox / BoxBorder / Min / Max (+Border) / Median (+Border) family
// executor (rank/order-statistics window filters).
//
// Semantics below are measured against NVIDIA NPP 13.4 on an RTX 5090
// (probes: scratch/probe_f9*.cu, notes: scratch/f9_semantics.md):
//
//   * window geometry: dst(x,y)[c] reads src(x + i - ax, y + j - ay)[c] for
//     i in [0, mw), j in [0, mh).  The non-border variants read raw memory
//     (no clamping); the border variants accept
//       NPP_BORDER_REPLICATE (2)  -> clamp to [0, size-1]
//       NPP_BORDER_MIRROR    (4)  -> reflect without repeating the edge
//                                    (-1 -> 1, n -> n-2)
//       value 5                   -> reflect repeating the edge
//                                    (-1 -> 0, n -> n-1)
//     BoxBorder additionally accepts 5, Max/MinBorder accept only REPLICATE,
//     MedianBorder accepts only REPLICATE and MIRROR.
//   * AC4 layouts filter the first three channels only; the destination alpha
//     channel is left untouched.  C3/C4 filter every channel.
//   * box: sum of the raw taps divided by mw*mh.  Integer destinations
//     truncate the quotient toward zero and saturate; 32f accumulates in
//     float, 64f in double.
//   * min/max: plain order statistics over the full window.
//   * median: k-th order statistic.  For an even number of taps NPP averages
//     the two middle samples (integers: trunc((a+b)/2) with C truncation).
//     The pBuffer argument is not validated and its contents are ignored.
//   * median GetBufferSize: measured, ROI-independent bytes value
//       49152 * elementSize * maskW * maskH  on the "buffered" kernel path,
//       0                                    otherwise.
//     Border variant: buffered iff maskW*maskH*elementSize > 192.
//     Plain variant: dispatch measured per dtype/layout (table generated from
//     the 5090 probe: mpp_scripts/gen_median_bufsize_table.py).
//   * validation order (measured, see f9b_valid.out / f9e_edge.out): differs
//     per family - see the entry points below.
#ifndef NPPI_FILTER_WINDOW_EXECUTOR_H
#define NPPI_FILTER_WINDOW_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace filter {

// nppiFilterBox_<dt>_<lay>R (no border, raw reads).
template <typename T, int Channels, bool AlphaPreserve>
NppStatus filterBox(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                    NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);

// nppiFilterBoxBorder_<dt>_<lay>R.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus filterBoxBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          T *pDst, int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize,
                          NppiPoint oAnchor, NppiBorderType eBorderType, cudaStream_t stream);

// nppiFilterMin_<dt>_<lay>R / nppiFilterMax_<dt>_<lay>R.
template <typename T, int Channels, bool AlphaPreserve, bool IsMax>
NppStatus filterMinMax(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                       NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream);

// nppiFilterMinBorder / nppiFilterMaxBorder.
template <typename T, int Channels, bool AlphaPreserve, bool IsMax>
NppStatus filterMinMaxBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             T *pDst, int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize,
                             NppiPoint oAnchor, NppiBorderType eBorderType, cudaStream_t stream);

// nppiFilterMedian_<dt>_<lay>R.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus filterMedian(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                       NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer, cudaStream_t stream);

// nppiFilterMedianBorder_<dt>_<lay>R.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus filterMedianBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             T *pDst, int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize,
                             NppiPoint oAnchor, Npp8u *pBuffer, NppiBorderType eBorderType,
                             cudaStream_t stream);

// nppiFilterMedianGetBufferSize_<dt>_<lay>R (host only).
template <typename T, int Channels>
NppStatus medianGetBufferSize(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u *pBufferSize);

// nppiFilterMedianBorderGetBufferSize_<dt>_<lay>R (host only).
template <typename T, int Channels>
NppStatus medianBorderGetBufferSize(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u *pBufferSize,
                                    NppiBorderType eBorderType);

}  // namespace filter
}  // namespace nppi

#endif  // NPPI_FILTER_WINDOW_EXECUTOR_H
