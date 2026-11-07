// Wiener filter family executor: nppiFilterWienerBorder.
//
// Host-visible surface: the entry-point template is declared here and defined +
// explicitly instantiated in nppi_filter_wiener_executor.cu.  The public
// wrappers are generated.
//
// Semantics (baseline NVIDIA NPP 13.4 in the reference implementation, reverse-engineered from the
// installed libnppif SASS and measured with
//   * Only NPP_BORDER_REPLICATE is supported; reads clamp to the source image
//     bounds relative to pSrc: [-offX, srcW-1-offX] x [-offY, srcH-1-offY].
//   * The statistics window is NPP's actual (quirky) loop:
//       - mask height h == 1: the mean visits row y+ay with columns
//         ax-(w-1)..ax, while the variance visits row y+ay-1 with the same
//         columns;
//       - otherwise the visited set has h-1 rows, starting at row dy = ay and
//         going down (dy = ay, ay-1,...).  The first row visits all w columns
//         ax-(w-1)..ax; every further row visits w-1 columns
//         ax-(w-1)+1..ax.
//     Duplicated cells from the border clamp are counted multiple times.  The
//     AC4 entry points are the exception: they use the plain w x h rectangle
//     ax-(w-1)..ax by ay-(h-1)..ay for both mean and variance (measured
//     impulse support), so K == N there.
//   * N = oMaskSize.width * oMaskSize.height (not the visited-cell count):
//       mean  = (sum over the mean set) * (1/N)
//       var   = (sum over the variance set of fma(v, v, -mean*mean)) * (1/N)
//     var may therefore be negative for the quirky windows, exactly like NPP.
//   * noise handling: with aNoise[ch] == 0 the output is the variance itself.
//     Otherwise, for integer dtypes aNoise[ch] is scaled by the dtype dynamic
//     range squared (255^2 for 8u, 32767^2 for 16s) on every entry point;
//     32f uses it directly.
//     dst = mean                                      if var <= noise
//     dst = mean + (var - noise)/var * (src - mean)   otherwise
//   * Integer destinations truncate toward zero and saturate.
//   * AC4 computes channels 0..2 only; C4 computes all four channels.
//   * Validation order (measured): aNoise range over the used channels
//     (OUT_OFF_RANGE) -> null src/dst (NULL_POINTER) -> negative ROI (SIZE) ->
//     oSrcSize <= 0 (SIZE) -> oSrcOffset range (OUT_OFF_RANGE) -> border type
//     (NOT_SUPPORTED_MODE) -> mask <= 0 (SIZE_ERROR) -> anchor.x >= mask.width
//     or anchor.y >= mask.height (OUT_OFF_RANGE; negative anchors pass) ->
//     zero-size ROI (NO_ERROR) -> dst step (STEP_ERROR / NOT_EVEN_STEP_ERROR).
//     A null aNoise segfaults on NPP 13.4; we return NULL_POINTER instead.
#ifndef NPPI_FILTER_WIENER_EXECUTOR_H
#define NPPI_FILTER_WIENER_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace filter {

template <typename T, int Channels, bool AlphaPreserve>
NppStatus filterWienerBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, T *pDst,
                             int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                             const Npp32f *aNoise, NppiBorderType eBorderType, cudaStream_t stream);

}  // namespace filter
}  // namespace nppi

#endif  // NPPI_FILTER_WIENER_EXECUTOR_H
