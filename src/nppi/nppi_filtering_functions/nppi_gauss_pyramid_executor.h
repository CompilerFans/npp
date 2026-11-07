// Gauss pyramid filter family executor: nppiFilterGaussPyramidLayer{Down,Up}Border.
//
// Host-visible surface (safe for plain.cpp translation units): the entry-point
// templates are declared here and defined + explicitly instantiated in
// nppi_gauss_pyramid_executor.cu.  The public wrappers are generated.
//
// Semantics (baseline NVIDIA NPP 13.4 in the reference implementation):
//   * Down: every destination pixel j samples the source at the fractional
//     position `j * rate + (t - center)` for each kernel tap t, with
//     `center = (nFilterTaps - 1) / 2` (integer division, so even tap counts
//     put the extra tap after the centre).  Non-integer coordinates are
//     resolved with a bilinear interpolation of the two neighbouring pixels;
//     integer rates therefore reduce to plain subsampling.
//   * Kernel taps are applied per axis and are NOT renormalised (the raw
//     coefficient values are used, so a kernel that sums to 1 preserves the
//     DC level; measurements with an unnormalised kernel match the raw tap sums).
//   * Border handling happens on the fractional tap coordinate: REPLICATE
//     clamps it and MIRROR reflects it (reflect-101: -1 -> 1, dim -> dim-2).
//     Only those two border types are accepted.
//   * Coordinates are relative to pSrc.  pSrc points at oSrcOffset inside the
//     oSrcSize image, so the clamp window is [-oSrcOffset, oSrcSize-1-oSrcOffset]
//     (the upstream f12 convention; measured with delta images).
//   * 8u/16u destinations accumulate in float and truncate toward zero, then
//     saturate; 32f passes the float result through unchanged.
//   * Validation order (measured): null src/dst/kernel (-8) -> oSrcSize
//     <= 0 (-6) -> negative ROI (-6) -> oSrcOffset invalid (-11) -> unsupported
//     border (-14) -> nFilterTaps < 3 or >= 100 (-33) -> rate outside
//     (1.0, 10.0] (-5) -> zero-size ROI (NPP_NO_ERROR, skips the step check)
//     -> dst step < row bytes (-16) -> dst step not a multiple of the channel
//     element size (-108).  The source step is never validated.
#ifndef NPPI_GAUSS_PYRAMID_EXECUTOR_H
#define NPPI_GAUSS_PYRAMID_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace filter {

enum PyramidDir {
  kPyramidDown,
  kPyramidUp,
};

// LayerDown/LayerUp Border entry point.  Dir selects the sampling model.
template <typename Src, typename Dst, int Channels, PyramidDir Dir>
NppStatus gaussPyramidBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Dst *pDst, int nDstStep, NppiSize oSizeROI, Npp32f nRate,
                             int nFilterTaps, const Npp32f *pKernel, NppiBorderType eBorderType,
                             cudaStream_t stream);

}  // namespace filter
}  // namespace nppi

#endif  // NPPI_GAUSS_PYRAMID_EXECUTOR_H
