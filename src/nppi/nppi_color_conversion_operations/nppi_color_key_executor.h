// Color-key composition executors (nppiCompColorKey / nppiAlphaCompColorKey,
// nppi_color_conversion.h).
//
// Host-visible surface: the entry-point templates are declared here and defined
// + explicitly instantiated in nppi_color_key_executor.cu.  The public wrappers
//
// Semantics (baseline NVIDIA NPP 13.4, measured in the reference implementation and validated
// by reference comparisons):
//   * compColorKey: a source-1 pixel whose *every* channel equals the color key
//     is replaced by the source-2 pixel (C4 compares the alpha channel too);
//     all other pixels are copied from source 1.
//   * alphaCompColorKey (8u AC4): source-1 pixels that match the 4-channel key
//     take the source-2 pixel verbatim (alpha included).  Other pixels blend
//     the RGB channels with the Porter-Duff operator selected by nppAlphaOp
//     (fixed-point model in the.cu file); the destination alpha is NOT
//     written for them.  Premultiplied operators (NPPI_OP_ALPHA_*_PREMUL,
//     NPPI_OP_ALPHA_PREMUL) and any out-of-range operator value are measured
//     no-ops that return NPP_NO_ERROR *before* any other validation.
//   * Only the ROI is written; row-tail padding is left untouched.
//   * Validation order: (alphaComp only: operator range) -> null pointers ->
//     negative ROI (NPP_SIZE_ERROR) -> empty ROI (no-op) -> step == 0 or
//     step < ROI row length (NPP_STEP_ERROR, unsigned compare) -> non-aligned
//     step for 4-byte pixel layouts (NPP_NOT_EVEN_STEP_ERROR; 8u C3 has no
//     alignment requirement).  A negative step passes every check and is
//     treated as a successful no-op (NVIDIA launches with it and walks
//     out-of-bounds rows; we refuse to reproduce that).
#ifndef NPPI_COLOR_KEY_EXECUTOR_H
#define NPPI_COLOR_KEY_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace color_key {

// dst = (src1 == key on every channel) ? src2: src1
template <typename T, int Channels>
NppStatus compColorKey(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                       NppiSize oSizeROI, const T *pColorKey, cudaStream_t stream);

// dst = (src1 == key) ? src2: blend(src1, src2, nAlpha1, nAlpha2, op);
// AC4 layouts only (NVIDIA 13.4 exposes nppiAlphaCompColorKey_8u_AC4R_Ctx).
template <typename T, int Channels>
NppStatus alphaCompColorKey(const T *pSrc1, int nSrc1Step, int nAlpha1, const T *pSrc2, int nSrc2Step, int nAlpha2,
                            T *pDst, int nDstStep, NppiSize oSizeROI, const T *pColorKey, NppiAlphaOp nppAlphaOp,
                            cudaStream_t stream);

}  // namespace color_key
}  // namespace nppi

#endif  // NPPI_COLOR_KEY_EXECUTOR_H
