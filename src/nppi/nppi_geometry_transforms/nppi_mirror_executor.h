// nppiMirror / nppiMirrorBatch family executor.
//
// Host-visible surface (safe for plain.cpp translation units): the entry-point
// templates are declared here and defined + explicitly instantiated in
// nppi_mirror_executor.cu.  The public wrappers are generated.
//
// Semantics (baseline NVIDIA NPP 13.4 in the reference implementation, validated by
// reference comparisons reference outputs):
//   * out-of-place: dst[y][x][c] = src[flipY(y)][flipX(x)][c]; row padding is
//     never touched.  AC4 (AlphaPreserve=true) mirrors the first three
//     channels only and leaves the destination alpha channel untouched.
//   * in-place (IR): the alpha channel of AC4 stays at its original position;
//     NVIDIA NPP 13.4 *requires* oROI.width and oROI.height to both be even
//     and returns NPP_SIZE_ERROR otherwise (verified for every plain IR
//     variant and for the 32f batch IR variants).  The swap kernel therefore
//     always sees complete pixel pairs.
//   * error codes, in validation order (each position verified against
//     NVIDIA with combined-contract measurements): negative size -> step <
//     width*channels*sizeof(T) -> empty ROI returns NPP_NO_ERROR (verified
//     with {0,5} and {5,0}) -> step not a multiple of the largest power of
//     two dividing the pixel size (8u C3 -> 1, 32f C3 -> 4, 16s C4 -> 8,
//     32f C4/AC4 -> 16; note a 32f C3 step only needs 4-byte, not 12-byte,
//     granularity) returns NPP_NOT_EVEN_STEP_ERROR -> null pointer ->
//     (in-place only) odd width or height -> invalid flip axis.
//   * nppiMirrorBatch: null batch list -> nBatchSize < 2 ->
//     NPP_BAD_ARGUMENT_ERROR (0 and 1 both verified) -> oSizeROI.width <= 0
//     or height <= 0 -> NPP_SIZE_ERROR -> (IR only) odd dimensions ->
//     invalid flip axis.  NVIDIA raises SIGFPE (divide by zero) for
//     width == 0 there; mcMPP returns the error instead of crashing.  Batch
//     steps live in device
//     memory and are not validated; NVIDIA's batch C4/AC4 kernels fault with
//     a device misaligned-address error when the step is not 16-byte aligned
//     (poisoning the CUDA context), while mcMPP's scalar kernel accepts any
//     step - the difference is not recordable and therefore untested.
#ifndef NPPI_MIRROR_EXECUTOR_H
#define NPPI_MIRROR_EXECUTOR_H

#include "npp.h"
#include "nppi_geometry_transforms.h"
#include <cuda_runtime.h>

namespace nppi {
namespace mirror {

// Out-of-place mirror over an ROI.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirror(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oROI, NppiAxis flip,
                 cudaStream_t stream);

// In-place mirror over an ROI (requires even width and height, see above).
template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirrorInPlace(T *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, cudaStream_t stream);

// Out-of-place mirror batch; each NppiMirrorBatchCXR entry is an image pair.
template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirrorBatch(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                      cudaStream_t stream);

// In-place mirror batch (pSrc is ignored, pDst is mirrored in place; requires
// even width and height).
template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirrorBatchInPlace(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                             cudaStream_t stream);

}  // namespace mirror
}  // namespace nppi

#endif  // NPPI_MIRROR_EXECUTOR_H
