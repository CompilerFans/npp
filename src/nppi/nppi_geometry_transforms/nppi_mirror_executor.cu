// nppiMirror / nppiMirrorBatch family executor: kernels + template
// definitions + explicit instantiations.  See nppi_mirror_executor.h for the
// measured NVIDIA NPP 13.4 semantics this implements.

#include "nppi_mirror_executor.h"

#include <cuda_runtime.h>

namespace nppi {
namespace mirror {
namespace {

constexpr int kBlockDim = 16;

inline bool validFlip(NppiAxis flip) {
  return flip == NPP_HORIZONTAL_AXIS || flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS;
}

inline void flipAxes(NppiAxis flip, int *flipX, int *flipY) {
  *flipX = (flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS) ? 1: 0;
  *flipY = (flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS) ? 1: 0;
}

// Out-of-place mirror kernel.
template <typename T, int Channels, bool AlphaPreserve>
__global__ void mirrorOopKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height, int flipX,
                                int flipY) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int srcX = flipX ? (width - 1 - x): x;
  const int srcY = flipY ? (height - 1 - y): y;
  constexpr int kWrite = AlphaPreserve ? 3: Channels;  // AC4 leaves the alpha untouched
  const T *s = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + (size_t)srcY * nSrcStep) +
               (size_t)srcX * Channels;
  T *d =
      reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + (size_t)y * nDstStep) + (size_t)x * Channels;
#pragma unroll
  for (int c = 0; c < kWrite; ++c) {
    d[c] = s[c];
  }
}

// In-place mirror kernel.  Width and height are validated to be even, so every
// pixel belongs to a disjoint pair (a, b); the thread owning the smaller index
// swaps the pair, which makes the kernel race-free without extra memory.
template <typename T, int Channels, bool AlphaPreserve>
__global__ void mirrorSwapKernel(T *pSrcDst, int nSrcDstStep, int width, int height, int flipX, int flipY) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int srcX = flipX ? (width - 1 - x): x;
  const int srcY = flipY ? (height - 1 - y): y;
  const size_t a = (size_t)y * width + x;
  const size_t b = (size_t)srcY * width + srcX;
  if (a >= b) {
    return;  // the pair is handled by the thread holding the smaller index
  }
  constexpr int kWrite = AlphaPreserve ? 3: Channels;  // AC4 leaves the alpha in place
  T *pa = reinterpret_cast<T *>(reinterpret_cast<char *>(pSrcDst) + (size_t)y * nSrcDstStep) + (size_t)x * Channels;
  T *pb =
      reinterpret_cast<T *>(reinterpret_cast<char *>(pSrcDst) + (size_t)srcY * nSrcDstStep) + (size_t)srcX * Channels;
#pragma unroll
  for (int c = 0; c < kWrite; ++c) {
    const T t = pa[c];
    pa[c] = pb[c];
    pb[c] = t;
  }
}

// Out-of-place mirror batch kernel; blockIdx.z selects the batch entry.
template <typename T, int Channels, bool AlphaPreserve>
__global__ void mirrorBatchOopKernel(const NppiMirrorBatchCXR *pBatchList, int width, int height, int flipX,
                                     int flipY) {
  const NppiMirrorBatchCXR entry = pBatchList[blockIdx.z];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int srcX = flipX ? (width - 1 - x): x;
  const int srcY = flipY ? (height - 1 - y): y;
  constexpr int kWrite = AlphaPreserve ? 3: Channels;  // AC4 leaves the alpha untouched
  const T *s =
      reinterpret_cast<const T *>(reinterpret_cast<const char *>(entry.pSrc) + (size_t)srcY * entry.nSrcStep) +
      (size_t)srcX * Channels;
  T *d = reinterpret_cast<T *>(reinterpret_cast<char *>(entry.pDst) + (size_t)y * entry.nDstStep) +
         (size_t)x * Channels;
#pragma unroll
  for (int c = 0; c < kWrite; ++c) {
    d[c] = s[c];
  }
}

// In-place mirror batch kernel (pSrc is ignored, pDst is mirrored).
template <typename T, int Channels, bool AlphaPreserve>
__global__ void mirrorBatchSwapKernel(const NppiMirrorBatchCXR *pBatchList, int width, int height, int flipX,
                                      int flipY) {
  const NppiMirrorBatchCXR entry = pBatchList[blockIdx.z];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int srcX = flipX ? (width - 1 - x): x;
  const int srcY = flipY ? (height - 1 - y): y;
  const size_t a = (size_t)y * width + x;
  const size_t b = (size_t)srcY * width + srcX;
  if (a >= b) {
    return;
  }
  constexpr int kWrite = AlphaPreserve ? 3: Channels;  // AC4 leaves the alpha in place
  T *pa = reinterpret_cast<T *>(reinterpret_cast<char *>(entry.pDst) + (size_t)y * entry.nDstStep) +
          (size_t)x * Channels;
  T *pb =
      reinterpret_cast<T *>(reinterpret_cast<char *>(entry.pDst) + (size_t)srcY * entry.nDstStep) +
      (size_t)srcX * Channels;
#pragma unroll
  for (int c = 0; c < kWrite; ++c) {
    const T t = pa[c];
    pa[c] = pb[c];
    pb[c] = t;
  }
}

inline dim3 gridSize(int width, int height, int nBatch) {
  const dim3 block(kBlockDim, kBlockDim);
  return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, nBatch);
}

inline NppStatus launchStatus() {
  const cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// NVIDIA NPP 13.4 (reference) requires each step to be a multiple of the
// largest power of two dividing the pixel size in bytes (8u C3 -> 1, 32f C3
// -> 4, 16s C4 -> 8, 32f C4/AC4 -> 16); anything else returns
// NPP_NOT_EVEN_STEP_ERROR. Note C3 is NOT required to be a full pixel
// multiple (32f C3 step 256 with pixel size 12 is accepted).
template <int PixelBytes>
inline bool stepAligned(int step) {
  constexpr int kAlign = (PixelBytes & -PixelBytes);
  static_assert(kAlign >= 1, "pixel size must be positive");
  return (step % kAlign) == 0;
}

}  // namespace

template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirror(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oROI, NppiAxis flip,
                 cudaStream_t stream) {
  static_assert(Channels >= 1 && Channels <= 4, "unsupported channel count");
  static_assert(!AlphaPreserve || Channels == 4, "alpha preservation requires 4 channels");
  if (oROI.width < 0 || oROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const int minStep = oROI.width * Channels * static_cast<int>(sizeof(T));
  if (nSrcStep < minStep || nDstStep < minStep) {
    return NPP_STEP_ERROR;
  }
  if (oROI.width == 0 || oROI.height == 0) {
    return NPP_NO_ERROR;  // empty ROI: NVIDIA NPP 13.4 reports success
  }
  if (!stepAligned<Channels * static_cast<int>(sizeof(T))>(nSrcStep) ||
      !stepAligned<Channels * static_cast<int>(sizeof(T))>(nDstStep)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!validFlip(flip)) {
    return NPP_MIRROR_FLIP_ERROR;
  }
  int flipX = 0, flipY = 0;
  flipAxes(flip, &flipX, &flipY);
  const dim3 block(kBlockDim, kBlockDim);
  mirrorOopKernel<T, Channels, AlphaPreserve>
      <<<gridSize(oROI.width, oROI.height, 1), block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oROI.width,
                                                                   oROI.height, flipX, flipY);
  return launchStatus();
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirrorInPlace(T *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, cudaStream_t stream) {
  static_assert(Channels >= 1 && Channels <= 4, "unsupported channel count");
  static_assert(!AlphaPreserve || Channels == 4, "alpha preservation requires 4 channels");
  if (oROI.width < 0 || oROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  const int minStep = oROI.width * Channels * static_cast<int>(sizeof(T));
  if (nSrcDstStep < minStep) {
    return NPP_STEP_ERROR;
  }
  if (oROI.width == 0 || oROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (!stepAligned<Channels * static_cast<int>(sizeof(T))>(nSrcDstStep)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (pSrcDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  // NVIDIA NPP 13.4 only accepts even width and height for in-place mirror;
  // the parity check fires before the flip check (measured in the reference implementation).
  if ((oROI.width % 2) != 0 || (oROI.height % 2) != 0) {
    return NPP_SIZE_ERROR;
  }
  if (!validFlip(flip)) {
    return NPP_MIRROR_FLIP_ERROR;
  }
  int flipX = 0, flipY = 0;
  flipAxes(flip, &flipX, &flipY);
  const dim3 block(kBlockDim, kBlockDim);
  mirrorSwapKernel<T, Channels, AlphaPreserve>
      <<<gridSize(oROI.width, oROI.height, 1), block, 0, stream>>>(pSrcDst, nSrcDstStep, oROI.width, oROI.height,
                                                                   flipX, flipY);
  return launchStatus();
}

namespace {

// Host-side validation shared by the batch entry points.  Measured order
// (NVIDIA NPP 13.4, reference): nBatchSize < 2 -> negative/zero size -> (IR
// only) odd dimensions -> invalid flip.
inline NppStatus validateBatch(NppiSize oSizeROI, NppiAxis flip, int nBatchSize, bool requireEven) {
  if (nBatchSize < 2) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (requireEven && ((oSizeROI.width % 2) != 0 || (oSizeROI.height % 2) != 0)) {
    return NPP_SIZE_ERROR;
  }
  if (!validFlip(flip)) {
    return NPP_MIRROR_FLIP_ERROR;
  }
  return NPP_NO_ERROR;
}

}  // namespace

template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirrorBatch(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                      cudaStream_t stream) {
  static_assert(Channels >= 1 && Channels <= 4, "unsupported channel count");
  static_assert(!AlphaPreserve || Channels == 4, "alpha preservation requires 4 channels");
  if (pBatchList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  const NppStatus status = validateBatch(oSizeROI, flip, nBatchSize, false);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  int flipX = 0, flipY = 0;
  flipAxes(flip, &flipX, &flipY);
  const dim3 block(kBlockDim, kBlockDim);
  mirrorBatchOopKernel<T, Channels, AlphaPreserve>
      <<<gridSize(oSizeROI.width, oSizeROI.height, nBatchSize), block, 0, stream>>>(pBatchList, oSizeROI.width,
                                                                                    oSizeROI.height, flipX, flipY);
  return launchStatus();
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus mirrorBatchInPlace(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                             cudaStream_t stream) {
  static_assert(Channels >= 1 && Channels <= 4, "unsupported channel count");
  static_assert(!AlphaPreserve || Channels == 4, "alpha preservation requires 4 channels");
  if (pBatchList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  // Same even-dimension restriction as the plain in-place variants; it is
  // enforced inside validateBatch, before the flip check.
  const NppStatus status = validateBatch(oSizeROI, flip, nBatchSize, true);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  int flipX = 0, flipY = 0;
  flipAxes(flip, &flipX, &flipY);
  const dim3 block(kBlockDim, kBlockDim);
  mirrorBatchSwapKernel<T, Channels, AlphaPreserve>
      <<<gridSize(oSizeROI.width, oSizeROI.height, nBatchSize), block, 0, stream>>>(pBatchList, oSizeROI.width,
                                                                                    oSizeROI.height, flipX, flipY);
  return launchStatus();
}

// ---------------------------------------------------------------------------
// Explicit instantiations (exactly the variants the generated wrappers use).
// ---------------------------------------------------------------------------

#define NPPI_INSTANTIATE_MIRROR(T)                                                                                        \
  template NppStatus mirror<T, 1, false>(const T *, int, T *, int, NppiSize, NppiAxis, cudaStream_t);                    \
  template NppStatus mirror<T, 3, false>(const T *, int, T *, int, NppiSize, NppiAxis, cudaStream_t);                    \
  template NppStatus mirror<T, 4, false>(const T *, int, T *, int, NppiSize, NppiAxis, cudaStream_t);                    \
  template NppStatus mirror<T, 4, true>(const T *, int, T *, int, NppiSize, NppiAxis, cudaStream_t);                     \
  template NppStatus mirrorInPlace<T, 1, false>(T *, int, NppiSize, NppiAxis, cudaStream_t);                             \
  template NppStatus mirrorInPlace<T, 3, false>(T *, int, NppiSize, NppiAxis, cudaStream_t);                             \
  template NppStatus mirrorInPlace<T, 4, false>(T *, int, NppiSize, NppiAxis, cudaStream_t);                             \
  template NppStatus mirrorInPlace<T, 4, true>(T *, int, NppiSize, NppiAxis, cudaStream_t);

NPPI_INSTANTIATE_MIRROR(Npp8u)
NPPI_INSTANTIATE_MIRROR(Npp16u)
NPPI_INSTANTIATE_MIRROR(Npp16s)
NPPI_INSTANTIATE_MIRROR(Npp32s)
NPPI_INSTANTIATE_MIRROR(Npp32f)

#define NPPI_INSTANTIATE_MIRROR_BATCH(T)                                                                                  \
  template NppStatus mirrorBatch<T, 1, false>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);              \
  template NppStatus mirrorBatch<T, 3, false>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);              \
  template NppStatus mirrorBatch<T, 4, false>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);              \
  template NppStatus mirrorBatch<T, 4, true>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);               \
  template NppStatus mirrorBatchInPlace<T, 1, false>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);       \
  template NppStatus mirrorBatchInPlace<T, 3, false>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);       \
  template NppStatus mirrorBatchInPlace<T, 4, false>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);       \
  template NppStatus mirrorBatchInPlace<T, 4, true>(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, cudaStream_t);

NPPI_INSTANTIATE_MIRROR_BATCH(Npp32f)

}  // namespace mirror
}  // namespace nppi
