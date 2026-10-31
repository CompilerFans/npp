#include "nppi_color_key_executor.h"

#include <cuda_runtime.h>

#include "npp_kernel_utils.h"

namespace nppi {
namespace color_key {

namespace {

// Color key passed to the kernel by value (plain arrays would decay to
// pointers, and the host-side key lives on the stack).
template <typename T, int Channels> struct ColorKeyVec {
  T v[Channels];
};

// dst = (src1 == key on every channel) ? src2 : src1
template <typename T, int Channels>
__global__ void compColorKeyKernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst,
                                   int nDstStep, NppiSize oSizeROI, ColorKeyVec<T, Channels> key) {
  int x = 0, y = 0;
  if (!npp_kernel::getThreadCoords(x, y, oSizeROI.width, oSizeROI.height)) {
    return;
  }
  const T *src1 = npp_kernel::getSrcRow(pSrc1, y, nSrc1Step) + x * Channels;
  const T *src2 = npp_kernel::getSrcRow(pSrc2, y, nSrc2Step) + x * Channels;
  T *dst = npp_kernel::getDstRow(pDst, y, nDstStep) + x * Channels;

  bool match = true;
#pragma unroll
  for (int c = 0; c < Channels; ++c) {
    match = match && (src1[c] == key.v[c]);
  }
  const T *from = match ? src2 : src1;
#pragma unroll
  for (int c = 0; c < Channels; ++c) {
    dst[c] = from[c];
  }
}

}  // namespace

template <typename T, int Channels>
NppStatus compColorKey(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                       NppiSize oSizeROI, const T *pColorKey, cudaStream_t stream) {
  if (pSrc1 == nullptr || pSrc2 == nullptr || pDst == nullptr || pColorKey == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int minStep = oSizeROI.width * Channels * static_cast<int>(sizeof(T));
  if (nSrc1Step < minStep || nSrc2Step < minStep || nDstStep < minStep) {
    return NPP_STEP_ERROR;
  }

  ColorKeyVec<T, Channels> key;
  for (int c = 0; c < Channels; ++c) {
    key.v[c] = pColorKey[c];
  }

  const dim3 blockSize = npp_kernel::BlockConfig<16, 16>::blockSize();
  const dim3 gridSize = npp_kernel::BlockConfig<16, 16>::gridSize(oSizeROI.width, oSizeROI.height);
  compColorKeyKernel<T, Channels>
      <<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, key);
  return npp_kernel::checkCudaKernelError();
}

// Explicit instantiations for the generated wrappers.
#define NPPI_INSTANTIATE_COMPCOLORKEY(T)                                                          \\
  template NppStatus compColorKey<T, 1>(const T *, int, const T *, int, T *, int, NppiSize, const T *,  \\
                                        cudaStream_t);                                            \\
  template NppStatus compColorKey<T, 3>(const T *, int, const T *, int, T *, int, NppiSize, const T *,  \\
                                        cudaStream_t);                                            \\
  template NppStatus compColorKey<T, 4>(const T *, int, const T *, int, T *, int, NppiSize, const T *,  \\
                                        cudaStream_t)

NPPI_INSTANTIATE_COMPCOLORKEY(Npp8u);

}  // namespace color_key
}  // namespace nppi
