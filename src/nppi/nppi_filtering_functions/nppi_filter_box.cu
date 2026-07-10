#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T> struct FilterBoxAccumulator {
  using Type = long long;
};
template <> struct FilterBoxAccumulator<Npp32f> {
  using Type = double;
};
template <> struct FilterBoxAccumulator<Npp64f> {
  using Type = double;
};

template <typename T>
__global__ void nppiFilterBox_CxR_kernel_impl(const T *pSrc, Npp32s nSrcStep, T *pDst, Npp32s nDstStep, int width,
                                              int height, int maskWidth, int maskHeight, int anchorX, int anchorY,
                                              int channels, int filteredChannels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  using Accumulator = typename FilterBoxAccumulator<T>::Type;
  Accumulator sums[4] = {0, 0, 0, 0};
  for (int maskY = 0; maskY < maskHeight; ++maskY) {
    const int sourceY = y + maskY - anchorY;
    for (int maskX = 0; maskX < maskWidth; ++maskX) {
      const int sourceX = x + maskX - anchorX;
      const T *sourceRow = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + sourceY * nSrcStep);
      for (int channel = 0; channel < filteredChannels; ++channel) {
        sums[channel] += static_cast<Accumulator>(sourceRow[sourceX * channels + channel]);
      }
    }
  }
  T *destinationRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  const Accumulator divisor = static_cast<Accumulator>(maskWidth) * maskHeight;
  for (int channel = 0; channel < filteredChannels; ++channel) {
    destinationRow[x * channels + channel] = static_cast<T>(sums[channel] / divisor);
  }
}

template <typename T>
cudaError_t launchFilterBoxCxR(const T *pSrc, Npp32s nSrcStep, T *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiSize oMaskSize, NppiPoint oAnchor, int channels, int filteredChannels,
                               cudaStream_t stream) {
  const dim3 blockSize(16, 16);
  const dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  nppiFilterBox_CxR_kernel_impl<T><<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, oMaskSize.width, oMaskSize.height, oAnchor.x,
      oAnchor.y, channels, filteredChannels);
  return cudaGetLastError();
}

extern "C" {
cudaError_t nppiFilterBox_8u_CxR_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                        int nFilteredChannels, cudaStream_t stream) {
  return launchFilterBoxCxR(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nChannels,
                            nFilteredChannels, stream);
}
cudaError_t nppiFilterBox_16u_CxR_kernel(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream) {
  return launchFilterBoxCxR(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nChannels,
                            nFilteredChannels, stream);
}
cudaError_t nppiFilterBox_16s_CxR_kernel(const Npp16s *pSrc, Npp32s nSrcStep, Npp16s *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream) {
  return launchFilterBoxCxR(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nChannels,
                            nFilteredChannels, stream);
}
cudaError_t nppiFilterBox_32f_CxR_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream) {
  return launchFilterBoxCxR(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nChannels,
                            nFilteredChannels, stream);
}
cudaError_t nppiFilterBox_64f_CxR_kernel(const Npp64f *pSrc, Npp32s nSrcStep, Npp64f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, int nChannels,
                                         int nFilteredChannels, cudaStream_t stream) {
  return launchFilterBoxCxR(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, nChannels,
                            nFilteredChannels, stream);
}
}
