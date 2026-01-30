#include "nppdefs.h"
#include <cuda_runtime.h>

__global__ void nv12_to_yuv420_y_kernel(const Npp8u *__restrict__ srcY, int srcYStep, Npp8u *__restrict__ dstY,
                                        int dstYStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)srcY + y * srcYStep);
  Npp8u *dstRow = (Npp8u *)((char *)dstY + y * dstYStep);
  dstRow[x] = srcRow[x];
}

__global__ void nv12_to_yuv420_uv_kernel(const Npp8u *__restrict__ srcUV, int srcUVStep, Npp8u *__restrict__ dstU,
                                         int dstUStep, Npp8u *__restrict__ dstV, int dstVStep, int width,
                                         int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int uvWidth = width >> 1;
  int uvHeight = height >> 1;
  if (x >= uvWidth || y >= uvHeight) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)srcUV + y * srcUVStep);
  int srcIdx = (x << 1);
  Npp8u u = srcRow[srcIdx];
  Npp8u v = srcRow[srcIdx + 1];

  Npp8u *dstRowU = (Npp8u *)((char *)dstU + y * dstUStep);
  Npp8u *dstRowV = (Npp8u *)((char *)dstV + y * dstVStep);
  dstRowU[x] = u;
  dstRowV[x] = v;
}

extern "C" cudaError_t nppiNV12ToYUV420_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                        int nSrcUVStep, Npp8u *pDstY, int nDstYStep, Npp8u *pDstU,
                                                        int nDstUStep, Npp8u *pDstV, int nDstVStep, NppiSize oSizeROI,
                                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  nv12_to_yuv420_y_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                              oSizeROI.height);

  dim3 uvGridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                  ((oSizeROI.height / 2) + blockSize.y - 1) / blockSize.y);
  nv12_to_yuv420_uv_kernel<<<uvGridSize, blockSize, 0, stream>>>(pSrcUV, nSrcUVStep, pDstU, nDstUStep, pDstV, nDstVStep,
                                                                 oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}
