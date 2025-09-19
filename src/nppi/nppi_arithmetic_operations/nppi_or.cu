#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA内核：两图像按位或
__global__ void nppiOr_8u_C1R_kernel_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrc1Row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *pSrc2Row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    pDstRow[x] = pSrc1Row[x] | pSrc2Row[x];
  }
}

// CUDA内核：图像与常数按位或
__global__ void nppiOrC_8u_C1R_kernel_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst,
                                           int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    pDstRow[x] = pSrcRow[x] | nConstant;
  }
}

extern "C" {
// 两图像按位或
cudaError_t nppiOr_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiOr_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                                oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// 图像与常数按位或
cudaError_t nppiOrC_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiOrC_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, nConstant, pDst, nDstStep,
                                                                 oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}
}