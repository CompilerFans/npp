#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// GPU kernel: bitwise AND of two images
__global__ void nppiAnd_8u_C1R_kernel_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                           Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrc1Row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *pSrc2Row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    pDstRow[x] = pSrc1Row[x] & pSrc2Row[x];
  }
}

// GPU kernel: bitwise AND of image with constant
__global__ void nppiAndC_8u_C1R_kernel_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst,
                                            int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    pDstRow[x] = pSrcRow[x] & nConstant;
  }
}

// GPU kernel: bitwise AND of 3-channel 8u image with constants
__global__ void nppiAndC_8u_C3R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u c0, Npp8u c1, Npp8u c2, Npp8u *pDst,
                                            int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    pDstRow[idx + 0] = pSrcRow[idx + 0] & c0;
    pDstRow[idx + 1] = pSrcRow[idx + 1] & c1;
    pDstRow[idx + 2] = pSrcRow[idx + 2] & c2;
  }
}

// GPU kernel: bitwise AND of 3-channel 16u image with constants
__global__ void nppiAndC_16u_C3R_kernel_impl(const Npp16u *pSrc, int nSrcStep, Npp16u c0, Npp16u c1, Npp16u c2,
                                             Npp16u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *pSrcRow = (const Npp16u *)((const char *)pSrc + y * nSrcStep);
    Npp16u *pDstRow = (Npp16u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    pDstRow[idx + 0] = pSrcRow[idx + 0] & c0;
    pDstRow[idx + 1] = pSrcRow[idx + 1] & c1;
    pDstRow[idx + 2] = pSrcRow[idx + 2] & c2;
  }
}

// GPU kernel: bitwise AND of 4-channel 16u image with constants
__global__ void nppiAndC_16u_C4R_kernel_impl(const Npp16u *pSrc, int nSrcStep, Npp16u c0, Npp16u c1, Npp16u c2,
                                             Npp16u c3, Npp16u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *pSrcRow = (const Npp16u *)((const char *)pSrc + y * nSrcStep);
    Npp16u *pDstRow = (Npp16u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    pDstRow[idx + 0] = pSrcRow[idx + 0] & c0;
    pDstRow[idx + 1] = pSrcRow[idx + 1] & c1;
    pDstRow[idx + 2] = pSrcRow[idx + 2] & c2;
    pDstRow[idx + 3] = pSrcRow[idx + 3] & c3;
  }
}

// GPU kernel: bitwise AND of 3-channel 32s image with constants
__global__ void nppiAndC_32s_C3R_kernel_impl(const Npp32s *pSrc, int nSrcStep, Npp32s c0, Npp32s c1, Npp32s c2,
                                             Npp32s *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32s *pSrcRow = (const Npp32s *)((const char *)pSrc + y * nSrcStep);
    Npp32s *pDstRow = (Npp32s *)((char *)pDst + y * nDstStep);

    int idx = x * 3;
    pDstRow[idx + 0] = pSrcRow[idx + 0] & c0;
    pDstRow[idx + 1] = pSrcRow[idx + 1] & c1;
    pDstRow[idx + 2] = pSrcRow[idx + 2] & c2;
  }
}

// GPU kernel: bitwise AND of 4-channel 32s image with constants
__global__ void nppiAndC_32s_C4R_kernel_impl(const Npp32s *pSrc, int nSrcStep, Npp32s c0, Npp32s c1, Npp32s c2,
                                             Npp32s c3, Npp32s *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32s *pSrcRow = (const Npp32s *)((const char *)pSrc + y * nSrcStep);
    Npp32s *pDstRow = (Npp32s *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    pDstRow[idx + 0] = pSrcRow[idx + 0] & c0;
    pDstRow[idx + 1] = pSrcRow[idx + 1] & c1;
    pDstRow[idx + 2] = pSrcRow[idx + 2] & c2;
    pDstRow[idx + 3] = pSrcRow[idx + 3] & c3;
  }
}

extern "C" {
// Bitwise AND of two images
cudaError_t nppiAnd_8u_C1R_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAnd_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                                 oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// Bitwise AND of image with constant
cudaError_t nppiAndC_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAndC_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, nConstant, pDst, nDstStep,
                                                                  oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// Bitwise AND of 3-channel 8u image with constants
cudaError_t nppiAndC_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, const Npp8u aConstants[3], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAndC_8u_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, aConstants[0], aConstants[1], aConstants[2], pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// Bitwise AND of 3-channel 16u image with constants
cudaError_t nppiAndC_16u_C3R_kernel(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[3], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAndC_16u_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, aConstants[0], aConstants[1], aConstants[2], pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// Bitwise AND of 4-channel 16u image with constants
cudaError_t nppiAndC_16u_C4R_kernel(const Npp16u *pSrc, int nSrcStep, const Npp16u aConstants[4], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAndC_16u_C4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, aConstants[0], aConstants[1],
                                                                   aConstants[2], aConstants[3], pDst, nDstStep,
                                                                   oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// Bitwise AND of 3-channel 32s image with constants
cudaError_t nppiAndC_32s_C3R_kernel(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[3], Npp32s *pDst,
                                    int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAndC_32s_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, aConstants[0], aConstants[1], aConstants[2], pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// Bitwise AND of 4-channel 32s image with constants
cudaError_t nppiAndC_32s_C4R_kernel(const Npp32s *pSrc, int nSrcStep, const Npp32s aConstants[4], Npp32s *pDst,
                                    int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAndC_32s_C4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, aConstants[0], aConstants[1],
                                                                   aConstants[2], aConstants[3], pDst, nDstStep,
                                                                   oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}
}
