#include "npp.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Absolute value function for different types
__device__ __forceinline__ Npp16s abs_op(Npp16s val) { return (val < 0) ? -val : val; }

__device__ __forceinline__ half abs_op(half val) { return __habs(val); }

__device__ __forceinline__ Npp32f abs_op(Npp32f val) { return fabsf(val); }

// Generic absolute value kernel (not in-place)
template <typename T, int nChannels>
__global__ void abs_kernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep, int width,
                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcPtr = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstPtr = (T *)((char *)pDst + y * nDstStep);

    int idx = x * nChannels;

#pragma unroll
    for (int c = 0; c < nChannels; c++) {
      dstPtr[idx + c] = abs_op(srcPtr[idx + c]);
    }
  }
}

// Generic absolute value kernel (in-place)
template <typename T, int nChannels>
__global__ void abs_inplace_kernel(T *__restrict__ pSrcDst, int nSrcDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    T *ptr = (T *)((char *)pSrcDst + y * nSrcDstStep);
    int idx = x * nChannels;

#pragma unroll
    for (int c = 0; c < nChannels; c++) {
      ptr[idx + c] = abs_op(ptr[idx + c]);
    }
  }
}

// Special kernel for AC4 (skip alpha channel)
template <typename T>
__global__ void abs_ac4_kernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep, int width,
                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcPtr = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstPtr = (T *)((char *)pDst + y * nDstStep);

    int idx = x * 4;

    // Process only R, G, B channels (skip alpha)
    dstPtr[idx + 0] = abs_op(srcPtr[idx + 0]);
    dstPtr[idx + 1] = abs_op(srcPtr[idx + 1]);
    dstPtr[idx + 2] = abs_op(srcPtr[idx + 2]);
    dstPtr[idx + 3] = srcPtr[idx + 3]; // Copy alpha unchanged
  }
}

// Special kernel for AC4 in-place
template <typename T>
__global__ void abs_ac4_inplace_kernel(T *__restrict__ pSrcDst, int nSrcDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    T *ptr = (T *)((char *)pSrcDst + y * nSrcDstStep);
    int idx = x * 4;

    // Process only R, G, B channels (skip alpha)
    ptr[idx + 0] = abs_op(ptr[idx + 0]);
    ptr[idx + 1] = abs_op(ptr[idx + 1]);
    ptr[idx + 2] = abs_op(ptr[idx + 2]);
    // Alpha channel unchanged
  }
}

template <typename T, int nChannels>
static NppStatus launchAbsKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (vendor NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  abs_kernel<T, nChannels><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

template <typename T, int nChannels>
static NppStatus launchAbsInplaceKernel(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (vendor NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  abs_inplace_kernel<T, nChannels>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

template <typename T>
static NppStatus launchAbsAC4Kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (vendor NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  abs_ac4_kernel<T><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                      oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

template <typename T>
static NppStatus launchAbsAC4InplaceKernel(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI (vendor NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  abs_ac4_inplace_kernel<T>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

extern "C" {

NppStatus nppiAbs_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<Npp16s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<Npp16s, 1>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<Npp16s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C3IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<Npp16s, 3>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsAC4Kernel<Npp16s>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_AC4IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  return launchAbsAC4InplaceKernel<Npp16s>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<Npp16s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C4IR_Ctx_impl(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<Npp16s, 4>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C1R_Ctx_impl(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<half, 1>((const half *)pSrc, nSrcStep, (half *)pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C1IR_Ctx_impl(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<half, 1>((half *)pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C3R_Ctx_impl(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<half, 3>((const half *)pSrc, nSrcStep, (half *)pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C3IR_Ctx_impl(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<half, 3>((half *)pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C4R_Ctx_impl(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<half, 4>((const half *)pSrc, nSrcStep, (half *)pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16f_C4IR_Ctx_impl(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<half, 4>((half *)pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<Npp32f, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<Npp32f, 1>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<Npp32f, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C3IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<Npp32f, 3>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsAC4Kernel<Npp32f>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_AC4IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  return launchAbsAC4InplaceKernel<Npp32f>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchAbsKernel<Npp32f, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C4IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return launchAbsInplaceKernel<Npp32f, 4>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}
}
