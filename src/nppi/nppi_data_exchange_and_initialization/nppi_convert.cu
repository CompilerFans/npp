#include "npp.h"
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kConvert32f8uElemsPerThread = 16;
constexpr int kConvertBlockX = 256;
__global__ void convert_8u32f_C1R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);

  // Convert 8-bit unsigned to 32-bit float (0-255 -> 0.0-255.0)
  dstRow[x] = (Npp32f)srcRow[x];
}

__global__ void convert_8u32f_C3R_kernel_vec4(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                              int nDstStep, int vec4PerRow, int height) {
  const int vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (y >= height || vecIdx >= vec4PerRow) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);
  const uchar4 s = reinterpret_cast<const uchar4 *>(srcRow)[vecIdx];
  const int base = vecIdx * 4;
  dstRow[base + 0] = (Npp32f)s.x;
  dstRow[base + 1] = (Npp32f)s.y;
  dstRow[base + 2] = (Npp32f)s.z;
  dstRow[base + 3] = (Npp32f)s.w;
}

__global__ void convert_8u32f_C3R_kernel_tail(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                              int nDstStep, int startElem, int tailElems, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (y >= height || x >= tailElems) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);
  dstRow[startElem + x] = (Npp32f)srcRow[startElem + x];
}

__global__ void convert_8u32f_C3R_kernel_scalar(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                                int nDstStep, int rowElems, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (y >= height || x >= rowElems) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);
  dstRow[x] = (Npp32f)srcRow[x];
}

__global__ void convert_8u32f_C4R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
  Npp32f *dstRow = reinterpret_cast<Npp32f *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  const int offset = x * 4;
  dstRow[offset] = static_cast<Npp32f>(srcRow[offset]);
  dstRow[offset + 1] = static_cast<Npp32f>(srcRow[offset + 1]);
  dstRow[offset + 2] = static_cast<Npp32f>(srcRow[offset + 2]);
  dstRow[offset + 3] = static_cast<Npp32f>(srcRow[offset + 3]);
}

__global__ void convert_8u16u_C1R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp16u *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp16u *dstRow = (Npp16u *)(((char *)pDst) + y * nDstStep);

  dstRow[x] = (Npp16u)srcRow[x];
}

template <bool FinancialMode>
__device__ __forceinline__ Npp8u round_and_clamp_32f_to_u8(Npp32f value) {
  int rounded;
  if constexpr (FinancialMode) {
    rounded = static_cast<int>(roundf(value));
  } else {
    rounded = __float2int_rn(value);
  }
  rounded = max(0, min(255, rounded));
  return static_cast<Npp8u>(rounded);
}

__device__ __forceinline__ Npp8u round_and_clamp_32f_to_u8(Npp32f value, NppRoundMode eRoundMode) {
  if (eRoundMode == NPP_RND_FINANCIAL) {
    return round_and_clamp_32f_to_u8<true>(value);
  }
  return round_and_clamp_32f_to_u8<false>(value);
}

__global__ void convert_32f8u_C1R_kernel(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                         int nDstStep, int width, int height, NppRoundMode eRoundMode) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32f *srcRow = (const Npp32f *)(((const char *)pSrc) + y * nSrcStep);
  Npp8u *dstRow = (Npp8u *)(((char *)pDst) + y * nDstStep);

  dstRow[x] = round_and_clamp_32f_to_u8(srcRow[x], eRoundMode);
}

template <bool FinancialMode>
__device__ __forceinline__ uchar4 round_and_clamp_32f4_to_u8(float4 value) {
  return make_uchar4(round_and_clamp_32f_to_u8<FinancialMode>(value.x), round_and_clamp_32f_to_u8<FinancialMode>(value.y),
                     round_and_clamp_32f_to_u8<FinancialMode>(value.z), round_and_clamp_32f_to_u8<FinancialMode>(value.w));
}

template <bool FinancialMode>
__global__ void convert_32f8u_C3R_kernel_opt(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                             int nDstStep, int rowElems, int height) {
  int elemBase = (blockIdx.x * blockDim.x + threadIdx.x) * kConvert32f8uElemsPerThread;
  int y = blockIdx.y;
  if (y >= height || elemBase >= rowElems) {
    return;
  }

  const Npp32f *srcRow = (const Npp32f *)(((const char *)pSrc) + y * nSrcStep);
  Npp8u *dstRow = (Npp8u *)(((char *)pDst) + y * nDstStep);
  const int remain = rowElems - elemBase;
  const Npp32f *srcPtr = srcRow + elemBase;
  Npp8u *dstPtr = dstRow + elemBase;
  const bool alignedSrc = ((reinterpret_cast<uintptr_t>(srcPtr) & 0xFu) == 0u);
  const bool alignedDst = ((reinterpret_cast<uintptr_t>(dstPtr) & 0x3u) == 0u);
  if (remain >= kConvert32f8uElemsPerThread && alignedSrc && alignedDst) {
    const float4 *src4 = reinterpret_cast<const float4 *>(srcPtr);
    uchar4 *dst4 = reinterpret_cast<uchar4 *>(dstPtr);
    dst4[0] = round_and_clamp_32f4_to_u8<FinancialMode>(src4[0]);
    dst4[1] = round_and_clamp_32f4_to_u8<FinancialMode>(src4[1]);
    dst4[2] = round_and_clamp_32f4_to_u8<FinancialMode>(src4[2]);
    dst4[3] = round_and_clamp_32f4_to_u8<FinancialMode>(src4[3]);
    return;
  }

  const int tail = min(kConvert32f8uElemsPerThread, remain);
#pragma unroll
  for (int i = 0; i < kConvert32f8uElemsPerThread; ++i) {
    if (i >= tail) {
      break;
    }
    dstPtr[i] = round_and_clamp_32f_to_u8<FinancialMode>(srcPtr[i]);
  }
}

__global__ void convert_16u32f_C1R_kernel(const Npp16u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                          int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp16u *srcRow = (const Npp16u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);

  dstRow[x] = (Npp32f)srcRow[x];
}

extern "C" {
NppStatus nppiConvert_8u32f_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  convert_8u32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_8u32f_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {

  // Parameter validation
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (nDstStep <= 0)
    return NPP_STEP_ERROR;
  if (nSrcStep <= 0)
    return NPP_NO_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;

  const int rowElems = oSizeROI.width * 3;
  dim3 blockSize(kConvertBlockX, 1);
  const int vecElems = rowElems & ~3;
  const int vec4PerRow = vecElems >> 2;
  const int tailElems = rowElems - vecElems;
  const bool canUseVec4 = (vec4PerRow > 0) && ((reinterpret_cast<uintptr_t>(pSrc) & 0x3u) == 0u) && ((nSrcStep & 0x3) == 0);

  if (canUseVec4) {
    dim3 gridVec((vec4PerRow + blockSize.x - 1) / blockSize.x, oSizeROI.height);
    convert_8u32f_C3R_kernel_vec4<<<gridVec, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, vec4PerRow, oSizeROI.height);

    if (tailElems > 0) {
      dim3 gridTail((tailElems + blockSize.x - 1) / blockSize.x, oSizeROI.height);
      convert_8u32f_C3R_kernel_tail<<<gridTail, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrc, nSrcStep, pDst, nDstStep, vecElems, tailElems, oSizeROI.height);
    }
  } else {
    dim3 gridScalar((rowElems + blockSize.x - 1) / blockSize.x, oSizeROI.height);
    convert_8u32f_C3R_kernel_scalar<<<gridScalar, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, rowElems, oSizeROI.height);
  }

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_8u32f_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  convert_8u32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_8u16u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  convert_8u16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_32f8u_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppRoundMode eRoundMode, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  convert_32f8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eRoundMode);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_32f8u_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppRoundMode eRoundMode, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (nDstStep <= 0)
    return NPP_STEP_ERROR;
  if (nSrcStep <= 0)
    return NPP_NO_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;

  const int rowElems = oSizeROI.width * 3;
  dim3 blockSize(kConvertBlockX, 1);
  dim3 gridSize((rowElems + blockSize.x * kConvert32f8uElemsPerThread - 1) /
                    (blockSize.x * kConvert32f8uElemsPerThread),
                oSizeROI.height);

  if (eRoundMode == NPP_RND_FINANCIAL) {
    convert_32f8u_C3R_kernel_opt<true><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, rowElems, oSizeROI.height);
  } else {
    convert_32f8u_C3R_kernel_opt<false><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, rowElems, oSizeROI.height);
  }

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_16u32f_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  convert_16u32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                              oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
}
