#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Generic multi-channel copy kernel template
template <typename T, int CHANNELS>
__global__ void nppiCopy_kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    dst_row[x * CHANNELS + c] = src_row[x * CHANNELS + c];
  }
}

// Optimized vectorized kernel for 8u_C4
__global__ void nppiCopy_8u_C4R_vectorized_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                  int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const uint32_t *src_row = (const uint32_t *)((const char *)pSrc + y * nSrcStep);
  uint32_t *dst_row = (uint32_t *)((char *)pDst + y * nDstStep);

  dst_row[x] = src_row[x];
}

// Optimized packed to planar copy kernel
template <typename T, int CHANNELS>
__global__ void nppiCopy_CxPxR_kernel_optimized(const T *pSrc, int nSrcStep, T *pDst0, T *pDst1, T *pDst2,
                                                int nDstStep, int width, int height) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  T *dst_row0 = (T *)((char *)pDst0 + y * nDstStep);
  T *dst_row1 = (T *)((char *)pDst1 + y * nDstStep);
  T *dst_row2 = (T *)((char *)pDst2 + y * nDstStep);

  if (x + 3 < width) {
    // Process 4 pixels at once
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = x + i;
      dst_row0[idx] = src_row[idx * CHANNELS];
      dst_row1[idx] = src_row[idx * CHANNELS + 1];
      if (CHANNELS >= 3) dst_row2[idx] = src_row[idx * CHANNELS + 2];
    }
  } else {
    // Handle remaining pixels
    for (int i = 0; i < 4 && x + i < width; i++) {
      int idx = x + i;
      dst_row0[idx] = src_row[idx * CHANNELS];
      dst_row1[idx] = src_row[idx * CHANNELS + 1];
      if (CHANNELS >= 3) dst_row2[idx] = src_row[idx * CHANNELS + 2];
    }
  }
}

// Legacy kernel for C4P4R
template <typename T, int CHANNELS>
__global__ void nppiCopy_CxPxR_kernel(const T *pSrc, int nSrcStep, T *const *pDst, int nDstStep, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    T *dst_row = (T *)((char *)pDst[c] + y * nDstStep);
    dst_row[x] = src_row[x * CHANNELS + c];
  }
}

// Optimized planar to packed copy kernel
template <typename T, int CHANNELS>
__global__ void nppiCopy_PxCxR_kernel_optimized(const T *pSrc0, const T *pSrc1, const T *pSrc2,
                                                int nSrcStep, T *pDst, int nDstStep, int width, int height) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= height)
    return;

  const T *src_row0 = (const T *)((const char *)pSrc0 + y * nSrcStep);
  const T *src_row1 = (const T *)((const char *)pSrc1 + y * nSrcStep);
  const T *src_row2 = (const T *)((const char *)pSrc2 + y * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

  if (x + 3 < width) {
    // Process 4 pixels at once
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = x + i;
      dst_row[idx * CHANNELS] = src_row0[idx];
      dst_row[idx * CHANNELS + 1] = src_row1[idx];
      if (CHANNELS >= 3) dst_row[idx * CHANNELS + 2] = src_row2[idx];
    }
  } else {
    // Handle remaining pixels
    for (int i = 0; i < 4 && x + i < width; i++) {
      int idx = x + i;
      dst_row[idx * CHANNELS] = src_row0[idx];
      dst_row[idx * CHANNELS + 1] = src_row1[idx];
      if (CHANNELS >= 3) dst_row[idx * CHANNELS + 2] = src_row2[idx];
    }
  }
}

// Legacy kernel for P4C4R
template <typename T, int CHANNELS>
__global__ void nppiCopy_PxCxR_kernel(T *const *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  T *dst_row = (T *)((char *)pDst + y * nDstStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    const T *src_row = (const T *)((const char *)pSrc[c] + y * nSrcStep);
    dst_row[x * CHANNELS + c] = src_row[x];
  }
}

// Generic copy implementation template (outside extern "C")
template <typename T, int CHANNELS>
NppStatus nppiCopy_impl(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                 oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

extern "C" {

// Explicit instantiations for 8u
NppStatus nppiCopy_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp8u, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp8u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  if (oSizeROI.width % 4 == 0 && nSrcStep % 4 == 0 && nDstStep % 4 == 0) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    nppiCopy_8u_C4R_vectorized_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else {
    return nppiCopy_impl<Npp8u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  return NPP_SUCCESS;
}

// Explicit instantiations for 16u
NppStatus nppiCopy_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16u, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 16s
NppStatus nppiCopy_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 32s
NppStatus nppiCopy_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 32f
NppStatus nppiCopy_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32f, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32f, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32f, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

} // extern "C"

// Optimized C3P3R implementation
template <typename T>
NppStatus nppiCopy_C3P3R_impl_optimized(const T *pSrc, int nSrcStep, T *const pDst[3], int nDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_CxPxR_kernel_optimized<T, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Generic packed to planar implementation (for C4P4R)
template <typename T, int CHANNELS>
NppStatus nppiCopy_CxPxR_impl(const T *pSrc, int nSrcStep, T *const pDst[], int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  T **d_pDst;

  cudaError_t cudaStatus = cudaMalloc(&d_pDst, CHANNELS * sizeof(T *));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpy(d_pDst, pDst, CHANNELS * sizeof(T *), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_pDst);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_CxPxR_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, d_pDst, nDstStep,
                                                                                       oSizeROI.width, oSizeROI.height);

  cudaStatus = cudaGetLastError();
  cudaFree(d_pDst);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Optimized P3C3R implementation
template <typename T>
NppStatus nppiCopy_P3C3R_impl_optimized(T *const pSrc[3], int nSrcStep, T *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_PxCxR_kernel_optimized<T, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Generic planar to packed implementation (for P4C4R)
template <typename T, int CHANNELS>
NppStatus nppiCopy_PxCxR_impl(T *const pSrc[], int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  T **d_pSrc;

  cudaError_t cudaStatus = cudaMalloc(&d_pSrc, CHANNELS * sizeof(T *));
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaStatus = cudaMemcpy(d_pSrc, pSrc, CHANNELS * sizeof(T *), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_pSrc);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopy_PxCxR_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(d_pSrc, nSrcStep, pDst, nDstStep,
                                                                                       oSizeROI.width, oSizeROI.height);

  cudaStatus = cudaGetLastError();
  cudaFree(d_pSrc);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

extern "C" {

// C3P3R implementations for all data types
NppStatus nppiCopy_8u_C3P3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[3], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_C3P3R_impl_optimized<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C3P3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_C3P3R_impl_optimized<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C3P3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_C3P3R_impl_optimized<Npp16s>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C3P3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_C3P3R_impl_optimized<Npp32s>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3P3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_C3P3R_impl_optimized<Npp32f>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// C4P4R implementations for all data types
NppStatus nppiCopy_8u_C4P4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp8u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C4P4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C4P4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32f, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// P3C3R implementations for all data types
NppStatus nppiCopy_8u_P3C3R_Ctx_impl(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_P3C3R_impl_optimized<Npp8u>(const_cast<Npp8u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_P3C3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_P3C3R_impl_optimized<Npp16u>(const_cast<Npp16u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_P3C3R_Ctx_impl(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_P3C3R_impl_optimized<Npp16s>(const_cast<Npp16s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_P3C3R_Ctx_impl(const Npp32s *const pSrc[3], int nSrcStep, Npp32s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_P3C3R_impl_optimized<Npp32s>(const_cast<Npp32s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_P3C3R_Ctx_impl(const Npp32f *const pSrc[3], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_P3C3R_impl_optimized<Npp32f>(const_cast<Npp32f **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// P4C4R implementations for all data types
NppStatus nppiCopy_8u_P4C4R_Ctx_impl(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp8u, 4>(const_cast<Npp8u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_P4C4R_Ctx_impl(const Npp16u *const pSrc[4], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16u, 4>(const_cast<Npp16u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_P4C4R_Ctx_impl(const Npp32f *const pSrc[4], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32f, 4>(const_cast<Npp32f **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
}
