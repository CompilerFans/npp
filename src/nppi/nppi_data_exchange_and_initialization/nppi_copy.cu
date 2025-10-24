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

// Packed to planar copy kernel (C3P3R, C4P4R)
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

// Planar to packed copy kernel (P3C3R, P4C4R)
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

// Generic packed to planar implementation (outside extern "C")
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

// Generic planar to packed implementation
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
  return nppiCopy_CxPxR_impl<Npp8u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C3P3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C3P3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C3P3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3P3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32f, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
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
  return nppiCopy_PxCxR_impl<Npp8u, 3>(const_cast<Npp8u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_P3C3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16u, 3>(const_cast<Npp16u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_P3C3R_Ctx_impl(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16s, 3>(const_cast<Npp16s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_P3C3R_Ctx_impl(const Npp32s *const pSrc[3], int nSrcStep, Npp32s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32s, 3>(const_cast<Npp32s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_P3C3R_Ctx_impl(const Npp32f *const pSrc[3], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32f, 3>(const_cast<Npp32f **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
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
