#include "npp.h"
#include <cuda_runtime.h>


/**
 * CUDA kernels for NPP Image Transpose operations
 * Optimized implementation using shared memory tiling
 */

#define TILE_SIZE 32 // Tile size for shared memory optimization
#define TILE_ROWS 8  // Block rows; each thread moves TILE_SIZE / TILE_ROWS rows

/**
 * Template CUDA kernel for transposing single-channel images
 * Uses shared memory tiling for optimal memory access patterns.  The block is
 * TILE_SIZE x TILE_ROWS (256 threads for 32x8): a 32-pixel row keeps the warp
 * accesses at full 32-byte sector utilisation while staying below the MACA
 * large-block recompile threshold.
 */
template <typename T>
__global__ void transpose_kernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep,
                                 int width, int height) {
  __shared__ T tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

  const int x0 = blockIdx.x * TILE_SIZE;
  const int y0 = blockIdx.y * TILE_SIZE;

  // Load data into shared memory
#pragma unroll
  for (int j = 0; j < TILE_SIZE / TILE_ROWS; ++j) {
    const int ty = threadIdx.y + j * TILE_ROWS;
    const int x = x0 + threadIdx.x;
    const int y = y0 + ty;
    if (x < width && y < height) {
      const T *srcPtr = (const T *)((const char *)pSrc + (size_t)y * nSrcStep);
      tile[ty][threadIdx.x] = srcPtr[x];
    }
  }

  __syncthreads();

  // Transpose coordinates
  const int x1 = blockIdx.y * TILE_SIZE;
  const int y1 = blockIdx.x * TILE_SIZE;

  // Write transposed data
#pragma unroll
  for (int j = 0; j < TILE_SIZE / TILE_ROWS; ++j) {
    const int ty = threadIdx.y + j * TILE_ROWS;
    const int x = x1 + threadIdx.x;
    const int y = y1 + ty;
    if (x < height && y < width) {
      T *dstPtr = (T *)((char *)pDst + (size_t)y * nDstStep);
      dstPtr[x] = tile[threadIdx.x][ty];
    }
  }
}

/**
 * Template CUDA kernel for transposing multi-channel images
 */
template <typename T, int channels>
__global__ void transpose_multichannel_kernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst,
                                              int nDstStep, int width, int height) {
  __shared__ T tile[TILE_SIZE][TILE_SIZE * channels + channels]; // Padding to avoid bank conflicts

  const int x0 = blockIdx.x * TILE_SIZE;
  const int y0 = blockIdx.y * TILE_SIZE;

  // Load data into shared memory
#pragma unroll
  for (int j = 0; j < TILE_SIZE / TILE_ROWS; ++j) {
    const int ty = threadIdx.y + j * TILE_ROWS;
    const int x = x0 + threadIdx.x;
    const int y = y0 + ty;
    if (x < width && y < height) {
      const T *srcPtr = (const T *)((const char *)pSrc + (size_t)y * nSrcStep);
      for (int c = 0; c < channels; c++) {
        tile[ty][threadIdx.x * channels + c] = srcPtr[x * channels + c];
      }
    }
  }

  __syncthreads();

  // Transpose coordinates
  const int x1 = blockIdx.y * TILE_SIZE;
  const int y1 = blockIdx.x * TILE_SIZE;

  // Write transposed data
#pragma unroll
  for (int j = 0; j < TILE_SIZE / TILE_ROWS; ++j) {
    const int ty = threadIdx.y + j * TILE_ROWS;
    const int x = x1 + threadIdx.x;
    const int y = y1 + ty;
    if (x < height && y < width) {
      T *dstPtr = (T *)((char *)pDst + (size_t)y * nDstStep);
      for (int c = 0; c < channels; c++) {
        dstPtr[x * channels + c] = tile[threadIdx.x][ty * channels + c];
      }
    }
  }
}

extern "C" {

// ============================================================================
// 8-bit unsigned implementations
// ============================================================================

NppStatus nppiTranspose_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_kernel<Npp8u>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp8u, 3>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSrcROI,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp8u, 4>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

// ============================================================================
// 16-bit unsigned implementations
// ============================================================================

NppStatus nppiTranspose_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_kernel<Npp16u>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp16u, 3>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp16u, 4>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

// ============================================================================
// 16-bit signed implementations
// ============================================================================

NppStatus nppiTranspose_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_kernel<Npp16s>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp16s, 3>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp16s, 4>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

// ============================================================================
// 32-bit signed implementations
// ============================================================================

NppStatus nppiTranspose_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_kernel<Npp32s>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp32s, 3>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp32s, 4>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

// ============================================================================
// 32-bit float implementations
// ============================================================================

NppStatus nppiTranspose_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_kernel<Npp32f>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp32f, 3>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiTranspose_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSrcROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(TILE_SIZE, TILE_ROWS);
  dim3 gridSize((oSrcROI.width + TILE_SIZE - 1) / TILE_SIZE, (oSrcROI.height + TILE_SIZE - 1) / TILE_SIZE);

  transpose_multichannel_kernel<Npp32f, 4>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oSrcROI.width, oSrcROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

} // extern "C"