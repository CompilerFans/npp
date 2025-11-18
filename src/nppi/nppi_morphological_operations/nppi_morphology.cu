#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Morphological operations kernels

// Device function for 3x3 erosion (minimum value in neighborhood)

template <typename T> __device__ inline T erode3x3(const T *pSrc, int nSrcStep, int x, int y, int width, int height) {
  T minVal = T(255); // Default maximum value for 8u

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
      T pixelValue = src_row[srcX];

      if (pixelValue < minVal) {
        minVal = pixelValue;
      }
    }
  }

  return minVal;
}

// Template specialization for Npp32f (float)
template <>

__device__ inline Npp32f erode3x3<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height) {
  Npp32f minVal = 1e10f; // Large value for float

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
      Npp32f pixelValue = src_row[srcX];

      if (pixelValue < minVal) {
        minVal = pixelValue;
      }
    }
  }

  return minVal;
}

// Device function for 3x3 dilation (maximum value in neighborhood)

template <typename T> __device__ inline T dilate3x3(const T *pSrc, int nSrcStep, int x, int y, int width, int height) {
  T maxVal = (T)0; // Start with minimum value

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
      T pixelValue = src_row[srcX];

      if (pixelValue > maxVal) {
        maxVal = pixelValue;
      }
    }
  }

  return maxVal;
}

// Template specialization for Npp32f (float) dilation
template <>

__device__ inline Npp32f dilate3x3<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height) {
  Npp32f maxVal = -1e10f; // Very small value for float

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
      Npp32f pixelValue = src_row[srcX];

      if (pixelValue > maxVal) {
        maxVal = pixelValue;
      }
    }
  }

  return maxVal;
}

//=============================================================================
// Erosion Kernels
//=============================================================================

// Kernel for 8-bit unsigned single channel 3x3 erosion
__global__ void nppiErode3x3_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = erode3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

// Kernel for 32-bit float single channel 3x3 erosion
__global__ void nppiErode3x3_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = erode3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

//=============================================================================
// Dilation Kernels
//=============================================================================

// Kernel for 8-bit unsigned single channel 3x3 dilation
__global__ void nppiDilate3x3_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = dilate3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

// Kernel for 32-bit float single channel 3x3 dilation
__global__ void nppiDilate3x3_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = dilate3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

//=============================================================================
// Host Functions
//=============================================================================

extern "C" {

// 8-bit unsigned single channel 3x3 erosion implementation
NppStatus nppiErode3x3_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode3x3_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                               oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel 3x3 erosion implementation
NppStatus nppiErode3x3_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode3x3_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel 3x3 dilation implementation
NppStatus nppiDilate3x3_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate3x3_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel 3x3 dilation implementation
NppStatus nppiDilate3x3_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate3x3_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                 oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}

//=============================================================================
// General Morphology Kernels with Arbitrary Structure Elements
//=============================================================================

// Generic erosion device function with arbitrary kernel
template <typename T>
__device__ inline T erodeGeneral(const T *pSrc, int nSrcStep, int x, int y, int width, int height, const Npp8u *pMask,
                                 int maskWidth, int maskHeight, int anchorX, int anchorY) {
  T minVal = T(255); // Default maximum value for integer types

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      // Check if mask element is active (non-zero)
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        // Handle boundaries by clamping
        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
        T pixelValue = src_row[srcX];

        if (pixelValue < minVal) {
          minVal = pixelValue;
        }
      }
    }
  }

  return minVal;
}

// Specialization for float
template <>
__device__ inline Npp32f erodeGeneral<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height,
                                              const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                              int anchorY) {
  Npp32f minVal = 1e10f; // Large value for float

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
        Npp32f pixelValue = src_row[srcX];

        if (pixelValue < minVal) {
          minVal = pixelValue;
        }
      }
    }
  }

  return minVal;
}

// Generic dilation device function with arbitrary kernel
template <typename T>
__device__ inline T dilateGeneral(const T *pSrc, int nSrcStep, int x, int y, int width, int height, const Npp8u *pMask,
                                  int maskWidth, int maskHeight, int anchorX, int anchorY) {
  T maxVal = T(0); // Default minimum value for integer types

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
        T pixelValue = src_row[srcX];

        if (pixelValue > maxVal) {
          maxVal = pixelValue;
        }
      }
    }
  }

  return maxVal;
}

// Specialization for float
template <>
__device__ inline Npp32f dilateGeneral<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height,
                                               const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                               int anchorY) {
  Npp32f maxVal = -1e10f; // Small value for float

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
        Npp32f pixelValue = src_row[srcX];

        if (pixelValue > maxVal) {
          maxVal = pixelValue;
        }
      }
    }
  }

  return maxVal;
}

// General erosion kernels
__global__ void nppiErode_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                        int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                        int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u result = erodeGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}
// Shared-memory tiled erosion kernels (8u)
__global__ void nppiErode_8u_C1R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                              int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                              int maskW, int maskH, int anchorX, int anchorY) {
  extern __shared__ unsigned char smem[];

  const int tileW_noPad = blockDim.x * 4 + maskW - 1;
  const int tileH = blockDim.y + maskH - 1;
  const int tilePitch = (tileW_noPad + 3) & ~3;

  unsigned char *tile = smem;
  unsigned char *mskSh = tile + tilePitch * tileH;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int tcount = blockDim.x * blockDim.y;

  // output pixel start
  const int outX0 = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  const int outY = blockIdx.y * blockDim.y + threadIdx.y;

  // The coordinates of the upper left corner of the corresponding tile in the original image(include anchor offset）
  const int tileOrgX = blockIdx.x * blockDim.x * 4 - anchorX;
  const int tileOrgY = blockIdx.y * blockDim.y - anchorY;

  // 1) load tile（clamp bound）
  const int tileElems = tilePitch * tileH;
  for (int idx = tid; idx < tileElems; idx += tcount) {
    int ly = idx / tilePitch;
    int lx = idx - ly * tilePitch;

    int sx = tileOrgX + lx;
    int sy = tileOrgY + ly;
    if (sx < 0)
      sx = 0;
    else if (sx >= width)
      sx = width - 1;
    if (sy < 0)
      sy = 0;
    else if (sy >= height)
      sy = height - 1;

    const Npp8u *srow = (const Npp8u *)((const char *)pSrc + sy * nSrcStep);
    tile[ly * tilePitch + lx] = srow[sx];
  }

  // 2) obtain the mask to the shared memory
  const int msize = maskW * maskH;
  for (int i = tid; i < msize; i += tcount)
    mskSh[i] = pMask[i];

  __syncthreads();

  // 3) calculation: each thread processes 4 adjacent output pixels
  if (outY < height) {
    const int baseTileX = threadIdx.x * 4;
    const int baseTileY = threadIdx.y;

    uchar4 minVal = make_uchar4(255, 255, 255, 255);

    for (int my = 0; my < maskH; ++my) {
      const unsigned char *mrow = mskSh + my * maskW;
      const unsigned char *trow = tile + (baseTileY + my) * tilePitch;
#pragma unroll 8
      for (int mx = 0; mx < maskW; ++mx) {
        if (mrow[mx]) {
          const unsigned char *p = trow + (baseTileX + mx);
          uchar4 pVec = make_uchar4(p[0], p[1], p[2], p[3]);
          minVal.x = (pVec.x < minVal.x) ? pVec.x : minVal.x;
          minVal.y = (pVec.y < minVal.y) ? pVec.y : minVal.y;
          minVal.z = (pVec.z < minVal.z) ? pVec.z : minVal.z;
          minVal.w = (pVec.w < minVal.w) ? pVec.w : minVal.w;
        }
      }
    }

    Npp8u *drow = (Npp8u *)((char *)pDst + outY * nDstStep);

    if (outX0 + 3 < width) {
      *((uchar4 *)(drow + outX0)) = minVal;
    } else {
      if (outX0 < width)
        drow[outX0] = minVal.x;
      if (outX0 + 1 < width)
        drow[outX0 + 1] = minVal.y;
      if (outX0 + 2 < width)
        drow[outX0 + 2] = minVal.z;
      if (outX0 + 3 < width)
        drow[outX0 + 3] = minVal.w;
    }
  }
}

__global__ void nppiErode_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                        int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                        int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      // Create temporary single-channel views
      const Npp8u *src_chan = pSrc + c;
      Npp8u *dst_chan = dst_row + x * 4 + c;

      Npp8u minVal = 255;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp8u *src_row = (const Npp8u *)((const char *)src_chan + srcY * nSrcStep);
            Npp8u pixelValue = src_row[srcX * 4];

            if (pixelValue < minVal) {
              minVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = minVal;
    }
  }
}
__global__ void nppiErode_8u_C4R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                              int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                              int maskWidth, int maskHeight, int anchorX, int anchorY) {
  extern __shared__ unsigned char shared[];

  constexpr int PIXELS_PER_THREAD_X = 4;
  const int blockSpanX = blockDim.x * PIXELS_PER_THREAD_X;

  // tile width covered the entire block pixel range
  const int tileWidth = blockSpanX + maskWidth - 1;
  const int tileHeight = blockDim.y + maskHeight - 1;

  uchar4 *tile = reinterpret_cast<uchar4 *>(shared);
  unsigned char *maskShared = reinterpret_cast<unsigned char *>(tile + tileWidth * tileHeight);

  const int tileOriginX = blockIdx.x * blockSpanX - anchorX;
  const int tileOriginY = blockIdx.y * blockDim.y - anchorY;

  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int threadsPerBlock = blockDim.x * blockDim.y;

  // load tile
  const int tileSize = tileWidth * tileHeight;
  for (int idx = linearThreadId; idx < tileSize; idx += threadsPerBlock) {
    int localY = idx / tileWidth;
    int localX = idx % tileWidth;

    int srcX = tileOriginX + localX;
    int srcY = tileOriginY + localY;

    srcX = max(0, min(srcX, width - 1));
    srcY = max(0, min(srcY, height - 1));

    const uchar4 *src_row = reinterpret_cast<const uchar4 *>((const char *)pSrc + srcY * nSrcStep);
    tile[idx] = src_row[srcX];
  }

  // load mask
  const int maskSize = maskWidth * maskHeight;
  for (int idx = linearThreadId; idx < maskSize; idx += threadsPerBlock) {
    maskShared[idx] = pMask[idx];
  }

  __syncthreads();

  const int outY = blockIdx.y * blockDim.y + threadIdx.y;
  const int outXBase = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD_X;
  if (outY >= height || outXBase >= width)
    return;

  uchar4 *dst_row = reinterpret_cast<uchar4 *>((char *)pDst + outY * nDstStep);

#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD_X; ++i) {
    int outX = outXBase + i;
    if (outX >= width)
      break;

    uchar4 minVal = make_uchar4(255, 255, 255, 255);

    for (int my = 0; my < maskHeight; ++my) {
      const unsigned char *maskRow = maskShared + my * maskWidth;
      const uchar4 *tileRow = tile + (threadIdx.y + my) * tileWidth;
#pragma unroll 4
      for (int mx = 0; mx < maskWidth; ++mx) {
        if (maskRow[mx]) {
          int tileX = threadIdx.x * PIXELS_PER_THREAD_X + i + mx;
          uchar4 pixelValue = tileRow[tileX];

          minVal.x = min(minVal.x, pixelValue.x);
          minVal.y = min(minVal.y, pixelValue.y);
          minVal.z = min(minVal.z, pixelValue.z);
          minVal.w = min(minVal.w, pixelValue.w);
        }
      }
    }

    dst_row[outX] = minVal;
  }
}

__global__ void nppiErode_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f result = erodeGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}

__global__ void nppiErode_32f_C4R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      const Npp32f *src_chan = pSrc + c;
      Npp32f *dst_chan = dst_row + x * 4 + c;

      Npp32f minVal = 1e10f;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp32f *src_row = (const Npp32f *)((const char *)src_chan + srcY * nSrcStep);
            Npp32f pixelValue = src_row[srcX * 4];

            if (pixelValue < minVal) {
              minVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = minVal;
    }
  }
}

__global__ void nppiErode_32f_C4R_kernel_opt(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                             int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                             int maskWidth, int maskHeight, int anchorX, int anchorY) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tile_x = blockIdx.x * blockDim.x;
  const int tile_y = blockIdx.y * blockDim.y;
  const int out_x = tile_x + tx;
  const int out_y = tile_y + ty;
  if (out_x >= width || out_y >= height)
    return;

  const int TILE_WIDTH = blockDim.x + maskWidth - 1;
  const int TILE_HEIGHT = blockDim.y + maskHeight - 1;
  const int maskSize = maskWidth * maskHeight;

  extern __shared__ unsigned char sdata[];

  float4 *tile = reinterpret_cast<float4 *>(sdata);
  Npp8u *sMask = reinterpret_cast<Npp8u *>(tile + TILE_WIDTH * TILE_HEIGHT);

  const int tid = ty * blockDim.x + tx;
  const int numThreads = blockDim.x * blockDim.y;

  // 1) mask -> shared
  for (int idx = tid; idx < maskSize; idx += numThreads) {
    sMask[idx] = pMask[idx];
  }
  __syncthreads();

  // 2) tile + halo -> shared
  for (int j = ty; j < TILE_HEIGHT; j += blockDim.y) {
    int gy = tile_y + j - anchorY;
    gy = max(0, min(gy, height - 1));
    const float4 *src_row = reinterpret_cast<const float4 *>((const char *)pSrc + gy * nSrcStep);

    for (int i = tx; i < TILE_WIDTH; i += blockDim.x) {
      int gx = tile_x + i - anchorX;
      gx = max(0, min(gx, width - 1));
      tile[j * TILE_WIDTH + i] = src_row[gx];
    }
  }
  __syncthreads();

  float4 minVal = make_float4(1e10f, 1e10f, 1e10f, 1e10f);

  for (int my = 0; my < maskHeight; ++my) {
    const int sy = ty + my;
    const int maskRow = my * maskWidth;

    for (int mx = 0; mx < maskWidth; ++mx) {
      if (!sMask[maskRow + mx])
        continue;

      const int sx = tx + mx;
      const float4 v = tile[sy * TILE_WIDTH + sx];

      minVal.x = fminf(minVal.x, v.x);
      minVal.y = fminf(minVal.y, v.y);
      minVal.z = fminf(minVal.z, v.z);
      minVal.w = fminf(minVal.w, v.w);
    }
  }

  float4 *dst_row = reinterpret_cast<float4 *>((char *)pDst + out_y * nDstStep);
  dst_row[out_x] = minVal;
}


__global__ void nppiDilate_8u_C1R_3x3_opt(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u maxVal = 0;
    for (int dy = -1; dy <= 1; dy++) {
      int srcY = min(max(y + dy, 0), height - 1);
      const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);

      for (int dx = -1; dx <= 1; dx++) {
        int srcX = min(max(x + dx, 0), width - 1);
        maxVal = max(maxVal, src_row[srcX]);
      }
    }
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = maxVal;
  }
}
// opt for masksize 3x3
__global__ void nppiDilate_8u_C4R_3x3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                             int height) {

  const int BLOCK_WIDTH = 32; // blockSize.x
  const int BLOCK_HEIGHT = 8; // blockSize.y
  const int SHARED_WIDTH = 34;
  const int SHARED_HEIGHT = 10;

  // fixed shared mem
  __shared__ Npp8u shared_mem[10][136]; // [blockSize.y + 2][(blockSize.x + 2)*4]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blockX = blockIdx.x * BLOCK_WIDTH;
  int blockY = blockIdx.y * BLOCK_HEIGHT;

  // load
  for (int pos = ty * blockDim.x + tx; pos < SHARED_HEIGHT * SHARED_WIDTH; pos += blockDim.x * blockDim.y) {
    int loadY = pos / SHARED_WIDTH;
    int loadX = pos % SHARED_WIDTH;

    int srcY = blockY + loadY - 1;
    int srcX = blockX + loadX - 1;

    if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
      const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
      const Npp8u *src_pixel = &src_row[srcX * 4];

      shared_mem[loadY][loadX * 4 + 0] = src_pixel[0];
      shared_mem[loadY][loadX * 4 + 1] = src_pixel[1];
      shared_mem[loadY][loadX * 4 + 2] = src_pixel[2];
      shared_mem[loadY][loadX * 4 + 3] = src_pixel[3];
    } else {
      shared_mem[loadY][loadX * 4 + 0] = 0;
      shared_mem[loadY][loadX * 4 + 1] = 0;
      shared_mem[loadY][loadX * 4 + 2] = 0;
      shared_mem[loadY][loadX * 4 + 3] = 0;
    }
  }

  __syncthreads();

  int x = blockX + tx;
  int y = blockY + ty;

  if (x < width && y < height) {
    int shared_y = ty + 1;
    int shared_x = tx + 1;
    int idx = shared_x * 4;

    Npp8u maxVal[4];

    // init as center pixel
    maxVal[0] = shared_mem[shared_y][idx + 0];
    maxVal[1] = shared_mem[shared_y][idx + 1];
    maxVal[2] = shared_mem[shared_y][idx + 2];
    maxVal[3] = shared_mem[shared_y][idx + 3];

    // 3x3 field compare
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0)
          continue;

        int neighbor_y = shared_y + dy;
        int neighbor_x = shared_x + dx;
        int neighbor_idx = neighbor_x * 4;

        maxVal[0] = max(maxVal[0], shared_mem[neighbor_y][neighbor_idx + 0]);
        maxVal[1] = max(maxVal[1], shared_mem[neighbor_y][neighbor_idx + 1]);
        maxVal[2] = max(maxVal[2], shared_mem[neighbor_y][neighbor_idx + 2]);
        maxVal[3] = max(maxVal[3], shared_mem[neighbor_y][neighbor_idx + 3]);
      }
    }

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    Npp8u *dst_pixel = &dst_row[x * 4];

    dst_pixel[0] = maxVal[0];
    dst_pixel[1] = maxVal[1];
    dst_pixel[2] = maxVal[2];
    dst_pixel[3] = maxVal[3];
  }
}

__global__ void nppiDilate_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u result = dilateGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}
__global__ void nppiDilate_8u_C1R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                               int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                               int maskW, int maskH, int anchorX, int anchorY) {
  extern __shared__ unsigned char smem[];

  const int tileW_noPad = blockDim.x * 4 + maskW - 1;
  const int tileH = blockDim.y + maskH - 1;
  const int tilePitch = (tileW_noPad + 3) & ~3;

  unsigned char *tile = smem;
  unsigned char *mskSh = tile + tilePitch * tileH;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int tcount = blockDim.x * blockDim.y;

  // output pixel start
  const int outX0 = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  const int outY = blockIdx.y * blockDim.y + threadIdx.y;

  // The coordinates of the upper left corner of the corresponding tile in the original image(include anchor offset）
  const int tileOrgX = blockIdx.x * blockDim.x * 4 - anchorX;
  const int tileOrgY = blockIdx.y * blockDim.y - anchorY;

  // 1) load tile（clamp bound）
  const int tileElems = tilePitch * tileH;
  for (int idx = tid; idx < tileElems; idx += tcount) {
    int ly = idx / tilePitch;
    int lx = idx - ly * tilePitch;

    int sx = tileOrgX + lx;
    int sy = tileOrgY + ly;
    if (sx < 0)
      sx = 0;
    else if (sx >= width)
      sx = width - 1;
    if (sy < 0)
      sy = 0;
    else if (sy >= height)
      sy = height - 1;

    const Npp8u *srow = (const Npp8u *)((const char *)pSrc + sy * nSrcStep);
    tile[ly * tilePitch + lx] = srow[sx];
  }

  // 2) obtain the mask to the shared memory
  const int msize = maskW * maskH;
  for (int i = tid; i < msize; i += tcount)
    mskSh[i] = pMask[i];

  __syncthreads();

  // 3) calculation: each thread processes 4 adjacent output pixels
  if (outY < height) {
    const int baseTileX = threadIdx.x * 4;
    const int baseTileY = threadIdx.y;

    uchar4 maxVal = make_uchar4(0, 0, 0, 0);

    for (int my = 0; my < maskH; ++my) {
      const unsigned char *mrow = mskSh + my * maskW;
      const unsigned char *trow = tile + (baseTileY + my) * tilePitch;
#pragma unroll 8
      for (int mx = 0; mx < maskW; ++mx) {
        if (mrow[mx]) {
          const unsigned char *p = trow + (baseTileX + mx);
          uchar4 pVec = make_uchar4(p[0], p[1], p[2], p[3]);
          maxVal.x = (pVec.x > maxVal.x) ? pVec.x : maxVal.x;
          maxVal.y = (pVec.y > maxVal.y) ? pVec.y : maxVal.y;
          maxVal.z = (pVec.z > maxVal.z) ? pVec.z : maxVal.z;
          maxVal.w = (pVec.w > maxVal.w) ? pVec.w : maxVal.w;
        }
      }
    }

    Npp8u *drow = (Npp8u *)((char *)pDst + outY * nDstStep);

    if (outX0 + 3 < width) {
      *((uchar4 *)(drow + outX0)) = maxVal;
    } else {
      if (outX0 < width)
        drow[outX0] = maxVal.x;
      if (outX0 + 1 < width)
        drow[outX0 + 1] = maxVal.y;
      if (outX0 + 2 < width)
        drow[outX0 + 2] = maxVal.z;
      if (outX0 + 3 < width)
        drow[outX0 + 3] = maxVal.w;
    }
  }
}

__global__ void nppiDilate_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      const Npp8u *src_chan = pSrc + c;
      Npp8u *dst_chan = dst_row + x * 4 + c;

      Npp8u maxVal = 0;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp8u *src_row = (const Npp8u *)((const char *)src_chan + srcY * nSrcStep);
            Npp8u pixelValue = src_row[srcX * 4];

            if (pixelValue > maxVal) {
              maxVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = maxVal;
    }
  }
}
__global__ void nppiDilate_8u_C4R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                               int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                               int maskWidth, int maskHeight, int anchorX, int anchorY) {
  extern __shared__ unsigned char shared[];

  constexpr int PIXELS_PER_THREAD_X = 4;
  const int blockSpanX = blockDim.x * PIXELS_PER_THREAD_X;

  // tile width covered the entire block pixel range
  const int tileWidth = blockSpanX + maskWidth - 1;
  const int tileHeight = blockDim.y + maskHeight - 1;

  uchar4 *tile = reinterpret_cast<uchar4 *>(shared);
  unsigned char *maskShared = reinterpret_cast<unsigned char *>(tile + tileWidth * tileHeight);

  const int tileOriginX = blockIdx.x * blockSpanX - anchorX;
  const int tileOriginY = blockIdx.y * blockDim.y - anchorY;

  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int threadsPerBlock = blockDim.x * blockDim.y;

  // load tile
  const int tileSize = tileWidth * tileHeight;
  for (int idx = linearThreadId; idx < tileSize; idx += threadsPerBlock) {
    int localY = idx / tileWidth;
    int localX = idx % tileWidth;

    int srcX = tileOriginX + localX;
    int srcY = tileOriginY + localY;

    srcX = max(0, min(srcX, width - 1));
    srcY = max(0, min(srcY, height - 1));

    const uchar4 *src_row = reinterpret_cast<const uchar4 *>((const char *)pSrc + srcY * nSrcStep);
    tile[idx] = src_row[srcX];
  }

  // load mask
  const int maskSize = maskWidth * maskHeight;
  for (int idx = linearThreadId; idx < maskSize; idx += threadsPerBlock) {
    maskShared[idx] = pMask[idx];
  }

  __syncthreads();

  const int outY = blockIdx.y * blockDim.y + threadIdx.y;
  const int outXBase = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD_X;
  if (outY >= height || outXBase >= width)
    return;

  uchar4 *dst_row = reinterpret_cast<uchar4 *>((char *)pDst + outY * nDstStep);

#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD_X; ++i) {
    int outX = outXBase + i;
    if (outX >= width)
      break;

    uchar4 minVal = make_uchar4(0, 0, 0, 0);

    for (int my = 0; my < maskHeight; ++my) {
      const unsigned char *maskRow = maskShared + my * maskWidth;
      const uchar4 *tileRow = tile + (threadIdx.y + my) * tileWidth;
#pragma unroll 4
      for (int mx = 0; mx < maskWidth; ++mx) {
        if (maskRow[mx]) {
          int tileX = threadIdx.x * PIXELS_PER_THREAD_X + i + mx;
          uchar4 pixelValue = tileRow[tileX];

          minVal.x = max(minVal.x, pixelValue.x);
          minVal.y = max(minVal.y, pixelValue.y);
          minVal.z = max(minVal.z, pixelValue.z);
          minVal.w = max(minVal.w, pixelValue.w);
        }
      }
    }

    dst_row[outX] = minVal;
  }
}

__global__ void nppiDilate_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                          int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                          int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f result = dilateGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}

__global__ void nppiDilate_32f_C4R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                          int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                          int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      const Npp32f *src_chan = pSrc + c;
      Npp32f *dst_chan = dst_row + x * 4 + c;

      Npp32f maxVal = -1e10f;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp32f *src_row = (const Npp32f *)((const char *)src_chan + srcY * nSrcStep);
            Npp32f pixelValue = src_row[srcX * 4];

            if (pixelValue > maxVal) {
              maxVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = maxVal;
    }
  }
}

__global__ void nppiDilate_32f_C4R_kernel_opt(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                              int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                              int maskWidth, int maskHeight, int anchorX, int anchorY) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tile_x = blockIdx.x * blockDim.x;
  const int tile_y = blockIdx.y * blockDim.y;
  const int out_x = tile_x + tx;
  const int out_y = tile_y + ty;
  if (out_x >= width || out_y >= height)
    return;

  const int TILE_WIDTH = blockDim.x + maskWidth - 1;
  const int TILE_HEIGHT = blockDim.y + maskHeight - 1;
  const int maskSize = maskWidth * maskHeight;

  extern __shared__ unsigned char sdata[];

  float4 *tile = reinterpret_cast<float4 *>(sdata);
  Npp8u *sMask = reinterpret_cast<Npp8u *>(tile + TILE_WIDTH * TILE_HEIGHT);

  const int tid = ty * blockDim.x + tx;
  const int numThreads = blockDim.x * blockDim.y;

  // mask -> shared
  for (int idx = tid; idx < maskSize; idx += numThreads) {
    sMask[idx] = pMask[idx];
  }
  __syncthreads();

  // tile + halo -> shared
  for (int j = ty; j < TILE_HEIGHT; j += blockDim.y) {
    int gy = tile_y + j - anchorY;
    gy = max(0, min(gy, height - 1));
    const float4 *src_row = reinterpret_cast<const float4 *>((const char *)pSrc + gy * nSrcStep);

    for (int i = tx; i < TILE_WIDTH; i += blockDim.x) {
      int gx = tile_x + i - anchorX;
      gx = max(0, min(gx, width - 1));
      tile[j * TILE_WIDTH + i] = src_row[gx];
    }
  }
  __syncthreads();

  float4 maxVal = make_float4(-1e10f, -1e10f, -1e10f, -1e10f);

  for (int my = 0; my < maskHeight; ++my) {
    const int sy = ty + my;
    const int maskRow = my * maskWidth;

    for (int mx = 0; mx < maskWidth; ++mx) {
      if (!sMask[maskRow + mx])
        continue;

      const int sx = tx + mx;
      const float4 v = tile[sy * TILE_WIDTH + sx];

      maxVal.x = fmaxf(maxVal.x, v.x);
      maxVal.y = fmaxf(maxVal.y, v.y);
      maxVal.z = fmaxf(maxVal.z, v.z);
      maxVal.w = fmaxf(maxVal.w, v.w);
    }
  }

  float4 *dst_row = reinterpret_cast<float4 *>((char *)pDst + out_y * nDstStep);
  dst_row[out_x] = maxVal;
}

//=============================================================================
// Host functions for general morphology operations
//=============================================================================

extern "C" {

NppStatus nppiErode_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {

  dim3 blockSize(32, 32);
  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW_noPad = blockSize.x * 4 + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;
  const int tilePitch = tileW_noPad + 1; // padding
  const size_t tileSharedElements = size_t(tilePitch) * tileH;
  const size_t maskSharedElements = size_t(oMaskSize.width) * oMaskSize.height;
  const size_t sharedBytes = (tileSharedElements + maskSharedElements) * sizeof(Npp8u);

  nppiErode_8u_C1R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  const dim3 blockSize(16, 16);
  const dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileWidth = blockSize.x * 4 + oMaskSize.width - 1;
  const int tileHeight = blockSize.y + oMaskSize.height - 1;
  const size_t tileSharedElements = static_cast<size_t>(tileWidth) * tileHeight;
  const size_t maskSharedElements = static_cast<size_t>(oMaskSize.width) * oMaskSize.height;
  const size_t sharedBytes = tileSharedElements * sizeof(uchar4) + maskSharedElements * sizeof(Npp8u);

  nppiErode_8u_C4R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW = blockSize.x + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;

  size_t sharedSize = static_cast<size_t>(tileW) * tileH * sizeof(float4) +
                      static_cast<size_t>(oMaskSize.width) * oMaskSize.height * sizeof(Npp8u);

  nppiErode_32f_C4R_kernel_opt<<<gridSize, blockSize, sharedSize, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {

  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    nppiDilate_8u_C1R_3x3_opt<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                oSizeROI.width, oSizeROI.height);
  } else {
    dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    const int tileW_noPad = blockSize.x * 4 + oMaskSize.width - 1;
    const int tileH = blockSize.y + oMaskSize.height - 1;
    const int tilePitch = tileW_noPad + 1; // padding
    const size_t tileSharedElements = size_t(tilePitch) * tileH;
    const size_t maskSharedElements = size_t(oMaskSize.width) * oMaskSize.height;
    const size_t sharedBytes = (tileSharedElements + maskSharedElements) * sizeof(Npp8u);

    nppiDilate_8u_C1R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
        oAnchor.x, oAnchor.y);
  }
  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y); // special for 3x3
  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    nppiDilate_8u_C4R_3x3_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);
  } else {
    dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    const int tileWidth = blockSize.x * 4 + oMaskSize.width - 1;
    const int tileHeight = blockSize.y + oMaskSize.height - 1;
    const size_t tileSharedElements = static_cast<size_t>(tileWidth) * tileHeight;
    const size_t maskSharedElements = static_cast<size_t>(oMaskSize.width) * oMaskSize.height;
    const size_t sharedBytes = tileSharedElements * sizeof(uchar4) + maskSharedElements * sizeof(Npp8u);

    nppiDilate_8u_C4R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
        oAnchor.x, oAnchor.y);
  }
  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW = blockSize.x + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;

  size_t sharedSize = static_cast<size_t>(tileW) * tileH * sizeof(float4) +
                      static_cast<size_t>(oMaskSize.width) * oMaskSize.height * sizeof(Npp8u);

  nppiDilate_32f_C4R_kernel_opt<<<gridSize, blockSize, sharedSize, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}
}
