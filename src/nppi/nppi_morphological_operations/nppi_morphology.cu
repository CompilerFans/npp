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

} // end extern "C"

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

// General dilation kernels
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

//=============================================================================
// Host functions for general morphology operations
//=============================================================================

extern "C" {

NppStatus nppiErode_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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

  nppiErode_32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

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

  nppiDilate_32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}
}