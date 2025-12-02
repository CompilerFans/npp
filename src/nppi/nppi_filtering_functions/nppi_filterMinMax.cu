// nppiFilterMax and nppiFilterMin CUDA kernel implementations
// FilterMax: morphological dilation (local maximum)
// FilterMin: morphological erosion (local minimum)

#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

// Enable zero padding for boundary handling
#ifndef MPP_FILTER_MINMAX_ZERO_PADDING
#define MPP_FILTER_MINMAX_ZERO_PADDING
#endif

// ============================================================================
// FilterMax kernel implementations
// ============================================================================

// FilterMax kernel for 8u_C1R
__global__ void nppiFilterMax_8u_C1R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp8u maxVal = 0;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          maxVal = max(maxVal, pSrcRow[i]);
        }
#else
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        maxVal = max(maxVal, pSrcRow[i]);
#endif
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = maxVal;
  }
}

// FilterMax kernel for 8u_C3R
__global__ void nppiFilterMax_8u_C3R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp8u maxVal[3] = {0, 0, 0};

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 3; c++) {
            maxVal[c] = max(maxVal[c], pSrcRow[i * 3 + c]);
          }
        }
#else
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 3; c++) {
          maxVal[c] = max(maxVal[c], pSrcRow[i * 3 + c]);
        }
#endif
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 3; c++) {
      pDstRow[x * 3 + c] = maxVal[c];
    }
  }
}

// FilterMax kernel for 8u_C4R
__global__ void nppiFilterMax_8u_C4R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp8u maxVal[4] = {0, 0, 0, 0};

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 4; c++) {
            maxVal[c] = max(maxVal[c], pSrcRow[i * 4 + c]);
          }
        }
#else
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 4; c++) {
          maxVal[c] = max(maxVal[c], pSrcRow[i * 4 + c]);
        }
#endif
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 4; c++) {
      pDstRow[x * 4 + c] = maxVal[c];
    }
  }
}

// FilterMax kernel for 16u_C1R
__global__ void nppiFilterMax_16u_C1R_kernel_impl(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp16u maxVal = 0;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp16u *pSrcRow = (const Npp16u *)((const char *)pSrc + j * nSrcStep);
          maxVal = max(maxVal, pSrcRow[i]);
        }
#else
        const Npp16u *pSrcRow = (const Npp16u *)((const char *)pSrc + j * nSrcStep);
        maxVal = max(maxVal, pSrcRow[i]);
#endif
      }
    }

    Npp16u *pDstRow = (Npp16u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = maxVal;
  }
}

// FilterMax kernel for 32f_C1R
__global__ void nppiFilterMax_32f_C1R_kernel_impl(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp32f maxVal = -3.402823466e+38f; // -FLT_MAX

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          maxVal = fmaxf(maxVal, pSrcRow[i]);
        }
#else
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
        maxVal = fmaxf(maxVal, pSrcRow[i]);
#endif
      }
    }

    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    pDstRow[x] = maxVal;
  }
}

// FilterMax kernel for 32f_C3R
__global__ void nppiFilterMax_32f_C3R_kernel_impl(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp32f maxVal[3] = {-3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f};

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 3; c++) {
            maxVal[c] = fmaxf(maxVal[c], pSrcRow[i * 3 + c]);
          }
        }
#else
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 3; c++) {
          maxVal[c] = fmaxf(maxVal[c], pSrcRow[i * 3 + c]);
        }
#endif
      }
    }

    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 3; c++) {
      pDstRow[x * 3 + c] = maxVal[c];
    }
  }
}

// ============================================================================
// FilterMin kernel implementations
// ============================================================================

// FilterMin kernel for 8u_C1R
__global__ void nppiFilterMin_8u_C1R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp8u minVal = 255;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          minVal = min(minVal, pSrcRow[i]);
        }
#else
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        minVal = min(minVal, pSrcRow[i]);
#endif
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = minVal;
  }
}

// FilterMin kernel for 8u_C3R
__global__ void nppiFilterMin_8u_C3R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp8u minVal[3] = {255, 255, 255};

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 3; c++) {
            minVal[c] = min(minVal[c], pSrcRow[i * 3 + c]);
          }
        }
#else
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 3; c++) {
          minVal[c] = min(minVal[c], pSrcRow[i * 3 + c]);
        }
#endif
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 3; c++) {
      pDstRow[x * 3 + c] = minVal[c];
    }
  }
}

// FilterMin kernel for 8u_C4R
__global__ void nppiFilterMin_8u_C4R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp8u minVal[4] = {255, 255, 255, 255};

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 4; c++) {
            minVal[c] = min(minVal[c], pSrcRow[i * 4 + c]);
          }
        }
#else
        const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 4; c++) {
          minVal[c] = min(minVal[c], pSrcRow[i * 4 + c]);
        }
#endif
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 4; c++) {
      pDstRow[x * 4 + c] = minVal[c];
    }
  }
}

// FilterMin kernel for 16u_C1R
__global__ void nppiFilterMin_16u_C1R_kernel_impl(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp16u minVal = 65535;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp16u *pSrcRow = (const Npp16u *)((const char *)pSrc + j * nSrcStep);
          minVal = min(minVal, pSrcRow[i]);
        }
#else
        const Npp16u *pSrcRow = (const Npp16u *)((const char *)pSrc + j * nSrcStep);
        minVal = min(minVal, pSrcRow[i]);
#endif
      }
    }

    Npp16u *pDstRow = (Npp16u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = minVal;
  }
}

// FilterMin kernel for 32f_C1R
__global__ void nppiFilterMin_32f_C1R_kernel_impl(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp32f minVal = 3.402823466e+38f; // FLT_MAX

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          minVal = fminf(minVal, pSrcRow[i]);
        }
#else
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
        minVal = fminf(minVal, pSrcRow[i]);
#endif
      }
    }

    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    pDstRow[x] = minVal;
  }
}

// FilterMin kernel for 32f_C3R
__global__ void nppiFilterMin_32f_C3R_kernel_impl(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    Npp32f minVal[3] = {3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f};

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
#ifdef MPP_FILTER_MINMAX_ZERO_PADDING
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 3; c++) {
            minVal[c] = fminf(minVal[c], pSrcRow[i * 3 + c]);
          }
        }
#else
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
        for (int c = 0; c < 3; c++) {
          minVal[c] = fminf(minVal[c], pSrcRow[i * 3 + c]);
        }
#endif
      }
    }

    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 3; c++) {
      pDstRow[x * 3 + c] = minVal[c];
    }
  }
}

// ============================================================================
// Kernel wrapper functions
// ============================================================================

extern "C" {

// FilterMax kernel wrappers
cudaError_t nppiFilterMax_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMax_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMax_8u_C3R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMax_8u_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMax_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMax_8u_C4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMax_16u_C1R_kernel(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMax_16u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMax_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMax_32f_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMax_32f_C3R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMax_32f_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

// FilterMin kernel wrappers
cudaError_t nppiFilterMin_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMin_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMin_8u_C3R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMin_8u_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMin_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMin_8u_C4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height, oMaskSize.width,
                                                                       oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMin_16u_C1R_kernel(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMin_16u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMin_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMin_32f_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

cudaError_t nppiFilterMin_32f_C3R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiFilterMin_32f_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height, oMaskSize.width,
                                                                        oMaskSize.height, oAnchor.x, oAnchor.y);

  return cudaGetLastError();
}

} // extern "C"
