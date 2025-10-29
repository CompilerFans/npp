#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Interpolation helper
__device__ inline float lerp(float a, float b, float t) { return a + t * (b - a); }

// Cubic interpolation weight function (Catmull-Rom)
__device__ inline float cubicWeight(float x) {
  x = fabsf(x);
  if (x <= 1.0f) {
    return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
  } else if (x < 2.0f) {
    return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
  }
  return 0.0f;
}

// Lanczos-3 interpolation weight function
__device__ inline float lanczosWeight(float x) {
  x = fabsf(x);
  if (x < 3.0f) {
    if (x < 0.0001f)
      return 1.0f;
    constexpr float pi = 3.14159265358979323846f;
    float pi_x = pi * x;
    float pi_x_3 = pi_x / 3.0f;
    return (sinf(pi_x) / pi_x) * (sinf(pi_x_3) / pi_x_3);
  }
  return 0.0f;
}

// Interpolation strategies (using template specialization)

// Strategy: Nearest Neighbor
template <typename T, int CHANNELS> struct NearestInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Map destination coordinates to source coordinates
    int sx = (int)(dx * scaleX + 0.5f) + oSrcRectROI.x;
    int sy = (int)(dy * scaleY + 0.5f) + oSrcRectROI.y;

    // Clamp to source bounds
    sx = min(max(sx, oSrcRectROI.x), oSrcRectROI.x + oSrcRectROI.width - 1);
    sy = min(max(sy, oSrcRectROI.y), oSrcRectROI.y + oSrcRectROI.height - 1);

    // Copy pixels
    const T *src_row = (const T *)((const char *)pSrc + sy * nSrcStep);
    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

    int src_idx = sx * CHANNELS;
    int dst_idx = (dx + oDstRectROI.x) * CHANNELS;

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      dst_row[dst_idx + c] = src_row[src_idx + c];
    }
  }
};

// Strategy: Bilinear Interpolation (generic for 8u and integer types)
template <typename T, int CHANNELS> struct BilinearInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Map destination coordinates to source coordinates
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x1 = (int)fx;
    int y1 = (int)fy;
    int x2 = min(x1 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
    int y2 = min(y1 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

    x1 = max(x1, oSrcRectROI.x);
    y1 = max(y1, oSrcRectROI.y);

    float tx = fx - x1;
    float ty = fy - y1;

    const T *src_row1 = (const T *)((const char *)pSrc + y1 * nSrcStep);
    const T *src_row2 = (const T *)((const char *)pSrc + y2 * nSrcStep);
    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float p11 = src_row1[x1 * CHANNELS + c];
      float p12 = src_row1[x2 * CHANNELS + c];
      float p21 = src_row2[x1 * CHANNELS + c];
      float p22 = src_row2[x2 * CHANNELS + c];

      float result = lerp(lerp(p11, p12, tx), lerp(p21, p22, tx), ty);
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(result + 0.5f);
    }
  }
};

// Strategy: Bilinear Interpolation (specialized for float)
template <int CHANNELS> struct BilinearInterpolator<float, CHANNELS> {
  __device__ static void interpolate(const float *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, float *pDst, int nDstStep,
                                     const NppiRect &oDstRectROI) {
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x1 = (int)fx;
    int y1 = (int)fy;
    int x2 = min(x1 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
    int y2 = min(y1 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

    x1 = max(x1, oSrcRectROI.x);
    y1 = max(y1, oSrcRectROI.y);

    float tx = fx - x1;
    float ty = fy - y1;

    const float *src_row1 = (const float *)((const char *)pSrc + y1 * nSrcStep);
    const float *src_row2 = (const float *)((const char *)pSrc + y2 * nSrcStep);
    float *dst_row = (float *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float p11 = src_row1[x1 * CHANNELS + c];
      float p12 = src_row1[x2 * CHANNELS + c];
      float p21 = src_row2[x1 * CHANNELS + c];
      float p22 = src_row2[x2 * CHANNELS + c];

      float result = lerp(lerp(p11, p12, tx), lerp(p21, p22, tx), ty);
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = result; // No rounding for float
    }
  }
};

// Strategy: Super Sampling (only for 8u C3R currently)
template <typename T, int CHANNELS> struct SuperSamplingInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Calculate source region bounds
    float src_x_start = dx * scaleX + oSrcRectROI.x;
    float src_y_start = dy * scaleY + oSrcRectROI.y;
    float src_x_end = (dx + 1) * scaleX + oSrcRectROI.x;
    float src_y_end = (dy + 1) * scaleY + oSrcRectROI.y;

    int x_start = max((int)src_x_start, oSrcRectROI.x);
    int y_start = max((int)src_y_start, oSrcRectROI.y);
    int x_end = min((int)src_x_end, oSrcRectROI.x + oSrcRectROI.width);
    int y_end = min((int)src_y_end, oSrcRectROI.y + oSrcRectROI.height);

    float sum[CHANNELS];
#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      sum[c] = 0.0f;
    }
    int count = 0;

    // Average all source pixels that contribute to this destination pixel
    for (int y = y_start; y < y_end; y++) {
      const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
      for (int x = x_start; x < x_end; x++) {
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c];
        }
        count++;
      }
    }

    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

    if (count > 0) {
#pragma unroll
      for (int c = 0; c < CHANNELS; c++) {
        dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(sum[c] / count + 0.5f);
      }
    } else {
#pragma unroll
      for (int c = 0; c < CHANNELS; c++) {
        dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = 0;
      }
    }
  }
};

// Strategy: Bicubic Interpolation (generic for 8u and integer types)
template <typename T, int CHANNELS> struct CubicInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Map destination coordinates to source coordinates
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x_center = (int)floorf(fx);
    int y_center = (int)floorf(fy);

    float tx = fx - x_center;
    float ty = fy - y_center;

    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float sum = 0.0f;

      // Sample 4x4 neighborhood
      for (int j = -1; j <= 2; j++) {
        int sy = y_center + j;
        sy = max(sy, oSrcRectROI.y);
        sy = min(sy, oSrcRectROI.y + oSrcRectROI.height - 1);

        float wy = cubicWeight(ty - j);

        const T *src_row = (const T *)((const char *)pSrc + sy * nSrcStep);

        for (int i = -1; i <= 2; i++) {
          int sx = x_center + i;
          sx = max(sx, oSrcRectROI.x);
          sx = min(sx, oSrcRectROI.x + oSrcRectROI.width - 1);

          float wx = cubicWeight(tx - i);
          float weight = wx * wy;

          sum += src_row[sx * CHANNELS + c] * weight;
        }
      }

      // Clamp result to valid range
      sum = max(0.0f, min(sum, 255.0f));
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(sum + 0.5f);
    }
  }
};

// Strategy: Bicubic Interpolation (specialized for float)
template <int CHANNELS> struct CubicInterpolator<float, CHANNELS> {
  __device__ static void interpolate(const float *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, float *pDst, int nDstStep,
                                     const NppiRect &oDstRectROI) {
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x_center = (int)floorf(fx);
    int y_center = (int)floorf(fy);

    float tx = fx - x_center;
    float ty = fy - y_center;

    float *dst_row = (float *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float sum = 0.0f;

      for (int j = -1; j <= 2; j++) {
        int sy = y_center + j;
        sy = max(sy, oSrcRectROI.y);
        sy = min(sy, oSrcRectROI.y + oSrcRectROI.height - 1);

        float wy = cubicWeight(ty - j);

        const float *src_row = (const float *)((const char *)pSrc + sy * nSrcStep);

        for (int i = -1; i <= 2; i++) {
          int sx = x_center + i;
          sx = max(sx, oSrcRectROI.x);
          sx = min(sx, oSrcRectROI.x + oSrcRectROI.width - 1);

          float wx = cubicWeight(tx - i);
          float weight = wx * wy;

          sum += src_row[sx * CHANNELS + c] * weight;
        }
      }

      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = sum; // No clamping or rounding for float
    }
  }
};

// Strategy: Lanczos-3 Interpolation (generic for 8u and integer types)
template <typename T, int CHANNELS> struct LanczosInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x_center = (int)floorf(fx);
    int y_center = (int)floorf(fy);

    float tx = fx - x_center;
    float ty = fy - y_center;

    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float sum = 0.0f;

      // Sample 6x6 neighborhood for Lanczos-3
      for (int j = -2; j <= 3; j++) {
        int sy = y_center + j;
        sy = max(sy, oSrcRectROI.y);
        sy = min(sy, oSrcRectROI.y + oSrcRectROI.height - 1);

        float wy = lanczosWeight(ty - j);

        const T *src_row = (const T *)((const char *)pSrc + sy * nSrcStep);

        for (int i = -2; i <= 3; i++) {
          int sx = x_center + i;
          sx = max(sx, oSrcRectROI.x);
          sx = min(sx, oSrcRectROI.x + oSrcRectROI.width - 1);

          float wx = lanczosWeight(tx - i);
          float weight = wx * wy;

          sum += src_row[sx * CHANNELS + c] * weight;
        }
      }

      // Clamp result to valid range
      sum = max(0.0f, min(sum, 255.0f));
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(sum + 0.5f);
    }
  }
};

// Strategy: Lanczos-3 Interpolation (specialized for float)
template <int CHANNELS> struct LanczosInterpolator<float, CHANNELS> {
  __device__ static void interpolate(const float *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, float *pDst, int nDstStep,
                                     const NppiRect &oDstRectROI) {
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x_center = (int)floorf(fx);
    int y_center = (int)floorf(fy);

    float tx = fx - x_center;
    float ty = fy - y_center;

    float *dst_row = (float *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float sum = 0.0f;

      for (int j = -2; j <= 3; j++) {
        int sy = y_center + j;
        sy = max(sy, oSrcRectROI.y);
        sy = min(sy, oSrcRectROI.y + oSrcRectROI.height - 1);

        float wy = lanczosWeight(ty - j);

        const float *src_row = (const float *)((const char *)pSrc + sy * nSrcStep);

        for (int i = -2; i <= 3; i++) {
          int sx = x_center + i;
          sx = max(sx, oSrcRectROI.x);
          sx = min(sx, oSrcRectROI.x + oSrcRectROI.width - 1);

          float wx = lanczosWeight(tx - i);
          float weight = wx * wy;

          sum += src_row[sx * CHANNELS + c] * weight;
        }
      }

      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = sum; // No clamping or rounding for float
    }
  }
};

// Unified resize kernel template
template <typename T, int CHANNELS, typename Interpolator>
__global__ void resizeKernel(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, T *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= oDstRectROI.width || dy >= oDstRectROI.height) {
    return;
  }

  // Calculate scale factors
  float scaleX = (float)oSrcRectROI.width / (float)oDstRectROI.width;
  float scaleY = (float)oSrcRectROI.height / (float)oDstRectROI.height;

  // Delegate to interpolation strategy
  Interpolator::interpolate(pSrc, nSrcStep, oSrcRectROI, dx, dy, scaleX, scaleY, pDst, nDstStep, oDstRectROI);
}

// Wrapper function template
template <typename T, int CHANNELS>
NppStatus resizeImpl(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, T *pDst, int nDstStep,
                     NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                (oDstRectROI.height + blockSize.y - 1) / blockSize.y);

  // Dispatch to appropriate kernel based on interpolation mode
  switch (eInterpolation) {
  case 1: // NPPI_INTER_NN
    resizeKernel<T, CHANNELS, NearestInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 2: // NPPI_INTER_LINEAR
    resizeKernel<T, CHANNELS, BilinearInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 4: // NPPI_INTER_CUBIC
    resizeKernel<T, CHANNELS, CubicInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 8: // NPPI_INTER_SUPER
    resizeKernel<T, CHANNELS, SuperSamplingInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 16: // NPPI_INTER_LANCZOS
    resizeKernel<T, CHANNELS, LanczosInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  default:
    // Default to linear
    resizeKernel<T, CHANNELS, BilinearInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Explicit instantiations for all supported types
extern "C" {

// 8u C1R
NppStatus nppiResize_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return resizeImpl<Npp8u, 1>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                              eInterpolation, nppStreamCtx);
}

// 8u C3R
NppStatus nppiResize_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return resizeImpl<Npp8u, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                              eInterpolation, nppStreamCtx);
}

// 16u C1R
NppStatus nppiResize_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return resizeImpl<Npp16u, 1>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 32f C1R
NppStatus nppiResize_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return resizeImpl<Npp32f, 1>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 32f C3R
NppStatus nppiResize_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return resizeImpl<Npp32f, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

} // extern "C"
