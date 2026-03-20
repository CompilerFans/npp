#include "npp.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__device__ T nearestInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int ix = (int)(fx + 0.5f);
  int iy = (int)(fy + 0.5f);

  if (ix < 0 || ix >= srcSize.width || iy < 0 || iy >= srcSize.height) {
    return T(0); // Return 0 for out of bounds
  }

  const T *src_pixel = (const T *)((const char *)pSrc + iy * nSrcStep) + ix;
  return *src_pixel;
}

template <typename T>
__device__ T bilinearInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = fx - x0;
  float dy = fy - y0;

  // Boundary check - use nearest neighbor interpolation if any point is out of bounds
  if (x0 < 0 || x1 >= srcSize.width || y0 < 0 || y1 >= srcSize.height) {
    // Use nearest neighbor interpolation for out of bounds
    int ix = (int)(fx + 0.5f);
    int iy = (int)(fy + 0.5f);

    // Clamp within boundaries
    ix = (ix < 0) ? 0 : ((ix >= srcSize.width) ? srcSize.width - 1 : ix);
    iy = (iy < 0) ? 0 : ((iy >= srcSize.height) ? srcSize.height - 1 : iy);

    const T *pixel = (const T *)((const char *)pSrc + iy * nSrcStep) + ix;
    return *pixel;
  }

  // Get four corner pixel values
  const T *p00 = (const T *)((const char *)pSrc + y0 * nSrcStep) + x0;
  const T *p01 = (const T *)((const char *)pSrc + y0 * nSrcStep) + x1;
  const T *p10 = (const T *)((const char *)pSrc + y1 * nSrcStep) + x0;
  const T *p11 = (const T *)((const char *)pSrc + y1 * nSrcStep) + x1;

  // Bilinear interpolation computation
  T v0 = *p00 * (1.0f - dx) + *p01 * dx;
  T v1 = *p10 * (1.0f - dx) + *p11 * dx;
  T result = v0 * (1.0f - dy) + v1 * dy;

  return result;
}

template <typename T>
__device__ T cubicInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  // Simplified version: fallback to bilinear interpolation
  // Full bicubic interpolation is computationally expensive, use bilinear approximation here
  return bilinearInterpolation<T>(pSrc, nSrcStep, srcSize, fx, fy);
}

__device__ inline float npp16fToFloatDevice(Npp16f value) {
  union {
    Npp16f npp;
    unsigned short bits;
  } conv{};
  conv.npp = value;
  __half_raw raw{conv.bits};
  __half halfValue = raw;
  return __half2float(halfValue);
}

__device__ inline Npp16f floatToNpp16fDevice(float value) {
  __half halfValue = __float2half(value);
  __half_raw raw = halfValue;
  union {
    Npp16f npp;
    unsigned short bits;
  } conv{};
  conv.bits = raw.x;
  return conv.npp;
}

template <>
__device__ Npp16f nearestInterpolation<Npp16f>(const Npp16f *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int ix = static_cast<int>(fx + 0.5f);
  int iy = static_cast<int>(fy + 0.5f);
  if (ix < 0 || ix >= srcSize.width || iy < 0 || iy >= srcSize.height) {
    return floatToNpp16fDevice(0.0f);
  }
  const Npp16f *srcPixel = reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + iy * nSrcStep) + ix;
  return *srcPixel;
}

template <>
__device__ Npp16f bilinearInterpolation<Npp16f>(const Npp16f *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int x0 = static_cast<int>(floorf(fx));
  int y0 = static_cast<int>(floorf(fy));
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = fx - x0;
  float dy = fy - y0;

  if (x0 < 0 || x1 >= srcSize.width || y0 < 0 || y1 >= srcSize.height) {
    return nearestInterpolation<Npp16f>(pSrc, nSrcStep, srcSize, fx, fy);
  }

  const Npp16f *p00 = reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y0 * nSrcStep) + x0;
  const Npp16f *p01 = reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y0 * nSrcStep) + x1;
  const Npp16f *p10 = reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y1 * nSrcStep) + x0;
  const Npp16f *p11 = reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y1 * nSrcStep) + x1;

  float v00 = npp16fToFloatDevice(*p00);
  float v01 = npp16fToFloatDevice(*p01);
  float v10 = npp16fToFloatDevice(*p10);
  float v11 = npp16fToFloatDevice(*p11);
  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;
  return floatToNpp16fDevice(v0 * (1.0f - dy) + v1 * dy);
}

template <>
__device__ Npp16f cubicInterpolation<Npp16f>(const Npp16f *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  return bilinearInterpolation<Npp16f>(pSrc, nSrcStep, srcSize, fx, fy);
}

template <typename T>
__global__ void mergeAlphaChannelC4Kernel(const T *pSrcWarped, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const T *srcPixel = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrcWarped) + y * nDstStep) + x * 4;
  T *dstPixel = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x * 4;
  T alpha = dstPixel[3];
  dstPixel[0] = srcPixel[0];
  dstPixel[1] = srcPixel[1];
  dstPixel[2] = srcPixel[2];
  dstPixel[3] = alpha;
}

__global__ void nppiWarpPerspective_8u_C1R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                  double c02, double c10, double c11, double c12, double c20,
                                                  double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // Absolute destination coordinates (full image coordinate system)
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // Apply perspective transform (inverse transform)
    // Transform destination coordinates to source coordinates:[x' y' w'] = [dst_x dst_y 1] * inv(M)
    // Perspective transform requires handling homogeneous coordinates
    double w = c20 * dst_x + c21 * dst_y + c22;

    // Prevent division by zero
    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    // Absolute source coordinates
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp8u result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    // Write destination pixel (y and x are already relative coordinates within ROI)
    Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspective_8u_C3R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                  double c02, double c10, double c11, double c12, double c20,
                                                  double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // Absolute destination coordinates (full image coordinate system)
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // Apply perspective transform
    double w = c20 * dst_x + c21 * dst_y + c22;

    // Prevent division by zero
    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    // Absolute source coordinates
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    // Process three channels
    for (int c = 0; c < 3; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 >= 0 && x1 < oSrcSize.width && y0 >= 0 && y1 < oSrcSize.height) {
          const Npp8u *p00 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
          const Npp8u *p01 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
          const Npp8u *p10 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
          const Npp8u *p11 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

          float v0 = (*p00) * (1.0f - dx) + (*p01) * dx;
          float v1 = (*p10) * (1.0f - dx) + (*p11) * dx;
          result = (Npp8u)(v0 * (1.0f - dy) + v1 * dy);
        }
        break;
      }
      case NPPI_INTER_CUBIC:
      default:
        // Simplified to nearest neighbor
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }

      // Write destination pixel (y and x are already relative coordinates within ROI)
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_32f_C1R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // Absolute destination coordinates (full image coordinate system)
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // Apply perspective transform
    double w = c20 * dst_x + c21 * dst_y + c22;

    // Prevent division by zero
    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    // Absolute source coordinates
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp32f result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0.0f;
      break;
    }

    // Write destination pixel
    Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspective_16u_C1R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp16u result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp16u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp16u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp16u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspective_16u_C3R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp16u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 >= 0 && x1 < oSrcSize.width && y0 >= 0 && y1 < oSrcSize.height) {
          const Npp16u *p00 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
          const Npp16u *p01 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
          const Npp16u *p10 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
          const Npp16u *p11 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

          float v0 = (*p00) * (1.0f - dx) + (*p01) * dx;
          float v1 = (*p10) * (1.0f - dx) + (*p11) * dx;
          result = (Npp16u)(v0 * (1.0f - dy) + v1 * dy);
        }
        break;
      }
      case NPPI_INTER_CUBIC:
      default:
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }

      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_16u_C4R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp16u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 >= 0 && x1 < oSrcSize.width && y0 >= 0 && y1 < oSrcSize.height) {
          const Npp16u *p00 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
          const Npp16u *p01 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
          const Npp16u *p10 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
          const Npp16u *p11 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

          float v0 = (*p00) * (1.0f - dx) + (*p01) * dx;
          float v1 = (*p10) * (1.0f - dx) + (*p11) * dx;
          result = (Npp16u)(v0 * (1.0f - dy) + v1 * dy);
        }
        break;
      }
      case NPPI_INTER_CUBIC:
      default:
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }

      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_32f_C4R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      float result = 0.0f;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);

        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));

        const Npp32f *pixel = (const Npp32f *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
        result = *pixel;
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        x0 = max(0, min(x0, oSrcSize.width - 1));
        x1 = max(0, min(x1, oSrcSize.width - 1));
        y0 = max(0, min(y0, oSrcSize.height - 1));
        y1 = max(0, min(y1, oSrcSize.height - 1));

        float dx = abs_fx - floorf(abs_fx);
        float dy = abs_fy - floorf(abs_fy);

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        dx = max(0.0f, min(1.0f, dx));
        dy = max(0.0f, min(1.0f, dy));

        const Npp32f *p00 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
        const Npp32f *p01 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
        const Npp32f *p10 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
        const Npp32f *p11 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

        float v00 = *p00;
        float v01 = *p01;
        float v10 = *p10;
        float v11 = *p11;

        result = v00 * (1.0f - dx) * (1.0f - dy) + v01 * dx * (1.0f - dy) + v10 * (1.0f - dx) * dy + v11 * dx * dy;
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        x0 = max(0, min(x0, oSrcSize.width - 1));
        x1 = max(0, min(x1, oSrcSize.width - 1));
        y0 = max(0, min(y0, oSrcSize.height - 1));
        y1 = max(0, min(y1, oSrcSize.height - 1));

        float dx = abs_fx - floorf(abs_fx);
        float dy = abs_fy - floorf(abs_fy);

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        dx = max(0.0f, min(1.0f, dx));
        dy = max(0.0f, min(1.0f, dy));

        const Npp32f *p00 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
        const Npp32f *p01 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
        const Npp32f *p10 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
        const Npp32f *p11 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

        float v00 = *p00;
        float v01 = *p01;
        float v10 = *p10;
        float v11 = *p11;

        result = v00 * (1.0f - dx) * (1.0f - dy) + v01 * dx * (1.0f - dy) + v10 * (1.0f - dx) * dy + v11 * dx * dy;
        break;
      }
      }

      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_32s_C1R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp32s result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspective_32f_C3R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp32f result = 0.0f;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);

        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));

        const Npp32f *pixel = (const Npp32f *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
        result = *pixel;
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp32f *p00 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
        const Npp32f *p01 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
        const Npp32f *p10 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
        const Npp32f *p11 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

        result = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) + (*p10) * (1.0f - dx) * dy +
                 (*p11) * dx * dy;
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp32f *p00 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
        const Npp32f *p01 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
        const Npp32f *p10 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
        const Npp32f *p11 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

        result = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) + (*p10) * (1.0f - dx) * dy +
                 (*p11) * dx * dy;
        break;
      }
      }

      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_32s_C3R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp32s result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 >= 0 && x1 < oSrcSize.width && y0 >= 0 && y1 < oSrcSize.height) {
          const Npp32s *p00 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
          const Npp32s *p01 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
          const Npp32s *p10 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
          const Npp32s *p11 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

          float v0 = (*p00) * (1.0f - dx) + (*p01) * dx;
          float v1 = (*p10) * (1.0f - dx) + (*p11) * dx;
          result = (Npp32s)(v0 * (1.0f - dy) + v1 * dy);
        }
        break;
      }
      case NPPI_INTER_CUBIC:
      default:
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }

      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_32s_C4R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp32s result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 >= 0 && x1 < oSrcSize.width && y0 >= 0 && y1 < oSrcSize.height) {
          const Npp32s *p00 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
          const Npp32s *p01 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
          const Npp32s *p10 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
          const Npp32s *p11 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

          float v0 = (*p00) * (1.0f - dx) + (*p01) * dx;
          float v1 = (*p10) * (1.0f - dx) + (*p11) * dx;
          result = (Npp32s)(v0 * (1.0f - dy) + v1 * dy);
        }
        break;
      }
      case NPPI_INTER_CUBIC:
      default:
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }

      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspective_8u_C4R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                  double c02, double c10, double c11, double c12, double c20,
                                                  double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);

        // check border
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // check border
        bool x0_valid = (x0 >= 0 && x0 < oSrcSize.width);
        bool x1_valid = (x1 >= 0 && x1 < oSrcSize.width);
        bool y0_valid = (y0 >= 0 && y0 < oSrcSize.height);
        bool y1_valid = (y1 >= 0 && y1 < oSrcSize.height);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        // ensure weight range
        dx = max(0.0f, min(1.0f, dx));
        dy = max(0.0f, min(1.0f, dy));

        // init
        float v00 = 0, v01 = 0, v10 = 0, v11 = 0;

        if (x0_valid && y0_valid) {
          const Npp8u *p00 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
          v00 = *p00;
        }
        if (x1_valid && y0_valid) {
          const Npp8u *p01 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
          v01 = *p01;
        }
        if (x0_valid && y1_valid) {
          const Npp8u *p10 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
          v10 = *p10;
        }
        if (x1_valid && y1_valid) {
          const Npp8u *p11 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;
          v11 = *p11;
        }

        float interpolated =
            v00 * (1.0f - dx) * (1.0f - dy) + v01 * dx * (1.0f - dy) + v10 * (1.0f - dx) * dy + v11 * dx * dy;

        result = (Npp8u)(interpolated + 0.5f); // round off
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);

        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

extern "C" {
NppStatus nppiWarpPerspective_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  // Convert 2D array to separate parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
NppStatus nppiWarpPerspective_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_16u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_16u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_32s_C1R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_32s_C3R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32s_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_32s_C4R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32s_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}

// ============================================================================
// nppiWarpPerspectiveBack API kernel
// WarpPerspectiveBack uses a forward transformation matrix
// ============================================================================

__global__ void nppiWarpPerspectiveBack_8u_C1R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                      NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                      double c00, double c01, double c02, double c10, double c11,
                                                      double c12, double c20, double c21, double c22,
                                                      int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // WarpPerspectiveBack: 使用正向变换矩阵将目标坐标映射到源坐标
    // [src_x, src_y, 1] = [dst_x, dst_y, 1] * M
    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp8u result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspectiveBack_8u_C3R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                      NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                      double c00, double c01, double c02, double c10, double c11,
                                                      double c12, double c20, double c21, double c22,
                                                      int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp8u *p00 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
        const Npp8u *p01 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
        const Npp8u *p10 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
        const Npp8u *p11 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

        float interpolated = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) +
                             (*p10) * (1.0f - dx) * dy + (*p11) * dx * dy;
        result = (Npp8u)(interpolated + 0.5f);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_8u_C4R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                      NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                      double c00, double c01, double c02, double c10, double c11,
                                                      double c12, double c20, double c21, double c22,
                                                      int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp8u *p00 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
        const Npp8u *p01 = (const Npp8u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
        const Npp8u *p10 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
        const Npp8u *p11 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

        float interpolated = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) +
                             (*p10) * (1.0f - dx) * dy + (*p11) * dx * dy;
        result = (Npp8u)(interpolated + 0.5f);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_16u_C1R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp16u result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp16u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp16u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp16u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspectiveBack_16u_C3R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp16u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp16u *p00 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
        const Npp16u *p01 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
        const Npp16u *p10 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
        const Npp16u *p11 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

        float interpolated = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) +
                             (*p10) * (1.0f - dx) * dy + (*p11) * dx * dy;
        result = (Npp16u)(interpolated + 0.5f);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_16u_C4R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp16u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp16u *p00 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
        const Npp16u *p01 = (const Npp16u *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
        const Npp16u *p10 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
        const Npp16u *p11 = (const Npp16u *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

        float interpolated = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) +
                             (*p10) * (1.0f - dx) * dy + (*p11) * dx * dy;
        result = (Npp16u)(interpolated + 0.5f);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16u *pixel = (const Npp16u *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_32f_C1R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp32f result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0.0f;
      break;
    }

    Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspectiveBack_32f_C3R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp32f result = 0.0f;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32f *pixel = (const Npp32f *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp32f *p00 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
        const Npp32f *p01 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
        const Npp32f *p10 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
        const Npp32f *p11 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

        result = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) + (*p10) * (1.0f - dx) * dy +
                 (*p11) * dx * dy;
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32f *pixel = (const Npp32f *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_32f_C4R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp32f result = 0.0f;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32f *pixel = (const Npp32f *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp32f *p00 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
        const Npp32f *p01 = (const Npp32f *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
        const Npp32f *p10 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
        const Npp32f *p11 = (const Npp32f *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

        result = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) + (*p10) * (1.0f - dx) * dy +
                 (*p11) * dx * dy;
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32f *pixel = (const Npp32f *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_32s_C1R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    Npp32s result;
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolation<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpPerspectiveBack_32s_C3R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp32s result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp32s *p00 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x0 * 3 + c;
        const Npp32s *p01 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x1 * 3 + c;
        const Npp32s *p10 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x0 * 3 + c;
        const Npp32s *p11 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x1 * 3 + c;

        float interpolated = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) +
                             (*p10) * (1.0f - dx) * dy + (*p11) * dx * dy;
        result = (Npp32s)(interpolated + 0.5f);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpPerspectiveBack_32s_C4R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                       NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                       double c00, double c01, double c02, double c10, double c11,
                                                       double c12, double c20, double c21, double c22,
                                                       int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double w = c20 * dst_x + c21 * dst_y + c22;

    if (fabs(w) < 1e-10) {
      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 4;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = dst_pixel[3] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp32s result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        abs_fx = max(0.0f, min(abs_fx, (float)(oSrcSize.width - 1)));
        abs_fy = max(0.0f, min(abs_fy, (float)(oSrcSize.height - 1)));

        int x0 = (int)floorf(abs_fx);
        int y0 = (int)floorf(abs_fy);
        int x1 = min(x0 + 1, oSrcSize.width - 1);
        int y1 = min(y0 + 1, oSrcSize.height - 1);

        float dx = abs_fx - x0;
        float dy = abs_fy - y0;

        if (x0 == x1)
          dx = 0.0f;
        if (y0 == y1)
          dy = 0.0f;

        const Npp32s *p00 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x0 * 4 + c;
        const Npp32s *p01 = (const Npp32s *)((const char *)pSrc + y0 * nSrcStep) + x1 * 4 + c;
        const Npp32s *p10 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x0 * 4 + c;
        const Npp32s *p11 = (const Npp32s *)((const char *)pSrc + y1 * nSrcStep) + x1 * 4 + c;

        float interpolated = (*p00) * (1.0f - dx) * (1.0f - dy) + (*p01) * dx * (1.0f - dy) +
                             (*p10) * (1.0f - dx) * dy + (*p11) * dx * dy;
        result = (Npp32s)(interpolated + 0.5f);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        int ix = (int)roundf(abs_fx);
        int iy = (int)roundf(abs_fy);
        ix = max(0, min(ix, oSrcSize.width - 1));
        iy = max(0, min(iy, oSrcSize.height - 1));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp32s *pixel = (const Npp32s *)((const char *)pSrc + iy * nSrcStep) + ix * 4 + c;
          result = *pixel;
        }
        break;
      }
      }

      Npp32s *dst_pixel = (Npp32s *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

// ============================================================================
// WarpPerspectiveBack implement
// ============================================================================

extern "C" {
// 8u C1R 实现
NppStatus nppiWarpPerspectiveBack_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                  const double aCoeffs[3][3], int eInterpolation,
                                                  NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8u C3R 实现
NppStatus nppiWarpPerspectiveBack_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                  const double aCoeffs[3][3], int eInterpolation,
                                                  NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8u C4R 实现
NppStatus nppiWarpPerspectiveBack_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                  const double aCoeffs[3][3], int eInterpolation,
                                                  NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16u C1R 实现
NppStatus nppiWarpPerspectiveBack_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16u C3R 实现
NppStatus nppiWarpPerspectiveBack_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_16u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16u C4R 实现
NppStatus nppiWarpPerspectiveBack_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_16u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32f C1R 实现
NppStatus nppiWarpPerspectiveBack_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32f C3R 实现
NppStatus nppiWarpPerspectiveBack_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32f C4R 实现
NppStatus nppiWarpPerspectiveBack_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32s C1R 实现
NppStatus nppiWarpPerspectiveBack_32s_C1R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_32s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32s C3R 实现
NppStatus nppiWarpPerspectiveBack_32s_C3R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_32s_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32s C4R 实现
NppStatus nppiWarpPerspectiveBack_32s_C4R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspectiveBack_32s_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  if (nppStreamCtx.hStream == 0) {
    cudaStatus = cudaDeviceSynchronize();
  } else {
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
  }
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

}

__global__ void nppiWarpPerspective_16f_C1R_kernel(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                   Npp16f *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                   double c02, double c10, double c11, double c12, double c20,
                                                   double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;
    double w = c20 * dst_x + c21 * dst_y + c22;
    Npp16f *dstPixel = reinterpret_cast<Npp16f *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x;
    if (fabs(w) < 1e-10) {
      *dstPixel = floatToNpp16fDevice(0.0f);
      return;
    }
    float srcX = static_cast<float>((c00 * dst_x + c01 * dst_y + c02) / w);
    float srcY = static_cast<float>((c10 * dst_x + c11 * dst_y + c12) / w);
    switch (eInterpolation) {
    case NPPI_INTER_NN:
      *dstPixel = nearestInterpolation<Npp16f>(pSrc, nSrcStep, oSrcSize, srcX, srcY);
      break;
    case NPPI_INTER_LINEAR:
      *dstPixel = bilinearInterpolation<Npp16f>(pSrc, nSrcStep, oSrcSize, srcX, srcY);
      break;
    default:
      *dstPixel = cubicInterpolation<Npp16f>(pSrc, nSrcStep, oSrcSize, srcX, srcY);
      break;
    }
  }
}

template <int channels>
__global__ void nppiWarpPerspective_16f_CxR_kernel(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                   Npp16f *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                   double c02, double c10, double c11, double c12, double c20,
                                                   double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;
    double w = c20 * dst_x + c21 * dst_y + c22;
    Npp16f *dstPixel = reinterpret_cast<Npp16f *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x * channels;
    if (fabs(w) < 1e-10) {
      for (int c = 0; c < channels; ++c) {
        dstPixel[c] = floatToNpp16fDevice(0.0f);
      }
      return;
    }

    float fx = static_cast<float>((c00 * dst_x + c01 * dst_y + c02) / w);
    float fy = static_cast<float>((c10 * dst_x + c11 * dst_y + c12) / w);
    for (int c = 0; c < channels; ++c) {
      float result = 0.0f;
      switch (eInterpolation) {
      case NPPI_INTER_NN: {
        int ix = static_cast<int>(roundf(fx));
        int iy = static_cast<int>(roundf(fy));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16f *pixel =
              reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + iy * nSrcStep) + ix * channels + c;
          result = npp16fToFloatDevice(*pixel);
        }
        break;
      }
      case NPPI_INTER_LINEAR: {
        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float dx = fx - x0;
        float dy = fy - y0;
        if (x0 >= 0 && x1 < oSrcSize.width && y0 >= 0 && y1 < oSrcSize.height) {
          const Npp16f *p00 =
              reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y0 * nSrcStep) + x0 * channels + c;
          const Npp16f *p01 =
              reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y0 * nSrcStep) + x1 * channels + c;
          const Npp16f *p10 =
              reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y1 * nSrcStep) + x0 * channels + c;
          const Npp16f *p11 =
              reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + y1 * nSrcStep) + x1 * channels + c;
          float v0 = npp16fToFloatDevice(*p00) * (1.0f - dx) + npp16fToFloatDevice(*p01) * dx;
          float v1 = npp16fToFloatDevice(*p10) * (1.0f - dx) + npp16fToFloatDevice(*p11) * dx;
          result = v0 * (1.0f - dy) + v1 * dy;
        }
        break;
      }
      default: {
        int ix = static_cast<int>(roundf(fx));
        int iy = static_cast<int>(roundf(fy));
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp16f *pixel =
              reinterpret_cast<const Npp16f *>(reinterpret_cast<const char *>(pSrc) + iy * nSrcStep) + ix * channels + c;
          result = npp16fToFloatDevice(*pixel);
        }
        break;
      }
      }
      dstPixel[c] = floatToNpp16fDevice(result);
    }
  }
}

template <typename T>
static NppStatus mergeAlphaLaunch(const T *pSrcWarped, T *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  mergeAlphaChannelC4Kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrcWarped, pDst, nDstStep, oSizeROI.width,
                                                                               oSizeROI.height);
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  cudaStatus = nppStreamCtx.hStream == 0 ? cudaDeviceSynchronize() : cudaStreamSynchronize(nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

extern "C" {
NppStatus nppiWarpPerspective_16f_C1R_Ctx_impl(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);
  nppiWarpPerspective_16f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  cudaStatus = nppStreamCtx.hStream == 0 ? cudaDeviceSynchronize() : cudaStreamSynchronize(nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiWarpPerspective_16f_C3R_Ctx_impl(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);
  nppiWarpPerspective_16f_CxR_kernel<3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  cudaStatus = nppStreamCtx.hStream == 0 ? cudaDeviceSynchronize() : cudaStreamSynchronize(nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiWarpPerspective_16f_C4R_Ctx_impl(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);
  nppiWarpPerspective_16f_CxR_kernel<4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], aCoeffs[2][0], aCoeffs[2][1], aCoeffs[2][2], eInterpolation);
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  cudaStatus = nppStreamCtx.hStream == 0 ? cudaDeviceSynchronize() : cudaStreamSynchronize(nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMergeAlpha_8u_C4R(const Npp8u *pSrcWarped, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx) {
  return mergeAlphaLaunch(pSrcWarped, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMergeAlpha_16u_C4R(const Npp16u *pSrcWarped, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return mergeAlphaLaunch(pSrcWarped, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMergeAlpha_32f_C4R(const Npp32f *pSrcWarped, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return mergeAlphaLaunch(pSrcWarped, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMergeAlpha_32s_C4R(const Npp32s *pSrcWarped, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx) {
  return mergeAlphaLaunch(pSrcWarped, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
}
