#include "npp.h"
#include <cmath>
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

  // Use constant border 0 for out-of-bounds samples
  float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;

  if (x0 >= 0 && x0 < srcSize.width && y0 >= 0 && y0 < srcSize.height) {
    const T *p00 = (const T *)((const char *)pSrc + y0 * nSrcStep) + x0;
    v00 = static_cast<float>(*p00);
  }
  if (x1 >= 0 && x1 < srcSize.width && y0 >= 0 && y0 < srcSize.height) {
    const T *p01 = (const T *)((const char *)pSrc + y0 * nSrcStep) + x1;
    v01 = static_cast<float>(*p01);
  }
  if (x0 >= 0 && x0 < srcSize.width && y1 >= 0 && y1 < srcSize.height) {
    const T *p10 = (const T *)((const char *)pSrc + y1 * nSrcStep) + x0;
    v10 = static_cast<float>(*p10);
  }
  if (x1 >= 0 && x1 < srcSize.width && y1 >= 0 && y1 < srcSize.height) {
    const T *p11 = (const T *)((const char *)pSrc + y1 * nSrcStep) + x1;
    v11 = static_cast<float>(*p11);
  }

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;
  float result = v0 * (1.0f - dy) + v1 * dy;

  return static_cast<T>(result);
}

__device__ __forceinline__ float bicubicCoeff(float x_) {
  float x = fabsf(x_);
  if (x <= 1.0f) {
    return x * x * (1.5f * x - 2.5f) + 1.0f;
  } else if (x < 2.0f) {
    return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
  }
  return 0.0f;
}

template <typename T>
__device__ T cubicInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {

  float xmin = ceilf(fx - 2.0f);
  float xmax = floorf(fx + 2.0f);
  float ymin = ceilf(fy - 2.0f);
  float ymax = floorf(fy + 2.0f);

  float sum = 0.0f;
  float wsum = 0.0f;

  for (float cy = ymin; cy <= ymax; cy += 1.0f) {
    int yy = (int)floorf(cy);
    for (float cx = xmin; cx <= xmax; cx += 1.0f) {
      int xx = (int)floorf(cx);
      float w = bicubicCoeff(fx - cx) * bicubicCoeff(fy - cy);
      if (xx >= 0 && xx < srcSize.width && yy >= 0 && yy < srcSize.height) {
        const T *p = (const T *)((const char *)pSrc + yy * nSrcStep) + xx;
        sum += static_cast<float>(*p) * w;
      }
      wsum += w;
    }
  }

  float res = (wsum == 0.0f) ? 0.0f : (sum / wsum);
  return static_cast<T>(res);
}

__device__ __forceinline__ Npp8u saturateCastU8(float v) {
  v = (v < 0.0f) ? 0.0f : ((v > 255.0f) ? 255.0f : v);
  return static_cast<Npp8u>(v + 0.5f);
}

__device__ __forceinline__ float getPixelU8(const Npp8u *pSrc, int nSrcStep, NppiSize srcSize, int x, int y, int c,
                                            int channels) {
  if (x < 0 || x >= srcSize.width || y < 0 || y >= srcSize.height) {
    return 0.0f;
  }
  const Npp8u *p = (const Npp8u *)((const char *)pSrc + y * nSrcStep) + x * channels + c;
  return static_cast<float>(*p);
}

__device__ Npp8u bilinearInterpolationU8(const Npp8u *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy, int c,
                                         int channels) {
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = fx - x0;
  float dy = fy - y0;

  float v00 = getPixelU8(pSrc, nSrcStep, srcSize, x0, y0, c, channels);
  float v01 = getPixelU8(pSrc, nSrcStep, srcSize, x1, y0, c, channels);
  float v10 = getPixelU8(pSrc, nSrcStep, srcSize, x0, y1, c, channels);
  float v11 = getPixelU8(pSrc, nSrcStep, srcSize, x1, y1, c, channels);

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;
  return saturateCastU8(v0 * (1.0f - dy) + v1 * dy);
}

__device__ Npp8u cubicInterpolationU8(const Npp8u *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy, int c,
                                      int channels) {
  float xmin = ceilf(fx - 2.0f);
  float xmax = floorf(fx + 2.0f);
  float ymin = ceilf(fy - 2.0f);
  float ymax = floorf(fy + 2.0f);

  float sum = 0.0f;
  float wsum = 0.0f;

  for (float cy = ymin; cy <= ymax; cy += 1.0f) {
    int yy = (int)floorf(cy);
    for (float cx = xmin; cx <= xmax; cx += 1.0f) {
      int xx = (int)floorf(cx);
      float w = bicubicCoeff(fx - cx) * bicubicCoeff(fy - cy);
      sum += getPixelU8(pSrc, nSrcStep, srcSize, xx, yy, c, channels) * w;
      wsum += w;
    }
  }

  float res = (wsum == 0.0f) ? 0.0f : (sum / wsum);
  return saturateCastU8(res);
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
      result = bilinearInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, 0, 1);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, 0, 1);
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
        result = bilinearInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      }
      case NPPI_INTER_CUBIC:
      default:
        result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
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
        result = bilinearInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      }
      }

      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

static inline bool invertPerspectiveCoeffs(const double in[3][3], double out[3][3]) {
  const double a00 = in[0][0], a01 = in[0][1], a02 = in[0][2];
  const double a10 = in[1][0], a11 = in[1][1], a12 = in[1][2];
  const double a20 = in[2][0], a21 = in[2][1], a22 = in[2][2];

  const double c00 = a11 * a22 - a12 * a21;
  const double c01 = a02 * a21 - a01 * a22;
  const double c02 = a01 * a12 - a02 * a11;
  const double c10 = a12 * a20 - a10 * a22;
  const double c11 = a00 * a22 - a02 * a20;
  const double c12 = a02 * a10 - a00 * a12;
  const double c20 = a10 * a21 - a11 * a20;
  const double c21 = a01 * a20 - a00 * a21;
  const double c22 = a00 * a11 - a01 * a10;

  const double det = a00 * c00 + a01 * c10 + a02 * c20;
  if (fabs(det) < 1e-12) {
    return false;
  }

  const double invDet = 1.0 / det;
  out[0][0] = c00 * invDet;
  out[0][1] = c01 * invDet;
  out[0][2] = c02 * invDet;
  out[1][0] = c10 * invDet;
  out[1][1] = c11 * invDet;
  out[1][2] = c12 * invDet;
  out[2][0] = c20 * invDet;
  out[2][1] = c21 * invDet;
  out[2][2] = c22 * invDet;
  return true;
}

extern "C" {
NppStatus nppiWarpPerspective_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  // Convert 2D array to separate parameters
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_16u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_16u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32s_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_32s_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
  double invCoeffs[3][3];
  if (!invertPerspectiveCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpPerspective_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
      invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2], invCoeffs[2][0], invCoeffs[2][1], invCoeffs[2][2],
      eInterpolation);

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
      result = bilinearInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, 0, 1);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, 0, 1);
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
        result = bilinearInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
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
        result = bilinearInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      }
      case NPPI_INTER_CUBIC:
      default: {
        result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
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
