#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float bicubicCoeff(float x_) {
  float x = fabsf(x_);
  if (x <= 1.0f) {
    return x * x * (1.5f * x - 2.5f) + 1.0f;
  } else if (x < 2.0f) {
    return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
  }
  return 0.0f;
}

__device__ __forceinline__ Npp8u saturateCastU8(float v) {
  v = (v < 0.0f) ? 0.0f : ((v > 255.0f) ? 255.0f : v);
  return static_cast<Npp8u>(v + 0.5f);
}

template <typename T>
__device__ __forceinline__ float getPixel(const T *pSrc, int nSrcStep, NppiSize srcSize, int x, int y) {
  if (x < 0 || x >= srcSize.width || y < 0 || y >= srcSize.height) {
    return 0.0f;
  }
  const T *p = (const T *)((const char *)pSrc + y * nSrcStep) + x;
  return static_cast<float>(*p);
}

template <typename T>
__device__ __forceinline__ float getPixelPacked(const T *pSrc, int nSrcStep, NppiSize srcSize, int x, int y, int c,
                                                int channels) {
  if (x < 0 || x >= srcSize.width || y < 0 || y >= srcSize.height) {
    return 0.0f;
  }
  const T *p = (const T *)((const char *)pSrc + y * nSrcStep) + x * channels + c;
  return static_cast<float>(*p);
}

template <typename T>
__device__ T nearestInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int ix = (int)(fx + 0.5f);
  int iy = (int)(fy + 0.5f);

  if (ix < 0 || ix >= srcSize.width || iy < 0 || iy >= srcSize.height) {
    return T(0); // Return 0 for out-of-bounds
  }

  const T *src_pixel = (const T *)((const char *)pSrc + iy * nSrcStep) + ix;
  return *src_pixel;
}
template <typename T>
__device__ __forceinline__ T nearestInterpolationPacked(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx,
                                                        float fy, int c, int channels) {
  int ix = (int)(fx + 0.5f);
  int iy = (int)(fy + 0.5f);

  if (ix < 0 || ix >= srcSize.width || iy < 0 || iy >= srcSize.height) {
    return T(0);
  }

  const T *p = (const T *)((const char *)pSrc + iy * nSrcStep) + ix * channels + c;
  return *p;
}

template <typename T>
__device__ T bilinearInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = fx - x0;
  float dy = fy - y0;

  float v00 = getPixel(pSrc, nSrcStep, srcSize, x0, y0);
  float v01 = getPixel(pSrc, nSrcStep, srcSize, x1, y0);
  float v10 = getPixel(pSrc, nSrcStep, srcSize, x0, y1);
  float v11 = getPixel(pSrc, nSrcStep, srcSize, x1, y1);

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;
  float result = v0 * (1.0f - dy) + v1 * dy;

  return static_cast<T>(result);
}

template <typename T>
__device__ T bilinearInterpolationPacked(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy, int c,
                                         int channels) {
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = fx - x0;
  float dy = fy - y0;

  float v00 = getPixelPacked(pSrc, nSrcStep, srcSize, x0, y0, c, channels);
  float v01 = getPixelPacked(pSrc, nSrcStep, srcSize, x1, y0, c, channels);
  float v10 = getPixelPacked(pSrc, nSrcStep, srcSize, x0, y1, c, channels);
  float v11 = getPixelPacked(pSrc, nSrcStep, srcSize, x1, y1, c, channels);

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;
  float result = v0 * (1.0f - dy) + v1 * dy;

  return static_cast<T>(result);
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
      sum += getPixel(pSrc, nSrcStep, srcSize, xx, yy) * w;
      wsum += w;
    }
  }

  float res = (wsum == 0.0f) ? 0.0f : (sum / wsum);
  return static_cast<T>(res);
}

__device__ Npp8u cubicInterpolationU8(const Npp8u *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
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
      sum += getPixel(pSrc, nSrcStep, srcSize, xx, yy) * w;
      wsum += w;
    }
  }

  float res = (wsum == 0.0f) ? 0.0f : (sum / wsum);
  return saturateCastU8(res);
}

template <typename T>
__device__ T cubicInterpolationPacked(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy, int c,
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
      sum += getPixelPacked(pSrc, nSrcStep, srcSize, xx, yy, c, channels) * w;
      wsum += w;
    }
  }

  float res = (wsum == 0.0f) ? 0.0f : (sum / wsum);
  return static_cast<T>(res);
}

__device__ Npp8u cubicInterpolationPackedU8(const Npp8u *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy,
                                            int c, int channels) {
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
      sum += getPixelPacked(pSrc, nSrcStep, srcSize, xx, yy, c, channels) * w;
      wsum += w;
    }
  }

  float res = (wsum == 0.0f) ? 0.0f : (sum / wsum);
  return saturateCastU8(res);
}

__global__ void nppiWarpAffine_8u_C1R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                             double c02, double c10, double c11, double c12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // Absolute destination coordinates (full image coordinates)
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // Apply affine transform (inverse)
    // Transform destination to source: [x' y' 1] = [dst_x dst_y 1] * inv(M)
    double src_x = c00 * dst_x + c01 * dst_y + c02;
    double src_y = c10 * dst_x + c11 * dst_y + c12;

    Npp8u result;
    // Absolute source coordinates, no ROI offset subtraction
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    switch (eInterpolation) {
    case NPPI_INTER_NN:
      result = nearestInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_LINEAR:
      result = bilinearInterpolation<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    case NPPI_INTER_CUBIC:
      result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    // Write to destination pixel (y and x are ROI-relative coordinates)
    Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpAffine_8u_C3R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                             double c02, double c10, double c11, double c12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // Absolute destination coordinates (full image coordinates)
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // Apply affine transform
    double src_x = c00 * dst_x + c01 * dst_y + c02;
    double src_y = c10 * dst_x + c11 * dst_y + c12;

    // Absolute source coordinates, no ROI offset subtraction
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    // Process three channels
    for (int c = 0; c < 3; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      case NPPI_INTER_CUBIC:
        result = cubicInterpolationPackedU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      default:
        result = 0;
        break;
      }

      // Write to destination pixel (y and x are ROI-relative coordinates)
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpAffine_32f_C1R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                              double c02, double c10, double c11, double c12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // Absolute destination coordinates (full image coordinates)
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // Apply affine transform
    double src_x = c00 * dst_x + c01 * dst_y + c02;
    double src_y = c10 * dst_x + c11 * dst_y + c12;

    Npp32f result;
    // Absolute source coordinates, no ROI offset subtraction
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

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

    // Write to destination pixel (y and x are ROI-relative coordinates)
    Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

static inline bool invertAffineCoeffs(const double in[2][3], double out[2][3]) {
  const double a00 = in[0][0];
  const double a01 = in[0][1];
  const double a02 = in[0][2];
  const double a10 = in[1][0];
  const double a11 = in[1][1];
  const double a12 = in[1][2];

  const double det = a00 * a11 - a01 * a10;
  if (fabs(det) < 1e-12) {
    return false;
  }

  const double invDet = 1.0 / det;
  const double b00 = a11 * invDet;
  const double b01 = -a01 * invDet;
  const double b10 = -a10 * invDet;
  const double b11 = a00 * invDet;

  out[0][0] = b00;
  out[0][1] = b01;
  out[1][0] = b10;
  out[1][1] = b11;
  out[0][2] = -(b00 * a02 + b01 * a12);
  out[1][2] = -(b10 * a02 + b11 * a12);
  return true;
}

extern "C" {
NppStatus nppiWarpAffine_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  double invCoeffs[2][3];
  if (!invertAffineCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  double flatCoeffs[6] = {invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
                          invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2]};

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffine_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, flatCoeffs[0], flatCoeffs[1], flatCoeffs[2],
      flatCoeffs[3], flatCoeffs[4], flatCoeffs[5], eInterpolation);

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

NppStatus nppiWarpAffine_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  double invCoeffs[2][3];
  if (!invertAffineCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  double flatCoeffs[6] = {invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
                          invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2]};

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffine_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, flatCoeffs[0], flatCoeffs[1], flatCoeffs[2],
      flatCoeffs[3], flatCoeffs[4], flatCoeffs[5], eInterpolation);

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

NppStatus nppiWarpAffine_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  double invCoeffs[2][3];
  if (!invertAffineCoeffs(aCoeffs, invCoeffs)) {
    return NPP_COEFFICIENT_ERROR;
  }
  double flatCoeffs[6] = {invCoeffs[0][0], invCoeffs[0][1], invCoeffs[0][2],
                          invCoeffs[1][0], invCoeffs[1][1], invCoeffs[1][2]};

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffine_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, flatCoeffs[0], flatCoeffs[1], flatCoeffs[2],
      flatCoeffs[3], flatCoeffs[4], flatCoeffs[5], eInterpolation);

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

// =============================================
// WarpAffineBack kernel
// WarpAffineBack use 2x3 matrix
// =============================================
__global__ void nppiWarpAffineBack_8u_C1R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                 Npp8u *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                 double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // WarpAffineBack: 使用正向仿射变换矩阵将目标坐标映射到源坐标
    // src_x = a00 * dst_x + a01 * dst_y + a02
    // src_y = a10 * dst_x + a11 * dst_y + a12
    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

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
      result = cubicInterpolationU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy);
      break;
    default:
      result = 0;
      break;
    }

    Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

__global__ void nppiWarpAffineBack_8u_C3R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                 Npp8u *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                 double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      case NPPI_INTER_CUBIC:
        result = cubicInterpolationPackedU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      default:
        result = 0;
        break;
      }

      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpAffineBack_8u_C4R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                 Npp8u *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                 double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp8u result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp8u>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      case NPPI_INTER_CUBIC:
        result = cubicInterpolationPackedU8(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      default:
        result = 0;
        break;
      }

      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 4 + c;
      *dst_pixel = result;
    }
  }
}

__global__ void nppiWarpAffineBack_16u_C1R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp16u *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

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

__global__ void nppiWarpAffineBack_16u_C3R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp16u *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

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

__global__ void nppiWarpAffineBack_16u_C4R_kernel(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp16u *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

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

__global__ void nppiWarpAffineBack_32f_C1R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp32f *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

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

__global__ void nppiWarpAffineBack_32f_C3R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp32f *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp32f result = 0.0f;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
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

__global__ void nppiWarpAffineBack_32f_C4R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp32f *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp32f result = 0.0f;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp32f>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
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

__global__ void nppiWarpAffineBack_32s_C1R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp32s *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

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

__global__ void nppiWarpAffineBack_32s_C3R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp32s *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 3; c++) {
      Npp32s result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 3);
        break;
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

__global__ void nppiWarpAffineBack_32s_C4R_kernel(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp32s *pDst, int nDstStep, NppiRect oDstROI, double a00, double a01,
                                                  double a02, double a10, double a11, double a12, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    double src_x = a00 * dst_x + a01 * dst_y + a02;
    double src_y = a10 * dst_x + a11 * dst_y + a12;

    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    for (int c = 0; c < 4; c++) {
      Npp32s result = 0;

      switch (eInterpolation) {
      case NPPI_INTER_NN:
        result = nearestInterpolationPacked<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
      case NPPI_INTER_LINEAR:
        result = bilinearInterpolationPacked<Npp32s>(pSrc, nSrcStep, oSrcSize, abs_fx, abs_fy, c, 4);
        break;
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

// =============================================
// WarpAffineBack implement
// =============================================

extern "C" {
NppStatus nppiWarpAffineBack_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_16u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_16u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_32f_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_32s_C1R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_32s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_32s_C3R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_32s_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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

NppStatus nppiWarpAffineBack_32s_C4R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  nppiWarpAffineBack_32s_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs[0][0], aCoeffs[0][1], aCoeffs[0][2],
      aCoeffs[1][0], aCoeffs[1][1], aCoeffs[1][2], eInterpolation);

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
