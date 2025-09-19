#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP Perspective Warp Operations
 * 实现透视变换的GPU加速计算
 */

/**
 * 最近邻插值
 */
template <typename T>
__device__ T nearestInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int ix = (int)(fx + 0.5f);
  int iy = (int)(fy + 0.5f);

  if (ix < 0 || ix >= srcSize.width || iy < 0 || iy >= srcSize.height) {
    return T(0); // 边界外返回0
  }

  const T *src_pixel = (const T *)((const char *)pSrc + iy * nSrcStep) + ix;
  return *src_pixel;
}

/**
 * 双线性插值
 */
template <typename T>
__device__ T bilinearInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = fx - x0;
  float dy = fy - y0;

  // 边界检查 - 如果任何一个点超出边界，使用最近邻插值
  if (x0 < 0 || x1 >= srcSize.width || y0 < 0 || y1 >= srcSize.height) {
    // 边界外使用最近邻插值
    int ix = (int)(fx + 0.5f);
    int iy = (int)(fy + 0.5f);

    // 限制在边界内
    ix = (ix < 0) ? 0 : ((ix >= srcSize.width) ? srcSize.width - 1 : ix);
    iy = (iy < 0) ? 0 : ((iy >= srcSize.height) ? srcSize.height - 1 : iy);

    const T *pixel = (const T *)((const char *)pSrc + iy * nSrcStep) + ix;
    return *pixel;
  }

  // 获取四个角点像素值
  const T *p00 = (const T *)((const char *)pSrc + y0 * nSrcStep) + x0;
  const T *p01 = (const T *)((const char *)pSrc + y0 * nSrcStep) + x1;
  const T *p10 = (const T *)((const char *)pSrc + y1 * nSrcStep) + x0;
  const T *p11 = (const T *)((const char *)pSrc + y1 * nSrcStep) + x1;

  // 双线性插值计算
  T v0 = *p00 * (1.0f - dx) + *p01 * dx;
  T v1 = *p10 * (1.0f - dx) + *p11 * dx;
  T result = v0 * (1.0f - dy) + v1 * dy;

  return result;
}

/**
 * 三次插值（简化的双三次插值）
 */
template <typename T>
__device__ T cubicInterpolation(const T *pSrc, int nSrcStep, NppiSize srcSize, float fx, float fy) {
  // 简化版本：降级为双线性插值
  // 完整的双三次插值计算量较大，这里使用双线性近似
  return bilinearInterpolation<T>(pSrc, nSrcStep, srcSize, fx, fy);
}

/**
 * 8位单通道透视变换内核
 */
__global__ void nppiWarpPerspective_8u_C1R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                  double c02, double c10, double c11, double c12, double c20,
                                                  double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // 绝对目标坐标（全图像坐标系）
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // 应用透视变换（逆变换）
    // 目标坐标变换到源坐标：[x' y' w'] = [dst_x dst_y 1] * inv(M)
    // 透视变换需要处理齐次坐标
    double w = c20 * dst_x + c21 * dst_y + c22;

    // 防止除零错误
    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    // 绝对源坐标
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

    // 写入目标像素（y和x已经是ROI内的相对坐标）
    Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

/**
 * 8位三通道透视变换内核
 */
__global__ void nppiWarpPerspective_8u_C3R_kernel(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI, double c00, double c01,
                                                  double c02, double c10, double c11, double c12, double c20,
                                                  double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // 绝对目标坐标（全图像坐标系）
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // 应用透视变换
    double w = c20 * dst_x + c21 * dst_y + c22;

    // 防止除零错误
    if (fabs(w) < 1e-10) {
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3;
      dst_pixel[0] = dst_pixel[1] = dst_pixel[2] = 0;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    // 绝对源坐标
    float abs_fx = (float)src_x;
    float abs_fy = (float)src_y;

    // 处理三个通道
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
        // 简化为最近邻
        int ix = (int)(abs_fx + 0.5f);
        int iy = (int)(abs_fy + 0.5f);
        if (ix >= 0 && ix < oSrcSize.width && iy >= 0 && iy < oSrcSize.height) {
          const Npp8u *pixel = (const Npp8u *)((const char *)pSrc + iy * nSrcStep) + ix * 3 + c;
          result = *pixel;
        }
        break;
      }

      // 写入目标像素（y和x已经是ROI内的相对坐标）
      Npp8u *dst_pixel = (Npp8u *)((char *)pDst + y * nDstStep) + x * 3 + c;
      *dst_pixel = result;
    }
  }
}

/**
 * 32位浮点单通道透视变换内核
 */
__global__ void nppiWarpPerspective_32f_C1R_kernel(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   double c00, double c01, double c02, double c10, double c11,
                                                   double c12, double c20, double c21, double c22, int eInterpolation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstROI.width && y < oDstROI.height) {
    // 绝对目标坐标（全图像坐标系）
    int dst_x = x + oDstROI.x;
    int dst_y = y + oDstROI.y;

    // 应用透视变换
    double w = c20 * dst_x + c21 * dst_y + c22;

    // 防止除零错误
    if (fabs(w) < 1e-10) {
      Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
      *dst_pixel = 0.0f;
      return;
    }

    double src_x = (c00 * dst_x + c01 * dst_y + c02) / w;
    double src_y = (c10 * dst_x + c11 * dst_y + c12) / w;

    // 绝对源坐标
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

    // 写入目标像素
    Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;
    *dst_pixel = result;
  }
}

extern "C" {

/**
 * 8位单通道透视变换CUDA实现
 */
NppStatus nppiWarpPerspective_8u_C1R_Ctx_cuda(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  // 将二维数组转换为单独的参数
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

/**
 * 8位三通道透视变换CUDA实现
 */
NppStatus nppiWarpPerspective_8u_C3R_Ctx_cuda(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
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

/**
 * 32位浮点单通道透视变换CUDA实现
 */
NppStatus nppiWarpPerspective_32f_C1R_Ctx_cuda(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
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

} // extern "C"