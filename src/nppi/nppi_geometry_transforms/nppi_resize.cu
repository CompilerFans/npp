#include "../../nppcore/logger.h"
#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Control which super sampling algorithm to use
// 0: Use SuperSamplingInterpolator (simple box filter, compatible with existing tests)
// 1: Use SuperSamplingV2Interpolator (weighted box filter, kunzmi/mpp algorithm, higher precision)
#ifndef USE_SUPER_V2
#define USE_SUPER_V2 1
#endif

// Control which linear interpolation algorithm to use
// 0: Use BilinearInterpolator (standard bilinear)
// 1: Use BilinearV2Interpolator (NPP-compatible algorithm, higher precision)
#ifndef USE_LINEAR_V2
#define USE_LINEAR_V2 1
#endif

// Include interpolator implementations
#include "interpolators/bilinear_interpolator.cuh"
#include "interpolators/bilinear_v2_interpolator.cuh"
#include "interpolators/cubic_interpolator.cuh"
#include "interpolators/interpolator_common.cuh"
#include "interpolators/lanczos_interpolator.cuh"
#include "interpolators/nearest_interpolator.cuh"
#include "interpolators/super_interpolator.cuh"
#include "interpolators/super_v2_interpolator.cuh"

__device__ inline float clampCoord(float value, float minVal, float maxVal) {
  return fminf(fmaxf(value, minVal), maxVal);
}

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
  LOG_DEBUG("=== nppiResize START ===");
  LOG_DEBUG("Source: size=(%d, %d), ROI=(%d, %d, %d, %d), step=%d bytes", oSrcSize.width, oSrcSize.height,
            oSrcRectROI.x, oSrcRectROI.y, oSrcRectROI.width, oSrcRectROI.height, nSrcStep);
  LOG_DEBUG("Destination: size=(%d, %d), ROI=(%d, %d, %d, %d), step=%d bytes", oDstSize.width, oDstSize.height,
            oDstRectROI.x, oDstRectROI.y, oDstRectROI.width, oDstRectROI.height, nDstStep);

  const char *interpMode = "UNKNOWN";
  switch (eInterpolation) {
  case 1:
    interpMode = "NEAREST_NEIGHBOR";
    break;
  case 2:
    interpMode = "LINEAR";
    break;
  case 4:
    interpMode = "CUBIC";
    break;
  case 8:
    interpMode = "SUPER";
    break;
  case 16:
    interpMode = "LANCZOS";
    break;
  }
  LOG_DEBUG("Interpolation mode: %s (%d), Channels: %d", interpMode, eInterpolation, CHANNELS);

  float scaleX = (float)oSrcRectROI.width / (float)oDstRectROI.width;
  float scaleY = (float)oSrcRectROI.height / (float)oDstRectROI.height;
  LOG_DEBUG("Scale factors: X=%.4f, Y=%.4f", scaleX, scaleY);

  dim3 blockSize(32, 8);
  dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                (oDstRectROI.height + blockSize.y - 1) / blockSize.y);
  LOG_DEBUG("Grid config: blocks=(%d, %d), threads=(%d, %d), total_threads=%d", gridSize.x, gridSize.y, blockSize.x,
            blockSize.y, gridSize.x * gridSize.y * blockSize.x * blockSize.y);

  // Dispatch to appropriate kernel based on interpolation mode
  switch (eInterpolation) {
  case 1: // NPPI_INTER_NN
    resizeKernel<T, CHANNELS, NearestInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 2: // NPPI_INTER_LINEAR
#if USE_LINEAR_V2
    resizeKernel<T, CHANNELS, BilinearV2Interpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#else
    resizeKernel<T, CHANNELS, BilinearInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#endif
    break;
  case 4: // NPPI_INTER_CUBIC
    resizeKernel<T, CHANNELS, CubicInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 8: // NPPI_INTER_SUPER
#if USE_SUPER_V2
    resizeKernel<T, CHANNELS, SuperSamplingV2Interpolator<T, CHANNELS>>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep,
                                                           oDstSize, oDstRectROI);
#else
    resizeKernel<T, CHANNELS, SuperSamplingInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#endif
    break;
  case 16: // NPPI_INTER_LANCZOS
    resizeKernel<T, CHANNELS, LanczosInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  default:
    // Default to linear
#if USE_LINEAR_V2
    resizeKernel<T, CHANNELS, BilinearV2Interpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#else
    resizeKernel<T, CHANNELS, BilinearInterpolator<T, CHANNELS>><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#endif
    break;
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    LOG_ERROR("Kernel launch failed: %s (%d)", cudaGetErrorString(cudaStatus), cudaStatus);
    LOG_DEBUG("=== nppiResize END (ERROR) ===");
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  LOG_DEBUG("Kernel launched successfully");
  LOG_DEBUG("=== nppiResize END (SUCCESS) ===");
  return NPP_SUCCESS;
}

__device__ __forceinline__ void process_pixel_channel(const Npp8u *src_row1, const Npp8u *src_row2, int x1, int x2,
                                                      float tx, float ty, int channel, Npp8u *dst_ptr) {
  float p11 = src_row1[x1 * 3 + channel];
  float p12 = src_row1[x2 * 3 + channel];
  float p21 = src_row2[x1 * 3 + channel];
  float p22 = src_row2[x2 * 3 + channel];

  float top = fmaf(tx, p12 - p11, p11);
  float bot = fmaf(tx, p22 - p21, p21);
  float v = fmaf(ty, bot - top, top);

  v = fminf(255.f, fmaxf(0.f, v + 0.5f));
  dst_ptr[channel] = (Npp8u)v;
}

__global__ void nppiResize_8u_C3R_linear_kernel_opt(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                    NppiRect oSrcRectROI, float scaleX, float scaleY, Npp8u *pDst,
                                                    int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= oDstRectROI.width || dy >= oDstRectROI.height)
    return;

  float fx = (dx + 0.5f) * scaleX - 0.5f + oSrcRectROI.x;
  float fy = (dy + 0.5f) * scaleY - 0.5f + oSrcRectROI.y;

  const float maxXf = (float)(oSrcRectROI.x + oSrcRectROI.width - 1);
  const float maxYf = (float)(oSrcRectROI.y + oSrcRectROI.height - 1);
  fx = fminf(maxXf, fmaxf((float)oSrcRectROI.x, fx));
  fy = fminf(maxYf, fmaxf((float)oSrcRectROI.y, fy));

  int x1 = (int)floorf(fx);
  int y1 = (int)floorf(fy);
  x1 = max(x1, oSrcRectROI.x);
  y1 = max(y1, oSrcRectROI.y);

  int x2 = min(x1 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
  int y2 = min(y1 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

  float tx = fx - (float)x1;
  float ty = fy - (float)y1;

  const Npp8u *src_row1 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep);
  const Npp8u *src_row2 = (const Npp8u *)((const char *)pSrc + y2 * nSrcStep);

  Npp8u *dst_ptr = (Npp8u *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep + (dx + oDstRectROI.x) * 3);

  process_pixel_channel(src_row1, src_row2, x1, x2, tx, ty, 0, dst_ptr);
  process_pixel_channel(src_row1, src_row2, x1, x2, tx, ty, 1, dst_ptr);
  process_pixel_channel(src_row1, src_row2, x1, x2, tx, ty, 2, dst_ptr);
}

// Explicit instantiations for all supported types
extern "C" {

// 8u C1R
NppStatus nppiResize_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_8u_C1R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp8u, 1>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                              eInterpolation, nppStreamCtx);
}

// 8u C3R
NppStatus nppiResize_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_8u_C3R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  if (eInterpolation == NPPI_INTER_LINEAR) {
    dim3 blockSize(32, 8);
    dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstRectROI.height + blockSize.y - 1) / blockSize.y);
    float scaleX = (float)oSrcRectROI.width / (float)oDstRectROI.width;
    float scaleY = (float)oSrcRectROI.height / (float)oDstRectROI.height;
    nppiResize_8u_C3R_linear_kernel_opt<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, scaleX, scaleY, pDst, nDstStep, oDstSize, oDstRectROI);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    return NPP_SUCCESS;
  }
  return resizeImpl<Npp8u, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                              eInterpolation, nppStreamCtx);
}

// 16u C1R
NppStatus nppiResize_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_16u_C1R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp16u, 1>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 32f C1R
NppStatus nppiResize_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_32f_C1R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp32f, 1>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 32f C3R
NppStatus nppiResize_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_32f_C3R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp32f, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

} // extern "C"

template <typename T, int CHANNELS>
__device__ void sampleNearest(const T *pSrc, int nSrcStep, const NppiRect &oSrcROI, float sx, float sy, T *dstPixel) {
  int minX = oSrcROI.x;
  int minY = oSrcROI.y;
  int maxX = oSrcROI.x + oSrcROI.width - 1;
  int maxY = oSrcROI.y + oSrcROI.height - 1;

  int ix = static_cast<int>(floorf(sx + 0.5f));
  int iy = static_cast<int>(floorf(sy + 0.5f));

  ix = max(minX, min(ix, maxX));
  iy = max(minY, min(iy, maxY));

  const T *srcRow = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + iy * nSrcStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; ++c) {
    dstPixel[c] = srcRow[ix * CHANNELS + c];
  }
}

template <typename T, int CHANNELS>
__device__ void sampleBilinear(const T *pSrc, int nSrcStep, const NppiRect &oSrcROI, float sx, float sy, T *dstPixel) {
  int minX = oSrcROI.x;
  int minY = oSrcROI.y;
  int maxX = oSrcROI.x + oSrcROI.width - 1;
  int maxY = oSrcROI.y + oSrcROI.height - 1;

  sx = clampCoord(sx, static_cast<float>(minX), static_cast<float>(maxX));
  sy = clampCoord(sy, static_cast<float>(minY), static_cast<float>(maxY));

  int x1 = static_cast<int>(floorf(sx));
  int y1 = static_cast<int>(floorf(sy));

  x1 = max(minX, min(x1, maxX));
  y1 = max(minY, min(y1, maxY));

  int x2 = min(x1 + 1, maxX);
  int y2 = min(y1 + 1, maxY);

  float tx = sx - static_cast<float>(x1);
  float ty = sy - static_cast<float>(y1);

  const T *row1 = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + y1 * nSrcStep);
  const T *row2 = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + y2 * nSrcStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; ++c) {
    float p11 = static_cast<float>(row1[x1 * CHANNELS + c]);
    float p12 = static_cast<float>(row1[x2 * CHANNELS + c]);
    float p21 = static_cast<float>(row2[x1 * CHANNELS + c]);
    float p22 = static_cast<float>(row2[x2 * CHANNELS + c]);

    float top = lerp(p11, p12, tx);
    float bottom = lerp(p21, p22, tx);
    float value = lerp(top, bottom, ty);
    dstPixel[c] = static_cast<T>(value + 0.5f);
  }
}

template <typename T, int CHANNELS>
__global__ void resizeSqrPixelKernel(const T *pSrc, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                                     NppiRect oDstROI, float invScaleX, float invScaleY, float shiftX, float shiftY,
                                     int eInterpolation) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= oDstROI.width || dy >= oDstROI.height) {
    return;
  }

  int destX = dx + oDstROI.x;
  int destY = dy + oDstROI.y;

  float sx = (static_cast<float>(destX) + 0.5f) * invScaleX + shiftX + static_cast<float>(oSrcROI.x) - 0.5f;
  float sy = (static_cast<float>(destY) + 0.5f) * invScaleY + shiftY + static_cast<float>(oSrcROI.y) - 0.5f;

  float minX = static_cast<float>(oSrcROI.x);
  float minY = static_cast<float>(oSrcROI.y);
  float maxX = static_cast<float>(oSrcROI.x + oSrcROI.width - 1);
  float maxY = static_cast<float>(oSrcROI.y + oSrcROI.height - 1);

  sx = clampCoord(sx, minX, maxX);
  sy = clampCoord(sy, minY, maxY);

  T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + destY * nDstStep);
  T *dstPixel = dstRow + destX * CHANNELS;

  if (eInterpolation == NPPI_INTER_NN) {
    sampleNearest<T, CHANNELS>(pSrc, nSrcStep, oSrcROI, sx, sy, dstPixel);
  } else {
    sampleBilinear<T, CHANNELS>(pSrc, nSrcStep, oSrcROI, sx, sy, dstPixel);
  }
}

extern "C" {
NppStatus nppiResizeSqrPixel_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, double nXFactor,
                                             double nYFactor, double nXShift, double nYShift, int eInterpolation,
                                             NppStreamContext nppStreamCtx) {
  (void)oSrcSize;
  // printf("nppiResizeSqrPixel_8u_C1R: srcROI=%dx%d dstROI=%dx%d factors=(%.3f, %.3f) shifts=(%.3f, %.3f)\n",
  //          oSrcROI.width, oSrcROI.height, oDstROI.width, oDstROI.height, nXFactor, nYFactor, nXShift, nYShift);

  if (nXFactor <= 0.0 || nYFactor <= 0.0) {
    return NPP_RESIZE_FACTOR_ERROR;
  }

  if (eInterpolation != NPPI_INTER_NN && eInterpolation != NPPI_INTER_LINEAR) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  float invScaleX = 1.0f / static_cast<float>(nXFactor);
  float invScaleY = 1.0f / static_cast<float>(nYFactor);
  float shiftX = static_cast<float>(nXShift);
  float shiftY = static_cast<float>(nYShift);

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  resizeSqrPixelKernel<Npp8u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, invScaleX, invScaleY, shiftX, shiftY, eInterpolation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    printf("ResizeSqrPixel kernel launch failed: %s (%d)", cudaGetErrorString(cudaStatus), cudaStatus);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
