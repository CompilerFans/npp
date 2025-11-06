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

// Include interpolator implementations
#include "interpolators/bilinear_interpolator.cuh"
#include "interpolators/cubic_interpolator.cuh"
#include "interpolators/interpolator_common.cuh"
#include "interpolators/lanczos_interpolator.cuh"
#include "interpolators/nearest_interpolator.cuh"
#include "interpolators/super_interpolator.cuh"
#include "interpolators/super_v2_interpolator.cuh"

// Unified resize kernel template
template <typename T, int CHANNELS, typename Interpolator, bool PRESERVE_ALPHA = false>
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

  if constexpr (PRESERVE_ALPHA && CHANNELS == 4) {
    int sx = static_cast<int>(dx * scaleX + 0.5f) + oSrcRectROI.x;
    int sy = static_cast<int>(dy * scaleY + 0.5f) + oSrcRectROI.y;

    sx = min(max(sx, oSrcRectROI.x), oSrcRectROI.x + oSrcRectROI.width - 1);
    sy = min(max(sy, oSrcRectROI.y), oSrcRectROI.y + oSrcRectROI.height - 1);

    const T *src_row = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + sy * nSrcStep);
    T *dst_row = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + (dy + oDstRectROI.y) * nDstStep);

    int src_idx = sx * CHANNELS + (CHANNELS - 1);
    int dst_idx = (dx + oDstRectROI.x) * CHANNELS + (CHANNELS - 1);
    dst_row[dst_idx] = src_row[src_idx];
  }
}

// Wrapper function template
template <typename T, int CHANNELS, bool PRESERVE_ALPHA = false>
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

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                (oDstRectROI.height + blockSize.y - 1) / blockSize.y);
  LOG_DEBUG("Grid config: blocks=(%d, %d), threads=(%d, %d), total_threads=%d", gridSize.x, gridSize.y, blockSize.x,
            blockSize.y, gridSize.x * gridSize.y * blockSize.x * blockSize.y);

  // Dispatch to appropriate kernel based on interpolation mode
  switch (eInterpolation) {
  case 1: // NPPI_INTER_NN
    resizeKernel<T, CHANNELS, NearestInterpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 2: // NPPI_INTER_LINEAR
    resizeKernel<T, CHANNELS, BilinearInterpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 4: // NPPI_INTER_CUBIC
    resizeKernel<T, CHANNELS, CubicInterpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 8: // NPPI_INTER_SUPER
#if USE_SUPER_V2
    resizeKernel<T, CHANNELS, SuperSamplingV2Interpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep,
                                                           oDstSize, oDstRectROI);
#else
    resizeKernel<T, CHANNELS, SuperSamplingInterpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#endif
    break;
  case 16: // NPPI_INTER_LANCZOS
    resizeKernel<T, CHANNELS, LanczosInterpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  default:
    // Default to linear
    resizeKernel<T, CHANNELS, BilinearInterpolator<T, CHANNELS>, PRESERVE_ALPHA>
        <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
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

// Explicit instantiations for all supported types
template <typename T, int CHANNELS>
NppStatus resizePlanarImpl(const T *const pSrc[CHANNELS], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                           T *const pDst[CHANNELS], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                           int eInterpolation, NppStreamContext nppStreamCtx) {
  for (int c = 0; c < CHANNELS; ++c) {
    if (!pSrc[c] || !pDst[c]) {
      LOG_ERROR("Planar resize received null pointer at channel %d", c);
      return NPP_NULL_POINTER_ERROR;
    }

    NppStatus status = resizeImpl<T, 1>(pSrc[c], nSrcStep, oSrcSize, oSrcRectROI, pDst[c], nDstStep, oDstSize,
                                        oDstRectROI, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

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
  return resizeImpl<Npp8u, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                              eInterpolation, nppStreamCtx);
}

// 8u AC4R (alpha preserved)
NppStatus nppiResize_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_8u_AC4R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp8u, 4, true>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

// 8u P3R
NppStatus nppiResize_8u_P3R_Ctx_impl(const Npp8u *const pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *const pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_8u_P3R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizePlanarImpl<Npp8u, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

// 8u P4R
NppStatus nppiResize_8u_P4R_Ctx_impl(const Npp8u *const pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *const pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_8u_P4R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizePlanarImpl<Npp8u, 4>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
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

// 16u P3R
NppStatus nppiResize_16u_P3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, NppiSize oSrcSize,
                                      NppiRect oSrcRectROI, Npp16u *const pDst[3], int nDstStep, NppiSize oDstSize,
                                      NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_16u_P3R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizePlanarImpl<Npp16u, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

// 16u P4R
NppStatus nppiResize_16u_P4R_Ctx_impl(const Npp16u *const pSrc[4], int nSrcStep, NppiSize oSrcSize,
                                      NppiRect oSrcRectROI, Npp16u *const pDst[4], int nDstStep, NppiSize oDstSize,
                                      NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_16u_P4R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizePlanarImpl<Npp16u, 4>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

// 16u AC4R (alpha preserved)
NppStatus nppiResize_16u_AC4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                       Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                       int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_16u_AC4R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp16u, 4, true>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

// 32f AC4R (alpha preserved)
NppStatus nppiResize_32f_AC4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                       Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                       int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_32f_AC4R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizeImpl<Npp32f, 4, true>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

// 32f P3R
NppStatus nppiResize_32f_P3R_Ctx_impl(const Npp32f *const pSrc[3], int nSrcStep, NppiSize oSrcSize,
                                      NppiRect oSrcRectROI, Npp32f *const pDst[3], int nDstStep, NppiSize oDstSize,
                                      NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_32f_P3R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizePlanarImpl<Npp32f, 3>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

// 32f P4R
NppStatus nppiResize_32f_P4R_Ctx_impl(const Npp32f *const pSrc[4], int nSrcStep, NppiSize oSrcSize,
                                      NppiRect oSrcRectROI, Npp32f *const pDst[4], int nDstStep, NppiSize oDstSize,
                                      NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
  LOG_INFO("nppiResize_32f_P4R called: %dx%d -> %dx%d", oSrcRectROI.width, oSrcRectROI.height, oDstRectROI.width,
           oDstRectROI.height);
  return resizePlanarImpl<Npp32f, 4>(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

} // extern "C"
