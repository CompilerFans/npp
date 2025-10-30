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

  dim3 blockSize(16, 16);
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
