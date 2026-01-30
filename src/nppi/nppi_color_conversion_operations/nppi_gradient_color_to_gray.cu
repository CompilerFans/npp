#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

namespace {

constexpr float kInvSqrt3 = 0.5773502691896257f;

template <typename DstT>
__device__ inline DstT convertValue(float value) {
  if constexpr (std::is_same_v<DstT, Npp8u>) {
    if (value < 0.0f) {
      value = 0.0f;
    } else if (value > 255.0f) {
      value = 255.0f;
    }
    return static_cast<Npp8u>(value);
  } else if constexpr (std::is_same_v<DstT, Npp16u>) {
    if (value < 0.0f) {
      value = 0.0f;
    } else if (value > 65535.0f) {
      value = 65535.0f;
    }
    return static_cast<Npp16u>(value);
  } else if constexpr (std::is_same_v<DstT, Npp16s>) {
    if (value < -32768.0f) {
      value = -32768.0f;
    } else if (value > 32767.0f) {
      value = 32767.0f;
    }
    return static_cast<Npp16s>(value);
  } else {
    return static_cast<Npp32f>(value);
  }
}

template <typename SrcT, typename DstT, bool kScale>
__global__ void nppiGradientColorToGray_C3C1R_kernel(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep,
                                                     int width, int height, NppiNorm eNorm) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const SrcT *srcPixel = reinterpret_cast<const SrcT *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep) + x * 3;
  float r = static_cast<float>(srcPixel[0]);
  float g = static_cast<float>(srcPixel[1]);
  float b = static_cast<float>(srcPixel[2]);

  float value = 0.0f;
  switch (eNorm) {
  case nppiNormL1:
    value = r + g + b;
    if constexpr (kScale) {
      value *= (1.0f / 3.0f);
    }
    break;
  case nppiNormL2:
    value = sqrtf(r * r + g * g + b * b);
    if constexpr (kScale) {
      value *= kInvSqrt3;
    }
    break;
  case nppiNormInf:
  default:
    value = fmaxf(r, fmaxf(g, b));
    break;
  }

  DstT *dstPixel = reinterpret_cast<DstT *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x;
  *dstPixel = convertValue<DstT>(value);
}

template <typename SrcT, typename DstT, bool kScale>
NppStatus nppiGradientColorToGray_C3C1R_Ctx_impl(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep,
                                                 NppiSize oSizeROI, NppiNorm eNorm,
                                                 NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiGradientColorToGray_C3C1R_kernel<SrcT, DstT, kScale><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eNorm);

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

} // namespace

extern "C" {
NppStatus nppiGradientColorToGray_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                    NppiSize oSizeROI, NppiNorm eNorm,
                                                    NppStreamContext nppStreamCtx) {
  return nppiGradientColorToGray_C3C1R_Ctx_impl<Npp8u, Npp8u, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm,
                                                                   nppStreamCtx);
}

NppStatus nppiGradientColorToGray_16u_C3C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                                     NppiSize oSizeROI, NppiNorm eNorm,
                                                     NppStreamContext nppStreamCtx) {
  return nppiGradientColorToGray_C3C1R_Ctx_impl<Npp16u, Npp16u, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm,
                                                                     nppStreamCtx);
}

NppStatus nppiGradientColorToGray_16s_C3C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                                     NppiSize oSizeROI, NppiNorm eNorm,
                                                     NppStreamContext nppStreamCtx) {
  return nppiGradientColorToGray_C3C1R_Ctx_impl<Npp16s, Npp16s, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm,
                                                                     nppStreamCtx);
}

NppStatus nppiGradientColorToGray_32f_C3C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                     NppiSize oSizeROI, NppiNorm eNorm,
                                                     NppStreamContext nppStreamCtx) {
  return nppiGradientColorToGray_C3C1R_Ctx_impl<Npp32f, Npp32f, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm,
                                                                      nppStreamCtx);
}
}
