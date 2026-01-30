#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline Npp8u clamp_to_8u(float v) {
  if (v < 0.0f) {
    v = 0.0f;
  } else if (v > 255.0f) {
    v = 255.0f;
  }
  return static_cast<Npp8u>(v + 0.5f);
}

__global__ void color_to_gray_8u_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                           int height, float3 coeffs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_pixel = pSrc + y * nSrcStep + x * 3;
    Npp8u *dst_pixel = pDst + y * nDstStep + x;

    float gray = coeffs.x * src_pixel[0] + coeffs.y * src_pixel[1] + coeffs.z * src_pixel[2];
    *dst_pixel = clamp_to_8u(gray);
  }
}

__global__ void color_to_gray_8u_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                            int height, float3 coeffs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_pixel = pSrc + y * nSrcStep + x * 4;
    Npp8u *dst_pixel = pDst + y * nDstStep + x;

    float gray = coeffs.x * src_pixel[0] + coeffs.y * src_pixel[1] + coeffs.z * src_pixel[2];
    *dst_pixel = clamp_to_8u(gray);
  }
}

__global__ void color_to_gray_8u_c4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                           int height, float4 coeffs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_pixel = pSrc + y * nSrcStep + x * 4;
    Npp8u *dst_pixel = pDst + y * nDstStep + x;

    float gray = coeffs.x * src_pixel[0] + coeffs.y * src_pixel[1] + coeffs.z * src_pixel[2] +
                 coeffs.w * src_pixel[3];
    *dst_pixel = clamp_to_8u(gray);
  }
}

extern "C" {
NppStatus nppiColorToGray_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                            NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  float3 coeffs = make_float3(aCoeffs[0], aCoeffs[1], aCoeffs[2]);
  color_to_gray_8u_c3_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                               oSizeROI.width, oSizeROI.height,
                                                                               coeffs);

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

NppStatus nppiColorToGray_8u_AC4C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, const Npp32f aCoeffs[3],
                                             NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  float3 coeffs = make_float3(aCoeffs[0], aCoeffs[1], aCoeffs[2]);
  color_to_gray_8u_ac4_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                oSizeROI.width, oSizeROI.height,
                                                                                coeffs);

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

NppStatus nppiColorToGray_8u_C4C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aCoeffs[4],
                                            NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  float4 coeffs = make_float4(aCoeffs[0], aCoeffs[1], aCoeffs[2], aCoeffs[3]);
  color_to_gray_8u_c4_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                               oSizeROI.width, oSizeROI.height,
                                                                               coeffs);

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
