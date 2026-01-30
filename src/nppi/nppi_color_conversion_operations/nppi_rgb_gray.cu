#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ITU-R BT.709 standard luminance weights
#define WEIGHT_R 0.299f
#define WEIGHT_G 0.587f
#define WEIGHT_B 0.114f

__device__ inline int round_to_int(float value) {
  return static_cast<int>(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

__device__ inline Npp16u clamp_u16(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > 65535) {
    return 65535;
  }
  return static_cast<Npp16u>(value);
}

__device__ inline Npp16s clamp_s16(int value) {
  if (value < -32768) {
    return -32768;
  }
  if (value > 32767) {
    return 32767;
  }
  return static_cast<Npp16s>(value);
}
__global__ void nppiRGBToGray_8u_C3C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                              int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate byte offsets correctly
    const Npp8u *src_pixel = pSrc + y * nSrcStep + x * 3;
    Npp8u *dst_pixel = pDst + y * nDstStep + x;

    // Get RGB values
    Npp8u r = src_pixel[0];
    Npp8u g = src_pixel[1];
    Npp8u b = src_pixel[2];

    // Calculate grayscale using ITU-R BT.709 weights
    float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    *dst_pixel = (Npp8u)(gray + 0.5f); // Round to nearest
  }
}

__global__ void nppiRGBToGray_16u_C3C1R_kernel(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, int width,
                                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src_pixel = (const Npp16u *)((const char *)pSrc + y * nSrcStep) + x * 3;
    Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;

    float r = static_cast<float>(src_pixel[0]);
    float g = static_cast<float>(src_pixel[1]);
    float b = static_cast<float>(src_pixel[2]);

    float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    int rounded = round_to_int(gray);
    *dst_pixel = clamp_u16(rounded);
  }
}

__global__ void nppiRGBToGray_8u_AC4C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate byte offsets correctly
    const Npp8u *src_pixel = pSrc + y * nSrcStep + x * 4;
    Npp8u *dst_pixel = pDst + y * nDstStep + x;

    // Get RGB values (ignore alpha channel)
    Npp8u r = src_pixel[0];
    Npp8u g = src_pixel[1];
    Npp8u b = src_pixel[2];
    // Alpha channel at index 3 is ignored

    // Calculate grayscale using ITU-R BT.709 weights
    float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    *dst_pixel = (Npp8u)(gray + 0.5f); // Round to nearest
  }
}

__global__ void nppiRGBToGray_16u_AC4C1R_kernel(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src_pixel = (const Npp16u *)((const char *)pSrc + y * nSrcStep) + x * 4;
    Npp16u *dst_pixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;

    float r = static_cast<float>(src_pixel[0]);
    float g = static_cast<float>(src_pixel[1]);
    float b = static_cast<float>(src_pixel[2]);

    float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    int rounded = round_to_int(gray);
    *dst_pixel = clamp_u16(rounded);
  }
}

__global__ void nppiRGBToGray_16s_C3C1R_kernel(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, int width,
                                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src_pixel = (const Npp16s *)((const char *)pSrc + y * nSrcStep) + x * 3;
    Npp16s *dst_pixel = (Npp16s *)((char *)pDst + y * nDstStep) + x;

    float r = static_cast<float>(src_pixel[0]);
    float g = static_cast<float>(src_pixel[1]);
    float b = static_cast<float>(src_pixel[2]);

    float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    int rounded = round_to_int(gray);
    *dst_pixel = clamp_s16(rounded);
  }
}

__global__ void nppiRGBToGray_16s_AC4C1R_kernel(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src_pixel = (const Npp16s *)((const char *)pSrc + y * nSrcStep) + x * 4;
    Npp16s *dst_pixel = (Npp16s *)((char *)pDst + y * nDstStep) + x;

    float r = static_cast<float>(src_pixel[0]);
    float g = static_cast<float>(src_pixel[1]);
    float b = static_cast<float>(src_pixel[2]);

    float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    int rounded = round_to_int(gray);
    *dst_pixel = clamp_s16(rounded);
  }
}

__global__ void nppiRGBToGray_32f_C3C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate byte offsets correctly for 32-bit float
    const Npp32f *src_pixel = (const Npp32f *)((const char *)pSrc + y * nSrcStep) + x * 3;
    Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;

    // Get RGB values
    Npp32f r = src_pixel[0];
    Npp32f g = src_pixel[1];
    Npp32f b = src_pixel[2];

    // Calculate grayscale using ITU-R BT.709 weights
    *dst_pixel = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
  }
}

__global__ void nppiRGBToGray_32f_AC4C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate byte offsets correctly for 32-bit float
    const Npp32f *src_pixel = (const Npp32f *)((const char *)pSrc + y * nSrcStep) + x * 4;
    Npp32f *dst_pixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;

    // Get RGB values (ignore alpha channel)
    Npp32f r = src_pixel[0];
    Npp32f g = src_pixel[1];
    Npp32f b = src_pixel[2];
    // Alpha channel at index 3 is ignored

    // Calculate grayscale using ITU-R BT.709 weights
    *dst_pixel = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
  }
}

extern "C" {
NppStatus nppiRGBToGray_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_8u_C3C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                  oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Synchronize to ensure kernel completion
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

NppStatus nppiRGBToGray_16u_C3C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_16u_C3C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);

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

NppStatus nppiRGBToGray_8u_AC4C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_8u_AC4C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Synchronize to ensure kernel completion
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

NppStatus nppiRGBToGray_16u_AC4C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_16u_AC4C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                    oSizeROI.width, oSizeROI.height);

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

NppStatus nppiRGBToGray_16s_C3C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_16s_C3C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);

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

NppStatus nppiRGBToGray_16s_AC4C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_16s_AC4C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                    oSizeROI.width, oSizeROI.height);

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

NppStatus nppiRGBToGray_32f_C3C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_32f_C3C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Synchronize to ensure kernel completion
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

NppStatus nppiRGBToGray_32f_AC4C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToGray_32f_AC4C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                    oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Synchronize to ensure kernel completion
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
