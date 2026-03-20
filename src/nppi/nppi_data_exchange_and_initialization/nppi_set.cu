#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__global__ void nppiSet_C1R_kernel(T nValue, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  dstRow[x] = nValue;
}

template <typename T, int CHANNELS>
__global__ void nppiSet_CxR_kernel(const T *aValue, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  int idx = x * CHANNELS;

#pragma unroll
  for (int c = 0; c < CHANNELS; ++c) {
    dstRow[idx + c] = aValue[c];
  }
}

template <typename T>
NppStatus launchSetScalar(T nValue, T *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSet_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(nValue, pDst, nDstStep, oSizeROI.width,
                                                                        oSizeROI.height);

  return cudaGetLastError() == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T, int CHANNELS>
NppStatus launchSetVector(const T *aValue, T *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  T *dValue = nullptr;
  cudaError_t status = cudaMalloc(&dValue, CHANNELS * sizeof(T));
  if (status != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  status = cudaMemcpyAsync(dValue, aValue, CHANNELS * sizeof(T), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (status != cudaSuccess) {
    cudaFree(dValue);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSet_CxR_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(dValue, pDst, nDstStep,
                                                                                      oSizeROI.width, oSizeROI.height);

  status = cudaGetLastError();
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(dValue);

  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

extern "C" {

NppStatus nppiSet_8u_C1R_Ctx_impl(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_8u_C3R_Ctx_impl(const Npp8u aValue[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp8u, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_8u_C4R_Ctx_impl(const Npp8u aValue[4], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp8u, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16u_C1R_Ctx_impl(Npp16u nValue, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16u_C3R_Ctx_impl(const Npp16u aValue[3], Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16u, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16u_C4R_Ctx_impl(const Npp16u aValue[4], Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16u, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16s_C1R_Ctx_impl(Npp16s nValue, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16s_C3R_Ctx_impl(const Npp16s aValue[3], Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16s, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16s_C4R_Ctx_impl(const Npp16s aValue[4], Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16s, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C1R_Ctx_impl(Npp32f nValue, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C3R_Ctx_impl(const Npp32f aValue[3], Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp32f, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C4R_Ctx_impl(const Npp32f aValue[4], Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp32f, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

} // extern "C"
