#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Device function for threshold operation

template <typename T> __device__ inline T performThreshold(T src, T threshold, NppCmpOp op) {
  switch (op) {
  case NPP_CMP_LESS:
    return (src < threshold) ? threshold : src;
  case NPP_CMP_GREATER:
    return (src > threshold) ? threshold : src;
  default:
    return src;
  }
}

// Kernel for 8-bit unsigned single channel threshold (non-inplace)
__global__ void nppiThreshold_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                            int height, Npp8u nThreshold, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    dst_row[x] = performThreshold(src_row[x], nThreshold, eComparisonOperation);
  }
}

// Kernel for 8-bit unsigned single channel threshold (inplace)
__global__ void nppiThreshold_8u_C1IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height, Npp8u nThreshold,
                                             NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    row[x] = performThreshold(row[x], nThreshold, eComparisonOperation);
  }
}

// Kernel for 32-bit float single channel threshold (non-inplace)
__global__ void nppiThreshold_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                             int height, Npp32f nThreshold, NppCmpOp eComparisonOperation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    dst_row[x] = performThreshold(src_row[x], nThreshold, eComparisonOperation);
  }
}

// Kernel for 8-bit threshold_ltval (non-inplace)
__global__ void nppiThreshold_LTVal_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                   int width, int height, Npp8u nThreshold, Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    dst_row[x] = (src_row[x] < nThreshold) ? nValue : src_row[x];
  }
}

// Kernel for 8-bit threshold_ltval (inplace)
__global__ void nppiThreshold_LTVal_8u_C1IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height,
                                                    Npp8u nThreshold, Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    row[x] = (row[x] < nThreshold) ? nValue : row[x];
  }
}

// Kernel for 8-bit threshold_gtval (non-inplace)
__global__ void nppiThreshold_GTVal_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                   int width, int height, Npp8u nThreshold, Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    dst_row[x] = (src_row[x] > nThreshold) ? nValue : src_row[x];
  }
}

// Kernel for 8-bit threshold_gtval (inplace)
__global__ void nppiThreshold_GTVal_8u_C1IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height,
                                                    Npp8u nThreshold, Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    row[x] = (row[x] > nThreshold) ? nValue : row[x];
  }
}

// Kernel for 8-bit threshold_ltvalgtval (non-inplace)
__global__ void nppiThreshold_LTValGTVal_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                        int width, int height, Npp8u nThresholdLT, Npp8u nValueLT,
                                                        Npp8u nThresholdGT, Npp8u nValueGT) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    Npp8u val = src_row[x];
    if (val < nThresholdLT) {
      dst_row[x] = nValueLT;
    } else if (val > nThresholdGT) {
      dst_row[x] = nValueGT;
    } else {
      dst_row[x] = val;
    }
  }
}

// Kernel for 32-bit float threshold_ltval (non-inplace)
__global__ void nppiThreshold_LTVal_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                    int width, int height, Npp32f nThreshold, Npp32f nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    dst_row[x] = (src_row[x] < nThreshold) ? nValue : src_row[x];
  }
}

// C3 Threshold_LTVal kernels
__global__ void nppiThreshold_LTVal_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                   int width, int height, const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    int idx = x * 3;

    for (int c = 0; c < 3; c++) {
      dst_row[idx + c] = (src_row[idx + c] < thresholds[c]) ? values[c] : src_row[idx + c];
    }
  }
}

__global__ void nppiThreshold_LTVal_8u_C3IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height,
                                                    const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    int idx = x * 3;

    for (int c = 0; c < 3; c++) {
      row[idx + c] = (row[idx + c] < thresholds[c]) ? values[c] : row[idx + c];
    }
  }
}

// C4 Threshold_LTVal kernels
__global__ void nppiThreshold_LTVal_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                   int width, int height, const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    int idx = x * 4;

    for (int c = 0; c < 4; c++) {
      dst_row[idx + c] = (src_row[idx + c] < thresholds[c]) ? values[c] : src_row[idx + c];
    }
  }
}

__global__ void nppiThreshold_LTVal_8u_C4IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height,
                                                    const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    int idx = x * 4;

    for (int c = 0; c < 4; c++) {
      row[idx + c] = (row[idx + c] < thresholds[c]) ? values[c] : row[idx + c];
    }
  }
}

// C3 Threshold_GTVal kernels
__global__ void nppiThreshold_GTVal_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                   int width, int height, const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    int idx = x * 3;

    for (int c = 0; c < 3; c++) {
      dst_row[idx + c] = (src_row[idx + c] > thresholds[c]) ? values[c] : src_row[idx + c];
    }
  }
}

__global__ void nppiThreshold_GTVal_8u_C3IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height,
                                                    const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    int idx = x * 3;

    for (int c = 0; c < 3; c++) {
      row[idx + c] = (row[idx + c] > thresholds[c]) ? values[c] : row[idx + c];
    }
  }
}

// C4 Threshold_GTVal kernels
__global__ void nppiThreshold_GTVal_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                   int width, int height, const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    int idx = x * 4;

    for (int c = 0; c < 4; c++) {
      dst_row[idx + c] = (src_row[idx + c] > thresholds[c]) ? values[c] : src_row[idx + c];
    }
  }
}

__global__ void nppiThreshold_GTVal_8u_C4IR_kernel(Npp8u *pSrcDst, int nSrcDstStep, int width, int height,
                                                    const Npp8u *thresholds, const Npp8u *values) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *row = (Npp8u *)((char *)pSrcDst + y * nSrcDstStep);
    int idx = x * 4;

    for (int c = 0; c < 4; c++) {
      row[idx + c] = (row[idx + c] > thresholds[c]) ? values[c] : row[idx + c];
    }
  }
}

extern "C" {

// 8-bit unsigned single channel threshold implementation (non-inplace)
NppStatus nppiThreshold_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        const Npp8u nThreshold, NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel threshold implementation (inplace)
NppStatus nppiThreshold_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp8u nThreshold,
                                         NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_8u_C1IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, nThreshold, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel threshold implementation (non-inplace)
NppStatus nppiThreshold_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, const Npp32f nThreshold, NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, eComparisonOperation);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Threshold_LTVal 8u C1R implementation (non-inplace)
NppStatus nppiThreshold_LTVal_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue,
                                               NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Threshold_LTVal 8u C1IR implementation (inplace)
NppStatus nppiThreshold_LTVal_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                Npp8u nThreshold, Npp8u nValue, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_8u_C1IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, nThreshold, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Threshold_GTVal 8u C1R implementation (non-inplace)
NppStatus nppiThreshold_GTVal_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, Npp8u nThreshold, Npp8u nValue,
                                               NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_GTVal_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Threshold_GTVal 8u C1IR implementation (inplace)
NppStatus nppiThreshold_GTVal_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                Npp8u nThreshold, Npp8u nValue, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_GTVal_8u_C1IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, nThreshold, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Threshold_LTValGTVal 8u C1R implementation (non-inplace)
NppStatus nppiThreshold_LTValGTVal_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                    NppiSize oSizeROI, Npp8u nThresholdLT, Npp8u nValueLT,
                                                    Npp8u nThresholdGT, Npp8u nValueGT, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTValGTVal_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThresholdLT, nValueLT, nThresholdGT, nValueGT);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Threshold_LTVal 32f C1R implementation (non-inplace)
NppStatus nppiThreshold_LTVal_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                NppiSize oSizeROI, Npp32f nThreshold, Npp32f nValue,
                                                NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nThreshold, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// C3 Threshold_LTVal implementations
NppStatus nppiThreshold_LTVal_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 3 * sizeof(Npp8u));
  cudaMalloc(&d_values, 3 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiThreshold_LTVal_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 3 * sizeof(Npp8u));
  cudaMalloc(&d_values, 3 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_8u_C3IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// C4 Threshold_LTVal implementations
NppStatus nppiThreshold_LTVal_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 4 * sizeof(Npp8u));
  cudaMalloc(&d_values, 4 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiThreshold_LTVal_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 4 * sizeof(Npp8u));
  cudaMalloc(&d_values, 4 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_LTVal_8u_C4IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// C3 Threshold_GTVal implementations
NppStatus nppiThreshold_GTVal_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 3 * sizeof(Npp8u));
  cudaMalloc(&d_values, 3 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_GTVal_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiThreshold_GTVal_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 3 * sizeof(Npp8u));
  cudaMalloc(&d_values, 3 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_GTVal_8u_C3IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// C4 Threshold_GTVal implementations
NppStatus nppiThreshold_GTVal_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp8u *pThresholds, const Npp8u *pValues,
                                               NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 4 * sizeof(Npp8u));
  cudaMalloc(&d_values, 4 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_GTVal_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiThreshold_GTVal_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                                const Npp8u *pThresholds, const Npp8u *pValues,
                                                NppStreamContext nppStreamCtx) {
  Npp8u *d_thresholds, *d_values;
  cudaMalloc(&d_thresholds, 4 * sizeof(Npp8u));
  cudaMalloc(&d_values, 4 * sizeof(Npp8u));
  cudaMemcpyAsync(d_thresholds, pThresholds, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  cudaMemcpyAsync(d_values, pValues, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiThreshold_GTVal_8u_C4IR_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, d_thresholds, d_values);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_thresholds);
  cudaFree(d_values);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
