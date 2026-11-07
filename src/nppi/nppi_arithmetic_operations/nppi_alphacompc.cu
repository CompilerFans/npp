#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// AlphaComp Implementation
// nppiAlphaCompC: Alpha composition with constant alpha values
// ============================================================================

template <typename T> class AlphaCompOpExecutor {
private:
  T alpha1, alpha2;
  NppiAlphaOp alphaOp;

public:
  __device__ __host__ AlphaCompOpExecutor(T a1, T a2, NppiAlphaOp op): alpha1(a1), alpha2(a2), alphaOp(op) {}

  __device__ __host__ T operator()(T src1, T src2, int = 0) const {
    return alpha_ref::color<T>(alphaOp, src1, src2, alpha1, alpha2);
  }
};

// AC4 constant-alpha composition: only channels 0..2 are composed.  NPP
// leaves the destination alpha untouched for every operator except PREMUL,
// which writes the first constant alpha.
template <typename T>
__global__ void alphaCompConstAC4Kernel(const T *pSrc1, int nSrc1Step, T alpha1, const T *pSrc2, int nSrc2Step,
                                        T alpha2, T *pDst, int nDstStep, int width, int height,
                                        NppiAlphaOp alphaOp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
    const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    AlphaCompOpExecutor<T> op(alpha1, alpha2, alphaOp);
    const int idx = x * 4;
    for (int c = 0; c < 3; ++c) dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c]);
    if (alphaOp == NPPI_OP_ALPHA_PREMUL) dstRow[idx + 3] = alpha1;
  }
}

// Custom kernel for AlphaComp operation
template <typename T, int Channels>
__global__ void alphaCompConstKernel(const T *pSrc1, int nSrc1Step, T alpha1, const T *pSrc2, int nSrc2Step, T alpha2,
                                     T *pDst, int nDstStep, int width, int height, NppiAlphaOp alphaOp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
    const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    AlphaCompOpExecutor<T> op(alpha1, alpha2, alphaOp);

    if constexpr (Channels == 1) {
      dstRow[x] = op(src1Row[x], src2Row[x]);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c]);
      }
    }
  }
}

extern "C" {

// 8u versions
NppStatus nppiAlphaCompC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                    Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check if the alpha operation is supported (now supports all operations)
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp8u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiAlphaCompC_8u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                   eAlphaOp, defaultCtx);
}

// 8u C3 versions
NppStatus nppiAlphaCompC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                    Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check if the alpha operation is supported (now supports all operations)
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp8u, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiAlphaCompC_8u_C3R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                   eAlphaOp, defaultCtx);
}

// 16u versions
NppStatus nppiAlphaCompC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2,
                                     int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check if the alpha operation is supported (now supports all operations)
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp16u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiAlphaCompC_16u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 32f versions
NppStatus nppiAlphaCompC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, Npp32f nAlpha1, const Npp32f *pSrc2,
                                     int nSrc2Step, Npp32f nAlpha2, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check if the alpha operation is supported (now supports all operations)
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp32f, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, Npp32f nAlpha1, const Npp32f *pSrc2, int nSrc2Step,
                                 Npp32f nAlpha2, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)
  return nppiAlphaCompC_32f_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 8u C4 versions
NppStatus nppiAlphaCompC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                    Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp8u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_8u_C4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                   eAlphaOp, defaultCtx);
}

// 8u AC4 versions
NppStatus nppiAlphaCompC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2,
                                     int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstAC4Kernel<Npp8u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_8u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 8s C1 versions
NppStatus nppiAlphaCompC_8s_C1R_Ctx(const Npp8s *pSrc1, int nSrc1Step, Npp8s nAlpha1, const Npp8s *pSrc2, int nSrc2Step,
                                    Npp8s nAlpha2, Npp8s *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp8s, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_8s_C1R(const Npp8s *pSrc1, int nSrc1Step, Npp8s nAlpha1, const Npp8s *pSrc2, int nSrc2Step,
                                Npp8s nAlpha2, Npp8s *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_8s_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                   eAlphaOp, defaultCtx);
}

// 16u C3 versions
NppStatus nppiAlphaCompC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2,
                                     int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp16u, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_16u_C3R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 16u C4 versions
NppStatus nppiAlphaCompC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2,
                                     int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp16u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_16u_C4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 16u AC4 versions
NppStatus nppiAlphaCompC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2,
                                      int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstAC4Kernel<Npp16u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u *pSrc2,
                                  int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_16u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                     eAlphaOp, defaultCtx);
}

// 16s C1 versions
NppStatus nppiAlphaCompC_16s_C1R_Ctx(const Npp16s *pSrc1, int nSrc1Step, Npp16s nAlpha1, const Npp16s *pSrc2,
                                     int nSrc2Step, Npp16s nAlpha2, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp16s, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_16s_C1R(const Npp16s *pSrc1, int nSrc1Step, Npp16s nAlpha1, const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s nAlpha2, Npp16s *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_16s_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 32u C1 versions
NppStatus nppiAlphaCompC_32u_C1R_Ctx(const Npp32u *pSrc1, int nSrc1Step, Npp32u nAlpha1, const Npp32u *pSrc2,
                                     int nSrc2Step, Npp32u nAlpha2, Npp32u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp32u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_32u_C1R(const Npp32u *pSrc1, int nSrc1Step, Npp32u nAlpha1, const Npp32u *pSrc2, int nSrc2Step,
                                 Npp32u nAlpha2, Npp32u *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_32u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

// 32s C1 versions
NppStatus nppiAlphaCompC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step, Npp32s nAlpha1, const Npp32s *pSrc2,
                                     int nSrc2Step, Npp32s nAlpha2, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx) {
  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (eAlphaOp < NPPI_OP_ALPHA_OVER || eAlphaOp > NPPI_OP_ALPHA_PREMUL) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  alphaCompConstKernel<Npp32s, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eAlphaOp);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaCompC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step, Npp32s nAlpha1, const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s nAlpha2, Npp32s *pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = nppGetStream();
  return nppiAlphaCompC_32s_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pSrc2, nSrc2Step, nAlpha2, pDst, nDstStep, oSizeROI,
                                    eAlphaOp, defaultCtx);
}

} // extern "C"