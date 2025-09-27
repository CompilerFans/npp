#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// AddWeighted Implementation
// nppiAddWeighted: result = src * alpha + dst * (1 - alpha) (in-place)
// ============================================================================

template <typename SrcType, typename DstType>
class AddWeightedInPlaceOp {
private:
  DstType alpha;

public:
  __device__ __host__ AddWeightedInPlaceOp(DstType a) : alpha(a) {}

  __device__ __host__ DstType operator()(SrcType src, DstType dst, int = 0) const {
    // AddWeighted: result = src * alpha + dst * (1 - alpha)
    if constexpr (std::is_same_v<DstType, Npp32f>) {
      return static_cast<DstType>(src) * alpha + dst * (DstType(1.0) - alpha);
    } else {
      double src_val = static_cast<double>(src);
      double dst_val = static_cast<double>(dst);
      double alpha_val = static_cast<double>(alpha);
      double result = src_val * alpha_val + dst_val * (1.0 - alpha_val);
      return saturate_cast<DstType>(result);
    }
  }
};

// Custom kernel for AddWeighted operation
template <typename SrcType, typename DstType, int Channels>
__global__ void addWeightedInPlaceKernel(const SrcType *pSrc, int nSrcStep, 
                                          DstType *pSrcDst, int nSrcDstStep,
                                          int width, int height, DstType alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
    DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

    if constexpr (Channels == 1) {
      AddWeightedInPlaceOp<SrcType, DstType> op(alpha);
      dstRow[x] = op(srcRow[x], dstRow[x]);
    } else {
      int idx = x * Channels;
      AddWeightedInPlaceOp<SrcType, DstType> op(alpha);
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(srcRow[idx + c], dstRow[idx + c]);
      }
    }
  }
}

extern "C" {

// 8u32f versions (8-bit source, 32-bit float destination)
NppStatus nppiAddWeighted_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                          NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addWeightedInPlaceKernel<Npp8u, Npp32f, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
      oSizeROI.width, oSizeROI.height, nAlpha);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddWeighted_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep,
                                      Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddWeighted_8u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, defaultCtx);
}

// 16u32f versions (16-bit unsigned source, 32-bit float destination)
NppStatus nppiAddWeighted_16u32f_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                           NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addWeightedInPlaceKernel<Npp16u, Npp32f, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
      oSizeROI.width, oSizeROI.height, nAlpha);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddWeighted_16u32f_C1IR(const Npp16u *pSrc, int nSrcStep,
                                       Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddWeighted_16u32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, defaultCtx);
}

// 32f versions (32-bit float source and destination)
NppStatus nppiAddWeighted_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                        NppStreamContext nppStreamCtx) {
  // Validate parameters
  if (!pSrc || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  addWeightedInPlaceKernel<Npp32f, Npp32f, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pSrcDst, nSrcDstStep, 
      oSizeROI.width, oSizeROI.height, nAlpha);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAddWeighted_32f_C1IR(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddWeighted_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, defaultCtx);
}

// Masked versions
NppStatus nppiAddWeighted_8u32f_C1IMR_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                           Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                           NppStreamContext nppStreamCtx) {
  // This is a masked version - more complex implementation needed
  // For now, return not supported
  return NPP_NOT_SUPPORTED_MODE_ERROR;
}

NppStatus nppiAddWeighted_8u32f_C1IMR(const Npp8u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                       Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddWeighted_8u32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, defaultCtx);
}

NppStatus nppiAddWeighted_16u32f_C1IMR_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                            Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                            NppStreamContext nppStreamCtx) {
  // This is a masked version - more complex implementation needed
  // For now, return not supported
  return NPP_NOT_SUPPORTED_MODE_ERROR;
}

NppStatus nppiAddWeighted_16u32f_C1IMR(const Npp16u *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                        Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddWeighted_16u32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, defaultCtx);
}

NppStatus nppiAddWeighted_32f_C1IMR_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                         Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha,
                                         NppStreamContext nppStreamCtx) {
  // This is a masked version - more complex implementation needed
  // For now, return not supported
  return NPP_NOT_SUPPORTED_MODE_ERROR;
}

NppStatus nppiAddWeighted_32f_C1IMR(const Npp32f *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                     Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha) {
  NppStreamContext defaultCtx = {};
  defaultCtx.hStream = 0; // Default CUDA stream
  return nppiAddWeighted_32f_C1IMR_Ctx(pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI, nAlpha, defaultCtx);
}

} // extern "C"