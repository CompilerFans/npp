#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// DeviceC (device memory constant) kernel for multi-channel operations
template <typename T, int Channels, typename ConstOpType>
__global__ void deviceConstKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                  const T *pConstants, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      ConstOpType op(pConstants[0]);
      dstRow[x] = op(srcRow[x], T(0), scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        ConstOpType op(pConstants[c]);
        dstRow[idx + c] = op(srcRow[idx + c], T(0), scaleFactor);
      }
    }
  }
}

// DeviceC operation executor for GPU device memory constants
template <typename T, int Channels, typename ConstOpType> class DeviceConstOperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream, const T *pDeviceConstants) {
    // Validate parameters
    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Validate device constant pointer
    if (!pDeviceConstants) {
      return NPP_NULL_POINTER_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    deviceConstKernel<T, Channels, ConstOpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pDeviceConstants, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// Implementation functions using the device constant executor
extern "C" {

// 8u C1 operations with scale factor
NppStatus nppiAddDeviceC_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstant, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp8u, 1, AddConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             nScaleFactor, nppStreamCtx.hStream,
                                                                             pConstant);
}

// 8u C3 operations with scale factor
NppStatus nppiAddDeviceC_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp8u, 3, AddConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             nScaleFactor, nppStreamCtx.hStream,
                                                                             pConstants);
}

// 8u C4 operations with scale factor
NppStatus nppiAddDeviceC_8u_C4RSfs_Ctx_impl(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp8u, 4, AddConstOp<Npp8u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                             nScaleFactor, nppStreamCtx.hStream,
                                                                             pConstants);
}

// 16u C1 operations with scale factor
NppStatus nppiAddDeviceC_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc, int nSrcStep, const Npp16u *pConstant, Npp16u *pDst,
                                             int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                             NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp16u, 1, AddConstOp<Npp16u>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                               nScaleFactor, nppStreamCtx.hStream,
                                                                               pConstant);
}

// 16s C1 operations with scale factor
NppStatus nppiAddDeviceC_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc, int nSrcStep, const Npp16s *pConstant, Npp16s *pDst,
                                             int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                             NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp16s, 1, AddConstOp<Npp16s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                               nScaleFactor, nppStreamCtx.hStream,
                                                                               pConstant);
}

// 32s C1 operations with scale factor
NppStatus nppiAddDeviceC_32s_C1RSfs_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp32s *pConstant, Npp32s *pDst,
                                             int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                             NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp32s, 1, AddConstOp<Npp32s>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                               nScaleFactor, nppStreamCtx.hStream,
                                                                               pConstant);
}

// 32f C1 operations (no scale factor)
NppStatus nppiAddDeviceC_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, const Npp32f *pConstant, Npp32f *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return DeviceConstOperationExecutor<Npp32f, 1, AddConstOp<Npp32f>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                                               0, nppStreamCtx.hStream, pConstant);
}

} // extern "C"

// Public API functions - only Ctx versions as per NPP API specification
NppStatus nppiAddDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiAddDeviceC_8u_C1RSfs_Ctx_impl(pSrc, nSrcStep, pConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                           nppStreamCtx);
}

NppStatus nppiAddDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiAddDeviceC_8u_C3RSfs_Ctx_impl(pSrc, nSrcStep, pConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                           nppStreamCtx);
}

NppStatus nppiAddDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiAddDeviceC_8u_C4RSfs_Ctx_impl(pSrc, nSrcStep, pConstants, pDst, nDstStep, oSizeROI, nScaleFactor,
                                           nppStreamCtx);
}

NppStatus nppiAddDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return nppiAddDeviceC_16u_C1RSfs_Ctx_impl(pSrc, nSrcStep, pConstant, pDst, nDstStep, oSizeROI, nScaleFactor,
                                            nppStreamCtx);
}

NppStatus nppiAddDeviceC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiAddDeviceC_32f_C1R_Ctx_impl(pSrc, nSrcStep, pConstant, pDst, nDstStep, oSizeROI, nppStreamCtx);
}