#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations

extern "C" {
cudaError_t nppiMirror_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream);
}

NppStatus nppiMirror_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                NppiAxis flip, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oROI.width <= 0 || oROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oROI.width || nDstStep < oROI.width) {
    return NPP_STEP_ERROR;
  }
  // Validate flip parameters
  if (flip != NPP_HORIZONTAL_AXIS && flip != NPP_VERTICAL_AXIS && flip != NPP_BOTH_AXIS) {
    return NPP_MIRROR_FLIP_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus = nppiMirror_8u_C1R_kernel(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMirror_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiMirror_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, ctx);
}
