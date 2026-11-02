// nppiMSSSIM (deprecated MS-SSIM*): best-effort implementation.
//
// NPP 13.4's MSSSIM is not the classic weighted MS-SSIM product (the
// WMSSSIM family, which this repository reproduces exactly).  Measured
// behaviour of the reference:
//   * identical images return exactly 1;
//   * two flat images with different levels return exactly 0;
//   * smooth textures land near the classic product (0.9948 vs 0.9945) while
//     noise is far below it (0.0189 vs 0.273), so the reference applies a
//     different per-scale combination that we have not decoded.
// This implementation therefore returns the classic multi-scale product (the
// WMSSSIM pipeline) for general inputs, with the two flat/identical reference
// behaviours special-cased.  ROI must be at least 176x176; an empty ROI is a
// successful no-op and source steps must be at least the ROI width.
#include "npp.h"

#include "nppi_ssim_executor.h"

#include <cuda_runtime.h>

namespace {

constexpr int kMinMSSSIMSize = 176;

__global__ void msssimFlagsKernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, int width,
                                  int height, int *pFlags) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const Npp8u a = pSrc1[(size_t)y * nSrc1Step + x];
  const Npp8u b = pSrc2[(size_t)y * nSrc2Step + x];
  const Npp8u a0 = pSrc1[0];
  const Npp8u b0 = pSrc2[0];
  if (a != b) atomicOr(&pFlags[0], 1);
  if (a != a0) atomicOr(&pFlags[1], 1);
  if (b != b0) atomicOr(&pFlags[2], 1);
}

__global__ void setZeroKernel(Npp32f *pOut) { *pOut = 0.0f; }

NppStatus msssimCommon(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                       Npp32f *pMSSSIM, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (pSrc1 == nullptr || pSrc2 == nullptr || pMSSSIM == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSizeROI.width < kMinMSSSIMSize || oSizeROI.height < kMinMSSSIMSize) return NPP_SIZE_ERROR;
  if (nSrc1Step < oSizeROI.width || nSrc2Step < oSizeROI.width) return NPP_STEP_ERROR;

  int *pFlags = nullptr;
  if (cudaMalloc(&pFlags, 3 * sizeof(int)) != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;
  cudaMemsetAsync(pFlags, 0, 3 * sizeof(int), stream);
  dim3 block(32, 8);
  dim3 grid((oSizeROI.width + 31) / 32, (oSizeROI.height + 7) / 8);
  msssimFlagsKernel<<<grid, block, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI.width, oSizeROI.height,
                                                pFlags);
  int hostFlags[3] = {0, 0, 0};
  cudaError_t err = cudaMemcpyAsync(hostFlags, pFlags, sizeof(hostFlags), cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  cudaFree(pFlags);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

  const bool equal = hostFlags[0] == 0;
  const bool flat1 = hostFlags[1] == 0;
  const bool flat2 = hostFlags[2] == 0;
  if (!equal && flat1 && flat2) {
    setZeroKernel<<<1, 1, 0, stream>>>(pMSSSIM);
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  return nppi::statistics::runWMSSSIM(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, 1, pMSSSIM, pDeviceBuffer,
                                      stream);
}

}  // namespace

NppStatus nppiMSSSIM_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                NppiSize oSizeROI, Npp32f *pMSSSIM, Npp8u *pDeviceBuffer,
                                NppStreamContext nppStreamCtx) {
  return msssimCommon(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pMSSSIM, pDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiMSSSIM_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                            Npp32f *pMSSSIM, Npp8u *pDeviceBuffer) {
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  return nppiMSSSIM_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, pMSSSIM, pDeviceBuffer, ctx);
}
