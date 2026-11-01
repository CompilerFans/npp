// nppiRotate family executor: kernel, entry point template and explicit
// instantiations.  See nppi_rotate_executor.h for the measured semantics.

#include "nppi_rotate_executor.h"
#include "nppi_warp_sampling.cuh"
#include "npp_kernel_utils.h"

#include <cmath>

namespace nppi {
namespace rotate {
namespace {

constexpr double kPi = 3.14159265358979323846;

inline bool interpValid(int eInterpolation) {
  return eInterpolation == NPPI_INTER_NN || eInterpolation == NPPI_INTER_LINEAR || eInterpolation == NPPI_INTER_CUBIC;
}

template <typename T, int C, bool AP>
__global__ void rotateKernel(const T *__restrict__ pSrc, int nSrcStepEl, NppiSize oSrcSize, NppiRect oSrcROI,
                             T *__restrict__ pDst, int nDstStepEl, NppiRect oDstROI, float ca, float sa, float shiftX,
                             float shiftY, int eInterpolation) {
  const int rx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ry = blockIdx.y * blockDim.y + threadIdx.y;
  if (rx >= oDstROI.width || ry >= oDstROI.height) return;
  const int x = rx + oDstROI.x;
  const int y = ry + oDstROI.y;
  const float dx = __fsub_rn((float)x, shiftX);
  const float dy = __fsub_rn((float)y, shiftY);
  const float fx = __fsub_rn(__fmul_rn(ca, dx), __fmul_rn(sa, dy));
  const float fy = __fadd_rn(__fmul_rn(sa, dx), __fmul_rn(ca, dy));
  const float loX = (float)oSrcROI.x - 0.5f;
  const float hiX = (float)(oSrcROI.x + oSrcROI.width - 1);
  const float loY = (float)oSrcROI.y - 0.5f;
  const float hiY = (float)(oSrcROI.y + oSrcROI.height - 1);
  if (!(fx >= loX && fx <= hiX && fy >= loY && fy <= hiY)) return;
  T *dstPixel = pDst + (size_t)y * nDstStepEl + (size_t)x * C;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    if (AP && c == 3) continue;
    dstPixel[c] = warp::sampleChannel<T>(pSrc + c, nSrcStepEl, C, oSrcSize.width, oSrcSize.height, (double)fx, (double)fy,
                                   eInterpolation);
  }
}

// Largest power of two dividing the pixel size (the NPP step granularity).
inline int stepGranularity(int channels, int elemSize) {
  int v = channels * elemSize;
  int g = 1;
  while (v > 1 && (v & 1) == 0) {
    g <<= 1;
    v >>= 1;
  }
  return g;
}

} // namespace

template <typename T, int Channels, bool AlphaPreserve>
NppStatus rotate(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                 NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY, int eInterpolation,
                 cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const int gran = stepGranularity(Channels, (int)sizeof(T));
  if (nSrcStep <= 0 || nDstStep <= 0) return NPP_STEP_ERROR;
  if ((nSrcStep % gran) != 0 || (nDstStep % gran) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  if (oSrcROI.x < 0 || oSrcROI.y < 0) return NPP_SIZE_ERROR;
  if (!interpValid(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) return NPP_WRONG_INTERSECTION_QUAD_WARNING;
  const double rad = nAngle * kPi / 180.0;
  const float ca = (float)cos(rad);
  const float sa = (float)sin(rad);
  const dim3 block(16, 16);
  const dim3 grid((oDstROI.width + block.x - 1) / block.x, (oDstROI.height + block.y - 1) / block.y);
  if (oDstROI.width <= 0 || oDstROI.height <= 0 || grid.x == 0 || grid.y == 0) return NPP_NO_ERROR;
  rotateKernel<T, Channels, AlphaPreserve><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep / (int)sizeof(T), oSrcSize, oSrcROI, pDst, nDstStep / (int)sizeof(T), oDstROI, ca, sa,
      (float)nShiftX, (float)nShiftY, eInterpolation);
#ifdef NPPI_GEOM_DEBUG_SYNC
  if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
#endif
  return npp_kernel::checkCudaKernelError();
}

#define NPPI_ROTATE_INSTANTIATE(T)                                                                                    \
  template NppStatus rotate<T, 1, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, double, \
                                         int, cudaStream_t);                                                          \
  template NppStatus rotate<T, 3, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, double, \
                                         int, cudaStream_t);                                                          \
  template NppStatus rotate<T, 4, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, double, \
                                         int, cudaStream_t);                                                          \
  template NppStatus rotate<T, 4, true>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, double,  \
                                        int, cudaStream_t);

NPPI_ROTATE_INSTANTIATE(Npp8u)
NPPI_ROTATE_INSTANTIATE(Npp16u)
NPPI_ROTATE_INSTANTIATE(Npp32f)
#undef NPPI_ROTATE_INSTANTIATE

} // namespace rotate
} // namespace nppi
