// Hough line transform — NPP-compatible accumulator model.
//
// Decoded from the 5090 accumulator dumps of nppiFilterHoughLine:
//   * thetaBins = ceil(180 / delta.theta) for delta.theta <= 1, else 180;
//   * rhoBins   = ceil(ceil(max(w,h) * sqrt(2)) / delta.rho) for delta.rho <= 1,
//                 else ceil(max(w,h) * sqrt(2));
//   * the accumulator is int[thetaBin + thetaBins * rhoBin];
//   * only source pixels equal to 255 vote;
//   * a vote lands in rhoBin = floor((rho / delta.rho) + R) with
//     rho = (x - w/2) cos(theta) + (y - h/2) sin(theta),
//     R = floor(max(w,h) * sqrt(2) / 2), theta = thetaBin * delta.theta
//     (for delta.theta > 1 the vote angle stays 1 degree per bin while the
//     reported theta is thetaBin * delta.theta);
//   * emitted rho/theta are bin * delta; candidates are strict local maxima
//     of the accumulator that reach nThreshold, filtered by the optional
//     region and truncated to nMaxLineCount.
//
// Known deviation: NPP's delta.theta < 1 (and some delta.rho != 1) peaks show
// angle/rho quirks that come from its internal trig/bin rounding; this
// implementation uses the physical angle.  Differential tests use
// delta = (1,1) clean lines, where the reference matches exactly.
#include "npp.h"

#include <cmath>
#include <cuda_runtime.h>

namespace {

__global__ void houghAccumulateKernel(const Npp8u *pSrc, int nSrcStep, int width, int height, int thetaBins,
                                      int rhoBins, float deltaRho, float deltaTheta, int *pAccum) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (*(const Npp8u *)((const char *)pSrc + (size_t)y * nSrcStep + x) != 255) return;

  const float R = floorf((float)(width > height ? width : height) * sqrtf(2.0f) * 0.5f);
  const float dx = (float)x - (float)(width / 2);
  const float dy = (float)y - (float)(height / 2);
  const float voteScale = deltaTheta > 1.0f ? 1.0f : deltaTheta;
  for (int t = 0; t < thetaBins; ++t) {
    const float radians = (float)t * voteScale * (float)(3.14159265358979323846 / 180.0);
    const float rho = dx * cosf(radians) + dy * sinf(radians);
    const int bin = (int)floorf(rho / deltaRho + R);
    if (bin >= 0 && bin < rhoBins) atomicAdd(&pAccum[t + thetaBins * bin], 1);
  }
}

__device__ bool isLocalMax(const int *pAccum, int thetaBins, int rhoBins, int t, int r) {
  const int v = pAccum[t + thetaBins * r];
  for (int dr = -1; dr <= 1; ++dr) {
    const int rr = r + dr;
    if (rr < 0 || rr >= rhoBins) continue;
    for (int dt = -1; dt <= 1; ++dt) {
      if (dt == 0 && dr == 0) continue;
      int tt = t + dt;
      if (tt < 0) tt += thetaBins;
      if (tt >= thetaBins) tt -= thetaBins;
      if (pAccum[tt + thetaBins * rr] > v) return false;
    }
  }
  return true;
}

__global__ void houghLinesKernel(const int *pAccum, int thetaBins, int rhoBins, float deltaRho, float deltaTheta,
                                 int threshold, bool useRegion, NppPointPolar rhoLo, NppPointPolar rhoHi,
                                 NppPointPolar *pLines, int maxLines, int *pLineCount) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= thetaBins || r >= rhoBins) return;
  const int v = pAccum[t + thetaBins * r];
  if (v < threshold || !isLocalMax(pAccum, thetaBins, rhoBins, t, r)) return;

  NppPointPolar line;
  line.rho = (Npp32f)r * deltaRho;
  line.theta = (Npp32f)t * deltaTheta;
  if (useRegion) {
    if (line.rho < rhoLo.rho || line.rho > rhoHi.rho || line.theta < rhoLo.theta || line.theta > rhoHi.theta) {
      return;
    }
  }
  const int slot = atomicAdd(pLineCount, 1);
  if (slot < maxLines) pLines[slot] = line;
}

__global__ void clampLineCountKernel(int *pLineCount, int maxLines) {
  if (*pLineCount > maxLines) *pLineCount = maxLines;
}

NppStatus validate(NppiSize roi, NppPointPolar delta, int threshold, NppPointPolar *pLines, int maxLineCount,
                   int *pLineCount, Npp8u *pDeviceBuffer) {
  if (pLines == nullptr || pLineCount == nullptr || pDeviceBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (roi.width <= 0 || roi.height <= 0) return NPP_SIZE_ERROR;
  if (!(delta.rho > 0.0f && delta.rho <= 3.0f)) return NPP_BAD_ARGUMENT_ERROR;
  if (!(delta.theta >= 0.25f && delta.theta <= 3.0f)) return NPP_BAD_ARGUMENT_ERROR;
  if (threshold == 0) return NPP_BAD_ARGUMENT_ERROR;
  if (maxLineCount <= 0) return NPP_BAD_ARGUMENT_ERROR;
  return NPP_NO_ERROR;
}

void houghLineCounts(NppiSize roi, NppPointPolar delta, int *thetaBins, int *rhoBins) {
  const int maxDim = roi.width > roi.height ? roi.width : roi.height;
  const float diag = ceilf((float)maxDim * sqrtf(2.0f));
  *thetaBins = delta.theta <= 1.0f ? (int)ceilf(180.0f / delta.theta) : 180;
  *rhoBins = delta.rho <= 1.0f ? (int)ceilf(diag / delta.rho) : (int)diag;
}

NppStatus houghLineCommon(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, NppPointPolar nDelta,
                          int nThreshold, NppPointPolar *pDeviceLines, int nMaxLineCount, int *pDeviceLineCount,
                          Npp8u *pDeviceBuffer, bool useRegion, NppPointPolar oDstROI[2], cudaStream_t stream) {
  if (pSrc == nullptr) return NPP_NULL_POINTER_ERROR;
  const NppStatus status = validate(oSizeROI, nDelta, nThreshold, pDeviceLines, nMaxLineCount, pDeviceLineCount,
                                    pDeviceBuffer);
  if (status != NPP_NO_ERROR) return status;
  if (useRegion) {
    if (oDstROI == nullptr) return NPP_NULL_POINTER_ERROR;
    if (oDstROI[0].rho > oDstROI[1].rho || oDstROI[0].theta > oDstROI[1].theta) return NPP_BAD_ARGUMENT_ERROR;
  }

  int thetaBins = 0, rhoBins = 0;
  houghLineCounts(oSizeROI, nDelta, &thetaBins, &rhoBins);
  int *pAccum = (int *)pDeviceBuffer;
  cudaMemsetAsync(pDeviceBuffer, 0, (size_t)thetaBins * rhoBins * sizeof(int), stream);
  cudaMemsetAsync(pDeviceLineCount, 0, sizeof(int), stream);

  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  houghAccumulateKernel<<<grid, block, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, thetaBins,
                                                    rhoBins, nDelta.rho, nDelta.theta, pAccum);

  NppPointPolar lo{}, hi{};
  if (useRegion) {
    lo = oDstROI[0];
    hi = oDstROI[1];
  }
  const int lineGrid = (thetaBins + 31) / 32;
  const int rhoGrid = (rhoBins + 7) / 8;
  houghLinesKernel<<<dim3(lineGrid, rhoGrid), dim3(32, 8), 0, stream>>>(
      pAccum, thetaBins, rhoBins, nDelta.rho, nDelta.theta, nThreshold, useRegion, lo, hi, pDeviceLines,
      nMaxLineCount, pDeviceLineCount);
  clampLineCountKernel<<<1, 1, 0, stream>>>(pDeviceLineCount, nMaxLineCount);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

NppStatus nppiFilterHoughLine_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI,
                                            NppPointPolar nDelta, int nThreshold, NppPointPolar *pDeviceLines,
                                            int nMaxLineCount, int *pDeviceLineCount, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx) {
  return houghLineCommon(pSrc, nSrcStep, oSizeROI, nDelta, nThreshold, pDeviceLines, nMaxLineCount,
                         pDeviceLineCount, pDeviceBuffer, false, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiFilterHoughLine_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, NppPointPolar nDelta,
                                        int nThreshold, NppPointPolar *pDeviceLines, int nMaxLineCount,
                                        int *pDeviceLineCount, Npp8u *pDeviceBuffer) {
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  return nppiFilterHoughLine_8u32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, nDelta, nThreshold, pDeviceLines,
                                           nMaxLineCount, pDeviceLineCount, pDeviceBuffer, ctx);
}

NppStatus nppiFilterHoughLineRegion_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI,
                                                  NppPointPolar nDelta, int nThreshold, NppPointPolar *pDeviceLines,
                                                  NppPointPolar oDstROI[2], int nMaxLineCount, int *pDeviceLineCount,
                                                  Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return houghLineCommon(pSrc, nSrcStep, oSizeROI, nDelta, nThreshold, pDeviceLines, nMaxLineCount,
                         pDeviceLineCount, pDeviceBuffer, true, oDstROI, nppStreamCtx.hStream);
}

NppStatus nppiFilterHoughLineRegion_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI,
                                              NppPointPolar nDelta, int nThreshold, NppPointPolar *pDeviceLines,
                                              NppPointPolar oDstROI[2], int nMaxLineCount, int *pDeviceLineCount,
                                              Npp8u *pDeviceBuffer) {
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  return nppiFilterHoughLineRegion_8u32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, nDelta, nThreshold, pDeviceLines, oDstROI,
                                                 nMaxLineCount, pDeviceLineCount, pDeviceBuffer, ctx);
}
