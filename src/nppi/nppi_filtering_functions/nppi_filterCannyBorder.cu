// Canny edge detector (nppiFilterCannyBorder_8u_C1R), decoded from NPP 13.4.
//
// The reference pipeline (measured on an RTX 5090; see
// test/unit/num/num_nppi_canny.h) is:
//   1. gradient: the GradientVectorBorder filter (Sobel or Scharr, mask 3x3 or
//      5x5) over the ROI, taps clamped to the oSrcSize image in pSrc-relative
//      coordinates, magnitude truncated to a saturated Npp16s, angle
//      atan2(Y, X) radians;
//   2. non-maximum suppression with the four direction bins of [0,180)
//      degrees (horizontal, anti-diagonal (1,-1), vertical, main diagonal
//      (1,1)); a pixel survives when its magnitude is positive and >= both
//      directional neighbours, which are clamped to the ROI edge;
//   3. single-pass hysteresis on the suppressed magnitudes: accept when
//      m >= high, or when m >= low and any of the eight neighbours in the
//      unsuppressed magnitude image is >= low;
//   4. output 0x00 / 0xFF; only NPP_BORDER_REPLICATE is supported.
#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>

namespace {

struct CannyCoef {
  int size;
  int k[5][5];
};

inline CannyCoef coefOf(const int *k, int n) {
  CannyCoef c{};
  c.size = n;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) c.k[a][b] = k[a * n + b];
  return c;
}

// Same tap tables as the GradientVectorBorder executor: X is the negated
// vertical response for Sobel and the positive one for Scharr; Y is the
// positive horizontal response.
void cannyTaps(NppiDifferentialKernel type, NppiMaskSize maskSize, CannyCoef &cx, CannyCoef &cy) {
  static const int SobelHoriz3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  static const int SobelVert3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const int SobelHoriz5[25] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0,
                                      2, 8,  12, 8,  2,  1,  4,  6,   4,  1};
  static const int SobelVert5[25] = {-1, -2, 0,  2,  1,  -4, -8, 0,  8,  4,  -6, -12, 0,  12, 6,
                                     -4, -8, 0,  8,  4,  -1, -2, 0,  2,  1};
  static const int ScharrHoriz3[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};
  static const int ScharrVert3[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
  const bool scharr = type == NPP_FILTER_SCHARR;
  if (!scharr && maskSize == NPP_MASK_SIZE_5_X_5) {
    cx = coefOf(SobelVert5, 5);
    cy = coefOf(SobelHoriz5, 5);
  } else {
    cx = coefOf(scharr ? ScharrVert3 : SobelVert3, 3);
    cy = coefOf(scharr ? ScharrHoriz3 : SobelHoriz3, 3);
  }
  if (!scharr)
    for (int a = 0; a < cx.size; ++a)
      for (int b = 0; b < cx.size; ++b) cx.k[a][b] = -cx.k[a][b];
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct CannyDirs {
  int dx1, dy1, dx2, dy2;
};

__device__ __forceinline__ CannyDirs cannyDirsOf(double deg) {
  while (deg < 0) deg += 180.0;
  if (deg >= 180.0) deg -= 180.0;
  if (deg < 22.5 || deg >= 157.5) return {1, 0, -1, 0};
  if (deg < 67.5) return {1, -1, -1, 1};
  if (deg < 112.5) return {0, 1, 0, -1};
  return {1, 1, -1, -1};
}

__global__ void cannyGradientKernel(const Npp8u *pSrc, int nSrcStep, Npp16s *pMag, int nMagStep,
                                    Npp32f *pAngle, int nAngleStep, int width, int height, CannyCoef cx,
                                    CannyCoef cy, int srcW, int srcH, int offX, int offY, int normL1) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int r = cx.size >> 1;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  int ax = 0, ay = 0;
  for (int a = 0; a < cx.size; ++a) {
    const int ry = clampi(y + a - r, loY, hiY);
    const Npp8u *row = (const Npp8u *)((const char *)pSrc + (size_t)ry * nSrcStep);
    for (int b = 0; b < cx.size; ++b) {
      const int rx = clampi(x + b - r, loX, hiX);
      const int v = (int)row[rx];
      ax += cx.k[a][b] * v;
      ay += cy.k[a][b] * v;
    }
  }
  const double mag = normL1 ? (std::fabs((double)ax) + std::fabs((double)ay))
                            : std::sqrt((double)ax * ax + (double)ay * ay);
  const Npp16s m16 = mag >= 32767.0 ? (Npp16s)32767 : (Npp16s)(float)mag;
  ((Npp16s *)((char *)pMag + (size_t)y * nMagStep))[x] = m16;
  ((Npp32f *)((char *)pAngle + (size_t)y * nAngleStep))[x] = (Npp32f)std::atan2((double)ay, (double)ax);
}

// Three-channel source (NPP 13.4 addition).  Measured: the channel is chosen
// by the largest X^2+Y^2 (always the L2 measure, like the GradientVector C3
// C1R family), independently of eNorm; the emitted magnitude is the chosen
// channel's magnitude under eNorm and the direction is the chosen channel's
// atan2.  A single NMS/hysteresis pass then runs on that combined map.
__global__ void cannyGradientKernelC3(const Npp8u *pSrc, int nSrcStep, Npp16s *pMag, int nMagStep,
                                      Npp32f *pAngle, int nAngleStep, int width, int height, CannyCoef cx,
                                      CannyCoef cy, int srcW, int srcH, int offX, int offY, int normL1) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int r = cx.size >> 1;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  double bestSq = -1.0, bestMag = 0.0, bestAngle = 0.0;
  for (int ch = 0; ch < 3; ++ch) {
    int ax = 0, ay = 0;
    for (int a = 0; a < cx.size; ++a) {
      const int ry = clampi(y + a - r, loY, hiY);
      const Npp8u *row = (const Npp8u *)((const char *)pSrc + (size_t)ry * nSrcStep);
      for (int b = 0; b < cx.size; ++b) {
        const int rx = clampi(x + b - r, loX, hiX);
        const int v = (int)row[rx * 3 + ch];
        ax += cx.k[a][b] * v;
        ay += cy.k[a][b] * v;
      }
    }
    const double sq = (double)ax * ax + (double)ay * ay;
    if (sq > bestSq) {
      bestSq = sq;
      bestMag = normL1 ? (std::fabs((double)ax) + std::fabs((double)ay)) : std::sqrt(sq);
      bestAngle = std::atan2((double)ay, (double)ax);
    }
  }
  const Npp16s m16 = bestMag >= 32767.0 ? (Npp16s)32767 : (Npp16s)(float)bestMag;
  ((Npp16s *)((char *)pMag + (size_t)y * nMagStep))[x] = m16;
  ((Npp32f *)((char *)pAngle + (size_t)y * nAngleStep))[x] = (Npp32f)bestAngle;
}

__device__ __forceinline__ int cannyMagAt(const Npp16s *pMag, int step, int x, int y, int w, int h) {
  return (int)((const Npp16s *)((const char *)pMag + (size_t)clampi(y, 0, h - 1) * step))[clampi(x, 0, w - 1)];
}

__global__ void cannyNmsKernel(const Npp16s *pMag, int nMagStep, const Npp32f *pAngle, int nAngleStep,
                               Npp16s *pNms, int nNmsStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const Npp32f angle = ((const Npp32f *)((const char *)pAngle + (size_t)y * nAngleStep))[x];
  const CannyDirs d = cannyDirsOf((double)angle * 180.0 / M_PI);
  const int m = cannyMagAt(pMag, nMagStep, x, y, width, height);
  const int m1 = cannyMagAt(pMag, nMagStep, x + d.dx1, y + d.dy1, width, height);
  const int m2 = cannyMagAt(pMag, nMagStep, x + d.dx2, y + d.dy2, width, height);
  const Npp16s out = (m > 0 && m >= m1 && m >= m2) ? (Npp16s)m : (Npp16s)0;
  ((Npp16s *)((char *)pNms + (size_t)y * nNmsStep))[x] = out;
}

__global__ void cannyHysteresisKernel(const Npp16s *pMag, int nMagStep, const Npp16s *pNms, int nNmsStep,
                                      Npp8u *pDst, int nDstStep, int width, int height, int lo, int hi) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int m = (int)((const Npp16s *)((const char *)pNms + (size_t)y * nNmsStep))[x];
  int rmax = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      if (dx != 0 || dy != 0) rmax = max(rmax, cannyMagAt(pMag, nMagStep, x + dx, y + dy, width, height));
  const bool accept = (m >= hi) || (m >= lo && rmax >= lo);
  ((Npp8u *)((char *)pDst + (size_t)y * nDstStep))[x] = accept ? (Npp8u)0xFF : (Npp8u)0x00;
}


}  // namespace

extern "C" {

NppStatus nppiFilterCannyBorderGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize) {
  const size_t imageSize = (size_t)oSizeROI.width * oSizeROI.height;
  // Magnitude (2 bytes), angle (4 bytes) and the suppressed magnitude
  // (2 bytes) per pixel.
  const size_t total = imageSize * 8;
  *hpBufferSize = (int)((total + 511) & ~(size_t)511);
  return NPP_SUCCESS;
}

NppStatus nppiFilterCannyBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                                                Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm,
                                                NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                                                NppStreamContext nppStreamCtx) {
  if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;

  const int width = oSizeROI.width, height = oSizeROI.height;
  const size_t numPixels = (size_t)width * height;
  Npp16s *pMag = (Npp16s *)pDeviceBuffer;
  Npp32f *pAngle = (Npp32f *)(pMag + numPixels);
  Npp16s *pNms = (Npp16s *)(pAngle + numPixels);

  CannyCoef cx, cy;
  cannyTaps(eFilterType, eMaskSize, cx, cy);
  const int normL1 = eNorm == nppiNormL1 ? 1 : 0;
  cudaStream_t stream = nppStreamCtx.hStream;

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  cannyGradientKernel<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pMag, width * (int)sizeof(Npp16s), pAngle,
                                                  width * (int)sizeof(Npp32f), width, height, cx, cy,
                                                  oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y,
                                                  normL1);
  cannyNmsKernel<<<grid, block, 0, stream>>>(pMag, width * (int)sizeof(Npp16s), pAngle,
                                             width * (int)sizeof(Npp32f), pNms, width * (int)sizeof(Npp16s),
                                             width, height);
  cannyHysteresisKernel<<<grid, block, 0, stream>>>(pMag, width * (int)sizeof(Npp16s), pNms,
                                                    width * (int)sizeof(Npp16s), pDst, nDstStep, width, height,
                                                    (int)nLowThreshold, (int)nHighThreshold);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_SUCCESS;
}


NppStatus nppiFilterCannyBorder_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                 NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                 NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                                                 Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm,
                                                 NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                                                 NppStreamContext nppStreamCtx) {
  if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;

  const int width = oSizeROI.width, height = oSizeROI.height;
  const size_t numPixels = (size_t)width * height;
  Npp16s *pMag = (Npp16s *)pDeviceBuffer;
  Npp32f *pAngle = (Npp32f *)(pMag + numPixels);
  Npp16s *pNms = (Npp16s *)(pAngle + numPixels);

  CannyCoef cx, cy;
  cannyTaps(eFilterType, eMaskSize, cx, cy);
  const int normL1 = eNorm == nppiNormL1 ? 1 : 0;
  cudaStream_t stream = nppStreamCtx.hStream;

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  cannyGradientKernelC3<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pMag, width * (int)sizeof(Npp16s), pAngle,
                                                   width * (int)sizeof(Npp32f), width, height, cx, cy,
                                                   oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y,
                                                   normL1);
  cannyNmsKernel<<<grid, block, 0, stream>>>(pMag, width * (int)sizeof(Npp16s), pAngle,
                                             width * (int)sizeof(Npp32f), pNms, width * (int)sizeof(Npp16s),
                                             width, height);
  cannyHysteresisKernel<<<grid, block, 0, stream>>>(pMag, width * (int)sizeof(Npp16s), pNms,
                                                    width * (int)sizeof(Npp16s), pDst, nDstStep, width, height,
                                                    (int)nLowThreshold, (int)nHighThreshold);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_SUCCESS;
}

}  // namespace
