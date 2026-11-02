// Kernels for the filtering long tail: bilateral Gauss, threshold adaptive box
// and Harris corners.  See nppi_filter_tail_executor.h for the contract.
#include "npp.h"
#include "nppi_filter_tail_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ __forceinline__ int mirror_index(int i, int dim) {
  while (i < 0 || i >= dim) {
    if (i < 0)
      i = -i;
    else
      i = 2 * dim - 2 - i;
  }
  return i;
}

__host__ __device__ __forceinline__ int border_index(int i, int lo, int hi, bool mirror) {
  if (lo > hi) return lo;
  if (mirror) return mirror_index(i - lo, hi - lo + 1) + lo;
  return i < lo ? lo : (i > hi ? hi : i);
}

template <typename T>
__host__ __device__ __forceinline__ float to_float(const T &s) {
  return static_cast<float>(s);
}

__device__ __forceinline__ Npp8u emit_u8(float v) {
  if (!(v > 0.0f)) return (Npp8u)0;
  if (v >= 255.0f) return (Npp8u)255;
  return (Npp8u)(int)v;
}

__device__ __forceinline__ Npp16u emit_u16(float v) {
  if (!(v > 0.0f)) return (Npp16u)0;
  if (v >= 65535.0f) return (Npp16u)65535;
  return (Npp16u)(int)v;
}

template <typename Dst>
__device__ __forceinline__ Dst emit(float v);
template <>
__device__ __forceinline__ Npp8u emit<Npp8u>(float v) {
  return emit_u8(v);
}
template <>
__device__ __forceinline__ Npp16u emit<Npp16u>(float v) {
  return emit_u16(v);
}
template <>
__device__ __forceinline__ Npp32f emit<Npp32f>(float v) {
  return v;
}

inline bool step_too_small(int nDstStep, int rowBytes, int H) {
  return H > 0 && (Npp32u)nDstStep < (Npp32u)rowBytes;
}

// ---------------------------------------------------------------- bilateral

template <typename Src, int CH>
__device__ __forceinline__ float bilateralFetch(const Src *pSrc, int nSrcStep, int fx, int fy,
                                                int ch, int loX, int hiX, int loY, int hiY,
                                                bool mirror) {
  const int ix = border_index(fx, loX, hiX, mirror);
  const int iy = border_index(fy, loY, hiY, mirror);
  const Src *row =
      reinterpret_cast<const Src *>(reinterpret_cast<const char *>(pSrc) + (long long)iy * nSrcStep);
  return to_float(row[(long long)ix * CH + ch]);
}

template <typename Src, typename Dst, int CH>
__global__ void bilateralKernel(const Src *pSrc, int nSrcStep, int srcW, int srcH, int offX,
                                int offY, Dst *pDst, int nDstStep, int dstW, int dstH,
                                int radius, int step, float valSigma2, float posSigma2,
                                int border) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstW || y >= dstH) return;
  const bool mirror = border == (int)NPP_BORDER_MIRROR;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  for (int ch = 0; ch < CH; ++ch) {
    auto fetch = [&](int fx, int fy) -> float {
      const int ix = border_index(fx, loX, hiX, mirror);
      const int iy = border_index(fy, loY, hiY, mirror);
      const Src *row = reinterpret_cast<const Src *>(reinterpret_cast<const char *>(pSrc) +
                                                     (long long)iy * nSrcStep);
      return to_float(row[(long long)ix * CH + ch]);
    };
    // NPP 13.4 snaps only the horizontal coordinate to the
    // nStepBetweenSrcPixels grid (measured with step 2: the vertical coordinate
    // stays untouched), so the sampler reproduces that quirk.
    const int cx = (step > 1) ? border_index((x + step / 2) / step * step, loX, hiX, mirror) : x;
    const float s00 = fetch(cx, y);
    float num = 0.0f, den = 0.0f;
    for (int h = -radius; h <= radius; ++h) {
      for (int w = -radius; w <= radius; ++w) {
        const float s = bilateralFetch<Src, CH>(pSrc, nSrcStep, cx + w * step, y + h * step, ch,
                                                loX, hiX, loY, hiY, mirror);
        const float d = s - s00;
        const float w1 = expf(-(d * d) / (2.0f * valSigma2));
        const float dist2 = (float)(h * h + w * w);
        const float w2 = expf(-dist2 / (2.0f * posSigma2));
        const float ww = w1 * w2;
        num += ww * s;
        den += ww;
      }
    }
    const float value = den != 0.0f ? num / den : s00;
    Dst *row = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + (long long)y * nDstStep);
    row[(long long)x * CH + ch] = emit<Dst>(value);
  }
}

// ------------------------------------------------- threshold adaptive box

__global__ void thresholdAdaptiveBoxKernel(const Npp8u *pSrc, int nSrcStep, int srcW, int srcH,
                                           int offX, int offY, Npp8u *pDst, int nDstStep, int dstW,
                                           int dstH, int maskW, int maskH, float delta, Npp8u gt,
                                           Npp8u le, int border) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstW || y >= dstH) return;
  const bool mirror = border == (int)NPP_BORDER_MIRROR;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  const int rx = maskW / 2, ry = maskH / 2;
  const Npp8u *centerRow =
      reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pSrc) +
                                      (long long)border_index(y, loY, hiY, mirror) * nSrcStep);
  const float srcVal = (float)centerRow[border_index(x, loX, hiX, mirror)];
  float sum = 0.0f;
  for (int j = -ry; j <= ry; ++j) {
    const Npp8u *row = reinterpret_cast<const Npp8u *>(
        reinterpret_cast<const char *>(pSrc) +
        (long long)border_index(y + j, loY, hiY, mirror) * nSrcStep);
    for (int i = -rx; i <= rx; ++i) sum += (float)row[border_index(x + i, loX, hiX, mirror)];
  }
  const float avg = sum / (float)(maskW * maskH);
  Npp8u *out = reinterpret_cast<Npp8u *>(reinterpret_cast<char *>(pDst) + (long long)y * nDstStep);
  out[x] = (srcVal > avg - delta) ? gt : le;
}

// -------------------------------------------------------------- Harris corners

// 3x3 / 5x5 gradient masks (row offset, col offset from the centre).
struct GradMask {
  int size;
  float kx[5][5];
  float ky[5][5];
};

__host__ __device__ __forceinline__ GradMask make_mask(int size, bool scharr) {
  GradMask m{};
  m.size = size;
  if (size == 3) {
    if (scharr) {
      const float sx[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
      const float sy[3][3] = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          m.kx[a][b] = sx[a][b];
          m.ky[a][b] = sy[a][b];
        }
    } else {
      const float sx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
      const float sy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          m.kx[a][b] = sx[a][b];
          m.ky[a][b] = sy[a][b];
        }
    }
  } else {
    const float sx[5][5] = {{-1, -2, 0, 2, 1}, {-4, -8, 0, 8, 4}, {-6, -12, 0, 12, 6},
                            {-4, -8, 0, 8, 4}, {-1, -2, 0, 2, 1}};
    const float sy[5][5] = {{-1, -4, -6, -4, -1}, {-2, -8, -12, -8, -2}, {0, 0, 0, 0, 0},
                            {2, 8, 12, 8, 2},     {1, 4, 6, 4, 1}};
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b) {
        m.kx[a][b] = sx[a][b];
        m.ky[a][b] = sy[a][b];
      }
  }
  return m;
}

__device__ __forceinline__ float byte_at(const Npp8u *pSrc, int nSrcStep, int ix, int iy, int loX,
                                         int hiX, int loY, int hiY) {
  ix = border_index(ix, loX, hiX, false);
  iy = border_index(iy, loY, hiY, false);
  return (float)pSrc[(long long)iy * nSrcStep + ix];
}

__global__ void harrisKernel(const Npp8u *pSrc, int nSrcStep, int srcW, int srcH, int offX,
                             int offY, Npp32f *pDst, int nDstStep, int dstW, int dstH, int maskSize,
                             bool scharr, int avgSize, float k, float scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstW || y >= dstH) return;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  const GradMask gm = make_mask(maskSize, scharr);
  const int mr = maskSize / 2;
  const int ar = avgSize / 2;
  float xx = 0.0f, yy = 0.0f, xy = 0.0f;
  for (int j = -ar; j <= ar; ++j) {
    for (int i = -ar; i <= ar; ++i) {
      // The averaging window clamps the *sample position* to the image first
      // and then computes the gradient there (probe-verified: clamping only
      // the individual gradient taps gives different border responses).
      const int px = border_index(x + i, loX, hiX, false);
      const int py = border_index(y + j, loY, hiY, false);
      float gx = 0.0f, gy = 0.0f;
      for (int a = -mr; a <= mr; ++a)
        for (int b = -mr; b <= mr; ++b) {
          const float s = byte_at(pSrc, nSrcStep, px + b, py + a, loX, hiX, loY, hiY);
          gx += gm.kx[a + mr][b + mr] * s;
          gy += gm.ky[a + mr][b + mr] * s;
        }
      xx += gx * gx;
      yy += gy * gy;
      xy += gx * gy;
    }
  }
  const float det = xx * yy - xy * xy;
  const float tr = xx + yy;
  const float h = (det - k * tr * tr) * scale;
  Npp32f *row = reinterpret_cast<Npp32f *>(reinterpret_cast<char *>(pDst) + (long long)y * nDstStep);
  row[x] = h;
}

}  // namespace

template <typename Src, typename Dst, int CH>
NppStatus bilateralGaussBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize,
                               NppiPoint oSrcOffset, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                               int nRadius, int nStepBetweenSrcPixels, Npp32f nValSquareSigma,
                               Npp32f nPosSquareSigma, NppiBorderType eBorderType,
                               cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (nStepBetweenSrcPixels <= 0) return NPP_STEP_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_MIRROR) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  if (nRadius < 1 || nRadius > 32) return NPP_MASK_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if ((nDstStep % (int)sizeof(Dst)) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  bilateralKernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, nRadius, nStepBetweenSrcPixels, nValSquareSigma,
      nPosSquareSigma, (int)eBorderType);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus thresholdAdaptiveBoxBorder(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                     NppiPoint oSrcOffset, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, Npp32f nDelta,
                                     Npp8u nValGT, Npp8u nValLE, NppiBorderType eBorderType,
                                     cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (oMaskSize.width < 1 || oMaskSize.height < 1 || (oMaskSize.width & 1) == 0 ||
      (oMaskSize.height & 1) == 0 || oMaskSize.width != oMaskSize.height) {
    return NPP_MASK_SIZE_ERROR;
  }
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int rowBytes = oSizeROI.width * (int)sizeof(Npp8u);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  thresholdAdaptiveBoxKernel<<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, oMaskSize.width, oMaskSize.height, nDelta, nValGT, nValLE,
      (int)eBorderType);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus harrisCornersBorder(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                              NppiPoint oSrcOffset, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize,
                              NppiMaskSize eAvgWindowSize, Npp32f nK, Npp32f nScale,
                              NppiBorderType eBorderType, Npp8u *pDeviceBuffer,
                              cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr || pDeviceBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (eFilterType != NPP_FILTER_SOBEL && eFilterType != NPP_FILTER_SCHARR) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  const int maskSize = (eMaskSize == NPP_MASK_SIZE_3_X_3)   ? 3
                       : (eMaskSize == NPP_MASK_SIZE_5_X_5) ? 5
                                                            : 0;
  const int avgSize = (eAvgWindowSize == NPP_MASK_SIZE_3_X_3)   ? 3
                      : (eAvgWindowSize == NPP_MASK_SIZE_5_X_5) ? 5
                                                                : 0;
  if (maskSize == 0 || avgSize == 0) return NPP_MASK_SIZE_ERROR;
  if (eFilterType == NPP_FILTER_SCHARR && maskSize != 3) return NPP_MASK_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int rowBytes = oSizeROI.width * (int)sizeof(Npp32f);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if ((nDstStep % (int)sizeof(Npp32f)) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  harrisKernel<<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, maskSize, eFilterType == NPP_FILTER_SCHARR, avgSize, nK,
      nScale);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template NppStatus bilateralGaussBorder<Npp8u, Npp8u, 1>(const Npp8u *, int, NppiSize, NppiPoint,
                                                          Npp8u *, int, NppiSize, int, int,
                                                          Npp32f, Npp32f, NppiBorderType,
                                                          cudaStream_t);
template NppStatus bilateralGaussBorder<Npp8u, Npp8u, 3>(const Npp8u *, int, NppiSize, NppiPoint,
                                                          Npp8u *, int, NppiSize, int, int,
                                                          Npp32f, Npp32f, NppiBorderType,
                                                          cudaStream_t);
template NppStatus bilateralGaussBorder<Npp16u, Npp16u, 1>(const Npp16u *, int, NppiSize,
                                                            NppiPoint, Npp16u *, int, NppiSize, int,
                                                            int, Npp32f, Npp32f, NppiBorderType,
                                                            cudaStream_t);
template NppStatus bilateralGaussBorder<Npp16u, Npp16u, 3>(const Npp16u *, int, NppiSize,
                                                            NppiPoint, Npp16u *, int, NppiSize, int,
                                                            int, Npp32f, Npp32f, NppiBorderType,
                                                            cudaStream_t);
template NppStatus bilateralGaussBorder<Npp32f, Npp32f, 1>(const Npp32f *, int, NppiSize,
                                                            NppiPoint, Npp32f *, int, NppiSize, int,
                                                            int, Npp32f, Npp32f, NppiBorderType,
                                                            cudaStream_t);
template NppStatus bilateralGaussBorder<Npp32f, Npp32f, 3>(const Npp32f *, int, NppiSize,
                                                            NppiPoint, Npp32f *, int, NppiSize, int,
                                                            int, Npp32f, Npp32f, NppiBorderType,
                                                            cudaStream_t);

}  // namespace filter
}  // namespace nppi
