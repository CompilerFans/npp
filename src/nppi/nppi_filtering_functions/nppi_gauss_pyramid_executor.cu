// Gauss pyramid filter kernels and executor entry points.
// See nppi_gauss_pyramid_executor.h for the family contract.
//
// The down path is probe-verified against NVIDIA NPP 13.4 on the RTX 5090:
// every destination pixel samples the source at `j * rate + (t - center)` per
// tap, with bilinear interpolation for the fractional part and REPLICATE /
// MIRROR (reflect-101) applied to the tap coordinate.  Kernel taps are used
// verbatim (no renormalisation).  Integer destinations truncate toward zero.
#include "npp.h"
#include "nppi_gauss_pyramid_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

// NPP takes kernel/mask arrays as HOST pointers and stages them internally.
// cudaMemcpyDefault lets the same path also accept a device pointer (the
// differential harness passes device-side kernel buffers).
template <typename K>
inline NppStatus stage_kernel(const K *pKernel, int count, cudaStream_t stream, K **out) {
  K *d = nullptr;
  if (cudaMallocAsync(reinterpret_cast<void **>(&d), (size_t)count * sizeof(K), stream) !=
      cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  if (cudaMemcpyAsync(d, pKernel, (size_t)count * sizeof(K), cudaMemcpyDefault, stream) !=
      cudaSuccess) {
    cudaFreeAsync(d, stream);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  *out = d;
  return NPP_NO_ERROR;
}

template <typename K, typename F>
inline NppStatus with_staged_kernel(const K *pKernel, int count, cudaStream_t stream, F &&launch) {
  K *d = nullptr;
  NppStatus st = stage_kernel(pKernel, count, stream, &d);
  if (st != NPP_NO_ERROR) return st;
  launch(d);
  cudaError_t e = cudaGetLastError();
  cudaFreeAsync(d, stream);
  return e == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

inline bool step_too_small(int nDstStep, int rowBytes, int H) {
  return H > 0 && (Npp32u)nDstStep < (Npp32u)rowBytes;
}

__host__ __device__ __forceinline__ int border_index(int i, int lo, int hi, bool mirror) {
  if (lo > hi) return lo;
  if (mirror) {
    const int dim = hi - lo + 1;
    i -= lo;
    // reflect-101, repeated until inside (tap offsets can exceed the image).
    while (i < 0 || i >= dim) {
      if (i < 0)
        i = -i;
      else
        i = 2 * dim - 2 - i;
    }
    return i + lo;
  }
  return i < lo ? lo : (i > hi ? hi : i);
}

// Source fetch in pSrc-relative coordinates, border applied per axis.
template <typename Src, int CH>
__device__ __forceinline__ float fetch(const Src *pSrc, int nSrcStep, int ix, int iy, int ch,
                                       int loX, int hiX, int loY, int hiY, bool mirror) {
  ix = border_index(ix, loX, hiX, mirror);
  iy = border_index(iy, loY, hiY, mirror);
  const Src *row = reinterpret_cast<const Src *>(reinterpret_cast<const char *>(pSrc) +
                                                 (long long)iy * nSrcStep);
  return static_cast<float>(row[(long long)ix * CH + ch]);
}

// Bilinear source sample at a fractional pSrc-relative coordinate.
template <typename Src, int CH>
__device__ __forceinline__ float sample2d(const Src *pSrc, int nSrcStep, float fx, float fy,
                                          int ch, int loX, int hiX, int loY, int hiY,
                                          bool mirror) {
  const float fxf = floorf(fx);
  const float fyf = floorf(fy);
  const int x0 = (int)fxf;
  const int y0 = (int)fyf;
  const float ax = fx - fxf;
  const float ay = fy - fyf;
  if (ax == 0.0f && ay == 0.0f) {
    return fetch<Src, CH>(pSrc, nSrcStep, x0, y0, ch, loX, hiX, loY, hiY, mirror);
  }
  const float v00 = fetch<Src, CH>(pSrc, nSrcStep, x0, y0, ch, loX, hiX, loY, hiY, mirror);
  const float v10 = fetch<Src, CH>(pSrc, nSrcStep, x0 + 1, y0, ch, loX, hiX, loY, hiY, mirror);
  const float v01 = fetch<Src, CH>(pSrc, nSrcStep, x0, y0 + 1, ch, loX, hiX, loY, hiY, mirror);
  const float v11 = fetch<Src, CH>(pSrc, nSrcStep, x0 + 1, y0 + 1, ch, loX, hiX, loY, hiY, mirror);
  const float top = v00 + (v10 - v00) * ax;
  const float bot = v01 + (v11 - v01) * ax;
  return top + (bot - top) * ay;
}

template <typename Dst>
__device__ __forceinline__ Dst emit(float v);
template <>
__device__ __forceinline__ Npp8u emit<Npp8u>(float v) {
  if (!(v > 0.0f)) return (Npp8u)0;
  if (v >= 255.0f) return (Npp8u)255;
  return (Npp8u)(int)v;
}
template <>
__device__ __forceinline__ Npp16u emit<Npp16u>(float v) {
  if (!(v > 0.0f)) return (Npp16u)0;
  if (v >= 65535.0f) return (Npp16u)65535;
  return (Npp16u)(int)v;
}
template <>
__device__ __forceinline__ Npp32f emit<Npp32f>(float v) {
  return v;
}

template <typename Src, typename Dst, int CH>
__global__ void pyramidDownKernel(const Src *pSrc, int nSrcStep, int srcW, int srcH, int offX,
                                  int offY, Dst *pDst, int nDstStep, int dstW, int dstH,
                                  const float *pK, int taps, float rate, int border) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstW || y >= dstH) return;
  const bool mirror = border == (int)NPP_BORDER_MIRROR;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  const int center = (taps - 1) / 2;
  for (int ch = 0; ch < CH; ++ch) {
    float acc = 0.0f;
    for (int s = 0; s < taps; ++s) {
      const float fy = (float)y * rate + (float)(s - center);
      float hsum = 0.0f;
      for (int t = 0; t < taps; ++t) {
        const float fx = (float)x * rate + (float)(t - center);
        hsum += pK[t] * sample2d<Src, CH>(pSrc, nSrcStep, fx, fy, ch, loX, hiX, loY, hiY, mirror);
      }
      acc += pK[s] * hsum;
    }
    Dst *row = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + (long long)y * nDstStep);
    row[(long long)x * CH + ch] = emit<Dst>(acc);
  }
}

// Best-effort upsampling model.  The reference's tap phase is driven by host
// uniforms and its exact Gaussian spreading was not decoded; the measured
// geometry is nearest-neighbour mapping `src = floor(dst / rate)` (exact for
// delta kernels), so this kernel applies the user taps in destination space
// over that nearest-neighbour upsampled image.  A unit-sum kernel therefore
// preserves constant images and delta kernels reproduce the reference mapping.
template <typename Src, typename Dst, int CH>
__global__ void pyramidUpKernel(const Src *pSrc, int nSrcStep, int srcW, int srcH, int offX,
                                int offY, Dst *pDst, int nDstStep, int dstW, int dstH,
                                const float *pK, int taps, float rate, int border) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstW || y >= dstH) return;
  const bool mirror = border == (int)NPP_BORDER_MIRROR;
  const int loX = -offX, hiX = srcW - 1 - offX;
  const int loY = -offY, hiY = srcH - 1 - offY;
  const int center = (taps - 1) / 2;
  const float invRate = 1.0f / rate;
  for (int ch = 0; ch < CH; ++ch) {
    float acc = 0.0f;
    for (int s = 0; s < taps; ++s) {
      const int sy = (int)floorf((float)(y + s - center) * invRate);
      float hsum = 0.0f;
      for (int t = 0; t < taps; ++t) {
        const int sx = (int)floorf((float)(x + t - center) * invRate);
        hsum += pK[t] * fetch<Src, CH>(pSrc, nSrcStep, sx, sy, ch, loX, hiX, loY, hiY, mirror);
      }
      acc += pK[s] * hsum;
    }
    Dst *row = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + (long long)y * nDstStep);
    row[(long long)x * CH + ch] = emit<Dst>(acc);
  }
}

}  // namespace

template <typename Src, typename Dst, int CH, PyramidDir Dir>
NppStatus gaussPyramidBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Dst *pDst, int nDstStep, NppiSize oSizeROI, Npp32f nRate,
                             int nFilterTaps, const Npp32f *pKernel, NppiBorderType eBorderType,
                             cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr || pKernel == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.y < 0 || oSrcOffset.x >= oSrcSize.width ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  // Measured on the 5090: NPP 13.4 reports an unsupported border type for
  // this family as -14 (the value of NPP_STEP_ERROR), not as
  // NPP_NOT_SUPPORTED_MODE_ERROR like the other filtering families.
  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_MIRROR) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  if (nFilterTaps < 3 || nFilterTaps >= 100) return NPP_MASK_SIZE_ERROR;
  if (!(nRate > 1.0f && nRate <= 10.0f)) return NPP_BAD_ARGUMENT_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  // Measured on the 5090: a destination step shorter than one ROI row comes
  // back as -16 (NPP_FFT_FLAG_ERROR, the code NPP 13.4 actually uses here),
  // while an unaligned step is the regular NPP_NOT_EVEN_STEP_ERROR (-108).
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if ((nDstStep % (int)sizeof(Dst)) != 0) return NPP_NOT_EVEN_STEP_ERROR;

  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nFilterTaps, stream, [&](const Npp32f *pk) {
    if (Dir == kPyramidDown) {
      pyramidDownKernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
          oSizeROI.width, oSizeROI.height, pk, nFilterTaps, nRate, (int)eBorderType);
    } else {
      pyramidUpKernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
          oSizeROI.width, oSizeROI.height, pk, nFilterTaps, nRate, (int)eBorderType);
    }
  });
}

#define NPPI_PYRAMID_INSTANTIATE(SRC, DST, CH, DIR)                               \
  template NppStatus gaussPyramidBorder<SRC, DST, CH, DIR>(                       \
      const SRC *, int, NppiSize, NppiPoint, DST *, int, NppiSize, Npp32f, int,   \
      const Npp32f *, NppiBorderType, cudaStream_t);

#define NPPI_PYRAMID_INSTANTIATE_BOTH(SRC, DST, CH)         \
  NPPI_PYRAMID_INSTANTIATE(SRC, DST, CH, kPyramidDown)      \
  NPPI_PYRAMID_INSTANTIATE(SRC, DST, CH, kPyramidUp)

NPPI_PYRAMID_INSTANTIATE_BOTH(Npp8u, Npp8u, 1)
NPPI_PYRAMID_INSTANTIATE_BOTH(Npp8u, Npp8u, 3)
NPPI_PYRAMID_INSTANTIATE_BOTH(Npp16u, Npp16u, 1)
NPPI_PYRAMID_INSTANTIATE_BOTH(Npp16u, Npp16u, 3)
NPPI_PYRAMID_INSTANTIATE_BOTH(Npp32f, Npp32f, 1)
NPPI_PYRAMID_INSTANTIATE_BOTH(Npp32f, Npp32f, 3)

}  // namespace filter
}  // namespace nppi
