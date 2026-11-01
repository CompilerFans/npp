// Bayer CFA demosaic kernels (package c4).
//
// Integer bilinear interpolation with chroma correlation, reverse-engineered
// from NVIDIA NPP 13.4 on the RTX 5090 and verified element-exact by
// test/unit/diff/test_nppi_cfa_generated.cpp.  See nppi_cfa_executor.h for the
// measured semantics and the deliberate validation deviations.

#include "nppi_cfa_executor.h"
#include "npp_kernel_utils.h"

using npp_kernel::BlockConfig;
using npp_kernel::getDstRow;
using npp_kernel::getSrcRow;
using npp_kernel::getThreadCoords;

namespace nppi {
namespace cfa {
namespace {

// Bayer colour of the pixel at (x, y) relative to the ROI origin.  0 = R,
// 1 = G, 2 = B.  Layout order is p = (y & 1) * 2 + (x & 1):
//   BGGR -> {2,1,1,0}   RGGB -> {0,1,1,2}
//   GBRG -> {1,2,0,1}   GRBG -> {1,0,2,1}
// (written as a switch so the table lives in code, not in host-only data).
__device__ __forceinline__ int colorAt(int grid, int x, int y) {
  const int p = ((y & 1) << 1) | (x & 1);
  switch (grid) {
  case 0: return (p == 0) ? 2 : ((p == 3) ? 0 : 1);  // NPPI_BAYER_BGGR
  case 1: return (p == 0) ? 0 : ((p == 3) ? 2 : 1);  // NPPI_BAYER_RGGB
  case 2: return (p == 0 || p == 3) ? 1 : ((p == 1) ? 2 : 0);  // NPPI_BAYER_GBRG
  default: return (p == 0 || p == 3) ? 1 : ((p == 1) ? 0 : 2);  // NPPI_BAYER_GRBG
  }
}

__device__ __forceinline__ int mirrorIndex(int i, int n) {
  if (i < 0) {
    return -i;
  }
  if (i >= n) {
    return 2 * n - 2 - i;
  }
  return i;
}

using Wide = unsigned long long;

template <typename T> struct SrcView {
  const T *p;
  int step;
  int width;
  int height;

  __device__ __forceinline__ T at(int x, int y) const {
    return getSrcRow(p, mirrorIndex(y, height), step)[mirrorIndex(x, width)];
  }
};

// Truncating two-sample average (NVIDIA floors every stage).
template <typename T> __device__ __forceinline__ T avg2(T a, T b) {
  return (T)(((Wide)a + (Wide)b) >> 1);
}

// Diagonal R/B average: even output rows pair the columns, odd rows pair the
// rows; each pairwise average is truncated.
template <typename T> __device__ __forceinline__ T avg4Diagonal(int y, T lt, T rt, T lb, T rb) {
  if ((y & 1) == 0) {
    return avg2(avg2(lt, lb), avg2(rt, rb));
  }
  return avg2(avg2(lt, rt), avg2(lb, rb));
}

template <typename T, int OutChannels>
__global__ void cfaKernel(const T *pSrc, int nSrcStep, NppiSize srcSize, NppiRect roi, T *pDst, int nDstStep,
                          int grid, T alpha) {
  int ox, oy;
  if (!getThreadCoords(ox, oy, roi.width - roi.x, roi.height - roi.y)) {
    return;
  }
  const SrcView<T> src{pSrc, nSrcStep, srcSize.width, srcSize.height};
  const int ax = roi.x + ox;
  const int ay = roi.y + oy;
  const int color = colorAt(grid, ox, oy);
  const T base = src.at(ax, ay);

  T r = 0, g = 0, b = 0;
  if (color == 1) {
    // Green base pixel: R and B come from the horizontal or vertical pair.
    g = base;
    const bool rHorizontal = (colorAt(grid, ox - 1, oy) == 0) || (colorAt(grid, ox + 1, oy) == 0);
    r = rHorizontal ? avg2<T>(src.at(ax - 1, ay), src.at(ax + 1, ay))
                    : avg2<T>(src.at(ax, ay - 1), src.at(ax, ay + 1));
    const bool bHorizontal = (colorAt(grid, ox - 1, oy) == 2) || (colorAt(grid, ox + 1, oy) == 2);
    b = bHorizontal ? avg2<T>(src.at(ax - 1, ay), src.at(ax + 1, ay))
                    : avg2<T>(src.at(ax, ay - 1), src.at(ax, ay + 1));
  } else {
    // Red or blue base pixel: chroma-correlated green plus the diagonal
    // average of the other colour.
    const long long c2 = 2LL * (long long)base;
    const long long diffH = c2 - ((long long)src.at(ax - 2, ay) + (long long)src.at(ax + 2, ay));
    const long long diffV = c2 - ((long long)src.at(ax, ay - 2) + (long long)src.at(ax, ay + 2));
    const long long dH = diffH < 0 ? -diffH : diffH;
    const long long dV = diffV < 0 ? -diffV : diffV;
    const T gl = src.at(ax - 1, ay);
    const T gr = src.at(ax + 1, ay);
    const T gu = src.at(ax, ay - 1);
    const T gd = src.at(ax, ay + 1);
    if (dH < dV) {
      g = avg2<T>(gl, gr);
    } else if (dV < dH) {
      g = avg2<T>(gu, gd);
    } else {
      g = avg2<T>(avg2<T>(gl, gr), avg2<T>(gu, gd));
    }
    const T other = avg4Diagonal<T>(oy, src.at(ax - 1, ay - 1), src.at(ax + 1, ay - 1), src.at(ax - 1, ay + 1),
                                    src.at(ax + 1, ay + 1));
    if (color == 0) {
      r = base;
      b = other;
    } else {
      b = base;
      r = other;
    }
  }

  T *dst = getDstRow(pDst, oy, nDstStep) + ox * OutChannels;
  dst[0] = r;
  dst[1] = g;
  dst[2] = b;
  if constexpr (OutChannels == 4) {
    dst[3] = alpha;
  }
}

template <typename T, int OutChannels>
NppStatus run(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *pDst, int nDstStep,
              NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation, T alpha, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (eInterpolation != NPPI_INTER_UNDEFINED) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (eGrid < NPPI_BAYER_BGGR || eGrid > NPPI_BAYER_GRBG) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (oSrcROI.x < 0 || oSrcROI.y < 0) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSrcROI.x & 1) != 0 || (oSrcROI.y & 1) != 0 || (oSrcROI.width & 1) != 0 || (oSrcROI.height & 1) != 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSrcROI.width < oSrcROI.x || oSrcROI.height < oSrcROI.y) {
    return NPP_SIZE_ERROR;
  }
  if (oSrcROI.x + oSrcROI.width > oSrcSize.width || oSrcROI.y + oSrcROI.height > oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }

  const int procWidth = oSrcROI.width - oSrcROI.x;
  const int procHeight = oSrcROI.height - oSrcROI.y;
  if (nDstStep < 0 || nSrcStep < 0) {
    return NPP_NO_ERROR;  // NVIDIA accepts negative steps and walks OOB rows
  }
  if (procWidth <= 0 || procHeight <= 0) {
    return NPP_NO_ERROR;
  }
  if (nDstStep < procWidth * OutChannels * (int)sizeof(T)) {
    return NPP_STEP_ERROR;
  }
  // The destination step must be a whole number of elements (RGBA also
  // requires whole 4-element pixels).
  const int stepAlign = (OutChannels == 4 ? 4 : 1) * (int)sizeof(T);
  if ((nDstStep % stepAlign) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  // NVIDIA never validates the source step; requiring a full row keeps the
  // kernel inside the caller's buffer (documented deviation).
  if (nSrcStep < procWidth * (int)sizeof(T)) {
    return NPP_STEP_ERROR;
  }

  const NppiRect procROI{0, 0, procWidth, procHeight};
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(procWidth, procHeight);
  cfaKernel<T, OutChannels><<<grid, block, 0, stream>>>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep,
                                                        (int)eGrid, alpha);
  return npp_kernel::checkCudaKernelError();
}

}  // namespace

#define NPPI_CFA_DEFINE_RGB(T)                                                                                         \
  NppStatus cfaToRgb(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *pDst, int nDstStep,           \
                     NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation, cudaStream_t stream) {          \
    return run<T, 3>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, eGrid, eInterpolation, (T)0, stream);           \
  }

#define NPPI_CFA_DEFINE_RGBA(T)                                                                                        \
  NppStatus cfaToRgba(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, T *pDst, int nDstStep,          \
                      NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation, T nAlpha,                      \
                      cudaStream_t stream) {                                                                           \
    return run<T, 4>(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, eGrid, eInterpolation, nAlpha, stream);         \
  }

NPPI_CFA_DEFINE_RGB(Npp8u)
NPPI_CFA_DEFINE_RGB(Npp16u)
NPPI_CFA_DEFINE_RGB(Npp32u)
NPPI_CFA_DEFINE_RGBA(Npp8u)
NPPI_CFA_DEFINE_RGBA(Npp16u)
NPPI_CFA_DEFINE_RGBA(Npp32u)

#undef NPPI_CFA_DEFINE_RGB
#undef NPPI_CFA_DEFINE_RGBA

}  // namespace cfa
}  // namespace nppi
