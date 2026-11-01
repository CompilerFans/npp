// CIE XYZ / LUV color-space conversion kernels (package c4).
//
// The arithmetic follows the formulas published in nppi_color_conversion.h
// (documented algorithm for nppiRGBToXYZ / nppiXYZToRGB / nppiRGBToLUV /
// nppiLUVToRGB) and was verified element-exact against NVIDIA NPP 13.4 on the
// RTX 5090 for random 8-bit images with both C3 and AC4 layouts.

#include "nppi_color_space_executor.h"
#include "npp_kernel_utils.h"

using npp_kernel::BlockConfig;
using npp_kernel::getDstRow;
using npp_kernel::getSrcRow;
using npp_kernel::getThreadCoords;

namespace nppi {
namespace color_space {
namespace {

// CIE D65 white point constants used by both LUV directions.
constexpr float kD65xn = 0.312713F;
constexpr float kD65yn = 0.329016F;
constexpr float kDivisor = -2.0F * kD65xn + 12.0F * kD65yn + 3.0F;
constexpr float kUn = 4.0F * kD65xn / kDivisor;
constexpr float kVn = 9.0F * kD65yn / kDivisor;

__device__ __forceinline__ Npp8u truncate8(float value) { return (Npp8u)value; }

__device__ __forceinline__ void rgbToXyzPixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &x, Npp8u &y, Npp8u &z) {
  const float nR = (float)r * 0.003921569F;
  const float nG = (float)g * 0.003921569F;
  const float nB = (float)b * 0.003921569F;

  float nX = 0.412453F * nR + 0.35758F * nG + 0.180423F * nB;
  if (nX > 1.0F) {
    nX = 1.0F;
  }
  float nY = 0.212671F * nR + 0.71516F * nG + 0.072169F * nB;
  if (nY > 1.0F) {
    nY = 1.0F;
  }
  float nZ = 0.019334F * nR + 0.119193F * nG + 0.950227F * nB;
  if (nZ > 1.0F) {
    nZ = 1.0F;
  }

  x = truncate8(nX * 255.0F);
  y = truncate8(nY * 255.0F);
  z = truncate8(nZ * 255.0F);
}

__device__ __forceinline__ void xyzToRgbPixel(Npp8u x, Npp8u y, Npp8u z, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float nX = (float)x * 0.003921569F;
  const float nY = (float)y * 0.003921569F;
  const float nZ = (float)z * 0.003921569F;

  // NPP clamps to [0, 1]: the lower bound matters because negative linear RGB
  // is common, and a float->u8 cast of a negative value wraps on MACA.
  float nR = 3.240479F * nX - 1.53715F * nY - 0.498535F * nZ;
  nR = nR < 0.0F ? 0.0F : (nR > 1.0F ? 1.0F : nR);
  float nG = -0.969256F * nX + 1.875991F * nY + 0.041556F * nZ;
  nG = nG < 0.0F ? 0.0F : (nG > 1.0F ? 1.0F : nG);
  float nB = 0.055648F * nX - 0.204043F * nY + 1.057311F * nZ;
  nB = nB < 0.0F ? 0.0F : (nB > 1.0F ? 1.0F : nB);

  r = truncate8(nR * 255.0F);
  g = truncate8(nG * 255.0F);
  b = truncate8(nB * 255.0F);
}

__device__ __forceinline__ void rgbToLuvPixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &l, Npp8u &u8, Npp8u &v8) {
  const float nR = (float)r * 0.003921569F;
  const float nG = (float)g * 0.003921569F;
  const float nB = (float)b * 0.003921569F;

  const float nX = 0.412453F * nR + 0.35758F * nG + 0.180423F * nB;
  const float nY = 0.212671F * nR + 0.71516F * nG + 0.072169F * nB;
  const float nZ = 0.019334F * nR + 0.119193F * nG + 0.950227F * nB;

  float nTemp = nX + 15.0F * nY + 3.0F * nZ;
  const float nu = 4.0F * nX / nTemp;
  const float nv = 9.0F * nY / nTemp;
  float nL = 116.0F * cbrtf(nY) - 16.0F;
  if (nL < 0.0F) {
    nL = 0.0F;
  }
  if (nL > 100.0F) {
    nL = 100.0F;
  }
  nTemp = 13.0F * nL;
  float nU = nTemp * (nu - kUn);
  if (nU < -134.0F) {
    nU = -134.0F;
  }
  if (nU > 220.0F) {
    nU = 220.0F;
  }
  float nV = nTemp * (nv - kVn);
  if (nV < -140.0F) {
    nV = -140.0F;
  }
  if (nV > 122.0F) {
    nV = 122.0F;
  }

  l = truncate8(nL * 255.0F * 0.01F);
  u8 = truncate8((nU + 134.0F) * 255.0F * 0.0028249F);
  v8 = truncate8((nV + 140.0F) * 255.0F * 0.0038168F);
}

__device__ __forceinline__ void luvToRgbPixel(Npp8u l, Npp8u u8, Npp8u v8, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float nL = (float)l * 100.0F * 0.003921569F;
  const float nU = ((float)u8 * 354.0F * 0.003921569F) - 134.0F;
  const float nV = ((float)v8 * 262.0F * 0.003921569F) - 140.0F;

  const float nTemp = 13.0F * nL;
  const float nu = nU / nTemp + kUn;
  const float nv = nV / nTemp + kVn;

  float nNormalizedY;
  if (nL > 7.9996248F) {
    nNormalizedY = (nL + 16.0F) * 0.008621F;
    nNormalizedY = powf(nNormalizedY, 3.0F);
  } else {
    nNormalizedY = nL * 0.001107F;
  }
  const float nNormalizedX = (-9.0F * nNormalizedY * nu) / ((nu - 4.0F) * nv - nu * nv);
  const float nNormalizedZ =
      (9.0F * nNormalizedY - 15.0F * nv * nNormalizedY - nv * nNormalizedX) / (3.0F * nv);

  float nR = 3.240479F * nNormalizedX - 1.53715F * nNormalizedY - 0.498535F * nNormalizedZ;
  if (nR > 1.0F) {
    nR = 1.0F;
  }
  if (nR < 0.0F) {
    nR = 0.0F;
  }
  float nG = -0.969256F * nNormalizedX + 1.875991F * nNormalizedY + 0.041556F * nNormalizedZ;
  if (nG > 1.0F) {
    nG = 1.0F;
  }
  if (nG < 0.0F) {
    nG = 0.0F;
  }
  float nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
  if (nB > 1.0F) {
    nB = 1.0F;
  }
  if (nB < 0.0F) {
    nB = 0.0F;
  }

  r = truncate8(nR * 255.0F);
  g = truncate8(nG * 255.0F);
  b = truncate8(nB * 255.0F);
}

// Mode: 0 = RGB->XYZ, 1 = XYZ->RGB, 2 = RGB->LUV, 3 = LUV->RGB.
template <int Mode, int Channels>
__global__ void colorSpaceKernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  int x, y;
  if (!getThreadCoords(x, y, oSizeROI.width, oSizeROI.height)) {
    return;
  }
  const Npp8u *srcRow = getSrcRow(pSrc, y, nSrcStep);
  Npp8u *dstRow = getDstRow(pDst, y, nDstStep);
  const Npp8u *src = srcRow + x * Channels;
  Npp8u *dst = dstRow + x * Channels;

  Npp8u c0, c1, c2;
  if constexpr (Mode == 0) {
    rgbToXyzPixel(src[0], src[1], src[2], c0, c1, c2);
  } else if constexpr (Mode == 1) {
    xyzToRgbPixel(src[0], src[1], src[2], c0, c1, c2);
  } else if constexpr (Mode == 2) {
    rgbToLuvPixel(src[0], src[1], src[2], c0, c1, c2);
  } else {
    luvToRgbPixel(src[0], src[1], src[2], c0, c1, c2);
  }
  dst[0] = c0;
  dst[1] = c1;
  dst[2] = c2;
  // AC4: the destination alpha (dst[3]) is intentionally left untouched.
}

inline NppStatus validate(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                          int channels) {
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const bool ac4 = (channels == 4);
  if (nSrcStep < 0 || nDstStep < 0) {
    // NVIDIA reports success for negative steps and walks out-of-bounds rows;
    // AC4 layouts still require 4-byte-aligned steps before that early exit.
    if (ac4 && ((nSrcStep % 4) != 0 || (nDstStep % 4) != 0)) {
      return NPP_NOT_EVEN_STEP_ERROR;
    }
    return NPP_NO_ERROR;
  }
  const int minStep = oSizeROI.width * channels;
  if (nSrcStep < minStep || nDstStep < minStep) {
    return NPP_STEP_ERROR;
  }
  if (ac4 && ((nSrcStep % 4) != 0 || (nDstStep % 4) != 0)) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <int Mode, int Channels>
NppStatus run(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus status = validate(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, Channels);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  // Empty ROI and negative steps are successful no-ops (see validate()).
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  colorSpaceKernel<Mode, Channels><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  return npp_kernel::checkCudaKernelError();
}

}  // namespace

NppStatus rgbToXyz(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int channels,
                   cudaStream_t stream) {
  if (channels == 4) {
    return run<0, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
  }
  return run<0, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
}

NppStatus xyzToRgb(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int channels,
                   cudaStream_t stream) {
  if (channels == 4) {
    return run<1, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
  }
  return run<1, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
}

NppStatus rgbToLuv(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int channels,
                   cudaStream_t stream) {
  if (channels == 4) {
    return run<2, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
  }
  return run<2, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
}

NppStatus luvToRgb(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int channels,
                   cudaStream_t stream) {
  if (channels == 4) {
    return run<3, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
  }
  return run<3, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, stream);
}

}  // namespace color_space
}  // namespace nppi
