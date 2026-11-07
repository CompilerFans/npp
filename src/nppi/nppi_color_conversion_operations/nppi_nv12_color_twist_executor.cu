// NV12 <-> RGB ColorTwist kernels.
//
// Semantics were reverse-engineered from NVIDIA NPP 13.4 in the reference implementation with
// identity/half-scale/clipping twist matrices and verified element-exact by
// the reference comparisons

#include "nppi_nv12_color_twist_executor.h"
#include "npp_kernel_utils.h"

using npp_kernel::BlockConfig;
using npp_kernel::getDstRow;
using npp_kernel::getSrcRow;
using npp_kernel::getThreadCoords;

namespace nppi {
namespace nv12 {
namespace {

struct Twist {
  float m[12];
};

template <typename T> __host__ __device__ constexpr float typeMax() { return (sizeof(T) == 1) ? 255.0F: 65535.0F; }

__device__ __forceinline__ int roundHalfUp(float value) { return (int)floorf(value + 0.5F); }

template <typename T> __device__ __forceinline__ T roundClamp(float value) {
  float r = floorf(value + 0.5F);
  if (r < 0.0F) {
    r = 0.0F;
  } else if (r > typeMax<T>()) {
    r = typeMax<T>();
  }
  return (T)r;
}

__device__ __forceinline__ void applyTwist(const Twist &t, float c0, float c1, float c2, float &o0, float &o1,
                                           float &o2) {
  o0 = t.m[0] * c0 + t.m[1] * c1 + t.m[2] * c2 + t.m[3];
  o1 = t.m[4] * c0 + t.m[5] * c1 + t.m[6] * c2 + t.m[7];
  o2 = t.m[8] * c0 + t.m[9] * c1 + t.m[10] * c2 + t.m[11];
}

// Right-associated fused multiply-add chain.  This reproduces the reference
// library's per-pixel evaluation bit-exactly (the left-associated form differs
// in the last ULP and flips rare half-way chroma averages).
__device__ __forceinline__ void applyTwistNoFma(const Twist &t, float c0, float c1, float c2, float &o0, float &o1,
                                                float &o2) {
  o0 = __fmaf_rn(t.m[0], c0, __fmaf_rn(t.m[1], c1, __fmaf_rn(t.m[2], c2, t.m[3])));
  o1 = __fmaf_rn(t.m[4], c0, __fmaf_rn(t.m[5], c1, __fmaf_rn(t.m[6], c2, t.m[7])));
  o2 = __fmaf_rn(t.m[8], c0, __fmaf_rn(t.m[9], c1, __fmaf_rn(t.m[10], c2, t.m[11])));
}

inline Twist makeTwist(const Npp32f aTwist[3][4]) {
  Twist t{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.m[i * 4 + j] = aTwist[i][j];
    }
  }
  return t;
}

// ---------------------------------------------------------------- NV12 -> RGB

template <typename T>
__global__ void nv12ToRgbKernel(const T *pSrcY, int nSrcYStep, const T *pSrcUV, int nSrcUVStep, T *pDst,
                                int nDstStep, NppiSize oSizeROI, Twist twist) {
  int x, y;
  if (!getThreadCoords(x, y, oSizeROI.width, oSizeROI.height)) {
    return;
  }
  const T yValue = getSrcRow(pSrcY, y, nSrcYStep)[x];
  const T *uvRow = getSrcRow(pSrcUV, y >> 1, nSrcUVStep);
  const T uValue = uvRow[(x >> 1) << 1];
  const T vValue = uvRow[((x >> 1) << 1) + 1];

  float r, g, b;
  applyTwist(twist, (float)yValue, (float)uValue, (float)vValue, r, g, b);
  T *dst = getDstRow(pDst, y, nDstStep) + x * 3;
  dst[0] = roundClamp<T>(r);
  dst[1] = roundClamp<T>(g);
  dst[2] = roundClamp<T>(b);
}

// ---------------------------------------------------------------- RGB -> NV12

// Plane pointers/steps are passed as individual scalars: a kernel parameter of
// array-of-pointer type would be a host pointer when dereferenced on device.
template <typename T, bool Planar>
__device__ __forceinline__ void loadRgb(const T *pSrc, int nSrcStep, const T *pPlane0, const T *pPlane1,
                                        const T *pPlane2, int s0, int s1, int s2, int x, int y, float &r, float &g,
                                        float &b) {
  if constexpr (Planar) {
    r = (float)getSrcRow(pPlane0, y, s0)[x];
    g = (float)getSrcRow(pPlane1, y, s1)[x];
    b = (float)getSrcRow(pPlane2, y, s2)[x];
  } else {
    const T *row = getSrcRow(pSrc, y, nSrcStep);
    r = (float)row[x * 3 + 0];
    g = (float)row[x * 3 + 1];
    b = (float)row[x * 3 + 2];
  }
}

template <typename T, bool Planar>
__global__ void rgbToNv12Kernel(const T *pSrc, int nSrcStep, const T *pPlane0, const T *pPlane1, const T *pPlane2,
                                int s0, int s1, int s2, T *pDstY, int nDstYStep, T *pDstUV, int nDstUVStep,
                                NppiSize oSizeROI, Twist twist) {
  int x, y;
  if (!getThreadCoords(x, y, oSizeROI.width, oSizeROI.height)) {
    return;
  }
  float r, g, b, fy, fu, fv;
  loadRgb<T, Planar>(pSrc, nSrcStep, pPlane0, pPlane1, pPlane2, s0, s1, s2, x, y, r, g, b);
  // The Y channel uses the default (contracted) evaluation; the chroma block
  // below uses the right-associated FMA chain.  Both match the reference
  // library bit-exactly, including half-way rounding ties.
  applyTwist(twist, r, g, b, fy, fu, fv);
  getDstRow(pDstY, y, nDstYStep)[x] = roundClamp<T>(fy);

  // The ROI is even in both dimensions (odd sizes are rejected before the
  // launch), so the top-left pixel of every 2x2 block computes its chroma.
  if ((x & 1) == 0 && (y & 1) == 0) {
    // Chroma is averaged in floating point over the 2x2 block and rounded
    // half-up once, after the average (measured on the reference).
    float uSum = fu;
    float vSum = fv;
    const int dx[3] = {1, 0, 1};
    const int dy[3] = {0, 1, 1};
    for (int k = 0; k < 3; ++k) {
      loadRgb<T, Planar>(pSrc, nSrcStep, pPlane0, pPlane1, pPlane2, s0, s1, s2, x + dx[k], y + dy[k], r, g, b);
      applyTwistNoFma(twist, r, g, b, fy, fu, fv);
      uSum += fu;
      vSum += fv;
    }
    T *uvRow = getDstRow(pDstUV, y >> 1, nDstUVStep);
    uvRow[x] = roundClamp<T>(uSum * 0.25F);
    uvRow[x + 1] = roundClamp<T>(vSum * 0.25F);
  }
}

// ---------------------------------------------------------------- Validation

template <typename T>
NppStatus validateNv12ToRgb(const T *const pSrc[2], const int aSrcStep[2], T *pDst, int nDstStep, NppiSize roi,
                            const Npp32f aTwist[3][4]) {
  if (pSrc == nullptr || pSrc[0] == nullptr || pSrc[1] == nullptr || pDst == nullptr || aTwist == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (roi.width < 0 || roi.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (roi.width == 0 || roi.height == 0) {
    return NPP_NO_ERROR;
  }
  if ((roi.width & 1) != 0) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  const int rowBytes = roi.width * (int)sizeof(T);
  if (aSrcStep == nullptr || aSrcStep[0] < rowBytes || aSrcStep[1] < rowBytes ||
      nDstStep < roi.width * 3 * (int)sizeof(T)) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <typename T>
NppStatus validateRgbToNv12(NppiSize roi, int nSrcStep, const int aSrcStep[3], int planarPlanes,
                            const int aDstStep[2]) {
  if (roi.width < 0 || roi.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (roi.width == 0 || roi.height == 0) {
    return NPP_NO_ERROR;
  }
  if ((roi.width & 1) != 0 || (roi.height & 1) != 0) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  const int srcRowBytes = roi.width * 3 * (int)sizeof(T);
  if (planarPlanes > 0) {
    for (int i = 0; i < planarPlanes; ++i) {
      if (aSrcStep == nullptr || aSrcStep[i] < roi.width * (int)sizeof(T)) {
        return NPP_STEP_ERROR;
      }
    }
  } else if (nSrcStep < srcRowBytes) {
    return NPP_STEP_ERROR;
  }
  const int dstRowBytes = roi.width * (int)sizeof(T);
  if (aDstStep == nullptr || aDstStep[0] < dstRowBytes || aDstStep[1] < dstRowBytes) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <typename T, bool Planar>
NppStatus runRgbToNv12(const T *pSrc, int nSrcStep, const T *const pSrcPlanes[3], const int aSrcStep[3], T *pDst[2],
                       const int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], cudaStream_t stream) {
  if (pDst == nullptr || pDst[0] == nullptr || pDst[1] == nullptr || aTwist == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if constexpr (Planar) {
    if (pSrcPlanes == nullptr || pSrcPlanes[0] == nullptr || pSrcPlanes[1] == nullptr || pSrcPlanes[2] == nullptr) {
      return NPP_NULL_POINTER_ERROR;
    }
  } else {
    if (pSrc == nullptr) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  NppStatus status = validateRgbToNv12<T>(oSizeROI, nSrcStep, aSrcStep, Planar ? 3: 0, aDstStep);
  if (status != NPP_NO_ERROR) {
    return status;  // includes NPP_DOUBLE_SIZE_WARNING, which never launches
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(oSizeROI.width, oSizeROI.height);
  const T *p0 = Planar ? pSrcPlanes[0]: nullptr;
  const T *p1 = Planar ? pSrcPlanes[1]: nullptr;
  const T *p2 = Planar ? pSrcPlanes[2]: nullptr;
  const int s0 = (Planar && aSrcStep != nullptr) ? aSrcStep[0]: 0;
  const int s1 = (Planar && aSrcStep != nullptr) ? aSrcStep[1]: 0;
  const int s2 = (Planar && aSrcStep != nullptr) ? aSrcStep[2]: 0;
  rgbToNv12Kernel<T, Planar><<<grid, block, 0, stream>>>(pSrc, nSrcStep, p0, p1, p2, s0, s1, s2, pDst[0], aDstStep[0],
                                                         pDst[1], aDstStep[1], oSizeROI, makeTwist(aTwist));
  return npp_kernel::checkCudaKernelError();
}

}  // namespace

template <typename T>
NppStatus nv12ToRgbColorTwist(const T *const pSrc[2], const int aSrcStep[2], T *pDst, int nDstStep, NppiSize oSizeROI,
                              const Npp32f aTwist[3][4], cudaStream_t stream) {
  NppStatus status = validateNv12ToRgb<T>(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;  // empty ROI: successful no-op
  }
  if (oSizeROI.height == 1) {
    // NVIDIA fails the zero-sized chroma launch (NPP_CUDA_KERNEL_EXECUTION_ERROR)
    // and writes nothing; we report the same no-op as an empty ROI.
    return NPP_NO_ERROR;
  }
  // NVIDIA processes one whole 2x2 chroma row group: an odd ROI height writes
  // ceil(height / 2) * 2 output rows.
  const NppiSize procROI{oSizeROI.width, (oSizeROI.height + 1) & ~1};
  dim3 block = BlockConfig<>::blockSize();
  dim3 grid = BlockConfig<>::gridSize(procROI.width, procROI.height);
  nv12ToRgbKernel<T><<<grid, block, 0, stream>>>(pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pDst, nDstStep, procROI,
                                                 makeTwist(aTwist));
  return npp_kernel::checkCudaKernelError();
}

template <typename T>
NppStatus rgbToNv12ColorTwist(const T *pSrc, int nSrcStep, T *pDst[2], const int aDstStep[2], NppiSize oSizeROI,
                              const Npp32f aTwist[3][4], cudaStream_t stream) {
  return runRgbToNv12<T, false>(pSrc, nSrcStep, nullptr, nullptr, pDst, aDstStep, oSizeROI, aTwist, stream);
}

template <typename T>
NppStatus rgbToNv12ColorTwistPlanar(const T *const pSrc[3], const int aSrcStep[3], T *pDst[2],
                                    const int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                    cudaStream_t stream) {
  return runRgbToNv12<T, true>(nullptr, 0, pSrc, aSrcStep, pDst, aDstStep, oSizeROI, aTwist, stream);
}

template NppStatus nv12ToRgbColorTwist<Npp8u>(const Npp8u *const[2], const int[2], Npp8u *, int, NppiSize,
                                              const Npp32f[3][4], cudaStream_t);
template NppStatus nv12ToRgbColorTwist<Npp16u>(const Npp16u *const[2], const int[2], Npp16u *, int, NppiSize,
                                               const Npp32f[3][4], cudaStream_t);
template NppStatus rgbToNv12ColorTwist<Npp8u>(const Npp8u *, int, Npp8u *[2], const int[2], NppiSize,
                                              const Npp32f[3][4], cudaStream_t);
template NppStatus rgbToNv12ColorTwist<Npp16u>(const Npp16u *, int, Npp16u *[2], const int[2], NppiSize,
                                               const Npp32f[3][4], cudaStream_t);
template NppStatus rgbToNv12ColorTwistPlanar<Npp8u>(const Npp8u *const[3], const int[3], Npp8u *[2], const int[2],
                                                    NppiSize, const Npp32f[3][4], cudaStream_t);
template NppStatus rgbToNv12ColorTwistPlanar<Npp16u>(const Npp16u *const[3], const int[3], Npp16u *[2], const int[2],
                                                     NppiSize, const Npp32f[3][4], cudaStream_t);

}  // namespace nv12
}  // namespace nppi
