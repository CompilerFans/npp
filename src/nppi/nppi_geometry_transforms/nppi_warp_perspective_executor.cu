// nppiWarpPerspective / Back / Quad / Batch family executor: kernels, entry
// point templates and their explicit instantiations.  See
// nppi_warp_perspective_executor.h for the measured semantics.

#include "nppi_warp_perspective_executor.h"
#include "nppi_warp_sampling.cuh"
#include "npp_kernel_utils.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace nppi {
namespace warp {
namespace {

__device__ __forceinline__ void mapPoint(const Mat3 &M, double x, double y, double &u, double &v, double &den) {
  den = M.m[2][0] * x + M.m[2][1] * y + M.m[2][2];
  const double inv = 1.0 / den;
  u = (M.m[0][0] * x + M.m[0][1] * y + M.m[0][2]) * inv;
  v = (M.m[1][0] * x + M.m[1][1] * y + M.m[1][2]) * inv;
}

// Per-pixel region test shared by every direction.
__device__ __forceinline__ bool pixelGate(const Mat3 &M, NppiRect src, int x, int y, double &u, double &v) {
  double den;
  mapPoint(M, (double)x, (double)y, u, v, den);
  if (!(den > 0.0)) return false;
  const double uhi = (double)src.x + src.width - 1.0;
  const double vhi = (double)src.y + src.height - 1.0;
  return u >= (double)src.x && u <= uhi && v >= (double)src.y && v <= vhi;
}

// Forward (source -> destination coefficients) region model: the x span of the
// source ROI's image at the row's left-edge v, then the per-pixel gate.  The
// batch kernel additionally drops whole rows whose left-edge v falls outside
// the source ROI (`gateRows`), which the single-image kernel does not do
// (measured: batch n3/persp skips dst row 1, planar persp_subroi_cubic still
// writes dst row 3 with v_row < srcROI.y).
__device__ __forceinline__ bool forwardGate(const Mat3 &M, const Mat3 &F, NppiRect src, int x, int y, bool gateRows,
                                            double &u, double &v) {
  const double left = (double)src.x;
  const double uhi = (double)src.x + src.width - 1.0;
  double denA, uA, vRow;
  mapPoint(M, left, (double)y, uA, vRow, denA);
  (void)uA;
  if (!(denA > 0.0)) return false;
  if (gateRows && (vRow < (double)src.y || vRow > (double)src.y + src.height)) return false;
  double d0, d1, xl, unused;
  mapPoint(F, left, vRow, xl, unused, d0);
  if (!(d0 > 0.0)) return false;
  double xr, unused2;
  mapPoint(F, uhi, vRow, xr, unused2, d1);
  if (!(d1 > 0.0)) return false;
  if ((double)x < ceil(xl) || (double)x > floor(xr)) return false;
  return pixelGate(M, src, x, y, u, v);
}

// Source quadrilateral passed by value.  Kernel parameters declared as plain
// arrays decay to pointers (the device would dereference a host address), so
// the eight corners travel inside a struct.
struct QuadF {
  float q[8];
};

// Half-plane test of the source quadrilateral in float32 (measured: NVIDIA
// evaluates the quad edges in reduced precision).
__device__ __forceinline__ bool insideQuadF32(const QuadF &quad_, float u, float v) {
  const float *q = quad_.q;
  bool allPos = true;
  bool allNeg = true;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) & 3;
    const float ax = q[2 * i], ay = q[2 * i + 1];
    const float bx = q[2 * j], by = q[2 * j + 1];
    const float t1 = fmul_rn(fsub_rn(bx, ax), fsub_rn(v, ay));
    const float t2 = fmul_rn(fsub_rn(by, ay), fsub_rn(u, ax));
    const float cross = fsub_rn(t1, t2);
    allPos = allPos && (cross >= 0.0f);
    allNeg = allNeg && (cross <= 0.0f);
  }
  return allPos || allNeg;
}

enum class Direction { Forward, Backward, Quad };

// Diagnostic helper: the CI/dev builds may define NPPI_GEOM_DEBUG_SYNC to
// surface asynchronous kernel faults as status errors (NPP itself stays
// asynchronous, so this is off by default).
inline NppStatus checkLaunch(cudaStream_t stream) {
  const NppStatus st = npp_kernel::checkCudaKernelError();
  if (st != NPP_NO_ERROR) return st;
#ifdef NPPI_GEOM_DEBUG_SYNC
  if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
#endif
  return NPP_NO_ERROR;
}

template <typename T, int C, bool AP, Direction Dir>
__global__ void warpPackedKernel(const T *__restrict__ pSrc, int nSrcStepEl, NppiSize oSrcSize, NppiRect oSrcROI,
                                 T *__restrict__ pDst, int nDstStepEl, NppiRect oDstROI, Mat3 M, Mat3 F,
                                 QuadF quad, int eInterpolation, bool gateRows) {
  const int rx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ry = blockIdx.y * blockDim.y + threadIdx.y;
  if (rx >= oDstROI.width || ry >= oDstROI.height) return;
  const int x = rx + oDstROI.x;
  const int y = ry + oDstROI.y;
  double u = 0.0, v = 0.0;
  if (Dir == Direction::Forward) {
    if (!forwardGate(M, F, oSrcROI, x, y, gateRows, u, v)) return;
  } else if (Dir == Direction::Backward) {
    if (!pixelGate(M, oSrcROI, x, y, u, v)) return;
  } else {
    double den;
    mapPoint(M, (double)x, (double)y, u, v, den);
    if (!(den > 0.0)) return;
    if (!(u >= 0.0 && u <= (double)oSrcSize.width - 1.0 && v >= 0.0 && v <= (double)oSrcSize.height - 1.0))
      return;
    if (!insideQuadF32(quad, (float)u, (float)v)) return;
  }
  T *dstPixel = pDst + (size_t)y * nDstStepEl + (size_t)x * C;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    if (AP && c == 3) continue;
    dstPixel[c] = sampleChannel<T>(pSrc + c, nSrcStepEl, C, oSrcSize.width, oSrcSize.height, u, v, eInterpolation);
  }
}

// Plane pointers travel by value: a kernel parameter declared as a pointer to
// an array of pointers would dereference the caller's host array on the device
// (illegal address / garbage data).  NVIDIA NPP takes the array on the host and
// stages the pointers itself; mcMPP copies them into the launch parameters.
template <typename T, int P> struct PlaneSet {
  const T *src[P];
  T *dst[P];
};

template <typename T, int P, Direction Dir>
__global__ void warpPlanarKernel(PlaneSet<T, P> planes, int nSrcStepEl, NppiSize oSrcSize, NppiRect oSrcROI,
                                 int nDstStepEl, NppiRect oDstROI, Mat3 M, Mat3 F, QuadF quad, int eInterpolation,
                                 bool gateRows) {
  const int rx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ry = blockIdx.y * blockDim.y + threadIdx.y;
  if (rx >= oDstROI.width || ry >= oDstROI.height) return;
  const int x = rx + oDstROI.x;
  const int y = ry + oDstROI.y;
  double u = 0.0, v = 0.0;
  if (Dir == Direction::Forward) {
    if (!forwardGate(M, F, oSrcROI, x, y, gateRows, u, v)) return;
  } else if (Dir == Direction::Backward) {
    if (!pixelGate(M, oSrcROI, x, y, u, v)) return;
  } else {
    double den;
    mapPoint(M, (double)x, (double)y, u, v, den);
    if (!(den > 0.0)) return;
    if (!(u >= 0.0 && u <= (double)oSrcSize.width - 1.0 && v >= 0.0 && v <= (double)oSrcSize.height - 1.0))
      return;
    if (!insideQuadF32(quad, (float)u, (float)v)) return;
  }
#pragma unroll
  for (int p = 0; p < P; ++p) {
    const T *srcPlane = planes.src[p];
    T *dstPlane = planes.dst[p];
    dstPlane[(size_t)y * nDstStepEl + x] =
        sampleChannel<T>(srcPlane, nSrcStepEl, 1, oSrcSize.width, oSrcSize.height, u, v, eInterpolation);
  }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

inline bool interpIsValid(int eInterpolation) {
  return eInterpolation == NPPI_INTER_NN || eInterpolation == NPPI_INTER_LINEAR || eInterpolation == NPPI_INTER_CUBIC;
}

inline bool planarPtrsValid(const void *const *p, int planes) {
  if (!p) return false;
  for (int i = 0; i < planes; ++i)
    if (!p[i]) return false;
  return true;
}

// Common validation for the perspective/back entry points (measured order).
inline NppStatus validateWarp(const void *pSrc, NppiSize oSrcSize, NppiRect oSrcROI, const void *pDst, int nDstStep,
                              int eInterpolation) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width < 0 || oSrcSize.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width == 0 || oSrcSize.height == 0) return NPP_SIZE_ERROR;
  if (nDstStep <= 0) return NPP_STEP_ERROR;
  if (!interpIsValid(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (oSrcROI.x < 0 || oSrcROI.y < 0) return NPP_SIZE_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) return NPP_RECTANGLE_ERROR;
  return NPP_NO_ERROR;
}

inline NppStatus validateQuadCall(const void *pSrc, NppiSize oSrcSize, NppiRect oSrcROI, const double aSrcQuad[4][2],
                                  const void *pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2],
                                  int eInterpolation) {
  // Quads first: the order_* probes show NPP_QUADRANGLE_ERROR wins over every
  // other contract error.
  if (isDegenerateQuad(aSrcQuad) || isDegenerateQuad(aDstQuad)) return NPP_QUADRANGLE_ERROR;
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (nDstStep <= 0) return NPP_STEP_ERROR;
  if (!interpIsValid(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0 || oDstROI.width <= 0 || oDstROI.height <= 0)
    return NPP_WRONG_INTERSECTION_QUAD_WARNING;
  return NPP_NO_ERROR;
}

inline void quadsToFloat(const double srcQuad[4][2], float out[8]) {
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = (float)srcQuad[i][0];
    out[2 * i + 1] = (float)srcQuad[i][1];
  }
}

template <typename T, int C, bool AP, Direction Dir>
inline NppStatus launchPackedT(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst,
                               int nDstStep, NppiRect oDstROI, const Mat3 &M, const Mat3 &F, const float quad[8],
                               int eInterpolation, bool gateRows, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((oDstROI.width + block.x - 1) / block.x, (oDstROI.height + block.y - 1) / block.y);
  if (grid.x == 0 || grid.y == 0) return NPP_NO_ERROR;
  QuadF q{};
  if (quad)
    for (int i = 0; i < 8; ++i) q.q[i] = quad[i];
  warpPackedKernel<T, C, AP, Dir><<<grid, block, 0, stream>>>(pSrc, nSrcStep / (int)sizeof(T), oSrcSize, oSrcROI, pDst,
                                                             nDstStep / (int)sizeof(T), oDstROI, M, F, q,
                                                             eInterpolation, gateRows);
  return checkLaunch(stream);
}

template <typename T, int P, Direction Dir>
inline NppStatus launchPlanarT(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                               T *const pDst[], int nDstStep, NppiRect oDstROI, const Mat3 &M, const Mat3 &F,
                               const float quad[8], int eInterpolation, bool gateRows, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((oDstROI.width + block.x - 1) / block.x, (oDstROI.height + block.y - 1) / block.y);
  if (grid.x == 0 || grid.y == 0) return NPP_NO_ERROR;
  QuadF q{};
  if (quad)
    for (int i = 0; i < 8; ++i) q.q[i] = quad[i];
  PlaneSet<T, P> planes{};
  for (int i = 0; i < P; ++i) {
    planes.src[i] = pSrc[i];
    planes.dst[i] = pDst[i];
  }
  warpPlanarKernel<T, P, Dir><<<grid, block, 0, stream>>>(planes, nSrcStep / (int)sizeof(T), oSrcSize, oSrcROI,
                                                          nDstStep / (int)sizeof(T), oDstROI, M, F, q, eInterpolation,
                                                          gateRows);
  return checkLaunch(stream);
}

} // namespace

// ---------------------------------------------------------------------------
// Quad homography solver + degenerate detection
// ---------------------------------------------------------------------------

bool solveQuadHomography(const double aSrcQuad[4][2], const double aDstQuad[4][2], Mat3 &out) {
  double a[8][9] = {};
  for (int i = 0; i < 4; ++i) {
    const double x = aSrcQuad[i][0];
    const double y = aSrcQuad[i][1];
    const double X = aDstQuad[i][0];
    const double Y = aDstQuad[i][1];
    a[2 * i][0] = x;
    a[2 * i][1] = y;
    a[2 * i][2] = 1.0;
    a[2 * i][6] = -X * x;
    a[2 * i][7] = -X * y;
    a[2 * i][8] = X;
    a[2 * i + 1][3] = x;
    a[2 * i + 1][4] = y;
    a[2 * i + 1][5] = 1.0;
    a[2 * i + 1][6] = -Y * x;
    a[2 * i + 1][7] = -Y * y;
    a[2 * i + 1][8] = Y;
  }
  for (int col = 0; col < 8; ++col) {
    int pivot = col;
    for (int r = col + 1; r < 8; ++r)
      if (fabs(a[r][col]) > fabs(a[pivot][col])) pivot = r;
    if (a[pivot][col] == 0.0) return false;
    if (pivot != col)
      for (int k = 0; k < 9; ++k) {
        const double t = a[col][k];
        a[col][k] = a[pivot][k];
        a[pivot][k] = t;
      }
    const double p = a[col][col];
    for (int r = 0; r < 8; ++r) {
      if (r == col) continue;
      const double f = a[r][col] / p;
      if (f == 0.0) continue;
      for (int k = col; k < 9; ++k) a[r][k] -= f * a[col][k];
    }
  }
  double h[8];
  for (int i = 0; i < 8; ++i) h[i] = a[i][8] / a[i][i];
  out.m[0][0] = h[0];
  out.m[0][1] = h[1];
  out.m[0][2] = h[2];
  out.m[1][0] = h[3];
  out.m[1][1] = h[4];
  out.m[1][2] = h[5];
  out.m[2][0] = h[6];
  out.m[2][1] = h[7];
  out.m[2][2] = 1.0;
  return true;
}

bool isDegenerateQuad(const double q[4][2]) {
  // NPP rejects quads with a duplicated vertex or with collinear edges
  // (NPP_QUADRANGLE_ERROR for both the duplicate and the collinear probes).
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      if (q[i][0] == q[j][0] && q[i][1] == q[j][1]) return true;
  // Any three collinear vertices leave the quad without area.
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) & 3;
    const int k = (i + 2) & 3;
    const double ax = q[j][0] - q[i][0];
    const double ay = q[j][1] - q[i][1];
    const double bx = q[k][0] - q[i][0];
    const double by = q[k][1] - q[i][1];
    if (ax * by - ay * bx == 0.0) return true;
  }
  return false;
}

namespace {

inline NppStatus quadSetup(const double aSrcQuad[4][2], const double aDstQuad[4][2], Mat3 &M, Mat3 &F, float quad[8]) {
  if (!solveQuadHomography(aSrcQuad, aDstQuad, F)) return NPP_QUADRANGLE_ERROR;
  const double d = det3(F);
  if (d == 0.0) return NPP_QUADRANGLE_ERROR;
  M = inverse3(F);
  quadsToFloat(aSrcQuad, quad);
  return NPP_NO_ERROR;
}

} // namespace

// ---------------------------------------------------------------------------
// Entry point templates
// ---------------------------------------------------------------------------

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpForward(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                      NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, cudaStream_t stream) {
  const NppStatus st = validateWarp(pSrc, oSrcSize, oSrcROI, pDst, nDstStep, eInterpolation);
  if (st != NPP_NO_ERROR) return st;
  const Mat3 F = mat3FromCoeffs(aCoeffs);
  if (det3(F) == 0.0) return NPP_COEFFICIENT_ERROR;
  const Mat3 M = inverse3(F);
  if (oDstROI.width <= 0 || oDstROI.height <= 0) return NPP_NO_ERROR;
  return launchPackedT<T, Channels, AlphaPreserve, Direction::Forward>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep,
                                                                      oDstROI, M, F, nullptr, eInterpolation, false, stream);
}

template <typename T, int Planes>
NppStatus warpForwardPlanar(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *const pDst[],
                            int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation,
                            cudaStream_t stream) {
  if (!planarPtrsValid((const void *const *)pSrc, Planes) || !planarPtrsValid((void *const *)pDst, Planes))
    return NPP_NULL_POINTER_ERROR;
  const NppStatus st = validateWarp(pSrc[0], oSrcSize, oSrcROI, pDst[0], nDstStep, eInterpolation);
  if (st != NPP_NO_ERROR) return st;
  const Mat3 F = mat3FromCoeffs(aCoeffs);
  if (det3(F) == 0.0) return NPP_COEFFICIENT_ERROR;
  const Mat3 M = inverse3(F);
  if (oDstROI.width <= 0 || oDstROI.height <= 0) return NPP_NO_ERROR;
  return launchPlanarT<T, Planes, Direction::Forward>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, M, F,
                                                      nullptr, eInterpolation, false, stream);
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpBackward(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                       NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, cudaStream_t stream) {
  const NppStatus st = validateWarp(pSrc, oSrcSize, oSrcROI, pDst, nDstStep, eInterpolation);
  if (st != NPP_NO_ERROR) return st;
  const Mat3 M = mat3FromCoeffs(aCoeffs);
  const Mat3 F = inverse3(M);
  if (oDstROI.width <= 0 || oDstROI.height <= 0) return NPP_NO_ERROR;
  return launchPackedT<T, Channels, AlphaPreserve, Direction::Backward>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst,
                                                                       nDstStep, oDstROI, M, F, nullptr, eInterpolation,
                                                                       false, stream);
}

template <typename T, int Planes>
NppStatus warpBackwardPlanar(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *const pDst[],
                             int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation,
                             cudaStream_t stream) {
  if (!planarPtrsValid((const void *const *)pSrc, Planes) || !planarPtrsValid((void *const *)pDst, Planes))
    return NPP_NULL_POINTER_ERROR;
  const NppStatus st = validateWarp(pSrc[0], oSrcSize, oSrcROI, pDst[0], nDstStep, eInterpolation);
  if (st != NPP_NO_ERROR) return st;
  const Mat3 M = mat3FromCoeffs(aCoeffs);
  const Mat3 F = inverse3(M);
  if (oDstROI.width <= 0 || oDstROI.height <= 0) return NPP_NO_ERROR;
  return launchPlanarT<T, Planes, Direction::Backward>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, M, F,
                                                       nullptr, eInterpolation, false, stream);
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpQuad(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2],
                   T *pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
                   cudaStream_t stream) {
  const NppStatus st = validateQuadCall(pSrc, oSrcSize, oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad,
                                        eInterpolation);
  if (st != NPP_NO_ERROR) return st;
  Mat3 M, F;
  float quad[8];
  const NppStatus qs = quadSetup(aSrcQuad, aDstQuad, M, F, quad);
  if (qs != NPP_NO_ERROR) return qs;
  return launchPackedT<T, Channels, AlphaPreserve, Direction::Quad>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep,
                                                                    oDstROI, M, F, quad, eInterpolation, false,
                                                                    stream);
}

template <typename T, int Planes>
NppStatus warpQuadPlanar(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                         const double aSrcQuad[4][2], T *const pDst[], int nDstStep, NppiRect oDstROI,
                         const double aDstQuad[4][2], int eInterpolation, cudaStream_t stream) {
  if (!planarPtrsValid((const void *const *)pSrc, Planes) || !planarPtrsValid((void *const *)pDst, Planes))
    return NPP_NULL_POINTER_ERROR;
  const NppStatus st = validateQuadCall(pSrc[0], oSrcSize, oSrcROI, aSrcQuad, pDst[0], nDstStep, oDstROI, aDstQuad,
                                        eInterpolation);
  if (st != NPP_NO_ERROR) return st;
  Mat3 M, F;
  float quad[8];
  const NppStatus qs = quadSetup(aSrcQuad, aDstQuad, M, F, quad);
  if (qs != NPP_NO_ERROR) return qs;
  return launchPlanarT<T, Planes, Direction::Quad>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, M, F,
                                                   quad, eInterpolation, false, stream);
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

NppStatus batchInit(NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream) {
  if (!pBatchList) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize == 0) return NPP_CUDA_KERNEL_EXECUTION_ERROR; // measured: NVIDIA launches an empty grid
  std::vector<NppiWarpPerspectiveBatchCXR> host(nBatchSize);
  if (cudaMemcpyAsync(host.data(), pBatchList, sizeof(NppiWarpPerspectiveBatchCXR) * nBatchSize,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  for (unsigned int i = 0; i < nBatchSize; ++i) {
    double coeffs[3][3];
    if (cudaMemcpyAsync(coeffs, host[i].pCoeffs, sizeof(coeffs), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    const Mat3 inv = inverse3(mat3FromCoeffs(coeffs));
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) host[i].aTransformedCoeffs[r][c] = inv.m[r][c];
  }
  if (cudaMemcpyAsync(pBatchList, host.data(), sizeof(NppiWarpPerspectiveBatchCXR) * nBatchSize,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_SUCCESS;
}

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation,
                    NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream) {
  if (!pBatchList) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize < 2) return NPP_BAD_ARGUMENT_ERROR;
  if (oSmallestSrcSize.width <= 0 || oSmallestSrcSize.height <= 0) return NPP_RECTANGLE_ERROR;
  if (oSrcRectROI.x < 0 || oSrcRectROI.y < 0) return NPP_SIZE_ERROR;
  if (oSrcRectROI.width <= 0 || oSrcRectROI.height <= 0) return NPP_RECTANGLE_ERROR;
  if (oDstRectROI.width <= 0 || oDstRectROI.height <= 0) return NPP_RECTANGLE_ERROR;
  if (!interpIsValid(eInterpolation)) return NPP_INTERPOLATION_ERROR;
  std::vector<NppiWarpPerspectiveBatchCXR> host(nBatchSize);
  if (cudaMemcpyAsync(host.data(), pBatchList, sizeof(NppiWarpPerspectiveBatchCXR) * nBatchSize,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  for (unsigned int i = 0; i < nBatchSize; ++i) {
    if (!host[i].pSrc || !host[i].pDst) return NPP_NULL_POINTER_ERROR;
    double coeffs[3][3];
    if (cudaMemcpyAsync(coeffs, host[i].pCoeffs, sizeof(coeffs), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    const Mat3 F = mat3FromCoeffs(coeffs);
    Mat3 M;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) M.m[r][c] = host[i].aTransformedCoeffs[r][c];
    const NppStatus st = launchPackedT<T, Channels, AlphaPreserve, Direction::Forward>(
        (const T *)host[i].pSrc, oSmallestSrcSize, host[i].nSrcStep, oSrcRectROI, (T *)host[i].pDst, host[i].nDstStep,
        oDstRectROI, M, F, nullptr, eInterpolation, true, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_SUCCESS;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define NPPI_GEOM_INSTANTIATE(T)                                                                                      \
  template NppStatus warpForward<T, 1, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, const double[3][3], \
                                              int, cudaStream_t);                                                      \
  template NppStatus warpForward<T, 3, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, const double[3][3], \
                                              int, cudaStream_t);                                                      \
  template NppStatus warpForward<T, 4, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, const double[3][3], \
                                              int, cudaStream_t);                                                      \
  template NppStatus warpForward<T, 4, true>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, const double[3][3],  \
                                             int, cudaStream_t);                                                       \
  template NppStatus warpBackward<T, 1, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect,                   \
                                               const double[3][3], int, cudaStream_t);                                 \
  template NppStatus warpBackward<T, 3, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect,                   \
                                               const double[3][3], int, cudaStream_t);                                 \
  template NppStatus warpBackward<T, 4, false>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect,                   \
                                               const double[3][3], int, cudaStream_t);                                 \
  template NppStatus warpBackward<T, 4, true>(const T *, NppiSize, int, NppiRect, T *, int, NppiRect,                    \
                                              const double[3][3], int, cudaStream_t);                                  \
  template NppStatus warpQuad<T, 1, false>(const T *, NppiSize, int, NppiRect, const double[4][2], T *, int, NppiRect,   \
                                           const double[4][2], int, cudaStream_t);                                     \
  template NppStatus warpQuad<T, 3, false>(const T *, NppiSize, int, NppiRect, const double[4][2], T *, int, NppiRect,   \
                                           const double[4][2], int, cudaStream_t);                                     \
  template NppStatus warpQuad<T, 4, false>(const T *, NppiSize, int, NppiRect, const double[4][2], T *, int, NppiRect,   \
                                           const double[4][2], int, cudaStream_t);                                     \
  template NppStatus warpQuad<T, 4, true>(const T *, NppiSize, int, NppiRect, const double[4][2], T *, int, NppiRect,    \
                                          const double[4][2], int, cudaStream_t);                                      \
  template NppStatus warpForwardPlanar<T, 3>(const T *const[], NppiSize, int, NppiRect, T *const[], int, NppiRect,       \
                                             const double[3][3], int, cudaStream_t);                                   \
  template NppStatus warpForwardPlanar<T, 4>(const T *const[], NppiSize, int, NppiRect, T *const[], int, NppiRect,       \
                                             const double[3][3], int, cudaStream_t);                                   \
  template NppStatus warpBackwardPlanar<T, 3>(const T *const[], NppiSize, int, NppiRect, T *const[], int, NppiRect,      \
                                              const double[3][3], int, cudaStream_t);                                  \
  template NppStatus warpBackwardPlanar<T, 4>(const T *const[], NppiSize, int, NppiRect, T *const[], int, NppiRect,      \
                                              const double[3][3], int, cudaStream_t);                                  \
  template NppStatus warpQuadPlanar<T, 3>(const T *const[], NppiSize, int, NppiRect, const double[4][2], T *const[], int, \
                                          NppiRect, const double[4][2], int, cudaStream_t);                            \
  template NppStatus warpQuadPlanar<T, 4>(const T *const[], NppiSize, int, NppiRect, const double[4][2], T *const[], int, \
                                          NppiRect, const double[4][2], int, cudaStream_t);

NPPI_GEOM_INSTANTIATE(Npp8u)
NPPI_GEOM_INSTANTIATE(Npp16u)
NPPI_GEOM_INSTANTIATE(Npp32s)
NPPI_GEOM_INSTANTIATE(Npp32f)
NPPI_GEOM_INSTANTIATE(Npp16f)
#undef NPPI_GEOM_INSTANTIATE

#define NPPI_GEOM_INSTANTIATE_BATCH(T)                                                                                \
  template NppStatus warpBatch<T, 1, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpPerspectiveBatchCXR *,          \
                                            unsigned int, cudaStream_t);                                              \
  template NppStatus warpBatch<T, 3, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpPerspectiveBatchCXR *,          \
                                            unsigned int, cudaStream_t);                                              \
  template NppStatus warpBatch<T, 4, false>(NppiSize, NppiRect, NppiRect, int, NppiWarpPerspectiveBatchCXR *,          \
                                            unsigned int, cudaStream_t);                                              \
  template NppStatus warpBatch<T, 4, true>(NppiSize, NppiRect, NppiRect, int, NppiWarpPerspectiveBatchCXR *,           \
                                           unsigned int, cudaStream_t);

NPPI_GEOM_INSTANTIATE_BATCH(Npp8u)
NPPI_GEOM_INSTANTIATE_BATCH(Npp16f)
NPPI_GEOM_INSTANTIATE_BATCH(Npp32f)
#undef NPPI_GEOM_INSTANTIATE_BATCH

} // namespace warp
} // namespace nppi
