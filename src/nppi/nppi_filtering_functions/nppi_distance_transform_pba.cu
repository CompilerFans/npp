// Exact unsigned PBA (distance transform + Voronoi) executor.
//
// Semantics measured on NVIDIA NPP 13.4 / RTX 5090 (probes pba1..pba4):
//   * sites are pixels with nMinSiteValue <= value <= nMaxSiteValue;
//   * every pixel reports the nearest site by squared Euclidean distance;
//     ties prefer the smaller x, then the larger y;
//   * with no sites the reported site is (-1,-1) for integer destinations
//     and (-1,0) for floating-point destinations (measured quirk);
//   * pDstVoronoi and pDstVoronoiIndices both hold (site.x, site.y);
//     the relative Manhattan output holds (x - site.x, y - site.y) and the
//     absolute one holds (|x - site.x|, |y - site.y|);
//   * the transform is sqrt(distance): 16u truncates a float sqrt, 32f is
//     float sqrt, 64f is double sqrt;
//   * the ROI must be at least 64x64; the source step is not validated;
//     every non-null output's step must be at least one row and aligned;
//     at least one output must be requested.
//
// A brute-force nearest-site scan is used for exactness; it is O(W*H*sites)
// and intended for correctness, not throughput.
#include "npp.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

namespace {

template <typename Src, typename Dst, bool Absolute>
__global__ void pbaBruteKernel(const Src *pSrc, int nSrcStep, Src minSiteValue, Src maxSiteValue,
                               Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices,
                               int nDstVoronoiIndicesStep, void *pManhattan, int nManhattanStep,
                               Dst *pDstTransform, int nDstTransformStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  int bestX = -1;
  int bestY = -1;
  long long bestDistance = 0;
  bool found = false;
  for (int sy = 0; sy < height; ++sy) {
    const Src *row = reinterpret_cast<const Src *>(reinterpret_cast<const char *>(pSrc) +
                                                   static_cast<long long>(sy) * nSrcStep);
    for (int sx = 0; sx < width; ++sx) {
      const Src value = row[sx];
      if (!(value >= minSiteValue && value <= maxSiteValue)) {
        continue;
      }
      const long long dx = static_cast<long long>(x) - sx;
      const long long dy = static_cast<long long>(y) - sy;
      const long long distance = dx * dx + dy * dy;
      if (!found || distance < bestDistance ||
          (distance == bestDistance && (sx < bestX || (sx == bestX && sy > bestY)))) {
        found = true;
        bestDistance = distance;
        bestX = sx;
        bestY = sy;
      }
    }
  }
  if (!found) {
    // Measured NPP 13.4 quirk: with no site at all the reported site is
    // (-1,-1) for integer destinations but (-1,0) for floating-point ones.
    const long long dx = static_cast<long long>(x) + 1;
    const long long dy = std::is_same_v<Dst, Npp64f> || std::is_same_v<Dst, Npp32f>
                             ? static_cast<long long>(y)
                             : static_cast<long long>(y) + 1;
    bestDistance = dx * dx + dy * dy;
  }

  if (pDstVoronoi != nullptr) {
    Npp16s *row = reinterpret_cast<Npp16s *>(reinterpret_cast<char *>(pDstVoronoi) +
                                             static_cast<long long>(y) * nDstVoronoiStep);
    row[2 * x] = static_cast<Npp16s>(bestX);
    row[2 * x + 1] = static_cast<Npp16s>(bestY);
  }
  if (pDstVoronoiIndices != nullptr) {
    Npp16s *row = reinterpret_cast<Npp16s *>(reinterpret_cast<char *>(pDstVoronoiIndices) +
                                             static_cast<long long>(y) * nDstVoronoiIndicesStep);
    row[2 * x] = static_cast<Npp16s>(bestX);
    row[2 * x + 1] = static_cast<Npp16s>(bestY);
  }
  if (pManhattan != nullptr) {
    const long long dx = static_cast<long long>(x) - bestX;
    const long long dy = static_cast<long long>(y) - bestY;
    if (Absolute) {
      Npp16u *row = reinterpret_cast<Npp16u *>(reinterpret_cast<char *>(pManhattan) +
                                               static_cast<long long>(y) * nManhattanStep);
      row[2 * x] = static_cast<Npp16u>(dx < 0 ? -dx : dx);
      row[2 * x + 1] = static_cast<Npp16u>(dy < 0 ? -dy : dy);
    } else {
      Npp16s *row = reinterpret_cast<Npp16s *>(reinterpret_cast<char *>(pManhattan) +
                                               static_cast<long long>(y) * nManhattanStep);
      row[2 * x] = static_cast<Npp16s>(dx);
      row[2 * x + 1] = static_cast<Npp16s>(dy);
    }
  }
  if (pDstTransform != nullptr) {
    Dst *row = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDstTransform) +
                                       static_cast<long long>(y) * nDstTransformStep);
    if constexpr (std::is_same_v<Dst, Npp64f>) {
      row[x] = static_cast<Npp64f>(std::sqrt(static_cast<double>(bestDistance)));
    } else {
      row[x] = static_cast<Dst>(std::sqrt(static_cast<float>(bestDistance)));
    }
  }
}

inline NppStatus checkPlaneStep(const void *pointer, int step, int width) {
  if (pointer == nullptr) {
    return NPP_NO_ERROR;
  }
  const int minimum = width * 2 * static_cast<int>(sizeof(Npp16s));
  if (step < minimum) {
    return NPP_STEP_ERROR;
  }
  if (step % static_cast<int>(sizeof(Npp16s)) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <typename Dst>
inline NppStatus checkTransformStep(const void *pointer, int step, int width) {
  if (pointer == nullptr) {
    return NPP_NO_ERROR;
  }
  const int minimum = width * static_cast<int>(sizeof(Dst));
  if (step < minimum) {
    return NPP_STEP_ERROR;
  }
  if (step % static_cast<int>(sizeof(Dst)) != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <typename Src, typename Dst, bool Absolute>
NppStatus pbaLaunch(const Src *pSrc, int nSrcStep, Src nMinSiteValue, Src nMaxSiteValue,
                    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices,
                    int nDstVoronoiIndicesStep, void *pManhattan, int nManhattanStep,
                    Dst *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
                    Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (pSrc == nullptr || pDeviceBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 64 || oSizeROI.height < 64) {
    return NPP_SIZE_ERROR;
  }
  if (pDstVoronoi == nullptr && pDstVoronoiIndices == nullptr && pManhattan == nullptr &&
      pDstTransform == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  NppStatus status = checkPlaneStep(pDstVoronoi, nDstVoronoiStep, oSizeROI.width);
  if (status != NPP_NO_ERROR) return status;
  status = checkPlaneStep(pDstVoronoiIndices, nDstVoronoiIndicesStep, oSizeROI.width);
  if (status != NPP_NO_ERROR) return status;
  status = checkPlaneStep(pManhattan, nManhattanStep, oSizeROI.width);
  if (status != NPP_NO_ERROR) return status;
  status = checkTransformStep<Dst>(pDstTransform, nDstTransformStep, oSizeROI.width);
  if (status != NPP_NO_ERROR) return status;

  const dim3 block(16, 16);
  const dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  pbaBruteKernel<Src, Dst, Absolute><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
      nDstVoronoiIndicesStep, pManhattan, nManhattanStep, pDstTransform, nDstTransformStep,
      oSizeROI.width, oSizeROI.height);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

// ---------------------------------------------------------------------------
// Generated wrappers: PBA (relative Manhattan) and AbsPBA (absolute Manhattan)
// for the 16u, 32f and 64f transform variants.  The 64f-source entry points
// are named without the doubled dtype suffix and are defined explicitly below.
// ---------------------------------------------------------------------------

#define DEFINE_PBA_16U(SRC, TOK)                                                                          \
  NppStatus nppiDistanceTransformPBA_##TOK##16u_C1R_Ctx(                                                  \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      NppStreamContext nppStreamCtx) {                                                                    \
    return pbaLaunch<SRC, Npp16u, false>(                                                                 \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformPBA_##TOK##16u_C1R(                                                      \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer) {            \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformPBA_##TOK##16u_C1R_Ctx(                                                   \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, ctx);                                                                              \
  }

#define DEFINE_ABS_16U(SRC, TOK)                                                                          \
  NppStatus nppiDistanceTransformAbsPBA_##TOK##16u_C1R_Ctx(                                               \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      NppStreamContext nppStreamCtx) {                                                                    \
    return pbaLaunch<SRC, Npp16u, true>(                                                                  \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformAbsPBA_##TOK##16u_C1R(                                                   \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer) {            \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformAbsPBA_##TOK##16u_C1R_Ctx(                                                \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, ctx);                                                                              \
  }

#define DEFINE_PBA_32F(SRC, TOK)                                                                          \
  NppStatus nppiDistanceTransformPBA_##TOK##32f_C1R_Ctx(                                                  \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      NppStreamContext nppStreamCtx) {                                                                    \
    return pbaLaunch<SRC, Npp32f, false>(                                                                 \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformPBA_##TOK##32f_C1R(                                                      \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer) {            \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformPBA_##TOK##32f_C1R_Ctx(                                                   \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, ctx);                                                                              \
  }

#define DEFINE_ABS_32F(SRC, TOK)                                                                          \
  NppStatus nppiDistanceTransformAbsPBA_##TOK##32f_C1R_Ctx(                                               \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      NppStreamContext nppStreamCtx) {                                                                    \
    return pbaLaunch<SRC, Npp32f, true>(                                                                  \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformAbsPBA_##TOK##32f_C1R(                                                   \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer) {            \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformAbsPBA_##TOK##32f_C1R_Ctx(                                                \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, ctx);                                                                              \
  }

#define DEFINE_PBA_64F(SRC, TOK)                                                                          \
  NppStatus nppiDistanceTransformPBA_##TOK##64f_C1R_Ctx(                                                  \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {                                  \
    (void)pAntialiasingDeviceBuffer;                                                                      \
    return pbaLaunch<SRC, Npp64f, false>(                                                                 \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformPBA_##TOK##64f_C1R(                                                      \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer) {                                                                 \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformPBA_##TOK##64f_C1R_Ctx(                                                   \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, pAntialiasingDeviceBuffer, ctx);                                                   \
  }

#define DEFINE_ABS_64F(SRC, TOK)                                                                          \
  NppStatus nppiDistanceTransformAbsPBA_##TOK##64f_C1R_Ctx(                                               \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {                                  \
    (void)pAntialiasingDeviceBuffer;                                                                      \
    return pbaLaunch<SRC, Npp64f, true>(                                                                  \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformAbsPBA_##TOK##64f_C1R(                                                   \
      SRC *pSrc, int nSrcStep, SRC nMinSiteValue, SRC nMaxSiteValue, Npp16s *pDstVoronoi,                 \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer) {                                                                 \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformAbsPBA_##TOK##64f_C1R_Ctx(                                                \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, pAntialiasingDeviceBuffer, ctx);                                                   \
  }

DEFINE_PBA_16U(Npp8u, 8u)
DEFINE_ABS_16U(Npp8u, 8u)
DEFINE_PBA_16U(Npp8s, 8s)
DEFINE_ABS_16U(Npp8s, 8s)
DEFINE_PBA_16U(Npp16u, 16u)
DEFINE_ABS_16U(Npp16u, 16u)
DEFINE_PBA_16U(Npp16s, 16s)
DEFINE_ABS_16U(Npp16s, 16s)

DEFINE_PBA_32F(Npp8u, 8u)
DEFINE_ABS_32F(Npp8u, 8u)
DEFINE_PBA_32F(Npp8s, 8s)
DEFINE_ABS_32F(Npp8s, 8s)
DEFINE_PBA_32F(Npp16u, 16u)
DEFINE_ABS_32F(Npp16u, 16u)
DEFINE_PBA_32F(Npp16s, 16s)
DEFINE_ABS_32F(Npp16s, 16s)

DEFINE_PBA_64F(Npp8u, 8u)
DEFINE_ABS_64F(Npp8u, 8u)
DEFINE_PBA_64F(Npp8s, 8s)
DEFINE_ABS_64F(Npp8s, 8s)
DEFINE_PBA_64F(Npp16u, 16u)
DEFINE_ABS_64F(Npp16u, 16u)
DEFINE_PBA_64F(Npp16s, 16s)
DEFINE_ABS_64F(Npp16s, 16s)
DEFINE_PBA_64F(Npp32f, 32f)
DEFINE_ABS_64F(Npp32f, 32f)

// 64f source: the public names are nppi*_64f_C1R_Ctx (no doubled suffix).
#define DEFINE_PBA_64F_SRC()                                                                              \
  NppStatus nppiDistanceTransformPBA_64f_C1R_Ctx(                                                         \
      Npp64f *pSrc, int nSrcStep, Npp64f nMinSiteValue, Npp64f nMaxSiteValue, Npp16s *pDstVoronoi,        \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {                                  \
    (void)pAntialiasingDeviceBuffer;                                                                      \
    return pbaLaunch<Npp64f, Npp64f, false>(                                                              \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformPBA_64f_C1R(                                                             \
      Npp64f *pSrc, int nSrcStep, Npp64f nMinSiteValue, Npp64f nMaxSiteValue, Npp16s *pDstVoronoi,        \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer) {                                                                 \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformPBA_64f_C1R_Ctx(                                                          \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances,                                    \
        nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, pAntialiasingDeviceBuffer, ctx);                                                   \
  }

#define DEFINE_ABS_64F_SRC()                                                                              \
  NppStatus nppiDistanceTransformAbsPBA_64f_C1R_Ctx(                                                      \
      Npp64f *pSrc, int nSrcStep, Npp64f nMinSiteValue, Npp64f nMaxSiteValue, Npp16s *pDstVoronoi,        \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {                                  \
    (void)pAntialiasingDeviceBuffer;                                                                      \
    return pbaLaunch<Npp64f, Npp64f, true>(                                                               \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, nppStreamCtx.hStream);                                                             \
  }                                                                                                       \
  NppStatus nppiDistanceTransformAbsPBA_64f_C1R(                                                          \
      Npp64f *pSrc, int nSrcStep, Npp64f nMinSiteValue, Npp64f nMaxSiteValue, Npp16s *pDstVoronoi,        \
      int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,                        \
      Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,       \
      Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,              \
      Npp8u *pAntialiasingDeviceBuffer) {                                                                 \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiDistanceTransformAbsPBA_64f_C1R_Ctx(                                                       \
        pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,   \
        nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,                                    \
        nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI,            \
        pDeviceBuffer, pAntialiasingDeviceBuffer, ctx);                                                   \
  }

DEFINE_PBA_64F_SRC()
DEFINE_ABS_64F_SRC()

// ---------------------------------------------------------------------------
// The public wrappers above are emitted by token-pasting macros, which the
// API catalog cannot see.  These literal forwarding *_impl definitions
// follow the project convention so the catalog marks both the Ctx and
// non-Ctx public entry points real.
// ---------------------------------------------------------------------------

NppStatus nppiDistanceTransformPBA_8u16u_C1R_Ctx_impl(Npp8u * pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_8u16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx_impl(Npp8u * pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_8s16u_C1R_Ctx_impl(Npp8s * pSrc, int nSrcStep, Npp8s nMinSiteValue, Npp8s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_8s16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_8s16u_C1R_Ctx_impl(Npp8s * pSrc, int nSrcStep, Npp8s nMinSiteValue, Npp8s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_8s16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_16u16u_C1R_Ctx_impl(Npp16u * pSrc, int nSrcStep, Npp16u nMinSiteValue, Npp16u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_16u16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_16u16u_C1R_Ctx_impl(Npp16u * pSrc, int nSrcStep, Npp16u nMinSiteValue, Npp16u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_16u16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_16s16u_C1R_Ctx_impl(Npp16s * pSrc, int nSrcStep, Npp16s nMinSiteValue, Npp16s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_16s16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_16s16u_C1R_Ctx_impl(Npp16s * pSrc, int nSrcStep, Npp16s nMinSiteValue, Npp16s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp16u * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_16s16u_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_8u32f_C1R_Ctx_impl(Npp8u * pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_8u32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx_impl(Npp8u * pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_8s32f_C1R_Ctx_impl(Npp8s * pSrc, int nSrcStep, Npp8s nMinSiteValue, Npp8s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_8s32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_8s32f_C1R_Ctx_impl(Npp8s * pSrc, int nSrcStep, Npp8s nMinSiteValue, Npp8s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_8s32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_16u32f_C1R_Ctx_impl(Npp16u * pSrc, int nSrcStep, Npp16u nMinSiteValue, Npp16u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_16u32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_16u32f_C1R_Ctx_impl(Npp16u * pSrc, int nSrcStep, Npp16u nMinSiteValue, Npp16u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_16u32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_16s32f_C1R_Ctx_impl(Npp16s * pSrc, int nSrcStep, Npp16s nMinSiteValue, Npp16s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_16s32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_16s32f_C1R_Ctx_impl(Npp16s * pSrc, int nSrcStep, Npp16s nMinSiteValue, Npp16s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp32f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_16s32f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_8u64f_C1R_Ctx_impl(Npp8u * pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_8u64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx_impl(Npp8u * pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_8s64f_C1R_Ctx_impl(Npp8s * pSrc, int nSrcStep, Npp8s nMinSiteValue, Npp8s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_8s64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_8s64f_C1R_Ctx_impl(Npp8s * pSrc, int nSrcStep, Npp8s nMinSiteValue, Npp8s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_8s64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_16u64f_C1R_Ctx_impl(Npp16u * pSrc, int nSrcStep, Npp16u nMinSiteValue, Npp16u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_16u64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_16u64f_C1R_Ctx_impl(Npp16u * pSrc, int nSrcStep, Npp16u nMinSiteValue, Npp16u nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_16u64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_16s64f_C1R_Ctx_impl(Npp16s * pSrc, int nSrcStep, Npp16s nMinSiteValue, Npp16s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_16s64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_16s64f_C1R_Ctx_impl(Npp16s * pSrc, int nSrcStep, Npp16s nMinSiteValue, Npp16s nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_16s64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_32f64f_C1R_Ctx_impl(Npp32f * pSrc, int nSrcStep, Npp32f nMinSiteValue, Npp32f nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_32f64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx_impl(Npp32f * pSrc, int nSrcStep, Npp32f nMinSiteValue, Npp32f nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformPBA_64f_C1R_Ctx_impl(Npp64f * pSrc, int nSrcStep, Npp64f nMinSiteValue, Npp64f nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s * pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformPBA_64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiRelativeManhattanDistances, nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

NppStatus nppiDistanceTransformAbsPBA_64f_C1R_Ctx_impl(Npp64f * pSrc, int nSrcStep, Npp64f nMinSiteValue, Npp64f nMaxSiteValue, Npp16s * pDstVoronoi, int nDstVoronoiStep, Npp16s * pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u * pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep, Npp64f * pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return nppiDistanceTransformAbsPBA_64f_C1R_Ctx(pSrc, nSrcStep, nMinSiteValue, nMaxSiteValue, pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances, nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, pAntialiasingDeviceBuffer, nppStreamCtx);
}

