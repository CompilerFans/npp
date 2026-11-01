// nppiWarpPerspective / Back / Quad / Batch family executor.
//
// Host-visible surface (safe for plain .cpp translation units): Mat3, the
// per-entry validation helpers and the entry-point templates.  The templates
// are defined and explicitly instantiated in nppi_warp_perspective_executor.cu;
// the public wrappers are generated from mpp_scripts/families/
// nppi_warp_perspective*.json by mpp_scripts/gen_family.py.
//
// Semantics (baseline NVIDIA NPP 13.4 on the RTX 5090, validated against the
// goldens of test/unit/diff/test_nppi_warp_perspective_{planar,quad,batch}_generated.cpp,
// see also docs/agent_packages/g5_notes.md):
//
//   * Direction.  nppiWarpPerspective/Quad/Batch take coefficients that map
//     source to destination; the kernels apply the inverse (dst -> src).
//     nppiWarpPerspectiveBack takes coefficients that already map
//     destination to source and applies them directly.
//   * Write region (forward/batch, measured): NPP does not simply test every
//     destination pixel.  Per destination row it gates the row on the source
//     v of the ROI's left edge (v < sy or v > sy + sh drops the row), clips the
//     x range to [ceil(xl), floor(xr)] where xl/xr are the forward-map images
//     of the source-ROI left/right edges evaluated at that row's v, and only
//     then applies the per-pixel test den > 0 and u/v inside the *source ROI
//     rect in absolute coordinates* (continuous, inclusive of x + w - 1).
//     The backward family only performs the per-pixel test (it writes rows the
//     forward kernel would drop - measured on the planar back goldens).
//   * Quad: the two quads define a src->dst homography solved from the eight
//     vertex equations; the kernel additionally restricts writes to the image
//     of the source quadrilateral (half-plane test in float32, matching the
//     reduced-precision edge arithmetic NVIDIA uses).
//   * Errors (measured, see the contract goldens): forward validates the
//     coefficient determinant (NPP_COEFFICIENT_ERROR on a singular matrix),
//     the backward entry point does not.  Steps: the destination step must be
//     positive (NPP_STEP_ERROR); the planar source step is not validated at all
//     (src_step0 -> NPP_NO_ERROR).  A negative source ROI origin is
//     NPP_SIZE_ERROR, an empty/negative source ROI rectangle is
//     NPP_RECTANGLE_ERROR (planar) or NPP_WRONG_INTERSECTION_QUAD_WARNING
//     (quad/rotate); an empty destination ROI rectangle is a no-op
//     (NPP_NO_ERROR) for the perspective family.  Quad geometries are validated
//     before anything else (NPP_QUADRANGLE_ERROR).  Batch: nBatchSize < 2 is
//     NPP_BAD_ARGUMENT_ERROR, an empty source/destination rect or a negative
//     ROI origin is NPP_RECTANGLE_ERROR/NPP_SIZE_ERROR, a null list is
//     NPP_NULL_POINTER_ERROR, and BatchInit with nBatchSize == 0 returns
//     NPP_CUDA_KERNEL_EXECUTION_ERROR (NVIDIA launches an empty grid).
#ifndef NPPI_WARP_PERSPECTIVE_EXECUTOR_H
#define NPPI_WARP_PERSPECTIVE_EXECUTOR_H

#include "npp.h"
#include "nppi_geometry_transforms.h"
#include <cuda_runtime.h>

namespace nppi {
namespace warp {

// 3x3 projective matrix in double precision (row major).
struct Mat3 {
  double m[3][3];
};

inline Mat3 mat3FromCoeffs(const double aCoeffs[3][3]) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = aCoeffs[i][j];
  return r;
}

inline double det3(const Mat3 &a) {
  return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
         a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
         a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}

inline Mat3 inverse3(const Mat3 &a) {
  Mat3 r;
  const double id = 1.0 / det3(a);
  r.m[0][0] = (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) * id;
  r.m[0][1] = (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) * id;
  r.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) * id;
  r.m[1][0] = (a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2]) * id;
  r.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) * id;
  r.m[1][2] = (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) * id;
  r.m[2][0] = (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]) * id;
  r.m[2][1] = (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) * id;
  r.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) * id;
  return r;
}

// Solves the 8x8 system of the quad-to-quad homography.  Vertices are ordered
// top-left, top-right, bottom-right, bottom-left; the resulting matrix maps
// source image coordinates to destination image coordinates.
bool solveQuadHomography(const double aSrcQuad[4][2], const double aDstQuad[4][2], Mat3 &out);

// Degenerate-quad detection used by the quad entry points.
bool isDegenerateQuad(const double aQuad[4][2]);

// ---------------------------------------------------------------------------
// Entry points (templates defined in the .cu, explicit instantiations only).
// ---------------------------------------------------------------------------

// Packed (interleaved) destination with `Channels` channels; AlphaPreserve
// leaves channel 3 of the destination untouched (AC4 layouts).
template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpForward(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                      NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, cudaStream_t stream);

template <typename T, int Planes>
NppStatus warpForwardPlanar(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *const pDst[],
                            int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation,
                            cudaStream_t stream);

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpBackward(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst, int nDstStep,
                       NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, cudaStream_t stream);

template <typename T, int Planes>
NppStatus warpBackwardPlanar(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *const pDst[],
                             int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation,
                             cudaStream_t stream);

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpQuad(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2],
                   T *pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
                   cudaStream_t stream);

template <typename T, int Planes>
NppStatus warpQuadPlanar(const T *const pSrc[], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                         const double aSrcQuad[4][2], T *const pDst[], int nDstStep, NppiRect oDstROI,
                         const double aDstQuad[4][2], int eInterpolation, cudaStream_t stream);

template <typename T, int Channels, bool AlphaPreserve>
NppStatus warpBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, int eInterpolation,
                    NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream);

// nppiWarpPerspectiveBatchInit: writes the true inverse of each entry's
// pCoeffs into aTransformedCoeffs.
NppStatus batchInit(NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize, cudaStream_t stream);

} // namespace warp
} // namespace nppi

#endif // NPPI_WARP_PERSPECTIVE_EXECUTOR_H
