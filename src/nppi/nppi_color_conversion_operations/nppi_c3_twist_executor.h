// c3 executor: ColorTwist sampling-format conversions
//
// NVIDIA NPP applies the user supplied 3x4 matrix to every pixel and rounds
// the result to nearest (floor(x + 0.5)) with saturation:
//   dst[c] = aTwist[c][0]*src[0] + aTwist[c][1]*src[1] + aTwist[c][2]*src[2]
//            + aTwist[c][3]
// Chroma for the subsampled formats is the average of the twisted per-pixel
// values over the 2x2 block (4:2:0) or the pixel pair (4:2:2), rounded to
// nearest.  The 4:2:2 packed layout is [U Y0 V Y1].  Odd extents leave the
// odd edge unconverted and report NPP_DOUBLE_SIZE_WARNING.
#ifndef NPPI_C3_TWIST_EXECUTOR_H
#define NPPI_C3_TWIST_EXECUTOR_H

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace c3twist {

// --- forward: packed RGB (C3) -> planar YUV 4:2:0 ------------------------
NppStatus fwd420(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                 cudaStream_t stream, const Npp32f aTwist[3][4]);
// --- forward: planar RGB (P3) -> planar YUV 4:2:0 ------------------------
NppStatus fwd420P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);

// --- forward: 16-bit variants -------------------------------------------
NppStatus fwd420_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                     cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus fwd420P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);

// --- forward 4:2:2 -------------------------------------------------------
// packed destination in [U Y0 V Y1] order.
NppStatus fwd422C2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                   cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus fwd422C2_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                       cudaStream_t stream, const Npp32f aTwist[3][4]);
// packed source -> planar 4:2:2 destination.
NppStatus fwd422C3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                     cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus fwd422C3P3_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                         cudaStream_t stream, const Npp32f aTwist[3][4]);
// planar source -> planar 4:2:2 destination.
NppStatus fwd422P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus fwd422P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);

// --- inverse: planar YUV 4:2:0 -> RGB ------------------------------------
// dstPp 3 or 4; pp 4 writes `alpha` into the last channel.
NppStatus inv420(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, int dstPp, Npp8u alpha,
                 NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv420P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv420_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst, int nDstStep, int dstPp,
                     Npp16u alpha, NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv420P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);

// --- inverse: planar / packed YUV 4:2:2 -> RGB ---------------------------
NppStatus inv422P3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                   NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv422P3C(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, int dstPp,
                    Npp8u alpha, NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv422( const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream,
                  const Npp32f aTwist[3][4]);
NppStatus inv422_16u(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                     cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv422P3_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst[3], const int aDstStep[3],
                       NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);
NppStatus inv422P3C_16u(const Npp16u *const pSrc[3], const int aSrcStep[3], Npp16u *pDst, int nDstStep, int dstPp,
                        Npp16u alpha, NppiSize oSizeROI, cudaStream_t stream, const Npp32f aTwist[3][4]);

} // namespace c3twist
} // namespace nppi

#endif // NPPI_C3_TWIST_EXECUTOR_H
