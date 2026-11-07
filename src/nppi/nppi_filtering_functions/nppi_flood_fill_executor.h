// Flood fill executor: nppiFloodFill / FloodFillBoundary / FloodFillRange /
// FloodFillGradient and their *Boundary forms (72 functions).
//
// Measured against NVIDIA NPP 13.4 in the reference implementation:
//
//  * The filled set ("region") is the connected component of the seed pixel:
//      - FloodFill / FloodFillBoundary: pixels whose value equals the seed
//        pixel value (all channels for C3),
//      - FloodFillRange: pixels inside [nMin, nMax] per channel,
//      - FloodFillGradient: pixels inside [seed - nMin, seed + nMax] per
//        channel, with the lower bound clamped at 0 (unsigned source types).
//    Connectivity is 4-way for nppiNormL1 and 8-way for nppiNormInf.  The seed
//    pixel is always part of the region, even when it does not satisfy the
//    range/gradient predicate.
//  * The *Boundary forms do NOT use nBoundaryValue as a stop value.  They fill
//    the same region as the matching non-Boundary form, but write
//    nBoundaryValue into the region pixels that have at least one of their
//    eight neighbours outside the region (neighbours outside the ROI do not
//    count), and nNewValue into the remaining region pixels.  Measured on
//    1-pixel lines, 3x3 blocks, ring-shaped regions and regions touching the
//    ROI edge.
//  * The seed pixel value is read at the byte offset
//    `oSeed.y * nSrcDstStep + oSeed.x * C * C * sizeof(T)` - the column term is
//    scaled by the channel count twice (measured with the index-encoding measurement
//    measurements).  For C1 that is the seed pixel itself, for C3 it is another
//    column (3*x) or, near the end of the image, memory past the image; NPP
//    then compares the neighbours against that value.  The executor
//    reproduces the quirk.
//  * Validation order: null pSrcDst, null pBuffer, non-positive
//    ROI dimensions (NPP_SIZE_ERROR), seed outside the ROI
//    (NPP_OUT_OFF_RANGE_ERROR), unsupported NppiNorm (NPP_BAD_ARGUMENT_ERROR,
//    nppiNormL2 is rejected), and nMin > nMax for the Range forms
//    (NPP_RANGE_ERROR; the Gradient forms do not check it).  nSrcDstStep is
//    not validated at all.
//  * nppiFloodFillGetBufferSize is exactly 4*width*height + 4*width + 64 bytes
//    for every channel count, independent of the image type.
#ifndef NPPI_FLOOD_FILL_EXECUTOR_H
#define NPPI_FLOOD_FILL_EXECUTOR_H

#include "npp.h"

namespace nppi {
namespace floodfill {

enum Variant {
  kPlain = 0,
  kBoundary = 1,
  kRange = 2,
  kGradient = 3,
  kRangeBoundary = 4,
  kGradientBoundary = 5
};

NppStatus getBufferSize(NppiSize oSizeROI, int *hpBufferSize);

// C1 entry points take a pointer to a single element (the generated wrappers
// pass the address of the by-value parameter); C3 entry points take a pointer
// to the three channel values.
template <typename T, int C>
NppStatus floodFill(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pNewValue,
                    NppiNorm eNorm, NppiSize oSizeROI, NppiConnectedRegion *pConnectedRegion,
                    Npp8u *pBuffer, cudaStream_t stream);

template <typename T, int C>
NppStatus floodFillBoundary(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pNewValue,
                            const T *pBoundaryValue, NppiNorm eNorm, NppiSize oSizeROI,
                            NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                            cudaStream_t stream);

template <typename T, int C>
NppStatus floodFillRange(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                         const T *pMax, const T *pNewValue, NppiNorm eNorm, NppiSize oSizeROI,
                         NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                         cudaStream_t stream);

template <typename T, int C>
NppStatus floodFillGradient(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                            const T *pMax, const T *pNewValue, NppiNorm eNorm, NppiSize oSizeROI,
                            NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                            cudaStream_t stream);

template <typename T, int C>
NppStatus floodFillRangeBoundary(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                                 const T *pMax, const T *pNewValue, const T *pBoundaryValue,
                                 NppiNorm eNorm, NppiSize oSizeROI,
                                 NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                                 cudaStream_t stream);

template <typename T, int C>
NppStatus floodFillGradientBoundary(T *pSrcDst, int nSrcDstStep, NppiPoint oSeed, const T *pMin,
                                    const T *pMax, const T *pNewValue, const T *pBoundaryValue,
                                    NppiNorm eNorm, NppiSize oSizeROI,
                                    NppiConnectedRegion *pConnectedRegion, Npp8u *pBuffer,
                                    cudaStream_t stream);

}  // namespace floodfill
}  // namespace nppi

#endif  // NPPI_FLOOD_FILL_EXECUTOR_H
