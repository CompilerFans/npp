#pragma once

// NPPI point / interpolated / palette LUT executors.
//
// Semantics are pinned by reference comparisons
// (nppiLUT, nppiLUT_Linear, nppiLUT_Cubic, nppiLUTPalette,
// nppiLUTPaletteSwap and nppiLUT_Trilinear) recorded against NVIDIA NPP 13.4
// in the reference implementation.  The important, non-obvious points:
//
//   * point / linear / cubic LUT: pixels outside [levels[0], levels[n-1])
//     are copied through unchanged; the interval search is a forward scan
//     (unsorted level arrays therefore behave exactly like NPP);
//   * the 8-bit families take a direct-index shortcut when *every* used
//     channel has exactly 256 levels: dst = values[src] (levels ignored);
//   * linear interpolation uses C integer truncation for integer dtypes;
//     cubic interpolation is the 4-point Newton/divided-difference cubic
//     evaluated in float (see the SASS-derived formula in the.cu file);
//   * row steps are never validated (NPP accepts short, zero and negative
//     steps), and a width-0 / height>0 ROI returns NPP_SUCCESS for the
//     point/interp families while height-0 launches an invalid grid and
//     reports NPP_CUDA_KERNEL_EXECUTION_ERROR;
//   * AC4 variants leave the alpha channel completely untouched.

#include "npp.h"

#include <cuda_runtime.h>

namespace nppi {
namespace lut {

enum class LutMode { Point, Linear, Cubic };

// 8u/16u/16s LUT tables are Npp32s; the 32f family uses Npp32f tables.
template <typename SrcT> struct LutValue {
  using type = Npp32s;
};
template <> struct LutValue<Npp32f> {
  using type = Npp32f;
};

// ---------------------------------------------------------------------------
// Point / linear / cubic LUT
// ---------------------------------------------------------------------------

// Single-channel (C1 / C1IR) variants.
template <typename SrcT, LutMode Mode>
NppStatus runC1(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
                const typename LutValue<SrcT>::type *pValues, const typename LutValue<SrcT>::type *pLevels,
                int nLevels, cudaStream_t stream);

// Multi-channel variants.  Channels is the number of channels addressed in
// the pixel (3 or 4); AlphaOnly marks the AC4 layouts that carry only three
// tables and must not touch the destination alpha byte.
template <typename SrcT, int Channels, bool AlphaOnly, LutMode Mode>
NppStatus run(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
              const typename LutValue<SrcT>::type *const *pValues,
              const typename LutValue<SrcT>::type *const *pLevels, const int *nLevels, cudaStream_t stream);

template <typename SrcT, int Channels, bool AlphaOnly, LutMode Mode>
inline NppStatus runInPlace(SrcT *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                            const typename LutValue<SrcT>::type *const *pValues,
                            const typename LutValue<SrcT>::type *const *pLevels, const int *nLevels,
                            cudaStream_t stream) {
  return run<SrcT, Channels, AlphaOnly, Mode>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels,
                                              nLevels, stream);
}

// ---------------------------------------------------------------------------
// Palette LUT
// ---------------------------------------------------------------------------

// One table, dst element = table[src & mask] (8u_C1R, 8u32u, 16u8u, 16u_C1R,
// 16u32u).
template <typename SrcT, typename TabT, typename DstT>
NppStatus paletteC1(const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep, NppiSize oSizeROI, const TabT *pTable,
                    int nBitSize, cudaStream_t stream);

// 8u24u / 16u24u: packed 3-byte table entries, 3 destination bytes per pixel.
template <typename SrcT>
NppStatus palette24(const SrcT *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp8u *pTable, int nBitSize, cudaStream_t stream);

// C3 / C4 / AC4 with one table per addressed channel and dst type == src type.
template <typename SrcT, int Channels, bool AlphaOnly>
NppStatus paletteMulti(const SrcT *pSrc, int nSrcStep, SrcT *pDst, int nDstStep, NppiSize oSizeROI,
                       const SrcT *const *pTables, int nBitSize, cudaStream_t stream);

// C3 source -> C4 destination: dst = (table2[idx2], table1[idx1], table0[idx0]).
template <typename T>
NppStatus paletteSwap(const T *pSrc, int nSrcStep, int nAlphaValue, T *pDst, int nDstStep, NppiSize oSizeROI,
                      const T *const *pTables, int nBitSize, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Trilinear LUT (8u C4 / AC4 / AC4IR)
// ---------------------------------------------------------------------------

// pLevels is a host array of three host pointers (NPP reads the level values
// on the host); pValues is device memory laid out x-fastest, then y, then z.
NppStatus trilinear(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp32u *pValues, const Npp8u *const pLevels[3], const int aLevels[3], bool alphaSrc,
                    cudaStream_t stream);

} // namespace lut
} // namespace nppi
