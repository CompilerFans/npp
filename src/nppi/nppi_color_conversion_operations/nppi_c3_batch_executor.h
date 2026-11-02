// c3 package executor: batch color conversions (nppi_color_conversion.h).
//
// The *_Batch entry points take arrays of NppiImageDescriptor in device
// memory plus one ROI (or, for the _Advanced variants, one ROI per
// destination descriptor) and run the corresponding single-image conversion
// on every pair.
//
// Semantics measured on NVIDIA NPP 13.4 (RTX 5090):
//   * nBatchSize <= 1 -> NPP_BAD_ARGUMENT_ERROR (checked before anything is
//     read; the reference dereferences the list pointers without a null
//     check, so a null list is an unrecoverable fault there and is instead
//     reported as NPP_NULL_POINTER_ERROR here).
//   * The plain variant applies oSizeROI to every pair.  The _Advanced
//     variant uses pDstBatchList[i].oSize as that pair's ROI; oMaxSizeROI is
//     not validated by the reference and is ignored here as well.
//   * An odd ROI is processed up to the largest even rectangle and the call
//     returns NPP_DOUBLE_SIZE_WARNING (4:2:2 shrinks the width, 4:2:0 the
//     width and height).  Recorded goldens pin the converted extent.
//   * Per-image data pointers and steps are not validated by the reference
//     (invalid ones fault the kernel); this executor reports
//     NPP_NULL_POINTER_ERROR / NPP_STEP_ERROR instead of touching memory.
#ifndef NPPI_C3_BATCH_EXECUTOR_H
#define NPPI_C3_BATCH_EXECUTOR_H

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace c3batch {

enum Conversion {
  kYCbCr420ToBGR,
  kYCbCr420ToRGB,
  kYUV420ToBGR,
  kYUV420ToRGB,
  kYCbCr422ToBGR,
  kYCbCr422ToRGB,
  kYUV422ToBGR,
  kYUV422ToRGB,
  kYCbCrToBGRP3,
  kYCbCrToRGBP3,
  kYUVToBGRP3,
  kYUVToRGBP3,
  kYCbCrToBGRC3,
  kYCbCrToRGBC3,
  kYUVToBGRC3,
  kYUVToRGBC3,
};

// Planar source (three per-image batch lists) -> packed destination.
NppStatus runPlanar(const NppiImageDescriptor *const pSrc[3], NppiImageDescriptor *pDst, int nBatchSize,
                    NppiSize oSizeROI, cudaStream_t stream, Conversion conv, int advanced);
// Packed source list -> packed destination.
NppStatus runPacked(const NppiImageDescriptor *pSrc, NppiImageDescriptor *pDst, int nBatchSize, NppiSize oSizeROI,
                    cudaStream_t stream, Conversion conv, int advanced);

} // namespace c3batch
} // namespace nppi

#endif // NPPI_C3_BATCH_EXECUTOR_H
