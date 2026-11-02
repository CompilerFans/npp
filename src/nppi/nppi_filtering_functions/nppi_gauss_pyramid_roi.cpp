// Host-only destination-ROI helpers for the Gauss pyramid border family.
//
// Reverse-engineered from the NPP 13.2 libnppif host code (disassembly).
// All arithmetic is single precision:
//   Down:    clamp(ceilf(w / rate), 1, w - 1)
//   Up max:  max(w + 1, truncf(w * rate))
//   Up min:  w == 1 ? 1 : max(w + 1, floorf((w - 1) * rate))
// Both helpers write (-1, -1) into every output before validating.  A rate
// outside (1.0, 10.0] is NPP_BAD_ARGUMENT_ERROR; a non-positive source ROI
// extent is NPP_SIZE_ERROR; a null output pointer is NPP_NULL_POINTER_ERROR.
#include "npp.h"

#include <algorithm>
#include <cmath>

namespace {

bool validRate(Npp32f rate) { return rate > 1.0f && rate <= 10.0f; }

int clampDown(float quotient, int sourceExtent) {
  // NPP computes ceilf(extent / rate) in float, then clamps to
  // [1, extent - 1] (the extent - 1 upper bound is what makes a 1-pixel
  // source ROI stay 1 and keeps small ROIs below the naive ceil).
  int size = static_cast<int>(std::ceil(quotient));
  size = std::min(size, sourceExtent - 1);
  return std::max(size, 1);
}

}  // namespace

NppStatus nppiGetFilterGaussPyramidLayerDownBorderDstROI(int nSrcROIWidth, int nSrcROIHeight,
                                                         NppiSize *pDstSizeROI, Npp32f nRate) {
  if (pDstSizeROI == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  pDstSizeROI->width = -1;
  pDstSizeROI->height = -1;
  if (nSrcROIWidth <= 0 || nSrcROIHeight <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (!validRate(nRate)) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  const float widthQuotient = static_cast<float>(nSrcROIWidth) / nRate;
  const float heightQuotient = static_cast<float>(nSrcROIHeight) / nRate;
  pDstSizeROI->width = clampDown(widthQuotient, nSrcROIWidth);
  pDstSizeROI->height = clampDown(heightQuotient, nSrcROIHeight);
  return NPP_NO_ERROR;
}

NppStatus nppiGetFilterGaussPyramidLayerUpBorderDstROI(int nSrcROIWidth, int nSrcROIHeight,
                                                       NppiSize *pDstSizeROIMin, NppiSize *pDstSizeROIMax,
                                                       Npp32f nRate) {
  if (pDstSizeROIMin == nullptr || pDstSizeROIMax == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  pDstSizeROIMin->width = -1;
  pDstSizeROIMin->height = -1;
  pDstSizeROIMax->width = -1;
  pDstSizeROIMax->height = -1;
  if (nSrcROIWidth <= 0 || nSrcROIHeight <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (!validRate(nRate)) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  const int maxWidth = std::max(nSrcROIWidth + 1, static_cast<int>(static_cast<float>(nSrcROIWidth) * nRate));
  const int maxHeight = std::max(nSrcROIHeight + 1, static_cast<int>(static_cast<float>(nSrcROIHeight) * nRate));
  pDstSizeROIMax->width = maxWidth;
  pDstSizeROIMax->height = maxHeight;

  const int minWidth =
      nSrcROIWidth == 1
          ? 1
          : std::max(nSrcROIWidth + 1, static_cast<int>(std::floor(static_cast<float>(nSrcROIWidth - 1) * nRate)));
  const int minHeight =
      nSrcROIHeight == 1
          ? 1
          : std::max(nSrcROIHeight + 1,
                     static_cast<int>(std::floor(static_cast<float>(nSrcROIHeight - 1) * nRate)));
  pDstSizeROIMin->width = minWidth;
  pDstSizeROIMin->height = minHeight;
  return NPP_NO_ERROR;
}
