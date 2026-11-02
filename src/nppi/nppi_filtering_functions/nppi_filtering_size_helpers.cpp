// Host-only scratch-buffer and descriptor-size helpers that were still
// missing from the filtering module.  Every formula and validation order was
// reverse-engineered from the NPP 13.2 libnppif host code (disassembly):
//
//   BoxBorderAdvanced:  4 * channels * w * h + 8      (_64: 8 * ... + 8)
//   HarrisCorners:      12 * w * h
//   HoughLine:          4 * ceil(max(w,h) * sqrt2 / (rho <= 1 ? rho : 1))
//                           * (theta <= 1 ? ceil(180 / theta) : 180)
//   HOG descriptors:    nLocations * (2 * blocksX - 1) * (2 * blocksY - 1)
//                           * bins * 16        (blocksX/Y from the window)
//   HOG buffer:         8 + 8 * spanX * spanY
//                           + 4 * nLocations * windowH * (windowW / cell)
//                               * (bins + 2)
//                       where span is the bounding box of all location
//                       windows: windowW + maxX - minX, likewise Y.
//   Watershed (16u):    2 * h * align64(16 * w) + 72
//   SignedPBA AA:       60 * w * h + 112  (64-bit wrap-around arithmetic)
//
// Non-positive ROI extents are NPP_SIZE_ERROR (-6), invalid configurations
// are NPP_NOT_SUPPORTED_MODE_ERROR (-14), out-of-range locations/counts are
// NPP_OUT_OFF_RANGE_ERROR (-11), invalid box channels or Hough deltas are
// NPP_BAD_ARGUMENT_ERROR (-5).  Outputs are written only on success.  NPP
// does not validate the size-pointer/hpLocations arguments; we return
// NPP_NULL_POINTER_ERROR instead of dereferencing null.
#include "npp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr float kHoughSqrt2 = 1.4142135381698608f;  // 0x3fb504f3, NPP's constant

bool validBoxChannels(int channels) {
  // NPP: (unsigned)(channels - 1) <= 3 and channels != 2 -> {1, 3, 4}.
  return channels == 1 || channels == 3 || channels == 4;
}

// Shared HOG configuration checks with the same order as NPP.  The window
// extents must already have been checked against zero by the caller.
bool hogConfigInvalid(const NppiHOGConfig &config) {
  if (config.cellSize < 2 || config.cellSize > 16) {
    return true;
  }
  if (config.cellSize > config.histogramBlockSize || config.histogramBlockSize > 64) {
    return true;
  }
  if (config.histogramBlockSize != 2 * config.cellSize) {
    return true;
  }
  if (config.detectionWindowSize.width % config.histogramBlockSize != 0 ||
      config.detectionWindowSize.height % config.histogramBlockSize != 0) {
    return true;
  }
  if (config.nHistogramBins < 2 || config.nHistogramBins > 16) {
    return true;
  }
  return false;
}

}  // namespace

NppStatus nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize oSizeROI, int nChannels,
                                                         int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (!validBoxChannels(nChannels)) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  *hpBufferSize = 4 * nChannels * oSizeROI.width * oSizeROI.height + 8;
  return NPP_NO_ERROR;
}

NppStatus nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize oSizeROI, int nChannels,
                                                            int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (!validBoxChannels(nChannels)) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  *hpBufferSize = 8 * nChannels * oSizeROI.width * oSizeROI.height + 8;
  return NPP_NO_ERROR;
}

NppStatus nppiFilterHarrisCornersBorderGetBufferSize(NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  *hpBufferSize = 12 * oSizeROI.width * oSizeROI.height;
  return NPP_NO_ERROR;
}

NppStatus nppiFilterHoughLineGetBufferSize(NppiSize oSizeROI, NppPointPolar nDelta,
                                           int nMaxLineCount, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (!(nDelta.theta > 0.25f && nDelta.theta <= 3.0f)) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (!(nDelta.rho > 0.0f && nDelta.rho <= 3.0f)) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (nMaxLineCount <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  const int largestExtent = std::max(oSizeROI.width, oSizeROI.height);
  float rhoBins = static_cast<float>(largestExtent) * kHoughSqrt2;
  rhoBins = rhoBins * 0.5f;
  rhoBins = rhoBins + rhoBins;
  if (nDelta.rho <= 1.0f) {
    rhoBins = rhoBins / nDelta.rho;
  }
  const int nRhoBins = static_cast<int>(std::ceil(rhoBins));
  const int nThetaBins =
      nDelta.theta <= 1.0f ? static_cast<int>(std::ceil(180.0f / nDelta.theta)) : 180;
  *hpBufferSize = 4 * nRhoBins * nThetaBins;
  return NPP_NO_ERROR;
}

NppStatus nppiHistogramOfGradientsBorderGetDescriptorsSize(const NppiHOGConfig oHOGConfig,
                                                           int nLocations, int *hpDescriptorsSize) {
  if (hpDescriptorsSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oHOGConfig.detectionWindowSize.width <= 0 || oHOGConfig.detectionWindowSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (hogConfigInvalid(oHOGConfig)) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  const int cellsX = oHOGConfig.detectionWindowSize.width / oHOGConfig.histogramBlockSize;
  const int cellsY = oHOGConfig.detectionWindowSize.height / oHOGConfig.histogramBlockSize;
  *hpDescriptorsSize =
      nLocations * (2 * cellsX - 1) * (2 * cellsY - 1) * oHOGConfig.nHistogramBins * 16;
  return NPP_NO_ERROR;
}

NppStatus nppiHistogramOfGradientsBorderGetBufferSize(const NppiHOGConfig oHOGConfig,
                                                      const NppiPoint *hpLocations, int nLocations,
                                                      NppiSize oSizeROI, int *hpBufferSize) {
  if (hpBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oHOGConfig.detectionWindowSize.width <= 0 || oHOGConfig.detectionWindowSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (hogConfigInvalid(oHOGConfig)) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  const int cellsPerDescriptor = (oHOGConfig.detectionWindowSize.width / oHOGConfig.cellSize) *
                                 (oHOGConfig.detectionWindowSize.height / oHOGConfig.cellSize);
  if (cellsPerDescriptor > 256) {
    // Measured: this particular limit reports OUT_OFF_RANGE (-11), unlike the
    // configuration checks above which report NOT_SUPPORTED_MODE (-14).
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (nLocations < 1 || nLocations > 1024) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (hpLocations == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  int minX = hpLocations[0].x;
  int minY = hpLocations[0].y;
  int maxX = hpLocations[0].x;
  int maxY = hpLocations[0].y;
  for (int i = 1; i < nLocations; ++i) {
    minX = std::min(minX, hpLocations[i].x);
    minY = std::min(minY, hpLocations[i].y);
    maxX = std::max(maxX, hpLocations[i].x);
    maxY = std::max(maxY, hpLocations[i].y);
  }
  if (minX < 0 || minY < 0) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (maxX + oHOGConfig.detectionWindowSize.width > oSizeROI.width ||
      maxY + oHOGConfig.detectionWindowSize.height > oSizeROI.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  const int spanX = oHOGConfig.detectionWindowSize.width + maxX - minX;
  const int spanY = oHOGConfig.detectionWindowSize.height + maxY - minY;
  const int cellsX = oHOGConfig.detectionWindowSize.width / oHOGConfig.cellSize;
  *hpBufferSize = 8 + 8 * spanX * spanY +
                  4 * nLocations * oHOGConfig.detectionWindowSize.height * cellsX *
                      (oHOGConfig.nHistogramBins + 2);
  return NPP_NO_ERROR;
}

extern "C" NppStatus nppiSegmentWatershedGetBufferSize_16u_C1R(NppiSize oSizeROI,
                                                               int *hpDeviceMemoryBufferSize) {
  if (hpDeviceMemoryBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  const size_t alignedWidth =
      (static_cast<size_t>(oSizeROI.width) * 16u + 63u) & ~static_cast<size_t>(63u);
  const size_t size = 2u * static_cast<size_t>(oSizeROI.height) * alignedWidth + 72u;
  *hpDeviceMemoryBufferSize = static_cast<int>(size);
  return NPP_NO_ERROR;
}

NppStatus nppiSegmentWatershedGetBufferSize_16u_C1R(NppiSize oSizeROI,
                                                    size_t *hpDeviceMemoryBufferSize) {
  if (hpDeviceMemoryBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  const size_t alignedWidth =
      (static_cast<size_t>(oSizeROI.width) * 16u + 63u) & ~static_cast<size_t>(63u);
  *hpDeviceMemoryBufferSize = 2u * static_cast<size_t>(oSizeROI.height) * alignedWidth + 72u;
  return NPP_NO_ERROR;
}

NppStatus nppiSignedDistanceTransformPBAGetAntialiasingBufferSize(
    NppiSize oSizeROI, size_t *hpAntialiasingBufferSize) {
  if (hpAntialiasingBufferSize == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  // NPP computes this with 64-bit wrap-around arithmetic and does not reject
  // negative extents; keep the same wrap-around.
  const uint64_t width = static_cast<uint64_t>(static_cast<int64_t>(oSizeROI.width));
  const uint64_t height = static_cast<uint64_t>(static_cast<int64_t>(oSizeROI.height));
  *hpAntialiasingBufferSize = static_cast<size_t>(60u * width * height + 112u);
  return NPP_NO_ERROR;
}
