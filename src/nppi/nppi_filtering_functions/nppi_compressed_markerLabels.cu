#include "npp.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

// Contour direction constants (from nppdefs.h)
#define CONTOUR_DIR_SE 1
#define CONTOUR_DIR_S 2
#define CONTOUR_DIR_SW 4
#define CONTOUR_DIR_W 8
#define CONTOUR_DIR_E 16
#define CONTOUR_DIR_NE 32
#define CONTOUR_DIR_N 64
#define CONTOUR_DIR_NW 128

namespace {

constexpr int kNeighborCount = 8;
constexpr int kNeighborDx[kNeighborCount] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kNeighborDy[kNeighborCount] = {0, -1, -1, -1, 0, 1, 1, 1};
constexpr Npp8u kNeighborBits[kNeighborCount] = {CONTOUR_DIR_E, CONTOUR_DIR_NE, CONTOUR_DIR_N, CONTOUR_DIR_NW,
                                                 CONTOUR_DIR_W, CONTOUR_DIR_SW, CONTOUR_DIR_S, CONTOUR_DIR_SE};
constexpr Npp32u kMaxContourPixelsPerSegment = 256u * 1024u;

struct HostCompressedMarkerContourData {
  std::vector<Npp8u> contourImage;
  std::vector<NppiContourPixelDirectionInfo> directionImage;
  std::vector<NppiCompressedMarkerLabelsInfo> infoList;
  std::vector<Npp32u> contourCounts;
  std::vector<Npp32u> contourOffsets;
  NppiContourTotalsInfo totalsInfo{};
};

inline bool isInBounds(int x, int y, int width, int height) { return x >= 0 && x < width && y >= 0 && y < height; }

inline size_t pixelOffset(int x, int y, int width) { return static_cast<size_t>(y) * width + x; }

template <typename T>
cudaError_t copyDeviceImageToHost(std::vector<T> &host, const T *device, int deviceStep, int width, int height) {
  host.assign(static_cast<size_t>(width) * height, T{});
  if (device == nullptr) {
    return cudaSuccess;
  }

  return cudaMemcpy2D(host.data(), static_cast<size_t>(width) * sizeof(T), device, deviceStep,
                      static_cast<size_t>(width) * sizeof(T), height, cudaMemcpyDeviceToHost);
}

template <typename T>
cudaError_t copyHostImageToDevice(T *device, int deviceStep, const std::vector<T> &host, int width, int height) {
  if (device == nullptr) {
    return cudaSuccess;
  }

  return cudaMemcpy2D(device, deviceStep, host.data(), static_cast<size_t>(width) * sizeof(T),
                      static_cast<size_t>(width) * sizeof(T), height, cudaMemcpyHostToDevice);
}

template <typename T> cudaError_t copyHostArrayToDevice(T *device, const std::vector<T> &host, size_t count) {
  if (device == nullptr || count == 0) {
    return cudaSuccess;
  }

  return cudaMemcpy(device, host.data(), count * sizeof(T), cudaMemcpyHostToDevice);
}

// Count pixels per label (including label 0, which is observable in info outputs).
__global__ void countPixelsPerLabel_kernel(const Npp32u *pMarkerLabels, int nMarkerStep, int width, int height,
                                           Npp32u *pPixelCounts, unsigned int nMaxMarkerLabelID) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp32u *row = reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + y * nMarkerStep);
  const Npp32u label = row[x];
  if (label <= nMaxMarkerLabelID && pPixelCounts != nullptr) {
    atomicAdd(&pPixelCounts[label], 1u);
  }
}

// Restored GPU path for contour extraction to avoid host-only regressions.
__global__ void detectContourPixels_kernel(const Npp32u *pMarkerLabels, int nMarkerStep, int width, int height,
                                           Npp8u *pContoursImage, int nContoursStep, Npp32u *pContourCounts,
                                           unsigned int nMaxMarkerLabelID) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp32u *labelRow = reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + y * nMarkerStep);
  const Npp32u label = labelRow[x];
  if (label > nMaxMarkerLabelID) {
    if (pContoursImage != nullptr) {
      Npp8u *contourRow = reinterpret_cast<Npp8u *>(reinterpret_cast<char *>(pContoursImage) + y * nContoursStep);
      contourRow[x] = 0;
    }
    return;
  }

  bool isContour = false;
  const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
  for (int i = 0; i < 8; ++i) {
    const int nx = x + dx[i];
    const int ny = y + dy[i];
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
      continue;
    }
    const Npp32u *neighborRow =
        reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + ny * nMarkerStep);
    if (neighborRow[nx] != label) {
      isContour = true;
      break;
    }
  }

  if (pContoursImage != nullptr) {
    Npp8u *contourRow = reinterpret_cast<Npp8u *>(reinterpret_cast<char *>(pContoursImage) + y * nContoursStep);
    contourRow[x] = isContour ? 255 : 0;
  }
  if (isContour && pContourCounts != nullptr) {
    atomicAdd(&pContourCounts[label], 1u);
  }
}

__global__ void updateBoundingBox_kernel(const Npp32u *pMarkerLabels, int nMarkerStep, int width, int height, int *pMinX,
                                         int *pMinY, int *pMaxX, int *pMaxY, unsigned int nMaxMarkerLabelID) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp32u *row = reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + y * nMarkerStep);
  const Npp32u label = row[x];
  if (label > nMaxMarkerLabelID) {
    return;
  }

  atomicMin(&pMinX[label], x);
  atomicMin(&pMinY[label], y);
  atomicMax(&pMaxX[label], x);
  atomicMax(&pMaxY[label], y);
}

__global__ void findFirstContourPixel_kernel(const Npp8u *pContoursImage, int nContoursStep, const Npp32u *pMarkerLabels,
                                             int nMarkerStep, int width, int height, int *pFirstContourEncoded,
                                             unsigned int nMaxMarkerLabelID) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *contourRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pContoursImage) + y * nContoursStep);
  if (contourRow[x] == 0) {
    return;
  }

  const Npp32u *labelRow = reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + y * nMarkerStep);
  const Npp32u label = labelRow[x];
  if (label > nMaxMarkerLabelID) {
    return;
  }

  atomicMin(&pFirstContourEncoded[label], y * width + x);
}

__global__ void computeContourDirection_kernel(const Npp32u *pMarkerLabels, int nMarkerStep, int width, int height,
                                               const Npp8u *pContoursImage, int nContoursStep,
                                               NppiContourPixelDirectionInfo *pDirectionImage, int nDirectionStep,
                                               unsigned int nMaxMarkerLabelID) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp32u *labelRow = reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + y * nMarkerStep);
  const Npp32u label = labelRow[x];
  NppiContourPixelDirectionInfo *dirRow =
      reinterpret_cast<NppiContourPixelDirectionInfo *>(reinterpret_cast<char *>(pDirectionImage) + y * nDirectionStep);
  NppiContourPixelDirectionInfo info{};
  info.nMarkerLabelID = label;
  if (label > nMaxMarkerLabelID) {
    dirRow[x] = info;
    return;
  }

  Npp8u contourNeighborMask = 0;
  Npp8u interiorMask = 0;
  const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
  const Npp8u bits[8] = {CONTOUR_DIR_E, CONTOUR_DIR_NE, CONTOUR_DIR_N, CONTOUR_DIR_NW,
                         CONTOUR_DIR_W, CONTOUR_DIR_SW, CONTOUR_DIR_S, CONTOUR_DIR_SE};

  for (int i = 0; i < 8; ++i) {
    const int nx = x + dx[i];
    const int ny = y + dy[i];
    bool sameLabel = true;
    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
      const Npp32u *neighborRow =
          reinterpret_cast<const Npp32u *>(reinterpret_cast<const char *>(pMarkerLabels) + ny * nMarkerStep);
      sameLabel = (neighborRow[nx] == label);
      if (sameLabel) {
        interiorMask |= bits[i];
        if (pContoursImage != nullptr) {
          const Npp8u *contourRow = reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pContoursImage) +
                                                                     ny * nContoursStep);
          if (contourRow[nx] != 0) {
            contourNeighborMask |= bits[i];
          }
        }
      }
      switch (i) {
      case 0:
        info.nEast1.x = nx;
        info.nEast1.y = ny;
        break;
      case 1:
        info.nNorthEast1.x = nx;
        info.nNorthEast1.y = ny;
        break;
      case 2:
        info.nNorth1.x = nx;
        info.nNorth1.y = ny;
        break;
      case 3:
        info.nNorthWest1.x = nx;
        info.nNorthWest1.y = ny;
        break;
      case 4:
        info.nWest1.x = nx;
        info.nWest1.y = ny;
        break;
      case 5:
        info.nSouthWest1.x = nx;
        info.nSouthWest1.y = ny;
        break;
      case 6:
        info.nSouth1.x = nx;
        info.nSouth1.y = ny;
        break;
      case 7:
        info.nSouthEast1.x = nx;
        info.nSouthEast1.y = ny;
        break;
      default:
        break;
      }
    }

    const Npp8u connected = sameLabel ? 1u : 0u;
    switch (i) {
    case 0:
      info.nTest1EastConnected = connected;
      break;
    case 1:
      info.nTest1NorthEastConnected = connected;
      break;
    case 2:
      info.nTest1NorthConnected = connected;
      break;
    case 3:
      info.nTest1NorthWestConnected = connected;
      break;
    case 4:
      info.nTest1WestConnected = connected;
      break;
    case 5:
      info.nTest1SouthWestConnected = connected;
      break;
    case 6:
      info.nTest1SouthConnected = connected;
      break;
    case 7:
      info.nTest1SouthEastConnected = connected;
      break;
    default:
      break;
    }
  }

  if (pContoursImage != nullptr) {
    const Npp8u *centerContourRow =
        reinterpret_cast<const Npp8u *>(reinterpret_cast<const char *>(pContoursImage) + y * nContoursStep);
    if (centerContourRow[x] != 0) {
      info.nContourDirectionCenterPixel = contourNeighborMask;
      info.nContourInteriorDirectionCenterPixel = interiorMask;
      info.nConnected = contourNeighborMask;
    }
  }
  dirRow[x] = info;
}

inline void setNeighborPoint(NppiContourPixelDirectionInfo &info, int direction, int x, int y) {
  switch (direction) {
  case 0:
    info.nEast1.x = x;
    info.nEast1.y = y;
    break;
  case 1:
    info.nNorthEast1.x = x;
    info.nNorthEast1.y = y;
    break;
  case 2:
    info.nNorth1.x = x;
    info.nNorth1.y = y;
    break;
  case 3:
    info.nNorthWest1.x = x;
    info.nNorthWest1.y = y;
    break;
  case 4:
    info.nWest1.x = x;
    info.nWest1.y = y;
    break;
  case 5:
    info.nSouthWest1.x = x;
    info.nSouthWest1.y = y;
    break;
  case 6:
    info.nSouth1.x = x;
    info.nSouth1.y = y;
    break;
  case 7:
    info.nSouthEast1.x = x;
    info.nSouthEast1.y = y;
    break;
  default:
    break;
  }
}

inline void setNeighborConnectedFlag(NppiContourPixelDirectionInfo &info, int direction, bool connected) {
  const Npp8u value = connected ? 1u : 0u;
  switch (direction) {
  case 0:
    info.nTest1EastConnected = value;
    break;
  case 1:
    info.nTest1NorthEastConnected = value;
    break;
  case 2:
    info.nTest1NorthConnected = value;
    break;
  case 3:
    info.nTest1NorthWestConnected = value;
    break;
  case 4:
    info.nTest1WestConnected = value;
    break;
  case 5:
    info.nTest1SouthWestConnected = value;
    break;
  case 6:
    info.nTest1SouthConnected = value;
    break;
  case 7:
    info.nTest1SouthEastConnected = value;
    break;
  default:
    break;
  }
}

inline bool hasMissingNeighbor(const NppiContourPixelDirectionInfo &info) {
  return info.nTest1EastConnected == 0 || info.nTest1NorthEastConnected == 0 || info.nTest1NorthConnected == 0 ||
         info.nTest1NorthWestConnected == 0 || info.nTest1WestConnected == 0 || info.nTest1SouthWestConnected == 0 ||
         info.nTest1SouthConnected == 0 || info.nTest1SouthEastConnected == 0;
}

inline Npp8u selectPrimaryDirection(Npp8u directionMask) {
  for (int i = 0; i < kNeighborCount; ++i) {
    if ((directionMask & kNeighborBits[i]) != 0) {
      return kNeighborBits[i];
    }
  }
  return 0;
}

[[maybe_unused]] bool analyzeCompressedMarkerContours(const std::vector<Npp32u> &labels, int width, int height,
                                                      unsigned int nMaxMarkerLabelID,
                                                      HostCompressedMarkerContourData *analysis) {
  if (analysis == nullptr) {
    return false;
  }

  const size_t pixelCount = static_cast<size_t>(width) * height;
  analysis->contourImage.assign(pixelCount, 0);
  analysis->directionImage.assign(pixelCount, NppiContourPixelDirectionInfo{});
  analysis->infoList.assign(static_cast<size_t>(nMaxMarkerLabelID + 1), NppiCompressedMarkerLabelsInfo{});
  analysis->contourCounts.assign(static_cast<size_t>(nMaxMarkerLabelID + 1), 0);
  analysis->contourOffsets.assign(static_cast<size_t>(nMaxMarkerLabelID + 1), 0);
  analysis->totalsInfo = NppiContourTotalsInfo{};

  std::vector<int> minX(nMaxMarkerLabelID + 1, width);
  std::vector<int> minY(nMaxMarkerLabelID + 1, height);
  std::vector<int> maxX(nMaxMarkerLabelID + 1, -1);
  std::vector<int> maxY(nMaxMarkerLabelID + 1, -1);
  std::vector<int> firstContourX(nMaxMarkerLabelID + 1, 0);
  std::vector<int> firstContourY(nMaxMarkerLabelID + 1, 0);
  std::vector<Npp8u> hasFirstContour(nMaxMarkerLabelID + 1, 0);
  std::vector<Npp8u> isContour(pixelCount, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = pixelOffset(x, y, width);
      const Npp32u label = labels[idx];
      // Labels outside declared range are ignored; label 0 (background) is kept
      // so callers can query background contour statistics.
      if (label > nMaxMarkerLabelID) {
        continue;
      }

      analysis->infoList[label].nMarkerLabelPixelCount++;
      minX[label] = std::min(minX[label], x);
      minY[label] = std::min(minY[label], y);
      maxX[label] = std::max(maxX[label], x);
      maxY[label] = std::max(maxY[label], y);
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = pixelOffset(x, y, width);
      const Npp32u label = labels[idx];
      if (label > nMaxMarkerLabelID) {
        continue;
      }

      bool contourPixel = false;
      for (int direction = 0; direction < kNeighborCount; ++direction) {
        const int nx = x + kNeighborDx[direction];
        const int ny = y + kNeighborDy[direction];
        if (!isInBounds(nx, ny, width, height)) {
          continue;
        }

        if (labels[pixelOffset(nx, ny, width)] != label) {
          contourPixel = true;
          break;
        }
      }

      if (!contourPixel) {
        continue;
      }

      isContour[idx] = 1;
      analysis->contourImage[idx] = 255;
      analysis->contourCounts[label]++;

      if (hasFirstContour[label] == 0 || x < firstContourX[label] ||
          (x == firstContourX[label] && y < firstContourY[label])) {
        firstContourX[label] = x;
        firstContourY[label] = y;
        hasFirstContour[label] = 1;
      }
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = pixelOffset(x, y, width);
      const Npp32u label = labels[idx];
      NppiContourPixelDirectionInfo info{};
      info.nMarkerLabelID = label;

      if (label <= nMaxMarkerLabelID) {
        Npp8u contourNeighborMask = 0;
        Npp8u interiorMask = 0;

        for (int direction = 0; direction < kNeighborCount; ++direction) {
          const int nx = x + kNeighborDx[direction];
          const int ny = y + kNeighborDy[direction];
          bool sameLabel = true;

          if (isInBounds(nx, ny, width, height)) {
            setNeighborPoint(info, direction, nx, ny);
            sameLabel = (labels[pixelOffset(nx, ny, width)] == label);
            if (sameLabel) {
              interiorMask |= kNeighborBits[direction];
              if (isContour[pixelOffset(nx, ny, width)] != 0) {
                contourNeighborMask |= kNeighborBits[direction];
              }
            }
          }

          setNeighborConnectedFlag(info, direction, sameLabel);
        }

        if (isContour[idx] != 0) {
          // Preserve the historical nConnected semantics used by callers that
          // inspect direction-image masks directly.
          info.nContourDirectionCenterPixel = contourNeighborMask;
          info.nContourInteriorDirectionCenterPixel = interiorMask;
          info.nConnected = contourNeighborMask;
        }
      }

      analysis->directionImage[idx] = info;
    }
  }

  Npp32u runningOffset = 0;
  Npp32u longestContour = 0;
  for (unsigned int label = 0; label <= nMaxMarkerLabelID; ++label) {
    NppiCompressedMarkerLabelsInfo &info = analysis->infoList[label];
    info.nContourPixelCount = analysis->contourCounts[label];
    info.nContourPixelsFound = 0;
    analysis->contourOffsets[label] = runningOffset;
    runningOffset += analysis->contourCounts[label];
    longestContour = std::max(longestContour, analysis->contourCounts[label]);

    if (maxX[label] >= 0) {
      info.oMarkerLabelBoundingBox.x = minX[label];
      info.oMarkerLabelBoundingBox.y = minY[label];
      info.oMarkerLabelBoundingBox.width = maxX[label];
      info.oMarkerLabelBoundingBox.height = maxY[label];
    } else {
      info.oMarkerLabelBoundingBox = NppiRect{0, 0, 0, 0};
    }

    if (hasFirstContour[label] != 0) {
      info.oContourFirstPixelLocation.x = firstContourX[label];
      info.oContourFirstPixelLocation.y = firstContourY[label];
    } else {
      info.oContourFirstPixelLocation = NppiPoint{0, 0};
    }
  }

  analysis->totalsInfo.nTotalImagePixelContourCount = runningOffset;
  analysis->totalsInfo.nLongestImageContourPixelCount = longestContour;
  return true;
}

std::vector<Npp32u> deriveContourList(const Npp32u *deviceList, size_t count) {
  std::vector<Npp32u> derived(count, 0);
  if (deviceList != nullptr && count > 0 &&
      cudaMemcpy(derived.data(), deviceList, count * sizeof(Npp32u), cudaMemcpyDeviceToHost) == cudaSuccess) {
    return derived;
  }
  return derived;
}

bool isContourDirectionPixel(const NppiContourPixelDirectionInfo &info, Npp32u labelID) {
  return info.nMarkerLabelID == labelID && hasMissingNeighbor(info);
}

int chooseStartDirection(int backtrackDirection, bool counterclockwise) {
  return counterclockwise ? (backtrackDirection + 6) % kNeighborCount : (backtrackDirection + 2) % kNeighborCount;
}

std::vector<NppiPoint> traceOrderedContourForLabel(const std::vector<NppiContourPixelDirectionInfo> &directionImage,
                                                   int width, int height, Npp32u labelID, bool counterclockwise) {
  std::vector<NppiPoint> contourPixels;
  std::vector<Npp8u> contourMask(static_cast<size_t>(width) * height, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = pixelOffset(x, y, width);
      if (!isContourDirectionPixel(directionImage[idx], labelID)) {
        continue;
      }

      contourMask[idx] = 1;
      contourPixels.push_back(NppiPoint{x, y});
    }
  }

  if (contourPixels.empty()) {
    return {};
  }

  std::sort(contourPixels.begin(), contourPixels.end(), [](const NppiPoint &lhs, const NppiPoint &rhs) {
    if (lhs.x != rhs.x) {
      return lhs.x < rhs.x;
    }
    return lhs.y < rhs.y;
  });

  const NppiPoint start = contourPixels.front();
  std::vector<Npp8u> visited(static_cast<size_t>(width) * height, 0);
  std::vector<NppiPoint> ordered;
  ordered.reserve(contourPixels.size());

  // Moore-neighborhood contour walk with deterministic left-top start point.
  int currentX = start.x;
  int currentY = start.y;
  int backtrackDirection = 4;
  int previousX = start.x - 1;
  int previousY = start.y;

  while (ordered.size() < contourPixels.size()) {
    const size_t currentIdx = pixelOffset(currentX, currentY, width);
    if (visited[currentIdx] == 0) {
      ordered.push_back(NppiPoint{currentX, currentY});
      visited[currentIdx] = 1;
    }

    bool foundNext = false;
    const int startDirection = chooseStartDirection(backtrackDirection, counterclockwise);
    for (int offset = 0; offset < kNeighborCount; ++offset) {
      const int direction = counterclockwise ? (startDirection + offset) % kNeighborCount
                                             : (startDirection + kNeighborCount - offset) % kNeighborCount;
      const int nextX = currentX + kNeighborDx[direction];
      const int nextY = currentY + kNeighborDy[direction];
      if (!isInBounds(nextX, nextY, width, height)) {
        continue;
      }

      const size_t nextIdx = pixelOffset(nextX, nextY, width);
      if (contourMask[nextIdx] == 0) {
        continue;
      }
      if (nextX == previousX && nextY == previousY) {
        continue;
      }
      if (visited[nextIdx] != 0 && !(nextX == start.x && nextY == start.y && ordered.size() == contourPixels.size())) {
        continue;
      }

      previousX = currentX;
      previousY = currentY;
      currentX = nextX;
      currentY = nextY;
      backtrackDirection = (direction + 4) % kNeighborCount;
      foundNext = true;
      break;
    }

    if (!foundNext) {
      break;
    }
    if (currentX == start.x && currentY == start.y) {
      break;
    }
  }

  for (const NppiPoint &point : contourPixels) {
    const size_t idx = pixelOffset(point.x, point.y, width);
    if (visited[idx] == 0) {
      ordered.push_back(point);
      visited[idx] = 1;
    }
  }

  if (!counterclockwise) {
    std::reverse(ordered.begin(), ordered.end());
  }
  return ordered;
}

void fillBlockSegments(std::vector<NppiContourBlockSegment> &segments, Npp32u labelID, Npp32u contourPixelCount,
                       Npp32u startingOffset) {
  if (contourPixelCount == 0) {
    return;
  }

  Npp32u segmentNum = 0;
  Npp32u remaining = contourPixelCount;
  Npp32u offset = startingOffset;
  while (remaining > 0) {
    const Npp32u segmentPixels = std::min(remaining, kMaxContourPixelsPerSegment);
    NppiContourBlockSegment segment{};
    segment.nMarkerLabelID = labelID;
    segment.nContourPixelCount = segmentPixels;
    segment.nContourStartingPixelOffset = offset;
    segment.nSegmentNum = segmentNum++;
    segments.push_back(segment);
    remaining -= segmentPixels;
    offset += segmentPixels;
  }
}

} // namespace

extern "C" {

NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                       NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                       int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                       NppStreamContext nppStreamCtx);

// Get info list buffer size
NppStatus nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R_impl(unsigned int nMaxMarkerLabelID,
                                                                   unsigned int *hpBufferSize) {
  // Need space for nMaxMarkerLabelID + 1 info structures (index 0 is unused)
  size_t size = (nMaxMarkerLabelID + 1) * sizeof(NppiCompressedMarkerLabelsInfo);
  *hpBufferSize = (unsigned int)((size + 255) & ~255); // Align to 256 bytes
  return NPP_SUCCESS;
}

// Get compressed marker labels info
NppStatus nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx_impl(
    Npp32u *pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI,
    unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoList, Npp8u *pContoursImage,
    Npp32s nContoursImageStep, NppiContourPixelDirectionInfo *pContoursDirectionImage,
    Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo *pContoursTotalsInfoHost,
    Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelCountsListHost, Npp32u *pContoursPixelStartingOffsetDev,
    Npp32u *pContoursPixelStartingOffsetHost, NppStreamContext nppStreamCtx) {
  const int width = oSizeROI.width;
  const int height = oSizeROI.height;
  const size_t pixelCount = static_cast<size_t>(width) * height;
  const size_t labelCount = static_cast<size_t>(nMaxMarkerLabelID + 1);
  const size_t countsSizeBytes = labelCount * sizeof(Npp32u);
  const size_t bboxSizeBytes = labelCount * sizeof(int);
  cudaStream_t stream = nppStreamCtx.hStream;

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  Npp8u *dInternalContoursImage = nullptr;
  Npp32u *dPixelCounts = nullptr;
  Npp32u *dContourCounts = nullptr;
  int *dMinX = nullptr;
  int *dMinY = nullptr;
  int *dMaxX = nullptr;
  int *dMaxY = nullptr;
  int *dFirstContourEncoded = nullptr;

  auto cleanup = [&]() {
    if (dInternalContoursImage != nullptr) {
      cudaFree(dInternalContoursImage);
    }
    if (dPixelCounts != nullptr) {
      cudaFree(dPixelCounts);
    }
    if (dContourCounts != nullptr) {
      cudaFree(dContourCounts);
    }
    if (dMinX != nullptr) {
      cudaFree(dMinX);
    }
    if (dMinY != nullptr) {
      cudaFree(dMinY);
    }
    if (dMaxX != nullptr) {
      cudaFree(dMaxX);
    }
    if (dMaxY != nullptr) {
      cudaFree(dMaxY);
    }
    if (dFirstContourEncoded != nullptr) {
      cudaFree(dFirstContourEncoded);
    }
  };

  Npp8u *contoursImageForKernels = pContoursImage;
  Npp32s contoursStepForKernels = nContoursImageStep;
  if (contoursImageForKernels == nullptr) {
    if (cudaMalloc(&dInternalContoursImage, pixelCount * sizeof(Npp8u)) != cudaSuccess) {
      cleanup();
      return NPP_MEMORY_ALLOCATION_ERR;
    }
    contoursImageForKernels = dInternalContoursImage;
    contoursStepForKernels = static_cast<Npp32s>(width * sizeof(Npp8u));
  }

  if (cudaMalloc(&dPixelCounts, countsSizeBytes) != cudaSuccess || cudaMalloc(&dContourCounts, countsSizeBytes) != cudaSuccess ||
      cudaMalloc(&dMinX, bboxSizeBytes) != cudaSuccess || cudaMalloc(&dMinY, bboxSizeBytes) != cudaSuccess ||
      cudaMalloc(&dMaxX, bboxSizeBytes) != cudaSuccess || cudaMalloc(&dMaxY, bboxSizeBytes) != cudaSuccess ||
      cudaMalloc(&dFirstContourEncoded, bboxSizeBytes) != cudaSuccess) {
    cleanup();
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  if (cudaMemsetAsync(dPixelCounts, 0, countsSizeBytes, stream) != cudaSuccess ||
      cudaMemsetAsync(dContourCounts, 0, countsSizeBytes, stream) != cudaSuccess) {
    cleanup();
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  std::vector<int> hMinInit(labelCount, width);
  std::vector<int> hMinYInit(labelCount, height);
  std::vector<int> hMaxInit(labelCount, -1);
  std::vector<int> hFirstInit(labelCount, INT_MAX);
  if (cudaMemcpyAsync(dMinX, hMinInit.data(), bboxSizeBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(dMinY, hMinYInit.data(), bboxSizeBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(dMaxX, hMaxInit.data(), bboxSizeBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(dMaxY, hMaxInit.data(), bboxSizeBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(dFirstContourEncoded, hFirstInit.data(), bboxSizeBytes, cudaMemcpyHostToDevice, stream) !=
          cudaSuccess) {
    cleanup();
    return NPP_MEMCPY_ERROR;
  }

  countPixelsPerLabel_kernel<<<gridSize, blockSize, 0, stream>>>(pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width,
                                                                  height, dPixelCounts, nMaxMarkerLabelID);
  detectContourPixels_kernel<<<gridSize, blockSize, 0, stream>>>(
      pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width, height, contoursImageForKernels,
      contoursStepForKernels, dContourCounts, nMaxMarkerLabelID);
  updateBoundingBox_kernel<<<gridSize, blockSize, 0, stream>>>(pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width,
                                                               height, dMinX, dMinY, dMaxX, dMaxY, nMaxMarkerLabelID);
  findFirstContourPixel_kernel<<<gridSize, blockSize, 0, stream>>>(
      contoursImageForKernels, contoursStepForKernels, pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width, height,
      dFirstContourEncoded, nMaxMarkerLabelID);

  if (pContoursDirectionImage != nullptr) {
    computeContourDirection_kernel<<<gridSize, blockSize, 0, stream>>>(
        pCompressedMarkerLabels, nCompressedMarkerLabelsStep, width, height, contoursImageForKernels,
        contoursStepForKernels, pContoursDirectionImage, nContoursDirectionImageStep, nMaxMarkerLabelID);
  }

  if (cudaStreamSynchronize(stream) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
    cleanup();
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  std::vector<Npp32u> hPixelCounts(labelCount, 0);
  std::vector<Npp32u> hContourCounts(labelCount, 0);
  std::vector<int> hMinX(labelCount, width);
  std::vector<int> hMinY(labelCount, height);
  std::vector<int> hMaxX(labelCount, -1);
  std::vector<int> hMaxY(labelCount, -1);
  std::vector<int> hFirstContourPos(labelCount, INT_MAX);
  if (cudaMemcpy(hPixelCounts.data(), dPixelCounts, countsSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hContourCounts.data(), dContourCounts, countsSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hMinX.data(), dMinX, bboxSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hMinY.data(), dMinY, bboxSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hMaxX.data(), dMaxX, bboxSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hMaxY.data(), dMaxY, bboxSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hFirstContourPos.data(), dFirstContourEncoded, bboxSizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cleanup();
    return NPP_MEMCPY_ERROR;
  }

  std::vector<NppiCompressedMarkerLabelsInfo> hInfoList(labelCount);
  std::vector<Npp32u> hContourOffsets(labelCount, 0);
  Npp32u runningOffset = 0;
  Npp32u longestContour = 0;
  for (unsigned int label = 0; label <= nMaxMarkerLabelID; ++label) {
    NppiCompressedMarkerLabelsInfo &info = hInfoList[label];
    info.nMarkerLabelPixelCount = hPixelCounts[label];
    info.nContourPixelCount = hContourCounts[label];
    info.nContourPixelsFound = 0;

    if (hMaxX[label] >= 0) {
      info.oMarkerLabelBoundingBox.x = hMinX[label];
      info.oMarkerLabelBoundingBox.y = hMinY[label];
      // NVIDIA stores right/bottom coordinates here, not geometric width/height.
      info.oMarkerLabelBoundingBox.width = hMaxX[label];
      info.oMarkerLabelBoundingBox.height = hMaxY[label];
    } else {
      info.oMarkerLabelBoundingBox = NppiRect{0, 0, 0, 0};
    }

    if (hFirstContourPos[label] != INT_MAX) {
      info.oContourFirstPixelLocation.x = hFirstContourPos[label] % width;
      info.oContourFirstPixelLocation.y = hFirstContourPos[label] / width;
    } else {
      info.oContourFirstPixelLocation = NppiPoint{0, 0};
    }

    hContourOffsets[label] = runningOffset;
    runningOffset += hContourCounts[label];
    longestContour = std::max(longestContour, hContourCounts[label]);
  }

  if (cudaMemcpy(pMarkerLabelsInfoList, hInfoList.data(), labelCount * sizeof(NppiCompressedMarkerLabelsInfo),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cleanup();
    return NPP_MEMCPY_ERROR;
  }

  if (pContoursPixelCountsListHost != nullptr) {
    std::memcpy(pContoursPixelCountsListHost, hContourCounts.data(), countsSizeBytes);
  }
  if (pContoursPixelStartingOffsetHost != nullptr) {
    std::memcpy(pContoursPixelStartingOffsetHost, hContourOffsets.data(), countsSizeBytes);
  }
  if (pContoursPixelCountsListDev != nullptr &&
      cudaMemcpy(pContoursPixelCountsListDev, hContourCounts.data(), countsSizeBytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    cleanup();
    return NPP_MEMCPY_ERROR;
  }
  if (pContoursPixelStartingOffsetDev != nullptr &&
      cudaMemcpy(pContoursPixelStartingOffsetDev, hContourOffsets.data(), countsSizeBytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    cleanup();
    return NPP_MEMCPY_ERROR;
  }

  if (pContoursTotalsInfoHost != nullptr) {
    pContoursTotalsInfoHost->nTotalImagePixelContourCount = runningOffset;
    pContoursTotalsInfoHost->nLongestImageContourPixelCount = longestContour;
  }

  cleanup();
  return NPP_SUCCESS;
}

// Get geometry lists buffer size
NppStatus nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R_impl(Npp32u nMaxContourPixelGeometryInfoCount,
                                                                    Npp32u *hpBufferSize) {
  size_t size = nMaxContourPixelGeometryInfoCount * sizeof(NppiContourPixelGeometryInfo);
  *hpBufferSize = (Npp32u)((size + 255) & ~255);
  return NPP_SUCCESS;
}

NppStatus nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R_impl(
    Npp32u *pContoursPixelCountsListHost, Npp32u nTotalImagePixelContourCount, Npp32u nCompressedLabelCount,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID, Npp32u *hpBufferSize) {
  constexpr Npp32u kMaxContourPixelsPerSegment = 256u * 1024u;

  if (nTotalImagePixelContourCount == 0) {
    *hpBufferSize = 0;
    return NPP_SUCCESS;
  }

  size_t segmentCount = 0;
  for (Npp32u labelID = nFirstContourGeometryListID;
       labelID < nLastContourGeometryListID && labelID <= nCompressedLabelCount; ++labelID) {
    Npp32u contourPixelCount = pContoursPixelCountsListHost[labelID];
    if (contourPixelCount == 0) {
      continue;
    }
    segmentCount += (contourPixelCount + kMaxContourPixelsPerSegment - 1) / kMaxContourPixelsPerSegment;
  }

  size_t size = segmentCount * sizeof(NppiContourBlockSegment);
  *hpBufferSize = static_cast<Npp32u>((size + 255) & ~255);
  return NPP_SUCCESS;
}

// Generate geometry lists
NppStatus nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx_impl(
    NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo *pMarkerLabelsInfoListHost,
    NppiContourPixelDirectionInfo *pContoursDirectionImageDev, Npp32s nContoursDirectionImageStep,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsDev,
    NppiContourPixelGeometryInfo *pContoursPixelGeometryListsHost, Npp8u *pContoursGeometryImageHost,
    Npp32s nContoursGeometryImageStep, Npp32u *pContoursPixelCountsListDev, Npp32u *pContoursPixelsFoundListDev,
    Npp32u *pContoursPixelsFoundListHost, Npp32u *pContoursPixelsStartingOffsetDev,
    Npp32u *pContoursPixelsStartingOffsetHost, Npp32u nTotalImagePixelContourCount, Npp32u nMaxMarkerLabelID,
    Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID,
    NppiContourBlockSegment *pContoursBlockSegmentListDev, NppiContourBlockSegment *pContoursBlockSegmentListHost,
    Npp32u bOutputInCounterclockwiseOrder, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  const int width = oSizeROI.width;
  const int height = oSizeROI.height;
  const size_t labelListSize = static_cast<size_t>(nMaxMarkerLabelID + 1);

  std::vector<NppiContourPixelDirectionInfo> hostDirectionImage;
  if (copyDeviceImageToHost(hostDirectionImage, pContoursDirectionImageDev, nContoursDirectionImageStep, width,
                            height) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  std::vector<Npp32u> contourCounts = deriveContourList(pContoursPixelCountsListDev, labelListSize);
  if (std::all_of(contourCounts.begin(), contourCounts.end(), [](Npp32u value) { return value == 0; })) {
    // Legacy caller may only provide direction image; reconstruct per-label counts.
    contourCounts.assign(labelListSize, 0);
    for (const NppiContourPixelDirectionInfo &info : hostDirectionImage) {
      if (info.nMarkerLabelID <= nMaxMarkerLabelID && isContourDirectionPixel(info, info.nMarkerLabelID)) {
        contourCounts[info.nMarkerLabelID]++;
      }
    }
  }

  std::vector<Npp32u> contourOffsets;
  if (pContoursPixelsStartingOffsetHost != nullptr) {
    contourOffsets.assign(pContoursPixelsStartingOffsetHost, pContoursPixelsStartingOffsetHost + labelListSize);
  } else {
    contourOffsets = deriveContourList(pContoursPixelsStartingOffsetDev, labelListSize);
    if (std::all_of(contourOffsets.begin(), contourOffsets.end(), [](Npp32u value) { return value == 0; })) {
      // Legacy path fallback: build packed offsets from the reconstructed counts.
      contourOffsets.assign(labelListSize, 0);
      Npp32u runningOffset = 0;
      for (size_t label = 0; label < labelListSize; ++label) {
        contourOffsets[label] = runningOffset;
        runningOffset += contourCounts[label];
      }
    }
  }

  std::vector<Npp32u> pixelsFound(labelListSize, 0);
  std::vector<NppiContourPixelGeometryInfo> geometryLists(nTotalImagePixelContourCount);
  std::vector<NppiContourBlockSegment> blockSegments;

  if (pContoursGeometryImageHost != nullptr && nContoursGeometryImageStep > 0) {
    for (int y = 0; y < height; ++y) {
      std::memset(static_cast<char *>(static_cast<void *>(pContoursGeometryImageHost + y * nContoursGeometryImageStep)),
                  0, width * sizeof(Npp8u));
    }
  }

  for (Npp32u labelID = nFirstContourGeometryListID;
       labelID < nLastContourGeometryListID && labelID <= nMaxMarkerLabelID; ++labelID) {
    NppiCompressedMarkerLabelsInfo &info = pMarkerLabelsInfoListHost[labelID];
    if (info.nContourPixelCount > kMaxContourPixelsPerSegment &&
        nLastContourGeometryListID - nFirstContourGeometryListID > 1) {
      fillBlockSegments(blockSegments, labelID, contourCounts[labelID], contourOffsets[labelID]);
      continue;
    }

    std::vector<NppiPoint> orderedPoints =
        traceOrderedContourForLabel(hostDirectionImage, width, height, labelID, bOutputInCounterclockwiseOrder != 0);
    const Npp32u baseOffset = contourOffsets[labelID];
    const Npp32u writableCount =
        baseOffset < nTotalImagePixelContourCount
            ? std::min<Npp32u>(static_cast<Npp32u>(orderedPoints.size()), nTotalImagePixelContourCount - baseOffset)
            : 0;

    pixelsFound[labelID] = writableCount;
    info.nContourPixelsFound = writableCount;
    fillBlockSegments(blockSegments, labelID, contourCounts[labelID], baseOffset);

    for (Npp32u i = 0; i < writableCount; ++i) {
      const NppiPoint &point = orderedPoints[i];
      const NppiPoint &prev = orderedPoints[(i + writableCount - 1) % writableCount];
      const NppiPoint &next = orderedPoints[(i + 1) % writableCount];
      const size_t geomIndex = static_cast<size_t>(baseOffset + i);
      NppiContourPixelGeometryInfo &geom = geometryLists[geomIndex];
      geom = NppiContourPixelGeometryInfo{};
      geom.oContourOrderedGeometryLocation = point;
      geom.oContourCenterPixelLocation = point;
      geom.oContourPrevPixelLocation = prev;
      geom.oContourNextPixelLocation = next;
      geom.nOrderIndex = static_cast<Npp32s>(i);
      geom.nReverseOrderIndex = static_cast<Npp32s>(writableCount - 1 - i);
      geom.nFirstIndex = baseOffset;
      geom.nLastIndex = baseOffset + writableCount - 1;
      geom.nNextContourPixelIndex = baseOffset + ((i + 1) % writableCount);
      geom.nPrevContourPixelIndex = baseOffset + ((i + writableCount - 1) % writableCount);
      geom.nPixelAlreadyUsed = 1;
      geom.nAlreadyLinked = 0;
      geom.nAlreadyOutput = 1;

      const Npp8u interiorMask =
          hostDirectionImage[pixelOffset(point.x, point.y, width)].nContourInteriorDirectionCenterPixel;
      geom.nContourInteriorDirection = selectPrimaryDirection(interiorMask);

      if (i == 0) {
        geom.oContourPrevPixelLocation.x = info.oMarkerLabelBoundingBox.x;
        geom.oContourPrevPixelLocation.y = info.oMarkerLabelBoundingBox.y;
        geom.oContourNextPixelLocation.x = info.oMarkerLabelBoundingBox.width;
        geom.oContourNextPixelLocation.y = info.oMarkerLabelBoundingBox.height;
      }

      NppiContourPixelDirectionInfo &dirInfo = hostDirectionImage[pixelOffset(point.x, point.y, width)];
      dirInfo.nGeometryInfoIsValid = 1;
      dirInfo.oContourPixelGeometryInfo = geom;

      if (pContoursGeometryImageHost != nullptr && nContoursGeometryImageStep > 0) {
        reinterpret_cast<Npp8u *>(reinterpret_cast<char *>(pContoursGeometryImageHost) +
                                  static_cast<size_t>(point.y) * nContoursGeometryImageStep)[point.x] = 255;
      }
    }
  }

  if (pContoursPixelGeometryListsHost != nullptr && !geometryLists.empty()) {
    std::memcpy(pContoursPixelGeometryListsHost, geometryLists.data(),
                geometryLists.size() * sizeof(NppiContourPixelGeometryInfo));
  }
  if (copyHostArrayToDevice(pContoursPixelGeometryListsDev, geometryLists, geometryLists.size()) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  if (pContoursPixelsFoundListHost != nullptr) {
    std::memcpy(pContoursPixelsFoundListHost, pixelsFound.data(), pixelsFound.size() * sizeof(Npp32u));
  }
  if (copyHostArrayToDevice(pContoursPixelsFoundListDev, pixelsFound, pixelsFound.size()) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  if (pContoursPixelsStartingOffsetHost != nullptr) {
    std::memcpy(pContoursPixelsStartingOffsetHost, contourOffsets.data(), contourOffsets.size() * sizeof(Npp32u));
  }
  if (copyHostArrayToDevice(pContoursPixelsStartingOffsetDev, contourOffsets, contourOffsets.size()) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  if (pContoursPixelCountsListDev != nullptr &&
      copyHostArrayToDevice(pContoursPixelCountsListDev, contourCounts, contourCounts.size()) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  if (copyHostImageToDevice(pContoursDirectionImageDev, nContoursDirectionImageStep, hostDirectionImage, width,
                            height) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  if (copyHostArrayToDevice(pMarkerLabelsInfoListDev,
                            std::vector<NppiCompressedMarkerLabelsInfo>(pMarkerLabelsInfoListHost,
                                                                        pMarkerLabelsInfoListHost + labelListSize),
                            labelListSize) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  if (pContoursBlockSegmentListHost != nullptr && !blockSegments.empty()) {
    std::memcpy(pContoursBlockSegmentListHost, blockSegments.data(),
                blockSegments.size() * sizeof(NppiContourBlockSegment));
  }
  if (pContoursBlockSegmentListDev != nullptr && !blockSegments.empty() &&
      cudaMemcpy(pContoursBlockSegmentListDev, blockSegments.data(),
                 blockSegments.size() * sizeof(NppiContourBlockSegment), cudaMemcpyHostToDevice) != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  return NPP_SUCCESS;
}

// Batch compress marker labels (advanced)
NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx_impl(
    NppiImageDescriptor *pSrcDstBatchList, NppiBufferDescriptor *pBufferList, unsigned int *pNewMaxLabelIDList,
    int nBatchSize, NppiSize oMaxSizeROI, int nLargestPerImageBufferSize, NppStreamContext nppStreamCtx) {

  cudaStream_t stream = nppStreamCtx.hStream;
  constexpr int kCompressBufferFixedOverhead = 65556;
  constexpr int kCompressBufferBytesPerLabel = static_cast<int>(sizeof(Npp32u) * 2);

  if (pSrcDstBatchList == nullptr || pBufferList == nullptr || pNewMaxLabelIDList == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize <= 1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (oMaxSizeROI.width <= 0 || oMaxSizeROI.height <= 0 || nLargestPerImageBufferSize <= 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  std::vector<NppiImageDescriptor> h_batchImages(static_cast<size_t>(nBatchSize));
  std::vector<NppiBufferDescriptor> h_batchBuffers(static_cast<size_t>(nBatchSize));
  std::vector<unsigned int> h_newMaxLabelIDList(static_cast<size_t>(nBatchSize), 0);

  cudaError_t cudaStatus =
      cudaMemcpyAsync(h_batchImages.data(), pSrcDstBatchList, h_batchImages.size() * sizeof(NppiImageDescriptor),
                      cudaMemcpyDeviceToHost, stream);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }
  cudaStatus = cudaMemcpyAsync(h_batchBuffers.data(), pBufferList, h_batchBuffers.size() * sizeof(NppiBufferDescriptor),
                               cudaMemcpyDeviceToHost, stream);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }
  cudaStatus = cudaStreamSynchronize(stream);
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  for (int batch = 0; batch < nBatchSize; ++batch) {
    const NppiImageDescriptor &img = h_batchImages[batch];
    const NppiBufferDescriptor &buf = h_batchBuffers[batch];

    if (img.pData == nullptr || buf.pData == nullptr) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (img.oSize.width <= 0 || img.oSize.height <= 0) {
      return NPP_SIZE_ERROR;
    }
    if (img.oSize.width > oMaxSizeROI.width || img.oSize.height > oMaxSizeROI.height) {
      return NPP_SIZE_ERROR;
    }
    if (img.nStep <= 0) {
      return NPP_STEP_ERROR;
    }
    if (buf.nBufferSize < kCompressBufferFixedOverhead) {
      return NPP_BAD_ARGUMENT_ERROR;
    }

    // Batch API encodes the effective starting label range in per-image buffer size.
    const int nStartingNumber = (buf.nBufferSize - kCompressBufferFixedOverhead) / kCompressBufferBytesPerLabel;
    if (nStartingNumber <= 0) {
      return NPP_BAD_ARGUMENT_ERROR;
    }

    int newMarkerLabelsNumber = 0;
    NppStatus status = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(reinterpret_cast<Npp32u *>(img.pData), img.nStep,
                                                                    img.oSize, nStartingNumber, &newMarkerLabelsNumber,
                                                                    reinterpret_cast<Npp8u *>(buf.pData), nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }

    h_newMaxLabelIDList[batch] = static_cast<unsigned int>(std::max(newMarkerLabelsNumber - 1, 0));
  }

  cudaStatus = cudaMemcpyAsync(pNewMaxLabelIDList, h_newMaxLabelIDList.data(),
                               h_newMaxLabelIDList.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMCPY_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
