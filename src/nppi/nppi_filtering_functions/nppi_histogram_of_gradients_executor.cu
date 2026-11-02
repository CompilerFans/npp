// HOG (HistogramOfGradientsBorder) executor.
//
// Semantics measured on NVIDIA NPP 13.4 / RTX 5090 (probes hog1..hog5):
//   * gradients are the centered [-1,0,+1] differences with REPLICATE
//     clamping at the SOURCE IMAGE bounds (not at the ROI);
//   * the unsigned gradient orientation is atan2(gy,gx) in degrees, negative
//     angles get +180; 180 degrees is left as-is and its bin index (== bins)
//     is dropped;
//   * each pixel's magnitude is added to the two nearest angular bins with
//     linear interpolation; the cell histogram is a float sum over the cell;
//   * every 2x2-cell block is normalized by its largest bin;
//   * the descriptor layout is [blockY][cellY][blockX][cellX][bin] (the
//     blockX axis is inside the cell axes), verified against ramp border
//     patterns and single-delta cell appearances;
//   * C3 sources use the per-pixel channel with the largest gradient
//     magnitude (ties are resolved by the first channel).
//
// The exact inlined atan approximation NPP uses is not reproduced bit-for-bit;
// for axis-aligned and 45-degree gradients (the differential test images) the
// results agree to float rounding.
#include "npp.h"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

template <typename Src, int Channels>
__device__ __forceinline__ float sampleChannel(const Src *pSrc, int nSrcStep, int x, int y, int channel,
                                               int loX, int hiX, int loY, int hiY) {
  x = x < loX ? loX : (x > hiX ? hiX : x);
  y = y < loY ? loY : (y > hiY ? hiY : y);
  const Src *row = reinterpret_cast<const Src *>(reinterpret_cast<const char *>(pSrc) +
                                                 static_cast<long long>(y) * nSrcStep);
  return static_cast<float>(row[static_cast<long long>(x) * Channels + channel]);
}

template <typename Src, int Channels>
__global__ void hogKernel(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          const NppiPoint *hpLocations, Npp32f *pDstWindowDescriptorBuffer,
                          NppiHOGConfig oHOGConfig) {
  extern __shared__ float shared[];
  const int windowW = oHOGConfig.detectionWindowSize.width;
  const int windowH = oHOGConfig.detectionWindowSize.height;
  const int cellSize = oHOGConfig.cellSize;
  const int blockSize = oHOGConfig.histogramBlockSize;
  const int bins = oHOGConfig.nHistogramBins;
  const int cellsX = windowW / cellSize;
  const int cellsY = windowH / cellSize;
  const int blocksX = 2 * (windowW / blockSize) - 1;
  const int blocksY = 2 * (windowH / blockSize) - 1;
  const int cellsPerBlock = blockSize / cellSize;
  const int histCount = cellsX * cellsY * bins;
  float *hist = shared;
  float *blockMax = shared + histCount;

  for (int i = threadIdx.x; i < histCount; i += blockDim.x) {
    hist[i] = 0.0f;
  }
  __syncthreads();

  const NppiPoint location = hpLocations[blockIdx.x];
  const int loX = -oSrcOffset.x;
  const int hiX = oSrcSize.width - 1 - oSrcOffset.x;
  const int loY = -oSrcOffset.y;
  const int hiY = oSrcSize.height - 1 - oSrcOffset.y;
  // One thread per cell; accumulate the cell histogram locally (no shared
  // atomics, which the MACA runtime does not support for float).
  const int cellCount = cellsX * cellsY;
  for (int cell = threadIdx.x; cell < cellCount; cell += blockDim.x) {
    const int cellX = cell % cellsX;
    const int cellY = cell / cellsX;
    float local[16];
    for (int bin = 0; bin < bins; ++bin) {
      local[bin] = 0.0f;
    }
    for (int y = cellY * cellSize; y < (cellY + 1) * cellSize; ++y) {
      for (int x = cellX * cellSize; x < (cellX + 1) * cellSize; ++x) {
        const int srcX = location.x + x;
        const int srcY = location.y + y;
        float bestMagnitude = -1.0f;
        float bestAngle = 0.0f;
        for (int channel = 0; channel < Channels; ++channel) {
          const float gx =
              sampleChannel<Src, Channels>(pSrc, nSrcStep, srcX + 1, srcY, channel, loX, hiX, loY, hiY) -
              sampleChannel<Src, Channels>(pSrc, nSrcStep, srcX - 1, srcY, channel, loX, hiX, loY, hiY);
          const float gy =
              sampleChannel<Src, Channels>(pSrc, nSrcStep, srcX, srcY + 1, channel, loX, hiX, loY, hiY) -
              sampleChannel<Src, Channels>(pSrc, nSrcStep, srcX, srcY - 1, channel, loX, hiX, loY, hiY);
          const float magnitude = sqrtf(gx * gx + gy * gy);
          if (magnitude > bestMagnitude) {
            bestMagnitude = magnitude;
            float angle = atan2f(gy, gx) * 57.2957763671875f;
            if (angle < 0.0f) {
              angle += 180.0f;
            }
            bestAngle = angle;
          }
        }
        if (bestMagnitude <= 0.0f) {
          continue;
        }
        const float position = bestAngle * (static_cast<float>(bins) / 180.0f);
        const int bin0 = static_cast<int>(floorf(position));
        if (bin0 >= bins) {
          continue;  // exactly 180 degrees: no valid bin
        }
        const float fraction = position - static_cast<float>(bin0);
        local[bin0] += bestMagnitude * (1.0f - fraction);
        if (bin0 + 1 < bins) {
          local[bin0 + 1] += bestMagnitude * fraction;
        }
      }
    }
    const int base = (cellY * cellsX + cellX) * bins;
    for (int bin = 0; bin < bins; ++bin) {
      hist[base + bin] = local[bin];
    }
  }
  __syncthreads();

  for (int b = threadIdx.x; b < blocksX * blocksY; b += blockDim.x) {
    const int by = b / blocksX;
    const int bx = b % blocksX;
    float maximum = 0.0f;
    for (int cy = 0; cy < cellsPerBlock; ++cy) {
      for (int cx = 0; cx < cellsPerBlock; ++cx) {
        const int base = ((by + cy) * cellsX + (bx + cx)) * bins;
        for (int bin = 0; bin < bins; ++bin) {
          maximum = fmaxf(maximum, hist[base + bin]);
        }
      }
    }
    blockMax[b] = maximum > 0.0f ? maximum : 1.0f;
  }
  __syncthreads();

  const int descriptorCount = blocksX * blocksY * cellsPerBlock * cellsPerBlock * bins;
  Npp32f *out = pDstWindowDescriptorBuffer + static_cast<size_t>(blockIdx.x) * descriptorCount;
  for (int i = threadIdx.x; i < descriptorCount; i += blockDim.x) {
    int rest = i;
    const int bin = rest % bins;
    rest /= bins;
    const int cx = rest % cellsPerBlock;
    rest /= cellsPerBlock;
    const int bx = rest % blocksX;
    rest /= blocksX;
    const int cy = rest % cellsPerBlock;
    const int by = rest / cellsPerBlock;
    const int cellIndex = (by + cy) * cellsX + (bx + cx);
    out[i] = hist[cellIndex * bins + bin] / blockMax[by * blocksX + bx];
  }
}

inline bool hogConfigInvalid(const NppiHOGConfig &config) {
  if (config.detectionWindowSize.width <= 0 || config.detectionWindowSize.height <= 0) {
    return true;
  }
  if (config.cellSize < 2 || config.cellSize > 16) {
    return true;
  }
  if (config.histogramBlockSize > 64 || config.cellSize > config.histogramBlockSize) {
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
  const int cellsPerDescriptor = (config.detectionWindowSize.width / config.cellSize) *
                                 (config.detectionWindowSize.height / config.cellSize);
  return cellsPerDescriptor > 256;
}

template <typename Src, int Channels>
NppStatus hogLaunch(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                    const NppiPoint *hpLocations, int nLocations, Npp32f *pDstWindowDescriptorBuffer,
                    NppiSize oSizeROI, NppiHOGConfig oHOGConfig, Npp8u *pScratchBuffer,
                    NppiBorderType eBorderType, cudaStream_t stream) {
  if (pSrc == nullptr || hpLocations == nullptr || pDstWindowDescriptorBuffer == nullptr ||
      pScratchBuffer == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || oSrcSize.width <= 0 || oSrcSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (hogConfigInvalid(oHOGConfig)) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  if (eBorderType != NPP_BORDER_REPLICATE) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  if (nLocations < 1 || nLocations > 1024) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (nSrcStep % 4 != 0) {
    return NPP_NOT_EVEN_STEP_ERROR;
  }
  if (nSrcStep < 64) {
    // Measured: the minimum HOG source step is 64 bytes and does not scale
    // with the number of locations.
    return NPP_STEP_ERROR;
  }
  const int windowW = oHOGConfig.detectionWindowSize.width;
  const int windowH = oHOGConfig.detectionWindowSize.height;
  for (int i = 0; i < nLocations; ++i) {
    const NppiPoint location = hpLocations[i];
    if (location.x < 0 || location.y < 0 || location.x + windowW > oSizeROI.width ||
        location.y + windowH > oSizeROI.height) {
      return NPP_OUT_OFF_RANGE_ERROR;
    }
  }

  const int cellsX = windowW / oHOGConfig.cellSize;
  const int cellsY = windowH / oHOGConfig.cellSize;
  const int blocksX = 2 * (windowW / oHOGConfig.histogramBlockSize) - 1;
  const int blocksY = 2 * (windowH / oHOGConfig.histogramBlockSize) - 1;
  const int histCount = cellsX * cellsY * oHOGConfig.nHistogramBins;
  const size_t sharedBytes = static_cast<size_t>(histCount + blocksX * blocksY) * sizeof(float);
  if (sharedBytes > 48 * 1024) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
  NppiPoint *dLocations = nullptr;
  if (cudaMalloc(&dLocations, static_cast<size_t>(nLocations) * sizeof(NppiPoint)) != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  if (cudaMemcpy(dLocations, hpLocations, static_cast<size_t>(nLocations) * sizeof(NppiPoint),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dLocations);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  const dim3 block(256);
  const dim3 grid(nLocations);
  hogKernel<Src, Channels><<<grid, block, sharedBytes, stream>>>(
      pSrc, nSrcStep, oSrcSize, oSrcOffset, dLocations, pDstWindowDescriptorBuffer, oHOGConfig);
  const cudaError_t error = cudaGetLastError();
  cudaFree(dLocations);
  return error == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

#define DEFINE_HOG(SRC, TOK, CHANNELS, SUFFIX)                                                           \
  NppStatus nppiHistogramOfGradientsBorder_##TOK##SUFFIX##_Ctx(                                          \
      const SRC *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,                             \
      const NppiPoint *hpLocations, int nLocations, Npp32f *pDstWindowDescriptorBuffer,                   \
      NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u *pScratchBuffer,                           \
      NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {                                        \
    return hogLaunch<SRC, CHANNELS>(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations,        \
                                    pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer,     \
                                    eBorderType, nppStreamCtx.hStream);                                   \
  }                                                                                                       \
  NppStatus nppiHistogramOfGradientsBorder_##TOK##SUFFIX(                                                 \
      const SRC *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,                             \
      const NppiPoint *hpLocations, int nLocations, Npp32f *pDstWindowDescriptorBuffer,                   \
      NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u *pScratchBuffer,                           \
      NppiBorderType eBorderType) {                                                                       \
    NppStreamContext ctx{};                                                                               \
    ctx.hStream = nppGetStream();                                                                         \
    return nppiHistogramOfGradientsBorder_##TOK##SUFFIX##_Ctx(                                            \
        pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer,         \
        oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, ctx);                                           \
  }

DEFINE_HOG(Npp8u, 8u32f, 1, _C1R)
DEFINE_HOG(Npp8u, 8u32f, 3, _C3R)
DEFINE_HOG(Npp16u, 16u32f, 1, _C1R)
DEFINE_HOG(Npp16u, 16u32f, 3, _C3R)
DEFINE_HOG(Npp16s, 16s32f, 1, _C1R)
DEFINE_HOG(Npp16s, 16s32f, 3, _C3R)
DEFINE_HOG(Npp32f, 32f, 1, _C1R)
DEFINE_HOG(Npp32f, 32f, 3, _C3R)

// Literal forwarding *_impl helpers (the public wrappers are macro-generated,
// so the API catalog needs these to mark the entry points real).

NppStatus nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx_impl(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx_impl(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx_impl(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_16u32f_C3R_Ctx_impl(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_16u32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx_impl(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_16s32f_C3R_Ctx_impl(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_16s32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_32f_C1R_Ctx_impl(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

NppStatus nppiHistogramOfGradientsBorder_32f_C3R_Ctx_impl(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
  return nppiHistogramOfGradientsBorder_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, hpLocations, nLocations, pDstWindowDescriptorBuffer, oSizeROI, oHOGConfig, pScratchBuffer, eBorderType, nppStreamCtx);
}

