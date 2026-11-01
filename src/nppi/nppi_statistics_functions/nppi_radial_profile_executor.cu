// Circular radial profile executor (see the header for the decoded NPP 13.4
// sampling rule).
#include "nppi_radial_profile_executor.h"

#include <cmath>
#include <cuda_runtime.h>

namespace nppi {
namespace statistics {
namespace {

constexpr int kBlock = 128;

__device__ __forceinline__ bool inRoi(int x, int y, int w, int h) { return x >= 0 && y >= 0 && x < w && y < h; }

// Collects the pixels of one ring.  Each block owns one ring; the threads split
// the ROI rows and the block reduces (count, sum, sum of squares).
__global__ void radialProfileKernel(const Npp32f *pSrc, int nSrcStep, int width, int height, float cx, float cy,
                                    int samples, NppiProfileData *pProfile) {
  __shared__ double sCount[kBlock];
  __shared__ double sSum[kBlock];
  __shared__ double sSq[kBlock];

  const int ring = blockIdx.x;
  double sum = 0.0, sq = 0.0, count = 0.0;
  const int stride = (int)(nSrcStep / sizeof(Npp32f));

  const bool finite = isfinite(cx) && isfinite(cy);
  if (ring == 0) {
    if (threadIdx.x == 0 && finite) {
      // Ring 0: every pixel whose centre is closer than one pixel to oCenter.
      const int x0 = max((int)floorf(cx) - 1, 0);
      const int y0 = max((int)floorf(cy) - 1, 0);
      const int x1 = min((int)ceilf(cx) + 1, width - 1);
      const int y1 = min((int)ceilf(cy) + 1, height - 1);
      for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
          const double dx = (double)x - (double)cx;
          const double dy = (double)y - (double)cy;
          if (dx * dx + dy * dy >= 1.0) continue;
          const double v = (double)pSrc[(size_t)y * stride + x];
          sum += v;
          sq += v * v;
          count += 1.0;
        }
      }
    }
  } else if (finite) {
    const double k = (double)ring;
    // Only ROI rows can contribute: clamp the row window to [0, height).
    int ylo = (int)ceil((double)cy - k);
    int yhi = (int)floor((double)cy + k);
    ylo = max(ylo, 0);
    yhi = min(yhi, height - 1);
    for (int y = ylo + threadIdx.x; y <= yhi; y += blockDim.x) {
      const double dy = (double)y - (double)cy;
      if (fabs(dy) > k) continue;
      const double v = sqrt(fmax(0.0, k * k - dy * dy));
      const int xl = (int)floor((double)cx - v);
      const int xr = (int)ceil((double)cx + v);
      if (inRoi(xl, y, width, height)) {
        const double p = (double)pSrc[(size_t)y * stride + xl];
        sum += p;
        sq += p * p;
        count += 1.0;
      }
      if (xr != xl && inRoi(xr, y, width, height)) {
        const double p = (double)pSrc[(size_t)y * stride + xr];
        sum += p;
        sq += p * p;
        count += 1.0;
      }
    }
  }

  sCount[threadIdx.x] = count;
  sSum[threadIdx.x] = sum;
  sSq[threadIdx.x] = sq;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sCount[threadIdx.x] += sCount[threadIdx.x + s];
      sSum[threadIdx.x] += sSum[threadIdx.x + s];
      sSq[threadIdx.x] += sSq[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    NppiProfileData out;
    out.nPixels = (int)sCount[0];
    if (sCount[0] > 0.0) {
      const double mean = sSum[0] / sCount[0];
      const double var = sSq[0] / sCount[0] - mean * mean;
      out.nMeanIntensity = (Npp32f)mean;
      out.nStdDevIntensity = (Npp32f)sqrt(var > 0.0 ? var : 0.0);
    } else {
      out.nMeanIntensity = 0.0f;
      out.nStdDevIntensity = 0.0f;
    }
    pProfile[ring] = out;
  }
}

}  // namespace

NppStatus runCircularRadialProfile(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, NppiPoint32f oCenter,
                                   NppiProfileData *pProfileData, Npp16u nProfileSamples, cudaStream_t stream) {
  if (!pSrc || !pProfileData) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (nProfileSamples == 0) return NPP_NO_ERROR;

  radialProfileKernel<<<nProfileSamples, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height,
                                                              oCenter.x, oCenter.y, (int)nProfileSamples,
                                                              pProfileData);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace statistics
}  // namespace nppi
