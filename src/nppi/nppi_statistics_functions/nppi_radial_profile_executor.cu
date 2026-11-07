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
      out.nStdDevIntensity = (Npp32f)sqrt(var > 0.0 ? var: 0.0);
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

  // NPP returns the profile through HOST memory; stage it on the device and
  // copy back before returning.
  NppiProfileData *dProfile = nullptr;
  if (cudaMalloc(&dProfile, sizeof(NppiProfileData) * (size_t)nProfileSamples) != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  radialProfileKernel<<<nProfileSamples, kBlock, 0, stream>>>(pSrc, nSrcStep, oSizeROI.width, oSizeROI.height,
                                                              oCenter.x, oCenter.y, (int)nProfileSamples,
                                                              dProfile);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpy(pProfileData, dProfile, sizeof(NppiProfileData) * (size_t)nProfileSamples,
                     cudaMemcpyDeviceToHost);
  }
  cudaFree(dProfile);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

namespace {

// Reproduces the NPP 13.4 elliptical kernel (decoded from the sm_120 SASS of
// executeComputeProfileEllipticalIfE... in libnppist.so.13).  The membership
// test is a dense band classification, ring k = k^2 <= t^2 < (k+1)^2 with
//
//   t^2 = (dx*f0 - dy*f1)^2 + (dx*f1 + dy*f0)^2 / ratio^2,
//
// where the host passes f0 = cos(orientation) and f1 = -sin(orientation).
// The reference scans only a cropped window per ring: columns
// [floor(cx - k*ratio), ceil(cx + k*ratio)] (the inner radius, not the outer
// one) and rows from floor(cy - k*ratio) for ceil(2*(k+1)*ratio) steps.  The
// crop is why the central columns above the centre are never sampled for some
// rings (24 pixels for 33x33, ratio 2, 24 samples).  All geometry is float32
// with the reference's FMA order; the horizontal inner bound is what makes
// the impulse map match exactly.
__global__ void ellipticalProfileKernel(const Npp32f *pSrc, int nSrcStep, int width, int height, float cx,
                                        float cy, float ratio, float f0, float f1,
                                        NppiProfileData *pProfile) {
  __shared__ double sCount[kBlock];
  __shared__ double sSum[kBlock];
  __shared__ double sSq[kBlock];

  const int ring = blockIdx.x;
  const float kf = (float)ring;
  const float k1f = (float)(ring + 1);
  const float k2 = __fmul_rn(kf, kf);
  const float k12 = __fmul_rn(k1f, k1f);
  const float invRatio2 = __frcp_rn(__fmul_rn(ratio, ratio));
  const int stride = (int)(nSrcStep / sizeof(Npp32f));

  const float xloF = __fmaf_rn(-kf, ratio, cx);
  const float xhiF = __fmaf_rn(kf, ratio, cx);
  const float ystartF = __fmaf_rn(-kf, ratio, cy);
  const float yspanF = __fmul_rn((float)(2 * (ring + 1)), ratio);
  int xlo = max((int)floorf(xloF), 0);
  int xhi = min((int)ceilf(xhiF), width - 1);
  const long long ystart = (long long)(int)floorf(ystartF);
  const long long yspan = (long long)(int)ceilf(yspanF);
  // The reference's row loop skips rows outside the ROI; intersect up front so
  // a pathological ratio cannot spin for 2^31 iterations.
  const long long ylo = max(0LL, ystart);
  const long long yhi = min((long long)height - 1, ystart + yspan);

  double sum = 0.0, sq = 0.0, count = 0.0;
  if (xlo <= xhi && ylo <= yhi) {
    const long long rows = yhi - ylo + 1;
    for (long long t = threadIdx.x; t < rows; t += blockDim.x) {
      const int y = (int)(ylo + t);
      const float b = __fadd_rn(cy, -(float)y);
      const float bf0 = __fmul_rn(b, f0);
      const float bf1 = __fmul_rn(b, f1);
      const Npp32f *row = pSrc + (size_t)y * stride;
      for (int x = xlo; x <= xhi; ++x) {
        const float a = __fadd_rn(cx, -(float)x);
        const float u = __fmaf_rn(a, f1, bf0);
        const float v = __fmaf_rn(a, f0, -bf1);
        const float t2 = __fmaf_rn(v, v, __fmul_rn(__fmul_rn(u, u), invRatio2));
        if (t2 >= k2 && t2 < k12) {
          const double p = (double)row[x];
          sum += p;
          sq += p * p;
          count += 1.0;
        }
      }
    }
  }

  sCount[threadIdx.x] = count;
  sSum[threadIdx.x] = sum;
  sSq[threadIdx.x] = sq;
  __syncthreads();
  for (unsigned step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) {
      sCount[threadIdx.x] += sCount[threadIdx.x + step];
      sSum[threadIdx.x] += sSum[threadIdx.x + step];
      sSq[threadIdx.x] += sSq[threadIdx.x + step];
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
      out.nStdDevIntensity = (Npp32f)sqrt(var > 0.0 ? var: 0.0);
    } else {
      out.nMeanIntensity = 0.0f;
      out.nStdDevIntensity = 0.0f;
    }
    pProfile[ring] = out;
  }
}

}  // namespace

NppStatus runEllipticalRadialProfile(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI,
                                     NppiPoint32f oCenter, Npp32f nAstigmatismRatio,
                                     Npp32f nOrientationRadians, NppiProfileData *pProfileData,
                                     Npp16u nProfileSamples, cudaStream_t stream) {
  if (pSrc == nullptr || pProfileData == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  // Measured order on the reference: null pointers, then negative ROI, then the
  // ratio range, and only then the empty-ROI/empty-samples no-ops.  The check
  // is a plain `ratio < 1`, so NaN and +inf are accepted; -inf is rejected.
  if (nAstigmatismRatio < 1.0f) {
    return NPP_RANGE_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0 || nProfileSamples == 0) {
    return NPP_NO_ERROR;
  }
  // The reference passes cos(orientation) and -sin(orientation) to the kernel,
  // so the trig is evaluated on the host as well.
  const float f0 = cosf(nOrientationRadians);
  const float f1 = -sinf(nOrientationRadians);
  // Host-memory result, staged through the device like the reference.
  NppiProfileData *dProfile = nullptr;
  if (cudaMalloc(&dProfile, sizeof(NppiProfileData) * (size_t)nProfileSamples) != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  ellipticalProfileKernel<<<nProfileSamples, kBlock, 0, stream>>>(
      pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, oCenter.x, oCenter.y, nAstigmatismRatio, f0, f1,
      dProfile);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpy(pProfileData, dProfile, sizeof(NppiProfileData) * (size_t)nProfileSamples,
                     cudaMemcpyDeviceToHost);
  }
  cudaFree(dProfile);
  return err == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace statistics
}  // namespace nppi
