// SSIM / WMSSSIM executor implementation (see the header for the decoded
// NVIDIA NPP 13.4 pipeline).
#include "nppi_ssim_executor.h"

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace nppi {
namespace statistics {
namespace {

constexpr int kBlock = 256;
constexpr float kC1 = 6.5025f;
constexpr float kC2 = 58.5225f;
constexpr float kWeights[5] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

struct Gauss {
  float w[11];
};

Gauss gauss_weights() {
  Gauss g{};
  double s = 0.0;
  for (int i = 0; i < 11; ++i) {
    const double x = i - 5;
    g.w[i] = (float)std::exp(-(x * x) / (2.0 * 1.5 * 1.5));
    s += g.w[i];
  }
  for (int i = 0; i < 11; ++i) g.w[i] = (float)(g.w[i] / s);
  return g;
}

__device__ double blockReduceDouble(double val) {
  __shared__ double shared[kBlock];
  shared[threadIdx.x] = val;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  return shared[0];
}

// Horizontal then vertical clamped Gaussian pass (in place through tmp).
__global__ void gaussianH(const float *in, float *out, int w, int h, Gauss g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  double acc = 0.0;
  for (int k = -5; k <= 5; ++k) {
    int xx = x + k;
    xx = xx < 0 ? 0: (xx >= w ? w - 1: xx);
    acc += (double)g.w[k + 5] * in[(size_t)y * w + xx];
  }
  out[(size_t)y * w + x] = (float)acc;
}

__global__ void gaussianV(const float *in, float *out, int w, int h, Gauss g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  double acc = 0.0;
  for (int k = -5; k <= 5; ++k) {
    int yy = y + k;
    yy = yy < 0 ? 0: (yy >= h ? h - 1: yy);
    acc += (double)g.w[k + 5] * in[(size_t)yy * w + x];
  }
  out[(size_t)y * w + x] = (float)acc;
}

void conv2d(const float *in, float *tmp, float *out, int w, int h, Gauss g, cudaStream_t s) {
  dim3 block(32, 8);
  dim3 grid((w + 31) / 32, (h + 7) / 8);
  gaussianH<<<grid, block, 0, s>>>(in, tmp, w, h, g);
  gaussianV<<<grid, block, 0, s>>>(tmp, out, w, h, g);
}

__global__ void extractPlane(const Npp8u *src, int step, int w, int h, int channels, int ch, float *out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const Npp8u *row = src + (size_t)y * step;
  out[(size_t)y * w + x] = (float)row[(size_t)x * channels + ch];
}

__global__ void squarePlane(const float *in, float *out, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  out[(size_t)y * w + x] = in[(size_t)y * w + x] * in[(size_t)y * w + x];
}

__global__ void productPlane(const float *a, const float *b, float *out, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  out[(size_t)y * w + x] = a[(size_t)y * w + x] * b[(size_t)y * w + x];
}

// SSIM / WMSSSIM per-level accumulation.
//  mode 0: sum of l*cs over all pixels (plain SSIM)
//  mode 1: sum of cs^w (levels 0..3 of WMSSSIM)
//  mode 2: sum of (l*cs)^w (level 4 of WMSSSIM)
__global__ void ssimAccum(const float *mu_a, const float *mu_b, const float *ea2, const float *eb2,
                          const float *eab, int w, int h, int mode, float weight, double *out) {
  double acc = 0.0;
  const long total = (long)w * h;
  for (long i = threadIdx.x; i < total; i += blockDim.x) {
    const float mua = mu_a[i];
    const float mub = mu_b[i];
    const float va = ea2[i] - mua * mua;
    const float vb = eb2[i] - mub * mub;
    const float cov = eab[i] - mua * mub;
    const float cs = (2.0f * fabsf(cov) + kC2) / (va + vb + kC2);
    const float l = (2.0f * mua * mub + kC1) / (mua * mua + mub * mub + kC1);
    float v;
    if (mode == 0) {
      v = l * cs;
    } else if (mode == 1) {
      v = powf(cs, weight);
    } else {
      v = powf(l * cs, weight);
    }
    acc += (double)v;
  }
  acc = blockReduceDouble(acc);
  if (threadIdx.x == 0) *out = acc;
}

__global__ void decimatePlane(const float *in, float *out, int w, int h, int ow, int oh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ow || y >= oh) return;
  const int y0 = y > 0 ? 2 * y - 1: 0;
  const int y1 = 2 * y < h ? 2 * y: h - 1;
  const int x0 = x > 0 ? 2 * x - 1: 0;
  const int x1 = 2 * x < w ? 2 * x: w - 1;
  out[(size_t)y * ow + x] =
      0.25f * (in[(size_t)y0 * w + x0] + in[(size_t)y0 * w + x1] + in[(size_t)y1 * w + x0] + in[(size_t)y1 * w + x1]);
}

// One level of the SSIM pipeline for a single channel.
// Returns the level accumulation in *hsum (host).
NppStatus ssimLevel(const float *a, const float *b, int w, int h, float *buf, int mode, float weight, double *hsum,
                    cudaStream_t stream) {
  const size_t plane = (size_t)w * h;
  float *mu_a = buf;
  float *mu_b = mu_a + plane;
  float *ea2 = mu_b + plane;
  float *eb2 = ea2 + plane;
  float *eab = eb2 + plane;
  float *tmp = eab + plane;
  float *sq = tmp + plane;
  float *pr = sq + plane;
  const Gauss g = gauss_weights();
  conv2d(a, tmp, mu_a, w, h, g, stream);
  conv2d(b, tmp, mu_b, w, h, g, stream);
  squarePlane<<<dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), 0, stream>>>(a, sq, w, h);
  conv2d(sq, tmp, ea2, w, h, g, stream);
  squarePlane<<<dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), 0, stream>>>(b, sq, w, h);
  conv2d(sq, tmp, eb2, w, h, g, stream);
  productPlane<<<dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), 0, stream>>>(a, b, pr, w, h);
  conv2d(pr, tmp, eab, w, h, g, stream);
  double *dacc = nullptr;
  if (cudaMalloc(&dacc, sizeof(double)) != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;
  ssimAccum<<<1, kBlock, 0, stream>>>(mu_a, mu_b, ea2, eb2, eab, w, h, mode, weight, dacc);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaMemcpyAsync(hsum, dacc, sizeof(double), cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  cudaFree(dacc);
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

constexpr int kPlanes = 12;  // a,b + 8 working planes + 2 decimated planes

NppStatus ssimChannel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize roi, int channels,
                      int ch, Npp32f *out, cudaStream_t stream) {
  const int w = roi.width, h = roi.height;
  const size_t plane = (size_t)w * h;
  float *buf = nullptr;
  if (cudaMalloc(&buf, sizeof(float) * plane * kPlanes) != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;
  float *a = buf + 0 * plane;
  float *b = buf + 1 * plane;
  dim3 block(32, 8), grid((w + 31) / 32, (h + 7) / 8);
  extractPlane<<<grid, block, 0, stream>>>(pSrc1, nSrc1Step, w, h, channels, ch, a);
  extractPlane<<<grid, block, 0, stream>>>(pSrc2, nSrc2Step, w, h, channels, ch, b);
  double hsum = 0.0;
  NppStatus st = ssimLevel(a, b, w, h, buf + 2 * plane, 0, 1.0f, &hsum, stream);
  cudaFree(buf);
  if (st != NPP_NO_ERROR) return st;
  const Npp32f value = (Npp32f)(hsum / (double)(w * h));
  return cudaMemcpyAsync(out + ch, &value, sizeof(value), cudaMemcpyHostToDevice, stream) == cudaSuccess
             ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus wmsssimChannel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize roi,
                         int channels, int ch, Npp32f *out, cudaStream_t stream) {
  const int w0 = roi.width, h0 = roi.height;
  float *buf = nullptr;
  if (cudaMalloc(&buf, sizeof(float) * (size_t)w0 * h0 * kPlanes) != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;
  float *a = buf + 0 * ((size_t)w0 * h0);
  float *b = buf + 1 * ((size_t)w0 * h0);
  const size_t plane0 = (size_t)w0 * h0;
  // Level buffers live after a/b; make sure the working set fits.
  float *work = buf + 2 * plane0;
  dim3 block(32, 8), grid((w0 + 31) / 32, (h0 + 7) / 8);
  extractPlane<<<grid, block, 0, stream>>>(pSrc1, nSrc1Step, w0, h0, channels, ch, a);
  extractPlane<<<grid, block, 0, stream>>>(pSrc2, nSrc2Step, w0, h0, channels, ch, b);
  int w = w0, h = h0;
  double result = 1.0;
  for (int level = 0; level < 5; ++level) {
    double hsum = 0.0;
    const int mode = level < 4 ? 1: 2;
    NppStatus st = ssimLevel(a, b, w, h, work, mode, kWeights[level], &hsum, stream);
    if (st != NPP_NO_ERROR) {
      cudaFree(buf);
      return st;
    }
    result *= hsum / (double)(w * h);
    if (level < 4) {
      const int nw = (w + 1) / 2, nh = (h + 1) / 2;
      float *na = work + (size_t)w * h * 8;
      float *nb = na + (size_t)nw * nh;
      dim3 g((nw + 31) / 32, (nh + 7) / 8);
      decimatePlane<<<g, block, 0, stream>>>(a, na, w, h, nw, nh);
      decimatePlane<<<g, block, 0, stream>>>(b, nb, w, h, nw, nh);
      const float *olda = a, *oldb = b;
      a = na;
      b = nb;
      (void)olda;
      (void)oldb;
      w = nw;
      h = nh;
    }
  }
  cudaFree(buf);
  const Npp32f value = (Npp32f)result;
  return cudaMemcpyAsync(out + ch, &value, sizeof(value), cudaMemcpyHostToDevice, stream) == cudaSuccess
             ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus copyBatchDesc(const NppiImageDescriptor *p, int n, std::vector<NppiImageDescriptor> &out,
                        cudaStream_t stream) {
  out.resize(n);
  cudaError_t err = cudaMemcpyAsync(out.data(), p, sizeof(NppiImageDescriptor) * (size_t)n, cudaMemcpyDeviceToHost,
                                    stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  return err == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

// The size helpers report a conservative requirement (all working maps for the
// largest level).  NPP's exact SSIM-family sizes are undocumented internals;
// the reference comparisons pin the status contract and sufficiency.
NppStatus ssimGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  // Measured: a negative ROI is accepted (NO_ERROR, no size written).
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_NO_ERROR;
  const long planes = (long)oSizeROI.width * oSizeROI.height;
  *hpBufferSize = (int)(channels * kPlanes * (planes + 2L * oSizeROI.width) * 4 + 256);
  return NPP_NO_ERROR;
}

NppStatus ssimBatchGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  return ssimGetBufferHostSize(oSizeROI, channels, hpBufferSize);
}

NppStatus wmsssimGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  // Measured: a negative ROI is accepted (NO_ERROR, no size written).
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_NO_ERROR;
  const long planes = (long)oSizeROI.width * oSizeROI.height;
  *hpBufferSize = (NPP_BUFFER_SIZE_TYPE)(channels * (kPlanes + 2) * (planes + 2L * oSizeROI.width) * 4 + 256);
  return NPP_NO_ERROR;
}

NppStatus wmsssimBatchGetBufferHostSize(NppiSize oSizeROI, int channels, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  return wmsssimGetBufferHostSize(oSizeROI, channels, hpBufferSize);
}

NppStatus msssimGetBufferHostSize(NppiSize oSizeROI, NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  return wmsssimGetBufferHostSize(oSizeROI, 1, hpBufferSize);
}

NppStatus runSSIM(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                  int channels, Npp32f *pSSIM, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pSSIM || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (nSrc1Step < oSizeROI.width * channels || nSrc2Step < oSizeROI.width * channels) return NPP_STEP_ERROR;
  for (int c = 0; c < channels; ++c) {
    NppStatus st = ssimChannel(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, channels, c, pSSIM, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_NO_ERROR;
}

NppStatus runSSIMBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                       NppiSize oSizeROI, int channels, Npp32f *pSSIM, bool advanced, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pSSIM) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize <= 0) return NPP_BAD_ARGUMENT_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  std::vector<NppiImageDescriptor> h1, h2;
  NppStatus st = copyBatchDesc(pSrc1, nBatchSize, h1, stream);
  if (st != NPP_NO_ERROR) return st;
  st = copyBatchDesc(pSrc2, nBatchSize, h2, stream);
  if (st != NPP_NO_ERROR) return st;
  for (int i = 0; i < nBatchSize; ++i) {
    const NppiSize roi = advanced ? h1[i].oSize: oSizeROI;
    if (roi.width <= 0 || roi.height <= 0) return NPP_SIZE_ERROR;
    st = runSSIM((const Npp8u *)h1[i].pData, h1[i].nStep, (const Npp8u *)h2[i].pData, h2[i].nStep, roi, channels,
                 pSSIM + (size_t)i * channels, (Npp8u *)pSSIM, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_NO_ERROR;
}

NppStatus runWMSSSIM(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, NppiSize oSizeROI,
                     int channels, Npp32f *pWMSSSIM, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pWMSSSIM || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oSizeROI.width < 16 || oSizeROI.height < 16) return NPP_SIZE_ERROR;
  if (nSrc1Step < oSizeROI.width * channels || nSrc2Step < oSizeROI.width * channels) return NPP_STEP_ERROR;
  for (int c = 0; c < channels; ++c) {
    NppStatus st = wmsssimChannel(pSrc1, nSrc1Step, pSrc2, nSrc2Step, oSizeROI, channels, c, pWMSSSIM, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_NO_ERROR;
}

NppStatus runWMSSSIMBatch(const NppiImageDescriptor *pSrc1, const NppiImageDescriptor *pSrc2, int nBatchSize,
                          NppiSize oSizeROI, int channels, Npp32f *pWMSSSIM, bool advanced, cudaStream_t stream) {
  if (!pSrc1 || !pSrc2 || !pWMSSSIM) return NPP_NULL_POINTER_ERROR;
  if (nBatchSize <= 0) return NPP_BAD_ARGUMENT_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  std::vector<NppiImageDescriptor> h1, h2;
  NppStatus st = copyBatchDesc(pSrc1, nBatchSize, h1, stream);
  if (st != NPP_NO_ERROR) return st;
  st = copyBatchDesc(pSrc2, nBatchSize, h2, stream);
  if (st != NPP_NO_ERROR) return st;
  for (int i = 0; i < nBatchSize; ++i) {
    const NppiSize roi = advanced ? h1[i].oSize: oSizeROI;
    if (roi.width <= 0 || roi.height <= 0) return NPP_SIZE_ERROR;
    st = runWMSSSIM((const Npp8u *)h1[i].pData, h1[i].nStep, (const Npp8u *)h2[i].pData, h2[i].nStep, roi, channels,
                    pWMSSSIM + (size_t)i * channels, (Npp8u *)pWMSSSIM, stream);
    if (st != NPP_NO_ERROR) return st;
  }
  return NPP_NO_ERROR;
}

}  // namespace statistics
}  // namespace nppi
