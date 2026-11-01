// Unsharp-mask filter executor (nppiFilterUnsharpBorder* /
// nppiFilterUnsharpGetBufferSize*).  See nppi_filter_unsharp_executor.h for the
// family contract and the 5090 probe references.
#include "npp.h"
#include "nppi_filter_unsharp_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

// Upper bound on the auto radius so a bogus sigma cannot loop forever.
constexpr int kMaxAutoRadius = 4096;

struct DtypeTraits {
  double thresholdScale;  // |src - blur| is compared against threshold * scale
  double tailEps;         // auto-radius tail criterion
  bool isFloat;
};

inline DtypeTraits traits(UnsharpDtype dtype) {
  switch (dtype) {
    case kUnsharpU8: return {128.0, 1.1813e-2, false};
    case kUnsharpU16: return {32768.0, 5.9274e-5, false};
    case kUnsharpS16: return {16384.0, 1.1490e-4, false};
    case kUnsharpF32: return {4294967296.0, 1.9930e-12, true};
  }
  return {128.0, 1.1813e-2, false};
}

// Smallest radius r >= 1 whose normalised 2-D Gaussian tail is below eps.
// The measured NPP 13.4 auto-radius behaviour follows this criterion for
// 16u / 16s / 32f and for 8u up to sigma ~ 2 (probe_j part3, probe_p part3).
inline int autoRadius(float sigma, double eps) {
  const double s2 = 2.0 * static_cast<double>(sigma) * static_cast<double>(sigma);
  // 1-D weights g_i = exp(-i^2 / (2 sigma^2)); the 2-D normalisation factor is
  // (g_0 + 2 sum_{i>=1} g_i)^2 and the window sum for radius r is
  // (g_0 + 2 sum_{i<=r} g_i)^2, so the normalised 2-D tail is 1 - ratio^2.
  double totalSign = 1.0;
  for (int i = 1; i <= kMaxAutoRadius * 2 + 64; ++i) {
    const double v = std::exp(-static_cast<double>(i) * i / s2);
    totalSign += 2.0 * v;
    if (v < 1e-18 && i > 64) break;
  }
  double window = 1.0;
  for (int r = 1; r <= kMaxAutoRadius; ++r) {
    const double v = std::exp(-static_cast<double>(r) * r / s2);
    window += 2.0 * v;
    const double ratio = window / totalSign;
    if (1.0 - ratio * ratio <= eps) return r;
  }
  return kMaxAutoRadius;
}

inline int kernelRadius(float radius, float sigma, UnsharpDtype dtype) {
  if (radius > 0.0f) {
    // k = 2*ceil(radius)+1 (probe_j part2: 0.5/1.0 -> 3, 1.5/2 -> 5, 3 -> 7).
    const double c = std::ceil(static_cast<double>(radius));
    if (c > 4096.0) return 4096;
    return static_cast<int>(c);
  }
  return autoRadius(sigma, traits(dtype).tailEps);
}

// Normalised Gaussian taps, float, row-major k x k.
inline void buildTaps(std::vector<float> &taps, int r, float sigma) {
  const int k = 2 * r + 1;
  taps.resize(static_cast<std::size_t>(k) * k);
  const double s2 = 2.0 * static_cast<double>(sigma) * static_cast<double>(sigma);
  double sum = 0.0;
  for (int a = 0; a < k; ++a) {
    for (int b = 0; b < k; ++b) {
      const double dy = a - r, dx = b - r;
      const double v = std::exp(-(dy * dy + dx * dx) / s2);
      taps[static_cast<std::size_t>(a) * k + b] = static_cast<float>(v);
      sum += v;
    }
  }
  const double inv = 1.0 / sum;
  for (auto &t : taps) t = static_cast<float>(static_cast<double>(t) * inv);
}

// Saturating integer emission: C truncation toward zero, then clamping.
template <typename Dst>
__host__ __device__ __forceinline__ Dst unsharp_emit(double v) {
  if (v != v) return Dst(0);  // NaN -> 0 (not reachable with finite inputs)
  if constexpr (std::is_same_v<Dst, Npp8u>) {
    return v <= 0.0 ? (Npp8u)0 : (v >= 255.0 ? (Npp8u)255 : (Npp8u)(int)v);
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    return v <= 0.0 ? (Npp16u)0 : (v >= 65535.0 ? (Npp16u)65535 : (Npp16u)(int)v);
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    return v <= -32768.0 ? (Npp16s)-32768 : (v >= 32767.0 ? (Npp16s)32767 : (Npp16s)(int)v);
  } else {
    static_assert(sizeof(Dst) == 0, "unexpected integer destination type");
  }
}

// Unaligned-safe element load (source steps may be odd for 16/32-bit data).
template <typename T> __device__ __forceinline__ T load_unaligned(const char *p) {
  T v;
  memcpy(&v, p, sizeof(T));
  return v;
}

// Unsharp kernel.
//   taps      : k x k normalised Gaussian in device memory (pDeviceBuffer)
//   clamp box : [loX, hiX] x [loY, hiY] in pSrc-relative coordinates, with
//               lo = -oSrcOffset and hi = ROI-1-oSrcOffset.
template <typename Src, typename Dst, int Channels, bool AlphaPr>
__global__ void unsharp_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst, int nDstStep,
                               int width, int height, const float *__restrict__ taps, int k, int loX, int hiX,
                               int loY, int hiY, float weight, float threshold, float thresholdScale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int r = k >> 1;
  const int nch = AlphaPr ? 3 : Channels;
  const char *const srcBase = reinterpret_cast<const char *>(pSrc);
  const int cx = x < loX ? loX : (x > hiX ? hiX : x);
  const int cy = y < loY ? loY : (y > hiY ? hiY : y);
  // Source steps are not required to be element-aligned (NPP accepts an odd
  // source step), so every sample is read byte-wise through memcpy; the MACA
  // device faults on unaligned 16/32-bit loads.
  const char *centre =
      srcBase + static_cast<std::ptrdiff_t>(cy) * nSrcStep +
      static_cast<std::ptrdiff_t>(cx) * Channels * static_cast<std::ptrdiff_t>(sizeof(Src));
  double acc[4] = {0.0, 0.0, 0.0, 0.0};
  for (int a = 0; a < k; ++a) {
    int ry = y + a - r;
    ry = ry < loY ? loY : (ry > hiY ? hiY : ry);
    const char *row = srcBase + static_cast<std::ptrdiff_t>(ry) * nSrcStep;
    for (int b = 0; b < k; ++b) {
      int rx = x + b - r;
      rx = rx < loX ? loX : (rx > hiX ? hiX : rx);
      const float t = taps[a * k + b];
      if (t == 0.0f) continue;
      const char *pix = row + (static_cast<std::ptrdiff_t>(rx) * Channels) * sizeof(Src);
      for (int c = 0; c < nch; ++c) {
        const Src v = load_unaligned<Src>(pix + static_cast<std::ptrdiff_t>(c) * sizeof(Src));
        acc[c] += static_cast<double>(t) * static_cast<double>(v);
      }
    }
  }
  Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + static_cast<std::ptrdiff_t>(y) * nDstStep) +
              static_cast<std::ptrdiff_t>(x) * Channels;
  const bool useThreshold = threshold > 0.0f && !(threshold != threshold);
  const double thr = static_cast<double>(threshold) * static_cast<double>(thresholdScale);
  for (int c = 0; c < nch; ++c) {
    const double s = static_cast<double>(load_unaligned<Src>(centre + static_cast<std::ptrdiff_t>(c) * sizeof(Src)));
    const double hp = s - acc[c];
    double out;
    if (useThreshold && fabs(hp) <= thr) {
      out = s;
    } else {
      out = s + static_cast<double>(weight) * hp;
    }
    if constexpr (std::is_floating_point_v<Dst>) {
      drow[c] = static_cast<Dst>(out);
    } else {
      drow[c] = unsharp_emit<Dst>(out);
    }
  }
}

inline NppStatus finish_launch(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Dst step validation (unsigned comparison, mirroring the fixed-filter family:
// a negative step passes).  Alignment is the element size for C1/C3 and the
// pixel size for C4/AC4.
template <typename Dst, int Channels>
NppStatus check_dst_step(int nDstStep, int roiWidth) {
  const int pix = static_cast<int>(sizeof(Dst)) * Channels;
  const int align = Channels == 4 ? pix : static_cast<int>(sizeof(Dst));
  const unsigned uStep = static_cast<unsigned>(nDstStep);
  if (uStep < static_cast<unsigned>(roiWidth * pix)) return NPP_STEP_ERROR;
  if (uStep % static_cast<unsigned>(align) != 0) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

int unsharpKernelSize(float radius, float sigma, UnsharpDtype dtype) {
  const int r = kernelRadius(radius, sigma, dtype);
  return 2 * r + 1;
}

int unsharpBufferSize(float radius, float sigma, UnsharpDtype dtype) {
  const int k = unsharpKernelSize(radius, sigma, dtype);
  return k * k * static_cast<int>(sizeof(Npp32f));
}

NppStatus unsharpGetBufferSize(float radius, float sigma, UnsharpDtype dtype, int *hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (radius < 0.0f) return NPP_SIZE_ERROR;
  // Measured (5090, NPP 13.4): an explicit positive radius pins the window, so
  // sigma == 0 is accepted here even though the filter itself rejects it.  A
  // sigma <= 0 with an auto radius is an error; NPP 13.4 never returns from
  // sigma == 0 (it hangs), so that case is reported as NPP_SIZE_ERROR.
  if (radius == 0.0f) {
    if (!(sigma > 0.0f)) return NPP_SIZE_ERROR;
  } else if (sigma < 0.0f) {
    return NPP_SIZE_ERROR;
  }
  *hpBufferSize = unsharpBufferSize(radius, sigma, dtype);
  return NPP_NO_ERROR;
}

template <typename Src>
NppStatus unsharpGetBufferSizeT(float radius, float sigma, int *hpBufferSize) {
  UnsharpDtype dtype = kUnsharpU8;
  if constexpr (std::is_same_v<Src, Npp16u>) {
    dtype = kUnsharpU16;
  } else if constexpr (std::is_same_v<Src, Npp16s>) {
    dtype = kUnsharpS16;
  } else if constexpr (std::is_same_v<Src, Npp32f>) {
    dtype = kUnsharpF32;
  }
  return unsharpGetBufferSize(radius, sigma, dtype, hpBufferSize);
}

template <typename Src, typename Dst, int Channels, bool AlphaPreserve>
NppStatus filterUnsharpBorder(const Src *pSrc, int nSrcStep, NppiPoint oSrcOffset, Dst *pDst, int nDstStep,
                              NppiSize oSizeROI, float nRadius, float nSigma, float nWeight, float nThreshold,
                              NppiBorderType eBorderType, Npp8u *pDeviceBuffer, cudaStream_t stream) {
  if (!pSrc || !pDst || !pDeviceBuffer) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (nRadius < 0.0f) return NPP_SIZE_ERROR;
  if (nSigma <= 0.0f) return NPP_SIZE_ERROR;
  UnsharpDtype dtype = kUnsharpU8;
  if constexpr (std::is_same_v<Dst, Npp16u>) {
    dtype = kUnsharpU16;
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    dtype = kUnsharpS16;
  } else if constexpr (std::is_same_v<Dst, Npp32f>) {
    dtype = kUnsharpF32;
  }
  const int r = kernelRadius(nRadius, nSigma, dtype);
  const int k = 2 * r + 1;
  // The kernel has to fit inside the ROI (this is also what makes a zero-sized
  // ROI return NPP_SIZE_ERROR, as measured).
  if (k > oSizeROI.width || k > oSizeROI.height) return NPP_SIZE_ERROR;
  NppStatus st = check_dst_step<Dst, Channels>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;

  std::vector<float> taps;
  buildTaps(taps, r, nSigma);
  (void)cudaGetLastError();
  const std::size_t tapBytes = taps.size() * sizeof(float);
  cudaError_t cerr = cudaMemcpyAsync(pDeviceBuffer, taps.data(), tapBytes, cudaMemcpyHostToDevice, stream);
  if (cerr != cudaSuccess) {
    (void)cudaGetLastError();
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  unsharp_kernel<Src, Dst, Channels, AlphaPreserve><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
      reinterpret_cast<const float *>(pDeviceBuffer), k, -oSrcOffset.x, oSizeROI.width - 1 - oSrcOffset.x,
      -oSrcOffset.y, oSizeROI.height - 1 - oSrcOffset.y, nWeight, nThreshold,
      static_cast<float>(traits(dtype).thresholdScale));
  return finish_launch(stream);
}

}  // namespace filter
}  // namespace nppi

// ---------------------------------------------------------------------------
// Explicit instantiations (f11 package: nppiFilterUnsharpBorder family).
// ---------------------------------------------------------------------------
#define NPPI_UNSHARP_INSTANTIATE(Src, Dst, Ch, Ac)                                                            \
  template NppStatus nppi::filter::filterUnsharpBorder<Src, Dst, Ch, Ac>(                                     \
      const Src *, int, NppiPoint, Dst *, int, NppiSize, float, float, float, float, NppiBorderType, Npp8u *, \
      cudaStream_t);

NPPI_UNSHARP_INSTANTIATE(Npp8u, Npp8u, 1, false)
NPPI_UNSHARP_INSTANTIATE(Npp8u, Npp8u, 3, false)
NPPI_UNSHARP_INSTANTIATE(Npp8u, Npp8u, 4, false)
NPPI_UNSHARP_INSTANTIATE(Npp8u, Npp8u, 4, true)
NPPI_UNSHARP_INSTANTIATE(Npp16u, Npp16u, 1, false)
NPPI_UNSHARP_INSTANTIATE(Npp16u, Npp16u, 3, false)
NPPI_UNSHARP_INSTANTIATE(Npp16u, Npp16u, 4, false)
NPPI_UNSHARP_INSTANTIATE(Npp16u, Npp16u, 4, true)
NPPI_UNSHARP_INSTANTIATE(Npp16s, Npp16s, 1, false)
NPPI_UNSHARP_INSTANTIATE(Npp16s, Npp16s, 3, false)
NPPI_UNSHARP_INSTANTIATE(Npp16s, Npp16s, 4, false)
NPPI_UNSHARP_INSTANTIATE(Npp16s, Npp16s, 4, true)
NPPI_UNSHARP_INSTANTIATE(Npp32f, Npp32f, 1, false)
NPPI_UNSHARP_INSTANTIATE(Npp32f, Npp32f, 3, false)
NPPI_UNSHARP_INSTANTIATE(Npp32f, Npp32f, 4, false)
NPPI_UNSHARP_INSTANTIATE(Npp32f, Npp32f, 4, true)

#undef NPPI_UNSHARP_INSTANTIATE

template NppStatus nppi::filter::unsharpGetBufferSizeT<Npp8u>(float, float, int *);
template NppStatus nppi::filter::unsharpGetBufferSizeT<Npp16u>(float, float, int *);
template NppStatus nppi::filter::unsharpGetBufferSizeT<Npp16s>(float, float, int *);
template NppStatus nppi::filter::unsharpGetBufferSizeT<Npp32f>(float, float, int *);
