// NPP 13.4 NV12 extended color conversions:
//   * NV12 -> RGB/BGR 601 (BT.601 full range, dedicated symbol)
//   * NV12 -> RGB/BGR BT2020
//   * NV12 -> RGB/BGR/  C4 variants with an explicit NppiColorSpace
//   * RGB -> NV12 (packed C3P2R and planar P3P2R)
//   * YUV420 -> NV12 plane repack
//
// All conversion arithmetic replicates the exact NVIDIA NPP 13.4 kernels:
//   * float32 FMA chains with the measured literals (see the tables below),
//   * results rounded with a +0.5f bias, clamped to [0, 255] and truncated,
//   * RGB->NV12 chroma is sampled from the EVEN source row of each 2x2 block
//     and averaged over the horizontal pixel pair only.
#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Exact NPP literals (read from the NPP 13.4 SASS).
struct RgbCoefs {
  float rv;  // R += rv * dV
  float gu;  // G += gu * dU
  float gv;  // G += gv * dV
  float bu;  // B += bu * dU
  bool guFirst;  // G chain order: true -> fma(dV, gv, fma(dU, gu, Y))
};

__device__ __forceinline__ float finish_pixel(float v) {
  // NPP: t = v + 0.5f; clamp to [0, 255]; truncate.  NaN collapses to 0.
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f: t;
  t = t > 255.0f ? 255.0f: t;
  return (float)(unsigned int)t;
}

template <bool BGR, bool C4, bool GU_FIRST>
__global__ void nv12_to_rgb_ext_kernel(const uint8_t *__restrict__ srcY, int nSrcStep,
                                       const uint8_t *__restrict__ srcUV, uint8_t *__restrict__ dst,
                                       int nDstStep, int width, int height, RgbCoefs cf, uint8_t alpha) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const float fy = (float)srcY[(size_t)y * nSrcStep + x];
  const uint8_t *uv = srcUV + (size_t)(y >> 1) * nSrcStep + (size_t)((x >> 1) << 1);
  const float du = __fadd_rn((float)uv[0], -128.0f);
  const float dv = __fadd_rn((float)uv[1], -128.0f);

  const float fr = finish_pixel(__fmaf_rn(dv, cf.rv, fy));
  float fg;
  if (GU_FIRST) {
    fg = __fmaf_rn(dv, cf.gv, __fmaf_rn(du, cf.gu, fy));
  } else {
    fg = __fmaf_rn(du, cf.gu, __fmaf_rn(dv, cf.gv, fy));
  }
  fg = finish_pixel(fg);
  const float fb = finish_pixel(__fmaf_rn(du, cf.bu, fy));

  const uint8_t c0 = (uint8_t)(BGR ? fb: fr);
  const uint8_t c1 = (uint8_t)fg;
  const uint8_t c2 = (uint8_t)(BGR ? fr: fb);
  uint8_t *out = dst + (size_t)y * nDstStep + (size_t)x * (C4 ? 4: 3);
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  if (C4) out[3] = alpha;
}

// Plain (non-extended) NV12->RGB keeps the historical functor arithmetic:
// R = Y + fma(dV, rv, 0.5f), G/B add 0.5f after the FMA chains.
__global__ void nv12_to_rgb_601_kernel(const uint8_t *__restrict__ srcY, int nSrcStep,
                                       const uint8_t *__restrict__ srcUV, uint8_t *__restrict__ dst, int nDstStep,
                                       int width, int height, bool bgr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const float fy = (float)srcY[(size_t)y * nSrcStep + x];
  const uint8_t *uv = srcUV + (size_t)(y >> 1) * nSrcStep + (size_t)((x >> 1) << 1);
  const float du = __fadd_rn((float)uv[0], -128.0f);
  const float dv = __fadd_rn((float)uv[1], -128.0f);

  const float fr = finish_pixel(__fadd_rn(fy, __fmaf_rn(dv, 1.4019999504089355469f, 0.5f)));
  float fg = __fmaf_rn(dv, -0.71413999795913696289f, __fmaf_rn(du, -0.34413999319076538086f, fy));
  fg = finish_pixel(__fadd_rn(fg, 0.5f));
  const float fb = finish_pixel(__fadd_rn(__fmaf_rn(du, 1.7719999551773071289f, fy), 0.5f));

  const uint8_t c0 = (uint8_t)(bgr ? fb: fr);
  const uint8_t c2 = (uint8_t)(bgr ? fr: fb);
  uint8_t *out = dst + (size_t)y * nDstStep + (size_t)x * 3;
  out[0] = c0;
  out[1] = (uint8_t)fg;
  out[2] = c2;
}

__device__ __forceinline__ float finish_chroma(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f: t;
  t = t > 255.0f ? 255.0f: t;
  return (float)(unsigned int)t;
}

struct Rgb2YuvCoefs {
  float kr, kg, kb;  // luma weights
  float ur, ug, ub;  // U = fma(R, ur, fma(G, ug, fma(B, ub, 128)))
  float vr, vg, vb;  // V = fma(R, vr, fma(G, vg, fma(B, vb, 128)))
};

template <bool PLANAR>
__global__ void rgb_to_nv12_ext_kernel(const uint8_t *__restrict__ srcPacked, const uint8_t *__restrict__ srcR,
                                       const uint8_t *__restrict__ srcG, const uint8_t *__restrict__ srcB,
                                       int nSrcStep, uint8_t *__restrict__ dstY, int nDstStep,
                                       uint8_t *__restrict__ dstUV, int width, int height, Rgb2YuvCoefs cf) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  auto load = [&](int ch, int xx, int yy) -> float {
    if (PLANAR) {
      const uint8_t *p = ch == 0 ? srcR: (ch == 1 ? srcG: srcB);
      return (float)p[(size_t)yy * nSrcStep + xx];
    }
    return (float)srcPacked[(size_t)yy * nSrcStep + (size_t)xx * 3 + ch];
  };

  const float r = load(0, x, y), g = load(1, x, y), b = load(2, x, y);
  const float ly = __fmaf_rn(r, cf.kr, __fmaf_rn(g, cf.kg, __fmul_rn(b, cf.kb)));
  dstY[(size_t)y * nDstStep + x] = (uint8_t)(unsigned int)finish_chroma(ly);

  // Chroma: one sample per horizontal pixel pair of the EVEN source row.
  if ((x & 1) == 0) {
    const int cy = y >> 1;
    const int sy = y & ~1;
    const float r0 = load(0, x, sy), g0 = load(1, x, sy), b0 = load(2, x, sy);
    const float r1 = load(0, x + 1, sy), g1 = load(1, x + 1, sy), b1 = load(2, x + 1, sy);
    const float ra = __fmul_rn(__fadd_rn(r0, r1), 0.5f);
    const float ga = __fmul_rn(__fadd_rn(g0, g1), 0.5f);
    const float ba = __fmul_rn(__fadd_rn(b0, b1), 0.5f);
    const float u = __fmaf_rn(ra, cf.ur, __fmaf_rn(ga, cf.ug, __fmaf_rn(ba, cf.ub, 128.0f)));
    const float v = __fmaf_rn(ra, cf.vr, __fmaf_rn(ga, cf.vg, __fmaf_rn(ba, cf.vb, 128.0f)));
    uint8_t *uv = dstUV + (size_t)cy * nDstStep + (size_t)x;
    uv[0] = (uint8_t)(unsigned int)finish_chroma(u);
    uv[1] = (uint8_t)(unsigned int)finish_chroma(v);
  }
}

__global__ void yuv420_to_nv12_kernel(const uint8_t *__restrict__ srcY, const uint8_t *__restrict__ srcU,
                                      const uint8_t *__restrict__ srcV, int nSrcYStep, int nSrcCStep,
                                      uint8_t *__restrict__ dstY, int nDstStep, uint8_t *__restrict__ dstUV,
                                      int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  dstY[(size_t)y * nDstStep + x] = srcY[(size_t)y * nSrcYStep + x];
  if ((x & 1) == 0 && (y & 1) == 0) {
    const int cx = x >> 1, cy = y >> 1;
    uint8_t *uv = dstUV + (size_t)cy * nDstStep + (size_t)x;
    uv[0] = srcU[(size_t)cy * nSrcCStep + cx];
    uv[1] = srcV[(size_t)cy * nSrcCStep + cx];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Launchers (extern "C" so the entry-point file stays free of CUDA syntax).
// ---------------------------------------------------------------------------
extern "C" {

cudaError_t nppiNv12ExtToRgbLaunch(const Npp8u *srcY, int nSrcStep, const Npp8u *srcUV, Npp8u *dst, int nDstStep,
                                   NppiSize roi, int coefSet, bool bgr, int dstC, Npp8u alpha,
                                   cudaStream_t stream) {
  // coefSet: 0 = BT.601 full range (1.402/1.772/0.34414/0.71414, dU-first G chain)
  //          1 = BT.709 HDTV     (1.5748/1.8556/0.187324/0.468124)
  //          2 = BT.2020 HDTV    (1.4746/1.8814/0.164553/0.571353)
  //          3 = BT.601 JPEG/plain (1.13983/2.03211/0.39465/0.5806)
  RgbCoefs cf{};
  switch (coefSet) {
    case 0: cf = {1.4019999504089355469f, -0.34413999319076538086f, -0.71413999795913696289f,
                  1.7719999551773071289f, true}; break;
    case 1: cf = {1.5748000144958496094f, -0.18732400238513946533f, -0.46812400221824645996f,
                  1.8555999994277954102f, false}; break;
    case 2: cf = {1.4745999574661254883f, -0.16455310583114624023f, -0.57135307788848876953f,
                  1.881399989128112793f, false}; break;
    default: cf = {1.1398299932479858398f, -0.39465001225471496582f, -0.58060002326965332031f,
                   2.0321099758148193359f, false}; break;
  }
  // The 601 dedicated symbol uses the dU-first G chain; every other variant
  // shares the generic dV-first chain.
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const bool c4 = dstC == 4;
  if (coefSet == 0 && !bgr && dstC == 3) {
    nv12_to_rgb_601_kernel<<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, dst, nDstStep, roi.width,
                                                       roi.height, false);
    return cudaGetLastError();
  }
  if (coefSet == 0 && bgr && dstC == 3) {
    nv12_to_rgb_601_kernel<<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, dst, nDstStep, roi.width,
                                                       roi.height, true);
    return cudaGetLastError();
  }
  if (c4) {
    if (bgr)
      nv12_to_rgb_ext_kernel<true, true, false><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, dst, nDstStep,
                                                                           roi.width, roi.height, cf, alpha);
    else
      nv12_to_rgb_ext_kernel<false, true, false><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, dst, nDstStep,
                                                                            roi.width, roi.height, cf, alpha);
  } else {
    if (bgr)
      nv12_to_rgb_ext_kernel<true, false, false><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, dst, nDstStep,
                                                                            roi.width, roi.height, cf, alpha);
    else
      nv12_to_rgb_ext_kernel<false, false, false><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, dst, nDstStep,
                                                                             roi.width, roi.height, cf, alpha);
  }
  return cudaGetLastError();
}

cudaError_t nppiRgbToNv12Launch(const Npp8u *src0, const Npp8u *src1, const Npp8u *src2, int nSrcStep, bool planar,
                                Npp8u *dstY, Npp8u *dstUV, int nDstStep, NppiSize roi, int colorSpace,
                                cudaStream_t stream) {
  Rgb2YuvCoefs cf{};
  switch (colorSpace) {
    case 1:  // NPP_BT601JPEG
      cf = {0.29899999499320983887f, 0.58700001239776611328f, 0.11400000005960464478f,
            -0.14712999761104583740f, -0.28885999321937561035f, 0.43599998950958251953f,
            0.61500000953674316406f, -0.51498997211456298828f, -0.10001000016927719116f};
      break;
    case 2:  // NPP_BT709HDTV
      cf = {0.21259999275207519531f, 0.71520000696182250977f, 0.07220000028610229492f,
            -0.11457200348377227783f, -0.38542801141738891602f, 0.5f,
            0.5f, -0.45415300130844116211f, -0.04584700241684913635f};
      break;
    default:  // NPP_BT2020HDTV
      cf = {0.26269999146461486816f, 0.67799997329711914062f, 0.05930000171065330505f,
            -0.13959999382495880127f, -0.36039999127388000488f, 0.5f,
            0.5f, -0.45980000495910644531f, -0.04020000249147415161f};
      break;
  }
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  if (planar)
    rgb_to_nv12_ext_kernel<true><<<grid, block, 0, stream>>>(nullptr, src0, src1, src2, nSrcStep, dstY, nDstStep,
                                                             dstUV, roi.width, roi.height, cf);
  else
    rgb_to_nv12_ext_kernel<false><<<grid, block, 0, stream>>>(src0, nullptr, nullptr, nullptr, nSrcStep, dstY,
                                                              nDstStep, dstUV, roi.width, roi.height, cf);
  return cudaGetLastError();
}

cudaError_t nppiYuv420ToNv12Launch(const Npp8u *const src[3], const int srcStep[3], Npp8u *dstY, Npp8u *dstUV,
                                   int nDstStep, NppiSize roi, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  yuv420_to_nv12_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], srcStep[0], srcStep[1], dstY, nDstStep,
                                                    dstUV, roi.width, roi.height);
  return cudaGetLastError();
}

}  // extern "C"
