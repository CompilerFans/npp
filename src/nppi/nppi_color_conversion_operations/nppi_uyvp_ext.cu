// NPP 13.4 UYVP_10u color conversions.
//
// UYVP_10u packed: 5 bytes hold one 4:2:2 group as four little-endian 10-bit
// fields U, Y0, V, Y1.  UYVP planar (the C3P3R encoder output): plane 0 holds
// one 10-bit luma per pixel, planes 1/2 one chroma value per horizontal pixel
// pair, each stored in a 16-bit little-endian container.
//
// Inputs are normalised into the 10-bit domain: 8-bit samples x4, 16-bit
// samples x1023/65535, 16f samples (clamped to [0,1]) x1023.  Decoding applies
// the measured per-colour-space matrix with chroma centred on 512 and scales
// to the destination type: 8-bit x0.25, 16-bit x65535/1023, 16f /1023.
#include "npp.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

struct CsCoefs {
  float rv, gu, gv, bu;
  float kr, kg, kb;
  float ur, ug, ub;
  float vr, vg, vb;
};

__device__ __host__ inline CsCoefs cs_coefs(int cs) {
  if (cs == 1)
    return {1.1398299932479858398f, -0.39465001225471496582f, -0.58060002326965332031f,
            2.0321099758148193359f,
            0.29899999499320983887f, 0.58700001239776611328f, 0.11400000005960464478f,
            -0.14712999761104583740f, -0.28885999321937561035f, 0.43599998950958251953f,
            0.61500000953674316406f, -0.51498997211456298828f, -0.10001000016927719116f};
  if (cs == 2)
    return {1.5748000144958496094f, -0.18732400238513946533f, -0.46812400221824645996f,
            1.8555999994277954102f,
            0.21259999275207519531f, 0.71520000696182250977f, 0.07220000028610229492f,
            -0.11457200348377227783f, -0.38542801141738891602f, 0.5f,
            0.5f, -0.45415300130844116211f, -0.04584700241684913635f};
  return {1.4745999574661254883f, -0.16455310583114624023f, -0.57135307788848876953f,
          1.881399989128112793f,
          0.26269999146461486816f, 0.67799997329711914062f, 0.05930000171065330505f,
          -0.13959999382495880127f, -0.36039999127388000488f, 0.5f,
          0.5f, -0.45980000495910644531f, -0.04020000249147415161f};
}

__device__ __forceinline__ float q10(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f : t;
  t = t > 1023.0f ? 1023.0f : t;
  return (float)(unsigned int)t;
}

__device__ __forceinline__ float q8(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f : t;
  t = t > 255.0f ? 255.0f : t;
  return (float)(unsigned int)t;
}

__device__ __forceinline__ float q16(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f : t;
  t = t > 65535.0f ? 65535.0f : t;
  return (float)(unsigned int)t;
}

__device__ __forceinline__ float to10_8u(float v) { return __fmul_rn(v, 4.0f); }
__device__ __forceinline__ float to10_16u(float v) { return __fmul_rn(v, 0.015609979629516601562f); }
__device__ __forceinline__ float to10_16f(float v) {
  v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  return __fmul_rn(v, 1023.0f);
}

// ---------------------------------------------------------------------------
// UYVP packed -> RGB (8u, 16u or 16f output; packed, AC4 or planar).
// OutKind: 0 = Npp8u, 1 = Npp16u, 2 = Npp16f.  Mode: 0 C3, 1 AC4, 2 P3.
// ---------------------------------------------------------------------------
template <int OutKind, int Mode>
__global__ void uyvp_to_rgb_kernel(const uint8_t *__restrict__ src, int nSrcStep, void *__restrict__ dst,
                                   void *__restrict__ dst1, void *__restrict__ dst2, int nDstStep, int width,
                                   int height, CsCoefs cf, float alpha) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const uint8_t *p = src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 5;
  unsigned long long g = 0;
  for (int i = 0; i < 5; ++i) g |= (unsigned long long)p[i] << (8 * i);
  const float u = (float)(g & 0x3FF);
  const float yv = (float)((g >> ((x & 1) ? 30 : 10)) & 0x3FF);
  const float v = (float)((g >> 20) & 0x3FF);
  const float du = __fadd_rn(u, -512.0f);
  const float dv = __fadd_rn(v, -512.0f);
  const float fr = __fmaf_rn(dv, cf.rv, yv);
  const float fg = __fmaf_rn(du, cf.gu, __fmaf_rn(dv, cf.gv, yv));
  const float fb = __fmaf_rn(du, cf.bu, yv);
  if constexpr (OutKind == 0) {
    const uint8_t R = (uint8_t)(unsigned int)q8(__fmul_rn(fr, 0.25f));
    const uint8_t G = (uint8_t)(unsigned int)q8(__fmul_rn(fg, 0.25f));
    const uint8_t B = (uint8_t)(unsigned int)q8(__fmul_rn(fb, 0.25f));
    if (Mode == 2) {
      uint8_t *d0 = (uint8_t *)dst, *d1 = (uint8_t *)dst1, *d2 = (uint8_t *)dst2;
      d0[(size_t)y * nDstStep + x] = R;
      d1[(size_t)y * nDstStep + x] = G;
      d2[(size_t)y * nDstStep + x] = B;
    } else {
      const int C = Mode == 1 ? 4 : 3;
      uint8_t *o = (uint8_t *)dst + (size_t)y * nDstStep + (size_t)x * C;
      o[0] = R;
      o[1] = G;
      o[2] = B;
      if (Mode == 1) o[3] = (uint8_t)(unsigned int)alpha;
    }
  } else if constexpr (OutKind == 1) {
    const uint16_t R = (uint16_t)(unsigned int)q16(__fmul_rn(fr, 64.06158447265625f));
    const uint16_t G = (uint16_t)(unsigned int)q16(__fmul_rn(fg, 64.06158447265625f));
    const uint16_t B = (uint16_t)(unsigned int)q16(__fmul_rn(fb, 64.06158447265625f));
    const int es = nDstStep / 2;
    if (Mode == 2) {
      uint16_t *d0 = (uint16_t *)dst, *d1 = (uint16_t *)dst1, *d2 = (uint16_t *)dst2;
      d0[(size_t)y * es + x] = R;
      d1[(size_t)y * es + x] = G;
      d2[(size_t)y * es + x] = B;
    } else {
      const int C = Mode == 1 ? 4 : 3;
      uint16_t *o = (uint16_t *)dst + (size_t)y * es + (size_t)x * C;
      o[0] = R;
      o[1] = G;
      o[2] = B;
      if (Mode == 1) o[3] = (uint16_t)(unsigned int)alpha;
    }
  } else {
    const float S = 1.0f / 1023.0f;
    const __half R = __float2half_rn(__fmul_rn(fr, S));
    const __half G = __float2half_rn(__fmul_rn(fg, S));
    const __half B = __float2half_rn(__fmul_rn(fb, S));
    const int es = nDstStep / 2;
    if (Mode == 2) {
      __half *d0 = (__half *)dst, *d1 = (__half *)dst1, *d2 = (__half *)dst2;
      d0[(size_t)y * es + x] = R;
      d1[(size_t)y * es + x] = G;
      d2[(size_t)y * es + x] = B;
    } else {
      const int C = Mode == 1 ? 4 : 3;
      __half *o = (__half *)dst + (size_t)y * es + (size_t)x * C;
      o[0] = R;
      o[1] = G;
      o[2] = B;
      if (Mode == 1) o[3] = __float2half_rn(alpha);
    }
  }
}

// ---------------------------------------------------------------------------
// RGB -> UYVP packed.  InKind: 0 = 8u, 1 = 16u, 2 = 16f.
// Mode: 0 packed C3, 1 planar C3 source, 2 packed AC4 source.
// ---------------------------------------------------------------------------
template <int InKind, int Mode>
__device__ __forceinline__ float load10(const void *src, const void *s1, const void *s2, int nSrcStep, int ch, int x,
                                        int y) {
  if constexpr (Mode == 1) {
    const void *p = ch == 0 ? src : (ch == 1 ? s1 : s2);
    if constexpr (InKind == 0) return to10_8u((float)((const uint8_t *)p)[(size_t)y * nSrcStep + x]);
    if constexpr (InKind == 1)
      return to10_16u((float)((const uint16_t *)p)[(size_t)y * (nSrcStep / 2) + x]);
    return to10_16f(__half2float(((const __half *)p)[(size_t)y * (nSrcStep / 2) + x]));
  } else {
    const int C = Mode == 2 ? 4 : 3;
    if constexpr (InKind == 0)
      return to10_8u((float)((const uint8_t *)src)[(size_t)y * nSrcStep + (size_t)x * C + ch]);
    if constexpr (InKind == 1)
      return to10_16u((float)((const uint16_t *)src)[(size_t)y * (nSrcStep / 2) + (size_t)x * C + ch]);
    return to10_16f(__half2float(((const __half *)src)[(size_t)y * (nSrcStep / 2) + (size_t)x * C + ch]));
  }
}

template <int InKind, int Mode>
__global__ void rgb_to_uyvp_kernel(const void *__restrict__ src, const void *__restrict__ s1,
                                   const void *__restrict__ s2, int nSrcStep, uint8_t *__restrict__ dst,
                                   uint8_t *__restrict__ p1, uint8_t *__restrict__ p2, int nDstStep, int width,
                                   int height, CsCoefs cf, bool planar) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const float r = load10<InKind, Mode>(src, s1, s2, nSrcStep, 0, x, y);
  const float g = load10<InKind, Mode>(src, s1, s2, nSrcStep, 1, x, y);
  const float b = load10<InKind, Mode>(src, s1, s2, nSrcStep, 2, x, y);
  const float ly = q10(__fmaf_rn(r, cf.kr, __fmaf_rn(g, cf.kg, __fmul_rn(b, cf.kb))));
  if (planar) {
    // plane 0: luma per pixel, 16-bit little-endian containers.
    uint16_t *py = (uint16_t *)dst;
    py[(size_t)y * (nDstStep / 2) + x] = (uint16_t)(unsigned int)ly;
    if ((x & 1) == 0) {
      const float r1 = load10<InKind, Mode>(src, s1, s2, nSrcStep, 0, x + 1, y);
      const float g1 = load10<InKind, Mode>(src, s1, s2, nSrcStep, 1, x + 1, y);
      const float b1 = load10<InKind, Mode>(src, s1, s2, nSrcStep, 2, x + 1, y);
      // Planar UYVP: per-pixel chroma floats are clamped to [0,1023] and then
      // averaged (measured NPP behaviour; distinct from the packed encoder).
      auto clamp10 = [](float w) { return w < 0.0f ? 0.0f : (w > 1023.0f ? 1023.0f : w); };
      const float u0 = clamp10(__fmaf_rn(r, cf.ur, __fmaf_rn(g, cf.ug, __fmaf_rn(b, cf.ub, 512.0f))));
      const float u1 = clamp10(__fmaf_rn(r1, cf.ur, __fmaf_rn(g1, cf.ug, __fmaf_rn(b1, cf.ub, 512.0f))));
      const float v0 = clamp10(__fmaf_rn(r, cf.vr, __fmaf_rn(g, cf.vg, __fmaf_rn(b, cf.vb, 512.0f))));
      const float v1 = clamp10(__fmaf_rn(r1, cf.vr, __fmaf_rn(g1, cf.vg, __fmaf_rn(b1, cf.vb, 512.0f))));
      const float u = q10(__fmul_rn(__fadd_rn(u0, u1), 0.5f));
      const float v = q10(__fmul_rn(__fadd_rn(v0, v1), 0.5f));
      uint16_t *pu = (uint16_t *)p1, *pv = (uint16_t *)p2;
      pu[(size_t)y * (nDstStep / 2) + (x >> 1)] = (uint16_t)(unsigned int)u;
      pv[(size_t)y * (nDstStep / 2) + (x >> 1)] = (uint16_t)(unsigned int)v;
    }
    return;
  }
  if ((x & 1) == 0) {
    const float r1 = load10<InKind, Mode>(src, s1, s2, nSrcStep, 0, x + 1, y);
    const float g1 = load10<InKind, Mode>(src, s1, s2, nSrcStep, 1, x + 1, y);
    const float b1 = load10<InKind, Mode>(src, s1, s2, nSrcStep, 2, x + 1, y);
    const float ly1 = q10(__fmaf_rn(r1, cf.kr, __fmaf_rn(g1, cf.kg, __fmul_rn(b1, cf.kb))));
    const float ra = __fmul_rn(__fadd_rn(r, r1), 0.5f);
    const float ga = __fmul_rn(__fadd_rn(g, g1), 0.5f);
    const float ba = __fmul_rn(__fadd_rn(b, b1), 0.5f);
    const float u = q10(__fmaf_rn(ra, cf.ur, __fmaf_rn(ga, cf.ug, __fmaf_rn(ba, cf.ub, 512.0f))));
    const float v = q10(__fmaf_rn(ra, cf.vr, __fmaf_rn(ga, cf.vg, __fmaf_rn(ba, cf.vb, 512.0f))));
    const uint64_t grp = ((uint64_t)((uint32_t)u & 0x3FF)) | ((uint64_t)((uint32_t)ly & 0x3FF) << 10) |
                         ((uint64_t)((uint32_t)v & 0x3FF) << 20) | ((uint64_t)((uint32_t)ly1 & 0x3FF) << 30);
    uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
    for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((grp >> (8 * i)) & 0xFF);
  }
}

}  // namespace

extern "C" {

cudaError_t nppiUyvpToRgbLaunch(const Npp8u *src, int nSrcStep, void *dst, void *dst1, void *dst2, int nDstStep,
                                NppiSize roi, int cs, int mode, int outKind, float alpha, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
#define NPPI_UYVP_DEC(K, M)                                                                                     \
  uyvp_to_rgb_kernel<K, M><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,     \
                                                       roi.height, cf, alpha)
  if (outKind == 0) {
    if (mode == 0) NPPI_UYVP_DEC(0, 0);
    else if (mode == 1) NPPI_UYVP_DEC(0, 1);
    else NPPI_UYVP_DEC(0, 2);
  } else if (outKind == 1) {
    if (mode == 0) NPPI_UYVP_DEC(1, 0);
    else if (mode == 1) NPPI_UYVP_DEC(1, 1);
    else NPPI_UYVP_DEC(1, 2);
  } else {
    if (mode == 0) NPPI_UYVP_DEC(2, 0);
    else if (mode == 1) NPPI_UYVP_DEC(2, 1);
    else NPPI_UYVP_DEC(2, 2);
  }
#undef NPPI_UYVP_DEC
  return cudaGetLastError();
}

cudaError_t nppiRgbToUyvpLaunch(const void *src, const void *s1, const void *s2, int nSrcStep, Npp8u *dst,
                                Npp8u *p1, Npp8u *p2, int nDstStep, NppiSize roi, int cs, int mode, int inKind,
                                bool planar, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
#define NPPI_UYVP_ENC(K, M)                                                                                     \
  rgb_to_uyvp_kernel<K, M><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, p1, p2, nDstStep, roi.width, \
                                                       roi.height, cf, planar)
  if (inKind == 0) {
    if (mode == 0) NPPI_UYVP_ENC(0, 0);
    else if (mode == 1) NPPI_UYVP_ENC(0, 1);
    else NPPI_UYVP_ENC(0, 2);
  } else if (inKind == 1) {
    if (mode == 0) NPPI_UYVP_ENC(1, 0);
    else if (mode == 1) NPPI_UYVP_ENC(1, 1);
    else NPPI_UYVP_ENC(1, 2);
  } else {
    if (mode == 0) NPPI_UYVP_ENC(2, 0);
    else if (mode == 1) NPPI_UYVP_ENC(2, 1);
    else NPPI_UYVP_ENC(2, 2);
  }
#undef NPPI_UYVP_ENC
  return cudaGetLastError();
}

}  // extern "C"
