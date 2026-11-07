// NPP 13.4 UYVY color conversions.
//
//   UYVY / UYVY16  <->  RGB / RGBA (packed or planar, 8- and 16-bit)
//
// Layout: packed 4:2:2 with U,V shared by each horizontal pixel pair; every
// luma row has its own chroma pair.  The 8-bit path uses the measured NPP
// arithmetic (float32 FMA chains, +0.5f rounding, truncation).  The 16-bit
// path works in the same 10-bit normalized domain NPP uses: samples are
// multiplied by 1023/65535, chroma is centred on 512, and results are scaled
// back by 65535/1023 before rounding.
#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float finish8(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f: t;
  t = t > 255.0f ? 255.0f: t;
  return (float)(unsigned int)t;
}

__device__ __forceinline__ float finish16(float v) {
  // v is already in 16-bit scale.
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f: t;
  t = t > 65535.0f ? 65535.0f: t;
  return (float)(unsigned int)t;
}

struct CsCoefs {
  float rv, gu, gv, bu;      // decode matrix
  float kr, kg, kb;          // encode luma
  float ur, ug, ub;          // encode U
  float vr, vg, vb;          // encode V
};

__device__ __host__ inline CsCoefs cs_coefs(int cs) {
  CsCoefs c{};
  if (cs == 1) {  // NPP_BT601JPEG (plain matrix)
    c = {1.1398299932479858398f, -0.39465001225471496582f, -0.58060002326965332031f,
         2.0321099758148193359f,
         0.29899999499320983887f, 0.58700001239776611328f, 0.11400000005960464478f,
         -0.14712999761104583740f, -0.28885999321937561035f, 0.43599998950958251953f,
         0.61500000953674316406f, -0.51498997211456298828f, -0.10001000016927719116f};
  } else if (cs == 2) {  // NPP_BT709HDTV
    c = {1.5748000144958496094f, -0.18732400238513946533f, -0.46812400221824645996f,
         1.8555999994277954102f,
         0.21259999275207519531f, 0.71520000696182250977f, 0.07220000028610229492f,
         -0.11457200348377227783f, -0.38542801141738891602f, 0.5f,
         0.5f, -0.45415300130844116211f, -0.04584700241684913635f};
  } else {  // NPP_BT2020HDTV
    c = {1.4745999574661254883f, -0.16455310583114624023f, -0.57135307788848876953f,
         1.881399989128112793f,
         0.26269999146461486816f, 0.67799997329711914062f, 0.05930000171065330505f,
         -0.13959999382495880127f, -0.36039999127388000488f, 0.5f,
         0.5f, -0.45980000495910644531f, -0.04020000249147415161f};
  }
  return c;
}

__device__ __forceinline__ void decode8(const CsCoefs &c, float y, float u, float v, float &r, float &g, float &b) {
  const float du = __fadd_rn(u, -128.0f);
  const float dv = __fadd_rn(v, -128.0f);
  r = __fmaf_rn(dv, c.rv, y);
  g = __fmaf_rn(du, c.gu, __fmaf_rn(dv, c.gv, y));
  b = __fmaf_rn(du, c.bu, y);
}

__device__ __forceinline__ void encode8(const CsCoefs &c, float r, float g, float b, float &y, float &u, float &v) {
  y = __fmaf_rn(r, c.kr, __fmaf_rn(g, c.kg, __fmul_rn(b, c.kb)));
  u = __fmaf_rn(r, c.ur, __fmaf_rn(g, c.ug, __fmaf_rn(b, c.ub, 128.0f)));
  v = __fmaf_rn(r, c.vr, __fmaf_rn(g, c.vg, __fmaf_rn(b, c.vb, 128.0f)));
}

__device__ __forceinline__ void decode16(const CsCoefs &c, float y, float u, float v, float &r, float &g, float &b) {
  const float S = 0.015609979629516601562f;  // 1023/65535
  const float du = __fmaf_rn(u, S, -512.0f);
  const float dv = __fmaf_rn(v, S, -512.0f);
  const float yy = __fmul_rn(y, S);
  const float I = 64.06158447265625f;  // 65535/1023
  r = __fmul_rn(__fmaf_rn(dv, c.rv, yy), I);
  g = __fmul_rn(__fmaf_rn(du, c.gu, __fmaf_rn(dv, c.gv, yy)), I);
  b = __fmul_rn(__fmaf_rn(du, c.bu, yy), I);
}

__device__ __forceinline__ void encode16(const CsCoefs &c, float r, float g, float b, float &y, float &u, float &v) {
  const float S = 0.015609979629516601562f;
  const float I = 64.06158447265625f;
  const float rr = __fmul_rn(r, S), gg = __fmul_rn(g, S), bb = __fmul_rn(b, S);
  y = __fmul_rn(__fmaf_rn(rr, c.kr, __fmaf_rn(gg, c.kg, __fmul_rn(bb, c.kb))), I);
  u = __fmul_rn(__fmaf_rn(rr, c.ur, __fmaf_rn(gg, c.ug, __fmaf_rn(bb, c.ub, 512.0f))), I);
  v = __fmul_rn(__fmaf_rn(rr, c.vr, __fmaf_rn(gg, c.vg, __fmaf_rn(bb, c.vb, 512.0f))), I);
}

// ---------------------------------------------------------------------------
// UYVY -> RGB  (8-bit)
// ---------------------------------------------------------------------------
// Mode 0: packed C3, 1: packed AC4 (alpha), 2: planar P3.
template <int Mode>
__global__ void uyvy8_to_rgb_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                    uint8_t *__restrict__ dst1, uint8_t *__restrict__ dst2, int nDstStep,
                                    int width, int height, CsCoefs cf, uint8_t alpha) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const uint8_t *p = src + (size_t)y * nSrcStep + (size_t)((x >> 1) << 2);
  const float u = (float)p[0];
  const float yv = (float)p[(x & 1) ? 3: 1];
  const float v = (float)p[2];
  float r, g, b;
  decode8(cf, yv, u, v, r, g, b);
  const uint8_t R = (uint8_t)(unsigned int)finish8(r);
  const uint8_t G = (uint8_t)(unsigned int)finish8(g);
  const uint8_t B = (uint8_t)(unsigned int)finish8(b);
  if (Mode == 2) {
    dst[(size_t)y * nDstStep + x] = R;
    dst1[(size_t)y * nDstStep + x] = G;
    dst2[(size_t)y * nDstStep + x] = B;
  } else {
    const int C = Mode == 1 ? 4: 3;
    uint8_t *o = dst + (size_t)y * nDstStep + (size_t)x * C;
    o[0] = R;
    o[1] = G;
    o[2] = B;
    if (Mode == 1) o[3] = alpha;
  }
}

// ---------------------------------------------------------------------------
// RGB -> UYVY  (8-bit)
// ---------------------------------------------------------------------------
template <int Mode>  // 0: packed C3, 1: planar P3, 2: packed AC4 (alpha ignored)
__global__ void rgb8_to_uyvy_kernel(const uint8_t *__restrict__ src, const uint8_t *__restrict__ s1,
                                    const uint8_t *__restrict__ s2, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height, CsCoefs cf) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  auto at = [&](int ch, int xx) -> float {
    if (Mode == 1) {
      const uint8_t *p = ch == 0 ? src: (ch == 1 ? s1: s2);
      return (float)p[(size_t)y * nSrcStep + xx];
    }
    const int C = Mode == 2 ? 4: 3;
    return (float)src[(size_t)y * nSrcStep + (size_t)xx * C + ch];
  };
  float yy, uu, vv;
  encode8(cf, at(0, x), at(1, x), at(2, x), yy, uu, vv);
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)((x >> 1) << 2);
  if ((x & 1) == 0) {
    const float ra = __fmul_rn(__fadd_rn(at(0, x), at(0, x + 1)), 0.5f);
    const float ga = __fmul_rn(__fadd_rn(at(1, x), at(1, x + 1)), 0.5f);
    const float ba = __fmul_rn(__fadd_rn(at(2, x), at(2, x + 1)), 0.5f);
    float dy, du, dv;
    encode8(cf, ra, ga, ba, dy, du, dv);
    o[0] = (uint8_t)(unsigned int)finish8(du);
    o[1] = (uint8_t)(unsigned int)finish8(yy);
    o[2] = (uint8_t)(unsigned int)finish8(dv);
  } else {
    o[3] = (uint8_t)(unsigned int)finish8(yy);
  }
}

// ---------------------------------------------------------------------------
// UYVY16 -> RGB16
// ---------------------------------------------------------------------------
template <int Mode>
__global__ void uyvy16_to_rgb16_kernel(const uint16_t *__restrict__ src, int nSrcStep, uint16_t *__restrict__ dst,
                                       uint16_t *__restrict__ dst1, uint16_t *__restrict__ dst2, int nDstStep,
                                       int width, int height, CsCoefs cf, uint16_t alpha) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int sstep = nSrcStep / 2;
  const int dstep = nDstStep / 2;
  const uint16_t *p = src + (size_t)y * sstep + (size_t)((x >> 1) << 2);
  const float u = (float)p[0];
  const float yv = (float)p[(x & 1) ? 3: 1];
  const float v = (float)p[2];
  float r, g, b;
  decode16(cf, yv, u, v, r, g, b);
  const uint16_t R = (uint16_t)(unsigned int)finish16(r);
  const uint16_t G = (uint16_t)(unsigned int)finish16(g);
  const uint16_t B = (uint16_t)(unsigned int)finish16(b);
  if (Mode == 2) {
    dst[(size_t)y * dstep + x] = R;
    dst1[(size_t)y * dstep + x] = G;
    dst2[(size_t)y * dstep + x] = B;
  } else {
    const int C = Mode == 1 ? 4: 3;
    uint16_t *o = dst + (size_t)y * dstep + (size_t)x * C;
    o[0] = R;
    o[1] = G;
    o[2] = B;
    if (Mode == 1) o[3] = alpha;
  }
}

template <int Mode>
__global__ void rgb16_to_uyvy16_kernel(const uint16_t *__restrict__ src, const uint16_t *__restrict__ s1,
                                       const uint16_t *__restrict__ s2, int nSrcStep, uint16_t *__restrict__ dst,
                                       int nDstStep, int width, int height, CsCoefs cf) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int sstep = nSrcStep / 2;
  const int dstep = nDstStep / 2;
  auto at = [&](int ch, int xx) -> float {
    if (Mode == 1) {
      const uint16_t *p = ch == 0 ? src: (ch == 1 ? s1: s2);
      return (float)p[(size_t)y * sstep + xx];
    }
    const int C = Mode == 2 ? 4: 3;
    return (float)src[(size_t)y * sstep + (size_t)xx * C + ch];
  };
  float yy, uu, vv;
  encode16(cf, at(0, x), at(1, x), at(2, x), yy, uu, vv);
  uint16_t *o = dst + (size_t)y * dstep + (size_t)((x >> 1) << 2);
  if ((x & 1) == 0) {
    const float ra = __fmul_rn(__fadd_rn(at(0, x), at(0, x + 1)), 0.5f);
    const float ga = __fmul_rn(__fadd_rn(at(1, x), at(1, x + 1)), 0.5f);
    const float ba = __fmul_rn(__fadd_rn(at(2, x), at(2, x + 1)), 0.5f);
    float dy, du, dv;
    encode16(cf, ra, ga, ba, dy, du, dv);
    o[0] = (uint16_t)(unsigned int)finish16(du);
    o[1] = (uint16_t)(unsigned int)finish16(yy);
    o[2] = (uint16_t)(unsigned int)finish16(dv);
  } else {
    o[3] = (uint16_t)(unsigned int)finish16(yy);
  }
}

}  // namespace

extern "C" {

cudaError_t nppiUyvy8ToRgbLaunch(const Npp8u *src, int nSrcStep, Npp8u *dst, Npp8u *dst1, Npp8u *dst2, int nDstStep,
                                 NppiSize roi, int mode, int cs, Npp8u alpha, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
  if (mode == 0)
    uyvy8_to_rgb_kernel<0><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,
                                                       roi.height, cf, alpha);
  else if (mode == 1)
    uyvy8_to_rgb_kernel<1><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,
                                                       roi.height, cf, alpha);
  else
    uyvy8_to_rgb_kernel<2><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,
                                                       roi.height, cf, alpha);
  return cudaGetLastError();
}

cudaError_t nppiRgb8ToUyvyLaunch(const Npp8u *src, const Npp8u *s1, const Npp8u *s2, int nSrcStep, Npp8u *dst,
                                 int nDstStep, NppiSize roi, int mode, int cs, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
  if (mode == 0)
    rgb8_to_uyvy_kernel<0><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, nDstStep, roi.width, roi.height, cf);
  else if (mode == 1)
    rgb8_to_uyvy_kernel<1><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, nDstStep, roi.width, roi.height, cf);
  else
    rgb8_to_uyvy_kernel<2><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, nDstStep, roi.width, roi.height, cf);
  return cudaGetLastError();
}

cudaError_t nppiUyvy16ToRgb16Launch(const Npp16u *src, int nSrcStep, Npp16u *dst, Npp16u *dst1, Npp16u *dst2,
                                    int nDstStep, NppiSize roi, int mode, int cs, Npp16u alpha,
                                    cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
  if (mode == 0)
    uyvy16_to_rgb16_kernel<0><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,
                                                          roi.height, cf, alpha);
  else if (mode == 1)
    uyvy16_to_rgb16_kernel<1><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,
                                                          roi.height, cf, alpha);
  else
    uyvy16_to_rgb16_kernel<2><<<grid, block, 0, stream>>>(src, nSrcStep, dst, dst1, dst2, nDstStep, roi.width,
                                                          roi.height, cf, alpha);
  return cudaGetLastError();
}

cudaError_t nppiRgb16ToUyvy16Launch(const Npp16u *src, const Npp16u *s1, const Npp16u *s2, int nSrcStep, Npp16u *dst,
                                    int nDstStep, NppiSize roi, int mode, int cs, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
  if (mode == 0)
    rgb16_to_uyvy16_kernel<0><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, nDstStep, roi.width, roi.height,
                                                          cf);
  else if (mode == 1)
    rgb16_to_uyvy16_kernel<1><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, nDstStep, roi.width, roi.height,
                                                          cf);
  else
    rgb16_to_uyvy16_kernel<2><<<grid, block, 0, stream>>>(src, s1, s2, nSrcStep, dst, nDstStep, roi.width, roi.height,
                                                          cf);
  return cudaGetLastError();
}

}  // extern "C"
