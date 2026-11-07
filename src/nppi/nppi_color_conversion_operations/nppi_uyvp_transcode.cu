// NPP 13.4 UYVP_10u transcoding kernels.
//
// These entry points are declared by the 13.4 headers but are not exported by
// the reference NPP 13.2.0.58 library; the conversions are defined here from
// the format layouts:
//   * 8-bit samples enter the 10-bit domain as x*4, P010/P210 16-bit
//     containers as x>>6 (MSB-aligned 10-bit), V210/Y410 are 10-bit already.
//   * 10-bit -> 8-bit output applies x0.25 with the +0.5f rounding bias.
//   * 4:2:0 sources (NV12/P010) replicate each chroma row across the pair of
//     luma rows; 4:4:4 sources downsample chroma by averaging each horizontal
//     pixel pair; the reverse 4:4:4 outputs replicate the pair's chroma.
//   * P208 uses the packed UYVY byte order (U,Y0,V,Y1); V210 the standard
//     4-word/6-pixel packing; Y410 A/U/Y/V in 32-bit words.
#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float q8f(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  t = t < 0.0f ? 0.0f : t;
  t = t > 255.0f ? 255.0f : t;
  return (float)(unsigned int)t;
}

__device__ __forceinline__ float to10_8u(float v) { return __fmul_rn(v, 4.0f); }
__device__ __forceinline__ float from10_8u(float v) { return q8f(__fmul_rn(v, 0.25f)); }

__device__ __forceinline__ uint64_t pack_group(float u, float y0, float v, float y1) {
  return ((uint64_t)((uint32_t)(unsigned int)u & 0x3FF)) |
         ((uint64_t)((uint32_t)(unsigned int)y0 & 0x3FF) << 10) |
         ((uint64_t)((uint32_t)(unsigned int)v & 0x3FF) << 20) |
         ((uint64_t)((uint32_t)(unsigned int)y1 & 0x3FF) << 30);
}

__device__ __forceinline__ void unpack_group(const uint8_t *p, float &u, float &y0, float &v, float &y1) {
  uint64_t g = 0;
  for (int i = 0; i < 5; ++i) g |= (uint64_t)p[i] << (8 * i);
  u = (float)(g & 0x3FF);
  y0 = (float)((g >> 10) & 0x3FF);
  v = (float)((g >> 20) & 0x3FF);
  y1 = (float)((g >> 30) & 0x3FF);
}

// ---------------------------------------------------------------------------
// Semi-planar 4:2:0 / 4:2:2 (NV12/NV16/P010/P210) -> UYVP packed.
// Wide = 16-bit containers (MSB-aligned 10-bit); Vert = 4:2:0 chroma.
// ---------------------------------------------------------------------------
template <typename T, bool Wide, bool Vert>
__global__ void semiplanar_to_uyvp_kernel(const T *__restrict__ srcY, int syStep, const T *__restrict__ srcUV,
                                          int suvStep, uint8_t *__restrict__ dst, int nDstStep, int width,
                                          int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  auto cvt = [](T v) -> float {
    if constexpr (Wide) return (float)((uint32_t)v >> 6);
    else return to10_8u((float)v);
  };
  const int syE = syStep / (int)sizeof(T), suvE = suvStep / (int)sizeof(T);
  const float yv = cvt(srcY[(size_t)y * syE + x]);
  const int cy = Vert ? (y >> 1) : y;
  const T *uv = srcUV + (size_t)cy * suvE + (size_t)((x >> 1) << 1);
  const float u = cvt(uv[0]);
  const float v = cvt(uv[1]);
  if ((x & 1) == 0) {
    const float y1 = cvt(srcY[(size_t)y * syE + x + 1]);
    const uint64_t g = pack_group(u, yv, v, y1);
    uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
    for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
  }
}

// P010 with three planar inputs (Y, U, V; U/V half width, 4:2:0).
__global__ void p010_p3_to_uyvp_kernel(const uint16_t *__restrict__ py, const uint16_t *__restrict__ pu,
                                       const uint16_t *__restrict__ pv, int nSrcStep, uint8_t *__restrict__ dst,
                                       int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int se = nSrcStep / 2;
  const float yv = (float)(py[(size_t)y * se + x] >> 6);
  const int cy = y >> 1;
  const float uu = (float)(pu[(size_t)cy * se + (x >> 1)] >> 6);
  const float vv = (float)(pv[(size_t)cy * se + (x >> 1)] >> 6);
  if ((x & 1) == 0) {
    const float y1 = (float)(py[(size_t)y * se + x + 1] >> 6);
    const uint64_t g = pack_group(uu, yv, vv, y1);
    uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
    for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
  }
}

// P208 packed UYVY 8-bit -> UYVP.
__global__ void p208_to_uyvp_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const uint8_t *p = src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 4;
  const float u = to10_8u((float)p[0]);
  const float yv = to10_8u((float)p[(x & 1) ? 3 : 1]);
  const float v = to10_8u((float)p[2]);
  if ((x & 1) == 0) {
    const float y1 = to10_8u((float)p[3]);
    const uint64_t g = pack_group(u, yv, v, y1);
    uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
    for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
  }
}

// Planar 4:4:4 8-bit (I444) -> UYVP: chroma pair average in the 10-bit domain.
__global__ void i444_to_uyvp_kernel(const uint8_t *__restrict__ py, const uint8_t *__restrict__ pu,
                                    const uint8_t *__restrict__ pv, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if ((x & 1) != 0) return;
  const size_t o0 = (size_t)y * nSrcStep + x;
  const float y0 = to10_8u((float)py[o0]), y1 = to10_8u((float)py[o0 + 1]);
  const float uu = __fmul_rn(__fadd_rn(to10_8u((float)pu[o0]), to10_8u((float)pu[o0 + 1])), 0.5f);
  const float vv = __fmul_rn(__fadd_rn(to10_8u((float)pv[o0]), to10_8u((float)pv[o0 + 1])), 0.5f);
  const uint64_t g = pack_group(__fadd_rn(uu, 0.0f), y0, vv, y1);
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
  for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
}

// Packed 4:4:4 8-bit (Y,U,V triples) -> UYVP.
__global__ void yuv444_to_uyvp_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                      int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if ((x & 1) != 0) return;
  const size_t p0 = (size_t)y * nSrcStep + (size_t)x * 3;
  const float y0 = to10_8u((float)src[p0]), y1 = to10_8u((float)src[p0 + 3]);
  const float uu = __fmul_rn(__fadd_rn(to10_8u((float)src[p0 + 1]), to10_8u((float)src[p0 + 4])), 0.5f);
  const float vv = __fmul_rn(__fadd_rn(to10_8u((float)src[p0 + 2]), to10_8u((float)src[p0 + 5])), 0.5f);
  const uint64_t g = pack_group(uu, y0, vv, y1);
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
  for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
}

// V210 (6 pixels / 16 bytes) -> UYVP.
__global__ void v210_to_uyvp_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;  // 16-byte group index
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int groups = width / 6;
  if (g >= groups || y >= height) return;
  const uint32_t *w = (const uint32_t *)(src + (size_t)y * nSrcStep + (size_t)g * 16);
  const uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
  // Standard v210 packing: U0 Y0 V0 | Y1 U1 Y2 | V1 Y3 U2 | Y4 V2 Y5.
  const float u0 = (float)(w0 & 0x3FF), y0 = (float)((w0 >> 10) & 0x3FF), v0 = (float)((w0 >> 20) & 0x3FF);
  const float y1 = (float)(w1 & 0x3FF), u1 = (float)((w1 >> 10) & 0x3FF), y2 = (float)((w1 >> 20) & 0x3FF);
  const float v1 = (float)(w2 & 0x3FF), y3 = (float)((w2 >> 10) & 0x3FF), u2 = (float)((w2 >> 20) & 0x3FF);
  const float y4 = (float)(w3 & 0x3FF), v2 = (float)((w3 >> 10) & 0x3FF), y5 = (float)((w3 >> 20) & 0x3FF);
  const uint64_t ga = pack_group(u0, y0, v0, y1);
  const uint64_t gb = pack_group(u1, y2, v1, y3);
  const uint64_t gc = pack_group(u2, y4, v2, y5);
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)g * 15;
  const uint64_t arr[3] = {ga, gb, gc};
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < 5; ++i) o[k * 5 + i] = (uint8_t)((arr[k] >> (8 * i)) & 0xFF);
}

// Y410 (A/U/Y/V per pixel in 32-bit words) -> UYVP.
__global__ void y410_to_uyvp_kernel(const uint16_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if ((x & 1) != 0) return;
  const uint32_t *w = (const uint32_t *)((const uint8_t *)src + (size_t)y * nSrcStep);
  const uint32_t a = w[x], b = w[x + 1];
  const float u0 = (float)(a & 0x3FF), y0 = (float)((a >> 10) & 0x3FF), v0 = (float)((a >> 20) & 0x3FF);
  const float y1 = (float)((b >> 10) & 0x3FF);
  const uint64_t g = pack_group(u0, y0, v0, y1);
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 5;
  for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
}

// ---------------------------------------------------------------------------
// UYVP -> other formats.
// ---------------------------------------------------------------------------
// 4:4:4 planar 8-bit output (I444).
__global__ void uyvp_to_i444_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dy,
                                    uint8_t *__restrict__ du, uint8_t *__restrict__ dv, int nDstStep, int width,
                                    int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float u, y0, v, y1;
  unpack_group(src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 5, u, y0, v, y1);
  const float yv = (x & 1) ? y1 : y0;
  const uint8_t Y = (uint8_t)(unsigned int)from10_8u(yv);
  const uint8_t U = (uint8_t)(unsigned int)from10_8u(u);
  const uint8_t V = (uint8_t)(unsigned int)from10_8u(v);
  dy[(size_t)y * nDstStep + x] = Y;
  du[(size_t)y * nDstStep + x] = U;
  dv[(size_t)y * nDstStep + x] = V;
}

// Semi-planar 4:2:2 8-bit output (NV16).
__global__ void uyvp_to_nv16_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dy,
                                    int dyStep, uint8_t *__restrict__ duv, int duvStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float u, y0, v, y1;
  unpack_group(src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 5, u, y0, v, y1);
  const float yv = (x & 1) ? y1 : y0;
  dy[(size_t)y * dyStep + x] = (uint8_t)(unsigned int)from10_8u(yv);
  if ((x & 1) == 0) {
    duv[(size_t)y * duvStep + (size_t)x] = (uint8_t)(unsigned int)from10_8u(u);
    duv[(size_t)y * duvStep + (size_t)x + 1] = (uint8_t)(unsigned int)from10_8u(v);
  }
}

// P208 packed UYVY output.
__global__ void uyvp_to_p208_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float u, y0, v, y1;
  unpack_group(src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 5, u, y0, v, y1);
  const float yv = (x & 1) ? y1 : y0;
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)(x >> 1) * 4;
  if ((x & 1) == 0) {
    o[0] = (uint8_t)(unsigned int)from10_8u(u);
    o[1] = (uint8_t)(unsigned int)from10_8u(yv);
    o[2] = (uint8_t)(unsigned int)from10_8u(v);
  } else {
    o[3] = (uint8_t)(unsigned int)from10_8u(yv);
  }
}

// V210 output (6 pixels / 16 bytes); the caller handles the remainder.
__global__ void uyvp_to_v210_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int groups = width / 6;
  if (g >= groups || y >= height) return;
  const uint8_t *p = src + (size_t)y * nSrcStep + (size_t)g * 15;
  float u0, y0, v0, y1, u1, y2, v1, y3, u2, y4, v2, y5;
  unpack_group(p, u0, y0, v0, y1);
  unpack_group(p + 5, u1, y2, v1, y3);
  unpack_group(p + 10, u2, y4, v2, y5);
  uint32_t *w = (uint32_t *)(dst + (size_t)y * nDstStep + (size_t)g * 16);
  w[0] = ((uint32_t)u0 & 0x3FF) | (((uint32_t)y0 & 0x3FF) << 10) | (((uint32_t)v0 & 0x3FF) << 20);
  w[1] = ((uint32_t)y1 & 0x3FF) | (((uint32_t)u1 & 0x3FF) << 10) | (((uint32_t)y2 & 0x3FF) << 20);
  w[2] = ((uint32_t)v1 & 0x3FF) | (((uint32_t)y3 & 0x3FF) << 10) | (((uint32_t)u2 & 0x3FF) << 20);
  w[3] = ((uint32_t)y4 & 0x3FF) | (((uint32_t)v2 & 0x3FF) << 10) | (((uint32_t)y5 & 0x3FF) << 20);
}

// Y410 output (A=3, U/Y/V per pixel).
__global__ void uyvp_to_y410_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint16_t *__restrict__ dst,
                                    int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float u, y0, v, y1;
  unpack_group(src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 5, u, y0, v, y1);
  const float yv = (x & 1) ? y1 : y0;
  uint32_t *w = (uint32_t *)((uint8_t *)dst + (size_t)y * nDstStep);
  w[x] = ((uint32_t)u & 0x3FF) | (((uint32_t)yv & 0x3FF) << 10) | (((uint32_t)v & 0x3FF) << 20) |
         (3u << 30);
}

// Packed 4:4:4 8-bit output.
__global__ void uyvp_to_yuv444_kernel(const uint8_t *__restrict__ src, int nSrcStep, uint8_t *__restrict__ dst,
                                      int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float u, y0, v, y1;
  unpack_group(src + (size_t)y * nSrcStep + (size_t)(x >> 1) * 5, u, y0, v, y1);
  const float yv = (x & 1) ? y1 : y0;
  uint8_t *o = dst + (size_t)y * nDstStep + (size_t)x * 3;
  o[0] = (uint8_t)(unsigned int)from10_8u(yv);
  o[1] = (uint8_t)(unsigned int)from10_8u(u);
  o[2] = (uint8_t)(unsigned int)from10_8u(v);
}

}  // namespace

extern "C" {

cudaError_t nppiUyvpTranscodeToLaunch(const void *s0, const void *s1, const void *s2, int st0, int st1, int kind,
                                      Npp8u *dst, Npp8u *d1, Npp8u *d2, int dstStep, NppiSize roi, int wide,
                                      cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
#define NPPI_T(T, WIDE, VERT)                                                                                    \
  semiplanar_to_uyvp_kernel<T, WIDE, VERT><<<grid, block, 0, stream>>>(                                        \
      (const T *)s0, st0, (const T *)s1, st1, (uint8_t *)dst, dstStep, roi.width, roi.height)
  switch (kind) {
    case 0:  // NV12 (8-bit 4:2:0)
      NPPI_T(uint8_t, false, true);
      break;
    case 1:  // NV16 (8-bit 4:2:2)
      NPPI_T(uint8_t, false, false);
      break;
    case 2:  // P010 (16-bit 4:2:0)
      NPPI_T(uint16_t, true, true);
      break;
    case 3:  // P210 (16-bit 4:2:2)
      NPPI_T(uint16_t, true, false);
      break;
    case 4:  // P010 as three planar inputs
      p010_p3_to_uyvp_kernel<<<grid, block, 0, stream>>>((const uint16_t *)s0, (const uint16_t *)s1,
                                                        (const uint16_t *)s2, st0, (uint8_t *)dst, dstStep,
                                                        roi.width, roi.height);
      break;
    case 5:  // P208 packed UYVY
      p208_to_uyvp_kernel<<<grid, block, 0, stream>>>((const uint8_t *)s0, st0, (uint8_t *)dst, dstStep, roi.width,
                                                     roi.height);
      break;
    case 6:  // I444 planar 4:4:4
      i444_to_uyvp_kernel<<<grid, block, 0, stream>>>((const uint8_t *)s0, (const uint8_t *)s1,
                                                     (const uint8_t *)s2, st0, (uint8_t *)dst, dstStep, roi.width,
                                                     roi.height);
      break;
    case 7:  // packed YUV444
      yuv444_to_uyvp_kernel<<<grid, block, 0, stream>>>((const uint8_t *)s0, st0, (uint8_t *)dst, dstStep,
                                                       roi.width, roi.height);
      break;
    case 8: {  // V210
      const dim3 g2((roi.width / 6 + 15) / 16, (roi.height + 15) / 16);
      v210_to_uyvp_kernel<<<g2, block, 0, stream>>>((const uint8_t *)s0, st0, (uint8_t *)dst, dstStep, roi.width,
                                                   roi.height);
      break;
    }
    default:  // Y410
      y410_to_uyvp_kernel<<<grid, block, 0, stream>>>((const uint16_t *)s0, st0, (uint8_t *)dst, dstStep, roi.width,
                                                     roi.height);
      break;
  }
#undef NPPI_T
  (void)wide;
  return cudaGetLastError();
}

cudaError_t nppiUyvpTranscodeFromLaunch(const Npp8u *src, int srcStep, Npp8u *d0, Npp8u *d1, Npp8u *d2, int dstStep,
                                        int dstStep1, NppiSize roi, int kind, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  switch (kind) {
    case 0:  // I444
      uyvp_to_i444_kernel<<<grid, block, 0, stream>>>(src, srcStep, d0, d1, d2, dstStep, roi.width, roi.height);
      break;
    case 1:  // NV16
      uyvp_to_nv16_kernel<<<grid, block, 0, stream>>>(src, srcStep, d0, dstStep, d1, dstStep1, roi.width, roi.height);
      break;
    case 2:  // P208
      uyvp_to_p208_kernel<<<grid, block, 0, stream>>>(src, srcStep, d0, dstStep, roi.width, roi.height);
      break;
    case 3: {  // V210
      const dim3 g2((roi.width / 6 + 15) / 16, (roi.height + 15) / 16);
      uyvp_to_v210_kernel<<<g2, block, 0, stream>>>(src, srcStep, d0, dstStep, roi.width, roi.height);
      break;
    }
    case 4:  // Y410
      uyvp_to_y410_kernel<<<grid, block, 0, stream>>>(src, srcStep, (uint16_t *)d0, dstStep, roi.width, roi.height);
      break;
    default:  // packed YUV444
      uyvp_to_yuv444_kernel<<<grid, block, 0, stream>>>(src, srcStep, d0, dstStep, roi.width, roi.height);
      break;
  }
  return cudaGetLastError();
}

}  // extern "C"
