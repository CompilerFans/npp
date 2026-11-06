// CPU-golden coverage for the pointwise XYZ / LUV / YCC / Lab conversions.
//
// Formulas mirror the measured NPP 13.4 pipelines (normalized float
// arithmetic, truncating 8u stores):
//   XYZ:  BT.709 matrix 0.412453/0.35758/0.180423, 0.212671/0.71516/0.072169,
//         0.019334/0.119193/0.950227 and its inverse 3.240479/-1.53715/
//         -0.498535, -0.969256/1.875991/0.041556, 0.055648/-0.204043/1.057311;
//   LUV:  D65 white (u'n=0.1978398, v'n=0.4683363), L in [0,100],
//         u in [-134,220], v in [-140,122];
//   YCC:  PhotoYCC with 111.4/156, 135.64/137 coefficients and 0.713267;
//   Lab:  CIE D65 (xn 0.950455, zn 1.088753) with BGR channel order.
// All AC4 variants leave the destination alpha untouched (measured).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace cs {

inline float fclamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
inline Npp8u q8(float v) { return v < 0.0f ? (Npp8u)0 : (v > 255.0f ? (Npp8u)255 : (Npp8u)v); }

inline void rgb_to_xyz(Npp8u r, Npp8u g, Npp8u b, Npp8u &x, Npp8u &y, Npp8u &z) {
  const float nr = (float)r * 0.003921569f, ng = (float)g * 0.003921569f, nb = (float)b * 0.003921569f;
  float nx = 0.412453f * nr + 0.35758f * ng + 0.180423f * nb;
  float ny = 0.212671f * nr + 0.71516f * ng + 0.072169f * nb;
  float nz = 0.019334f * nr + 0.119193f * ng + 0.950227f * nb;
  if (nx > 1.0f) nx = 1.0f;
  if (ny > 1.0f) ny = 1.0f;
  if (nz > 1.0f) nz = 1.0f;
  x = q8(nx * 255.0f);
  y = q8(ny * 255.0f);
  z = q8(nz * 255.0f);
}

inline void xyz_to_rgb(Npp8u x, Npp8u y, Npp8u z, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float nx = (float)x * 0.003921569f, ny = (float)y * 0.003921569f, nz = (float)z * 0.003921569f;
  float nr = 3.240479f * nx - 1.53715f * ny - 0.498535f * nz;
  float ng = -0.969256f * nx + 1.875991f * ny + 0.041556f * nz;
  float nb = 0.055648f * nx - 0.204043f * ny + 1.057311f * nz;
  r = q8(fclamp01(nr) * 255.0f);
  g = q8(fclamp01(ng) * 255.0f);
  b = q8(fclamp01(nb) * 255.0f);
}

constexpr float kD65xn = 0.312713f, kD65yn = 0.329016f;
constexpr float kDivisor = -2.0f * kD65xn + 12.0f * kD65yn + 3.0f;
constexpr float kUn = 4.0f * kD65xn / kDivisor;
constexpr float kVn = 9.0f * kD65yn / kDivisor;

inline void rgb_to_luv(Npp8u r, Npp8u g, Npp8u b, Npp8u &l, Npp8u &u8, Npp8u &v8) {
  const float nr = (float)r * 0.003921569f, ng = (float)g * 0.003921569f, nb = (float)b * 0.003921569f;
  const float nx = 0.412453f * nr + 0.35758f * ng + 0.180423f * nb;
  const float ny = 0.212671f * nr + 0.71516f * ng + 0.072169f * nb;
  const float nz = 0.019334f * nr + 0.119193f * ng + 0.950227f * nb;
  float temp = nx + 15.0f * ny + 3.0f * nz;
  const float nu = 4.0f * nx / temp;
  const float nv = 9.0f * ny / temp;
  float nl = 116.0f * (float)std::cbrt((double)ny) - 16.0f;
  if (nl < 0.0f) nl = 0.0f;
  if (nl > 100.0f) nl = 100.0f;
  temp = 13.0f * nl;
  float nu8 = temp * (nu - kUn);
  if (nu8 < -134.0f) nu8 = -134.0f;
  if (nu8 > 220.0f) nu8 = 220.0f;
  float nv8 = temp * (nv - kVn);
  if (nv8 < -140.0f) nv8 = -140.0f;
  if (nv8 > 122.0f) nv8 = 122.0f;
  l = q8(nl * 255.0f * 0.01f);
  u8 = q8((nu8 + 134.0f) * 255.0f * 0.0028249f);
  v8 = q8((nv8 + 140.0f) * 255.0f * 0.0038168f);
}

inline void luv_to_rgb(Npp8u l, Npp8u u8, Npp8u v8, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float nl = (float)l * 100.0f * 0.003921569f;
  const float nu8 = ((float)u8 * 354.0f * 0.003921569f) - 134.0f;
  const float nv8 = ((float)v8 * 262.0f * 0.003921569f) - 140.0f;
  const float temp = 13.0f * nl;
  const float nu = nu8 / temp + kUn;
  const float nv = nv8 / temp + kVn;
  float ny;
  if (nl > 7.9996248f) {
    ny = (nl + 16.0f) * 0.008621f;
    ny = std::pow(ny, 3.0f);
  } else {
    ny = nl * 0.001107f;
  }
  const float nx = (-9.0f * ny * nu) / ((nu - 4.0f) * nv - nu * nv);
  const float nz = (9.0f * ny - 15.0f * nv * ny - nv * nx) / (3.0f * nv);
  float nr = 3.240479f * nx - 1.53715f * ny - 0.498535f * nz;
  float ng = -0.969256f * nx + 1.875991f * ny + 0.041556f * nz;
  float nb = 0.055648f * nx - 0.204043f * ny + 1.057311f * nz;
  r = q8(fclamp01(nr) * 255.0f);
  g = q8(fclamp01(ng) * 255.0f);
  b = q8(fclamp01(nb) * 255.0f);
}

inline void rgb_to_ycc(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &c1o, Npp8u &c2o) {
  const float nr = (float)r * 0.003921569f, ng = (float)g * 0.003921569f, nb = (float)b * 0.003921569f;
  const float ny = 0.299f * nr + 0.587f * ng + 0.114f * nb;
  float c1 = nb - ny;
  c1 = 111.4f * 0.003921569f * c1 + 156.0f * 0.003921569f;
  float c2 = nr - ny;
  c2 = 135.64f * 0.003921569f * c2 + 137.0f * 0.003921569f;
  const float yv = 0.713267f * ny;
  y = q8(yv * 255.0f);
  c1o = q8(c1 * 255.0f);
  c2o = q8(c2 * 255.0f);
}

inline void ycc_to_rgb(Npp8u y, Npp8u c1v, Npp8u c2v, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float ny = ((float)y * 0.003921569f) * 1.3584f;
  const float c1 = (((float)c1v * 0.003921569f) - 156.0f * 0.003921569f) * 2.2179f;
  const float c2 = (((float)c2v * 0.003921569f) - 137.0f * 0.003921569f) * 1.8215f;
  r = q8((ny + c2) * 255.0f);
  g = q8((ny - 0.194f * c1 - 0.509f * c2) * 255.0f);
  b = q8((ny + c1) * 255.0f);
}

inline void bgr_to_lab(Npp8u bb, Npp8u g, Npp8u r, Npp8u &l, Npp8u &a, Npp8u &labB) {
  const float xn = 0.950455f, zn = 1.088753f;
  const float nr = (float)r * 0.003921569f, ng = (float)g * 0.003921569f, nb = (float)bb * 0.003921569f;
  const float x = 0.412453f * nr + 0.35758f * ng + 0.180423f * nb;
  const float y = 0.212671f * nr + 0.71516f * ng + 0.072169f * nb;
  const float z = 0.019334f * nr + 0.119193f * ng + 0.950227f * nb;
  const float cbrtY = (float)std::cbrt((double)y);
  const float fy = cbrtY - 16.0f;
  float nl = 116.0f * cbrtY - 16.0f;
  float na = 500.0f * (((float)std::cbrt((double)(x / xn)) - 16.0f) - fy);
  float nlabB = 200.0f * (fy - ((float)std::cbrt((double)(z / zn)) - 16.0f));
  nl = nl * 255.0f * 0.01f;
  na = na + 128.0f;
  nlabB = nlabB + 128.0f;
  l = q8(nl);
  a = q8(na);
  labB = q8(nlabB);
}

inline void lab_to_bgr(Npp8u l, Npp8u a, Npp8u labB, Npp8u &bb, Npp8u &g, Npp8u &r) {
  const float xn = 0.950455f, zn = 1.088753f;
  const float nl = (float)l * 100.0f * 0.003921569f;
  const float na = (float)a - 128.0f;
  const float nb = (float)labB - 128.0f;
  const float p = (nl + 16.0f) * 0.008621f;
  const float ny = p * p * p;
  const float nx = xn * std::pow(p + na * 0.002f, 3.0f);
  const float nz = zn * std::pow(p - nb * 0.005f, 3.0f);
  float nr = 3.240479f * nx - 1.53715f * ny - 0.498535f * nz;
  float ng = -0.969256f * nx + 1.875991f * ny + 0.041556f * nz;
  float nbb = 0.055648f * nx - 0.204043f * ny + 1.057311f * nz;
  r = q8(fclamp01(nr) * 255.0f);
  g = q8(fclamp01(ng) * 255.0f);
  bb = q8(fclamp01(nbb) * 255.0f);
}

enum class CS { XYZ2RGB, RGB2XYZ, RGB2LUV, LUV2RGB, RGB2YCC, YCC2RGB };

template <CS K, bool AC4, class Call>
void run_cs(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  constexpr int C = AC4 ? 4 : 3;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<Npp8u> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<Npp8u> d(c.w, c.h, C, 0, 0x5A);
    fill_regime(s, C, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false, false);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const Npp8u *hsrc = s.hptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, C);
    const Npp8u *dsrc = s.dptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, C);
    const std::vector<uint8_t> pre = d.host;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(dsrc, s.step, d.dptr(), d.step, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = pre;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const size_t so = (size_t)y * s.step + (size_t)x * C;
        const size_t db = off_bytes<Npp8u>(y, x, d.step, C);
        Npp8u o0, o1, o2;
        if constexpr (K == CS::RGB2XYZ) rgb_to_xyz(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        else if constexpr (K == CS::XYZ2RGB) xyz_to_rgb(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        else if constexpr (K == CS::RGB2LUV) rgb_to_luv(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        else if constexpr (K == CS::LUV2RGB) luv_to_rgb(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        else if constexpr (K == CS::RGB2YCC) rgb_to_ycc(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        else ycc_to_rgb(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        want[db] = o0;
        want[db + 1] = o1;
        want[db + 2] = o2;
      }
    const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
    if (!rotate_check<Npp8u, C, 3>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                   std::vector<uint8_t>(), cnt, first)) {
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

enum class LabDir { BGR2Lab, Lab2BGR };

template <LabDir K, class Call>
void run_lab(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<Npp8u> s(c.w + c.ox, c.h + c.oy, 3, c.pad, 0);
    Buf<Npp8u> d(c.w, c.h, 3, 0, 0x5A);
    fill_regime(s, 3, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false, false);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const Npp8u *hsrc = s.hptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, 3);
    const Npp8u *dsrc = s.dptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, 3);
    const std::vector<uint8_t> pre = d.host;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(dsrc, s.step, d.dptr(), d.step, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = pre;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const size_t so = (size_t)y * s.step + (size_t)x * 3;
        const size_t db = off_bytes<Npp8u>(y, x, d.step, 3);
        Npp8u o0, o1, o2;
        if constexpr (K == LabDir::BGR2Lab) bgr_to_lab(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        else lab_to_bgr(hsrc[so], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        want[db] = o0;
        want[db + 1] = o1;
        want[db + 2] = o2;
      }
    const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
    if (!rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                   std::vector<uint8_t>(), cnt, first)) {
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace cs
}  // namespace num
