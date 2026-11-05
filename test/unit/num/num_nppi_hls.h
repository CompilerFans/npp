// CPU-golden coverage for the packed nppiRGBToHLS / nppiHLSToRGB /
// nppiRGBToHSV / nppiHSVToRGB 8u conversions (C3R and AC4R).
//
// The formulas reproduce the measured conversion pipeline: normalized float
// arithmetic on the 0..255 channels, clamped and quantized with lrintf
// (round half to even).  The AC4 variants leave the destination alpha
// untouched (measured: a 0x5A filled destination keeps 0x5A).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace hls {

inline Npp8u q(float v) {
  if (v < 0.0f) return 0;
  if (v > 255.0f) return 255;
  return (Npp8u)std::lrint(v);
}

inline void rgb_to_hls(Npp8u r, Npp8u g, Npp8u b, Npp8u &h, Npp8u &l, Npp8u &s) {
  const float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
  const float maxv = std::max(rf, std::max(gf, bf));
  const float minv = std::min(rf, std::min(gf, bf));
  const float lv = 0.5f * (maxv + minv);
  float hv = 0.0f, sv = 0.0f;
  const float d = maxv - minv;
  if (d > 0.0f) {
    sv = (lv < 0.5f) ? d / (maxv + minv) : d / (2.0f - maxv - minv);
    if (maxv == rf) {
      hv = (gf - bf) / d;
      if (gf < bf) hv += 6.0f;
    } else if (maxv == gf) {
      hv = 2.0f + (bf - rf) / d;
    } else {
      hv = 4.0f + (rf - gf) / d;
    }
    hv /= 6.0f;
  }
  h = q(hv * 255.0f);
  l = q(lv * 255.0f);
  s = q(sv * 255.0f);
}

inline float hue_to_rgb(float p, float qf, float t) {
  if (t < 0.0f) t += 1.0f;
  if (t > 1.0f) t -= 1.0f;
  if (t < 1.0f / 6.0f) return p + (qf - p) * 6.0f * t;
  if (t < 0.5f) return qf;
  if (t < 2.0f / 3.0f) return p + (qf - p) * (2.0f / 3.0f - t) * 6.0f;
  return p;
}

inline void hls_to_rgb(Npp8u h, Npp8u l, Npp8u s, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float hf = h / 255.0f, lf = l / 255.0f, sf = s / 255.0f;
  float rf, gf, bf;
  if (sf <= 0.0f) {
    rf = gf = bf = lf;
  } else {
    const float qf = (lf < 0.5f) ? (lf * (1.0f + sf)) : (lf + sf - lf * sf);
    const float p = 2.0f * lf - qf;
    rf = hue_to_rgb(p, qf, hf + 1.0f / 3.0f);
    gf = hue_to_rgb(p, qf, hf);
    bf = hue_to_rgb(p, qf, hf - 1.0f / 3.0f);
  }
  r = q(rf * 255.0f);
  g = q(gf * 255.0f);
  b = q(bf * 255.0f);
}

inline void rgb_to_hsv(Npp8u r, Npp8u g, Npp8u b, Npp8u &h, Npp8u &s, Npp8u &v) {
  const float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
  const float maxv = std::max(rf, std::max(gf, bf));
  const float minv = std::min(rf, std::min(gf, bf));
  const float d = maxv - minv;
  float hv = 0.0f;
  const float sv = (maxv <= 0.0f) ? 0.0f : d / maxv;
  if (d > 0.0f) {
    if (maxv == rf) {
      hv = (gf - bf) / d;
      if (gf < bf) hv += 6.0f;
    } else if (maxv == gf) {
      hv = 2.0f + (bf - rf) / d;
    } else {
      hv = 4.0f + (rf - gf) / d;
    }
    hv /= 6.0f;
  }
  h = q(hv * 255.0f);
  s = q(sv * 255.0f);
  v = q(maxv * 255.0f);
}

inline void hsv_to_rgb(Npp8u h, Npp8u s, Npp8u v, Npp8u &r, Npp8u &g, Npp8u &b) {
  const float hf = h / 255.0f, sf = s / 255.0f, vf = v / 255.0f;
  if (sf <= 0.0f) {
    const Npp8u gv = q(vf * 255.0f);
    r = g = b = gv;
    return;
  }
  const float h6 = hf * 6.0f;
  const int i = (int)std::floor(h6);
  const float f = h6 - (float)i;
  const float p = vf * (1.0f - sf);
  const float qq = vf * (1.0f - sf * f);
  const float t = vf * (1.0f - sf * (1.0f - f));
  float rf = 0, gf = 0, bf = 0;
  switch (i % 6) {
    case 0: rf = vf; gf = t; bf = p; break;
    case 1: rf = qq; gf = vf; bf = p; break;
    case 2: rf = p; gf = vf; bf = t; break;
    case 3: rf = p; gf = qq; bf = vf; break;
    case 4: rf = t; gf = p; bf = vf; break;
    default: rf = vf; gf = p; bf = qq; break;
  }
  r = q(rf * 255.0f);
  g = q(gf * 255.0f);
  b = q(bf * 255.0f);
}

enum class Kind { RGB2HLS, HLS2RGB, RGB2HSV, HSV2RGB };

template <Kind K, bool AC4, class Call>
void run_hls(const char *id, Call call) {
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
        if constexpr (K == Kind::RGB2HLS) {
          rgb_to_hls(hsrc[so + 0], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        } else if constexpr (K == Kind::HLS2RGB) {
          hls_to_rgb(hsrc[so + 0], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        } else if constexpr (K == Kind::RGB2HSV) {
          rgb_to_hsv(hsrc[so + 0], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        } else {
          hsv_to_rgb(hsrc[so + 0], hsrc[so + 1], hsrc[so + 2], o0, o1, o2);
        }
        want[db + 0] = o0;
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

}  // namespace hls
}  // namespace num
