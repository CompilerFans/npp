// CPU-reference output coverage for the NPP 13.4 UYVY families:
//   nppiUYVY[16]_<T>_ToRGB_<T>_{C3C3R,C3AC4R,C3P3R}
//   nppiRGB[16]_<T>_ToUYVY[16]_<T>_{C3C3R,P3C3R} and the RGBA AC4C3R forms.
//
// Layout: packed 4:2:2 (U,Y0,V,Y1); each luma row has its own chroma pair.
// The 8-bit path uses float32 FMA chains with the NPP literals and +0.5f
// rounding; the 16-bit path converts samples to the 10-bit domain
// (x * 1023/65535, chroma centre 512), converts, then scales back by
// 65535/1023 and rounds.  Both were fitted bit-exactly against NPP.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace uyvy {

struct Coefs {
  float rv, gu, gv, bu;
  float kr, kg, kb;
  float ur, ug, ub;
  float vr, vg, vb;
};

inline Coefs coefs(int cs) {
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

inline float fma32(float a, float b, float c) { return std::fma(a, b, c); }

// Reference NPP schedules the encode/decode FMA chains differently per
// colour-space instantiation, so a boundary value can land one LSB away.  The
// tests allow exactly one LSB (the framework's 8-bit colour comparisons use a
// wider 4-LSB envelope).
template <typename T>
inline bool within_one_lsb(const std::vector<uint8_t> &got, const std::vector<uint8_t> &want, size_t &off) {
  if (got.size() != want.size()) {
    off = std::min(got.size(), want.size());
    return false;
  }
  for (size_t i = 0; i + sizeof(T) <= got.size(); i += sizeof(T)) {
    T g, w;
    std::memcpy(&g, got.data() + i, sizeof(T));
    std::memcpy(&w, want.data() + i, sizeof(T));
    const long long d = (long long)g - (long long)w;
    if (d < -1 || d > 1) {
      off = i;
      return false;
    }
  }
  return true;
}

inline void report_first_diff(const char *id, const std::vector<uint8_t> &got, const std::vector<uint8_t> &want) {
  const size_t n = std::min(got.size(), want.size());
  for (size_t i = 0; i < n; ++i)
    if (got[i] != want[i]) {
      std::fprintf(stderr, "[UYVY-DIFF] %s offset=%zu got=%u want=%u\n", id, i, (unsigned)got[i],
                   (unsigned)want[i]);
      return;
    }
  std::fprintf(stderr, "[UYVY-DIFF] %s size mismatch got=%zu want=%zu\n", id, got.size(), want.size());
}
// Correctly rounded single-operation helpers: immune to FP contraction, so the
// host model performs exactly the same operations as the device intrinsics.
inline float mul32(float a, float b) { return std::fma(a, b, 0.0f); }
inline float add32(float a, float b) { return std::fma(a, 1.0f, b); }

template <typename T>
inline T qv(float v) {
  if (v != v) return 0;
  float t = v + 0.5f;
  const float mx = std::is_same_v<T, Npp8u> ? 255.0f: 65535.0f;
  t = t < 0.0f ? 0.0f: (t > mx ? mx: t);
  return (T)(unsigned int)t;
}

inline void decode(const Coefs &c, float y, float u, float v, bool wide, float &r, float &g, float &b) {
  float du, dv, yy, I = 1.0f;
  if (wide) {
    const float S = 0.015609979629516601562f;
    I = 64.06158447265625f;
    du = fma32(u, S, -512.0f);
    dv = fma32(v, S, -512.0f);
    yy = mul32(y, S);
  } else {
    du = add32(u, -128.0f);
    dv = add32(v, -128.0f);
    yy = y;
  }
  r = mul32(fma32(dv, c.rv, yy), I);
  g = mul32(fma32(du, c.gu, fma32(dv, c.gv, yy)), I);
  b = mul32(fma32(du, c.bu, yy), I);
}

inline void encode(const Coefs &c, float r, float g, float b, bool wide, float &y, float &u, float &v) {
  float I = 1.0f, ctr = 128.0f;
  if (wide) {
    const float S = 0.015609979629516601562f;
    I = 64.06158447265625f;
    ctr = 512.0f;
    r = mul32(r, S);
    g = mul32(g, S);
    b = mul32(b, S);
  }
  y = mul32(fma32(r, c.kr, fma32(g, c.kg, mul32(b, c.kb))), I);
  u = mul32(fma32(r, c.ur, fma32(g, c.ug, fma32(b, c.ub, ctr))), I);
  v = mul32(fma32(r, c.vr, fma32(g, c.vg, fma32(b, c.vb, ctr))), I);
}

template <typename T, int Mode, class Call>
void run_uyvy_dec(const char *id, int cs, T alpha, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  const Coefs cf = coefs(cs);
  const bool wide = std::is_same_v<T, Npp16u>;
  const int C = Mode == 1 ? 4: 3;
  for (const Cfg &c: cfgs) {
    const int sstep = align16((size_t)c.w * 2 * sizeof(T));
    const int dstep = align16((size_t)c.w * C * sizeof(T));
    Buf<T> src(c.w, c.h, 2, 0, 0);
    fill_regime(src, 2, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, wide);
    src.dev.copyFromHost(src.host);
    Buf<T> d(c.w, c.h, C, 0, 0x5A);
    Buf<T> d1(c.w, c.h, 1, Mode == 2 ? dstep - align16((size_t)c.w * sizeof(T)): 0, 0x33);
    Buf<T> d2(c.w, c.h, 1, Mode == 2 ? dstep - align16((size_t)c.w * sizeof(T)): 0, 0x33);
    d.dev.copyFromHost(d.host);
    d1.dev.copyFromHost(d1.host);
    d2.dev.copyFromHost(d2.host);
    const NppStatus st = [&]() {
      T *sp = reinterpret_cast<T *>(src.dptr());
      T *dp = reinterpret_cast<T *>(d.dptr());
      if constexpr (Mode == 0) {
        return call(sp, sstep, NppiPoint{0, 0}, dp, dstep, NppiSize{c.w, c.h}, (NppiColorSpace)cs, ctx);
      } else if constexpr (Mode == 1) {
        return call(sp, sstep, NppiPoint{0, 0}, dp, dstep, NppiSize{c.w, c.h}, (NppiColorSpace)cs, alpha, ctx);
      } else {
        T *planes[3] = {dp, reinterpret_cast<T *>(d1.dptr()), reinterpret_cast<T *>(d2.dptr())};
        return call(sp, sstep, NppiPoint{0, 0}, planes, dstep, NppiSize{c.w, c.h}, (NppiColorSpace)cs, ctx);
      }
    }();
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    d1.dev.copyToHost(d1.host);
    d2.dev.copyToHost(d2.host);
    ++cnt.executed;
    std::vector<uint8_t> w0 = d.host, w1 = d1.host, w2 = d2.host;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const int si = (y * sstep / (int)sizeof(T)) + ((x >> 1) << 2);
        const T *hp = src.hptr() + si;
        const float u = (float)hp[0];
        const float yv = (float)hp[(x & 1) ? 3: 1];
        const float v = (float)hp[2];
        float r, g, b;
        decode(cf, yv, u, v, wide, r, g, b);
        const T R = qv<T>(r), G = qv<T>(g), B = qv<T>(b);
        if constexpr (Mode == 2) {
          const size_t offp = ((size_t)y * dstep / (int)sizeof(T) + (size_t)x) * sizeof(T);
          std::memcpy(w0.data() + offp, &R, sizeof(T));
          std::memcpy(w1.data() + offp, &G, sizeof(T));
          std::memcpy(w2.data() + offp, &B, sizeof(T));
        } else {
          const size_t off = (size_t)y * dstep + (size_t)x * C * sizeof(T);
          std::memcpy(w0.data() + off, &R, sizeof(T));
          std::memcpy(w0.data() + off + sizeof(T), &G, sizeof(T));
          std::memcpy(w0.data() + off + 2 * sizeof(T), &B, sizeof(T));
          if constexpr (Mode == 1) {
            const T A = alpha;
            std::memcpy(w0.data() + off + 3 * sizeof(T), &A, sizeof(T));
          }
        }
      }
    size_t badOff = 0;
    bool bad = !within_one_lsb<T>(d.host, w0, badOff);
    if constexpr (Mode == 2)
      bad = bad || !within_one_lsb<T>(d1.host, w1, badOff) || !within_one_lsb<T>(d2.host, w2, badOff);
    if (bad) {
      report_first_diff(id, d.host, w0);
      if constexpr (Mode == 2) {
        report_first_diff(id, d1.host, w1);
        report_first_diff(id, d2.host, w2);
      }
      ++cnt.compared;
      ++cnt.mismatched;
      if (first.empty()) first = std::string("[") + id + "] UYVY decode mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}

template <typename T, int Mode, class Call>  // Mode 0: C3 packed, 1: P3 planar, 2: AC4 packed
void run_rgb_enc(const char *id, int cs, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  const Coefs cf = coefs(cs);
  const bool wide = std::is_same_v<T, Npp16u>;
  const int SC = Mode == 2 ? 4: 3;
  for (const Cfg &c: cfgs) {
    const int sstep = align16((size_t)c.w * SC * sizeof(T));
    const int dstep = align16((size_t)c.w * 2 * sizeof(T));
    auto at = [](const Buf<T> &b, int ch, int x, int y, int C) -> float {
      return (float)b.hptr()[(size_t)y * (b.step / (int)sizeof(T)) + (size_t)x * C + ch];
    };
    const int ppad = Mode == 1 ? sstep - align16((size_t)c.w * sizeof(T)): 0;
    Buf<T> src(c.w, c.h, SC, 0, 0);
    Buf<T> s1(c.w, c.h, 1, ppad, 0), s2(c.w, c.h, 1, ppad, 0);
    fill_regime(src, SC, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, wide);
    if constexpr (Mode == 1) {
      fill_regime(s1, 1, case_seed(id, c.w, c.h, 1, 0, 0, 1), kWide, 0, false, false, wide);
      fill_regime(s2, 1, case_seed(id, c.w, c.h, 2, 0, 0, 1), kWide, 0, false, false, wide);
    }
    src.dev.copyFromHost(src.host);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<T> d(c.w, c.h, 2, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = [&]() {
      T *dp = reinterpret_cast<T *>(d.dptr());
      if constexpr (Mode == 1) {
        const T *planes[3] = {reinterpret_cast<const T *>(src.dptr()),
                              reinterpret_cast<const T *>(s1.dptr()),
                              reinterpret_cast<const T *>(s2.dptr())};
        return call(planes, sstep, NppiPoint{0, 0}, dp, dstep, NppiSize{c.w, c.h}, (NppiColorSpace)cs, ctx);
      } else {
        (void)s1;
        (void)s2;
        return call(reinterpret_cast<T *>(src.dptr()), sstep, NppiPoint{0, 0}, dp, dstep, NppiSize{c.w, c.h},
                    (NppiColorSpace)cs, ctx);
      }
    }();
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        auto px = [&](int ch, int xx) -> float {
          if constexpr (Mode == 1) {
            const Buf<T> *b = ch == 0 ? &src: (ch == 1 ? &s1: &s2);
            return (float)b->hptr()[(size_t)y * (b->step / (int)sizeof(T)) + (size_t)xx];
          }
          return at(src, ch, xx, y, SC);
        };
        float yy, uu, vv;
        encode(cf, px(0, x), px(1, x), px(2, x), wide, yy, uu, vv);
        const size_t off = (size_t)y * dstep + (size_t)((x >> 1) << 2) * sizeof(T);
        if ((x & 1) == 0) {
          const float ra = (px(0, x) + px(0, x + 1)) * 0.5f;
          const float ga = (px(1, x) + px(1, x + 1)) * 0.5f;
          const float ba = (px(2, x) + px(2, x + 1)) * 0.5f;
          float dy, du, dv;
          encode(cf, ra, ga, ba, wide, dy, du, dv);
          const T U = qv<T>(du), Y = qv<T>(yy), V = qv<T>(dv);
          std::memcpy(want.data() + off, &U, sizeof(T));
          std::memcpy(want.data() + off + sizeof(T), &Y, sizeof(T));
          std::memcpy(want.data() + off + 2 * sizeof(T), &V, sizeof(T));
        } else {
          const T Y = qv<T>(yy);
          std::memcpy(want.data() + off + 3 * sizeof(T), &Y, sizeof(T));
        }
      }
    size_t badOff = 0;
    if (!within_one_lsb<T>(d.host, want, badOff)) {
      report_first_diff(id, d.host, want);
      {
        size_t off = 0;
        const size_t n = std::min(want.size(), d.host.size());
        while (off < n && want[off] == d.host[off]) ++off;
        const int elem = (int)sizeof(T);
        const int row = (int)(off / (size_t)dstep);
        const int byteCol = (int)(off % (size_t)dstep);
        const int pair = byteCol / (4 * elem);
        std::fprintf(stderr, "[UYVY-AT] row=%d pair=%d cfg=%dx%d want=", row, pair, c.w, c.h);
        for (int b = 0; b < 2 * elem; ++b) std::fprintf(stderr, "%u,", (unsigned)want[(size_t)row * dstep + byteCol - (byteCol % (2 * elem)) + b]);
        std::fprintf(stderr, " got=");
        for (int b = 0; b < 2 * elem; ++b) std::fprintf(stderr, "%u,", (unsigned)d.host[(size_t)row * dstep + byteCol - (byteCol % (2 * elem)) + b]);
        std::fprintf(stderr, "\n");
        {
          const int sx0 = pair * 2, sx1 = pair * 2 + 1;
          std::fprintf(stderr, "[UYVY-SRC] row=%d pair=%d px0=", row, pair);
          for (int ch = 0; ch < 3; ++ch) {
            float v;
            if constexpr (Mode == 1) {
              const Buf<T> *b = ch == 0 ? &src: (ch == 1 ? &s1: &s2);
              v = (float)b->hptr()[(size_t)row * (b->step / (int)sizeof(T)) + (size_t)sx0];
            } else {
              v = (float)src.hptr()[(size_t)row * (src.step / (int)sizeof(T)) + (size_t)sx0 * SC + ch];
            }
            std::fprintf(stderr, "%d,", (int)v);
          }
          std::fprintf(stderr, " px1=");
          for (int ch = 0; ch < 3; ++ch) {
            float v;
            if constexpr (Mode == 1) {
              const Buf<T> *b = ch == 0 ? &src: (ch == 1 ? &s1: &s2);
              v = (float)b->hptr()[(size_t)row * (b->step / (int)sizeof(T)) + (size_t)sx1];
            } else {
              v = (float)src.hptr()[(size_t)row * (src.step / (int)sizeof(T)) + (size_t)sx1 * SC + ch];
            }
            std::fprintf(stderr, "%d,", (int)v);
          }
          std::fprintf(stderr, "\n");
        }
      }
      ++cnt.compared;
      ++cnt.mismatched;
      if (first.empty()) first = std::string("[") + id + "] UYVY encode mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}

}  // namespace uyvy
}  // namespace num
