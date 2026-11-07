// CPU-golden coverage for the NPP 13.4 UYVP_10u conversions.
//
// Packed UYVP holds one 4:2:2 group in 5 bytes as four little-endian 10-bit
// fields U, Y0, V, Y1.  Planar UYVP stores luma per pixel and chroma per
// horizontal pair in 16-bit little-endian containers.
// Inputs normalise to the 10-bit domain: 8-bit x4, 16-bit x1023/65535, 16f
// (clamped to [0,1]) x1023.  Decoding uses the per-colour-space matrix with
// chroma centred on 512 and scales to the destination type (8-bit x0.25,
// 16-bit x65535/1023, 16f /1023).
//
// The reference NPP 13.2.0.58 build returns all-zero results for the 16f
// entry points, so those tests skip the comparison on the reference build and
// run fully on MACA / the in-tree CUDA build.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace uyvp {

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
inline float mul32(float a, float b) { return std::fma(a, b, 0.0f); }
inline float add32(float a, float b) { return std::fma(a, 1.0f, b); }

inline float q10f(float v) {
  if (v != v) return 0;
  float t = add32(v, 0.5f);
  t = t < 0.0f ? 0.0f : (t > 1023.0f ? 1023.0f : t);
  return (float)(unsigned int)t;
}

template <typename T>
inline T qout(float v) {
  if constexpr (std::is_same_v<T, Npp16f>) return half_round(v * (1.0f / 1023.0f));
  else {
    if (v != v) return 0;
    float t = add32(v, 0.5f);
    const float mx = std::is_same_v<T, Npp8u> ? 255.0f : 65535.0f;
    t = t < 0.0f ? 0.0f : (t > mx ? mx : t);
    return (T)(unsigned int)t;
  }
}

inline void decode10(const Coefs &c, float yv, float u, float v, float &r, float &g, float &b) {
  const float du = add32(u, -512.0f);
  const float dv = add32(v, -512.0f);
  r = fma32(dv, c.rv, yv);
  g = fma32(du, c.gu, fma32(dv, c.gv, yv));
  b = fma32(du, c.bu, yv);
}

inline void encode10(const Coefs &c, float r, float g, float b, float &y, float &u, float &v) {
  y = q10f(fma32(r, c.kr, fma32(g, c.kg, mul32(b, c.kb))));
  u = q10f(fma32(r, c.ur, fma32(g, c.ug, fma32(b, c.ub, 512.0f))));
  v = q10f(fma32(r, c.vr, fma32(g, c.vg, fma32(b, c.vb, 512.0f))));
}

template <typename T>
inline float to10(float v) {
  if constexpr (std::is_same_v<T, Npp8u>) return mul32(v, 4.0f);
  else if constexpr (std::is_same_v<T, Npp16u>) return mul32(v, 0.015609979629516601562f);
  else {
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return mul32(v, 1023.0f);
  }
}

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
    if constexpr (std::is_same_v<T, Npp16f>) {
      const float d = half_to_float_host(g) - half_to_float_host(w);
      if (std::fabs(d) > 1.0f / 1023.0f + 1e-6f) {
        off = i;
        return false;
      }
    } else {
      const long long d = (long long)g - (long long)w;
      if (d < -1 || d > 1) {
        off = i;
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// UYVP packed -> RGB.  Mode: 0 C3, 1 AC4, 2 P3.
// ---------------------------------------------------------------------------
template <typename T, int Mode, class Call>
void run_uyvp_decode(const char *id, int cs, T alpha, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
#ifdef USE_NVIDIA_NPP_TESTS
  if constexpr (std::is_same_v<T, Npp16f>) {
    GTEST_SKIP() << "[" << id << "] reference NPP 13.2.0.58 returns all-zero 16f output";
    return;
  }
#endif
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  const Coefs cf = coefs(cs);
  const int C = Mode == 1 ? 4 : 3;
  for (const Cfg &c : cfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)c.w * C * sizeof(T));
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<T> d(c.w, c.h, C, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const int pad = Mode == 2 ? dstep - align16((size_t)c.w * sizeof(T)) : 0;
    Buf<T> d1(c.w, c.h, 1, pad, 0x33), d2(c.w, c.h, 1, pad, 0x33);
    d1.dev.copyFromHost(d1.host);
    d2.dev.copyFromHost(d2.host);
    const NppStatus st = [&]() {
      if constexpr (Mode == 0) {
        return call(src.dptr(), sstep, reinterpret_cast<T *>(d.dptr()), dstep, NppiSize{c.w, c.h},
                    (NppiColorSpace)cs, ctx);
      } else if constexpr (Mode == 1) {
        return call(src.dptr(), sstep, reinterpret_cast<T *>(d.dptr()), dstep, NppiSize{c.w, c.h},
                    (NppiColorSpace)cs, alpha, ctx);
      } else {
        T *planes[3] = {reinterpret_cast<T *>(d.dptr()), reinterpret_cast<T *>(d1.dptr()),
                        reinterpret_cast<T *>(d2.dptr())};
        return call(src.dptr(), sstep, planes, dstep, NppiSize{c.w, c.h}, (NppiColorSpace)cs, ctx);
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
        const Npp8u *p = src.hptr() + (size_t)y * src.step + (size_t)(x >> 1) * 5;
        uint64_t g = 0;
        for (int i = 0; i < 5; ++i) g |= (uint64_t)p[i] << (8 * i);
        const float u = (float)(g & 0x3FF);
        const float yv = (float)((g >> ((x & 1) ? 30 : 10)) & 0x3FF);
        const float v = (float)((g >> 20) & 0x3FF);
        float r, gg, b;
        decode10(cf, yv, u, v, r, gg, b);
        float outScale = 0.25f;
        if constexpr (std::is_same_v<T, Npp16u>) outScale = 64.06158447265625f;
        else if constexpr (std::is_same_v<T, Npp16f>) outScale = 1.0f;
        const T R = qout<T>(mul32(r, outScale)), G = qout<T>(mul32(gg, outScale)), B = qout<T>(mul32(b, outScale));
        if constexpr (Mode == 2) {
          const size_t off = (size_t)y * dstep + (size_t)x * sizeof(T);
          std::memcpy(w0.data() + off, &R, sizeof(T));
          std::memcpy(w1.data() + off, &G, sizeof(T));
          std::memcpy(w2.data() + off, &B, sizeof(T));
        } else {
          const size_t off = (size_t)y * dstep + (size_t)x * C * sizeof(T);
          std::memcpy(w0.data() + off, &R, sizeof(T));
          std::memcpy(w0.data() + off + sizeof(T), &G, sizeof(T));
          std::memcpy(w0.data() + off + 2 * sizeof(T), &B, sizeof(T));
          if constexpr (Mode == 1) std::memcpy(w0.data() + off + 3 * sizeof(T), &alpha, sizeof(T));
        }
      }
    size_t badOff = 0;
    bool bad = !within_one_lsb<T>(d.host, w0, badOff);
    if constexpr (Mode == 2)
      bad = bad || !within_one_lsb<T>(d1.host, w1, badOff) || !within_one_lsb<T>(d2.host, w2, badOff);
    if (bad) {
      ++cnt.compared;
      ++cnt.mismatched;
      std::fprintf(stderr, "[UYVP-DIFF] %s off=%zu cfg=%dx%d\n", id, badOff, c.w, c.h);
      if (first.empty()) first = std::string("[") + id + "] UYVP decode mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// RGB -> UYVP packed.  Mode: 0 packed C3, 1 planar C3 source, 2 packed AC4.
// ---------------------------------------------------------------------------
template <typename T, int Mode, class Call>
void run_uyvp_encode(const char *id, int cs, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
#ifdef USE_NVIDIA_NPP_TESTS
  if constexpr (std::is_same_v<T, Npp16f>) {
    GTEST_SKIP() << "[" << id << "] reference NPP 13.2.0.58 returns all-zero 16f output";
    return;
  }
#endif
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  const Coefs cf = coefs(cs);
  const int SC = Mode == 2 ? 4 : 3;
  const bool fp = std::is_same_v<T, Npp16f>;
  for (const Cfg &c : cfgs) {
    const int sstep = align16((size_t)c.w * SC * sizeof(T));
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<T> src(c.w, c.h, SC, 0, 0);
    const int pad = Mode == 1 ? sstep - align16((size_t)c.w * sizeof(T)) : 0;
    Buf<T> s1(c.w, c.h, 1, pad, 0), s2(c.w, c.h, 1, pad, 0);
    fill_regime(src, SC, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, fp);
    if constexpr (Mode == 1) {
      fill_regime(s1, 1, case_seed(id, c.w, c.h, 1, 0, 0, 1), kWide, 0, false, false, fp);
      fill_regime(s2, 1, case_seed(id, c.w, c.h, 2, 0, 0, 1), kWide, 0, false, false, fp);
    }
    src.dev.copyFromHost(src.host);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = [&]() {
      if constexpr (Mode == 1) {
        const T *planes[3] = {reinterpret_cast<const T *>(src.dptr()), reinterpret_cast<const T *>(s1.dptr()),
                              reinterpret_cast<const T *>(s2.dptr())};
        return call(planes, sstep, reinterpret_cast<Npp8u *>(d.dptr()), dstep, NppiSize{c.w, c.h},
                    (NppiColorSpace)cs, ctx);
      } else {
        (void)s1;
        (void)s2;
        return call(reinterpret_cast<T *>(src.dptr()), sstep, reinterpret_cast<Npp8u *>(d.dptr()), dstep,
                    NppiSize{c.w, c.h}, (NppiColorSpace)cs, ctx);
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
    auto px = [&](int ch, int x, int y) -> float {
      if constexpr (Mode == 1) {
        const Buf<T> *b = ch == 0 ? &src : (ch == 1 ? &s1 : &s2);
        return to10<T>((float)as_double(b->hptr()[(size_t)y * (b->step / (int)sizeof(T)) + (size_t)x]));
      }
      return to10<T>((float)as_double(src.hptr()[(size_t)y * (src.step / (int)sizeof(T)) + (size_t)x * SC + ch]));
    };
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; x += 2) {
        float y0, u0, v0, y1, u1, v1;
        encode10(cf, px(0, x, y), px(1, x, y), px(2, x, y), y0, u0, v0);
        encode10(cf, px(0, x + 1, y), px(1, x + 1, y), px(2, x + 1, y), y1, u1, v1);
        const float ra = mul32(add32(px(0, x, y), px(0, x + 1, y)), 0.5f);
        const float ga = mul32(add32(px(1, x, y), px(1, x + 1, y)), 0.5f);
        const float ba = mul32(add32(px(2, x, y), px(2, x + 1, y)), 0.5f);
        float dy, du, dv;
        encode10(cf, ra, ga, ba, dy, du, dv);
        uint64_t g = ((uint64_t)((uint32_t)q10f(du) & 0x3FF)) |
                     ((uint64_t)((uint32_t)q10f(y0) & 0x3FF) << 10) |
                     ((uint64_t)((uint32_t)q10f(dv) & 0x3FF) << 20) |
                     ((uint64_t)((uint32_t)q10f(y1) & 0x3FF) << 30);
        uint8_t *o = want.data() + (size_t)y * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    size_t badOff = 0;
    if (!within_one_lsb<T>(d.host, want, badOff)) {
      ++cnt.compared;
      ++cnt.mismatched;
      std::fprintf(stderr, "[UYVP-ENC-DIFF] %s off=%zu cfg=%dx%d\n", id, badOff, c.w, c.h);
      if (first.empty()) first = std::string("[") + id + "] UYVP encode mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// RGB 8u -> UYVP planar (Y per pixel, U/V per pair, 16-bit containers).
// ---------------------------------------------------------------------------
template <class Call>
void run_uyvp_encode_planar(const char *id, int cs, Call call) {
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
  for (const Cfg &c : cfgs) {
    const int sstep = align16((size_t)c.w * 3);
    const int dstep = align16((size_t)c.w * 2);
    Buf<Npp8u> src(c.w, c.h, 3, 0, 0);
    fill_regime(src, 3, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w, c.h, 2, 0, 0x5A), d1(c.w, c.h, 2, 0, 0x33), d2(c.w, c.h, 2, 0, 0x33);
    d.dev.copyFromHost(d.host);
    d1.dev.copyFromHost(d1.host);
    d2.dev.copyFromHost(d2.host);
    Npp8u *planes[3] = {d.dptr(), d1.dptr(), d2.dptr()};
    const NppStatus st = call(src.dptr(), sstep, planes, dstep, NppiSize{c.w, c.h}, (NppiColorSpace)cs, ctx);
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
    auto px = [&](int ch, int x, int y) -> float {
      return to10<Npp8u>((float)src.hptr()[(size_t)y * (src.step) + (size_t)x * 3 + ch]);
    };
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        float ly, lu, lv;
        encode10(cf, px(0, x, y), px(1, x, y), px(2, x, y), ly, lu, lv);
        const uint16_t Y = (uint16_t)(unsigned int)q10f(ly);
        std::memcpy(w0.data() + (size_t)y * dstep + (size_t)x * 2, &Y, 2);
        if ((x & 1) == 0) {
          auto clamp10 = [](float w) { return w < 0.0f ? 0.0f : (w > 1023.0f ? 1023.0f : w); };
          const float u0 = clamp10(fma32(px(0, x, y), cf.ur, fma32(px(1, x, y), cf.ug, fma32(px(2, x, y), cf.ub, 512.0f))));
          const float u1 = clamp10(fma32(px(0, x + 1, y), cf.ur, fma32(px(1, x + 1, y), cf.ug, fma32(px(2, x + 1, y), cf.ub, 512.0f))));
          const float v0 = clamp10(fma32(px(0, x, y), cf.vr, fma32(px(1, x, y), cf.vg, fma32(px(2, x, y), cf.vb, 512.0f))));
          const float v1 = clamp10(fma32(px(0, x + 1, y), cf.vr, fma32(px(1, x + 1, y), cf.vg, fma32(px(2, x + 1, y), cf.vb, 512.0f))));
          const uint16_t U = (uint16_t)(unsigned int)q10f(mul32(add32(u0, u1), 0.5f));
          const uint16_t V = (uint16_t)(unsigned int)q10f(mul32(add32(v0, v1), 0.5f));
          std::memcpy(w1.data() + (size_t)y * dstep + (size_t)(x >> 1) * 2, &U, 2);
          std::memcpy(w2.data() + (size_t)y * dstep + (size_t)(x >> 1) * 2, &V, 2);
        }
      }
    size_t badOff = 0;
    bool bad0 = !within_one_lsb<uint16_t>(d.host, w0, badOff);
    size_t off1 = 0, off2 = 0;
    bool bad1 = !within_one_lsb<uint16_t>(d1.host, w1, off1);
    bool bad2 = !within_one_lsb<uint16_t>(d2.host, w2, off2);
    bool bad = bad0 || bad1 || bad2;
    if (bad) {
      std::fprintf(stderr, "[UYVP-P3] %s cfg=%dx%d dstep=%d bad0=%d@%zu bad1=%d@%zu bad2=%d@%zu\n", id, c.w, c.h,
                   dstep, (int)bad0, badOff, (int)bad1, off1, (int)bad2, off2);
      auto show = [&](const char *nm, const std::vector<uint8_t> &got, const std::vector<uint8_t> &wnt,
                      size_t off) {
        std::fprintf(stderr, "  %s off=%zu got=", nm, off);
        for (int b = 0; b < 8; ++b) std::fprintf(stderr, "%u,", (unsigned)got[off + b]);
        std::fprintf(stderr, " want=");
        for (int b = 0; b < 8; ++b) std::fprintf(stderr, "%u,", (unsigned)wnt[off + b]);
        std::fprintf(stderr, "\n");
      };
      if (bad0) show("Y", d.host, w0, badOff);
      if (bad1) show("U", d1.host, w1, off1);
      if (bad2) show("V", d2.host, w2, off2);
    }
    if (bad) {
      ++cnt.compared;
      ++cnt.mismatched;
      if (first.empty()) first = std::string("[") + id + "] UYVP planar encode mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}

}  // namespace uyvp
}  // namespace num
