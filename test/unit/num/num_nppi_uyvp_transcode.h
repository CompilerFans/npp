// CPU-golden coverage for the NPP 13.4 UYVP_10u transcoders.
//
// These entry points are declared by the 13.4 headers but are not exported by
// the reference NPP 13.2.0.58 library, so every test resolves the symbol with
// dlsym at run time and skips when the reference does not provide it.  On the
// in-tree builds the symbols resolve and the full comparison runs.
//
// Layouts implemented (see nppi_uyvp_transcode.cu):
//   8-bit sources enter the 10-bit domain as x*4; P010/P210 16-bit containers
//   as x>>6 (MSB-aligned 10-bit); V210/Y410 are already 10-bit.
//   10-bit -> 8-bit output uses x0.25 with the +0.5f bias.
//   4:2:0 sources replicate chroma over the luma-row pair; 4:4:4 sources
//   average the horizontal pair; 4:4:4 outputs replicate the pair's chroma.
#pragma once

#include "num_nppi_geom.h"
#include <dlfcn.h>

namespace num {
namespace tc {

inline float q8f(float v) {
  if (v != v) return 0;
  const float t = v + 0.5f;
  return (float)(unsigned int)(t < 0.0f ? 0.0f : (t > 255.0f ? 255.0f : t));
}
inline float to10(float v) { return v * 4.0f; }
inline float from10(float v) { return q8f(v * 0.25f); }

template <typename Fn>
Fn resolve(const char *name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, name));
}

inline uint64_t pack(float u, float y0, float v, float y1) {
  return ((uint64_t)((uint32_t)u & 0x3FF)) | ((uint64_t)((uint32_t)y0 & 0x3FF) << 10) |
         ((uint64_t)((uint32_t)v & 0x3FF) << 20) | ((uint64_t)((uint32_t)y1 & 0x3FF) << 30);
}
inline void unpack(const uint8_t *p, float &u, float &y0, float &v, float &y1) {
  uint64_t g = 0;
  for (int i = 0; i < 5; ++i) g |= (uint64_t)p[i] << (8 * i);
  u = (float)(g & 0x3FF);
  y0 = (float)((g >> 10) & 0x3FF);
  v = (float)((g >> 20) & 0x3FF);
  y1 = (float)((g >> 30) & 0x3FF);
}

struct Cfg { int w, h; };
static const Cfg kCfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
static const Cfg kV210Cfgs[] = {{6, 2}, {12, 4}, {24, 8}, {48, 16}, {96, 8}, {384, 6}};

inline bool diff_ok(const char *id, const std::vector<uint8_t> &got, const std::vector<uint8_t> &want) {
  if (got.size() != want.size()) {
    std::fprintf(stderr, "[TC-DIFF] %s size got=%zu want=%zu\n", id, got.size(), want.size());
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i)
    if (std::abs((int)got[i] - (int)want[i]) > 1) {
      std::fprintf(stderr, "[TC-DIFF] %s off=%zu got=%u want=%u\n", id, i, (unsigned)got[i], (unsigned)want[i]);
      return false;
    }
  return true;
}

// ---------------------------------------------------------------------------
// Semi-planar 4:2:0 / 4:2:2 -> UYVP packed (NV12 / NV16 / P010 / P210).
// ---------------------------------------------------------------------------
template <typename T, bool Wide, bool Vert>
void run_semiplanar_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const T *const *, const int *, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)c.w * sizeof(T));
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<T> y(c.w, Vert ? c.h + c.h / 2 : c.h * 2, 1, 0, 0);
    fill_regime(y, 1, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    y.dev.copyFromHost(y.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const int se = y.step / (int)sizeof(T);
    const T *py = reinterpret_cast<const T *>(y.dptr());
    const T *puv = py + (size_t)(Vert ? c.h : c.h) * se;
    const T *planes[2] = {py, puv};
    const int steps[2] = {sstep, sstep};
    const NppStatus st = fn(planes, steps, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    auto cvt = [](T v) -> float {
      if constexpr (Wide) return (float)((uint32_t)v >> 6);
      else return to10((float)v);
    };
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        const int cy = Vert ? (yy >> 1) : yy;
        const T *uv = y.hptr() + (size_t)(c.h + cy) * se + (size_t)x;
        const uint64_t g = pack(cvt(uv[0]), cvt(y.hptr()[(size_t)yy * se + x]), cvt(uv[1]),
                                cvt(y.hptr()[(size_t)yy * se + x + 1]));
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.compared;
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// P010 with three planar inputs -> UYVP (4:2:0).
inline void run_p010_p3_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp16u *const *, const int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)c.w * 2);
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<Npp16u> yp(c.w, c.h, 1, 0, 0);
    // All three planes share the API's single nSrcStep.
    const int cpad = sstep - align16((size_t)(c.w / 2) * 2);
    Buf<Npp16u> up(c.w / 2, c.h / 2, 1, cpad, 0), vp(c.w / 2, c.h / 2, 1, cpad, 0);
    fill_regime(yp, 1, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    fill_regime(up, 1, case_seed(id, c.w, c.h, 1, 0, 0, 0), kWide, 0, false, false, false);
    fill_regime(vp, 1, case_seed(id, c.w, c.h, 2, 0, 0, 0), kWide, 0, false, false, false);
    yp.dev.copyFromHost(yp.host);
    up.dev.copyFromHost(up.host);
    vp.dev.copyFromHost(vp.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const Npp16u *planes[3] = {reinterpret_cast<const Npp16u *>(yp.dptr()),
                               reinterpret_cast<const Npp16u *>(up.dptr()),
                               reinterpret_cast<const Npp16u *>(vp.dptr())};
    const NppStatus st = fn(planes, sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    const int yse = yp.step / 2, cse = up.step / 2;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        const uint64_t g = pack((float)(up.hptr()[(size_t)(yy / 2) * cse + x / 2] >> 6),
                                (float)(yp.hptr()[(size_t)yy * yse + x] >> 6),
                                (float)(vp.hptr()[(size_t)(yy / 2) * cse + x / 2] >> 6),
                                (float)(yp.hptr()[(size_t)yy * yse + x + 1] >> 6));
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// P208 (packed UYVY 8u) -> UYVP.
inline void run_p208_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)c.w * 2);
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<Npp8u> src(c.w / 2, c.h, 4, 0, 0);
    fill_regime(src, 4, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        const Npp8u *p = src.hptr() + (size_t)yy * sstep + (size_t)x * 2;
        const uint64_t g = pack(to10((float)p[0]), to10((float)p[1]), to10((float)p[2]), to10((float)p[3]));
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// I444 (planar 4:4:4 8u) -> UYVP.
inline void run_i444_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *const *, const int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)c.w);
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<Npp8u> py(c.w, c.h, 1, 0, 0), pu(c.w, c.h, 1, 0, 0), pv(c.w, c.h, 1, 0, 0);
    fill_regime(py, 1, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    fill_regime(pu, 1, case_seed(id, c.w, c.h, 1, 0, 0, 0), kWide, 0, false, false, false);
    fill_regime(pv, 1, case_seed(id, c.w, c.h, 2, 0, 0, 0), kWide, 0, false, false, false);
    py.dev.copyFromHost(py.host);
    pu.dev.copyFromHost(pu.host);
    pv.dev.copyFromHost(pv.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const Npp8u *planes[3] = {py.dptr(), pu.dptr(), pv.dptr()};
    const NppStatus st = fn(planes, sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        const size_t i0 = (size_t)yy * sstep + x;
        const uint64_t g = pack((to10((float)pu.hptr()[i0]) + to10((float)pu.hptr()[i0 + 1])) * 0.5f,
                                to10((float)py.hptr()[i0]),
                                (to10((float)pv.hptr()[i0]) + to10((float)pv.hptr()[i0 + 1])) * 0.5f,
                                to10((float)py.hptr()[i0 + 1]));
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// Packed YUV444 8u -> UYVP.
inline void run_yuv444_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)c.w * 3);
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<Npp8u> src(c.w, c.h, 3, 0, 0);
    fill_regime(src, 3, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        const size_t i0 = (size_t)yy * sstep + (size_t)x * 3;
        const uint64_t g = pack((to10((float)src.hptr()[i0 + 1]) + to10((float)src.hptr()[i0 + 4])) * 0.5f,
                                to10((float)src.hptr()[i0]),
                                (to10((float)src.hptr()[i0 + 2]) + to10((float)src.hptr()[i0 + 5])) * 0.5f,
                                to10((float)src.hptr()[i0 + 3]));
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// V210 -> UYVP and back.
inline void run_v210_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kV210Cfgs) {
    const int sstep = align16((size_t)(c.w / 6) * 16);
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<Npp8u> src(c.w / 6, c.h, 16, 0, 0);
    fill_regime(src, 16, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int g = 0; g < c.w / 6; ++g) {
        const uint32_t *w = reinterpret_cast<const uint32_t *>(src.hptr() + (size_t)yy * sstep + (size_t)g * 16);
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
        const uint64_t ga = pack((float)(w0 & 0x3FF), (float)((w0 >> 10) & 0x3FF), (float)((w0 >> 20) & 0x3FF),
                                 (float)(w1 & 0x3FF));
        const uint64_t gb = pack((float)((w1 >> 10) & 0x3FF), (float)((w1 >> 20) & 0x3FF),
                                 (float)(w2 & 0x3FF), (float)((w2 >> 10) & 0x3FF));
        const uint64_t gc = pack((float)((w2 >> 20) & 0x3FF), (float)(w3 & 0x3FF),
                                 (float)((w3 >> 10) & 0x3FF), (float)((w3 >> 20) & 0x3FF));
        const uint64_t arr[3] = {ga, gb, gc};
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)g * 15;
        for (int k = 0; k < 3; ++k)
          for (int i = 0; i < 5; ++i) o[k * 5 + i] = (uint8_t)((arr[k] >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

inline void run_uyvp_to_v210(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kV210Cfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)(c.w / 6) * 16);
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w / 6, c.h, 16, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int g = 0; g < c.w / 6; ++g) {
        const uint8_t *p = src.hptr() + (size_t)yy * sstep + (size_t)g * 15;
        float u0, y0, v0, y1, u1, y2, v1, y3, u2, y4, v2, y5;
        unpack(p, u0, y0, v0, y1);
        unpack(p + 5, u1, y2, v1, y3);
        unpack(p + 10, u2, y4, v2, y5);
        uint32_t *w = reinterpret_cast<uint32_t *>(want.data() + (size_t)yy * dstep + (size_t)g * 16);
        w[0] = ((uint32_t)u0 & 0x3FF) | (((uint32_t)y0 & 0x3FF) << 10) | (((uint32_t)v0 & 0x3FF) << 20);
        w[1] = ((uint32_t)y1 & 0x3FF) | (((uint32_t)u1 & 0x3FF) << 10) | (((uint32_t)y2 & 0x3FF) << 20);
        w[2] = ((uint32_t)v1 & 0x3FF) | (((uint32_t)y3 & 0x3FF) << 10) | (((uint32_t)u2 & 0x3FF) << 20);
        w[3] = ((uint32_t)y4 & 0x3FF) | (((uint32_t)v2 & 0x3FF) << 10) | (((uint32_t)y5 & 0x3FF) << 20);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// Y410 <-> UYVP.
inline void run_y410_to_uyvp(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp16u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)c.w * 4);
    const int dstep = align16((size_t)(c.w / 2) * 5);
    Buf<Npp16u> src(c.w * 2, c.h, 1, 0, 0);
    fill_regime(src, 1, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w / 2, c.h, 5, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(reinterpret_cast<const Npp16u *>(src.dptr()), sstep, NppiPoint{0, 0}, d.dptr(), dstep,
                            NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        const uint32_t *w = reinterpret_cast<const uint32_t *>(src.hptr() + (size_t)yy * (sstep / 2) + (size_t)x * 2);
        const float u0 = (float)(w[0] & 0x3FF), y0 = (float)((w[0] >> 10) & 0x3FF), v0 = (float)((w[0] >> 20) & 0x3FF);
        const float y1 = (float)((w[1] >> 10) & 0x3FF);
        const uint64_t g = pack(u0, y0, v0, y1);
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)(x >> 1) * 5;
        for (int i = 0; i < 5; ++i) o[i] = (uint8_t)((g >> (8 * i)) & 0xFF);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

inline void run_uyvp_to_y410(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)c.w * 4);
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w * 2, c.h, 2, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; ++x) {
        float u, y0, v, y1;
        unpack(src.hptr() + (size_t)yy * sstep + (size_t)(x >> 1) * 5, u, y0, v, y1);
        const float yv = (x & 1) ? y1 : y0;
        uint32_t *w = reinterpret_cast<uint32_t *>(want.data() + (size_t)yy * dstep);
        w[x] = ((uint32_t)u & 0x3FF) | (((uint32_t)yv & 0x3FF) << 10) | (((uint32_t)v & 0x3FF) << 20) | (3u << 30);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

// UYVP -> I444 / NV16 / P208 / YUV444.
inline void run_uyvp_to_i444(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *const *, const int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)c.w);
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> dy(c.w, c.h, 1, 0, 0x5A), du(c.w, c.h, 1, 0, 0x33), dv(c.w, c.h, 1, 0, 0x33);
    dy.dev.copyFromHost(dy.host);
    du.dev.copyFromHost(du.host);
    dv.dev.copyFromHost(dv.host);
    Npp8u *planes[3] = {dy.dptr(), du.dptr(), dv.dptr()};
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, planes, dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    dy.dev.copyToHost(dy.host);
    du.dev.copyToHost(du.host);
    dv.dev.copyToHost(dv.host);
    ++cnt.executed;
    std::vector<uint8_t> wy = dy.host, wu = du.host, wv = dv.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; ++x) {
        float u, y0, v, y1;
        unpack(src.hptr() + (size_t)yy * sstep + (size_t)(x >> 1) * 5, u, y0, v, y1);
        const float yv = (x & 1) ? y1 : y0;
        wy[(size_t)yy * dstep + x] = (uint8_t)(unsigned int)from10(yv);
        wu[(size_t)yy * dstep + x] = (uint8_t)(unsigned int)from10(u);
        wv[(size_t)yy * dstep + x] = (uint8_t)(unsigned int)from10(v);
      }
    bool bad = !diff_ok(id, dy.host, wy) || !diff_ok(id, du.host, wu) || !diff_ok(id, dv.host, wv);
    if (bad) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

inline void run_uyvp_to_nv16(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *const, const int, NppiPoint, Npp8u **, const int *, NppiSize,
                           NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)c.w);
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> dy(c.w, c.h, 1, 0, 0x5A), duv(c.w / 2, c.h, 2, 0, 0x33);
    dy.dev.copyFromHost(dy.host);
    duv.dev.copyFromHost(duv.host);
    Npp8u *planes[2] = {dy.dptr(), duv.dptr()};
    const int steps[2] = {dstep, dstep};
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, planes, steps, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    dy.dev.copyToHost(dy.host);
    duv.dev.copyToHost(duv.host);
    ++cnt.executed;
    std::vector<uint8_t> wy = dy.host, wuv = duv.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; ++x) {
        float u, y0, v, y1;
        unpack(src.hptr() + (size_t)yy * sstep + (size_t)(x >> 1) * 5, u, y0, v, y1);
        const float yv = (x & 1) ? y1 : y0;
        wy[(size_t)yy * dstep + x] = (uint8_t)(unsigned int)from10(yv);
        if ((x & 1) == 0) {
          wuv[(size_t)yy * dstep + x] = (uint8_t)(unsigned int)from10(u);
          wuv[(size_t)yy * dstep + x + 1] = (uint8_t)(unsigned int)from10(v);
        }
      }
    if (!diff_ok(id, dy.host, wy) || !diff_ok(id, duv.host, wuv)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

inline void run_uyvp_to_p208(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)c.w * 2);
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w / 2, c.h, 4, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; x += 2) {
        float u, y0, v, y1;
        unpack(src.hptr() + (size_t)yy * sstep + (size_t)(x >> 1) * 5, u, y0, v, y1);
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)x * 2;
        o[0] = (uint8_t)(unsigned int)from10(u);
        o[1] = (uint8_t)(unsigned int)from10(y0);
        o[2] = (uint8_t)(unsigned int)from10(v);
        o[3] = (uint8_t)(unsigned int)from10(y1);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

inline void run_uyvp_to_yuv444(const char *id, const char *name) {
  using Fn = NppStatus (*)(const Npp8u *, int, NppiPoint, Npp8u *, int, NppiSize, NppStreamContext);
  Fn fn = resolve<Fn>(name);
  if (!fn) {
    GTEST_SKIP() << "[" << id << "] not exported by the reference NPP";
    return;
  }
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  for (const Cfg &c : kCfgs) {
    const int sstep = align16((size_t)(c.w / 2) * 5);
    const int dstep = align16((size_t)c.w * 3);
    Buf<Npp8u> src(c.w / 2, c.h, 5, 0, 0);
    fill_regime(src, 5, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, false);
    src.dev.copyFromHost(src.host);
    Buf<Npp8u> d(c.w, c.h, 3, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const NppStatus st = fn(src.dptr(), sstep, NppiPoint{0, 0}, d.dptr(), dstep, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      continue;
    }
    cudaDeviceSynchronize();
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    std::vector<uint8_t> want = d.host;
    for (int yy = 0; yy < c.h; ++yy)
      for (int x = 0; x < c.w; ++x) {
        float u, y0, v, y1;
        unpack(src.hptr() + (size_t)yy * sstep + (size_t)(x >> 1) * 5, u, y0, v, y1);
        const float yv = (x & 1) ? y1 : y0;
        uint8_t *o = want.data() + (size_t)yy * dstep + (size_t)x * 3;
        o[0] = (uint8_t)(unsigned int)from10(yv);
        o[1] = (uint8_t)(unsigned int)from10(u);
        o[2] = (uint8_t)(unsigned int)from10(v);
      }
    if (!diff_ok(id, d.host, want)) {
      ++cnt.mismatched;
      EXPECT_TRUE(false) << "[" << id << "] transcode mismatch";
      break;
    }
  }
  finish(id, cnt, std::string());
}

}  // namespace tc
}  // namespace num
