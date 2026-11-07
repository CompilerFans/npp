// CPU-reference output coverage for the NPP 13.4 P010 / P016 to RGB entry points.
//
// Both layouts parse as 16-bit semi-planar (Y plane, interleaved U/V pairs at
// half resolution).  NPP normalises each sample into its 10-bit conversion
// domain (x * 1023/65535), decodes with chroma centred on 512 and scales the
// result to the destination type (x0.25 for 8-bit output, x65535/1023 for
// 16-bit output), then rounds with the +0.5f bias.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace p010 {

struct Coefs {
  float rv, gu, gv, bu;
};

inline Coefs coefs(int cs) {
  if (cs == 1)
    return {1.1398299932479858398f, -0.39465001225471496582f, -0.58060002326965332031f,
            2.0321099758148193359f};
  if (cs == 2)
    return {1.5748000144958496094f, -0.18732400238513946533f, -0.46812400221824645996f,
            1.8555999994277954102f};
  return {1.4745999574661254883f, -0.16455310583114624023f, -0.57135307788848876953f,
          1.881399989128112793f};
}

inline float fma32(float a, float b, float c) { return std::fma(a, b, c); }
inline float mul32(float a, float b) { return std::fma(a, b, 0.0f); }
inline float add32(float a, float b) { return std::fma(a, 1.0f, b); }

template <typename T>
inline T qv(float v) {
  if (v != v) return 0;
  float t = add32(v, 0.5f);
  const float mx = std::is_same_v<T, Npp8u> ? 255.0f: 65535.0f;
  t = t < 0.0f ? 0.0f: (t > mx ? mx: t);
  return (T)(unsigned int)t;
}

inline void decode(const Coefs &c, float y16, float u16, float v16, bool wide, float &r, float &g, float &b) {
  const float S = 0.015609979629516601562f;
  const float O = wide ? 64.06158447265625f: 0.25f;
  const float yy = mul32(y16, S);
  const float du = fma32(u16, S, -512.0f);
  const float dv = fma32(v16, S, -512.0f);
  r = mul32(fma32(dv, c.rv, yy), O);
  g = mul32(fma32(du, c.gu, fma32(dv, c.gv, yy)), O);
  b = mul32(fma32(du, c.bu, yy), O);
}

// The reference NPP 13.2.0.58 build renders these entry points through the
// same 16-bit-stride format-conversion kernel it wrongly uses for the 8-bit
// NV12 extended conversions: a constant image must produce identical pixels,
// but the reference alternates values.  Skip the comparison on the reference
// build only.
template <class Call>
bool reference_sane(int dstC, Call call) {
  const int W = 4, H = 2;
  Npp16u *dY = nullptr, *dUV = nullptr, *dOut = nullptr;
  cudaMalloc((void **)&dY, (size_t)W * H * 2);
  cudaMalloc((void **)&dUV, (size_t)W * H * 4);
  cudaMalloc((void **)&dOut, (size_t)W * H * 8 * 2);
  std::vector<Npp16u> y((size_t)W * H, 20000), uv((size_t)W * H * 2, 12000);
  for (size_t i = 1; i < uv.size(); i += 2) uv[i] = 30000;
  cudaMemcpy(dY, y.data(), y.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dUV, uv.data(), uv.size() * 2, cudaMemcpyHostToDevice);
  NppStreamContext ctx{};
  bool sane = true;
  if (nppGetStreamContext(&ctx) == NPP_NO_ERROR) {
    const Npp16u *planes[2] = {dY, dUV};
    NppStatus st = call(planes, W * 2, dOut, W * dstC * 2, NppiSize{W, H}, ctx);
    if (st == NPP_NO_ERROR) {
      cudaDeviceSynchronize();
      std::vector<Npp8u> o((size_t)W * H * dstC * 2);
      cudaMemcpy(o.data(), dOut, o.size(), cudaMemcpyDeviceToHost);
      for (int x = 1; x < W; ++x)
        for (int b = 0; b < dstC * 2; ++b)
          if (o[(size_t)b] != o[(size_t)x * dstC * 2 + b]) sane = false;
    }
  }
  cudaFree(dY);
  cudaFree(dUV);
  cudaFree(dOut);
  return sane;
}

template <typename T, bool C4, class Call>
void run_p010(const char *id, int cs, T alpha, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
#ifdef USE_NVIDIA_NPP_TESTS
  if (!reference_sane(C4 ? 4: 3, [&](const Npp16u *const *sp, int ss, void *d, int ds, NppiSize r,
                                      NppStreamContext c) {
        if constexpr (C4)
          return call(sp, ss, (T *)d, ds, r, (NppiColorSpace)cs, alpha, c);
        else
          return call(sp, ss, (T *)d, ds, r, (NppiColorSpace)cs, c);
      })) {
    GTEST_SKIP() << "[" << id << "] reference NPP 13.2.0.58 extended kernel is broken "
                    "(16-bit sample addressing applied to the wrong plane layout)";
    return;
  }
#endif
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {32, 24}, {64, 48}, {16, 12}, {128, 96}};
  const Coefs cf = coefs(cs);
  const bool wide = std::is_same_v<T, Npp16u>;
  const int C = C4 ? 4: 3;
  for (const Cfg &c: cfgs) {
    const int sstep = align16((size_t)c.w * 2);
    const int dstep = align16((size_t)c.w * C * sizeof(T));
    // Semi-planar source: H luma rows followed by H/2 interleaved UV rows.
    Buf<Npp16u> src(c.w, c.h + c.h / 2, 1, 0, 0);
    fill_regime(src, 1, case_seed(id, c.w, c.h, 0, 0, 0, 0), kWide, 0, false, false, true);
    src.dev.copyFromHost(src.host);
    Buf<T> d(c.w, c.h, C, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    const int sElemL = src.step / (int)sizeof(Npp16u);
    const Npp16u *dY = reinterpret_cast<const Npp16u *>(src.dptr());
    const Npp16u *dUV = dY + (size_t)c.h * sElemL;
    const Npp16u *planes[2] = {dY, dUV};
    const NppStatus st = [&]() {
      if constexpr (C4)
        return call(planes, sstep, reinterpret_cast<T *>(d.dptr()), dstep, NppiSize{c.w, c.h},
                    (NppiColorSpace)cs, alpha, ctx);
      else
        return call(planes, sstep, reinterpret_cast<T *>(d.dptr()), dstep, NppiSize{c.w, c.h},
                    (NppiColorSpace)cs, ctx);
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
    const int sElem = src.step / 2;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const Npp16u *sp = src.hptr();
        const int yv = sp[(size_t)y * sElem + x];
        const size_t uvi = (size_t)(c.h + y / 2) * sElem + (size_t)(x / 2) * 2;
        float r, g, b;
        decode(cf, (float)yv, (float)sp[uvi], (float)sp[uvi + 1], wide, r, g, b);
        const T R = qv<T>(r), G = qv<T>(g), B = qv<T>(b);
        const size_t off = (size_t)y * dstep + (size_t)x * C * sizeof(T);
        std::memcpy(want.data() + off, &R, sizeof(T));
        std::memcpy(want.data() + off + sizeof(T), &G, sizeof(T));
        std::memcpy(want.data() + off + 2 * sizeof(T), &B, sizeof(T));
        if constexpr (C4) std::memcpy(want.data() + off + 3 * sizeof(T), &alpha, sizeof(T));
      }
    size_t badOff = 0;
    bool bad = false;
    if (d.host.size() != want.size()) {
      bad = true;
    } else {
      for (size_t i = 0; i + sizeof(T) <= d.host.size(); i += sizeof(T)) {
        T got, wnt;
        std::memcpy(&got, d.host.data() + i, sizeof(T));
        std::memcpy(&wnt, want.data() + i, sizeof(T));
        const long long diff = (long long)got - (long long)wnt;
        if (diff < -1 || diff > 1) {
          badOff = i;
          bad = true;
          break;
        }
      }
    }
    if (bad) {
      ++cnt.compared;
      ++cnt.mismatched;
      std::fprintf(stderr, "[P010-DIFF] %s off=%zu cfg=%dx%d\n", id, badOff, c.w, c.h);
      if (first.empty()) first = std::string("[") + id + "] P010/P016 decode mismatch";
      EXPECT_TRUE(false) << first;
      break;
    }
  }
  finish(id, cnt, first);
}

}  // namespace p010
}  // namespace num
