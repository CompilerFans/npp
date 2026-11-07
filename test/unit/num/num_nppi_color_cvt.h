// CPU-reference output coverage for the 8u gamma transforms and the RGB/color-to-gray
// and gradient-to-gray families.
//
// The gamma tables below are the 256-entry LUTs measured from NVIDIA NPP 13.4
// (identical to the in-tree device tables).  Gray conversions use the
// BT.601 luminance weights for nppiRGBToGray and caller-supplied coefficients
// for nppiColorToGray; gradient-to-gray evaluates L1 / L2 / Inf norms.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace cvt {

static const unsigned char kGammaFwd[256] = {
      0,   4,   9,  13,  18,  22,  26,  30,  33,  36,  40,  42,  45,  48,  50,  53,
     55,  57,  59,  61,  63,  65,  67,  69,  71,  73,  75,  76,  78,  80,  81,  83,
     84,  86,  87,  89,  90,  92,  93,  95,  96,  97,  99, 100, 101, 103, 104, 105,
    106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 151, 152, 153, 154,
    155, 156, 156, 157, 158, 159, 160, 161, 161, 162, 163, 164, 165, 165, 166, 167,
    168, 169, 169, 170, 171, 172, 172, 173, 174, 175, 175, 176, 177, 178, 178, 179,
    180, 180, 181, 182, 183, 183, 184, 185, 185, 186, 187, 188, 188, 189, 190, 190,
    191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 198, 198, 199, 200, 200, 201,
    201, 202, 203, 203, 204, 205, 205, 206, 207, 207, 208, 208, 209, 210, 210, 211,
    211, 212, 213, 213, 214, 214, 215, 216, 216, 217, 217, 218, 219, 219, 220, 220,
    221, 221, 222, 223, 223, 224, 224, 225, 225, 226, 227, 227, 228, 228, 229, 229,
    230, 231, 231, 232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 238, 238,
    239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246,
    247, 247, 248, 248, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255,
};

static const unsigned char kGammaInv[256] = {
      0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   2,   2,   2,   2,   3,   3,
      3,   3,   3,   4,   4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   7,
      7,   7,   8,   8,   8,   9,   9,   9,  10,  10,  10,  11,  11,  11,  12,  12,
     12,  13,  13,  14,  14,  15,  15,  15,  16,  16,  17,  17,  18,  18,  19,  19,
     20,  20,  21,  21,  22,  22,  23,  23,  24,  24,  25,  26,  26,  27,  27,  28,
     28,  29,  30,  30,  31,  32,  32,  33,  34,  34,  35,  36,  36,  37,  38,  38,
     39,  40,  41,  41,  42,  43,  44,  44,  45,  46,  47,  48,  48,  49,  50,  51,
     52,  53,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  62,  63,  64,  65,
     66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  81,  82,
     83,  84,  85,  86,  87,  88,  89,  91,  92,  93,  94,  95,  96,  98,  99, 100,
    101, 102, 104, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 118, 119, 120,
    122, 123, 124, 126, 127, 129, 130, 131, 133, 134, 136, 137, 139, 140, 141, 143,
    144, 146, 147, 149, 150, 152, 153, 155, 157, 158, 160, 161, 163, 164, 166, 168,
    169, 171, 172, 174, 176, 177, 179, 181, 182, 184, 186, 187, 189, 191, 193, 194,
    196, 198, 200, 201, 203, 205, 207, 209, 210, 212, 214, 216, 218, 220, 221, 223,
    225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 246, 248, 250, 252, 255,
};

template <typename S>
inline S round_clamp(double v) {
  if constexpr (std::is_same_v<S, Npp32f>) {
    return (S)v;
  } else {
    double lo = 0.0, hi = 0.0;
    if constexpr (std::is_same_v<S, Npp8u>) { lo = 0.0; hi = 255.0; }
    else if constexpr (std::is_same_v<S, Npp16u>) { lo = 0.0; hi = 65535.0; }
    else if constexpr (std::is_same_v<S, Npp16s>) { lo = -32768.0; hi = 32767.0; }
    const double r = v >= 0.0 ? std::floor(v + 0.5): std::ceil(v - 0.5);
    if (r < lo) return (S)lo;
    if (r > hi) return (S)hi;
    return (S)r;
  }
}

template <typename S>
inline S trunc_clamp(double v) {
  if constexpr (std::is_same_v<S, Npp32f>) {
    return (S)v;
  } else {
    double lo = 0.0, hi = 0.0;
    if constexpr (std::is_same_v<S, Npp8u>) { lo = 0.0; hi = 255.0; }
    else if constexpr (std::is_same_v<S, Npp16u>) { lo = 0.0; hi = 65535.0; }
    else if constexpr (std::is_same_v<S, Npp16s>) { lo = -32768.0; hi = 32767.0; }
    const double r = std::trunc(v);
    if (r < lo) return (S)lo;
    if (r > hi) return (S)hi;
    return (S)r;
  }
}

// ---- gamma -----------------------------------------------------------------
// layout: 0 = C3 packed, 1 = AC4 packed, 2 = planar P3.
// The packed C3 variants write the three RGB channels; the AC4 variants
// (measured on NPP 13.4) leave the destination alpha untouched.
template <int Layout, bool InPlace, bool Fwd, class Call>
void run_gamma(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const unsigned char *lut = Fwd ? kGammaFwd: kGammaInv;
  for (const GenCase &c: make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    if constexpr (Layout == 2) {
      std::vector<Buf<Npp8u> *> srcs;
      const uint64_t seed = case_seed(id, c.w, c.h, c.pad, 0, (int)c.regime, 0);
      planar_fill_src(srcs, 3, c.w, c.h, c.ox, c.oy, c.pad, seed, c.regime);
      std::vector<Buf<Npp8u> *> dsts;
      std::vector<std::vector<uint8_t>> pre(3), orig(3);
      for (int p = 0; p < 3; ++p) {
        Buf<Npp8u> *d = new Buf<Npp8u>(c.w, c.h, 1, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        dsts.push_back(d);
        pre[(size_t)p] = d->host;
        orig[(size_t)p] = srcs[(size_t)p]->host;
      }
      const Npp8u *hsrc[4] = {nullptr, nullptr, nullptr, nullptr};
      const Npp8u *dsrc[4] = {nullptr, nullptr, nullptr, nullptr};
      Npp8u *ddst[4] = {nullptr, nullptr, nullptr, nullptr};
      Npp8u *msrc[4] = {nullptr, nullptr, nullptr, nullptr};
      int sstep[4] = {0, 0, 0, 0};
      planar_ptrs(srcs, dsts, 3, c.ox, c.oy, hsrc, dsrc, ddst, sstep);
      for (int p = 0; p < 3; ++p)
        msrc[p] = reinterpret_cast<Npp8u *>(srcs[(size_t)p]->dptr() +
                                            off_bytes<Npp8u>(c.oy, c.ox, srcs[(size_t)p]->step, 1));
      if constexpr (InPlace) {
        for (int p = 0; p < 3; ++p) {
          srcs[(size_t)p]->host = orig[(size_t)p];
          srcs[(size_t)p]->dev.copyFromHost(srcs[(size_t)p]->host);
        }
        npp_diff::clear_cuda_error();
        const NppStatus st = call(msrc, srcs[0]->step, NppiSize{c.w, c.h}, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          for (auto *p: srcs) delete p;
          for (auto *p: dsts) delete p;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
        ++cnt.executed;
        bool failed = false;
        for (int p = 0; p < 3 && !failed; ++p) {
          std::vector<uint8_t> got;
          srcs[(size_t)p]->dev.copyToHost(got);
          std::vector<uint8_t> want = orig[(size_t)p];
          for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
              const size_t so = (size_t)y * (srcs[(size_t)p]->step) + (size_t)x;
              const size_t db = off_bytes<Npp8u>(c.oy + y, c.ox + x, srcs[(size_t)p]->step, 1);
              want[db] = lut[hsrc[p][so]];
            }
          const GenCase dc{c.w, c.h, c.ox, c.oy, 0, kSmall, 0};
          if (!rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, orig[(size_t)p], got, want,
                                         srcs[(size_t)p]->step, std::vector<uint8_t>(), cnt, first))
            failed = true;
        }
        for (auto *p: srcs) delete p;
        for (auto *p: dsts) delete p;
        if (failed || cnt.mismatched >= 8) break;
      } else {
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, srcs[0]->step, ddst, dsts[0]->step, NppiSize{c.w, c.h}, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          for (auto *p: srcs) delete p;
          for (auto *p: dsts) delete p;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
        ++cnt.executed;
        bool failed = false;
        for (int p = 0; p < 3 && !failed; ++p) {
          dsts[(size_t)p]->dev.copyToHost(dsts[(size_t)p]->host);
          std::vector<uint8_t> want = pre[(size_t)p];
          for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
              const size_t so = (size_t)y * (srcs[(size_t)p]->step) + (size_t)x;
              const size_t db = off_bytes<Npp8u>(y, x, dsts[(size_t)p]->step, 1);
              want[db] = lut[hsrc[p][so]];
            }
          const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
          if (!rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre[(size_t)p], dsts[(size_t)p]->host,
                                         want, dsts[(size_t)p]->step, std::vector<uint8_t>(), cnt, first))
            failed = true;
        }
        for (auto *p: srcs) delete p;
        for (auto *p: dsts) delete p;
        if (failed || cnt.mismatched >= 8) break;
      }
    } else {
      constexpr int CH = Layout == 0 ? 3: 4;
      Buf<Npp8u> s(c.w + c.ox, c.h + c.oy, CH, c.pad, 0);
      Buf<Npp8u> d(c.w, c.h, CH, 0, 0x5A);
      fill_regime(s, CH, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false, false);
      const std::vector<uint8_t> orig = s.host;
      s.dev.copyFromHost(s.host);
      d.dev.copyFromHost(d.host);
      const Npp8u *hsrc = s.hptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, CH);
      const Npp8u *dsrc = s.dptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, CH);
      if constexpr (InPlace) {
        npp_diff::clear_cuda_error();
        const NppStatus st = call(reinterpret_cast<Npp8u *>(s.dptr() + off_bytes<Npp8u>(c.oy, c.ox, s.step, CH)),
                                  s.step, NppiSize{c.w, c.h}, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
        ++cnt.executed;
        std::vector<uint8_t> got;
        s.dev.copyToHost(got);
        std::vector<uint8_t> want = orig;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            const size_t so = (size_t)y * (s.step) + (size_t)x * CH;
            const size_t db = off_bytes<Npp8u>(c.oy + y, c.ox + x, s.step, CH);
            for (int ch = 0; ch < 3; ++ch) want[db + (size_t)ch] = lut[hsrc[so + (size_t)ch]];
          }
        const GenCase dc{c.w, c.h, c.ox, c.oy, 0, kSmall, 0};
        if (!rotate_check<Npp8u, CH, 3>(id, dc, NPPI_INTER_NN, orig, got, want, s.step,
                                        std::vector<uint8_t>(), cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
      } else {
        const std::vector<uint8_t> pre = d.host;
        npp_diff::clear_cuda_error();
        const NppStatus st = call(dsrc, s.step, reinterpret_cast<Npp8u *>(d.dptr()), d.step, NppiSize{c.w, c.h}, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        const cudaError_t se = cudaDeviceSynchronize();
        ASSERT_EQ(se, cudaSuccess) << "[" << id << "] sync: " << cudaGetErrorString(se);
        d.dev.copyToHost(d.host);
        ++cnt.executed;
        std::vector<uint8_t> want = pre;
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            const size_t so = (size_t)y * (s.step) + (size_t)x * CH;
            const size_t db = off_bytes<Npp8u>(y, x, d.step, CH);
            for (int ch = 0; ch < 3; ++ch) want[db + (size_t)ch] = lut[hsrc[so + (size_t)ch]];
          }
        const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<Npp8u, CH, 3>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                        std::vector<uint8_t>(), cnt, first)) {
          if (cnt.mismatched >= 8) break;
        }
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---- RGB / color to gray ---------------------------------------------------
// Layout: 0 = C3 source, 1 = C4 source (four coefficients incl. alpha),
// 2 = AC4 source (three coefficients, alpha ignored).  Destination is C1.
template <typename S, int Layout, bool FixedWeights, class Call>
void run_color_to_gray(const char *id, Call call, int ncoef) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  static const float coeffSets[2][4] = {
      {0.299f, 0.587f, 0.114f, 0.0f},
      {0.25f, 0.5f, 0.25f, 0.1f},
  };
  constexpr int CH = Layout == 0 ? 3: 4;
  for (const GenCase &c: make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, CH, c.pad, 0);
    Buf<S> d(c.w, c.h, 1, 0, 0x5A);
    fill_regime(s, CH, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false,
                std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      const size_t n = (size_t)(s.step / sizeof(S)) * (size_t)s.bh;
      for (size_t i = 0; i < n; ++i) {
        const double v = (double)s.hptr()[i];
        if (v > 1e3) s.hptr()[i] = S(1e3);
        else if (v < -1e3) s.hptr()[i] = S(-1e3);
        else if (v != v) s.hptr()[i] = S(0);
      }
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, CH) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, CH));
    const int ncases = FixedWeights ? 1: 2;
    static const float fixedCo[4] = {0.299f, 0.587f, 0.114f, 0.0f};
    for (int ci = 0; ci < ncases; ++ci) {
      float co[4];
      std::memcpy(co, FixedWeights ? fixedCo: coeffSets[ci], sizeof(co));
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(dsrc, s.step, reinterpret_cast<S *>(d.dptr()), d.step, NppiSize{c.w, c.h}, co,
                                ncoef, ctx);
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
          const size_t so = (size_t)y * (s.step / (int)sizeof(S)) + (size_t)x * CH;
          const size_t db = off_bytes<S>(y, x, d.step, 1);
          double acc = 0.0;
          for (int ch = 0; ch < ncoef; ++ch) acc += (double)co[ch] * (double)as_double(hsrc[so + (size_t)ch]);
          const float accf = (float)acc;
          S out;
          if constexpr (std::is_same_v<S, Npp32f>) out = (S)accf;
          else out = round_clamp<S>((double)accf);
          std::memcpy(want.data() + db, &out, sizeof(S));
        }
      const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, 1, 1>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---- gradient color to gray ------------------------------------------------
template <typename S, class Call>
void run_gradient_gray(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int norms[3] = {nppiNormL1, nppiNormL2, nppiNormInf};
  for (const GenCase &c: make_gen_cases(0)) {
    if (c.w < 2 || c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, 3, c.pad, 0);
    Buf<S> d(c.w, c.h, 1, 0, 0x5A);
    fill_regime(s, 3, case_seed(id, c.w, c.h, s.step, d.step, 0, 0), c.regime, 0, false, false,
                std::is_floating_point_v<S>);
    if constexpr (std::is_floating_point_v<S>) {
      const size_t n = (size_t)(s.step / sizeof(S)) * (size_t)s.bh;
      for (size_t i = 0; i < n; ++i) {
        const double v = (double)s.hptr()[i];
        if (v > 1e3) s.hptr()[i] = S(1e3);
        else if (v < -1e3) s.hptr()[i] = S(-1e3);
        else if (v != v) s.hptr()[i] = S(0);
      }
    }
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const S *hsrc = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, 3) / sizeof(S);
    const S *dsrc = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, 3));
    for (int ni = 0; ni < 3; ++ni) {
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(dsrc, s.step, reinterpret_cast<S *>(d.dptr()), d.step, NppiSize{c.w, c.h},
                                norms[ni], ctx);
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
          const size_t so = (size_t)y * (s.step / (int)sizeof(S)) + (size_t)x * 3;
          const size_t db = off_bytes<S>(y, x, d.step, 1);
          const float r = (float)as_double(hsrc[so + 0]);
          const float g = (float)as_double(hsrc[so + 1]);
          const float b = (float)as_double(hsrc[so + 2]);
          float value = 0.0f;
          constexpr bool kScale = !std::is_same_v<S, Npp32f>;
          if (norms[ni] == nppiNormL1) value = kScale ? (r + g + b) * (1.0f / 3.0f): (r + g + b);
          else if (norms[ni] == nppiNormL2)
            value = kScale ? std::sqrt(r * r + g * g + b * b) * 0.5773502691896257f: std::sqrt(r * r + g * g + b * b);
          else value = std::max(r, std::max(g, b));
          S out;
          if constexpr (std::is_same_v<S, Npp32f>) out = (S)value;
          else out = trunc_clamp<S>((double)value);
          std::memcpy(want.data() + db, &out, sizeof(S));
        }
      const GenCase dc{c.w, c.h, 0, 0, 0, kSmall, 0};
      if (!rotate_check<S, 1, 1>(id, dc, NPPI_INTER_NN, pre, d.host, want, d.step,
                                 std::vector<uint8_t>(), cnt, first)) {
        if (cnt.mismatched >= 8) break;
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace cvt
}  // namespace num
