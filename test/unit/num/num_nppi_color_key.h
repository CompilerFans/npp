// CPU-reference output coverage for the color-key helpers of the color conversion
// module: nppiCompColorKey_{C1,C3,C4} and nppiAlphaCompColorKey_AC4.
//
// Measured NPP 13.4 semantics:
//   * CompColorKey: a pixel whose every channel equals the key is replaced by
//     the second source, otherwise the first source is copied;
//   * AlphaCompColorKey (AC4): a pixel whose first four channels equal the key
//     is replaced by the second source (all four channels); otherwise the RGB
//     channels are alpha-blended with the measured fixed-point coefficients
//     (opacities are promoted 255 -> 256, every division rounds half down),
//     and the destination alpha is left untouched.  Premultiplied and
//     out-of-range operators are no-ops.
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace colorkey {

inline int rhd(int n, int d) { return (2 * n + d - 1) / (2 * d); }
inline int clamp255(int v) { return v < 0 ? 0: (v > 255 ? 255: v); }

struct Coeffs {
  int a1, a2, c0, c1, c2, c0x, c2x;
};
inline Coeffs coeffs(int alpha1, int alpha2) {
  Coeffs k;
  k.a1 = alpha1 == 255 ? 256: alpha1;
  k.a2 = alpha2 == 255 ? 256: alpha2;
  k.c2 = rhd((256 - k.a1) * k.a2, 256);
  k.c1 = rhd(k.a1 * k.a2, 256);
  k.c0 = rhd(k.a1 * (256 - k.a2), 256);
  k.c0x = k.c0 > 255 ? 255: k.c0;
  k.c2x = k.c2 > 255 ? 255: k.c2;
  return k;
}

inline int blend(int op, const Coeffs &k, int s1, int s2) {
  int r;
  switch (op) {
    case NPPI_OP_ALPHA_OVER: r = rhd(s1 * k.a1, 256) + rhd(s2 * k.c2, 256); break;
    case NPPI_OP_ALPHA_IN: r = rhd(s1 * k.c1, 256); break;
    case NPPI_OP_ALPHA_OUT: r = rhd(s1 * k.c0, 256); break;
    case NPPI_OP_ALPHA_ATOP: r = rhd(rhd(s1 * k.a1, 256) * k.a2 + s2 * k.c2, 256); break;
    case NPPI_OP_ALPHA_XOR: r = rhd(s1 * k.c0x + s2 * k.c2x, 256); break;
    default: r = rhd(s1 * k.a1, 256) + rhd(s2 * k.a2, 256); break;
  }
  return clamp255(r);
}

// C = 1, 3 or 4 channels; the key array holds C values.
template <int C, class Call>
void run_comp_key(const char *id, const Npp8u *key, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {5, 4}, {17, 13}};
  for (const Cfg &cf: cfgs) {
    const int w = cf.w, h = cf.h;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    Buf<Npp8u> *s1 = new Buf<Npp8u>(w, h, C, 0, 0);
    Buf<Npp8u> *s2 = new Buf<Npp8u>(w, h, C, 0, 0);
    Buf<Npp8u> *d = new Buf<Npp8u>(w, h, C, 0, 0x5A);
    fill_regime(*s1, C, seed, kWide, 0, false, false, false);
    fill_regime(*s2, C, seed ^ 0x9E3779B9ull, kWide, 0, false, false, false);
    // Make a few pixels match the key so both branches are exercised.
    for (int k = 0; k < 3 && k < w; ++k)
      for (int c = 0; c < C; ++c) s1->hptr()[k * C + c] = key[c];
    s1->dev.copyFromHost(s1->host);
    s2->dev.copyFromHost(s2->host);
    d->dev.copyFromHost(d->host);
    const Npp8u *ss1 = s1->dptr(), *ss2 = s2->dptr();
    Npp8u *dd = d->dptr();
    const std::vector<uint8_t> pre = d->host;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(ss1, s1->step, ss2, s2->step, dd, d->step, NppiSize{w, h}, key, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      delete s1; delete s2; delete d;
      continue;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
    ++cnt.executed;
    std::vector<uint8_t> want = pre;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        bool match = true;
        for (int c = 0; c < C; ++c) match = match && s1->hptr()[(size_t)y * s1->step + (size_t)x * C + c] == key[c];
        const Buf<Npp8u> *from = match ? s2: s1;
        for (int c = 0; c < C; ++c)
          want[(size_t)y * d->step + (size_t)x * C + c] =
              from->hptr()[(size_t)y * from->step + (size_t)x * C + c];
      }
    d->dev.copyToHost(d->host);
    const GenCase dc{w, h, 0, 0, 0, kSmall, 0};
    bool ok;
    if (C == 1)
      ok = rotate_check<Npp8u, 1, 1>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {}, cnt, first);
    else if (C == 3)
      ok = rotate_check<Npp8u, 3, 3>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {}, cnt, first);
    else
      ok = rotate_check<Npp8u, 4, 4>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {}, cnt, first);
    (void)ok;
    delete s1; delete s2; delete d;
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// AC4 alpha composition with color key; sweeps the six blend operators and a
// few opacity pairs per size.
template <class Call>
void run_alpha_comp_key(const char *id, const Npp8u *key, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  struct Cfg { int w, h; };
  static const Cfg cfgs[] = {{4, 4}, {8, 6}, {16, 12}, {32, 24}, {64, 48}, {5, 4}, {17, 13}};
  static const int ops[] = {NPPI_OP_ALPHA_OVER, NPPI_OP_ALPHA_IN, NPPI_OP_ALPHA_OUT,
                            NPPI_OP_ALPHA_ATOP, NPPI_OP_ALPHA_XOR, NPPI_OP_ALPHA_PLUS};
  static const int alphas[][2] = {{255, 255}, {128, 64}, {0, 200}, {77, 255}};
  for (const Cfg &cf: cfgs) {
    const int w = cf.w, h = cf.h;
    const uint64_t seed = case_seed(id, w, h, 0, 0, 0, 0);
    Buf<Npp8u> *s1 = new Buf<Npp8u>(w, h, 4, 0, 0);
    Buf<Npp8u> *s2 = new Buf<Npp8u>(w, h, 4, 0, 0);
    fill_regime(*s1, 4, seed, kWide, 0, false, false, false);
    fill_regime(*s2, 4, seed ^ 0x9E3779B9ull, kWide, 0, false, false, false);
    for (int k = 0; k < 2 && k < w; ++k)
      for (int c = 0; c < 4; ++c) s1->hptr()[k * 4 + c] = key[c];
    s1->dev.copyFromHost(s1->host);
    s2->dev.copyFromHost(s2->host);
    bool failed = false;
    for (int op: ops) {
      for (const auto &al: alphas) {
        Buf<Npp8u> *d = new Buf<Npp8u>(w, h, 4, 0, 0x5A);
        d->dev.copyFromHost(d->host);
        npp_diff::clear_cuda_error();
        const NppStatus st = call(s1->dptr(), s1->step, (Npp8u)al[0], s2->dptr(), s2->step, (Npp8u)al[1],
                                  d->dptr(), d->step, NppiSize{w, h}, key, (NppiAlphaOp)op, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          delete d;
          continue;
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
        ++cnt.executed;
        const std::vector<uint8_t> pre = d->host;
        std::vector<uint8_t> want = pre;
        const Coeffs k = coeffs(al[0], al[1]);
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x) {
            bool match = true;
            for (int c = 0; c < 4; ++c)
              match = match && s1->hptr()[(size_t)y * s1->step + (size_t)x * 4 + c] == key[c];
            if (match) {
              for (int c = 0; c < 4; ++c)
                want[(size_t)y * d->step + (size_t)x * 4 + c] =
                    s2->hptr()[(size_t)y * s2->step + (size_t)x * 4 + c];
            } else {
              for (int c = 0; c < 3; ++c) {
                const int s1v = s1->hptr()[(size_t)y * s1->step + (size_t)x * 4 + c];
                const int s2v = s2->hptr()[(size_t)y * s2->step + (size_t)x * 4 + c];
                want[(size_t)y * d->step + (size_t)x * 4 + c] = (uint8_t)blend(op, k, s1v, s2v);
              }
            }
          }
        d->dev.copyToHost(d->host);
        const GenCase dc{w, h, 0, 0, 0, kSmall, 0};
        if (!rotate_check<Npp8u, 4, 4>(id, dc, NPPI_INTER_NN, pre, d->host, want, d->step, {}, cnt, first))
          failed = true;
        delete d;
        if (failed || cnt.mismatched >= 8) break;
      }
      if (failed || cnt.mismatched >= 8) break;
    }
    delete s1; delete s2;
    if (failed || cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace colorkey
}  // namespace num
