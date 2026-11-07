#pragma once

// CPU-reference output drivers for the nppi morphological operations.
//
// Measured NPP 13.4 contracts reproduced here:
//  * The plain Erode/Dilate (and Erode3x3/Dilate3x3) entry points read outside
//    the ROI for border pixels.  The NPP documentation states that sampling
//    beyond the image is the caller's responsibility, so those results are
//    undefined; the drivers only compare pixels whose whole mask window lies
//    inside the ROI (and skip ROIs with no such pixel).
//  * Border and composite families clamp every tap to the source image
//    (replicate).  The 8u masked variants only accept NPP_BORDER_REPLICATE,
//    which is the mode used by the tests.
//  * nppiMorphGradientBorder is measured to produce close - src (the same
//    result as BlackHat), not dilate - erode.
//  * Gray erode/dilate is min/max over every mask tap of (src + mask), with
//    replicate sampling; 8u saturates the result to [0, 255].
//  * AC4 layouts process the first C-1 channels and leave the destination
//    alpha byte untouched.

#include "num_suite.h"

namespace num {

template <typename T> inline T morph_pick(bool dilate, T a, T b) {
  if (dilate) return b > a ? b: a;
  return b < a ? b: a;
}

inline int morph_clamp(int v, int lo, int hi) { return v < lo ? lo: (v > hi ? hi: v); }

// One output sample.  border=false samples only taps inside the image (the
// caller guarantees the whole window is interior).
template <typename T, int C>
inline T morph_sample(const T *src, int stepElems, int w, int h, int x, int y, const uint8_t *mask, int mw,
                      int mh, int ax, int ay, bool dilate, bool border, int ch) {
  bool any = false;
  T acc{};
  for (int my = 0; my < mh; ++my) {
    int cy = y + my - ay;
    if (border) {
      cy = morph_clamp(cy, 0, h - 1);
    } else if (cy < 0 || cy >= h) {
      continue;
    }
    for (int mx = 0; mx < mw; ++mx) {
      if (mask != nullptr && mask[(size_t)my * mw + mx] == 0) continue;
      int cx = x + mx - ax;
      if (border) {
        cx = morph_clamp(cx, 0, w - 1);
      } else if (cx < 0 || cx >= w) {
        continue;
      }
      const T v = src[(size_t)cy * stepElems + (size_t)cx * C + ch];
      if (!any) {
        acc = v;
        any = true;
      } else {
        acc = morph_pick(dilate, acc, v);
      }
    }
  }
  return any ? acc: T(0);
}

// Interior window for the plain (non-border) families.
inline bool morph_interior(int w, int h, int mw, int mh, int ax, int ay, int *x0, int *x1, int *y0, int *y1) {
  *x0 = ax;
  *x1 = w - mw + ax;
  *y0 = ay;
  *y1 = h - mh + ay;
  return *x0 <= *x1 && *y0 <= *y1;
}

inline bool morph_interior3(int w, int h, int *x0, int *x1, int *y0, int *y1) {
  return morph_interior(w, h, 3, 3, 1, 1, x0, x1, y0, y1);
}

// Device mask: tests pass the mask through device memory as the NPP contract
// requires.
struct MorphMask {
  int mw = 0, mh = 0, ax = 0, ay = 0;
  std::vector<uint8_t> host;
  npp_functional_test::DeviceMemory<uint8_t> dev;
  MorphMask(int w, int h, int anchorX, int anchorY, const std::vector<uint8_t> &values): mw(w), mh(h), ax(anchorX), ay(anchorY), host(values), dev(values.size()) {
    dev.copyFromHost(host);
  }
};

inline std::vector<MorphMask *> morph_masks() {
  return {new MorphMask(3, 3, 1, 1, {0, 1, 0, 1, 1, 1, 0, 1, 0}),
          new MorphMask(3, 5, 0, 0, std::vector<uint8_t>(15, 1)),
          new MorphMask(5, 3, 2, 1, std::vector<uint8_t>(15, 1))};
}

template <typename T> inline void morph_tol(double &rtol, double &atol) {
  if constexpr (std::is_same_v<T, Npp64f>) {
    rtol = 0.0;
    atol = 1e-12;
  } else if constexpr (std::is_floating_point_v<T>) {
    rtol = 1e-6;
    atol = 1e-6;
  } else {
    rtol = 0.0;
    atol = 0.0;
  }
}

// ---------------------------------------------------------------------------
// Plain families: compare interior pixels only.
// ---------------------------------------------------------------------------
template <typename S, int C, int A, bool Dilate, class Call>
void run_morph_plain3(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c: make_gen_cases(0)) {
    int x0, x1, y0, y1;
    if (!morph_interior3(c.w, c.h, &x0, &x1, &y0, &y1)) continue;  // no defined output pixel
    // The plain entry points sample outside the ROI at the borders.  NPP
    // declares those results undefined, but the memory must still exist:
    // allocate a two-pixel margin and only compare interior pixels.
    const int margin = 2;
    const int sx = margin + c.ox, sy = margin + c.oy;
    Buf<S> s(c.w + 2 * margin + c.ox, c.h + 2 * margin + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    const NppStatus st = call(
        reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(sy, sx, s.step, C)), s.step,
        reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step, NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    const S *src = s.hptr() + off_bytes<S>(sy, sx, s.step, C) / sizeof(S);
    const int stepElems = s.step / (int)sizeof(S);
    bool failed = false;
    for (int y = y0; y <= y1 && !failed; ++y)
      for (int x = x0; x <= x1 && !failed; ++x) {
        const size_t base = off_bytes<S>(c.oy + y, c.ox + x, d.step, C);
        for (int ch = 0; ch < A && !failed; ++ch) {
          const S want = morph_sample<S, C>(src, stepElems, c.w, c.h, x, y, nullptr, 3, 3, 1, 1, Dilate, false, ch);
          ++cnt.compared;
          if (!equal_elem<S>(d.host.data() + base + (size_t)ch * sizeof(S),
                             reinterpret_cast<const uint8_t *>(&want), 0.0, 0.0)) {
            char b[360];
            std::snprintf(b, sizeof(b), "[%s] %s 3x3 (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(), x, y, ch,
                          dbg_elem<S>(d.host.data() + base + (size_t)ch * sizeof(S)).c_str(),
                          dbg_elem<S>(reinterpret_cast<const uint8_t *>(&want)).c_str());
            if (first.empty()) first = b;
            EXPECT_TRUE(false) << b;
            ++cnt.mismatched;
            failed = true;
          }
        }
      }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool Dilate, class Call>
void run_morph_plain_mask(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const std::vector<MorphMask *> masks = morph_masks();
  for (const GenCase &c: make_gen_cases(0)) {
    // Same undefined-border memory requirement as the 3x3 driver; the margin
    // covers the largest mask (5x3 / 3x5) used below.
    const int margin = 6;
    const int sx = margin + c.ox, sy = margin + c.oy;
    Buf<S> s(c.w + 2 * margin + c.ox, c.h + 2 * margin + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    for (MorphMask *m: masks) {
      int x0, x1, y0, y1;
      if (!morph_interior(c.w, c.h, m->mw, m->mh, m->ax, m->ay, &x0, &x1, &y0, &y1)) continue;
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(
          reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(sy, sx, s.step, C)), s.step,
          reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step, NppiSize{c.w, c.h},
          m->dev.get(), NppiSize{m->mw, m->mh}, NppiPoint{m->ax, m->ay}, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      const S *src = s.hptr() + off_bytes<S>(sy, sx, s.step, C) / sizeof(S);
      const int stepElems = s.step / (int)sizeof(S);
      bool failed = false;
      for (int y = y0; y <= y1 && !failed; ++y)
        for (int x = x0; x <= x1 && !failed; ++x) {
          const size_t base = off_bytes<S>(c.oy + y, c.ox + x, d.step, C);
          for (int ch = 0; ch < A && !failed; ++ch) {
            const S want = morph_sample<S, C>(src, stepElems, c.w, c.h, x, y, m->host.data(), m->mw, m->mh, m->ax,
                                              m->ay, Dilate, false, ch);
            ++cnt.compared;
            if (!equal_elem<S>(d.host.data() + base + (size_t)ch * sizeof(S),
                               reinterpret_cast<const uint8_t *>(&want), 0.0, 0.0)) {
              char b[360];
              std::snprintf(b, sizeof(b), "[%s] %s mask %dx%d a%d,%d (%d,%d,ch=%d) got=%s want=%s", id,
                            c.id().c_str(), m->mw, m->mh, m->ax, m->ay, x, y, ch,
                            dbg_elem<S>(d.host.data() + base + (size_t)ch * sizeof(S)).c_str(),
                            dbg_elem<S>(reinterpret_cast<const uint8_t *>(&want)).c_str());
              if (first.empty()) first = b;
              EXPECT_TRUE(false) << b;
              ++cnt.mismatched;
              failed = true;
            }
          }
        }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  for (auto *m: masks) delete m;
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Full-ROI comparison: colour channels against the reference, alpha channels
// against the pre-call destination bytes (AC4 preserves them).
// ---------------------------------------------------------------------------
template <typename S, int C, int A>
inline bool morph_check(const char *id, const GenCase &c, const char *what, const std::vector<uint8_t> &pre,
                        const std::vector<uint8_t> &got, const std::vector<uint8_t> &want, int step,
                        RunCounts &cnt, std::string &first) {
  double rtol, atol;
  morph_tol<S>(rtol, atol);
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      const size_t base = off_bytes<S>(c.oy + y, c.ox + x, step, C);
      for (int ch = 0; ch < C; ++ch) {
        const size_t off = base + (size_t)ch * sizeof(S);
        ++cnt.compared;
        bool ok;
        if (ch >= A) {
          ok = std::memcmp(got.data() + off, pre.data() + off, sizeof(S)) == 0;
        } else {
          ok = equal_elem<S>(got.data() + off, want.data() + off, rtol, atol);
        }
        if (!ok) {
          char b[400];
          std::snprintf(b, sizeof(b), "[%s] %s %s (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(), what, x, y,
                        ch, dbg_elem<S>(got.data() + off).c_str(), dbg_elem<S>(want.data() + off).c_str());
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          return false;
        }
      }
    }
  return true;
}

inline constexpr size_t kMorphScratch = 8 << 20;

template <typename S, int C, int A, bool Dilate, class Call>
void run_morph_border3(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c: make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const std::vector<uint8_t> pre = d.host;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(
        reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step, NppiSize{c.w, c.h},
        NppiPoint{0, 0}, reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
        NppiSize{c.w, c.h}, NPP_BORDER_REPLICATE, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    const S *src = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
    const int stepElems = s.step / (int)sizeof(S);
    std::vector<uint8_t> want = pre;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const size_t base = off_bytes<S>(c.oy + y, c.ox + x, d.step, C);
        for (int ch = 0; ch < A; ++ch) {
          const S v = morph_sample<S, C>(src, stepElems, c.w, c.h, x, y, nullptr, 3, 3, 1, 1, Dilate, true, ch);
          std::memcpy(want.data() + base + (size_t)ch * sizeof(S), &v, sizeof(S));
        }
      }
    if (!morph_check<S, C, A>(id, c, "3x3 border", pre, d.host, want, d.step, cnt, first)) {
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool Dilate, class Call>
void run_morph_border_mask(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const std::vector<MorphMask *> masks = morph_masks();
  for (const GenCase &c: make_gen_cases(0)) {
    // Same undefined-border memory requirement as the 3x3 driver; the margin
    // covers the largest mask (5x3 / 3x5) used below.
    const int margin = 6;
    const int sx = margin + c.ox, sy = margin + c.oy;
    Buf<S> s(c.w + 2 * margin + c.ox, c.h + 2 * margin + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    for (MorphMask *m: masks) {
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(
          reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step, NppiSize{c.w, c.h},
          NppiPoint{0, 0}, reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, m->dev.get(), NppiSize{m->mw, m->mh}, NppiPoint{m->ax, m->ay},
          NPP_BORDER_REPLICATE, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      const S *src = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
      const int stepElems = s.step / (int)sizeof(S);
      std::vector<uint8_t> want = pre;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const size_t base = off_bytes<S>(c.oy + y, c.ox + x, d.step, C);
          for (int ch = 0; ch < A; ++ch) {
            const S v = morph_sample<S, C>(src, stepElems, c.w, c.h, x, y, m->host.data(), m->mw, m->mh, m->ax,
                                           m->ay, Dilate, true, ch);
            std::memcpy(want.data() + base + (size_t)ch * sizeof(S), &v, sizeof(S));
          }
        }
      if (!morph_check<S, C, A>(id, c, "mask border", pre, d.host, want, d.step, cnt, first)) break;
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  for (auto *m: masks) delete m;
  finish(id, cnt, first);
}

// kind: 0 Open, 1 Close, 2 Gradient, 3 TopHat, 4 BlackHat.
template <typename S, int C, int A, int Kind, class Call>
void run_morph_combo(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const std::vector<MorphMask *> masks = morph_masks();
  npp_functional_test::DeviceMemory<uint8_t> scratch(kMorphScratch);
  const bool firstDilate = (Kind == 1 || Kind == 2 || Kind == 4);
  for (const GenCase &c: make_gen_cases(0)) {
    // Same undefined-border memory requirement as the 3x3 driver; the margin
    // covers the largest mask (5x3 / 3x5) used below.
    const int margin = 6;
    const int sx = margin + c.ox, sy = margin + c.oy;
    Buf<S> s(c.w + 2 * margin + c.ox, c.h + 2 * margin + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    for (MorphMask *m: masks) {
      const std::vector<uint8_t> pre = d.host;
      npp_diff::clear_cuda_error();
      const NppStatus st = call(
          reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step, NppiSize{c.w, c.h},
          NppiPoint{0, 0}, reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
          NppiSize{c.w, c.h}, m->dev.get(), NppiSize{m->mw, m->mh}, NppiPoint{m->ax, m->ay}, scratch.get(),
          NPP_BORDER_REPLICATE, ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      const cudaError_t se = cudaDeviceSynchronize();
      ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
      d.dev.copyToHost(d.host);
      ++cnt.executed;
      const S *src = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, C) / sizeof(S);
      const int stepElems = s.step / (int)sizeof(S);
      const int tmpStep = c.w * C;
      std::vector<S> tmp((size_t)tmpStep * c.h), morph((size_t)tmpStep * c.h);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < A; ++ch)
            tmp[(size_t)y * tmpStep + (size_t)x * C + ch] = morph_sample<S, C>(
                src, stepElems, c.w, c.h, x, y, m->host.data(), m->mw, m->mh, m->ax, m->ay, firstDilate, true, ch);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          for (int ch = 0; ch < A; ++ch)
            morph[(size_t)y * tmpStep + (size_t)x * C + ch] = morph_sample<S, C>(
                tmp.data(), tmpStep, c.w, c.h, x, y, m->host.data(), m->mw, m->mh, m->ax, m->ay, !firstDilate,
                true, ch);
      std::vector<uint8_t> want = pre;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const size_t base = off_bytes<S>(c.oy + y, c.ox + x, d.step, C);
          const size_t mbase = (size_t)y * tmpStep + (size_t)x * C;
          for (int ch = 0; ch < A; ++ch) {
            const S a = morph[mbase + ch];
            const S b = src[(size_t)y * stepElems + (size_t)x * C + ch];
            S v;
            if constexpr (std::is_floating_point_v<S>) {
              v = (Kind == 2 || Kind == 4) ? (S)(a - b): (Kind == 3 ? (S)(b - a): a);
            } else {
              const long long av = (long long)a, bv = (long long)b;
              const long long r = (Kind == 2 || Kind == 4) ? av - bv: (Kind == 3 ? bv - av: av);
              v = (S)r;
            }
            std::memcpy(want.data() + base + (size_t)ch * sizeof(S), &v, sizeof(S));
          }
        }
      if (!morph_check<S, C, A>(id, c, "composite", pre, d.host, want, d.step, cnt, first)) break;
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  for (auto *m: masks) delete m;
  finish(id, cnt, first);
}

template <typename S, typename M, bool Dilate, class Call>
void run_morph_gray(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const std::vector<uint8_t> maskHost = {0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<M> maskVals(9);
  for (int i = 0; i < 9; ++i) maskVals[i] = (M)maskHost[i];
  npp_functional_test::DeviceMemory<M> maskDev(9);
  {
    std::vector<M> hv(maskVals);
    maskDev.copyFromHost(hv);
  }
  const int mw = 3, mh = 3, ax = 1, ay = 1;
  for (const GenCase &c: make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, 1, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, 1, seed, c.regime, 0, false, false, std::is_floating_point_v<S>);
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    const std::vector<uint8_t> pre = d.host;
    npp_diff::clear_cuda_error();
    const NppStatus st = call(
        reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, 1)), s.step, NppiSize{c.w, c.h},
        NppiPoint{0, 0}, reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, 1)), d.step,
        NppiSize{c.w, c.h}, maskDev.get(), NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    const S *src = s.hptr() + off_bytes<S>(c.oy, c.ox, s.step, 1) / sizeof(S);
    const int stepElems = s.step / (int)sizeof(S);
    std::vector<uint8_t> want = pre;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const size_t base = off_bytes<S>(c.oy + y, c.ox + x, d.step, 1);
        bool any = false;
        double acc = 0.0;
        for (int my = 0; my < mh; ++my) {
          const int cy = morph_clamp(y + my - ay, 0, c.h - 1);
          for (int mx = 0; mx < mw; ++mx) {
            const int cx = morph_clamp(x + mx - ax, 0, c.w - 1);
            // Every tap participates: the mask is an additive structuring
            // element, not a selector (measured on the reference).
            const double v = (double)src[(size_t)cy * stepElems + cx] +
                             (double)maskVals[(size_t)my * mw + mx];
            if (!any) {
              acc = v;
              any = true;
            } else {
              acc = Dilate ? (v > acc ? v: acc): (v < acc ? v: acc);
            }
          }
        }
        S v;
        if constexpr (std::is_same_v<S, Npp8u>) {
          acc = acc < 0.0 ? 0.0: (acc > 255.0 ? 255.0: acc);
          v = (S)acc;
        } else {
          v = (S)acc;
        }
        std::memcpy(want.data() + base, &v, sizeof(S));
      }
    if (!morph_check<S, 1, 1>(id, c, "gray", pre, d.host, want, d.step, cnt, first)) {
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace num
