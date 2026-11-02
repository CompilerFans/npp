#pragma once

// CPU-golden drivers for the nppi statistics reduction families.
//
// Outputs are compared against formulas computed in-process from the ROI;
// there are no recorded goldens.  The same generated tests build against the
// in-tree library and against NVIDIA NPP 13.4.

#include "num_suite.h"

namespace num {

enum class NppiStatKind {
  Sum, Mean, NormL1, NormL2, NormInf,
  NormDiffL1, NormDiffL2, NormDiffInf,
  NormRelL1, NormRelL2, NormRelInf,
  MaxAbsErr, AvgAbsErr, MaxRelErr, AvgRelErr
};

inline constexpr int kNppiStatScratch = 4 << 20;

template <typename D> inline void nppi_stat_tol(double &rtol, double &atol) {
  if constexpr (std::is_same_v<D, Npp32f>) { rtol = 1e-5; atol = 1e-4; }
  else if constexpr (std::is_same_v<D, Npp64f>) { rtol = 1e-10; atol = 1e-7; }
  else { rtol = 0; atol = 0; }
}

template <typename S, typename D> inline void nppi_stat_tol_s(double &rtol, double &atol) {
  nppi_stat_tol<D>(rtol, atol);
  if constexpr (std::is_same_v<S, Npp32f> && std::is_same_v<D, Npp64f>) {
    // NPP accumulates 32f reductions in single precision; the CPU reference
    // uses double, so allow the corresponding reduction-order envelope.
    rtol = 1e-5;
    atol = 1e-2;
  }
}

// Fill the ROI with a deterministic pattern that has unique per-channel
// minima and maxima (at the first and last pixel).  NPP's multi-block index
// reduction does not define a tie-break order, so index tests use unique
// extrema; non-index extrema tests keep the random regimes.
template <typename S, int C>
inline void nppi_fill_unique_extrema(Buf<S> &b, const GenCase &c, int active) {
  S base;
  if constexpr (std::is_signed_v<S> || std::is_floating_point_v<S>) base = S(0);
  else base = S(1);
  const size_t total = (size_t)(b.step / (int)sizeof(S)) * (size_t)b.bh;
  for (size_t i = 0; i < total; ++i) b.hptr()[i] = base;
  const int area = c.w * c.h;
  if (area < 2) return;
  auto setpix = [&](int pix, S v) {
    const int y = pix / c.w, x = pix % c.w;
    for (int ch = 0; ch < active; ++ch)
      b.hptr()[(size_t)(c.oy + y) * (b.step / (int)sizeof(S)) + (size_t)(c.ox + x) * C + ch] = v;
  };
  setpix(0, (S)(base - 1));
  setpix(area - 1, (S)(base + 1));
}

template <typename S> inline void nppi_clamp_stat_inputs(Buf<S> &b) {
  if constexpr (std::is_floating_point_v<S>) {
    const size_t total = (size_t)(b.step / (int)sizeof(S)) * (size_t)b.bh;
    for (size_t i = 0; i < total; ++i) {
      if (b.hptr()[i] > S(1)) b.hptr()[i] = S(1);
      if (b.hptr()[i] < S(-1)) b.hptr()[i] = S(-1);
    }
  } else {
    (void)b;
  }
}

template <typename S, int C>
inline S nppi_roi_at(const Buf<S> &b, const GenCase &c, int x, int y, int ch) {
  return b.hptr()[(size_t)(c.oy + y) * (b.step / (int)sizeof(S)) + (size_t)(c.ox + x) * C + ch];
}

template <typename S, int C, int A, NppiStatKind K, typename Out, class Call>
void run_nppi_reduce1(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Out>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    nppi_clamp_stat_inputs(s);
    s.dev.copyFromHost(s.host);
    Buf<Out> out(4, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)),
                        s.step, NppiSize{c.w, c.h}, reinterpret_cast<Out *>(out.dptr()),
                        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      double sum = 0.0, sq = 0.0, inf = 0.0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double v = (double)nppi_roi_at<S, C>(s, c, x, y, ch);
          sum += v;
          sq += v * v;
          double av;
          if constexpr (K == NppiStatKind::NormInf && std::is_same_v<S, Npp16s> && C == 4 && A == 3) {
            // NPP's 16s AC4 Norm_Inf path compares the raw 16-bit patterns as
            // unsigned values (no sign/abs), a measured 13.4 quirk.
            av = (double)(uint16_t)(int16_t)nppi_roi_at<S, C>(s, c, x, y, ch);
          } else {
            av = std::fabs(v);
          }
          if (av > inf) inf = av;
        }
      double want_d = 0.0;
      long long want_ll = 0;
      switch (K) {
        case NppiStatKind::Sum: sum = sum; break;
        case NppiStatKind::Mean: sum /= (double)(c.w * c.h); break;
        case NppiStatKind::NormL1: {
          double l1 = 0.0;
          for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) l1 += std::fabs((double)nppi_roi_at<S, C>(s, c, x, y, ch));
          sum = l1;
          break;
        }
        case NppiStatKind::NormL2: sum = std::sqrt(sq); break;
        default: sum = inf; break;
      }
      want_d = sum;
      if constexpr (std::is_integral_v<Out>) want_ll = (long long)std::llrint(want_d);
      Out want{};
      if constexpr (std::is_integral_v<Out>) want = (Out)want_ll;
      else want = (Out)want_d;
      ++cnt.compared;
      const size_t off = (size_t)ch * sizeof(Out);
      if (!equal_elem<Out>(out.host.data() + off, reinterpret_cast<const uint8_t *>(&want), rtol, atol)) {
        char b[340];
        std::snprintf(b, sizeof(b), "[%s] %s ch=%d got=%s want=%s", id, c.id().c_str(), ch,
                      dbg_elem<Out>(out.host.data() + off).c_str(),
                      dbg_elem<Out>(reinterpret_cast<const uint8_t *>(&want)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, NppiStatKind K, class Call>
void run_nppi_reduce2(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Npp64f>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    // Clamped extremes can leave near-zero variance, where the quality-index
    // ratio is numerically meaningless; use the wide regime instead.
    const Regime qr = c.regime == kExtremes ? kWide : c.regime;
    fill_regime(s1, C, seed, qr, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, qr, 1, false, false, finite);
    nppi_clamp_stat_inputs(s1);
    nppi_clamp_stat_inputs(s2);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Npp64f> out(4, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)), s1.step,
        reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)), s2.step,
        NppiSize{c.w, c.h}, reinterpret_cast<Npp64f *>(out.dptr()),
        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      double dsum = 0.0, dsq = 0.0, dinf = 0.0;
      double nsum = 0.0, nsq = 0.0, ninf = 0.0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double a = (double)nppi_roi_at<S, C>(s1, c, x, y, ch);
          const double b = (double)nppi_roi_at<S, C>(s2, c, x, y, ch);
          const double d = a - b;
          dsum += std::fabs(d);
          dsq += d * d;
          dinf = std::max(dinf, std::fabs(d));
          nsum += std::fabs(b);
          nsq += b * b;
          if constexpr (K == NppiStatKind::NormRelInf && std::is_same_v<S, Npp16s> && C == 4 && A == 3) {
            ninf = std::max(ninf, (double)(uint16_t)(int16_t)nppi_roi_at<S, C>(s2, c, x, y, ch));
          } else {
            ninf = std::max(ninf, std::fabs(b));
          }
        }
      double want = 0.0;
      if constexpr (K == NppiStatKind::NormDiffL1) want = dsum;
      else if constexpr (K == NppiStatKind::NormDiffL2) want = std::sqrt(dsq);
      else if constexpr (K == NppiStatKind::NormDiffInf) want = dinf;
      else if constexpr (K == NppiStatKind::NormRelL1)
        want = (nsum > 0.0) ? dsum / nsum : (dsum == 0.0 ? 0.0 : (double)std::numeric_limits<S>::max());
      else if constexpr (K == NppiStatKind::NormRelL2)
        want = (nsq > 0.0) ? std::sqrt(dsq) / std::sqrt(nsq)
                           : (dsq == 0.0 ? 0.0 : (double)std::numeric_limits<S>::max());
      else
        want = (ninf > 0.0) ? dinf / ninf
                            : (dinf == 0.0 ? 0.0
                                           : (C == 1 ? dinf : (double)std::numeric_limits<S>::max()));
      ++cnt.compared;
      const size_t off = (size_t)ch * sizeof(Npp64f);
      if (!equal_elem<Npp64f>(out.host.data() + off, reinterpret_cast<const uint8_t *>(&want), rtol, atol)) {
        char b[340];
        std::snprintf(b, sizeof(b), "[%s] %s ch=%d got=%s want=%s", id, c.id().c_str(), ch,
                      dbg_elem<Npp64f>(out.host.data() + off).c_str(),
                      dbg_elem<Npp64f>(reinterpret_cast<const uint8_t *>(&want)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, NppiStatKind K, class Call>
void run_nppi_error(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Npp64f>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  const bool rel = (K == NppiStatKind::MaxRelErr || K == NppiStatKind::AvgRelErr);
  const bool avg = (K == NppiStatKind::AvgAbsErr || K == NppiStatKind::AvgRelErr);
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    // Clamped extremes can leave near-zero variance, where the quality-index
    // ratio is numerically meaningless; use the wide regime instead.
    const Regime qr = c.regime == kExtremes ? kWide : c.regime;
    fill_regime(s1, C, seed, qr, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, qr, 1, false, false, finite);
    nppi_clamp_stat_inputs(s1);
    nppi_clamp_stat_inputs(s2);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Npp64f> out(1, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)), s1.step,
        reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)), s2.step,
        NppiSize{c.w, c.h}, reinterpret_cast<Npp64f *>(out.dptr()),
        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    double acc = 0.0, mx = 0.0;
    const double eps = std::numeric_limits<double>::min();
    for (int ch = 0; ch < A; ++ch)
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double a = (double)nppi_roi_at<S, C>(s1, c, x, y, ch);
          const double b = (double)nppi_roi_at<S, C>(s2, c, x, y, ch);
          double e = std::fabs(a - b);
          if (rel) {
            const double den = std::max(std::fabs(a), std::fabs(b));
            e = (den > 0.0) ? e / den : 0.0;
          }
          acc += e;
          mx = std::max(mx, e);
        }
    (void)eps;
    const double want = avg ? acc / (double)(c.w * c.h * A) : mx;
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<Npp64f>(out.host.data(), reinterpret_cast<const uint8_t *>(&want), rtol, atol)) {
      char b[340];
      std::snprintf(b, sizeof(b), "[%s] %s got=%s want=%s", id, c.id().c_str(),
                    dbg_elem<Npp64f>(out.host.data()).c_str(),
                    dbg_elem<Npp64f>(reinterpret_cast<const uint8_t *>(&want)).c_str());
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, typename Out, class Call>
void run_nppi_dotprod(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Out>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    // Clamped extremes can leave near-zero variance, where the quality-index
    // ratio is numerically meaningless; use the wide regime instead.
    const Regime qr = c.regime == kExtremes ? kWide : c.regime;
    fill_regime(s1, C, seed, qr, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, qr, 1, false, false, finite);
    nppi_clamp_stat_inputs(s1);
    nppi_clamp_stat_inputs(s2);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Out> out(4, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)), s1.step,
        reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)), s2.step,
        NppiSize{c.w, c.h}, reinterpret_cast<Out *>(out.dptr()),
        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      double acc = 0.0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
          acc += (double)nppi_roi_at<S, C>(s1, c, x, y, ch) *
                 (double)nppi_roi_at<S, C>(s2, c, x, y, ch);
      const Out want = (Out)acc;
      ++cnt.compared;
      const size_t off = (size_t)ch * sizeof(Out);
      if (!equal_elem<Out>(out.host.data() + off, reinterpret_cast<const uint8_t *>(&want), rtol, atol)) {
        char b[340];
        std::snprintf(b, sizeof(b), "[%s] %s ch=%d got=%s want=%s", id, c.id().c_str(), ch,
                      dbg_elem<Out>(out.host.data() + off).c_str(),
                      dbg_elem<Out>(reinterpret_cast<const uint8_t *>(&want)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, class Call>
void run_nppi_meanstd(const char *id, Call call, int coi = -1) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Npp64f>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    nppi_clamp_stat_inputs(s);
    s.dev.copyFromHost(s.host);
    Buf<Npp64f> mo(1, 1, 1, 0, 0x5A), so(1, 1, 1, 0, 0x5A);
    mo.dev.copyFromHost(mo.host);
    so.dev.copyFromHost(so.host);
    npp_diff::clear_cuda_error();
    NppStatus st;
    const S *p = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    if constexpr (C > 1)
      st = call(p, s.step, NppiSize{c.w, c.h}, coi, reinterpret_cast<Npp64f *>(mo.dptr()),
                reinterpret_cast<Npp64f *>(so.dptr()), reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    else
      st = call(p, s.step, NppiSize{c.w, c.h}, reinterpret_cast<Npp64f *>(mo.dptr()),
                reinterpret_cast<Npp64f *>(so.dptr()), reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    mo.dev.copyToHost(mo.host);
    so.dev.copyToHost(so.host);
    const int ch = C > 1 ? (coi - 1) : 0;
    double sum = 0.0, sq = 0.0;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const double v = (double)nppi_roi_at<S, C>(s, c, x, y, ch);
        sum += v;
        sq += v * v;
      }
    const double n = (double)(c.w * c.h);
    const double mean = sum / n;
    double var = sq / n - mean * mean;
    if (!(var > 0.0)) var = 0.0;
    const double sd = std::sqrt(var);
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<Npp64f>(mo.host.data(), reinterpret_cast<const uint8_t *>(&mean), rtol, atol) ||
        !equal_elem<Npp64f>(so.host.data(), reinterpret_cast<const uint8_t *>(&sd), rtol, atol)) {
      double gm = 0.0, gs = 0.0;
      std::memcpy(&gm, mo.host.data(), sizeof(gm));
      std::memcpy(&gs, so.host.data(), sizeof(gs));
      char b[340];
      std::snprintf(b, sizeof(b), "[%s] %s got=(%.17g,%.17g) want=(%.17g,%.17g)", id, c.id().c_str(),
                    gm, gs, mean, sd);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool WantMin, bool WantMax, bool WantMinIdx, bool WantMaxIdx, class Call>
void run_nppi_extrema(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    if constexpr (WantMinIdx || WantMaxIdx) {
      if (c.w * c.h < 2) continue;
      nppi_fill_unique_extrema<S, C>(s, c, A);
    }
    s.dev.copyFromHost(s.host);
    Buf<S> mn(4, 1, 1, 0, 0x5A), mx(4, 1, 1, 0, 0x5A);
    Buf<Npp32s> mnx(4, 1, 1, 0, 0x5A), mny(4, 1, 1, 0, 0x5A);
    Buf<Npp32s> mxx(4, 1, 1, 0, 0x5A), mxy(4, 1, 1, 0, 0x5A);
    mn.dev.copyFromHost(mn.host); mx.dev.copyFromHost(mx.host);
    mnx.dev.copyFromHost(mnx.host); mny.dev.copyFromHost(mny.host);
    mxx.dev.copyFromHost(mxx.host); mxy.dev.copyFromHost(mxy.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
        NppiSize{c.w, c.h}, reinterpret_cast<Npp8u *>(scratch.dptr()),
        WantMin ? reinterpret_cast<S *>(mn.dptr()) : nullptr,
        WantMax ? reinterpret_cast<S *>(mx.dptr()) : nullptr,
        WantMinIdx ? reinterpret_cast<int *>(mnx.dptr()) : nullptr,
        WantMinIdx ? reinterpret_cast<int *>(mny.dptr()) : nullptr,
        WantMaxIdx ? reinterpret_cast<int *>(mxx.dptr()) : nullptr,
        WantMaxIdx ? reinterpret_cast<int *>(mxy.dptr()) : nullptr, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    mn.dev.copyToHost(mn.host); mx.dev.copyToHost(mx.host);
    mnx.dev.copyToHost(mnx.host); mny.dev.copyToHost(mny.host);
    mxx.dev.copyToHost(mxx.host); mxy.dev.copyToHost(mxy.host);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      S vmin = nppi_roi_at<S, C>(s, c, 0, 0, ch), vmax = vmin;
      int minx = 0, miny = 0, maxx = 0, maxy = 0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const S v = nppi_roi_at<S, C>(s, c, x, y, ch);
          if (v < vmin) { vmin = v; minx = x; miny = y; }
          if (v > vmax) { vmax = v; maxx = x; maxy = y; }
        }
      auto chk = [&](bool enabled, const uint8_t *got, const void *wantp, const char *tag) {
        if (!enabled) return;
        ++cnt.compared;
        if (!equal_elem<S>(got, reinterpret_cast<const uint8_t *>(wantp), 0, 0)) {
          char b[320];
          std::snprintf(b, sizeof(b), "[%s] %s ch=%d %s got=%s want=%s", id, c.id().c_str(), ch, tag,
                        dbg_elem<S>(got).c_str(), dbg_elem<S>(reinterpret_cast<const uint8_t *>(wantp)).c_str());
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
        }
      };
      if constexpr (WantMin) chk(true, mn.host.data() + (size_t)ch * sizeof(S), &vmin, "min");
      if constexpr (WantMax) chk(true, mx.host.data() + (size_t)ch * sizeof(S), &vmax, "max");
      if constexpr (WantMinIdx) {
        const int gx = reinterpret_cast<const int *>(mnx.host.data())[ch];
        const int gy = reinterpret_cast<const int *>(mny.host.data())[ch];
        ++cnt.compared;
        if (gx != minx || gy != miny) {
          char b[320];
          std::snprintf(b, sizeof(b), "[%s] %s ch=%d minidx got=(%d,%d) want=(%d,%d)", id, c.id().c_str(),
                        ch, gx, gy, minx, miny);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
        }
      }
      if constexpr (WantMaxIdx) {
        const int gx = reinterpret_cast<const int *>(mxx.host.data())[ch];
        const int gy = reinterpret_cast<const int *>(mxy.host.data())[ch];
        ++cnt.compared;
        if (gx != maxx || gy != maxy) {
          char b[320];
          std::snprintf(b, sizeof(b), "[%s] %s ch=%d maxidx got=(%d,%d) want=(%d,%d)", id, c.id().c_str(),
                        ch, gx, gy, maxx, maxy);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
        }
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, class Call>
void run_nppi_minmaxindx(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    if (c.w * c.h < 2) continue;
    nppi_fill_unique_extrema<S, C>(s, c, C);
    s.dev.copyFromHost(s.host);
    Buf<S> mn(4, 1, 1, 0, 0x5A), mx(4, 1, 1, 0, 0x5A);
    Buf<Npp32s> pi(8, 1, 1, 0, 0x5A), qi(8, 1, 1, 0, 0x5A);
    mn.dev.copyFromHost(mn.host); mx.dev.copyFromHost(mx.host);
    pi.dev.copyFromHost(pi.host); qi.dev.copyFromHost(qi.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
        NppiSize{c.w, c.h}, reinterpret_cast<S *>(mn.dptr()), reinterpret_cast<S *>(mx.dptr()),
        reinterpret_cast<NppiPoint *>(pi.dptr()), reinterpret_cast<NppiPoint *>(qi.dptr()),
        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    mn.dev.copyToHost(mn.host); mx.dev.copyToHost(mx.host);
    pi.dev.copyToHost(pi.host); qi.dev.copyToHost(qi.host);
    ++cnt.executed;
    const S v0 = nppi_roi_at<S, C>(s, c, 0, 0, 0);
    S vmin = v0, vmax = v0;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const S v = nppi_roi_at<S, C>(s, c, x, y, 0);
        if (v < vmin) { vmin = v; minx = x; miny = y; }
        if (v > vmax) { vmax = v; maxx = x; maxy = y; }
      }
    const NppiPoint *gp = reinterpret_cast<const NppiPoint *>(pi.host.data());
    const NppiPoint *gq = reinterpret_cast<const NppiPoint *>(qi.host.data());
    ++cnt.compared;
    if (!equal_elem<S>(mn.host.data(), reinterpret_cast<const uint8_t *>(&vmin), 0, 0) ||
        !equal_elem<S>(mx.host.data(), reinterpret_cast<const uint8_t *>(&vmax), 0, 0) ||
        gp->x != minx || gp->y != miny || gq->x != maxx || gq->y != maxy) {
      char b[360];
      std::snprintf(b, sizeof(b), "[%s] %s got=(%d@%d,%d,%d@%d,%d) want=(%d@%d,%d,%d@%d,%d)",
                    id, c.id().c_str(), (int)mn.host[0], gp->x, gp->y, (int)mx.host[0], gq->x, gq->y,
                    (int)vmin, minx, miny, (int)vmax, maxx, maxy);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool IsMax, class Call>
void run_nppi_every(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> d(c.w + c.ox, c.h + c.oy, C, c.pad, 0x5A);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, d.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    fill_regime(d, C, seed ^ 0x2545f491ull, c.regime, 2, false, false, finite);
    const std::vector<uint8_t> old_dst = d.host;
    s.dev.copyFromHost(s.host);
    d.dev.copyFromHost(d.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)), s.step,
        reinterpret_cast<S *>(d.dptr() + off_bytes<S>(c.oy, c.ox, d.step, C)), d.step,
        NppiSize{c.w, c.h}, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x)
        for (int ch = 0; ch < C; ++ch) {
          const size_t off = off_bytes<S>(c.oy + y, c.ox + x, d.step, C) + (size_t)ch * sizeof(S);
          ++cnt.compared;
          if (ch >= A) {
            const uint8_t *wantp = old_dst.data() + off;  // alpha untouched
            if (!equal_elem<S>(d.host.data() + off, wantp, 0, 0)) {
              ++cnt.mismatched;
            }
            continue;
          }
          const S sv = nppi_roi_at<S, C>(s, c, x, y, ch);
          const S dv = reinterpret_cast<const S *>(old_dst.data())[off / sizeof(S)];
          const S want = IsMax ? (dv > sv ? dv : sv) : (dv < sv ? dv : sv);
          if (!equal_elem<S>(d.host.data() + off, reinterpret_cast<const uint8_t *>(&want), 0, 0)) {
            char b[340];
            std::snprintf(b, sizeof(b), "[%s] %s (x=%d,y=%d,ch=%d) got=%s want=%s", id, c.id().c_str(), x, y,
                          ch, dbg_elem<S>(d.host.data() + off).c_str(),
                          dbg_elem<S>(reinterpret_cast<const uint8_t *>(&want)).c_str());
            if (first.empty()) first = b;
            EXPECT_TRUE(false) << b;
            ++cnt.mismatched;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <class Call>
void run_nppi_buffer_size(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const NppiSize rois[] = {{1, 1}, {3, 2}, {7, 5}, {33, 17}, {257, 129}};
  for (const NppiSize &r : rois) {
    int size = -1;
    NppStatus st = call(r, &size, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ++cnt.executed;
    ++cnt.compared;
    if (size < 0) {
      char b[240];
      std::snprintf(b, sizeof(b), "[%s] %dx%d negative buffer size %d", id, r.width, r.height, size);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Histograms (Even / Range).  NPP histograms have nLevels-1 bins per active
// channel; Even bins are floor((v - lower) * (nLevels-1) / (upper - lower))
// for lower <= v < upper.  Range levels are device-resident arrays and a
// sample belongs to bin i when levels[i] <= v < levels[i+1].
// ---------------------------------------------------------------------------
// Wrap a cumulative sum the way NPP's 32-bit integral accumulators do.
template <typename D> inline D nppi_integral_cast(double v) {
  if constexpr (std::is_integral_v<D> && sizeof(D) <= 4)
    return (D)(uint32_t)(uint64_t)(long long)v;
  else
    return (D)v;
}

enum class NppiHistKind { Even, Range };

template <typename S, typename LV, int C, int A, NppiHistKind K, class Call>
void run_nppi_histogram(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    s.dev.copyFromHost(s.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    int nlevels[4];
    LV lower[4], upper[4];
    LV hlevels[4][8];
    std::vector<LV> lvhost[(4)];
    DeviceMemory<LV> lvdev[4] = {DeviceMemory<LV>(0), DeviceMemory<LV>(0), DeviceMemory<LV>(0), DeviceMemory<LV>(0)};
    int maxlev = 2;
    for (int ch = 0; ch < A; ++ch) {
      if constexpr (K == NppiHistKind::Even) {
        int n = 3 + (int)(rng.next() % 4);  // level boundaries -> n-1 bins
        nlevels[ch] = n;
        if (n > maxlev) maxlev = n;
        if constexpr (std::is_same_v<S, Npp8u>) {
          lower[ch] = (LV)(rng.next() % 32);
          upper[ch] = (LV)((long long)lower[ch] + 32 + (long long)(rng.next() % 180));
          if ((long long)upper[ch] > 255) upper[ch] = (LV)256;
        } else if constexpr (std::is_same_v<S, Npp16u>) {
          lower[ch] = (LV)(rng.next() % 2000);
          upper[ch] = (LV)((long long)lower[ch] + 500 + (long long)(rng.next() % 4000));
          if ((long long)upper[ch] > 65535) upper[ch] = (LV)65536;
        } else {
          lower[ch] = (LV)(-((long long)(rng.next() % 1000)));
          upper[ch] = (LV)((long long)lower[ch] + 500 + (long long)(rng.next() % 4000));
        }
      } else {
        int n = 3 + (int)(rng.next() % 4);
        nlevels[ch] = n;
        if (n > maxlev) maxlev = n;
        const long long step = 1 + (long long)(rng.next() % 20);
        const long long lo = (long long)(rng.next() % 10) - 5;
        for (int i = 0; i < n; ++i) hlevels[ch][i] = (LV)(lo + step * i);
        lvhost[ch].assign(hlevels[ch], hlevels[ch] + n);
        lvdev[ch].allocate((size_t)n);
        lvdev[ch].copyFromHost(lvhost[ch]);
      }
    }
    Buf<Npp32s> hist((maxlev - 1) * C + 4, 1, 1, 0, 0x5A);
    hist.dev.copyFromHost(hist.host);
    Npp32s *hptr[4];
    for (int ch = 0; ch < (C == 1 ? 1 : A); ++ch)
      hptr[ch] = reinterpret_cast<Npp32s *>(hist.dptr()) + (size_t)ch * (maxlev - 1);
    npp_diff::clear_cuda_error();
    NppStatus st;
    if constexpr (K == NppiHistKind::Even) {
      st = call(reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)),
                s.step, NppiSize{c.w, c.h}, hptr, nlevels, lower, upper,
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    } else {
      const LV *lptr[4];
      for (int ch = 0; ch < (C == 1 ? 1 : A); ++ch) lptr[ch] = lvdev[ch].get();
      st = call(reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)),
                s.step, NppiSize{c.w, c.h}, hptr, lptr, nlevels,
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    hist.dev.copyToHost(hist.host);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      const int nb = nlevels[ch] - 1;
      std::vector<long long> want((size_t)nb, 0);
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double v = (double)nppi_roi_at<S, C>(s, c, x, y, ch);
          int bin = -1;
          if constexpr (K == NppiHistKind::Even) {
            const double lo = (double)lower[ch], hi = (double)upper[ch];
            if (v >= lo && v < hi) {
              bin = (int)std::floor((v - lo) * (double)nb / (hi - lo));
              if (bin < 0) bin = 0;
              if (bin >= nb) bin = nb - 1;
            }
          } else {
            for (int i = 0; i < nb; ++i)
              if (v >= (double)hlevels[ch][i] && v < (double)hlevels[ch][i + 1]) {
                bin = i;
                break;
              }
          }
          if (bin >= 0) ++want[(size_t)bin];
        }
      for (int i = 0; i < nb; ++i) {
        ++cnt.compared;
        const size_t off = ((size_t)ch * (maxlev - 1) + (size_t)i) * sizeof(Npp32s);
        Npp32s w32 = (Npp32s)want[(size_t)i];
        if (!equal_elem<Npp32s>(hist.host.data() + off, reinterpret_cast<const uint8_t *>(&w32),
                                0, 0)) {
          char b[340];
          std::snprintf(b, sizeof(b), "[%s] %s ch=%d bin=%d got=%d want=%d", id, c.id().c_str(),
                        ch, i, (int)hist.host[((size_t)ch * (maxlev - 1) + (size_t)i)],
                        (int)w32);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          break;
        }
      }
      if (cnt.mismatched >= 8) break;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// ---------------------------------------------------------------------------
// Integral images.  The destination is (w+1) x (h+1); the first row and
// column hold nVal (or nValSqr for the squared image) and each interior entry
// is the cumulative sum of the source over [0..x][0..y].
// ---------------------------------------------------------------------------
template <typename S, typename D, class Call>
void run_nppi_integral(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.pad / (int)sizeof(S) + 1, c.h, 1, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, 1, seed, c.regime, 0);
    s.dev.copyFromHost(s.host);
    Buf<D> d(c.w + 2, c.h + 1, 1, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    const D nval = (D)((double)(rng.next() % 9) + 1.0);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s.dptr()), s.step,
                        reinterpret_cast<D *>(d.dptr()), d.step, NppiSize{c.w, c.h}, nval, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    ++cnt.executed;
    const int ds = (int)(d.step / sizeof(D));
    for (int y = 0; y <= c.h; ++y)
      for (int x = 0; x <= c.w; ++x) {
        double want_d;
        if (x == 0 || y == 0) {
          want_d = (double)nval;
        } else {
          double acc = (double)nval;
          for (int j = 0; j < y; ++j)
            for (int i = 0; i < x; ++i)
              acc += (double)s.hptr()[(size_t)j * (s.step / sizeof(S)) + (size_t)i];
          want_d = acc;
        }
        D want = nppi_integral_cast<D>(want_d);
        ++cnt.compared;
        const size_t off = (size_t)y * d.step + (size_t)x * sizeof(D);
        if (!equal_elem<D>(d.host.data() + off, reinterpret_cast<const uint8_t *>(&want), 1e-6,
                           1e-4)) {
          char b[340];
          std::snprintf(b, sizeof(b), "[%s] %s (%d,%d) got=%s want=%s", id, c.id().c_str(), x, y,
                        dbg_elem<D>(d.host.data() + off).c_str(),
                        dbg_elem<D>(reinterpret_cast<const uint8_t *>(&want)).c_str());
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          break;
        }
        if (cnt.mismatched >= 8) break;
      }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, typename DS, typename DQ, class Call>
void run_nppi_sqr_integral(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + 1, c.h, 1, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, 1, seed, c.regime, 0);
    s.dev.copyFromHost(s.host);
    Buf<DS> d(c.w + 2, c.h + 1, 1, 0, 0x5A);
    Buf<DQ> q(c.w + 2, c.h + 1, 1, 0, 0x5A);
    d.dev.copyFromHost(d.host);
    q.dev.copyFromHost(q.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    const DS nval = (DS)((double)(rng.next() % 9) + 1.0);
    const DQ nvalsqr = (DQ)((double)(rng.next() % 9) + 1.0);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s.dptr()), s.step,
                        reinterpret_cast<DS *>(d.dptr()), d.step,
                        reinterpret_cast<DQ *>(q.dptr()), q.step, NppiSize{c.w, c.h}, nval,
                        nvalsqr, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    d.dev.copyToHost(d.host);
    q.dev.copyToHost(q.host);
    ++cnt.executed;
    for (int y = 0; y <= c.h; ++y)
      for (int x = 0; x <= c.w; ++x) {
        double want_d, want_q;
        if (x == 0 || y == 0) {
          want_d = (double)nval;
          want_q = (double)nvalsqr;
        } else {
          double acc = (double)nval, accq = (double)nvalsqr;
          for (int j = 0; j < y; ++j)
            for (int i = 0; i < x; ++i) {
              const double v = (double)s.hptr()[(size_t)j * (s.step / sizeof(S)) + (size_t)i];
              acc += v;
              accq += v * v;
            }
          want_d = acc;
          want_q = accq;
        }
        DS wd = nppi_integral_cast<DS>(want_d);
        DQ wq = nppi_integral_cast<DQ>(want_q);
        ++cnt.compared;
        const size_t offd = (size_t)y * d.step + (size_t)x * sizeof(DS);
        const size_t offq = (size_t)y * q.step + (size_t)x * sizeof(DQ);
        if (!equal_elem<DS>(d.host.data() + offd, reinterpret_cast<const uint8_t *>(&wd), 1e-6,
                            1e-4) ||
            !equal_elem<DQ>(q.host.data() + offq, reinterpret_cast<const uint8_t *>(&wq), 1e-6,
                            1e-4)) {
          char b[380];
          std::snprintf(b, sizeof(b), "[%s] %s (%d,%d) got=(%s,%s)", id, c.id().c_str(), x, y,
                        dbg_elem<DS>(d.host.data() + offd).c_str(),
                        dbg_elem<DQ>(q.host.data() + offq).c_str());
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          break;
        }
        if (cnt.mismatched >= 8) break;
      }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// CountInRange with per-channel bounds (C3 / AC4): bounds are host arrays and
// the counts are written to device memory.
template <typename S, int C, int A, class Call>
void run_nppi_count_range_multi(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, 0, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    s.dev.copyFromHost(s.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    S lo[4], hi[4];
    for (int ch = 0; ch < A; ++ch) {
      lo[ch] = regime_value<S>(rng, c.regime, (size_t)ch + 3);
      hi[ch] = regime_value<S>(rng, c.regime, (size_t)ch + 9);
      if constexpr (std::is_floating_point_v<S>) {
        if (lo[ch] > hi[ch]) std::swap(lo[ch], hi[ch]);
      } else if constexpr (std::is_same_v<S, Npp8s>) {
        if (lo[ch] > hi[ch]) std::swap(lo[ch], hi[ch]);
      } else {
        if (lo[ch] > hi[ch]) std::swap(lo[ch], hi[ch]);
      }
    }
    DeviceMemory<int> counts((size_t)A);
    std::vector<int> counts0((size_t)A, -1);
    counts.copyFromHost(counts0);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C)),
                        s.step, NppiSize{c.w, c.h}, counts.get(), lo, hi,
                        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    std::vector<int> got((size_t)A, -1);
    counts.copyToHost(got);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      long long want = 0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const S v = nppi_roi_at<S, C>(s, c, x, y, ch);
          if (v >= lo[ch] && v <= hi[ch]) ++want;
        }
      ++cnt.compared;
      if (got[(size_t)ch] != (int)want) {
        char b[340];
        std::snprintf(b, sizeof(b), "[%s] %s ch=%d got=%d want=%lld", id, c.id().c_str(), ch,
                      got[(size_t)ch], want);
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        break;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// Histogram GetBufferSize smoke test: nLevels may be scalar (C1) or an array
// (C3/C4/AC4); only non-negative sizes are asserted.
template <class Call>
void run_nppi_histogram_buffer_size(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const NppiSize rois[] = {{1, 1}, {3, 2}, {7, 5}, {33, 17}, {257, 129}};
  for (const NppiSize &r : rois) {
    int nl[3] = {2, 3, 4};
    int size = -1;
    NppStatus st = call(r, nl, &size, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    ++cnt.executed;
    ++cnt.compared;
    if (size < 0) {
      char b[240];
      std::snprintf(b, sizeof(b), "[%s] %dx%d negative buffer size %d", id, r.width, r.height, size);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Quality metrics: QualityIndex, MSE, PSNR and RectStdDev.
//   QualityIndex (per channel, NPP 13.4): means/variances use the sample
//   divisor (n-1); the (mean^2 + mean^2)(var + var) denominator and the
//   numerator means come from channel 0 while each channel's covariance uses
//   its own means.
//   PSNR = 10*log10(65025 / MSE); a zero MSE yields +inf.
// ---------------------------------------------------------------------------
template <typename S, int C, int A, class Call>
void run_nppi_quality_index(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    // Clamped extremes can leave near-zero variance, where the quality-index
    // ratio is numerically meaningless; use the wide regime instead.
    const Regime qr = c.regime == kExtremes ? kWide : c.regime;
    fill_regime(s1, C, seed, qr, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, qr, 1, false, false, finite);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Npp32f> out(A + 1, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)),
                        s1.step,
                        reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)),
                        s2.step, NppiSize{c.w, c.h}, reinterpret_cast<Npp32f *>(out.dptr()),
                        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    const double n = (double)c.w * c.h;
    double sum[4][5];
    for (int ch = 0; ch < A; ++ch)
      for (int k = 0; k < 5; ++k) sum[ch][k] = 0.0;
    for (int ch = 0; ch < A; ++ch)
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double a = (double)nppi_roi_at<S, C>(s1, c, x, y, ch);
          const double b = (double)nppi_roi_at<S, C>(s2, c, x, y, ch);
          sum[ch][0] += a;
          sum[ch][1] += b;
          sum[ch][2] += a * a;
          sum[ch][3] += b * b;
          sum[ch][4] += a * b;
        }
    const double ma = sum[0][0] / n, mb = sum[0][1] / n;
    const double va = (sum[0][2] - n * ma * ma) / (n - 1.0);
    const double vb = (sum[0][3] - n * mb * mb) / (n - 1.0);
    const double denom = (ma * ma + mb * mb) * (va + vb);
    // Near-zero variance makes the ratio numerically meaningless (and NPP's
    // own accumulation is order-dependent there), so skip those regimes.
    if (!(va > 0.0) || !(vb > 0.0) || !(denom > 1e-10)) {
      ++cnt.skipped;
      continue;
    }
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      const double ca = sum[ch][0] / n, cb = sum[ch][1] / n;
      const double cov = (sum[ch][4] - n * ca * cb) / (n - 1.0);
      const float want = (float)(4.0 * cov * ma * mb / denom);
      ++cnt.compared;
      const size_t off = (size_t)ch * sizeof(Npp32f);
      if (!equal_elem<Npp32f>(out.host.data() + off, reinterpret_cast<const uint8_t *>(&want),
                              1e-5, 1e-5)) {
        char b[340];
        std::snprintf(b, sizeof(b), "[%s] %s ch=%d got=%s want=%s", id, c.id().c_str(), ch,
                      dbg_elem<Npp32f>(out.host.data() + off).c_str(),
                      dbg_elem<Npp32f>(reinterpret_cast<const uint8_t *>(&want)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        break;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, bool Psnr, class Call>
void run_nppi_mse(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    fill_regime(s1, C, seed, c.regime, 0);
    fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Npp32f> out(A + 1, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)),
                        s1.step,
                        reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)),
                        s2.step, NppiSize{c.w, c.h}, reinterpret_cast<Npp32f *>(out.dptr()),
                        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    ++cnt.executed;
    for (int ch = 0; ch < A; ++ch) {
      double acc = 0.0;
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const double d = (double)nppi_roi_at<S, C>(s1, c, x, y, ch) -
                           (double)nppi_roi_at<S, C>(s2, c, x, y, ch);
          acc += d * d;
        }
      const double mse = acc / ((double)c.w * c.h);
      float want;
      if constexpr (Psnr) want = mse <= 0.0 ? std::numeric_limits<float>::infinity()
                                            : (float)(10.0 * std::log10(65025.0 / mse));
      else want = (float)mse;
      ++cnt.compared;
      const size_t off = (size_t)ch * sizeof(Npp32f);
      if (!equal_elem<Npp32f>(out.host.data() + off, reinterpret_cast<const uint8_t *>(&want),
                              1e-5, 1e-4)) {
        char b[340];
        std::snprintf(b, sizeof(b), "[%s] %s ch=%d got=%s want=%s", id, c.id().c_str(), ch,
                      dbg_elem<Npp32f>(out.host.data() + off).c_str(),
                      dbg_elem<Npp32f>(reinterpret_cast<const uint8_t *>(&want)).c_str());
        if (first.empty()) first = b;
        EXPECT_TRUE(false) << b;
        ++cnt.mismatched;
        break;
      }
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// RectStdDev reads precomputed integral images; both images are built here
// with a zero border.  oRect offsets are in integral-image coordinates.
template <typename S, typename DS, typename DQ, bool HasSf, class Call>
void run_nppi_rect_stddev(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w < 3 || c.h < 3) continue;
    const int w = c.w, h = c.h;
    Buf<S> sum((w + 1) * (h + 1), 1, 1, 0, 0);
    Buf<DQ> sqr((w + 1) * (h + 1), 1, 1, 0, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, 0, 0, (int)c.regime, 0);
    Rng rng(seed);
    std::vector<double> src((size_t)w * h);
    for (size_t i = 0; i < src.size(); ++i)
      src[i] = (double)(rng.next() % ((S)sizeof(S) == 1 ? 4 : 16));
    for (int y = 0; y <= h; ++y)
      for (int x = 0; x <= w; ++x) {
        double s = 0.0, q = 0.0;
        if (x > 0 && y > 0) {
          for (int j = 0; j < y; ++j)
            for (int i = 0; i < x; ++i) {
              const double v = src[(size_t)j * w + i];
              s += v;
              q += v * v;
            }
        }
        sum.hptr()[(size_t)y * (w + 1) + x] = (S)s;
        sqr.hptr()[(size_t)y * (w + 1) + x] = (DQ)q;
      }
    sum.dev.copyFromHost(sum.host);
    sqr.dev.copyFromHost(sqr.host);
    const int ow = w - 2, oh = h - 2;
    Buf<DS> dst(ow + 1, oh, 1, 0, 0x5A);
    dst.dev.copyFromHost(dst.host);
    const NppiRect rect{1, 1, 2, 2};
    const int sf = HasSf ? (int)(rng.next() % 3) : 0;
    npp_diff::clear_cuda_error();
    NppStatus st;
    if constexpr (HasSf) {
      st = call(reinterpret_cast<const S *>(sum.dptr()), (w + 1) * (int)sizeof(S),
                reinterpret_cast<const DQ *>(sqr.dptr()), (w + 1) * (int)sizeof(DQ),
                reinterpret_cast<DS *>(dst.dptr()), (int)dst.step, NppiSize{ow, oh}, rect,
                sf, ctx);
    } else {
      st = call(reinterpret_cast<const S *>(sum.dptr()), (w + 1) * (int)sizeof(S),
                reinterpret_cast<const DQ *>(sqr.dptr()), (w + 1) * (int)sizeof(DQ),
                reinterpret_cast<DS *>(dst.dptr()), (int)dst.step, NppiSize{ow, oh}, rect,
                ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    dst.dev.copyToHost(dst.host);
    ++cnt.executed;
    const int sstep = w + 1;
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) {
        const int x1 = x + rect.x, y1 = y + rect.y;
        const int x2 = x1 + rect.width, y2 = y1 + rect.height;
        const double s = (double)sum.hptr()[(size_t)y2 * sstep + x2] -
                         (double)sum.hptr()[(size_t)y1 * sstep + x2] -
                         (double)sum.hptr()[(size_t)y2 * sstep + x1] +
                         (double)sum.hptr()[(size_t)y1 * sstep + x1];
        const double q = (double)sqr.hptr()[(size_t)y2 * sstep + x2] -
                         (double)sqr.hptr()[(size_t)y1 * sstep + x2] -
                         (double)sqr.hptr()[(size_t)y2 * sstep + x1] +
                         (double)sqr.hptr()[(size_t)y1 * sstep + x1];
        const double count = (double)rect.width * rect.height;
        double var = (count * q - s * s) / (count * count);
        if (var < 0.0) var = 0.0;
        DS want;
        if constexpr (std::is_integral_v<DS>) {
          long long v = (long long)std::sqrt(var);
          if (HasSf) {
            // NPP truncates the square root, then scales half-to-even.
            if (sf > 0) v = (long long)std::llrint((double)v / (double)(1ll << sf));
            else if (sf < 0) v = (long long)((unsigned long long)v << (unsigned)(-sf));
          }
          want = (DS)v;
        } else {
          want = (DS)std::sqrt(var);
        }
        ++cnt.compared;
        const size_t off = (size_t)y * dst.step + (size_t)x * sizeof(DS);
        if (!equal_elem<DS>(dst.host.data() + off, reinterpret_cast<const uint8_t *>(&want), 1e-6,
                            1e-4)) {
          char b[360];
          std::snprintf(b, sizeof(b), "[%s] %s (%d,%d) got=%s want=%s", id, c.id().c_str(), x, y,
                        dbg_elem<DS>(dst.host.data() + off).c_str(),
                        dbg_elem<DS>(reinterpret_cast<const uint8_t *>(&want)).c_str());
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          break;
        }
        if (cnt.mismatched >= 8) break;
      }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

// nppiEvenLevelsHost_32s: level[i] = lower + ceil(i*(upper-lower)/(nLevels-1)).
template <class Call>
void run_nppi_even_levels(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  const int levels[] = {2, 3, 4, 5, 8, 17};
  const Npp32s ranges[][2] = {{0, 10}, {-50, 50}, {10, 90}, {0, 255}, {-1000, 1000}};
  for (int nl : levels)
    for (const auto &rg : ranges) {
      std::vector<Npp32s> out((size_t)nl, -12345);
      npp_diff::clear_cuda_error();
      NppStatus st = call(out.data(), nl, rg[0], rg[1], ctx);
      if (st != NPP_NO_ERROR) {
        ++cnt.skipped;
        if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
        continue;
      }
      ++cnt.executed;
      const long long span = (long long)rg[1] - (long long)rg[0];
      const long long den = nl - 1;
      for (int i = 0; i < nl; ++i) {
        const long long num = (long long)i * span;
        const long long q = num / den + ((num % den) != 0 ? 1 : 0);
        const Npp32s want = (Npp32s)((long long)rg[0] + q);
        ++cnt.compared;
        if (out[(size_t)i] != want) {
          char b[280];
          std::snprintf(b, sizeof(b), "[%s] nl=%d range=[%d,%d] i=%d got=%d want=%d", id, nl,
                        (int)rg[0], (int)rg[1], i, (int)out[(size_t)i], (int)want);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          break;
        }
      }
      if (cnt.mismatched >= 8) break;
    }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// Masked variants.  nMaskStep is a byte stride, the mask is one byte per
// pixel, and only pixels with a non-zero mask byte contribute.  C3CM variants
// additionally select a single 1-based channel of interest.
// ---------------------------------------------------------------------------
template <typename S, int C>
inline void nppi_fill_mask(Buf<Npp8u> &m) {
  for (size_t i = 0; i < m.host.size(); ++i)
    m.host[i] = (i % 3 == 0) ? 0 : (Npp8u)(i % 5 == 0 ? 1 : 255);
}

template <typename S, int C, int A, NppiStatKind K, typename Out, bool UseCOI, class Call>
void run_nppi_reduce1_masked(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Out>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, m.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    nppi_clamp_stat_inputs(s);
    nppi_fill_mask<S, C>(m);
    s.dev.copyFromHost(s.host);
    m.dev.copyFromHost(m.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    const int coi = C > 1 ? 1 + (int)(rng.next() % C) : 1;
    Buf<Out> out(4, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    const S *sp = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
    NppStatus st;
    if constexpr (UseCOI) {
      st = call(sp, s.step, mp, m.step, NppiSize{c.w, c.h}, coi,
                reinterpret_cast<Out *>(out.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    } else {
      st = call(sp, s.step, mp, m.step, NppiSize{c.w, c.h},
                reinterpret_cast<Out *>(out.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    const int ch = UseCOI ? coi - 1 : 0;
    double sum = 0.0, sq = 0.0, l1 = 0.0, inf = 0.0;
    long long n = 0;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        if (m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] == 0) continue;
        const double v = (double)nppi_roi_at<S, C>(s, c, x, y, ch);
        ++n;
        sum += v;
        sq += v * v;
        l1 += std::fabs(v);
        double av;
        if constexpr (K == NppiStatKind::NormInf && std::is_same_v<S, Npp16s> && C == 4) {
          av = (double)(uint16_t)(int16_t)nppi_roi_at<S, C>(s, c, x, y, ch);
        } else {
          av = std::fabs(v);
        }
        inf = std::max(inf, av);
      }
    if (n == 0) {
      // NPP leaves its max accumulator at -DBL_MAX when no pixel is masked
      // in; treat empty masks as an undefined case.
      ++cnt.skipped;
      continue;
    }
    double want_d = 0.0;
    if constexpr (K == NppiStatKind::Sum) want_d = sum;
    else if constexpr (K == NppiStatKind::Mean) want_d = n ? sum / (double)n : 0.0;
    else if constexpr (K == NppiStatKind::NormL1) want_d = l1;
    else if constexpr (K == NppiStatKind::NormL2) want_d = std::sqrt(sq);
    else want_d = inf;
    Out want{};
    if constexpr (std::is_integral_v<Out>) want = (Out)(long long)std::llrint(want_d);
    else want = (Out)want_d;
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<Out>(out.host.data(), reinterpret_cast<const uint8_t *>(&want), rtol, atol)) {
      char b[340];
      std::snprintf(b, sizeof(b), "[%s] %s got=%s want=%s", id, c.id().c_str(),
                    dbg_elem<Out>(out.host.data()).c_str(),
                    dbg_elem<Out>(reinterpret_cast<const uint8_t *>(&want)).c_str());
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, int A, NppiStatKind K, bool UseCOI, class Call>
void run_nppi_reduce2_masked(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  double rtol, atol;
  nppi_stat_tol_s<S, Npp64f>(rtol, atol);
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    fill_regime(s1, C, seed, c.regime, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1, false, false, finite);
    nppi_clamp_stat_inputs(s1);
    nppi_clamp_stat_inputs(s2);
    nppi_fill_mask<S, C>(m);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    m.dev.copyFromHost(m.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    const int coi = C > 1 ? 1 + (int)(rng.next() % C) : 1;
    Buf<Npp64f> out(1, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    const S *p1 = reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C));
    const S *p2 = reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C));
    const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
    NppStatus st;
    if constexpr (UseCOI) {
      st = call(p1, s1.step, p2, s2.step, mp, m.step, NppiSize{c.w, c.h}, coi,
                reinterpret_cast<Npp64f *>(out.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    } else {
      st = call(p1, s1.step, p2, s2.step, mp, m.step, NppiSize{c.w, c.h},
                reinterpret_cast<Npp64f *>(out.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    const int ch = UseCOI ? coi - 1 : 0;
    double dsum = 0.0, dsq = 0.0, dinf = 0.0, nsum = 0.0, nsq = 0.0, ninf = 0.0;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        if (m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] == 0) continue;
        const double a = (double)nppi_roi_at<S, C>(s1, c, x, y, ch);
        const double b = (double)nppi_roi_at<S, C>(s2, c, x, y, ch);
        const double d = a - b;
        dsum += std::fabs(d);
        dsq += d * d;
        dinf = std::max(dinf, std::fabs(d));
        nsum += std::fabs(b);
        nsq += b * b;
        if constexpr (K == NppiStatKind::NormRelInf && std::is_same_v<S, Npp16s> && C == 4) {
          ninf = std::max(ninf, (double)(uint16_t)(int16_t)nppi_roi_at<S, C>(s2, c, x, y, ch));
        } else {
          ninf = std::max(ninf, std::fabs(b));
        }
      }
    if constexpr (K == NppiStatKind::NormRelL1 || K == NppiStatKind::NormRelL2 ||
                  K == NppiStatKind::NormRelInf) {
      // Masked NormRel's zero-denominator convention differs from the
      // unmasked family on NPP; skip those degenerate cases.
      if (nsum == 0.0 && nsq == 0.0 && ninf == 0.0) {
        ++cnt.skipped;
        continue;
      }
    }
    if (dsum == 0.0 && dsq == 0.0 && dinf == 0.0 && nsum == 0.0 && nsq == 0.0 && ninf == 0.0) {
      ++cnt.skipped;
      continue;
    }
    double want = 0.0;
    if constexpr (K == NppiStatKind::NormDiffL1) want = dsum;
    else if constexpr (K == NppiStatKind::NormDiffL2) want = std::sqrt(dsq);
    else if constexpr (K == NppiStatKind::NormDiffInf) want = dinf;
    else if constexpr (K == NppiStatKind::NormRelL1)
      want = (nsum > 0.0) ? dsum / nsum : (dsum == 0.0 ? 0.0 : (double)std::numeric_limits<S>::max());
    else if constexpr (K == NppiStatKind::NormRelL2)
      want = (nsq > 0.0) ? std::sqrt(dsq) / std::sqrt(nsq)
                         : (dsq == 0.0 ? 0.0 : (double)std::numeric_limits<S>::max());
    else
      want = (ninf > 0.0) ? dinf / ninf : (dinf == 0.0 ? 0.0 : (double)std::numeric_limits<S>::max());
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<Npp64f>(out.host.data(), reinterpret_cast<const uint8_t *>(&want), rtol, atol)) {
      char b[340];
      std::snprintf(b, sizeof(b), "[%s] %s got=%s want=%s", id, c.id().c_str(),
                    dbg_elem<Npp64f>(out.host.data()).c_str(),
                    dbg_elem<Npp64f>(reinterpret_cast<const uint8_t *>(&want)).c_str());
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, bool UseCOI, class Call>
void run_nppi_meanstd_masked(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, m.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    nppi_clamp_stat_inputs(s);
    nppi_fill_mask<S, C>(m);
    s.dev.copyFromHost(s.host);
    m.dev.copyFromHost(m.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    const int coi = C > 1 ? 1 + (int)(rng.next() % C) : 1;
    Buf<Npp64f> mo(1, 1, 1, 0, 0x5A), so(1, 1, 1, 0, 0x5A);
    mo.dev.copyFromHost(mo.host);
    so.dev.copyFromHost(so.host);
    npp_diff::clear_cuda_error();
    const S *sp = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
    NppStatus st;
    if constexpr (UseCOI) {
      st = call(sp, s.step, mp, m.step, NppiSize{c.w, c.h}, coi,
                reinterpret_cast<Npp64f *>(mo.dptr()), reinterpret_cast<Npp64f *>(so.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    } else {
      st = call(sp, s.step, mp, m.step, NppiSize{c.w, c.h},
                reinterpret_cast<Npp64f *>(mo.dptr()), reinterpret_cast<Npp64f *>(so.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    mo.dev.copyToHost(mo.host);
    so.dev.copyToHost(so.host);
    const int ch = UseCOI ? coi - 1 : 0;
    double sum = 0.0, sq = 0.0;
    long long n = 0;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        if (m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] == 0) continue;
        const double v = (double)nppi_roi_at<S, C>(s, c, x, y, ch);
        ++n;
        sum += v;
        sq += v * v;
      }
    if (n == 0) {
      ++cnt.skipped;
      continue;
    }
    const double mean = n ? sum / (double)n : 0.0;
    double var = n ? sq / (double)n - mean * mean : 0.0;
    if (!(var > 0.0)) var = 0.0;
    const double sd = std::sqrt(var);
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<Npp64f>(mo.host.data(), reinterpret_cast<const uint8_t *>(&mean), 1e-9, 1e-7) ||
        !equal_elem<Npp64f>(so.host.data(), reinterpret_cast<const uint8_t *>(&sd), 1e-9, 1e-7)) {
      double gm = 0.0, gs = 0.0;
      std::memcpy(&gm, mo.host.data(), sizeof(gm));
      std::memcpy(&gs, so.host.data(), sizeof(gs));
      char b[340];
      std::snprintf(b, sizeof(b), "[%s] %s got=(%.17g,%.17g) want=(%.17g,%.17g)", id,
                    c.id().c_str(), gm, gs, mean, sd);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

template <typename S, int C, bool UseCOI, class Call>
void run_nppi_minmaxindx_masked(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  const bool finite = std::is_floating_point_v<S>;
  for (const GenCase &c : make_gen_cases(0)) {
    if (c.w * c.h < 2) continue;
    Buf<S> s(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<Npp8u> m(c.w + c.ox, c.h + c.oy, 1, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s.step, m.step, (int)c.regime, 0);
    fill_regime(s, C, seed, c.regime, 0, false, false, finite);
    nppi_clamp_stat_inputs(s);
    nppi_fill_unique_extrema<S, C>(s, c, C);
    nppi_fill_mask<S, C>(m);
    // Keep the unique extrema inside the mask.
    m.hptr()[(size_t)c.oy * m.step + (size_t)c.ox] = 255;
    m.hptr()[(size_t)(c.oy + c.h - 1) * m.step + (size_t)(c.ox + c.w - 1)] = 255;
    s.dev.copyFromHost(s.host);
    m.dev.copyFromHost(m.host);
    Rng rng(seed ^ 0x9e3779b97f4a7c15ull);
    const int coi = C > 1 ? 1 + (int)(rng.next() % C) : 1;
    Buf<S> mn(4, 1, 1, 0, 0x5A), mx(4, 1, 1, 0, 0x5A);
    Buf<Npp32s> mnx(4, 1, 1, 0, 0x5A), mny(4, 1, 1, 0, 0x5A);
    Buf<Npp32s> mxx(4, 1, 1, 0, 0x5A), mxy(4, 1, 1, 0, 0x5A);
    mn.dev.copyFromHost(mn.host); mx.dev.copyFromHost(mx.host);
    mnx.dev.copyFromHost(mnx.host); mny.dev.copyFromHost(mny.host);
    mxx.dev.copyFromHost(mxx.host); mxy.dev.copyFromHost(mxy.host);
    npp_diff::clear_cuda_error();
    const S *sp = reinterpret_cast<const S *>(s.dptr() + off_bytes<S>(c.oy, c.ox, s.step, C));
    const Npp8u *mp = m.dptr() + (size_t)c.oy * m.step + (size_t)c.ox;
    NppStatus st;
    if constexpr (UseCOI) {
      st = call(sp, s.step, mp, m.step, NppiSize{c.w, c.h}, coi,
                reinterpret_cast<S *>(mn.dptr()), reinterpret_cast<S *>(mx.dptr()),
                reinterpret_cast<NppiPoint *>(mnx.dptr()),
                reinterpret_cast<NppiPoint *>(mxx.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    } else {
      st = call(sp, s.step, mp, m.step, NppiSize{c.w, c.h},
                reinterpret_cast<S *>(mn.dptr()), reinterpret_cast<S *>(mx.dptr()),
                reinterpret_cast<NppiPoint *>(mnx.dptr()),
                reinterpret_cast<NppiPoint *>(mxx.dptr()),
                reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    }
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    mn.dev.copyToHost(mn.host); mx.dev.copyToHost(mx.host);
    mnx.dev.copyToHost(mnx.host); mny.dev.copyToHost(mny.host);
    mxx.dev.copyToHost(mxx.host); mxy.dev.copyToHost(mxy.host);
    const int ch = UseCOI ? coi - 1 : 0;
    bool have = false;
    S vmin{}, vmax{};
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        if (m.hptr()[(size_t)(c.oy + y) * m.step + (size_t)(c.ox + x)] == 0) continue;
        const S v = nppi_roi_at<S, C>(s, c, x, y, ch);
        if (!have) { vmin = vmax = v; minx = maxx = x; miny = maxy = y; have = true; continue; }
        if (v < vmin) { vmin = v; minx = x; miny = y; }
        if (v > vmax) { vmax = v; maxx = x; maxy = y; }
      }
    const NppiPoint *gp = reinterpret_cast<const NppiPoint *>(mnx.host.data());
    const NppiPoint *gq = reinterpret_cast<const NppiPoint *>(mxx.host.data());
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<S>(mn.host.data(), reinterpret_cast<const uint8_t *>(&vmin), 0, 0) ||
        !equal_elem<S>(mx.host.data(), reinterpret_cast<const uint8_t *>(&vmax), 0, 0) ||
        gp->x != minx || gp->y != miny || gq->x != maxx || gq->y != maxy) {
      char b[380];
      std::snprintf(b, sizeof(b), "[%s] %s got=(%d@%d,%d,%d@%d,%d) want=(%d@%d,%d,%d@%d,%d)", id,
                    c.id().c_str(), (int)mn.host[0], gp->x, gp->y, (int)mx.host[0], gq->x, gq->y,
                    (int)vmin, minx, miny, (int)vmax, maxx, maxy);
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// ---------------------------------------------------------------------------
// CrossCorr / SqrDistance.  The window origin and normalized statistics
// reproduce the 5090-measured NPP 13.4 contract: Full anchors at
// (px-(tw-1), py-(th-1)), Same at (px-tw/2, py-th/2), Valid at (px, py);
// out-of-ROI source samples contribute 0 while the template is always read in
// full.  NormLevel uses the padded sample count for its moments.  64f
// destinations round the denominator through float.
// ---------------------------------------------------------------------------
enum class XCorrMode { Full, Valid, Same };
enum class XCorrOp { Plain, Norm, NormLevel, NormLevelAdvanced, SqrDistance };

template <typename D> inline D xcorr_store(double v, int sf) {
  if constexpr (sizeof(D) == 1) {
    if (std::isnan(v)) return (D)0;
    const double scaled = std::trunc(std::ldexp(std::trunc(v), -sf));
    if (scaled < 0.0) return (D)0;
    if (scaled > 255.0) return (D)255;
    return (D)scaled;
  } else {
    return (D)v;
  }
}

template <typename S, typename D, int C, int A, XCorrMode M, XCorrOp Op, bool HasSf, bool HasBuf,
          bool Advanced, class Call>
void run_nppi_crosscorr(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  // The Advanced kernels work on large templates and need a per-source-pixel
  // double-precision work area (several planes, all channels).  A fixed 4 MB
  // buffer is silently overrun for 512x257 multi-channel cases: the context
  // survives but adjacent allocations are corrupted, which made results
  // depend on test order.  Size the work area from the largest case.
  size_t adv_bytes = 1;
  if constexpr (Advanced) {
    for (const GenCase &gc : make_gen_cases(0)) {
      const int gtw = std::min(gc.w, 2 + (int)(gc.param % 4) + 1);
      const int gth = std::min(gc.h, 2 + (int)((gc.pad / 16) % 3) + 1);
      const size_t need = 4 * sizeof(double) * (size_t)(gc.w + gtw + 2) *
                          (size_t)(gc.h + gth + 2) * (size_t)C;
      if (need > adv_bytes) adv_bytes = need;
    }
  }
  Buf<uint8_t> scratch2((int)adv_bytes, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  scratch2.dev.copyFromHost(scratch2.host);
  const bool finite = std::is_floating_point_v<S>;
  double rtol = 0.0, atol = 0.0;
  if constexpr (std::is_same_v<D, Npp32f>) { rtol = 1e-5; atol = 1e-5; }
  else if constexpr (std::is_same_v<D, Npp64f>) { rtol = 1e-10; atol = 1e-10; }
  for (const GenCase &c : make_gen_cases(0)) {
    const int sw = c.w, sh = c.h;
    const int tw = std::min(sw, 2 + (int)(c.param % 4) + 1);
    const int th = std::min(sh, 2 + (int)((c.pad / 16) % 3) + 1);
    // NPP's Advanced kernels read uninitialized advanced-scratch data when
    // the template is one column wide: the output changes with the scratch
    // prefill (measured), so there is no CPU formula to check.  The plain
    // NormLevel path is well defined; keep Advanced on tw >= 2.
    if constexpr (Advanced) {
      if (tw < 2) continue;
    }
    Buf<S> src(sw + (int)c.pad / (int)sizeof(S) + 1, sh, C, c.pad, 0);
    Buf<S> tpl(tw, th, C, 0, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, src.step, 0, (int)c.regime, c.param);
    fill_regime(src, C, seed, c.regime, 0, false, false, finite);
    fill_regime(tpl, C, seed ^ 0x51633e2dull, c.regime, 1, false, false, finite);
    // Extreme float magnitudes overflow NPP's single-precision accumulators
    // (NaN/0 results); clamp them like the other statistics drivers.
    nppi_clamp_stat_inputs(src);
    nppi_clamp_stat_inputs(tpl);
    if constexpr (std::is_floating_point_v<S>) {
      // NPP accumulates 32f norm statistics in single precision; denormal
      // inputs square to zero there and skew the ratio.  Flush them.
      auto flush_tiny = [](Buf<S> &b) {
        const size_t n = (size_t)(b.step / sizeof(S)) * (size_t)b.bh;
        for (size_t i = 0; i < n; ++i)
          if (std::isfinite((double)b.hptr()[i]) && std::fabs((double)b.hptr()[i]) < 1e-30)
            b.hptr()[i] = S(0);
      };
      flush_tiny(src);
      flush_tiny(tpl);
    }
    src.dev.copyFromHost(src.host);
    tpl.dev.copyFromHost(tpl.host);
    int ow, oh;
    if constexpr (M == XCorrMode::Full) { ow = sw + tw - 1; oh = sh + th - 1; }
    else if constexpr (M == XCorrMode::Same) { ow = sw; oh = sh; }
    else { ow = sw - tw + 1; oh = sh - th + 1; }
    if (ow <= 0 || oh <= 0) continue;
    Buf<D> dst(ow + 1, oh, C, c.pad, 0x5A);
    const std::vector<uint8_t> old_dst = dst.host;
    // Zero-variance windows make the normalized value mathematically
    // undefined.  NPP's 32f/64f kernels return the dtype maximum when the
    // float accumulation cancels exactly (template areas that are powers of
    // two) and otherwise leave epsilon-scale residue.  Accept either.
    std::vector<uint8_t> undef((size_t)ow * oh * C, 0);
    dst.dev.copyFromHost(dst.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(src.dptr()), src.step, NppiSize{sw, sh},
        reinterpret_cast<const S *>(tpl.dptr()), tpl.step, NppiSize{tw, th},
        reinterpret_cast<D *>(dst.dptr()), dst.step, HasSf ? (int)(c.param % 3) : 0,
        reinterpret_cast<Npp8u *>(scratch.dptr()),
        Advanced ? reinterpret_cast<Npp8u *>(scratch2.dptr()) : nullptr, ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    dst.dev.copyToHost(dst.host);
    std::vector<uint8_t> want = old_dst;
    const int sf = HasSf ? (int)(c.param % 3) : 0;
    for (int py = 0; py < oh; ++py)
      for (int px = 0; px < ow; ++px) {
        int wx, wy;
        if constexpr (M == XCorrMode::Full) { wx = px - (tw - 1); wy = py - (th - 1); }
        else if constexpr (M == XCorrMode::Same) { wx = px - tw / 2; wy = py - th / 2; }
        else { wx = px; wy = py; }
        for (int ci = 0; ci < A; ++ci) {
          double rst = 0.0, rss = 0.0, rtt = 0.0, swv = 0.0, stv = 0.0, sd = 0.0;
          for (int v = 0; v < th; ++v) {
            const int sy = wy + v;
            const bool row_in = (sy >= 0 && sy < sh);
            for (int u = 0; u < tw; ++u) {
              const double t = (double)tpl.hptr()[(size_t)v * (tpl.step / sizeof(S)) +
                                                 (size_t)u * C + ci];
              rtt += t * t;
              stv += t;
              double w = 0.0;
              const int sx = wx + u;
              if (row_in && sx >= 0 && sx < sw) {
                w = (double)src.hptr()[(size_t)sy * (src.step / sizeof(S)) + (size_t)sx * C + ci];
                swv += w;
                rss += w * w;
                rst += w * t;
              }
              if constexpr (Op == XCorrOp::SqrDistance) {
                const double d = w - t;
                sd += d * d;
              }
            }
          }
          double value = 0.0;
          if constexpr (Op == XCorrOp::Plain) {
            value = rst;
          } else if constexpr (Op == XCorrOp::Norm || Op == XCorrOp::SqrDistance) {
            double den;
            if constexpr (std::is_same_v<D, Npp64f>) den = (double)std::sqrt((float)(rss * rtt));
            else den = std::sqrt(rss * rtt);
            value = (Op == XCorrOp::SqrDistance ? sd : rst) / den;
          } else {
            const double n = (double)tw * (double)th;
            const double num = (n * rst - swv * stv) / n;
            const double var_w = (n * rss - swv * swv) / n;
            const double var_t = (n * rtt - stv * stv) / n;
            const double prod = var_w * var_t;
            if (prod == 0.0) {
              undef[((size_t)py * ow + px) * C + ci] = 1;
              value = sizeof(D) <= 4 ? (double)std::numeric_limits<float>::max()
                                     : (double)std::numeric_limits<double>::max();
            } else {
              double den;
              if constexpr (std::is_same_v<D, Npp64f>) den = (double)std::sqrt((float)prod);
              else den = std::sqrt(prod);
              value = num / den;
            }
          }
          const size_t off = (size_t)py * dst.step + ((size_t)px * C + ci) * sizeof(D);
          const D r = xcorr_store<D>(value, sf);
          std::memcpy(want.data() + off, &r, sizeof(D));
        }
      }
    ++cnt.executed;
    bool case_failed = false;
    for (int py = 0; py < oh && !case_failed; ++py)
      for (int px = 0; px < ow && !case_failed; ++px)
        for (int ci = 0; ci < C; ++ci) {
          const size_t off = (size_t)py * dst.step + ((size_t)px * C + ci) * sizeof(D);
          ++cnt.compared;
          bool ok;
          if (undef[((size_t)py * ow + px) * C + ci]) {
            D gv{};
            std::memcpy(&gv, dst.host.data() + off, sizeof(D));
            if constexpr (std::is_floating_point_v<D>) {
              const double gd = (double)gv;
              const double sentinel = sizeof(D) <= 4
                                          ? (double)std::numeric_limits<float>::max()
                                          : (double)std::numeric_limits<double>::max();
              ok = gd == sentinel || std::fabs(gd) <= 1e-6;
            } else {
              // Unsigned 8-bit destinations store either 0 (NaN folded by
              // the float->u8 conversion) or the dtype maximum (inf clamped).
              ok = gv == D(0) || gv == std::numeric_limits<D>::max();
            }
          } else if constexpr (std::is_floating_point_v<D>) {
            ok = equal_elem<D>(dst.host.data() + off, want.data() + off, rtol, atol);
          } else {
            // NPP accumulates the coefficient in float and then truncates.
            // When the exact value lands on an integer boundary the float
            // result can fall just below it, so a one-step lower truncation
            // is still the same value within float rounding.
            D gv{}, wv{};
            std::memcpy(&gv, dst.host.data() + off, sizeof(D));
            std::memcpy(&wv, want.data() + off, sizeof(D));
            ok = gv == wv || (long long)gv + 1 == (long long)wv;
          }
          if (!ok) {
            char b[380];
            std::snprintf(b, sizeof(b), "[%s] %s (%d,%d,ch=%d) got=%s want=%s", id, c.id().c_str(),
                          px, py, ci, dbg_elem<D>(dst.host.data() + off).c_str(),
                          dbg_elem<D>(want.data() + off).c_str());
            if (first.empty()) first = b;
            EXPECT_TRUE(false) << b;
            ++cnt.mismatched;
            case_failed = true;
            break;
          }
        }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}


// Complex error metrics: each sample contributes the complex modulus of the
// difference; relative variants divide by max(|a|, |b|), 0 when both are 0.
template <typename S, int C, int A, NppiStatKind K, class Call>
void run_nppi_error_complex(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  Buf<uint8_t> scratch(kNppiStatScratch, 1, 1, 0, 0);
  scratch.dev.copyFromHost(scratch.host);
  for (const GenCase &c : make_gen_cases(0)) {
    Buf<S> s1(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    Buf<S> s2(c.w + c.ox, c.h + c.oy, C, c.pad, 0);
    const uint64_t seed = case_seed(id, c.w, c.h, s1.step, s2.step, (int)c.regime, 0);
    fill_regime(s1, C, seed, c.regime, 0);
    fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1);
    s1.dev.copyFromHost(s1.host);
    s2.dev.copyFromHost(s2.host);
    Buf<Npp64f> out(1, 1, 1, 0, 0x5A);
    out.dev.copyFromHost(out.host);
    npp_diff::clear_cuda_error();
    NppStatus st = call(
        reinterpret_cast<const S *>(s1.dptr() + off_bytes<S>(c.oy, c.ox, s1.step, C)), s1.step,
        reinterpret_cast<const S *>(s2.dptr() + off_bytes<S>(c.oy, c.ox, s2.step, C)), s2.step,
        NppiSize{c.w, c.h}, reinterpret_cast<Npp64f *>(out.dptr()),
        reinterpret_cast<Npp8u *>(scratch.dptr()), ctx);
    if (st != NPP_NO_ERROR) {
      ++cnt.skipped;
      if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
      continue;
    }
    const cudaError_t se = cudaDeviceSynchronize();
    ASSERT_EQ(se, cudaSuccess) << "[" << id << "] " << c.id() << " sync: " << cudaGetErrorString(se);
    out.dev.copyToHost(out.host);
    const bool rel = (K == NppiStatKind::MaxRelErr || K == NppiStatKind::AvgRelErr);
    const bool avg = (K == NppiStatKind::AvgAbsErr || K == NppiStatKind::AvgRelErr);
    double acc = 0.0, mx = 0.0;
    for (int ch = 0; ch < A; ++ch)
      for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
          const S a = nppi_roi_at<S, C>(s1, c, x, y, ch);
          const S b = nppi_roi_at<S, C>(s2, c, x, y, ch);
          const double d = std::hypot((double)a.re - (double)b.re, (double)a.im - (double)b.im);
          double e = d;
          if (rel) {
            const double den = std::max(std::hypot((double)a.re, (double)a.im),
                                        std::hypot((double)b.re, (double)b.im));
            e = den > 0.0 ? d / den : 0.0;
          }
          acc += e;
          mx = std::max(mx, e);
        }
    const double want = avg ? acc / ((double)c.w * c.h * A) : mx;
    ++cnt.executed;
    ++cnt.compared;
    if (!equal_elem<Npp64f>(out.host.data(), reinterpret_cast<const uint8_t *>(&want), 1e-11, 1e-11)) {
      char b[340];
      std::snprintf(b, sizeof(b), "[%s] %s got=%s want=%s", id, c.id().c_str(),
                    dbg_elem<Npp64f>(out.host.data()).c_str(),
                    dbg_elem<Npp64f>(reinterpret_cast<const uint8_t *>(&want)).c_str());
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
    }
    if (cnt.mismatched >= 8) break;
  }
  finish(id, cnt, first);
}

}  // namespace num






