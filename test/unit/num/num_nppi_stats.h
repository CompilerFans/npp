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
    fill_regime(s1, C, seed, c.regime, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1, false, false, finite);
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
    fill_regime(s1, C, seed, c.regime, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1, false, false, finite);
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
    fill_regime(s1, C, seed, c.regime, 0, false, false, finite);
    fill_regime(s2, C, seed ^ 0x51633e2dull, c.regime, 1, false, false, finite);
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

}  // namespace num
