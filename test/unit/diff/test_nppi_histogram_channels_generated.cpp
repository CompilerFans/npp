// Differential tests for the package-s2 histogram variants that were still
// stubs: nppiHistogramEven_*_C3R/_AC4R and nppiHistogramRange_*_C3R/_AC4R.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiHistogramChannelsDiff.*'
//
// Binning pinned by the 5090 probes: bin k covers [level[k], level[k+1]) with
// the final level exclusive; Even levels are half-open with integer width
// (value - lower) * (nLevels - 1) / (upper - lower); alpha is ignored by the
// AC4R forms.  Only the null-source contract is exercised on the error side:
// NPP 13.4 faults on null histogram/level arrays, so those paths are not
// recorded.
//
// Scope note: the 16-bit Even C3R/AC4R entry points on the 5090 build alias the
// channel-0 and channel-2 histogram buffers (both receive the same accumulated
// bins), so no golden is recorded for them.  The 16-bit code path is exercised
// through the Range forms below; the 8-bit Even forms cover the even binning.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

template <typename T> std::vector<T> raster(int w, int h, int ch, uint64_t seed) {
  std::vector<T> v((size_t)w * h * ch);
  Rng rng(seed);
  for (auto &x : v) {
    if constexpr (std::is_floating_point_v<T>) {
      x = static_cast<T>(rng.unit_double() * 8.0 - 4.0);
    } else if constexpr (sizeof(T) == 1) {
      x = static_cast<T>(rng.integer<unsigned>(0, 255));
    } else if constexpr (std::is_signed_v<T>) {
      x = static_cast<T>(rng.integer<int>(-30000, 30000));
    } else {
      x = static_cast<T>(rng.integer<unsigned>(0, 65535));
    }
  }
  return v;
}

template <typename T>
void even_case(const char *fname, const std::string &id, int w, int h, int ch, bool alpha, int pad, uint64_t seed,
               const std::vector<int> &levels, const std::vector<Npp32s> &lo, const std::vector<Npp32s> &hi,
               NppStatus (*f)(const T *, int, NppiSize, Npp32s **, int *, Npp32s *, Npp32s *, Npp8u *,
                              NppStreamContext)) {
  const int se = (int)sizeof(T);
  const int step = ((w * ch * se + pad + se - 1) / se) * se;
  const std::vector<T> packed = raster<T>(w, h, ch, seed);
  std::vector<T> img((size_t)(step / se) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w * ch; ++x) img[(size_t)y * (step / se) + x] = packed[(size_t)y * w * ch + x];
  const int nch = alpha ? ch - 1 : ch;
  std::vector<std::vector<Npp32s>> bins(nch);
  std::vector<DeviceMemory<Npp32s>> hist;
  std::vector<Npp32s *> ptrs(nch);
  hist.reserve(nch);
  for (int c = 0; c < nch; ++c) {
    // A little slack: NPP's channel histogram kernels are documented to write
    // nLevels-1 bins, but keeping spare entries avoids any out-of-bounds write
    // from poisoning neighbouring allocations in a long gtest run.
    bins[c].assign((size_t)levels[c] + 64, (Npp32s)-1);
    hist.emplace_back(bins[c].size());
    hist.back().copyFromHost(bins[c]);
    ptrs[c] = hist.back().get();
  }
  DeviceMemory<T> d(img.size());
  d.copyFromHost(img);
  DeviceMemory<Npp8u> buf((size_t)1 << 20);
  clear_cuda_error();
  NppStatus st = f(d.get(), step, NppiSize{w, h}, ptrs.data(), const_cast<int *>(levels.data()),
                   const_cast<Npp32s *>(lo.data()), const_cast<Npp32s *>(hi.data()), buf.get(), current_context());
  Case c(id, fname);
  c.input(img.data(), img.size() * sizeof(T));
  c.status(st);
  if (st == NPP_NO_ERROR) {
    for (int k = 0; k < nch; ++k) c.device_output(ptrs[k], (size_t)(levels[k] - 1), ValueKind::I32);
  }
  c.finish();
}

template <typename SrcT, typename LevT>
void range_case(const char *fname, const std::string &id, int w, int h, int ch, bool alpha, int pad, uint64_t seed,
                const std::vector<std::vector<LevT>> &levels,
                NppStatus (*f)(const SrcT *, int, NppiSize, Npp32s **, const LevT **, int *, Npp8u *,
                               NppStreamContext)) {
  const int se = (int)sizeof(SrcT);
  const int step = ((w * ch * se + pad + se - 1) / se) * se;
  const std::vector<SrcT> packed = raster<SrcT>(w, h, ch, seed);
  std::vector<SrcT> img((size_t)(step / se) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w * ch; ++x) img[(size_t)y * (step / se) + x] = packed[(size_t)y * w * ch + x];
  const int nch = alpha ? ch - 1 : ch;
  std::vector<DeviceMemory<Npp32s>> hist(nch);
  std::vector<DeviceMemory<LevT>> lev(nch);
  std::vector<Npp32s *> histPtr(nch);
  std::vector<const LevT *> levPtr(nch);
  std::vector<int> nl(nch);
  for (int c = 0; c < nch; ++c) {
    std::vector<Npp32s> hb((size_t)levels[c].size() - 1 + 64, -1);
    hist[c].allocate(hb.size());
    hist[c].copyFromHost(hb);
    lev[c].allocate(levels[c].size());
    lev[c].copyFromHost(levels[c]);
    histPtr[c] = hist[c].get();
    levPtr[c] = lev[c].get();
    nl[c] = (int)levels[c].size();
  }
  DeviceMemory<SrcT> d(img.size());
  d.copyFromHost(img);
  DeviceMemory<Npp8u> buf((size_t)1 << 20);
  clear_cuda_error();
  NppStatus st = f(d.get(), step, NppiSize{w, h}, histPtr.data(), levPtr.data(), nl.data(), buf.get(),
                   current_context());
  Case c(id, fname);
  c.input(img.data(), img.size() * sizeof(SrcT));
  c.status(st);
  if (st == NPP_NO_ERROR) {
    for (int k = 0; k < nch; ++k) c.device_output(histPtr[k], (size_t)nl[k] - 1, ValueKind::I32);
  }
  c.finish();
}

}  // namespace

class NppiHistogramChannelsDiff : public NppTestBase {};

TEST_F(NppiHistogramChannelsDiff, Even_8u_C3) {
  const std::vector<int> lv{257, 257, 257};
  const std::vector<Npp32s> lo{0, 0, 0}, hi{256, 256, 256};
  even_case<Npp8u>("nppiHistogramEven_8u_C3R_Ctx", "s2/HistEven/8u_C3", 32, 16, 3, false, 0, 0x501, lv, lo, hi,
                   &nppiHistogramEven_8u_C3R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Even_8u_C3_Narrow) {
  const std::vector<int> lv{3, 5, 3};
  const std::vector<Npp32s> lo{64, 0, 100}, hi{192, 256, 256};
  even_case<Npp8u>("nppiHistogramEven_8u_C3R_Ctx", "s2/HistEven/8u_C3_narrow", 33, 9, 3, false, 4, 0x502, lv, lo, hi,
                   &nppiHistogramEven_8u_C3R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Even_8u_AC4) {
  const std::vector<int> lv{257, 257, 257};
  const std::vector<Npp32s> lo{0, 0, 0}, hi{256, 256, 256};
  even_case<Npp8u>("nppiHistogramEven_8u_AC4R_Ctx", "s2/HistEven/8u_AC4", 31, 17, 4, true, 8, 0x503, lv, lo, hi,
                   &nppiHistogramEven_8u_AC4R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Even_NullSrc) {
  NppStreamContext ctx = current_context();
  Npp32s hbuf[4] = {0, 0, 0, 0};
  Npp32s *hist[3] = {hbuf, hbuf, hbuf};
  int lv[3] = {4, 4, 4};
  Npp32s lo[3] = {0, 0, 0}, hi[3] = {256, 256, 256};
  DeviceMemory<Npp8u> buf((size_t)1 << 16);
  NppStatus st =
      nppiHistogramEven_8u_C3R_Ctx(nullptr, 24, NppiSize{8, 4}, hist, lv, lo, hi, buf.get(), ctx);
  Case c("s2/HistEven/err/null", "nppiHistogramEven_8u_C3R_Ctx");
  c.status(st);
  c.finish();
}

TEST_F(NppiHistogramChannelsDiff, Range_8u_C3) {
  std::vector<std::vector<Npp32s>> lv{{0, 64, 128, 192, 256}, {0, 85, 170, 256}, {0, 128, 129, 256}};
  range_case<Npp8u, Npp32s>("nppiHistogramRange_8u_C3R_Ctx", "s2/HistRange/8u_C3", 32, 16, 3, false, 0, 0x511, lv,
                     &nppiHistogramRange_8u_C3R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Range_8u_AC4) {
  std::vector<std::vector<Npp32s>> lv{{0, 64, 128, 192, 256}, {0, 100, 200, 256}, {0, 50, 150, 250, 256}};
  range_case<Npp8u, Npp32s>("nppiHistogramRange_8u_AC4R_Ctx", "s2/HistRange/8u_AC4", 29, 13, 4, true, 0, 0x512, lv,
                     &nppiHistogramRange_8u_AC4R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Range_16u_C3) {
  std::vector<std::vector<Npp32s>> lv{{0, 1000, 5000, 65536}, {0, 32768, 65536}, {0, 16, 32, 64, 65536}};
  range_case<Npp16u, Npp32s>("nppiHistogramRange_16u_C3R_Ctx", "s2/HistRange/16u_C3", 23, 9, 3, false, 0, 0x513, lv,
                     &nppiHistogramRange_16u_C3R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Range_16s_AC4) {
  std::vector<std::vector<Npp32s>> lv{{-1000, 0, 1000, 32768}, {-32768, 0, 32768}, {-100, 0, 100}};
  range_case<Npp16s, Npp32s>("nppiHistogramRange_16s_AC4R_Ctx", "s2/HistRange/16s_AC4", 21, 11, 4, true, 0, 0x514, lv,
                     &nppiHistogramRange_16s_AC4R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Range_32f_C3) {
  std::vector<std::vector<Npp32f>> lv{{-4.f, -1.f, 0.f, 1.f, 4.f}, {-2.f, 0.f, 2.f}, {-4.f, 0.f, 4.f}};
  range_case<Npp32f, Npp32f>("nppiHistogramRange_32f_C3R_Ctx", "s2/HistRange/32f_C3", 19, 7, 3, false, 6, 0x515, lv,
                     &nppiHistogramRange_32f_C3R_Ctx);
}
TEST_F(NppiHistogramChannelsDiff, Range_NullSrc) {
  Npp32s hbuf[4] = {0, 0, 0, 0};
  Npp32s *hist[3] = {hbuf, hbuf, hbuf};
  Npp32s lvbuf[4] = {0, 1, 2, 3};
  const Npp32s *lev[3] = {lvbuf, lvbuf, lvbuf};
  int lv[3] = {4, 4, 4};
  DeviceMemory<Npp8u> buf((size_t)1 << 16);
  NppStatus st =
      nppiHistogramRange_8u_C3R_Ctx(nullptr, 24, NppiSize{8, 4}, hist, lev, lv, buf.get(), current_context());
  Case c("s2/HistRange/err/null", "nppiHistogramRange_8u_C3R_Ctx");
  c.status(st);
  c.finish();
}
