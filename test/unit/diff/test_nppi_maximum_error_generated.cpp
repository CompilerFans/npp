// Differential tests for nppiMaximumError / nppiMaximumRelativeError and
// their GetBufferHostSize helpers (package p1.3d).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiMaximumErrorDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiMaximumErrorDiff.*'
//
// MaximumError         = max |pSrc1 - pSrc2|                     over all samples
// MaximumRelativeError = max |pSrc1 - pSrc2| / max(|pSrc1|,|pSrc2|)
// Complex sources use the magnitudes |z1-z2| and max(|z1|,|z2|).
//
// The NPP documentation claims C3/C4 write a per-channel array while the
// public signature is a single `Npp64f *pError` in every layout; every case
// below therefore allocates FOUR output doubles, pre-fills them with the
// sentinel -12345.0 and records all four, so the golden decodes exactly how
// many doubles NVIDIA writes (1 scalar vs Channels per-channel values) and
// which slots stay untouched.
//
// The zero-denominator convention of MaximumRelativeError (denominator
// max(|a|,|b|) == 0, i.e. both samples zero) is NOT documented; the crafted
// probes below pin it.  Buffer-size helpers are only checked for status and
// sufficiency; exact bytes are not a contract.
//
// Max-vs-mean decoding: crafted ladders give distinct per-position values, so
// a max (expected) differs from a sum or mean of the same data.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

// Sentinel pre-fill for the 4-double output buffer: every metric result is
// non-negative, so a negative sentinel can never be a legitimate value.
constexpr double kSentinel = -12345.0;
constexpr size_t kOutDoubles = 4;

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

inline int align_step(int step, int esize) { return ((step + esize - 1) / esize) * esize; }

inline Npp8u *scratch() { return arena<Npp8u>(4 << 20).get(); }

// Sentinel-prefilled output buffer (record + verify both start identically).
DeviceMemory<Npp64f> &sentinel_output() {
  DeviceMemory<Npp64f> &o = arena<Npp64f>(kOutDoubles);
  o.copyFromHost(std::vector<Npp64f>(kOutDoubles, kSentinel));
  return o;
}

// fill_uniform leaves complex structs zeroed, so complex images get their own
// deterministic RNG fill (same element distributions as the real dtypes).
template <typename D> std::vector<D> make_image(std::size_t count, uint64_t seed) {
  if constexpr (std::is_same_v<D, Npp16sc> || std::is_same_v<D, Npp32sc> || std::is_same_v<D, Npp32fc>) {
    using E = decltype(D::re);
    std::vector<D> v(count);
    Rng rng(seed);
    for (auto &z : v) {
      if constexpr (std::is_integral_v<E>) {
        using U = std::make_unsigned_t<E>;
        z.re = static_cast<E>(rng.integer<U>(0, static_cast<U>(~U{0})));
        z.im = static_cast<E>(rng.integer<U>(0, static_cast<U>(~U{0})));
      } else {
        z.re = static_cast<E>(rng.unit_double() * 4.0 - 2.0);
        z.im = static_cast<E>(rng.unit_double() * 4.0 - 2.0);
      }
    }
    return v;
  } else {
    return make_uniform<D>(count, seed);
  }
}

// Two-source max-error metric (MaximumError and MaximumRelativeError share
// this signature).  srcCh selects the packed channel count; four sentinel
// doubles are recorded so the output layout is pinned by the golden.
template <typename D>
void twosrc_metric(const char *fname, const std::string &idBase, int width, int height, int pad, int srcCh,
                   uint64_t seed, NppStatus (*f)(const D *, int, const D *, int, NppiSize, Npp64f *, Npp8u *,
                                                 NppStreamContext),
                   double rtol = 1e-5, double atol = 1e-5, Npp8u *scratchOverride = nullptr) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + pad, se);
  const size_t elems = (size_t)height * (size_t)(step / se);
  std::vector<D> h1 = make_image<D>(elems, seed);
  std::vector<D> h2 = make_image<D>(elems, seed ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<D> &s1 = arena<D>(elems);
  DeviceMemory<D> &s2 = arena<D>(elems);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  DeviceMemory<Npp64f> &o = sentinel_output();
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st =
      f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(), scratchOverride ? scratchOverride : scratch(),
        current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), kOutDoubles, ValueKind::F64, rtol, atol);
  c.finish();
}

// Two-source case with caller-crafted host data and an explicit channel count
// (constant/ladder images pin the max semantics and the output layout; mixed
// zero samples pin the zero-denominator convention).  step = align_step(w*srcCh*se + 8, se).
template <typename D>
void twosrc_data(const char *fname, const std::string &id, const std::vector<D> &h1, const std::vector<D> &h2,
                 int width, int height, int srcCh,
                 NppStatus (*f)(const D *, int, const D *, int, NppiSize, Npp64f *, Npp8u *, NppStreamContext)) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + 8, se);
  ASSERT_EQ(h1.size(), (size_t)height * (size_t)(step / se));
  ASSERT_EQ(h2.size(), h1.size());
  DeviceMemory<D> &s1 = arena<D>(h1.size());
  DeviceMemory<D> &s2 = arena<D>(h2.size());
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  DeviceMemory<Npp64f> &o = sentinel_output();
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), kOutDoubles, ValueKind::F64);
  c.finish();
}

// Crafted constant image of a packed layout: every channel of every padded
// row holds `value` (padding bytes are never read).
template <typename D> std::vector<D> constant_image(int width, int height, int srcCh, D value) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + 8, se);
  std::vector<D> v((size_t)height * (size_t)(step / se), D{});
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < srcCh; ++c) v[(size_t)y * (step / se) + (size_t)x * srcCh + c] = value;
  return v;
}

// Crafted per-channel image: channel c of every pixel holds value_of(c).
// Distinct per-channel values decode a per-channel output array from the
// golden (a single scalar would hold only the overall maximum).
template <typename D, typename F> std::vector<D> channel_ladder(int width, int height, int srcCh, F &&value_of) {
  const int se = (int)sizeof(D);
  const int step = align_step(width * srcCh * se + 8, se);
  std::vector<D> v((size_t)height * (size_t)(step / se), D{});
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < srcCh; ++c) v[(size_t)y * (step / se) + (size_t)x * srcCh + c] = value_of(c);
  return v;
}

}  // namespace

class NppiMaximumErrorDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiMaximumError ===============================
// One representative per dtype x channel layout, plus 1x1 and 64x32-padded
// ROIs.  RNG images exercise the full dtype range (including negatives for
// the signed types) and, for complex, non-trivial |z1-z2| magnitudes.
TEST_F(NppiMaximumErrorDiff, MaximumError_Plain) {
  twosrc_metric<Npp8u>("nppiMaximumError_8u_C1R_Ctx", "maxErr/8u/C1R", 17, 13, 16, 1, 0xB01,
                       &nppiMaximumError_8u_C1R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumError_8u_C1R_Ctx", "maxErr/8u/C1R", 1, 1, 8, 1, 0xB02,
                       &nppiMaximumError_8u_C1R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumError_8u_C2R_Ctx", "maxErr/8u/C2R", 17, 13, 16, 2, 0xB03,
                       &nppiMaximumError_8u_C2R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumError_8u_C3R_Ctx", "maxErr/8u/C3R", 17, 13, 16, 3, 0xB04,
                       &nppiMaximumError_8u_C3R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumError_8u_C4R_Ctx", "maxErr/8u/C4R", 64, 32, 32, 4, 0xB05,
                       &nppiMaximumError_8u_C4R_Ctx);
  twosrc_metric<Npp8s>("nppiMaximumError_8s_C1R_Ctx", "maxErr/8s/C1R", 17, 13, 16, 1, 0xB11,
                       &nppiMaximumError_8s_C1R_Ctx);
  twosrc_metric<Npp8s>("nppiMaximumError_8s_C3R_Ctx", "maxErr/8s/C3R", 17, 13, 16, 3, 0xB12,
                       &nppiMaximumError_8s_C3R_Ctx);
  twosrc_metric<Npp8s>("nppiMaximumError_8s_C4R_Ctx", "maxErr/8s/C4R", 17, 13, 16, 4, 0xB13,
                       &nppiMaximumError_8s_C4R_Ctx);
  twosrc_metric<Npp16u>("nppiMaximumError_16u_C1R_Ctx", "maxErr/16u/C1R", 17, 13, 16, 1, 0xB21,
                        &nppiMaximumError_16u_C1R_Ctx);
  twosrc_metric<Npp16u>("nppiMaximumError_16u_C3R_Ctx", "maxErr/16u/C3R", 64, 32, 32, 3, 0xB22,
                        &nppiMaximumError_16u_C3R_Ctx);
  twosrc_metric<Npp16u>("nppiMaximumError_16u_C4R_Ctx", "maxErr/16u/C4R", 17, 13, 16, 4, 0xB23,
                        &nppiMaximumError_16u_C4R_Ctx);
  twosrc_metric<Npp16s>("nppiMaximumError_16s_C1R_Ctx", "maxErr/16s/C1R", 17, 13, 16, 1, 0xB31,
                        &nppiMaximumError_16s_C1R_Ctx);
  twosrc_metric<Npp16s>("nppiMaximumError_16s_C2R_Ctx", "maxErr/16s/C2R", 17, 13, 16, 2, 0xB32,
                        &nppiMaximumError_16s_C2R_Ctx);
  twosrc_metric<Npp16s>("nppiMaximumError_16s_C4R_Ctx", "maxErr/16s/C4R", 64, 32, 32, 4, 0xB33,
                        &nppiMaximumError_16s_C4R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumError_16sc_C1R_Ctx", "maxErr/16sc/C1R", 17, 13, 16, 1, 0xB41,
                         &nppiMaximumError_16sc_C1R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumError_16sc_C1R_Ctx", "maxErr/16sc/C1R", 1, 1, 8, 1, 0xB42,
                         &nppiMaximumError_16sc_C1R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumError_16sc_C3R_Ctx", "maxErr/16sc/C3R", 17, 13, 16, 3, 0xB43,
                         &nppiMaximumError_16sc_C3R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumError_16sc_C4R_Ctx", "maxErr/16sc/C4R", 17, 13, 16, 4, 0xB44,
                         &nppiMaximumError_16sc_C4R_Ctx);
  twosrc_metric<Npp32u>("nppiMaximumError_32u_C1R_Ctx", "maxErr/32u/C1R", 17, 13, 16, 1, 0xB51,
                        &nppiMaximumError_32u_C1R_Ctx);
  twosrc_metric<Npp32u>("nppiMaximumError_32u_C3R_Ctx", "maxErr/32u/C3R", 17, 13, 16, 3, 0xB52,
                        &nppiMaximumError_32u_C3R_Ctx);
  twosrc_metric<Npp32s>("nppiMaximumError_32s_C1R_Ctx", "maxErr/32s/C1R", 17, 13, 16, 1, 0xB61,
                        &nppiMaximumError_32s_C1R_Ctx);
  twosrc_metric<Npp32s>("nppiMaximumError_32s_C3R_Ctx", "maxErr/32s/C3R", 17, 13, 16, 3, 0xB62,
                        &nppiMaximumError_32s_C3R_Ctx);
  twosrc_metric<Npp32s>("nppiMaximumError_32s_C4R_Ctx", "maxErr/32s/C4R", 64, 32, 32, 4, 0xB63,
                        &nppiMaximumError_32s_C4R_Ctx);
  twosrc_metric<Npp32sc>("nppiMaximumError_32sc_C1R_Ctx", "maxErr/32sc/C1R", 17, 13, 16, 1, 0xB71,
                         &nppiMaximumError_32sc_C1R_Ctx);
  twosrc_metric<Npp32sc>("nppiMaximumError_32sc_C3R_Ctx", "maxErr/32sc/C3R", 17, 13, 16, 3, 0xB72,
                         &nppiMaximumError_32sc_C3R_Ctx);
  twosrc_metric<Npp32sc>("nppiMaximumError_32sc_C4R_Ctx", "maxErr/32sc/C4R", 17, 13, 16, 4, 0xB73,
                         &nppiMaximumError_32sc_C4R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumError_32f_C1R_Ctx", "maxErr/32f/C1R", 17, 13, 16, 1, 0xB81,
                        &nppiMaximumError_32f_C1R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumError_32f_C1R_Ctx", "maxErr/32f/C1R", 1, 1, 8, 1, 0xB82,
                        &nppiMaximumError_32f_C1R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumError_32f_C2R_Ctx", "maxErr/32f/C2R", 17, 13, 16, 2, 0xB83,
                        &nppiMaximumError_32f_C2R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumError_32f_C3R_Ctx", "maxErr/32f/C3R", 64, 32, 32, 3, 0xB84,
                        &nppiMaximumError_32f_C3R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumError_32f_C4R_Ctx", "maxErr/32f/C4R", 17, 13, 16, 4, 0xB85,
                        &nppiMaximumError_32f_C4R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumError_32fc_C1R_Ctx", "maxErr/32fc/C1R", 17, 13, 16, 1, 0xB91,
                         &nppiMaximumError_32fc_C1R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumError_32fc_C1R_Ctx", "maxErr/32fc/C1R", 1, 1, 8, 1, 0xB92,
                         &nppiMaximumError_32fc_C1R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumError_32fc_C3R_Ctx", "maxErr/32fc/C3R", 17, 13, 16, 3, 0xB93,
                         &nppiMaximumError_32fc_C3R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumError_32fc_C4R_Ctx", "maxErr/32fc/C4R", 17, 13, 16, 4, 0xB94,
                         &nppiMaximumError_32fc_C4R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumError_64f_C1R_Ctx", "maxErr/64f/C1R", 17, 13, 16, 1, 0xBa1,
                        &nppiMaximumError_64f_C1R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumError_64f_C2R_Ctx", "maxErr/64f/C2R", 17, 13, 16, 2, 0xBa2,
                        &nppiMaximumError_64f_C2R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumError_64f_C3R_Ctx", "maxErr/64f/C3R", 17, 13, 16, 3, 0xBa3,
                        &nppiMaximumError_64f_C3R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumError_64f_C4R_Ctx", "maxErr/64f/C4R", 17, 13, 16, 4, 0xBa4,
                        &nppiMaximumError_64f_C4R_Ctx);
}

// ======================= nppiMaximumRelativeError ===========================
TEST_F(NppiMaximumErrorDiff, MaximumRelativeError_Plain) {
  twosrc_metric<Npp8u>("nppiMaximumRelativeError_8u_C1R_Ctx", "maxRelErr/8u/C1R", 17, 13, 16, 1, 0xC01,
                       &nppiMaximumRelativeError_8u_C1R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumRelativeError_8u_C1R_Ctx", "maxRelErr/8u/C1R", 1, 1, 8, 1, 0xC02,
                       &nppiMaximumRelativeError_8u_C1R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumRelativeError_8u_C2R_Ctx", "maxRelErr/8u/C2R", 17, 13, 16, 2, 0xC03,
                       &nppiMaximumRelativeError_8u_C2R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumRelativeError_8u_C3R_Ctx", "maxRelErr/8u/C3R", 17, 13, 16, 3, 0xC04,
                       &nppiMaximumRelativeError_8u_C3R_Ctx);
  twosrc_metric<Npp8u>("nppiMaximumRelativeError_8u_C4R_Ctx", "maxRelErr/8u/C4R", 64, 32, 32, 4, 0xC05,
                       &nppiMaximumRelativeError_8u_C4R_Ctx);
  twosrc_metric<Npp8s>("nppiMaximumRelativeError_8s_C1R_Ctx", "maxRelErr/8s/C1R", 17, 13, 16, 1, 0xC11,
                       &nppiMaximumRelativeError_8s_C1R_Ctx);
  twosrc_metric<Npp8s>("nppiMaximumRelativeError_8s_C3R_Ctx", "maxRelErr/8s/C3R", 17, 13, 16, 3, 0xC12,
                       &nppiMaximumRelativeError_8s_C3R_Ctx);
  twosrc_metric<Npp8s>("nppiMaximumRelativeError_8s_C4R_Ctx", "maxRelErr/8s/C4R", 17, 13, 16, 4, 0xC13,
                       &nppiMaximumRelativeError_8s_C4R_Ctx);
  twosrc_metric<Npp16u>("nppiMaximumRelativeError_16u_C1R_Ctx", "maxRelErr/16u/C1R", 17, 13, 16, 1, 0xC21,
                        &nppiMaximumRelativeError_16u_C1R_Ctx);
  twosrc_metric<Npp16u>("nppiMaximumRelativeError_16u_C3R_Ctx", "maxRelErr/16u/C3R", 64, 32, 32, 3, 0xC22,
                        &nppiMaximumRelativeError_16u_C3R_Ctx);
  twosrc_metric<Npp16u>("nppiMaximumRelativeError_16u_C4R_Ctx", "maxRelErr/16u/C4R", 17, 13, 16, 4, 0xC23,
                        &nppiMaximumRelativeError_16u_C4R_Ctx);
  twosrc_metric<Npp16s>("nppiMaximumRelativeError_16s_C1R_Ctx", "maxRelErr/16s/C1R", 17, 13, 16, 1, 0xC31,
                        &nppiMaximumRelativeError_16s_C1R_Ctx);
  twosrc_metric<Npp16s>("nppiMaximumRelativeError_16s_C2R_Ctx", "maxRelErr/16s/C2R", 17, 13, 16, 2, 0xC32,
                        &nppiMaximumRelativeError_16s_C2R_Ctx);
  twosrc_metric<Npp16s>("nppiMaximumRelativeError_16s_C4R_Ctx", "maxRelErr/16s/C4R", 64, 32, 32, 4, 0xC33,
                        &nppiMaximumRelativeError_16s_C4R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumRelativeError_16sc_C1R_Ctx", "maxRelErr/16sc/C1R", 17, 13, 16, 1, 0xC41,
                         &nppiMaximumRelativeError_16sc_C1R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumRelativeError_16sc_C1R_Ctx", "maxRelErr/16sc/C1R", 1, 1, 8, 1, 0xC42,
                         &nppiMaximumRelativeError_16sc_C1R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumRelativeError_16sc_C3R_Ctx", "maxRelErr/16sc/C3R", 17, 13, 16, 3, 0xC43,
                         &nppiMaximumRelativeError_16sc_C3R_Ctx);
  twosrc_metric<Npp16sc>("nppiMaximumRelativeError_16sc_C4R_Ctx", "maxRelErr/16sc/C4R", 17, 13, 16, 4, 0xC44,
                         &nppiMaximumRelativeError_16sc_C4R_Ctx);
  twosrc_metric<Npp32u>("nppiMaximumRelativeError_32u_C1R_Ctx", "maxRelErr/32u/C1R", 17, 13, 16, 1, 0xC51,
                        &nppiMaximumRelativeError_32u_C1R_Ctx);
  twosrc_metric<Npp32u>("nppiMaximumRelativeError_32u_C3R_Ctx", "maxRelErr/32u/C3R", 17, 13, 16, 3, 0xC52,
                        &nppiMaximumRelativeError_32u_C3R_Ctx);
  twosrc_metric<Npp32s>("nppiMaximumRelativeError_32s_C1R_Ctx", "maxRelErr/32s/C1R", 17, 13, 16, 1, 0xC61,
                        &nppiMaximumRelativeError_32s_C1R_Ctx);
  twosrc_metric<Npp32s>("nppiMaximumRelativeError_32s_C3R_Ctx", "maxRelErr/32s/C3R", 17, 13, 16, 3, 0xC62,
                        &nppiMaximumRelativeError_32s_C3R_Ctx);
  twosrc_metric<Npp32s>("nppiMaximumRelativeError_32s_C4R_Ctx", "maxRelErr/32s/C4R", 64, 32, 32, 4, 0xC63,
                        &nppiMaximumRelativeError_32s_C4R_Ctx);
  twosrc_metric<Npp32sc>("nppiMaximumRelativeError_32sc_C1R_Ctx", "maxRelErr/32sc/C1R", 17, 13, 16, 1, 0xC71,
                         &nppiMaximumRelativeError_32sc_C1R_Ctx);
  twosrc_metric<Npp32sc>("nppiMaximumRelativeError_32sc_C3R_Ctx", "maxRelErr/32sc/C3R", 17, 13, 16, 3, 0xC72,
                         &nppiMaximumRelativeError_32sc_C3R_Ctx);
  twosrc_metric<Npp32sc>("nppiMaximumRelativeError_32sc_C4R_Ctx", "maxRelErr/32sc/C4R", 17, 13, 16, 4, 0xC73,
                         &nppiMaximumRelativeError_32sc_C4R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumRelativeError_32f_C1R_Ctx", "maxRelErr/32f/C1R", 17, 13, 16, 1, 0xC81,
                        &nppiMaximumRelativeError_32f_C1R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumRelativeError_32f_C1R_Ctx", "maxRelErr/32f/C1R", 1, 1, 8, 1, 0xC82,
                        &nppiMaximumRelativeError_32f_C1R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumRelativeError_32f_C2R_Ctx", "maxRelErr/32f/C2R", 17, 13, 16, 2, 0xC83,
                        &nppiMaximumRelativeError_32f_C2R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumRelativeError_32f_C3R_Ctx", "maxRelErr/32f/C3R", 64, 32, 32, 3, 0xC84,
                        &nppiMaximumRelativeError_32f_C3R_Ctx);
  twosrc_metric<Npp32f>("nppiMaximumRelativeError_32f_C4R_Ctx", "maxRelErr/32f/C4R", 17, 13, 16, 4, 0xC85,
                        &nppiMaximumRelativeError_32f_C4R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumRelativeError_32fc_C1R_Ctx", "maxRelErr/32fc/C1R", 17, 13, 16, 1, 0xC91,
                         &nppiMaximumRelativeError_32fc_C1R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumRelativeError_32fc_C1R_Ctx", "maxRelErr/32fc/C1R", 1, 1, 8, 1, 0xC92,
                         &nppiMaximumRelativeError_32fc_C1R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumRelativeError_32fc_C3R_Ctx", "maxRelErr/32fc/C3R", 17, 13, 16, 3, 0xC93,
                         &nppiMaximumRelativeError_32fc_C3R_Ctx);
  twosrc_metric<Npp32fc>("nppiMaximumRelativeError_32fc_C4R_Ctx", "maxRelErr/32fc/C4R", 17, 13, 16, 4, 0xC94,
                         &nppiMaximumRelativeError_32fc_C4R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumRelativeError_64f_C1R_Ctx", "maxRelErr/64f/C1R", 17, 13, 16, 1, 0xCa1,
                        &nppiMaximumRelativeError_64f_C1R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumRelativeError_64f_C2R_Ctx", "maxRelErr/64f/C2R", 17, 13, 16, 2, 0xCa2,
                        &nppiMaximumRelativeError_64f_C2R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumRelativeError_64f_C3R_Ctx", "maxRelErr/64f/C3R", 17, 13, 16, 3, 0xCa3,
                        &nppiMaximumRelativeError_64f_C3R_Ctx);
  twosrc_metric<Npp64f>("nppiMaximumRelativeError_64f_C4R_Ctx", "maxRelErr/64f/C4R", 17, 13, 16, 4, 0xCa4,
                        &nppiMaximumRelativeError_64f_C4R_Ctx);
}

// ===================== crafted constant / ladder cases ======================
// Constant images give exact expected values; per-channel ladders additionally
// decode a per-channel output array from the golden (a scalar output would
// hold only the overall maximum).  Opposite-sign sources drive the per-pixel
// ratio to its 2.0 maximum, pinning that no clamping is applied.
TEST_F(NppiMaximumErrorDiff, MaximumError_Crafted) {
  {  // 8u C4 2x1, all channels |10-6| = 4 -> 4.0 (or [4,4,4,4] per channel)
    twosrc_data<Npp8u>("nppiMaximumError_8u_C4R_Ctx", "maxErr/8u/C4R_const_2x1",
                       constant_image<Npp8u>(2, 1, 4, Npp8u{10}), constant_image<Npp8u>(2, 1, 4, Npp8u{6}), 2, 1, 4,
                       &nppiMaximumError_8u_C4R_Ctx);
  }
  {  // 8u C4 2x1 ladder: per-channel diffs 3,6,9,12 -> [3,6,9,12] or scalar 12
    twosrc_data<Npp8u>("nppiMaximumError_8u_C4R_Ctx", "maxErr/8u/C4R_ladder_2x1",
                       channel_ladder<Npp8u>(2, 1, 4, [](int c) { return Npp8u((c + 1) * 3); }),
                       constant_image<Npp8u>(2, 1, 4, Npp8u{0}), 2, 1, 4, &nppiMaximumError_8u_C4R_Ctx);
  }
  {  // 32f C3 2x1 ladder: per-channel diffs 5,10,15
    twosrc_data<Npp32f>("nppiMaximumError_32f_C3R_Ctx", "maxErr/32f/C3R_ladder_2x1",
                        channel_ladder<Npp32f>(2, 1, 3, [](int c) { return Npp32f((c + 1) * 5.0f); }),
                        constant_image<Npp32f>(2, 1, 3, Npp32f{0}), 2, 1, 3, &nppiMaximumError_32f_C3R_Ctx);
  }
  {  // 8u C2 3x1 const -> 4.0
    twosrc_data<Npp8u>("nppiMaximumError_8u_C2R_Ctx", "maxErr/8u/C2R_const_3x1", constant_image<Npp8u>(3, 1, 2, Npp8u{10}),
                       constant_image<Npp8u>(3, 1, 2, Npp8u{6}), 3, 1, 2, &nppiMaximumError_8u_C2R_Ctx);
  }
  {  // 16s C3 1x1, -5 vs 5 -> 10.0
    twosrc_data<Npp16s>("nppiMaximumError_16s_C3R_Ctx", "maxErr/16s/C3R_sign_1x1",
                        constant_image<Npp16s>(1, 1, 3, Npp16s{-5}), constant_image<Npp16s>(1, 1, 3, Npp16s{5}), 1, 1, 3,
                        &nppiMaximumError_16s_C3R_Ctx);
  }
  {  // 16s C1 1x1 full-range: -32768 vs 32767 -> 65535.0
    twosrc_data<Npp16s>("nppiMaximumError_16s_C1R_Ctx", "maxErr/16s/C1R_extreme_1x1",
                        constant_image<Npp16s>(1, 1, 1, Npp16s{-32768}),
                        constant_image<Npp16s>(1, 1, 1, Npp16s{32767}), 1, 1, 1, &nppiMaximumError_16s_C1R_Ctx);
  }
  {  // 32f C1 3x1: diffs 1.5, 5.0, 2.5 -> max 5.0 (a mean would be 3.0)
    std::vector<Npp32f> h1 = constant_image<Npp32f>(3, 1, 1, Npp32f{0});
    std::vector<Npp32f> h2 = constant_image<Npp32f>(3, 1, 1, Npp32f{0});
    h1[0] = 1.25f; h1[1] = -2.5f; h1[2] = 1.0f;
    h2[0] = -0.25f; h2[1] = 2.5f; h2[2] = -1.5f;
    twosrc_data<Npp32f>("nppiMaximumError_32f_C1R_Ctx", "maxErr/32f/C1R_sign_3x1", h1, h2, 3, 1, 1,
                        &nppiMaximumError_32f_C1R_Ctx);
  }
  {  // 32fc C1 2x1: |(3,4)-(0,0)| = 5 and |(0,0)-(3,-4)| = 5 -> 5.0
    std::vector<Npp32fc> h1 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    std::vector<Npp32fc> h2 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    h1[0] = Npp32fc{3.0f, 4.0f};
    h2[1] = Npp32fc{3.0f, -4.0f};
    twosrc_data<Npp32fc>("nppiMaximumError_32fc_C1R_Ctx", "maxErr/32fc/C1R_mag_2x1", h1, h2, 2, 1, 1,
                         &nppiMaximumError_32fc_C1R_Ctx);
  }
  {  // 16sc C2 1x1 ladder: channel magnitudes 5 and 10 -> [5,10] or scalar 10
    twosrc_data<Npp16sc>("nppiMaximumError_16sc_C2R_Ctx", "maxErr/16sc/C2R_ladder_1x1",
                         channel_ladder<Npp16sc>(1, 1, 2, [](int c) { return c == 0 ? Npp16sc{3, 4} : Npp16sc{6, 8}; }),
                         constant_image<Npp16sc>(1, 1, 2, Npp16sc{}), 1, 1, 2, &nppiMaximumError_16sc_C2R_Ctx);
  }
  {  // 64f C1 17x13: +1.5 vs -1.5 -> 3.0
    twosrc_data<Npp64f>("nppiMaximumError_64f_C1R_Ctx", "maxErr/64f/C1R_sign_17x13",
                        constant_image<Npp64f>(17, 13, 1, Npp64f{1.5}),
                        constant_image<Npp64f>(17, 13, 1, Npp64f{-1.5}), 17, 13, 1, &nppiMaximumError_64f_C1R_Ctx);
  }
  {  // 8u C1 all-zero 17x13 -> 0.0 (pins the reduction's initial value)
    twosrc_data<Npp8u>("nppiMaximumError_8u_C1R_Ctx", "maxErr/8u/C1R_allzero_17x13",
                       constant_image<Npp8u>(17, 13, 1, Npp8u{0}), constant_image<Npp8u>(17, 13, 1, Npp8u{0}), 17, 13, 1,
                       &nppiMaximumError_8u_C1R_Ctx);
  }
}

TEST_F(NppiMaximumErrorDiff, MaximumRelativeError_Crafted) {
  {  // 8u C4 2x1 const: 4/10 -> 0.4
    twosrc_data<Npp8u>("nppiMaximumRelativeError_8u_C4R_Ctx", "maxRelErr/8u/C4R_const_2x1",
                       constant_image<Npp8u>(2, 1, 4, Npp8u{10}), constant_image<Npp8u>(2, 1, 4, Npp8u{6}), 2, 1, 4,
                       &nppiMaximumRelativeError_8u_C4R_Ctx);
  }
  {  // 32f C4 2x1 ratio ladder: 0.5, 0.75, 2/3, 1.0 (last channel b=0)
    twosrc_data<Npp32f>("nppiMaximumRelativeError_32f_C4R_Ctx", "maxRelErr/32f/C4R_ladder_2x1",
                        channel_ladder<Npp32f>(2, 1, 4, [](int c) { return Npp32f((c + 1) * 10.0f); }),
                        channel_ladder<Npp32f>(2, 1, 4, [](int c) { return Npp32f(c * 5.0f); }), 2, 1, 4,
                        &nppiMaximumRelativeError_32f_C4R_Ctx);
  }
  {  // 32f C2 3x1 const: 1.5/2.5 -> 0.6
    twosrc_data<Npp32f>("nppiMaximumRelativeError_32f_C2R_Ctx", "maxRelErr/32f/C2R_const_3x1",
                        constant_image<Npp32f>(3, 1, 2, Npp32f{2.5}), constant_image<Npp32f>(3, 1, 2, Npp32f{1.0}), 3, 1,
                        2, &nppiMaximumRelativeError_32f_C2R_Ctx);
  }
  {  // 8s C1 17x13 opposite signs: |100-(-100)|/100 -> 2.0 exactly (no clamp)
    twosrc_data<Npp8s>("nppiMaximumRelativeError_8s_C1R_Ctx", "maxRelErr/8s/C1R_sign_17x13",
                       constant_image<Npp8s>(17, 13, 1, Npp8s{100}),
                       constant_image<Npp8s>(17, 13, 1, Npp8s{-100}), 17, 13, 1, &nppiMaximumRelativeError_8s_C1R_Ctx);
  }
  {  // 16s C1 1x1 extreme: |(-32768)-32767|/32768 = 65535/32768 (double math)
    twosrc_data<Npp16s>("nppiMaximumRelativeError_16s_C1R_Ctx", "maxRelErr/16s/C1R_extreme_1x1",
                        constant_image<Npp16s>(1, 1, 1, Npp16s{-32768}),
                        constant_image<Npp16s>(1, 1, 1, Npp16s{32767}), 1, 1, 1,
                        &nppiMaximumRelativeError_16s_C1R_Ctx);
  }
  {  // 64f C1 17x13: 3.0/1.5 -> 2.0
    twosrc_data<Npp64f>("nppiMaximumRelativeError_64f_C1R_Ctx", "maxRelErr/64f/C1R_sign_17x13",
                        constant_image<Npp64f>(17, 13, 1, Npp64f{1.5}),
                        constant_image<Npp64f>(17, 13, 1, Npp64f{-1.5}), 17, 13, 1,
                        &nppiMaximumRelativeError_64f_C1R_Ctx);
  }
  {  // 32fc C1 1x1: |(3,4)-(-3,-4)| / |(-3,-4)| = 10/5 -> 2.0
    twosrc_data<Npp32fc>("nppiMaximumRelativeError_32fc_C1R_Ctx", "maxRelErr/32fc/C1R_sign_1x1",
                         constant_image<Npp32fc>(1, 1, 1, Npp32fc{3.0f, 4.0f}),
                         constant_image<Npp32fc>(1, 1, 1, Npp32fc{-3.0f, -4.0f}), 1, 1, 1,
                         &nppiMaximumRelativeError_32fc_C1R_Ctx);
  }
  {  // 16sc C3 1x1: same complex opposite -> 2.0
    twosrc_data<Npp16sc>("nppiMaximumRelativeError_16sc_C3R_Ctx", "maxRelErr/16sc/C3R_sign_1x1",
                         constant_image<Npp16sc>(1, 1, 3, Npp16sc{3, 4}),
                         constant_image<Npp16sc>(1, 1, 3, Npp16sc{-3, -4}), 1, 1, 3,
                         &nppiMaximumRelativeError_16sc_C3R_Ctx);
  }
  {  // 32f C1 3x1: ratios 0.6, 2.0, 0.5 -> max 2.0 (a mean would be ~1.033)
    std::vector<Npp32f> h1 = constant_image<Npp32f>(3, 1, 1, Npp32f{0});
    std::vector<Npp32f> h2 = constant_image<Npp32f>(3, 1, 1, Npp32f{0});
    h1[0] = 2.5f; h1[1] = 1.5f; h1[2] = 1.0f;
    h2[0] = 1.0f; h2[1] = -1.5f; h2[2] = 0.5f;
    twosrc_data<Npp32f>("nppiMaximumRelativeError_32f_C1R_Ctx", "maxRelErr/32f/C1R_sign_3x1", h1, h2, 3, 1, 1,
                        &nppiMaximumRelativeError_32f_C1R_Ctx);
  }
}

// ===================== MaximumRelativeError zero denominator ================
// The per-sample denominator max(|a|,|b|) is zero only when a == b == 0.  The
// probes decode NVIDIA's convention from a single golden each:
//   sample 0 = (0,0)   zero denominator (0/0 sample)
//   sample 1 = (a,b)   known ratio (1.0 in every probe below)
// skip / count-as-0 -> golden 1.0 (a max cannot distinguish the two)
// dtype max         -> golden (huge)
// NaN               -> golden NaN
// The all-zero probes additionally pin the every-sample-zero case (0.0 vs
// dtype max vs NaN).
TEST_F(NppiMaximumErrorDiff, MaximumRelativeError_ZeroDenominator) {
  {  // 8u 2x1 mixed: [0 vs 0] and [5 vs 0]
    std::vector<Npp8u> h1 = constant_image<Npp8u>(2, 1, 1, Npp8u{0});
    std::vector<Npp8u> h2 = constant_image<Npp8u>(2, 1, 1, Npp8u{0});
    h1[1] = 5;
    twosrc_data<Npp8u>("nppiMaximumRelativeError_8u_C1R_Ctx", "maxRelErr/8u/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1, 1,
                       &nppiMaximumRelativeError_8u_C1R_Ctx);
  }
  {  // 8u all-zero 17x13
    twosrc_data<Npp8u>("nppiMaximumRelativeError_8u_C1R_Ctx", "maxRelErr/8u/C1R_allzero_17x13",
                       constant_image<Npp8u>(17, 13, 1, Npp8u{0}), constant_image<Npp8u>(17, 13, 1, Npp8u{0}), 17, 13, 1,
                       &nppiMaximumRelativeError_8u_C1R_Ctx);
  }
  {  // 16s 2x1 mixed with a negative sample: [0 vs 0] and [-5 vs 0]
    std::vector<Npp16s> h1 = constant_image<Npp16s>(2, 1, 1, Npp16s{0});
    std::vector<Npp16s> h2 = constant_image<Npp16s>(2, 1, 1, Npp16s{0});
    h1[1] = -5;
    twosrc_data<Npp16s>("nppiMaximumRelativeError_16s_C1R_Ctx", "maxRelErr/16s/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1, 1,
                        &nppiMaximumRelativeError_16s_C1R_Ctx);
  }
  {  // 32f all-zero 17x13 (float NaN / zero / FLT_MAX)
    twosrc_data<Npp32f>("nppiMaximumRelativeError_32f_C1R_Ctx", "maxRelErr/32f/C1R_allzero_17x13",
                        constant_image<Npp32f>(17, 13, 1, Npp32f{0}), constant_image<Npp32f>(17, 13, 1, Npp32f{0}), 17,
                        13, 1, &nppiMaximumRelativeError_32f_C1R_Ctx);
  }
  {  // 64f 2x1 mixed: [0 vs 0] and [1.5 vs 0]
    std::vector<Npp64f> h1 = constant_image<Npp64f>(2, 1, 1, Npp64f{0});
    std::vector<Npp64f> h2 = constant_image<Npp64f>(2, 1, 1, Npp64f{0});
    h1[1] = 1.5;
    twosrc_data<Npp64f>("nppiMaximumRelativeError_64f_C1R_Ctx", "maxRelErr/64f/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1, 1,
                        &nppiMaximumRelativeError_64f_C1R_Ctx);
  }
  {  // 32fc 2x1 mixed: [(0,0) vs (0,0)] and [(3,4) vs (0,0)]
    std::vector<Npp32fc> h1 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    std::vector<Npp32fc> h2 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    h1[1] = Npp32fc{3.0f, 4.0f};
    twosrc_data<Npp32fc>("nppiMaximumRelativeError_32fc_C1R_Ctx", "maxRelErr/32fc/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1,
                         1, &nppiMaximumRelativeError_32fc_C1R_Ctx);
  }
  {  // 16sc all-zero 17x13 (complex zero path)
    twosrc_data<Npp16sc>("nppiMaximumRelativeError_16sc_C1R_Ctx", "maxRelErr/16sc/C1R_allzero_17x13",
                         constant_image<Npp16sc>(17, 13, 1, Npp16sc{}), constant_image<Npp16sc>(17, 13, 1, Npp16sc{}),
                         17, 13, 1, &nppiMaximumRelativeError_16sc_C1R_Ctx);
  }
}

// ==================== buffer-size helpers (status only) =====================
TEST_F(NppiMaximumErrorDiff, BufferSize_Helpers) {
  SIZE_TYPE n = 0;
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiMaximumErrorGetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/maxErr_8u_C4R", "nppiMaximumErrorGetBufferHostSize_8u_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumErrorGetBufferHostSize_16s_C1R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/maxErr_16s_C1R", "nppiMaximumErrorGetBufferHostSize_16s_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumErrorGetBufferHostSize_32fc_C3R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/maxErr_32fc_C3R", "nppiMaximumErrorGetBufferHostSize_32fc_C3R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumErrorGetBufferHostSize_64f_C2R_Ctx(NppiSize{1, 1}, &n, ctx);
    Case c("bufferSize/maxErr_64f_C2R", "nppiMaximumErrorGetBufferHostSize_64f_C2R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/maxRelErr_8u_C1R", "nppiMaximumRelativeErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_16sc_C4R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/maxRelErr_16sc_C4R", "nppiMaximumRelativeErrorGetBufferHostSize_16sc_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_32f_C3R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/maxRelErr_32f_C3R", "nppiMaximumRelativeErrorGetBufferHostSize_32f_C3R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_32s_C2R_Ctx(NppiSize{1, 1}, &n, ctx);
    Case c("bufferSize/maxRelErr_32s_C2R", "nppiMaximumRelativeErrorGetBufferHostSize_32s_C2R_Ctx");
    c.status(st); c.finish();
  }
}

// Buffer-size helper contract: null out pointer, zero ROI, negative ROI.
// (p1.3c found the AverageError helpers accept a zero ROI; whether the
// MaximumError helpers do too is decoded from these goldens.)
TEST_F(NppiMaximumErrorDiff, BufferSize_Contract) {
  NppStreamContext ctx = current_context();
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{17, 13}, static_cast<SIZE_TYPE *>(nullptr), ctx);
    Case c("bufferSize/maxErr_contract/null_ptr", "nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{0, 2}, &n, ctx);
    Case c("bufferSize/maxErr_contract/zro_roi", "nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{-1, 2}, &n, ctx);
    Case c("bufferSize/maxErr_contract/neg_size", "nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_16u_C1R_Ctx(NppiSize{17, 13}, static_cast<SIZE_TYPE *>(nullptr), ctx);
    Case c("bufferSize/maxRelErr_contract/null_ptr", "nppiMaximumRelativeErrorGetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_16u_C1R_Ctx(NppiSize{0, 2}, &n, ctx);
    Case c("bufferSize/maxRelErr_contract/zro_roi", "nppiMaximumRelativeErrorGetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiMaximumRelativeErrorGetBufferHostSize_16u_C1R_Ctx(NppiSize{-1, 2}, &n, ctx);
    Case c("bufferSize/maxRelErr_contract/neg_size", "nppiMaximumRelativeErrorGetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
}

// ==================== buffer-size sufficiency ===============================
// Allocate exactly the reported scratch and run the matching reduction: the
// call must succeed and match the golden the 5090 recorded with NVIDIA's own
// buffer size.  Exact bytes are not a contract; sufficiency is.
TEST_F(NppiMaximumErrorDiff, BufferSize_Sufficiency) {
  NppStreamContext ctx = current_context();
  const NppiSize roi{64, 32};
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiMaximumErrorGetBufferHostSize_16u_C1R_Ctx(roi, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_metric<Npp16u>("nppiMaximumError_16u_C1R_Ctx", "maxErr/16u/C1R_exactbuf", 64, 32, 32, 1, 0xD01,
                          &nppiMaximumError_16u_C1R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiMaximumErrorGetBufferHostSize_32fc_C3R_Ctx(roi, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_metric<Npp32fc>("nppiMaximumError_32fc_C3R_Ctx", "maxErr/32fc/C3R_exactbuf", 64, 32, 32, 3, 0xD02,
                           &nppiMaximumError_32fc_C3R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiMaximumRelativeErrorGetBufferHostSize_8u_C1R_Ctx(roi, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_metric<Npp8u>("nppiMaximumRelativeError_8u_C1R_Ctx", "maxRelErr/8u/C1R_exactbuf", 64, 32, 32, 1, 0xD03,
                         &nppiMaximumRelativeError_8u_C1R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiMaximumRelativeErrorGetBufferHostSize_32sc_C4R_Ctx(NppiSize{17, 13}, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_metric<Npp32sc>("nppiMaximumRelativeError_32sc_C4R_Ctx", "maxRelErr/32sc/C4R_exactbuf", 17, 13, 16, 4, 0xD04,
                           &nppiMaximumRelativeError_32sc_C4R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
}

// ============================ non-Ctx adapters ==============================
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiMaximumErrorDiff, NonCtx_MaximumError) {
  // Reproduce the maxErr/8u/C1R_17x13 golden so both entry points agree.
  const int se = (int)sizeof(Npp8u);
  const int step = align_step(17 * se + 16, se);
  const size_t elems = (size_t)13 * (step / se);
  std::vector<Npp8u> h1 = make_image<Npp8u>(elems, 0xB01);
  std::vector<Npp8u> h2 = make_image<Npp8u>(elems, 0xB01 ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(elems);
  DeviceMemory<Npp64f> &o = sentinel_output();
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c("maxErr/8u/C1R_17x13", "nppiMaximumError_8u_C1R");  // reuses the golden id above
  clear_cuda_error();
  NppStatus st = nppiMaximumError_8u_C1R(s1.get(), step, s2.get(), step, NppiSize{17, 13}, o.get(), scratch());
  c.input(h1.data(), h1.size() * sizeof(Npp8u));
  c.input(h2.data(), h2.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output(o.get(), kOutDoubles, ValueKind::F64);
  c.finish();
}

TEST_F(NppiMaximumErrorDiff, NonCtx_MaximumRelativeError) {
  // Reproduce the maxRelErr/32f/C1R_17x13 golden so both entry points agree.
  const int se = (int)sizeof(Npp32f);
  const int step = align_step(17 * se + 16, se);
  const size_t elems = (size_t)13 * (step / se);
  std::vector<Npp32f> h1 = make_image<Npp32f>(elems, 0xC81);
  std::vector<Npp32f> h2 = make_image<Npp32f>(elems, 0xC81 ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<Npp32f> &s1 = arena<Npp32f>(elems);
  DeviceMemory<Npp32f> &s2 = arena<Npp32f>(elems);
  DeviceMemory<Npp64f> &o = sentinel_output();
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c("maxRelErr/32f/C1R_17x13", "nppiMaximumRelativeError_32f_C1R");  // reuses the golden id above
  clear_cuda_error();
  NppStatus st =
      nppiMaximumRelativeError_32f_C1R(s1.get(), step, s2.get(), step, NppiSize{17, 13}, o.get(), scratch());
  c.input(h1.data(), h1.size() * sizeof(Npp32f));
  c.input(h2.data(), h2.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output(o.get(), kOutDoubles, ValueKind::F64);
  c.finish();
}
#endif

// ============================ contract ======================================
// Illegal-input probes follow the P3.1b1 convention: only scenarios that do
// not fault the NVIDIA device are recorded (null src, null out, zero /
// negative ROI, short step).  Status is captured via Case so verify compares
// the golden status name returned by the 5090.  The zero-ROI case also
// records the sentinel output to pin whether anything is written.
TEST_F(NppiMaximumErrorDiff, Contract_MaximumError) {
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = sentinel_output();
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(nullptr, 16, s2.get(), 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("maxErr/contract/null_src1", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(s1.get(), 16, nullptr, 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("maxErr/contract/null_src2", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("maxErr/contract/null_out", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    DeviceMemory<Npp64f> &oz = sentinel_output();
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{0, 2}, oz.get(), scratch(), ctx);
    Case c("maxErr/contract/zro_roi", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st);
    c.device_output(oz.get(), kOutDoubles, ValueKind::F64);
    c.finish();
  }
  {
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("maxErr/contract/neg_size", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(s1.get(), 2, s2.get(), 16, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("maxErr/contract/short_step", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // multi-channel step contract: step < width*4*4
    DeviceMemory<Npp32f> &t1 = arena<Npp32f>(16);
    DeviceMemory<Npp32f> &t2 = arena<Npp32f>(16);
    NppStatus st = nppiMaximumError_32f_C4R_Ctx(t1.get(), 16, t2.get(), 32, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("maxErr/contract/short_step_c4", "nppiMaximumError_32f_C4R_Ctx");
    c.status(st); c.finish();
  }
  {  // null scratch is validated (like AverageError, unlike Sum/Mean)
    NppStatus st = nppiMaximumError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, o.get(), nullptr, ctx);
    Case c("maxErr/contract/null_buf", "nppiMaximumError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // step not a multiple of sizeof(Npp32f)*4 (C4 alignment) -> NPP_NOT_EVEN_STEP_ERROR
    DeviceMemory<Npp32f> &t1 = arena<Npp32f>(48);
    DeviceMemory<Npp32f> &t2 = arena<Npp32f>(48);
    NppStatus st = nppiMaximumError_32f_C4R_Ctx(t1.get(), 36, t2.get(), 36, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("maxErr/contract/noteven_step_c4", "nppiMaximumError_32f_C4R_Ctx");
    c.status(st); c.finish();
  }
}

TEST_F(NppiMaximumErrorDiff, Contract_MaximumRelativeError) {
  DeviceMemory<Npp16u> &s1 = arena<Npp16u>(32);
  DeviceMemory<Npp16u> &s2 = arena<Npp16u>(32);
  DeviceMemory<Npp64f> &o = sentinel_output();
  NppStreamContext ctx = current_context();
  {
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(nullptr, 8, s2.get(), 8, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("maxRelErr/contract/null_src1", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(s1.get(), 8, nullptr, 8, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("maxRelErr/contract/null_src2", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("maxRelErr/contract/null_out", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    DeviceMemory<Npp64f> &oz = sentinel_output();
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{0, 2}, oz.get(), scratch(), ctx);
    Case c("maxRelErr/contract/zro_roi", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st);
    c.device_output(oz.get(), kOutDoubles, ValueKind::F64);
    c.finish();
  }
  {
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("maxRelErr/contract/neg_size", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(s1.get(), 2, s2.get(), 8, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("maxRelErr/contract/short_step", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // null scratch is validated (like AverageError, unlike Sum/Mean)
    NppStatus st =
        nppiMaximumRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{2, 2}, o.get(), nullptr, ctx);
    Case c("maxRelErr/contract/null_buf", "nppiMaximumRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // step not a multiple of sizeof(Npp16sc)*2 (C2 alignment) -> NPP_NOT_EVEN_STEP_ERROR
    DeviceMemory<Npp16sc> &t1 = arena<Npp16sc>(32);
    DeviceMemory<Npp16sc> &t2 = arena<Npp16sc>(32);
    NppStatus st = nppiMaximumRelativeError_16sc_C2R_Ctx(t1.get(), 20, t2.get(), 20, NppiSize{2, 2}, o.get(),
                                                         scratch(), ctx);
    Case c("maxRelErr/contract/noteven_step_c2", "nppiMaximumRelativeError_16sc_C2R_Ctx");
    c.status(st); c.finish();
  }
}
