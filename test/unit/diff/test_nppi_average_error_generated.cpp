// Differential tests for nppiAverageError / nppiAverageRelativeError and
// their GetBufferHostSize helpers (package p1.3c).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiAverageErrorDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiAverageErrorDiff.*'
//
// AverageError         = 1/(W*H*N) * sum |pSrc1 - pSrc2|          (single scalar)
// AverageRelativeError = 1/(W*H*N) * sum |pSrc1 - pSrc2| / max(|pSrc1|,|pSrc2|)
// Complex sources use the magnitudes |z1-z2| and max(|z1|,|z2|).  Both write a
// single device double regardless of the channel count.
//
// The per-pixel zero-denominator convention of AverageRelativeError (the
// denominator max(|a|,|b|) is zero only when both sources are zero) is NOT
// documented; the crafted probes below pin it against the 5090 golden before
// the executor convention is frozen.  Buffer-size helpers are only checked
// for status and sufficiency; exact bytes are not a contract.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

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

// Two-source scalar error metric (AverageError and AverageRelativeError share
// this signature).  srcCh selects the packed channel count; the output is a
// single Npp64f in all cases.  id is the golden id prefix; the ROI size is
// appended.
template <typename D>
void twosrc_scalar(const char *fname, const std::string &idBase, int width, int height, int pad, int srcCh,
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
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c(idBase + "_" + std::to_string(width) + "x" + std::to_string(height), fname);
  clear_cuda_error();
  NppStatus st =
      f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(), scratchOverride ? scratchOverride : scratch(),
        current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64, rtol, atol);
  c.finish();
}

// Two-source case with caller-crafted host data and an explicit channel count
// (constant images pin the W*H*N normalization; mixed zero pixels pin the
// zero-denominator convention).  step = align_step(width*srcCh*se + 8, se).
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
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s1.get(), step, s2.get(), step, NppiSize{width, height}, o.get(), scratch(), current_context());
  c.input(h1.data(), h1.size() * sizeof(D));
  c.input(h2.data(), h2.size() * sizeof(D));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
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

}  // namespace

class NppiAverageErrorDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiAverageError ===============================
// One representative per dtype x channel layout, plus 1x1 and 64x32-padded
// ROIs.  RNG images exercise the full dtype range (including negatives for
// the signed types) and, for complex, non-trivial |z1-z2| magnitudes.
TEST_F(NppiAverageErrorDiff, AverageError_Plain) {
  twosrc_scalar<Npp8u>("nppiAverageError_8u_C1R_Ctx", "avgErr/8u/C1R", 17, 13, 16, 1, 0x501,
                       &nppiAverageError_8u_C1R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageError_8u_C1R_Ctx", "avgErr/8u/C1R", 1, 1, 8, 1, 0x502,
                       &nppiAverageError_8u_C1R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageError_8u_C2R_Ctx", "avgErr/8u/C2R", 17, 13, 16, 2, 0x503,
                       &nppiAverageError_8u_C2R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageError_8u_C3R_Ctx", "avgErr/8u/C3R", 17, 13, 16, 3, 0x504,
                       &nppiAverageError_8u_C3R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageError_8u_C4R_Ctx", "avgErr/8u/C4R", 64, 32, 32, 4, 0x505,
                       &nppiAverageError_8u_C4R_Ctx);
  twosrc_scalar<Npp8s>("nppiAverageError_8s_C1R_Ctx", "avgErr/8s/C1R", 17, 13, 16, 1, 0x511,
                       &nppiAverageError_8s_C1R_Ctx);
  twosrc_scalar<Npp8s>("nppiAverageError_8s_C3R_Ctx", "avgErr/8s/C3R", 17, 13, 16, 3, 0x512,
                       &nppiAverageError_8s_C3R_Ctx);
  twosrc_scalar<Npp8s>("nppiAverageError_8s_C4R_Ctx", "avgErr/8s/C4R", 17, 13, 16, 4, 0x513,
                       &nppiAverageError_8s_C4R_Ctx);
  twosrc_scalar<Npp16u>("nppiAverageError_16u_C1R_Ctx", "avgErr/16u/C1R", 17, 13, 16, 1, 0x521,
                        &nppiAverageError_16u_C1R_Ctx);
  twosrc_scalar<Npp16u>("nppiAverageError_16u_C3R_Ctx", "avgErr/16u/C3R", 64, 32, 32, 3, 0x522,
                        &nppiAverageError_16u_C3R_Ctx);
  twosrc_scalar<Npp16u>("nppiAverageError_16u_C4R_Ctx", "avgErr/16u/C4R", 17, 13, 16, 4, 0x523,
                        &nppiAverageError_16u_C4R_Ctx);
  twosrc_scalar<Npp16s>("nppiAverageError_16s_C1R_Ctx", "avgErr/16s/C1R", 17, 13, 16, 1, 0x531,
                        &nppiAverageError_16s_C1R_Ctx);
  twosrc_scalar<Npp16s>("nppiAverageError_16s_C2R_Ctx", "avgErr/16s/C2R", 17, 13, 16, 2, 0x532,
                        &nppiAverageError_16s_C2R_Ctx);
  twosrc_scalar<Npp16s>("nppiAverageError_16s_C4R_Ctx", "avgErr/16s/C4R", 64, 32, 32, 4, 0x533,
                        &nppiAverageError_16s_C4R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageError_16sc_C1R_Ctx", "avgErr/16sc/C1R", 17, 13, 16, 1, 0x541,
                         &nppiAverageError_16sc_C1R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageError_16sc_C1R_Ctx", "avgErr/16sc/C1R", 1, 1, 8, 1, 0x542,
                         &nppiAverageError_16sc_C1R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageError_16sc_C3R_Ctx", "avgErr/16sc/C3R", 17, 13, 16, 3, 0x543,
                         &nppiAverageError_16sc_C3R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageError_16sc_C4R_Ctx", "avgErr/16sc/C4R", 17, 13, 16, 4, 0x544,
                         &nppiAverageError_16sc_C4R_Ctx);
  twosrc_scalar<Npp32u>("nppiAverageError_32u_C1R_Ctx", "avgErr/32u/C1R", 17, 13, 16, 1, 0x551,
                        &nppiAverageError_32u_C1R_Ctx);
  twosrc_scalar<Npp32u>("nppiAverageError_32u_C3R_Ctx", "avgErr/32u/C3R", 17, 13, 16, 3, 0x552,
                        &nppiAverageError_32u_C3R_Ctx);
  twosrc_scalar<Npp32s>("nppiAverageError_32s_C1R_Ctx", "avgErr/32s/C1R", 17, 13, 16, 1, 0x561,
                        &nppiAverageError_32s_C1R_Ctx);
  twosrc_scalar<Npp32s>("nppiAverageError_32s_C3R_Ctx", "avgErr/32s/C3R", 17, 13, 16, 3, 0x562,
                        &nppiAverageError_32s_C3R_Ctx);
  twosrc_scalar<Npp32s>("nppiAverageError_32s_C4R_Ctx", "avgErr/32s/C4R", 64, 32, 32, 4, 0x563,
                        &nppiAverageError_32s_C4R_Ctx);
  twosrc_scalar<Npp32sc>("nppiAverageError_32sc_C1R_Ctx", "avgErr/32sc/C1R", 17, 13, 16, 1, 0x571,
                         &nppiAverageError_32sc_C1R_Ctx);
  twosrc_scalar<Npp32sc>("nppiAverageError_32sc_C3R_Ctx", "avgErr/32sc/C3R", 17, 13, 16, 3, 0x572,
                         &nppiAverageError_32sc_C3R_Ctx);
  twosrc_scalar<Npp32sc>("nppiAverageError_32sc_C4R_Ctx", "avgErr/32sc/C4R", 17, 13, 16, 4, 0x573,
                         &nppiAverageError_32sc_C4R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageError_32f_C1R_Ctx", "avgErr/32f/C1R", 17, 13, 16, 1, 0x581,
                        &nppiAverageError_32f_C1R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageError_32f_C1R_Ctx", "avgErr/32f/C1R", 1, 1, 8, 1, 0x582,
                        &nppiAverageError_32f_C1R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageError_32f_C2R_Ctx", "avgErr/32f/C2R", 17, 13, 16, 2, 0x583,
                        &nppiAverageError_32f_C2R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageError_32f_C3R_Ctx", "avgErr/32f/C3R", 64, 32, 32, 3, 0x584,
                        &nppiAverageError_32f_C3R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageError_32f_C4R_Ctx", "avgErr/32f/C4R", 17, 13, 16, 4, 0x585,
                        &nppiAverageError_32f_C4R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageError_32fc_C1R_Ctx", "avgErr/32fc/C1R", 17, 13, 16, 1, 0x591,
                         &nppiAverageError_32fc_C1R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageError_32fc_C1R_Ctx", "avgErr/32fc/C1R", 1, 1, 8, 1, 0x592,
                         &nppiAverageError_32fc_C1R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageError_32fc_C3R_Ctx", "avgErr/32fc/C3R", 17, 13, 16, 3, 0x593,
                         &nppiAverageError_32fc_C3R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageError_32fc_C4R_Ctx", "avgErr/32fc/C4R", 17, 13, 16, 4, 0x594,
                         &nppiAverageError_32fc_C4R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageError_64f_C1R_Ctx", "avgErr/64f/C1R", 17, 13, 16, 1, 0x5a1,
                        &nppiAverageError_64f_C1R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageError_64f_C2R_Ctx", "avgErr/64f/C2R", 17, 13, 16, 2, 0x5a2,
                        &nppiAverageError_64f_C2R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageError_64f_C3R_Ctx", "avgErr/64f/C3R", 17, 13, 16, 3, 0x5a3,
                        &nppiAverageError_64f_C3R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageError_64f_C4R_Ctx", "avgErr/64f/C4R", 17, 13, 16, 4, 0x5a4,
                        &nppiAverageError_64f_C4R_Ctx);
}

// ======================= nppiAverageRelativeError ===========================
TEST_F(NppiAverageErrorDiff, AverageRelativeError_Plain) {
  twosrc_scalar<Npp8u>("nppiAverageRelativeError_8u_C1R_Ctx", "avgRelErr/8u/C1R", 17, 13, 16, 1, 0x601,
                       &nppiAverageRelativeError_8u_C1R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageRelativeError_8u_C1R_Ctx", "avgRelErr/8u/C1R", 1, 1, 8, 1, 0x602,
                       &nppiAverageRelativeError_8u_C1R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageRelativeError_8u_C2R_Ctx", "avgRelErr/8u/C2R", 17, 13, 16, 2, 0x603,
                       &nppiAverageRelativeError_8u_C2R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageRelativeError_8u_C3R_Ctx", "avgRelErr/8u/C3R", 17, 13, 16, 3, 0x604,
                       &nppiAverageRelativeError_8u_C3R_Ctx);
  twosrc_scalar<Npp8u>("nppiAverageRelativeError_8u_C4R_Ctx", "avgRelErr/8u/C4R", 64, 32, 32, 4, 0x605,
                       &nppiAverageRelativeError_8u_C4R_Ctx);
  twosrc_scalar<Npp8s>("nppiAverageRelativeError_8s_C1R_Ctx", "avgRelErr/8s/C1R", 17, 13, 16, 1, 0x611,
                       &nppiAverageRelativeError_8s_C1R_Ctx);
  twosrc_scalar<Npp8s>("nppiAverageRelativeError_8s_C3R_Ctx", "avgRelErr/8s/C3R", 17, 13, 16, 3, 0x612,
                       &nppiAverageRelativeError_8s_C3R_Ctx);
  twosrc_scalar<Npp8s>("nppiAverageRelativeError_8s_C4R_Ctx", "avgRelErr/8s/C4R", 17, 13, 16, 4, 0x613,
                       &nppiAverageRelativeError_8s_C4R_Ctx);
  twosrc_scalar<Npp16u>("nppiAverageRelativeError_16u_C1R_Ctx", "avgRelErr/16u/C1R", 17, 13, 16, 1, 0x621,
                        &nppiAverageRelativeError_16u_C1R_Ctx);
  twosrc_scalar<Npp16u>("nppiAverageRelativeError_16u_C3R_Ctx", "avgRelErr/16u/C3R", 64, 32, 32, 3, 0x622,
                        &nppiAverageRelativeError_16u_C3R_Ctx);
  twosrc_scalar<Npp16u>("nppiAverageRelativeError_16u_C4R_Ctx", "avgRelErr/16u/C4R", 17, 13, 16, 4, 0x623,
                        &nppiAverageRelativeError_16u_C4R_Ctx);
  twosrc_scalar<Npp16s>("nppiAverageRelativeError_16s_C1R_Ctx", "avgRelErr/16s/C1R", 17, 13, 16, 1, 0x631,
                        &nppiAverageRelativeError_16s_C1R_Ctx);
  twosrc_scalar<Npp16s>("nppiAverageRelativeError_16s_C2R_Ctx", "avgRelErr/16s/C2R", 17, 13, 16, 2, 0x632,
                        &nppiAverageRelativeError_16s_C2R_Ctx);
  twosrc_scalar<Npp16s>("nppiAverageRelativeError_16s_C4R_Ctx", "avgRelErr/16s/C4R", 64, 32, 32, 4, 0x633,
                        &nppiAverageRelativeError_16s_C4R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageRelativeError_16sc_C1R_Ctx", "avgRelErr/16sc/C1R", 17, 13, 16, 1, 0x641,
                         &nppiAverageRelativeError_16sc_C1R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageRelativeError_16sc_C1R_Ctx", "avgRelErr/16sc/C1R", 1, 1, 8, 1, 0x642,
                         &nppiAverageRelativeError_16sc_C1R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageRelativeError_16sc_C3R_Ctx", "avgRelErr/16sc/C3R", 17, 13, 16, 3, 0x643,
                         &nppiAverageRelativeError_16sc_C3R_Ctx);
  twosrc_scalar<Npp16sc>("nppiAverageRelativeError_16sc_C4R_Ctx", "avgRelErr/16sc/C4R", 17, 13, 16, 4, 0x644,
                         &nppiAverageRelativeError_16sc_C4R_Ctx);
  twosrc_scalar<Npp32u>("nppiAverageRelativeError_32u_C1R_Ctx", "avgRelErr/32u/C1R", 17, 13, 16, 1, 0x651,
                        &nppiAverageRelativeError_32u_C1R_Ctx);
  twosrc_scalar<Npp32u>("nppiAverageRelativeError_32u_C3R_Ctx", "avgRelErr/32u/C3R", 17, 13, 16, 3, 0x652,
                        &nppiAverageRelativeError_32u_C3R_Ctx);
  twosrc_scalar<Npp32s>("nppiAverageRelativeError_32s_C1R_Ctx", "avgRelErr/32s/C1R", 17, 13, 16, 1, 0x661,
                        &nppiAverageRelativeError_32s_C1R_Ctx);
  twosrc_scalar<Npp32s>("nppiAverageRelativeError_32s_C3R_Ctx", "avgRelErr/32s/C3R", 17, 13, 16, 3, 0x662,
                        &nppiAverageRelativeError_32s_C3R_Ctx);
  twosrc_scalar<Npp32s>("nppiAverageRelativeError_32s_C4R_Ctx", "avgRelErr/32s/C4R", 64, 32, 32, 4, 0x663,
                        &nppiAverageRelativeError_32s_C4R_Ctx);
  twosrc_scalar<Npp32sc>("nppiAverageRelativeError_32sc_C1R_Ctx", "avgRelErr/32sc/C1R", 17, 13, 16, 1, 0x671,
                         &nppiAverageRelativeError_32sc_C1R_Ctx);
  twosrc_scalar<Npp32sc>("nppiAverageRelativeError_32sc_C3R_Ctx", "avgRelErr/32sc/C3R", 17, 13, 16, 3, 0x672,
                         &nppiAverageRelativeError_32sc_C3R_Ctx);
  twosrc_scalar<Npp32sc>("nppiAverageRelativeError_32sc_C4R_Ctx", "avgRelErr/32sc/C4R", 17, 13, 16, 4, 0x673,
                         &nppiAverageRelativeError_32sc_C4R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageRelativeError_32f_C1R_Ctx", "avgRelErr/32f/C1R", 17, 13, 16, 1, 0x681,
                        &nppiAverageRelativeError_32f_C1R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageRelativeError_32f_C1R_Ctx", "avgRelErr/32f/C1R", 1, 1, 8, 1, 0x682,
                        &nppiAverageRelativeError_32f_C1R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageRelativeError_32f_C2R_Ctx", "avgRelErr/32f/C2R", 17, 13, 16, 2, 0x683,
                        &nppiAverageRelativeError_32f_C2R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageRelativeError_32f_C3R_Ctx", "avgRelErr/32f/C3R", 64, 32, 32, 3, 0x684,
                        &nppiAverageRelativeError_32f_C3R_Ctx);
  twosrc_scalar<Npp32f>("nppiAverageRelativeError_32f_C4R_Ctx", "avgRelErr/32f/C4R", 17, 13, 16, 4, 0x685,
                        &nppiAverageRelativeError_32f_C4R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageRelativeError_32fc_C1R_Ctx", "avgRelErr/32fc/C1R", 17, 13, 16, 1, 0x691,
                         &nppiAverageRelativeError_32fc_C1R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageRelativeError_32fc_C1R_Ctx", "avgRelErr/32fc/C1R", 1, 1, 8, 1, 0x692,
                         &nppiAverageRelativeError_32fc_C1R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageRelativeError_32fc_C3R_Ctx", "avgRelErr/32fc/C3R", 17, 13, 16, 3, 0x693,
                         &nppiAverageRelativeError_32fc_C3R_Ctx);
  twosrc_scalar<Npp32fc>("nppiAverageRelativeError_32fc_C4R_Ctx", "avgRelErr/32fc/C4R", 17, 13, 16, 4, 0x694,
                         &nppiAverageRelativeError_32fc_C4R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageRelativeError_64f_C1R_Ctx", "avgRelErr/64f/C1R", 17, 13, 16, 1, 0x6a1,
                        &nppiAverageRelativeError_64f_C1R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageRelativeError_64f_C2R_Ctx", "avgRelErr/64f/C2R", 17, 13, 16, 2, 0x6a2,
                        &nppiAverageRelativeError_64f_C2R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageRelativeError_64f_C3R_Ctx", "avgRelErr/64f/C3R", 17, 13, 16, 3, 0x6a3,
                        &nppiAverageRelativeError_64f_C3R_Ctx);
  twosrc_scalar<Npp64f>("nppiAverageRelativeError_64f_C4R_Ctx", "avgRelErr/64f/C4R", 17, 13, 16, 4, 0x6a4,
                        &nppiAverageRelativeError_64f_C4R_Ctx);
}

// ===================== crafted constant / sign cases ========================
// Constant images give exact expected values, pinning the W*H*N normalization
// (a per-channel average, an unnormalized sum, or a division by W*H only would
// all differ).  Opposite-sign sources drive the per-pixel ratio to its 2.0
// maximum, pinning that no clamping is applied.
TEST_F(NppiAverageErrorDiff, AverageError_Crafted) {
  {  // 8u C4 2x1, all channels |10-6|=4 -> 4.0
    twosrc_data<Npp8u>("nppiAverageError_8u_C4R_Ctx", "avgErr/8u/C4R_const_2x1", constant_image<Npp8u>(2, 1, 4, Npp8u{10}),
                       constant_image<Npp8u>(2, 1, 4, Npp8u{6}), 2, 1, 4, &nppiAverageError_8u_C4R_Ctx);
  }
  {  // 8u C2 3x1 -> 4.0
    twosrc_data<Npp8u>("nppiAverageError_8u_C2R_Ctx", "avgErr/8u/C2R_const_3x1", constant_image<Npp8u>(3, 1, 2, Npp8u{10}),
                       constant_image<Npp8u>(3, 1, 2, Npp8u{6}), 3, 1, 2, &nppiAverageError_8u_C2R_Ctx);
  }
  {  // 16s C3 1x1, -5 vs 5 -> 10.0
    twosrc_data<Npp16s>("nppiAverageError_16s_C3R_Ctx", "avgErr/16s/C3R_sign_1x1",
                        constant_image<Npp16s>(1, 1, 3, Npp16s{-5}), constant_image<Npp16s>(1, 1, 3, Npp16s{5}), 1, 1, 3,
                        &nppiAverageError_16s_C3R_Ctx);
  }
  {  // 16s C1 1x1 full-range: -32768 vs 32767 -> 65535.0
    twosrc_data<Npp16s>("nppiAverageError_16s_C1R_Ctx", "avgErr/16s/C1R_extreme_1x1",
                        constant_image<Npp16s>(1, 1, 1, Npp16s{-32768}),
                        constant_image<Npp16s>(1, 1, 1, Npp16s{32767}), 1, 1, 1, &nppiAverageError_16s_C1R_Ctx);
  }
  {  // 32f C1 2x1: |1.25-(-0.25)| + |(-2.5)-2.5| = 1.5 + 5 -> 3.25
    std::vector<Npp32f> h1 = constant_image<Npp32f>(2, 1, 1, Npp32f{0});
    std::vector<Npp32f> h2 = constant_image<Npp32f>(2, 1, 1, Npp32f{0});
    h1[0] = 1.25f; h1[1] = -2.5f;
    h2[0] = -0.25f; h2[1] = 2.5f;
    twosrc_data<Npp32f>("nppiAverageError_32f_C1R_Ctx", "avgErr/32f/C1R_sign_2x1", h1, h2, 2, 1, 1,
                        &nppiAverageError_32f_C1R_Ctx);
  }
  {  // 32fc C1 2x1: |(3,4)-(0,0)| + |(0,0)-(3,-4)| = 5 + 5 -> 5.0
    std::vector<Npp32fc> h1 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    std::vector<Npp32fc> h2 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    h1[0] = Npp32fc{3.0f, 4.0f};
    h2[1] = Npp32fc{3.0f, -4.0f};
    twosrc_data<Npp32fc>("nppiAverageError_32fc_C1R_Ctx", "avgErr/32fc/C1R_mag_2x1", h1, h2, 2, 1, 1,
                         &nppiAverageError_32fc_C1R_Ctx);
  }
  {  // 16sc C2 1x1: both channels |(3,4)|=5 -> (5+5)/2 = 5.0
    twosrc_data<Npp16sc>("nppiAverageError_16sc_C2R_Ctx", "avgErr/16sc/C2R_mag_1x1",
                         constant_image<Npp16sc>(1, 1, 2, Npp16sc{3, 4}), constant_image<Npp16sc>(1, 1, 2, Npp16sc{}), 1,
                         1, 2, &nppiAverageError_16sc_C2R_Ctx);
  }
  {  // 64f C1 17x13: +1.5 vs -1.5 -> 3.0
    twosrc_data<Npp64f>("nppiAverageError_64f_C1R_Ctx", "avgErr/64f/C1R_sign_17x13",
                        constant_image<Npp64f>(17, 13, 1, Npp64f{1.5}),
                        constant_image<Npp64f>(17, 13, 1, Npp64f{-1.5}), 17, 13, 1, &nppiAverageError_64f_C1R_Ctx);
  }
}

TEST_F(NppiAverageErrorDiff, AverageRelativeError_Crafted) {
  {  // 8u C4 2x1 const: 4/10 -> 0.4
    twosrc_data<Npp8u>("nppiAverageRelativeError_8u_C4R_Ctx", "avgRelErr/8u/C4R_const_2x1",
                       constant_image<Npp8u>(2, 1, 4, Npp8u{10}), constant_image<Npp8u>(2, 1, 4, Npp8u{6}), 2, 1, 4,
                       &nppiAverageRelativeError_8u_C4R_Ctx);
  }
  {  // 32f C2 3x1 const: 1.5/2.5 -> 0.6
    twosrc_data<Npp32f>("nppiAverageRelativeError_32f_C2R_Ctx", "avgRelErr/32f/C2R_const_3x1",
                        constant_image<Npp32f>(3, 1, 2, Npp32f{2.5}), constant_image<Npp32f>(3, 1, 2, Npp32f{1.0}), 3, 1,
                        2, &nppiAverageRelativeError_32f_C2R_Ctx);
  }
  {  // 8s C1 17x13 opposite signs: |100-(-100)|/100 -> 2.0 exactly (no clamp)
    twosrc_data<Npp8s>("nppiAverageRelativeError_8s_C1R_Ctx", "avgRelErr/8s/C1R_sign_17x13",
                       constant_image<Npp8s>(17, 13, 1, Npp8s{100}),
                       constant_image<Npp8s>(17, 13, 1, Npp8s{-100}), 17, 13, 1, &nppiAverageRelativeError_8s_C1R_Ctx);
  }
  {  // 16s C1 1x1 extreme: |(-32768)-32767|/32768 = 65535/32768 (double math)
    twosrc_data<Npp16s>("nppiAverageRelativeError_16s_C1R_Ctx", "avgRelErr/16s/C1R_extreme_1x1",
                        constant_image<Npp16s>(1, 1, 1, Npp16s{-32768}),
                        constant_image<Npp16s>(1, 1, 1, Npp16s{32767}), 1, 1, 1,
                        &nppiAverageRelativeError_16s_C1R_Ctx);
  }
  {  // 64f C1 17x13: 3.0/1.5 -> 2.0
    twosrc_data<Npp64f>("nppiAverageRelativeError_64f_C1R_Ctx", "avgRelErr/64f/C1R_sign_17x13",
                        constant_image<Npp64f>(17, 13, 1, Npp64f{1.5}),
                        constant_image<Npp64f>(17, 13, 1, Npp64f{-1.5}), 17, 13, 1,
                        &nppiAverageRelativeError_64f_C1R_Ctx);
  }
  {  // 32fc C1 1x1: |(3,4)-(-3,-4)| / |(-3,-4)| = 10/5 -> 2.0
    twosrc_data<Npp32fc>("nppiAverageRelativeError_32fc_C1R_Ctx", "avgRelErr/32fc/C1R_sign_1x1",
                         constant_image<Npp32fc>(1, 1, 1, Npp32fc{3.0f, 4.0f}),
                         constant_image<Npp32fc>(1, 1, 1, Npp32fc{-3.0f, -4.0f}), 1, 1, 1,
                         &nppiAverageRelativeError_32fc_C1R_Ctx);
  }
  {  // 16sc C3 1x1: same complex opposite -> 2.0
    twosrc_data<Npp16sc>("nppiAverageRelativeError_16sc_C3R_Ctx", "avgRelErr/16sc/C3R_sign_1x1",
                         constant_image<Npp16sc>(1, 1, 3, Npp16sc{3, 4}),
                         constant_image<Npp16sc>(1, 1, 3, Npp16sc{-3, -4}), 1, 1, 3,
                         &nppiAverageRelativeError_16sc_C3R_Ctx);
  }
}

// ===================== AverageRelativeError zero denominator ================
// The per-pixel denominator max(|a|,|b|) is zero only when a == b == 0.  The
// mixed probes decode NVIDIA's convention from a single golden each:
//   pixel 0 = (0,0)   zero denominator (0/0 term)
//   pixel 1 = (a,b)   known ratio (1.0 in every probe below)
// count-as-0     -> golden 1/2 = 0.5
// skip-and-renorm-> golden 1/1 = 1.0
// dtype max      -> golden (max + 1)/2
// NaN            -> golden NaN
// The all-zero probes additionally pin the every-pixel-zero case.
TEST_F(NppiAverageErrorDiff, AverageRelativeError_ZeroDenominator) {
  {  // 8u 2x1 mixed: [0 vs 0] and [5 vs 0]
    std::vector<Npp8u> h1 = constant_image<Npp8u>(2, 1, 1, Npp8u{0});
    std::vector<Npp8u> h2 = constant_image<Npp8u>(2, 1, 1, Npp8u{0});
    h1[0] = 0; h1[1] = 5;
    twosrc_data<Npp8u>("nppiAverageRelativeError_8u_C1R_Ctx", "avgRelErr/8u/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1, 1,
                       &nppiAverageRelativeError_8u_C1R_Ctx);
  }
  {  // 8u all-zero 17x13
    twosrc_data<Npp8u>("nppiAverageRelativeError_8u_C1R_Ctx", "avgRelErr/8u/C1R_allzero_17x13",
                       constant_image<Npp8u>(17, 13, 1, Npp8u{0}), constant_image<Npp8u>(17, 13, 1, Npp8u{0}), 17, 13, 1,
                       &nppiAverageRelativeError_8u_C1R_Ctx);
  }
  {  // 16s 2x1 mixed with a negative ratio pixel: [0 vs 0] and [-5 vs 0]
    std::vector<Npp16s> h1 = constant_image<Npp16s>(2, 1, 1, Npp16s{0});
    std::vector<Npp16s> h2 = constant_image<Npp16s>(2, 1, 1, Npp16s{0});
    h1[1] = -5;
    twosrc_data<Npp16s>("nppiAverageRelativeError_16s_C1R_Ctx", "avgRelErr/16s/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1, 1,
                        &nppiAverageRelativeError_16s_C1R_Ctx);
  }
  {  // 32f all-zero 17x13 (float NaN / zero / FLT_MAX)
    twosrc_data<Npp32f>("nppiAverageRelativeError_32f_C1R_Ctx", "avgRelErr/32f/C1R_allzero_17x13",
                        constant_image<Npp32f>(17, 13, 1, Npp32f{0}), constant_image<Npp32f>(17, 13, 1, Npp32f{0}), 17,
                        13, 1, &nppiAverageRelativeError_32f_C1R_Ctx);
  }
  {  // 64f 2x1 mixed: [0 vs 0] and [1.5 vs 0]
    std::vector<Npp64f> h1 = constant_image<Npp64f>(2, 1, 1, Npp64f{0});
    std::vector<Npp64f> h2 = constant_image<Npp64f>(2, 1, 1, Npp64f{0});
    h1[1] = 1.5;
    twosrc_data<Npp64f>("nppiAverageRelativeError_64f_C1R_Ctx", "avgRelErr/64f/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1, 1,
                        &nppiAverageRelativeError_64f_C1R_Ctx);
  }
  {  // 32fc 2x1 mixed: [(0,0) vs (0,0)] and [(3,4) vs (0,0)]
    std::vector<Npp32fc> h1 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    std::vector<Npp32fc> h2 = constant_image<Npp32fc>(2, 1, 1, Npp32fc{});
    h1[1] = Npp32fc{3.0f, 4.0f};
    twosrc_data<Npp32fc>("nppiAverageRelativeError_32fc_C1R_Ctx", "avgRelErr/32fc/C1R_zerodenom_mixed_2x1", h1, h2, 2, 1,
                         1, &nppiAverageRelativeError_32fc_C1R_Ctx);
  }
  {  // 16sc all-zero 17x13 (complex zero path)
    twosrc_data<Npp16sc>("nppiAverageRelativeError_16sc_C1R_Ctx", "avgRelErr/16sc/C1R_allzero_17x13",
                         constant_image<Npp16sc>(17, 13, 1, Npp16sc{}), constant_image<Npp16sc>(17, 13, 1, Npp16sc{}),
                         17, 13, 1, &nppiAverageRelativeError_16sc_C1R_Ctx);
  }
}

// ==================== buffer-size helpers (status only) =====================
TEST_F(NppiAverageErrorDiff, BufferSize_Helpers) {
  SIZE_TYPE n = 0;
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiAverageErrorGetBufferHostSize_8u_C4R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/avgErr_8u_C4R", "nppiAverageErrorGetBufferHostSize_8u_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageErrorGetBufferHostSize_16s_C1R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/avgErr_16s_C1R", "nppiAverageErrorGetBufferHostSize_16s_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageErrorGetBufferHostSize_32fc_C3R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/avgErr_32fc_C3R", "nppiAverageErrorGetBufferHostSize_32fc_C3R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageErrorGetBufferHostSize_64f_C2R_Ctx(NppiSize{1, 1}, &n, ctx);
    Case c("bufferSize/avgErr_64f_C2R", "nppiAverageErrorGetBufferHostSize_64f_C2R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/avgRelErr_8u_C1R", "nppiAverageRelativeErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_16sc_C4R_Ctx(NppiSize{64, 32}, &n, ctx);
    Case c("bufferSize/avgRelErr_16sc_C4R", "nppiAverageRelativeErrorGetBufferHostSize_16sc_C4R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_32f_C3R_Ctx(NppiSize{17, 13}, &n, ctx);
    Case c("bufferSize/avgRelErr_32f_C3R", "nppiAverageRelativeErrorGetBufferHostSize_32f_C3R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_32s_C2R_Ctx(NppiSize{1, 1}, &n, ctx);
    Case c("bufferSize/avgRelErr_32s_C2R", "nppiAverageRelativeErrorGetBufferHostSize_32s_C2R_Ctx");
    c.status(st); c.finish();
  }
}

// Buffer-size helper contract: null out pointer, zero ROI, negative ROI.
TEST_F(NppiAverageErrorDiff, BufferSize_Contract) {
  NppStreamContext ctx = current_context();
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{17, 13}, static_cast<SIZE_TYPE *>(nullptr), ctx);
    Case c("bufferSize/avgErr_contract/null_ptr", "nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{0, 2}, &n, ctx);
    Case c("bufferSize/avgErr_contract/zro_roi", "nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(NppiSize{-1, 2}, &n, ctx);
    Case c("bufferSize/avgErr_contract/neg_size", "nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_16u_C1R_Ctx(NppiSize{17, 13}, static_cast<SIZE_TYPE *>(nullptr), ctx);
    Case c("bufferSize/avgRelErr_contract/null_ptr", "nppiAverageRelativeErrorGetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_16u_C1R_Ctx(NppiSize{0, 2}, &n, ctx);
    Case c("bufferSize/avgRelErr_contract/zro_roi", "nppiAverageRelativeErrorGetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    SIZE_TYPE n = 0;
    NppStatus st = nppiAverageRelativeErrorGetBufferHostSize_16u_C1R_Ctx(NppiSize{-1, 2}, &n, ctx);
    Case c("bufferSize/avgRelErr_contract/neg_size", "nppiAverageRelativeErrorGetBufferHostSize_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
}

// ==================== buffer-size sufficiency ===============================
// Allocate exactly the reported scratch and run the matching reduction: the
// call must succeed and match the golden the 5090 recorded with NVIDIA's own
// buffer size.  Exact bytes are not a contract; sufficiency is.
TEST_F(NppiAverageErrorDiff, BufferSize_Sufficiency) {
  NppStreamContext ctx = current_context();
  const NppiSize roi{64, 32};
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiAverageErrorGetBufferHostSize_16u_C1R_Ctx(roi, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_scalar<Npp16u>("nppiAverageError_16u_C1R_Ctx", "avgErr/16u/C1R_exactbuf", 64, 32, 32, 1, 0x711,
                          &nppiAverageError_16u_C1R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiAverageErrorGetBufferHostSize_32fc_C3R_Ctx(roi, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_scalar<Npp32fc>("nppiAverageError_32fc_C3R_Ctx", "avgErr/32fc/C3R_exactbuf", 64, 32, 32, 3, 0x712,
                           &nppiAverageError_32fc_C3R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiAverageRelativeErrorGetBufferHostSize_8u_C1R_Ctx(roi, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_scalar<Npp8u>("nppiAverageRelativeError_8u_C1R_Ctx", "avgRelErr/8u/C1R_exactbuf", 64, 32, 32, 1, 0x721,
                         &nppiAverageRelativeError_8u_C1R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
  {
    SIZE_TYPE sz = 0;
    ASSERT_EQ(nppiAverageRelativeErrorGetBufferHostSize_32sc_C4R_Ctx(NppiSize{17, 13}, &sz, ctx), NPP_NO_ERROR);
    ASSERT_GT(sz, (SIZE_TYPE)0);
    twosrc_scalar<Npp32sc>("nppiAverageRelativeError_32sc_C4R_Ctx", "avgRelErr/32sc/C4R_exactbuf", 17, 13, 16, 4, 0x722,
                           &nppiAverageRelativeError_32sc_C4R_Ctx, 1e-5, 1e-5, arena<Npp8u>(sz).get());
  }
}

// ============================ non-Ctx adapters ==============================
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiAverageErrorDiff, NonCtx_AverageError) {
  // Reproduce the avgErr/8u/C1R_17x13 golden so both entry points agree.
  const int se = (int)sizeof(Npp8u);
  const int step = align_step(17 * se + 16, se);
  const size_t elems = (size_t)13 * (step / se);
  std::vector<Npp8u> h1 = make_image<Npp8u>(elems, 0x501);
  std::vector<Npp8u> h2 = make_image<Npp8u>(elems, 0x501 ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(elems);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(elems);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c("avgErr/8u/C1R_17x13", "nppiAverageError_8u_C1R");  // reuses the golden id above
  clear_cuda_error();
  NppStatus st = nppiAverageError_8u_C1R(s1.get(), step, s2.get(), step, NppiSize{17, 13}, o.get(), scratch());
  c.input(h1.data(), h1.size() * sizeof(Npp8u));
  c.input(h2.data(), h2.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
  c.finish();
}

TEST_F(NppiAverageErrorDiff, NonCtx_AverageRelativeError) {
  // Reproduce the avgRelErr/32f/C1R_17x13 golden so both entry points agree.
  const int se = (int)sizeof(Npp32f);
  const int step = align_step(17 * se + 16, se);
  const size_t elems = (size_t)13 * (step / se);
  std::vector<Npp32f> h1 = make_image<Npp32f>(elems, 0x681);
  std::vector<Npp32f> h2 = make_image<Npp32f>(elems, 0x681 ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<Npp32f> &s1 = arena<Npp32f>(elems);
  DeviceMemory<Npp32f> &s2 = arena<Npp32f>(elems);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  s1.copyFromHost(h1);
  s2.copyFromHost(h2);
  Case c("avgRelErr/32f/C1R_17x13", "nppiAverageRelativeError_32f_C1R");  // reuses the golden id above
  clear_cuda_error();
  NppStatus st =
      nppiAverageRelativeError_32f_C1R(s1.get(), step, s2.get(), step, NppiSize{17, 13}, o.get(), scratch());
  c.input(h1.data(), h1.size() * sizeof(Npp32f));
  c.input(h2.data(), h2.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output(o.get(), 1, ValueKind::F64);
  c.finish();
}
#endif

// ============================ contract ======================================
// Illegal-input probes follow the P3.1b1 convention: only scenarios that do
// not fault the NVIDIA device are recorded (null src, null out, zero /
// negative ROI, short step).  Status is captured via Case so verify compares
// the golden status name returned by the 5090.
TEST_F(NppiAverageErrorDiff, Contract_AverageError) {
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(16);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(16);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    NppStatus st = nppiAverageError_8u_C1R_Ctx(nullptr, 16, s2.get(), 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/null_src1", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageError_8u_C1R_Ctx(s1.get(), 16, nullptr, 16, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/null_src2", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("avgErr/contract/null_out", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/zro_roi", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/neg_size", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st = nppiAverageError_8u_C1R_Ctx(s1.get(), 2, s2.get(), 16, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/short_step", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // multi-channel step contract: step < width*4*4
    DeviceMemory<Npp32f> &t1 = arena<Npp32f>(16);
    DeviceMemory<Npp32f> &t2 = arena<Npp32f>(16);
    NppStatus st = nppiAverageError_32f_C4R_Ctx(t1.get(), 16, t2.get(), 32, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/short_step_c4", "nppiAverageError_32f_C4R_Ctx");
    c.status(st); c.finish();
  }
  {  // null scratch is validated (unlike Sum/Mean which no-op)
    NppStatus st = nppiAverageError_8u_C1R_Ctx(s1.get(), 16, s2.get(), 16, NppiSize{2, 2}, o.get(), nullptr, ctx);
    Case c("avgErr/contract/null_buf", "nppiAverageError_8u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // step not a multiple of sizeof(Npp32f)*4 (C4 alignment) -> NPP_NOT_EVEN_STEP_ERROR
    DeviceMemory<Npp32f> &t1 = arena<Npp32f>(48);
    DeviceMemory<Npp32f> &t2 = arena<Npp32f>(48);
    NppStatus st = nppiAverageError_32f_C4R_Ctx(t1.get(), 36, t2.get(), 36, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("avgErr/contract/noteven_step_c4", "nppiAverageError_32f_C4R_Ctx");
    c.status(st); c.finish();
  }
}

TEST_F(NppiAverageErrorDiff, Contract_AverageRelativeError) {
  DeviceMemory<Npp16u> &s1 = arena<Npp16u>(32);
  DeviceMemory<Npp16u> &s2 = arena<Npp16u>(32);
  DeviceMemory<Npp64f> &o = arena<Npp64f>(1);
  NppStreamContext ctx = current_context();
  {
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(nullptr, 8, s2.get(), 8, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("avgRelErr/contract/null_src1", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(s1.get(), 8, nullptr, 8, NppiSize{2, 2}, o.get(), scratch(), ctx);
    Case c("avgRelErr/contract/null_src2", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{2, 2}, nullptr, scratch(), ctx);
    Case c("avgRelErr/contract/null_out", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{0, 2}, o.get(), scratch(), ctx);
    Case c("avgRelErr/contract/zro_roi", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{-1, 2}, o.get(), scratch(), ctx);
    Case c("avgRelErr/contract/neg_size", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(s1.get(), 2, s2.get(), 8, NppiSize{4, 2}, o.get(), scratch(), ctx);
    Case c("avgRelErr/contract/short_step", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // null scratch is validated (unlike Sum/Mean which no-op)
    NppStatus st =
        nppiAverageRelativeError_16u_C1R_Ctx(s1.get(), 8, s2.get(), 8, NppiSize{2, 2}, o.get(), nullptr, ctx);
    Case c("avgRelErr/contract/null_buf", "nppiAverageRelativeError_16u_C1R_Ctx");
    c.status(st); c.finish();
  }
  {  // step not a multiple of sizeof(Npp16sc)*2 (C2 alignment) -> NPP_NOT_EVEN_STEP_ERROR
    DeviceMemory<Npp16sc> &t1 = arena<Npp16sc>(32);
    DeviceMemory<Npp16sc> &t2 = arena<Npp16sc>(32);
    NppStatus st = nppiAverageRelativeError_16sc_C2R_Ctx(t1.get(), 20, t2.get(), 20, NppiSize{2, 2}, o.get(),
                                                         scratch(), ctx);
    Case c("avgRelErr/contract/noteven_step_c2", "nppiAverageRelativeError_16sc_C2R_Ctx");
    c.status(st); c.finish();
  }
}
