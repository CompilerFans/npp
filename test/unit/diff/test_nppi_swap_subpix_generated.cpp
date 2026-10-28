// Differential tests for package p2.2a: nppiSwapChannels (C3/C4 channel remap,
// incl. in-place and C3C4 fill) and nppiCopySubpix (sub-pixel bilinear copy).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiSwapSubpixDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiSwapSubpixDiff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else return ValueKind::Bytes;
}

// Out-of-place channel permutation (C3R / C4R / C4C3R / AC4R share this shape).
template <typename T>
void swap_perm(const char *fname, const std::string &key, int width, int height, int srcStr, int dstStr, int pad,
               const int order[], int nOrder,
               NppStatus (*f)(const T *, int, T *, int, NppiSize, const int *, NppStreamContext), uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * srcStr * es + pad, es);
  const int dStep = align_step(width * dstStr * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  int ord[4];
  for (int i = 0; i < 4; ++i) ord[i] = i < nOrder ? order[i] : 0;
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, ord, current_ctx());
  Case c("nppiSwapChannels/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// In-place channel permutation.
template <typename T>
void swap_inplace(const char *fname, const std::string &key, int width, int height, int stride, int pad,
                  const int *order, int nOrder, NppStatus (*f)(T *, int, NppiSize, const int *, NppStreamContext),
                  uint64_t seed) {
  const int es = (int)sizeof(T);
  const int step = align_step(width * stride * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<T> hs = make_uniform<T>(ne, seed + 11);
  DeviceMemory<T> &s = arena<T>(ne);
  s.copyFromHost(hs);
  int ord[4];
  for (int i = 0; i < 4; ++i) ord[i] = i < nOrder ? order[i] : 0;
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, ord, current_ctx());
  Case c("nppiSwapChannels/" + key, fname);
  c.status(st);
  c.device_output<T>(s.get(), ne, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// 3ch -> 4ch expand with a per-channel fill value.
template <typename T>
void swap_c3c4(const char *fname, const std::string &key, int width, int height, int pad, const int order[4],
               T nValue, uint64_t seed, NppStatus (*f)(const T *, int, T *, int, NppiSize, const int *,
                                                       const T, NppStreamContext)) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * 3 * es + pad, es);
  const int dStep = align_step(width * 4 * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, order, nValue, current_ctx());
  Case c("nppiSwapChannels/C3C4/" + key, fname);
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// CopySubpix with sub-pixel offsets.
template <typename T>
void subpix_run(const char *fname, const std::string &key, int width, int height, int stride, int pad, float dx,
                float dy, NppStatus (*f)(const T *, int, T *, int, NppiSize, Npp32f, Npp32f, NppStreamContext),
                uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * stride * es + pad, es);
  const int dStep = align_step(width * stride * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus status = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, dx, dy, current_ctx());
  std::string id = "nppiCopySubpix/" + key;
  Case c(id, fname);
  c.status(status);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

}  // namespace

class NppiSwapSubpixDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== SwapChannels: out-of-place permutations =============================
TEST_F(NppiSwapSubpixDiff, Swap_C3R) {
  const int o1[3] = {2, 1, 0};
  swap_perm("nppiSwapChannels_8u_C3R", "8u/rev", 17, 13, 3, 3, 16, o1, 3, &nppiSwapChannels_8u_C3R_Ctx, 100);
  const int o2[3] = {1, 2, 0};
  swap_perm("nppiSwapChannels_16u_C3R", "16u/rot", 5, 7, 3, 3, 16, o2, 3, &nppiSwapChannels_16u_C3R_Ctx, 101);
  const int o3[3] = {0, 2, 1};
  swap_perm("nppiSwapChannels_32f_C3R", "32f/rev", 1, 1, 3, 3, 12, o3, 3, &nppiSwapChannels_32f_C3R_Ctx, 102);
}
TEST_F(NppiSwapSubpixDiff, Swap_C4R) {
  const int o1[4] = {3, 2, 1, 0};
  swap_perm("nppiSwapChannels_32s_C4R", "32s/bgra", 17, 13, 4, 4, 16, o1, 4, &nppiSwapChannels_32s_C4R_Ctx, 110);
  const int o2[4] = {1, 3, 0, 2};
  swap_perm("nppiSwapChannels_16s_C4R", "16s/rot", 64, 32, 4, 4, 32, o2, 4, &nppiSwapChannels_16s_C4R_Ctx, 111);
}
TEST_F(NppiSwapSubpixDiff, Swap_C4C3R) {
  const int o1[3] = {0, 1, 2};
  swap_perm("nppiSwapChannels_8u_C4C3R", "8u/dropA", 17, 13, 4, 3, 16, o1, 3, &nppiSwapChannels_8u_C4C3R_Ctx, 120);
  const int o2[3] = {2, 1, 0};
  swap_perm("nppiSwapChannels_32f_C4C3R", "32f/BGR", 1, 1, 4, 3, 12, o2, 3, &nppiSwapChannels_32f_C4C3R_Ctx, 121);
}
TEST_F(NppiSwapSubpixDiff, Swap_C3C4R) {
  int o1[4] = {0, 1, 2, 3};
  swap_c3c4<Npp8u>("nppiSwapChannels_8u_C3C4R", "8u", 17, 13, 16, o1, (Npp8u)0xFF, 130, &nppiSwapChannels_8u_C3C4R_Ctx);
  int o2[4] = {2, 3, 0, 1};
  swap_c3c4<Npp16u>("nppiSwapChannels_16u_C3C4R", "16u", 17, 13, 16, o2, (Npp16u)1234, 131, &nppiSwapChannels_16u_C3C4R_Ctx);
  int o3[4] = {3, 0, 2, 7};  // entry 7 (>3) leaves destination channel untouched
  swap_c3c4<Npp32f>("nppiSwapChannels_32f_C3C4R", "32f", 64, 32, 32, o3, 0.5f, 132, &nppiSwapChannels_32f_C3C4R_Ctx);
}
TEST_F(NppiSwapSubpixDiff, Swap_AC4R) {
  const int o1[3] = {2, 1, 0};
  swap_perm("nppiSwapChannels_32s_AC4R", "32s/BGRA", 17, 13, 4, 4, 16, o1, 3, &nppiSwapChannels_32s_AC4R_Ctx, 140);
  const int o2[3] = {1, 0, 2};
  swap_perm("nppiSwapChannels_8u_AC4R", "8u/rot", 17, 13, 4, 4, 16, o2, 3, &nppiSwapChannels_8u_AC4R_Ctx, 141);
}
TEST_F(NppiSwapSubpixDiff, Swap_Inplace) {
  const int o1[3] = {2, 0, 1};
  swap_inplace<Npp8u>("nppiSwapChannels_8u_C3IR", "8u_C3IR", 17, 13, 3, 16, o1, 3, &nppiSwapChannels_8u_C3IR_Ctx, 150);
  const int o2[4] = {3, 2, 1, 0};
  swap_inplace<Npp16u>("nppiSwapChannels_16u_C4IR", "16u_C4IR", 17, 13, 4, 16, o2, 4, &nppiSwapChannels_16u_C4IR_Ctx, 151);
  const int o3[3] = {1, 2, 0};
  swap_inplace<Npp32f>("nppiSwapChannels_32f_C3IR", "32f_C3IR", 1, 1, 3, 12, o3, 3, &nppiSwapChannels_32f_C3IR_Ctx, 152);
}
TEST_F(NppiSwapSubpixDiff, Swap_Contract) {
  {
    DeviceMemory<Npp8u> &s = arena<Npp8u>(64);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(64);
    s.copyFromHost(std::vector<Npp8u>(64, 1));
    d.copyFromHost(std::vector<Npp8u>(64, 2));
    int ord3[3] = {1, 2, 0};
    clear_cuda_error();
    NppStatus st = nppiSwapChannels_8u_C3R_Ctx(s.get(), 24, d.get(), 24, NppiSize{0, 5}, ord3, current_ctx());
    Case c("nppiSwapChannels/8u_C3R/empty", "nppiSwapChannels_8u_C3R");
    c.status(st);
    c.finish();
    ord3[0] = 7;  // illegal channel-order value
    clear_cuda_error();
    st = nppiSwapChannels_8u_C3R_Ctx(s.get(), 24, d.get(), 24, NppiSize{4, 1}, ord3, current_ctx());
    Case c2("nppiSwapChannels/8u_C3R/badorder", "nppiSwapChannels_8u_C3R");
    c2.status(st);
    c2.finish();
  }
}

// ===== CopySubpix ==========================================================
TEST_F(NppiSwapSubpixDiff, Subpix_Offsets_Zero) {
  subpix_run<Npp8u>("nppiCopySubpix_8u_C1R", "8u_C1_dx0", 17, 13, 1, 16, 0.f, 0.f, &nppiCopySubpix_8u_C1R_Ctx, 200);
  subpix_run<Npp32f>("nppiCopySubpix_32f_C3R", "32f_C3_dx0", 17, 13, 3, 16, 0.f, 0.f, &nppiCopySubpix_32f_C3R_Ctx, 201);
}
TEST_F(NppiSwapSubpixDiff, Subpix_Offsets_Fraction) {
  subpix_run<Npp8u>("nppiCopySubpix_8u_C1R", "8u_C1_dx025", 17, 13, 1, 16, 0.25f, 0.25f, &nppiCopySubpix_8u_C1R_Ctx, 210);
  subpix_run<Npp16u>("nppiCopySubpix_16u_C3R", "16u_C3_dy05", 17, 13, 3, 16, 0.0f, 0.5f, &nppiCopySubpix_16u_C3R_Ctx, 211);
  subpix_run<Npp16s>("nppiCopySubpix_16s_C4R", "16s_C4_dx-025", 17, 13, 4, 16, -0.25f, 0.0f, &nppiCopySubpix_16s_C4R_Ctx, 212);
}
TEST_F(NppiSwapSubpixDiff, Subpix_Offsets_Boundary) {
  subpix_run<Npp8u>("nppiCopySubpix_8u_C4R", "8u_C4_dx050", 1, 1, 4, 16, 0.5f, 0.0f, &nppiCopySubpix_8u_C4R_Ctx, 220);
  subpix_run<Npp32f>("nppiCopySubpix_32f_C1R", "32f_C1_dx-05", 5, 7, 1, 8, -0.5f, 0.5f, &nppiCopySubpix_32f_C1R_Ctx, 221);
  subpix_run<Npp32s>("nppiCopySubpix_32s_C3R", "32s_C3_dy075", 7, 9, 3, 8, 0.75f, 0.75f, &nppiCopySubpix_32s_C3R_Ctx, 222);
}
TEST_F(NppiSwapSubpixDiff, Subpix_AC4) {
  subpix_run<Npp8u>("nppiCopySubpix_8u_AC4R", "8u_AC4_dx025", 17, 13, 4, 16, 0.25f, 0.0f, &nppiCopySubpix_8u_AC4R_Ctx, 230);
  subpix_run<Npp32f>("nppiCopySubpix_32f_AC4R", "32f_AC4_dy05", 17, 13, 4, 16, 0.0f, 0.5f, &nppiCopySubpix_32f_AC4R_Ctx, 231);
}

// ===== Non-Ctx adapters (reproduce representative _Ctx cases -> same golden) =
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiSwapSubpixDiff, NonCtx_Swap) {
  // Reproduces swap_perm(8u_C3R "8u/rev", 17,13,3,3,16, {2,1,0}, seed=100).
  const int es = 1, srcStr = 3, dstStr = 3, pad = 16, width = 17, height = 13;
  const int sStep = ((width * srcStr * es + pad + es - 1) / es) * es;
  const int dStep = sStep;
  const size_t se = (size_t)height * sStep / es;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(se, 100 + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(se, 100 + 29);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(se);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  int ord[3] = {2, 1, 0};
  clear_cuda_error();
  NppStatus st = nppiSwapChannels_8u_C3R(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, ord);
  Case c("nppiSwapChannels/8u/rev", "nppiSwapChannels_8u_C3R");
  c.status(st);
  c.device_output<Npp8u>(d.get(), se, ValueKind::U8, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppiSwapSubpixDiff, NonCtx_Subpix) {
  // Reproduces subpix_run(32f_C3R "32f_C3_dx0", 17,13,3,16, dx0 dy0, seed=201).
  const int es = (int)sizeof(Npp32f), stride = 3, pad = 16, width = 17, height = 13;
  const int sStep = ((width * stride * es + pad + es - 1) / es) * es;
  const size_t se = (size_t)height * sStep / es;
  std::vector<Npp32f> hs = make_uniform<Npp32f>(se, 201 + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(se, 201 + 29);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(se);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiCopySubpix_32f_C3R(s.get(), sStep, d.get(), sStep, NppiSize{width, height}, 0.f, 0.f);
  Case c("nppiCopySubpix/32f_C3_dx0", "nppiCopySubpix_32f_C3R");
  c.status(st);
  c.device_output<Npp32f>(d.get(), se, ValueKind::F32, 1e-5, 1e-6);
  c.finish();
}
#endif
