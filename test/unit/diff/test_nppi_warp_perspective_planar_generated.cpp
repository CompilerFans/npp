// Differential tests for package g5: nppiWarpPerspective and
// nppiWarpPerspectiveBack on the planar layouts P3R/P4R (8u/16u/32f/32s, Ctx
// and non-Ctx).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiWarpPerspectivePlanarDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiWarpPerspectivePlanarDiff.*'
//
// Direction contract (NPP docs): nppiWarpPerspective's aCoeffs maps source
// coordinates to destination coordinates (x' = C x, applied with the
// denominator), while nppiWarpPerspectiveBack's aCoeffs maps destination
// coordinates to source coordinates.  The same coefficient matrix therefore
// produces visibly different outputs from the two families - every
// perspective case below is run through both.
//
// Sources are position-encoded (plane 0 = x, plane 1 = y, remaining planes =
// 7) so the sampling / rounding convention of each interpolation mode can be
// read directly out of the recorded bytes.

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

// Steps are rounded up to 256 bytes: the NVIDIA 32f C4R batch warp
// kernel uses 16-byte vectorized accesses and faults on element-aligned
// steps (observed on the RTX 5090, NPP 13.4).
int align_step(int step, int es) { (void)es; return (step + 255) & ~255; }

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else return ValueKind::Bytes;
}

template <typename T>
using PlanarFn = NppStatus (*)(const T **, NppiSize, int, NppiRect, T **, int, NppiRect, const double (*)[3], int,
                               NppStreamContext);

// Position-encoded plane p: plane 0 = x, plane 1 = y, others = 7.
template <typename T>
std::vector<T> make_planar_positional(size_t count, int plane, int row_elems) {
  std::vector<T> v(count);
  for (size_t i = 0; i < count; ++i) {
    if (plane == 0) {
      v[i] = (T)(i % (size_t)row_elems);
    } else if (plane == 1) {
      v[i] = (T)(i / (size_t)row_elems);
    } else {
      v[i] = (T)7;
    }
  }
  return v;
}

// Generic planar warp case.  Buffers are per-plane with independent device
// allocations; the golden contains every plane of the destination.
template <typename T, int Planes>
void warp_planar_case(const char *fname, const std::string &key, NppiSize srcSize, NppiRect srcROI, NppiSize dstSize,
                      NppiRect dstROI, const double coeffs[3][3], int interp, bool positional, PlanarFn<T> f,
                      uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcSize.width * es + 8, es);
  const int dStep = align_step(dstSize.width * es + 8, es);
  const size_t se = (size_t)srcSize.height * sStep / es;
  const size_t de = (size_t)dstSize.height * dStep / es;

  std::vector<const T *> srcPtrs(Planes);
  std::vector<T *> dstPtrs(Planes);
  std::vector<std::vector<T>> hsrc(Planes);
  std::vector<std::vector<T>> hdst(Planes, std::vector<T>(de, (T)0xA5));
  for (int p = 0; p < Planes; ++p) {
    hsrc[p] = positional ? make_planar_positional<T>(se, p, sStep / es) : make_uniform<T>(se, seed + 11 + p * 7);
    DeviceMemory<T> &s = arena<T>(se);
    DeviceMemory<T> &d = arena<T>(de);
    s.copyFromHost(hsrc[p]);
    d.copyFromHost(hdst[p]);
    srcPtrs[p] = s.get();
    dstPtrs[p] = d.get();
  }

  clear_cuda_error();
  NppStatus st = f(srcPtrs.data(), srcSize, sStep, srcROI, dstPtrs.data(), dStep, dstROI, coeffs, interp,
                   current_ctx());
  Case c("nppiWarpPerspective/" + key, fname);
  for (int p = 0; p < Planes; ++p) {
    c.input(hsrc[p].data(), hsrc[p].size() * sizeof(T));
  }
  c.status(st);
  for (int p = 0; p < Planes; ++p) {
    c.device_output<T>(dstPtrs[p], de, out_kind<T>(), 1e-5, 1e-6);
  }
  c.finish();
}

constexpr double kIdentity[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
constexpr double kScale2[3][3] = {{2, 0, 0}, {0, 2, 0}, {0, 0, 1}};
constexpr double kPersp[3][3] = {{1.5, 0.1, 2.0}, {-0.05, 0.9, 1.0}, {0.004, 0.002, 1.0}};
constexpr double kShift[3][3] = {{1, 0, 4}, {0, 1, -2}, {0, 0, 1}};
constexpr double kSingular[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
constexpr double kZero[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

}  // namespace

class NppiWarpPerspectivePlanarDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

#define PLANAR_CASES(TAG, DT, DTS, FWD, BACK)                                                          \
  TEST_F(NppiWarpPerspectivePlanarDiff, Warp_##TAG##_P3) {                                             \
    warp_planar_case<DT, 3>(#FWD "_P3R", "fwd/" DTS "_P3R/identity", NppiSize{20, 14},                 \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14},          \
                            kIdentity, NPPI_INTER_NN, true, &FWD##_P3R_Ctx, 600);                      \
    warp_planar_case<DT, 3>(#FWD "_P3R", "fwd/" DTS "_P3R/scale2", NppiSize{12, 9},                    \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},           \
                            kScale2, NPPI_INTER_NN, true, &FWD##_P3R_Ctx, 601);                        \
    warp_planar_case<DT, 3>(#FWD "_P3R", "fwd/" DTS "_P3R/persp_nn", NppiSize{20, 14},                 \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_NN, true, &FWD##_P3R_Ctx, 602);                         \
    warp_planar_case<DT, 3>(#FWD "_P3R", "fwd/" DTS "_P3R/persp_linear", NppiSize{20, 14},             \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_LINEAR, true, &FWD##_P3R_Ctx, 603);                     \
    warp_planar_case<DT, 3>(#FWD "_P3R", "fwd/" DTS "_P3R/persp_cubic", NppiSize{20, 14},              \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_CUBIC, true, &FWD##_P3R_Ctx, 604);                      \
    warp_planar_case<DT, 3>(#FWD "_P3R", "fwd/" DTS "_P3R/shift_subroi", NppiSize{24, 16},             \
                            NppiRect{4, 3, 16, 11}, NppiSize{24, 16}, NppiRect{6, 2, 16, 11},          \
                            kShift, NPPI_INTER_NN, false, &FWD##_P3R_Ctx, 605);                        \
  }                                                                                                    \
  TEST_F(NppiWarpPerspectivePlanarDiff, Warp_##TAG##_P4) {                                             \
    warp_planar_case<DT, 4>(#FWD "_P4R", "fwd/" DTS "_P4R/identity", NppiSize{20, 14},                 \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14},          \
                            kIdentity, NPPI_INTER_NN, true, &FWD##_P4R_Ctx, 610);                      \
    warp_planar_case<DT, 4>(#FWD "_P4R", "fwd/" DTS "_P4R/persp_nn", NppiSize{20, 14},                 \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_NN, true, &FWD##_P4R_Ctx, 611);                         \
    warp_planar_case<DT, 4>(#FWD "_P4R", "fwd/" DTS "_P4R/persp_linear", NppiSize{20, 14},             \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_LINEAR, true, &FWD##_P4R_Ctx, 612);                     \
  }                                                                                                    \
  TEST_F(NppiWarpPerspectivePlanarDiff, Back_##TAG##_P3) {                                             \
    warp_planar_case<DT, 3>(#BACK "_P3R", "back/" DTS "_P3R/identity", NppiSize{20, 14},               \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14},          \
                            kIdentity, NPPI_INTER_NN, true, &BACK##_P3R_Ctx, 620);                     \
    warp_planar_case<DT, 3>(#BACK "_P3R", "back/" DTS "_P3R/scale2", NppiSize{12, 9},                  \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},           \
                            kScale2, NPPI_INTER_NN, true, &BACK##_P3R_Ctx, 621);                       \
    warp_planar_case<DT, 3>(#BACK "_P3R", "back/" DTS "_P3R/persp_nn", NppiSize{20, 14},               \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_NN, true, &BACK##_P3R_Ctx, 622);                        \
    warp_planar_case<DT, 3>(#BACK "_P3R", "back/" DTS "_P3R/persp_linear", NppiSize{20, 14},           \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_LINEAR, true, &BACK##_P3R_Ctx, 623);                    \
  }                                                                                                    \
  TEST_F(NppiWarpPerspectivePlanarDiff, Back_##TAG##_P4) {                                             \
    warp_planar_case<DT, 4>(#BACK "_P4R", "back/" DTS "_P4R/identity", NppiSize{20, 14},               \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14},          \
                            kIdentity, NPPI_INTER_NN, true, &BACK##_P4R_Ctx, 630);                     \
    warp_planar_case<DT, 4>(#BACK "_P4R", "back/" DTS "_P4R/persp_nn", NppiSize{20, 14},               \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_NN, true, &BACK##_P4R_Ctx, 631);                        \
    warp_planar_case<DT, 4>(#BACK "_P4R", "back/" DTS "_P4R/persp_linear", NppiSize{20, 14},           \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{2, 3, 24, 18},          \
                            kPersp, NPPI_INTER_LINEAR, true, &BACK##_P4R_Ctx, 632);                    \
  }

PLANAR_CASES(8u, Npp8u, "8u", nppiWarpPerspective_8u, nppiWarpPerspectiveBack_8u)
PLANAR_CASES(16u, Npp16u, "16u", nppiWarpPerspective_16u, nppiWarpPerspectiveBack_16u)
PLANAR_CASES(32f, Npp32f, "32f", nppiWarpPerspective_32f, nppiWarpPerspectiveBack_32f)
PLANAR_CASES(32s, Npp32s, "32s", nppiWarpPerspective_32s, nppiWarpPerspectiveBack_32s)

// ===== contract: null planes, steps, sizes, ROIs, interpolation, coeffs ====
TEST_F(NppiWarpPerspectivePlanarDiff, Warp_Planar_Contract) {
  const NppiSize srcSize{20, 14};
  const NppiRect srcROI{0, 0, 20, 14};
  const NppiSize dstSize{32, 24};
  const NppiRect dstROI{2, 3, 24, 18};
  const int es = 1;
  const int sStep = align_step(srcSize.width * es + 8, es);
  const int dStep = align_step(dstSize.width * es + 8, es);
  const size_t se = (size_t)srcSize.height * sStep;
  const size_t de = (size_t)dstSize.height * dStep;
  // Oversized allocation: the src_roi_oob probe reads up to (33, 13) if the
  // implementation does not validate the ROI, which must not crash.
  const size_t bigSrc = 200 * 40;

  DeviceMemory<Npp8u> &s0 = arena<Npp8u>(bigSrc);
  DeviceMemory<Npp8u> &s1 = arena<Npp8u>(bigSrc);
  DeviceMemory<Npp8u> &s2 = arena<Npp8u>(bigSrc);
  DeviceMemory<Npp8u> &d0 = arena<Npp8u>(de);
  DeviceMemory<Npp8u> &d1 = arena<Npp8u>(de);
  DeviceMemory<Npp8u> &d2 = arena<Npp8u>(de);
  for (auto *m : {&s0, &s1, &s2}) m->copyFromHost(std::vector<Npp8u>(bigSrc, 3));
  for (auto *m : {&d0, &d1, &d2}) m->copyFromHost(std::vector<Npp8u>(de, 0xA5));
  const Npp8u *srcOk[3] = {s0.get(), s1.get(), s2.get()};
  Npp8u *dstOk[3] = {d0.get(), d1.get(), d2.get()};

  auto run = [&](const std::string &key, const Npp8u **psrc, NppiSize ssize, int sstep, NppiRect sroi, Npp8u **pdst,
                 int dstep, NppiRect droi, const double c[3][3], int interp) {
    clear_cuda_error();
    NppStatus st = nppiWarpPerspective_8u_P3R_Ctx(psrc, ssize, sstep, sroi, pdst, dstep, droi, c, interp,
                                                  current_ctx());
    Case cc("nppiWarpPerspective/contract/" + key, "nppiWarpPerspective_8u_P3R");
    cc.status(st);
    cc.finish();
  };
  const Npp8u *nullPlane[3] = {s0.get(), nullptr, s2.get()};
  Npp8u *nullDstPlane[3] = {d0.get(), nullptr, d2.get()};
  run("null_src_array", nullptr, srcSize, sStep, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("null_dst_array", srcOk, srcSize, sStep, srcROI, nullptr, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("null_src_plane", nullPlane, srcSize, sStep, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("null_dst_plane", srcOk, srcSize, sStep, srcROI, nullDstPlane, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("src_step0", srcOk, srcSize, 0, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("dst_step0", srcOk, srcSize, sStep, srcROI, dstOk, 0, dstROI, kIdentity, NPPI_INTER_NN);
  run("src_size_empty", srcOk, NppiSize{0, 14}, sStep, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("src_size_neg", srcOk, NppiSize{20, -14}, sStep, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("src_roi_w0", srcOk, srcSize, sStep, NppiRect{0, 0, 0, 14}, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_NN);
  run("src_roi_neg", srcOk, srcSize, sStep, NppiRect{0, 0, -20, 14}, dstOk, dStep, dstROI, kIdentity,
      NPPI_INTER_NN);
  run("dst_roi_w0", srcOk, srcSize, sStep, srcROI, dstOk, dStep, NppiRect{2, 3, 0, 18}, kIdentity, NPPI_INTER_NN);
  run("dst_roi_h0", srcOk, srcSize, sStep, srcROI, dstOk, dStep, NppiRect{2, 3, 24, 0}, kIdentity, NPPI_INTER_NN);
  run("src_roi_oob", srcOk, srcSize, sStep, NppiRect{2, 2, 32, 12}, dstOk, dStep, dstROI, kIdentity,
      NPPI_INTER_NN);
  run("src_roi_neg_xy", srcOk, srcSize, sStep, NppiRect{-2, 0, 20, 14}, dstOk, dStep, dstROI, kIdentity,
      NPPI_INTER_NN);
  run("interp_undefined", srcOk, srcSize, sStep, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_UNDEFINED);
  run("interp_super", srcOk, srcSize, sStep, srcROI, dstOk, dStep, dstROI, kIdentity, NPPI_INTER_SUPER);
  run("coeffs_singular", srcOk, srcSize, sStep, srcROI, dstOk, dStep, dstROI, kSingular, NPPI_INTER_NN);
  run("coeffs_zero", srcOk, srcSize, sStep, srcROI, dstOk, dStep, dstROI, kZero, NPPI_INTER_NN);

  auto runBack = [&](const std::string &key, const double c[3][3], int interp) {
    clear_cuda_error();
    NppStatus st = nppiWarpPerspectiveBack_8u_P3R_Ctx(srcOk, srcSize, sStep, srcROI, dstOk, dStep, dstROI, c, interp,
                                                      current_ctx());
    Case cc("nppiWarpPerspective/contract/" + key, "nppiWarpPerspectiveBack_8u_P3R");
    cc.status(st);
    cc.finish();
  };
  runBack("back_coeffs_singular", kSingular, NPPI_INTER_NN);
  runBack("back_interp_undefined", kIdentity, NPPI_INTER_UNDEFINED);
}

// ===== Non-Ctx adapters (representative _Ctx cases -> same golden) =========
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiWarpPerspectivePlanarDiff, NonCtx_WarpPlanar) {
  // Reproduces warp_planar_case<8u,3>(fwd identity, positional, seed=600).
  const int es = 1;
  const int step = align_step(20 * es + 8, es);
  const size_t ne = (size_t)14 * step / es;
  std::vector<const Npp8u *> srcPtrs(3);
  std::vector<Npp8u *> dstPtrs(3);
  std::vector<std::vector<Npp8u>> hsrc(3);
  for (int p = 0; p < 3; ++p) {
    hsrc[p] = make_planar_positional<Npp8u>(ne, p, step / es);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(ne);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(ne);
    s.copyFromHost(hsrc[p]);
    d.copyFromHost(std::vector<Npp8u>(ne, 0xA5));
    srcPtrs[p] = s.get();
    dstPtrs[p] = d.get();
  }
  clear_cuda_error();
  NppStatus st = nppiWarpPerspective_8u_P3R(srcPtrs.data(), NppiSize{20, 14}, step, NppiRect{0, 0, 20, 14},
                                            dstPtrs.data(), step, NppiRect{0, 0, 20, 14}, kIdentity, NPPI_INTER_NN);
  Case c("nppiWarpPerspective/fwd/8u_P3R/identity", "nppiWarpPerspective_8u_P3R");
  for (int p = 0; p < 3; ++p) {
    c.input(hsrc[p].data(), hsrc[p].size() * sizeof(Npp8u));
  }
  c.status(st);
  for (int p = 0; p < 3; ++p) {
    c.device_output<Npp8u>(dstPtrs[p], ne, ValueKind::U8, 1e-5, 1e-6);
  }
  c.finish();

  // Reproduces warp_planar_case<32f,4>(back persp_nn, positional, seed=631).
  const int es32 = 4;
  const int sStep32 = align_step(20 * es32 + 8, es32);
  const int dStep32 = align_step(32 * es32 + 8, es32);
  const size_t se32 = (size_t)14 * sStep32 / es32;
  const size_t de32 = (size_t)24 * dStep32 / es32;
  std::vector<const Npp32f *> srcPtrs32(4);
  std::vector<Npp32f *> dstPtrs32(4);
  std::vector<std::vector<Npp32f>> hsrc32(4);
  for (int p = 0; p < 4; ++p) {
    hsrc32[p] = make_planar_positional<Npp32f>(se32, p, sStep32 / es32);
    DeviceMemory<Npp32f> &s = arena<Npp32f>(se32);
    DeviceMemory<Npp32f> &d = arena<Npp32f>(de32);
    s.copyFromHost(hsrc32[p]);
    d.copyFromHost(std::vector<Npp32f>(de32, (Npp32f)0xA5));
    srcPtrs32[p] = s.get();
    dstPtrs32[p] = d.get();
  }
  clear_cuda_error();
  NppStatus st32 = nppiWarpPerspectiveBack_32f_P4R(srcPtrs32.data(), NppiSize{20, 14}, sStep32,
                                                   NppiRect{0, 0, 20, 14}, dstPtrs32.data(), dStep32,
                                                   NppiRect{2, 3, 24, 18}, kPersp, NPPI_INTER_NN);
  Case c32("nppiWarpPerspective/back/32f_P4R/persp_nn", "nppiWarpPerspectiveBack_32f_P4R");
  for (int p = 0; p < 4; ++p) {
    c32.input(hsrc32[p].data(), hsrc32[p].size() * sizeof(Npp32f));
  }
  c32.status(st32);
  for (int p = 0; p < 4; ++p) {
    c32.device_output<Npp32f>(dstPtrs32[p], de32, ValueKind::F32, 1e-5, 1e-6);
  }
  c32.finish();
}
#endif
