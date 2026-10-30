// Differential tests for package g5: nppiRotate 8u/16u/32f AC4R (Ctx and
// non-Ctx).  The C1R/C3R/C4R variants are already implemented; a handful of
// probe cases on those functions pin down the border / rounding / alpha
// semantics that the new AC4 kernels must match.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiRotateDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiRotateDiff.*'
//
// The destination is always pre-filled with the sentinel 0xA5 so that the
// golden shows, per destination pixel:
//   * pixels outside the destination ROI are never written;
//   * pixels whose rotated source position falls outside the source ROI are
//     either zeroed or left untouched;
//   * for AC4 the alpha byte is either moved with the pixel, left untouched
//     or zeroed.
// Interpolation probes use a position-encoded source pattern (channel 0 = x,
// channel 1 = y) so the sampling / rounding convention can be read directly
// out of the recorded bytes.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstring>
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

template <typename T>
using RotFn = NppStatus (*)(const T *, NppiSize, int, NppiRect, T *, int, NppiRect, double, double, double, int,
                            NppStreamContext);

// Position-encoded source: channel 0 = x, channel 1 = y, channel 2 = 7,
// channel 3 = alpha sentinel (255 / 65535 / -1.5).
template <typename T> std::vector<T> make_positional(size_t count, int stride, int row_elems) {
  std::vector<T> v(count);
  for (size_t i = 0; i < count; ++i) {
    const int col = (int)((i % (size_t)row_elems) / (size_t)stride);
    const int row = (int)(i / (size_t)row_elems);
    switch (i % (size_t)stride) {
    case 0: v[i] = (T)(col); break;
    case 1: v[i] = (T)(row); break;
    case 2: v[i] = (T)7; break;
    default: v[i] = (T)255; break;
    }
  }
  return v;
}

// Generic out-of-place rotate case.  The source is either position-encoded
// (positional=true) or RNG-filled (positional=false); the destination is
// always pre-filled with 0xA5 so untouched bytes stay visible.
template <typename T>
void rotate_case(const char *fname, const std::string &key, NppiSize srcSize, NppiRect srcROI, NppiSize dstSize,
                 NppiRect dstROI, double angle, double shiftX, double shiftY, int interp, bool positional,
                 RotFn<T> f, uint64_t seed) {
  const int es = (int)sizeof(T);
  const int stride = fname[strlen(fname) - 2] == '4' ? 4 : (fname[strlen(fname) - 2] == '3' ? 3 : 1);
  const int sStep = align_step(srcSize.width * stride * es + 8, es);
  const int dStep = align_step(dstSize.width * stride * es + 8, es);
  const size_t se = (size_t)srcSize.height * sStep / es;
  const size_t de = (size_t)dstSize.height * dStep / es;
  std::vector<T> hs = positional ? make_positional<T>(se, stride, sStep / es) : make_uniform<T>(se, seed + 11);
  std::vector<T> hd(de, (T)0xA5);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, angle, shiftX, shiftY, interp,
                   current_ctx());
  Case c("nppiRotate/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

}  // namespace

class NppiRotateDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== 8u AC4R ==============================================================
TEST_F(NppiRotateDiff, Rotate_8u_AC4) {
  const RotFn<Npp8u> f = &nppiRotate_8u_AC4R_Ctx;
  // Identity: angle 0, no shift, matching ROIs.
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/angle0", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{24, 16}, NppiRect{0, 0, 24, 16}, 0.0, 0.0, 0.0, NPPI_INTER_NN, true, f, 500);
  // Rotation with parts of the destination outside the source: pins the
  // border rule together with the alpha rule.
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/angle30_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN, true, f, 501);
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/angle90_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{16, 24}, NppiRect{0, 0, 16, 24}, 90.0, 16.0, 0.0, NPPI_INTER_NN, true, f, 502);
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/angle_neg25_5_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{6, 4, 28, 24}, -25.5, 10.0, 8.0, NPPI_INTER_NN, true, f, 503);
  // Interpolation: position-encoded source so the sampling convention is
  // readable from the golden.
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/angle30_linear", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_LINEAR, true, f, 504);
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/angle33_25_cubic", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 33.25, 0.0, 0.0, NPPI_INTER_CUBIC, true, f, 505);
  // Sub-rect source ROI and offset destination ROI.
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/subroi_nn", NppiSize{32, 24}, NppiRect{4, 3, 16, 12},
                     NppiSize{24, 16}, NppiRect{5, 2, 16, 12}, 20.0, 8.0, 6.0, NPPI_INTER_NN, false, f, 506);
  // RNG-filled source, larger image.
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/rng_angle30_nn", NppiSize{37, 23}, NppiRect{0, 0, 37, 23},
                     NppiSize{64, 48}, NppiRect{8, 10, 44, 30}, 30.0, 18.0, 11.0, NPPI_INTER_NN, false, f, 507);
  // 1x1 destination ROI inside a larger destination.
  rotate_case<Npp8u>("nppiRotate_8u_AC4R", "8u_AC4R/one_pixel", NppiSize{8, 8}, NppiRect{0, 0, 8, 8}, NppiSize{16, 16},
                     NppiRect{3, 5, 1, 1}, 45.0, 4.0, 4.0, NPPI_INTER_NN, true, f, 508);
}

// ===== 16u AC4R =============================================================
TEST_F(NppiRotateDiff, Rotate_16u_AC4) {
  const RotFn<Npp16u> f = &nppiRotate_16u_AC4R_Ctx;
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/angle0", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{24, 16}, NppiRect{0, 0, 24, 16}, 0.0, 0.0, 0.0, NPPI_INTER_NN, true, f, 510);
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/angle30_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN, true, f, 511);
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/angle_neg25_5_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{6, 4, 28, 24}, -25.5, 10.0, 8.0, NPPI_INTER_NN, true, f, 512);
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/angle30_linear", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_LINEAR, true, f, 513);
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/angle33_25_cubic", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 33.25, 0.0, 0.0, NPPI_INTER_CUBIC, true, f, 514);
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/subroi_nn", NppiSize{32, 24}, NppiRect{4, 3, 16, 12},
                      NppiSize{24, 16}, NppiRect{5, 2, 16, 12}, 20.0, 8.0, 6.0, NPPI_INTER_NN, false, f, 515);
  rotate_case<Npp16u>("nppiRotate_16u_AC4R", "16u_AC4R/rng_angle30_nn", NppiSize{37, 23}, NppiRect{0, 0, 37, 23},
                      NppiSize{64, 48}, NppiRect{8, 10, 44, 30}, 30.0, 18.0, 11.0, NPPI_INTER_NN, false, f, 516);
}

// ===== 32f AC4R =============================================================
TEST_F(NppiRotateDiff, Rotate_32f_AC4) {
  const RotFn<Npp32f> f = &nppiRotate_32f_AC4R_Ctx;
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/angle0", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{24, 16}, NppiRect{0, 0, 24, 16}, 0.0, 0.0, 0.0, NPPI_INTER_NN, true, f, 520);
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/angle30_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN, true, f, 521);
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/angle_neg25_5_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{6, 4, 28, 24}, -25.5, 10.0, 8.0, NPPI_INTER_NN, true, f, 522);
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/angle30_linear", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_LINEAR, true, f, 523);
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/angle33_25_cubic", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 33.25, 0.0, 0.0, NPPI_INTER_CUBIC, true, f, 524);
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/subroi_nn", NppiSize{32, 24}, NppiRect{4, 3, 16, 12},
                      NppiSize{24, 16}, NppiRect{5, 2, 16, 12}, 20.0, 8.0, 6.0, NPPI_INTER_NN, false, f, 525);
  rotate_case<Npp32f>("nppiRotate_32f_AC4R", "32f_AC4R/rng_angle30_nn", NppiSize{37, 23}, NppiRect{0, 0, 37, 23},
                      NppiSize{64, 48}, NppiRect{8, 10, 44, 30}, 30.0, 18.0, 11.0, NPPI_INTER_NN, false, f, 526);
}

// ===== probes on the already-implemented variants ==========================
// These pin the semantics shared with the AC4 kernels (border rule, rounding,
// interpolation behaviour of the existing NN-only implementation).
TEST_F(NppiRotateDiff, Rotate_ExistingProbes) {
  rotate_case<Npp8u>("nppiRotate_8u_C1R", "8u_C1R/angle30_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN, true,
                     &nppiRotate_8u_C1R_Ctx, 530);
  rotate_case<Npp8u>("nppiRotate_8u_C1R", "8u_C1R/angle30_linear", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_LINEAR, true,
                     &nppiRotate_8u_C1R_Ctx, 531);
  rotate_case<Npp8u>("nppiRotate_8u_C4R", "8u_C4R/angle30_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                     NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN, true,
                     &nppiRotate_8u_C4R_Ctx, 532);
  rotate_case<Npp16u>("nppiRotate_16u_C3R", "16u_C3R/angle30_nn", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN, true,
                      &nppiRotate_16u_C3R_Ctx, 533);
  rotate_case<Npp32f>("nppiRotate_32f_C1R", "32f_C1R/angle30_linear", NppiSize{24, 16}, NppiRect{0, 0, 24, 16},
                      NppiSize{40, 32}, NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_LINEAR, true,
                      &nppiRotate_32f_C1R_Ctx, 534);
}

// ===== contract: null pointer, steps, sizes, ROIs ===========================
TEST_F(NppiRotateDiff, Rotate_Contract) {
  const NppiSize srcSize{24, 16};
  const NppiRect srcROI{0, 0, 24, 16};
  const NppiSize dstSize{40, 32};
  const int es = 4;
  const int sStep = align_step(srcSize.width * es + 8, 1);
  const int dStep = align_step(dstSize.width * es + 8, 1);
  const size_t se = (size_t)srcSize.height * sStep;
  const size_t de = (size_t)dstSize.height * dStep;
  // Oversized allocation: the src_roi_oob probe reads up to (35, 15) if the
  // implementation does not validate the ROI, which must not crash.
  DeviceMemory<Npp8u> &s = arena<Npp8u>(8000);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(std::vector<Npp8u>(8000, 3));
  d.copyFromHost(std::vector<Npp8u>(de, 0xA5));

  auto run = [&](const std::string &key, const Npp8u *psrc, NppiSize ssize, int sstep, NppiRect sroi, Npp8u *pdst,
                 int dstep, NppiRect droi, int interp) {
    clear_cuda_error();
    NppStatus st = nppiRotate_8u_AC4R_Ctx(psrc, ssize, sstep, sroi, pdst, dstep, droi, 30.0, 0.0, 0.0, interp,
                                          current_ctx());
    Case c("nppiRotate/contract/" + key, "nppiRotate_8u_AC4R");
    c.status(st);
    c.finish();
  };
  run("null_src", nullptr, srcSize, sStep, srcROI, d.get(), dStep, NppiRect{4, 6, 32, 22}, NPPI_INTER_NN);
  run("null_dst", s.get(), srcSize, sStep, srcROI, nullptr, dStep, NppiRect{4, 6, 32, 22}, NPPI_INTER_NN);
  run("src_step0", s.get(), srcSize, 0, srcROI, d.get(), dStep, NppiRect{4, 6, 32, 22}, NPPI_INTER_NN);
  run("dst_step0", s.get(), srcSize, sStep, srcROI, d.get(), 0, NppiRect{4, 6, 32, 22}, NPPI_INTER_NN);
  run("src_size_empty", s.get(), NppiSize{0, 16}, sStep, srcROI, d.get(), dStep, NppiRect{4, 6, 32, 22},
      NPPI_INTER_NN);
  run("src_roi_w0", s.get(), srcSize, sStep, NppiRect{0, 0, 0, 16}, d.get(), dStep, NppiRect{4, 6, 32, 22},
      NPPI_INTER_NN);
  run("src_roi_neg", s.get(), srcSize, sStep, NppiRect{0, 0, -8, 16}, d.get(), dStep, NppiRect{4, 6, 32, 22},
      NPPI_INTER_NN);
  run("dst_roi_w0", s.get(), srcSize, sStep, srcROI, d.get(), dStep, NppiRect{4, 6, 0, 22}, NPPI_INTER_NN);
  run("dst_roi_h0", s.get(), srcSize, sStep, srcROI, d.get(), dStep, NppiRect{4, 6, 32, 0}, NPPI_INTER_NN);
  run("src_roi_oob", s.get(), srcSize, sStep, NppiRect{4, 4, 32, 12}, d.get(), dStep, NppiRect{4, 6, 32, 22},
      NPPI_INTER_NN);
  run("src_roi_neg_xy", s.get(), srcSize, sStep, NppiRect{-2, 0, 24, 16}, d.get(), dStep, NppiRect{4, 6, 32, 22},
      NPPI_INTER_NN);
  run("interp_undefined", s.get(), srcSize, sStep, srcROI, d.get(), dStep, NppiRect{4, 6, 32, 22},
      NPPI_INTER_UNDEFINED);
  run("interp_super", s.get(), srcSize, sStep, srcROI, d.get(), dStep, NppiRect{4, 6, 32, 22}, NPPI_INTER_SUPER);
}

// ===== Non-Ctx adapters (representative _Ctx cases -> same golden) ==========
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiRotateDiff, NonCtx_Rotate) {
  // Reproduces rotate_case(8u_AC4R, angle30_nn, positional, seed=501).
  const int stride = 4, es = 1;
  const int sStep = align_step(24 * stride * es + 8, es);
  const int dStep = align_step(40 * stride * es + 8, es);
  const size_t se = (size_t)16 * sStep / es;
  const size_t de = (size_t)32 * dStep / es;
  std::vector<Npp8u> hs = make_positional<Npp8u>(se, stride, sStep / es);
  std::vector<Npp8u> hd(de, 0xA5);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiRotate_8u_AC4R(s.get(), NppiSize{24, 16}, sStep, NppiRect{0, 0, 24, 16}, d.get(), dStep,
                                    NppiRect{4, 6, 32, 22}, 30.0, 0.0, 0.0, NPPI_INTER_NN);
  Case c("nppiRotate/8u_AC4R/angle30_nn", "nppiRotate_8u_AC4R");
  c.input(hs.data(), hs.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 1e-5, 1e-6);
  c.finish();

  // Reproduces rotate_case(16u_AC4R, angle0, positional, seed=510).
  const int sStep16 = align_step(24 * 4 * 2 + 8, 2);
  const size_t se16 = (size_t)16 * sStep16 / 2;
  std::vector<Npp16u> hs16 = make_positional<Npp16u>(se16, 4, sStep16 / 2);
  std::vector<Npp16u> hd16((size_t)16 * sStep16 / 2, 0xA5);
  DeviceMemory<Npp16u> &s16 = arena<Npp16u>(se16);
  DeviceMemory<Npp16u> &d16 = arena<Npp16u>(se16);
  s16.copyFromHost(hs16);
  d16.copyFromHost(hd16);
  clear_cuda_error();
  NppStatus st16 = nppiRotate_16u_AC4R(s16.get(), NppiSize{24, 16}, sStep16, NppiRect{0, 0, 24, 16}, d16.get(),
                                       sStep16, NppiRect{0, 0, 24, 16}, 0.0, 0.0, 0.0, NPPI_INTER_NN);
  Case c16("nppiRotate/16u_AC4R/angle0", "nppiRotate_16u_AC4R");
  c16.input(hs16.data(), hs16.size() * sizeof(Npp16u));
  c16.status(st16);
  c16.device_output<Npp16u>(d16.get(), se16, ValueKind::U16, 1e-5, 1e-6);
  c16.finish();

  // Reproduces rotate_case(32f_AC4R, angle_neg25_5_nn, positional, seed=522).
  const int sStep32 = align_step(24 * 4 * 4 + 8, 4);
  const int dStep32 = align_step(40 * 4 * 4 + 8, 4);
  const size_t se32 = (size_t)16 * sStep32 / 4;
  const size_t de32 = (size_t)32 * dStep32 / 4;
  std::vector<Npp32f> hs32 = make_positional<Npp32f>(se32, 4, sStep32 / 4);
  std::vector<Npp32f> hd32(de32, (Npp32f)0xA5);
  DeviceMemory<Npp32f> &s32 = arena<Npp32f>(se32);
  DeviceMemory<Npp32f> &d32 = arena<Npp32f>(de32);
  s32.copyFromHost(hs32);
  d32.copyFromHost(hd32);
  clear_cuda_error();
  NppStatus st32 = nppiRotate_32f_AC4R(s32.get(), NppiSize{24, 16}, sStep32, NppiRect{0, 0, 24, 16}, d32.get(),
                                       dStep32, NppiRect{6, 4, 28, 24}, -25.5, 10.0, 8.0, NPPI_INTER_NN);
  Case c32("nppiRotate/32f_AC4R/angle_neg25_5_nn", "nppiRotate_32f_AC4R");
  c32.input(hs32.data(), hs32.size() * sizeof(Npp32f));
  c32.status(st32);
  c32.device_output<Npp32f>(d32.get(), de32, ValueKind::F32, 1e-5, 1e-6);
  c32.finish();
}
#endif
