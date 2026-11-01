// Package g4 probe suite: interpolation kernels, clip rules and the
// nppiResize / nppiResizeBatch ROI mapping, measured on the RTX 5090 against
// NVIDIA NPP 13.4.
//
// Record:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiResizeProbe*.*'
//
// The impulse-response cases put a single non-zero sample in the source and a
// zero destination, so the golden output *is* the interpolation kernel for
// that phase; the coordinate-coded cases let the reader decode the exact
// source pixel and the exact destination columns/rows that NVIDIA writes.

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

int align_step(int step, int es) { return ((step + 15) / 16) * 16; }

// Single-unit impulse in a zero 32f image, then one SqrPixel call.
//   fx/fy/sx/sy: the resize factors / shifts
//   chans: only the channel-0 plane of a packed image is excited when > 1
void impulse_sqr(const char *fname, const std::string &key, int srcW, int srcH, int px, int py, float amp, double fx,
                 double fy, double sx, double sy, int interp, int chans, int dstW = 0, int dstH = 0) {
  if (dstW == 0) dstW = srcW;
  if (dstH == 0) dstH = srcH;
  const int sStep = align_step(srcW * chans * 4, 4);
  const int dStep = align_step(dstW * chans * 4, 4);
  const size_t se = (size_t)srcH * sStep / 4;
  const size_t de = (size_t)dstH * dStep / 4;
  std::vector<Npp32f> hs(se, 0.0f);
  hs[(size_t)py * (sStep / 4) + (size_t)px * chans] = amp;
  std::vector<Npp32f> hd(de, 0.0f);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiResizeSqrPixel_32f_C1R_Ctx(s.get(), NppiSize{srcW, srcH}, sStep, NppiRect{0, 0, srcW, srcH},
                                                d.get(), dStep, NppiRect{0, 0, dstW, dstH}, fx, fy, sx, sy, interp,
                                                current_ctx());
  Case c("nppiResizeProbe/sqr_impulse/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-6, 1e-6);
  c.finish();
}

// Impulse through nppiResizeSqrPixel_8u_C1R_Advanced (LANCZOS3_ADVANCED).
void impulse_sqr_advanced(const char *fname, const std::string &key, int srcW, int srcH, int px, int py, int amp,
                          double fx, double fy, double sx, double sy, int dstW, int dstH) {
  const int sStep = align_step(srcW, 1);
  const int dStep = align_step(dstW, 1);
  const size_t se = (size_t)srcH * sStep;
  const size_t de = (size_t)dstH * dStep;
  std::vector<Npp8u> hs(se, 0);
  hs[(size_t)py * sStep + px] = (Npp8u)amp;
  std::vector<Npp8u> hd(de, 0xEE);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  const size_t bufSize = 4u * (size_t)dstW * srcH + 65536u;
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(bufSize);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  buf.copyFromHost(std::vector<Npp8u>(bufSize, 0));
  clear_cuda_error();
  NppStatus st = nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(
      s.get(), NppiSize{srcW, srcH}, sStep, NppiRect{0, 0, srcW, srcH}, d.get(), dStep, NppiRect{0, 0, dstW, dstH}, fx,
      fy, buf.get(), NPPI_INTER_LANCZOS3_ADVANCED, current_ctx());
  Case c("nppiResizeProbe/sqr_advanced_impulse/" + key, fname);
  c.input(hs.data(), hs.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8);
  c.finish();
}

// Coordinate-coded 8u source: value = y * w + x (w <= 15, h <= 15 so it fits).
std::vector<Npp8u> code_image_u8(int step, int w, int h) {
  std::vector<Npp8u> v((size_t)h * step, 0x7B);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) v[(size_t)y * step + x] = (Npp8u)(y * w + x);
  }
  return v;
}

// One SqrPixel 8u C1 call on a coordinate-coded source, poison destination.
void code_sqr(const std::string &key, int srcW, int srcH, NppiRect srcROI, int dstW, int dstH, NppiRect dstROI, double fx,
              double fy, double sx, double sy, int interp) {
  const int sStep = align_step(srcW, 1);
  const int dStep = align_step(dstW, 1);
  const size_t de = (size_t)dstH * dStep;
  std::vector<Npp8u> hs = code_image_u8(sStep, srcW, srcH);
  std::vector<Npp8u> hd(de, 0xEE);
  DeviceMemory<Npp8u> &s = arena<Npp8u>((size_t)srcH * sStep);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiResizeSqrPixel_8u_C1R_Ctx(s.get(), NppiSize{srcW, srcH}, sStep, srcROI, d.get(), dStep, dstROI,
                                               fx, fy, sx, sy, interp, current_ctx());
  Case c(std::string("nppiResizeProbe/sqr_code/") + key, "nppiResizeSqrPixel_8u_C1R_Ctx");
  c.input(hs.data(), hs.size());
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8);
  c.finish();
}

// Coordinate-coded 32f SqrPixel (decode code = y * 1024 + x).
void code_sqr_32f(const std::string &key, int srcW, int srcH, NppiRect srcROI, int dstW, int dstH, NppiRect dstROI, double fx,
                  double fy, double sx, double sy, int interp) {
  const int sStep = align_step(srcW * 4, 4);
  const int dStep = align_step(dstW * 4, 4);
  const size_t se = (size_t)srcH * sStep / 4;
  const size_t de = (size_t)dstH * dStep / 4;
  std::vector<Npp32f> hs(se, -7.0f);
  for (int y = 0; y < srcH; ++y)
    for (int x = 0; x < srcW; ++x) hs[(size_t)y * (sStep / 4) + x] = (Npp32f)(y * 1024 + x);
  std::vector<Npp32f> hd(de, -1e30f);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiResizeSqrPixel_32f_C1R_Ctx(s.get(), NppiSize{srcW, srcH}, sStep, srcROI, d.get(), dStep, dstROI,
                                                fx, fy, sx, sy, interp, current_ctx());
  Case c(std::string("nppiResizeProbe/sqr_code32f/") + key, "nppiResizeSqrPixel_32f_C1R_Ctx");
  c.input(hs.data(), hs.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-6, 1e-6);
  c.finish();
}

constexpr int kInterpNN = NPPI_INTER_NN;
constexpr int kInterpLinear = NPPI_INTER_LINEAR;
constexpr int kInterpCubic = NPPI_INTER_CUBIC;
constexpr int kInterpBSpline = NPPI_INTER_CUBIC2P_BSPLINE;
constexpr int kInterpCatmullRom = NPPI_INTER_CUBIC2P_CATMULLROM;
constexpr int kInterpB05C03 = NPPI_INTER_CUBIC2P_B05C03;
constexpr int kInterpSuper = NPPI_INTER_SUPER;
constexpr int kInterpLanczos = NPPI_INTER_LANCZOS;

struct ModeName {
  int mode;
  const char *name;
};
constexpr ModeName kModes[] = {{kInterpLinear, "linear"},   {kInterpCubic, "cubic"},
                               {kInterpBSpline, "bspline"}, {kInterpCatmullRom, "catmullrom"},
                               {kInterpB05C03, "b05c03"},   {kInterpLanczos, "lanczos"}};

} // namespace

class NppiResizeProbeDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== Impulse responses: one phase per 1/16 step, factors 1/2 =============
TEST_F(NppiResizeProbeDiff, Impulse_32f_C1_Phases) {
  const int W = 32, H = 32;
  for (const auto &m : kModes) {
    for (int k = 0; k < 16; ++k) {
      const double s = k / 16.0;
      const std::string key = std::string(m.name) + "/phase" + std::to_string(k) + "_f1";
      impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", key, W, H, 16, 16, 1.0f, 1.0, 1.0, s, s, m.mode, 1);
    }
    // Upscale and downscale phases for the same kernel.
    impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", std::string(m.name) + "/up2_phase0", W, H, 16, 16, 1.0f, 2.0, 2.0,
                0.0, 0.0, m.mode, 1, W * 2, H * 2);
    impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", std::string(m.name) + "/up2_phase8", W, H, 16, 16, 1.0f, 2.0, 2.0,
                0.5, 0.5, m.mode, 1, W * 2, H * 2);
    impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", std::string(m.name) + "/down2_phase0", W, H, 16, 16, 1.0f, 0.5, 0.5,
                0.0, 0.0, m.mode, 1, W / 2, H / 2);
    impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", std::string(m.name) + "/down2_phase8", W, H, 16, 16, 1.0f, 0.5, 0.5,
                0.5, 0.5, m.mode, 1, W / 2, H / 2);
  }
}

// NN and SUPER have their own sampling rules; record the same impulse set.
TEST_F(NppiResizeProbeDiff, Impulse_NN_Super) {
  const int W = 32, H = 32;
  for (int k = 0; k < 16; ++k) {
    impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", "nn/phase" + std::to_string(k) + "_f1", W, H, 16, 16, 1.0f, 1.0,
                1.0, k / 16.0, k / 16.0, kInterpNN, 1);
  }
  for (int k = 0; k < 8; ++k) {
    const double s = k / 8.0;
    impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", "super/down2_phase" + std::to_string(k), W, H, 16, 16, 1.0f, 0.5, 0.5,
                s, s, kInterpSuper, 1, W / 2, H / 2);
  }
  impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", "super/down4_phase0", W, H, 16, 16, 1.0f, 0.25, 0.25, 0.0, 0.0,
              kInterpSuper, 1, W / 4, H / 4);
  impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", "super/down3_phase0", W, H, 12, 12, 1.0f, 0.75, 0.75, 0.0, 0.0,
              kInterpSuper, 1, 16, 16);
  impulse_sqr("nppiResizeSqrPixel_32f_C1R_Ctx", "super/down13_phase0", W, H, 16, 16, 1.0f, 0.75, 0.75, 0.0, 0.0,
              kInterpSuper, 1, 24, 24);
}

// ===== Advanced (LANCZOS3_ADVANCED) impulse response ======================
TEST_F(NppiResizeProbeDiff, Impulse_Advanced_8u_C1) {
  const int W = 32, H = 32;
  for (int k = 0; k < 8; ++k) {
    const double s = k / 8.0;
    impulse_sqr_advanced("nppiResizeSqrPixel_8u_C1R_Advanced_Ctx", "phase" + std::to_string(k) + "_f1", W, H, 16, 16,
                         255, 1.0, 1.0, s, s, W, H);
  }
  for (int k = 0; k < 4; ++k) {
    const double s = k / 4.0;
    impulse_sqr_advanced("nppiResizeSqrPixel_8u_C1R_Advanced_Ctx", "down2_phase" + std::to_string(k), W, H, 16, 16, 255,
                         0.5, 0.5, s, s, W / 2, H / 2);
  }
  impulse_sqr_advanced("nppiResizeSqrPixel_8u_C1R_Advanced_Ctx", "up2_phase0", W, H, 16, 16, 255, 2.0, 2.0, 0.0, 0.0,
                       W * 2, H * 2);
}

// ===== Mapping / clip, coordinate coded ===================================
TEST_F(NppiResizeProbeDiff, Clip_8u_C1) {
  const int W = 12, H = 12;
  const NppiRect roi{3, 2, 7, 8};
  const NppiRect roiOut{8, 0, 10, 10};
  const NppiRect dstFull{0, 0, 18, 18};
  const int modes[] = {kInterpNN, kInterpLinear, kInterpCubic, kInterpLanczos, kInterpBSpline, kInterpCatmullRom};
  const char *names[] = {"nn", "linear", "cubic", "lanczos", "bspline", "catmullrom"};
  for (int i = 0; i < 6; ++i) {
    for (double f : {1.0, 2.0, 0.5}) {
      const int dw = (int)(W * f), dh = (int)(H * f);
      code_sqr(std::string(names[i]) + "_roi_f" + std::to_string(f), W, H, roi, dw, dh, NppiRect{0, 0, dw, dh}, f, f,
               0.0, 0.0, modes[i]);
      code_sqr(std::string(names[i]) + "_roi_shift_f" + std::to_string(f), W, H, roi, dw, dh, NppiRect{0, 0, dw, dh},
               f, f, 0.3, 0.6, modes[i]);
    }
    // ROI extending beyond the image and a destination ROI that is a sub-rect.
    code_sqr(std::string(names[i]) + "_roi_beyond", W, H, roiOut, 14, 14, dstFull, 1.0, 1.0, 0.0, 0.0, modes[i]);
    code_sqr(std::string(names[i]) + "_dst_sub", W, H, roi, 20, 20, NppiRect{2, 3, 11, 10}, 1.5, 1.5, 0.0, 0.0,
             modes[i]);
  }
}

TEST_F(NppiResizeProbeDiff, Clip_32f_C1) {
  const int W = 12, H = 12;
  const NppiRect roi{3, 2, 7, 8};
  const int modes[] = {kInterpNN, kInterpLinear, kInterpCubic, kInterpLanczos, kInterpSuper};
  const char *names[] = {"nn", "linear", "cubic", "lanczos", "super"};
  for (int i = 0; i < 5; ++i) {
    code_sqr_32f(std::string(names[i]) + "_roi_f2", W, H, roi, 24, 24, NppiRect{0, 0, 24, 24}, 2.0, 2.0, 0.0, 0.0,
                 modes[i]);
    code_sqr_32f(std::string(names[i]) + "_roi_f15_shift", W, H, roi, 18, 18, NppiRect{0, 0, 18, 18}, 1.5, 1.5, 0.25,
                 0.75, modes[i]);
  }
}

// ===== nppiResize family: ROI mapping =====================================
namespace {
void code_resize_32f(const std::string &key, int srcW, int srcH, NppiRect srcROI, int dstW, int dstH, NppiRect dstROI,
                     int interp, int chans = 1) {
  const int sStep = align_step(srcW * chans * 4, 4);
  const int dStep = align_step(dstW * chans * 4, 4);
  const size_t se = (size_t)srcH * sStep / 4;
  const size_t de = (size_t)dstH * dStep / 4;
  std::vector<Npp32f> hs(se, -7.0f);
  for (int y = 0; y < srcH; ++y)
    for (int x = 0; x < srcW; ++x)
      for (int c = 0; c < chans; ++c)
        hs[(size_t)y * (sStep / 4) + (size_t)x * chans + c] = (Npp32f)(c * 65536 + y * 1024 + x);
  std::vector<Npp32f> hd(de, -1e30f);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiResize_32f_C1R_Ctx(s.get(), sStep, NppiSize{srcW, srcH}, srcROI, d.get(), dStep,
                                        NppiSize{dstW, dstH}, dstROI, interp, current_ctx());
  Case c("nppiResizeProbe/resize/" + std::string(key), "nppiResize_32f_C1R_Ctx");
  c.input(hs.data(), hs.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-6, 1e-6);
  c.finish();
}
} // namespace

class NppiResizeFamilyProbeDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiResizeFamilyProbeDiff, Mapping_32f_C1) {
  const int W = 12, H = 12;
  const int modes[] = {kInterpNN, kInterpLinear, kInterpCubic, kInterpSuper};
  const char *names[] = {"nn", "linear", "cubic", "super"};
  for (int i = 0; i < 4; ++i) {
    code_resize_32f(std::string(names[i]) + "_full_full", W, H, NppiRect{0, 0, W, H}, 24, 24,
                    NppiRect{0, 0, 24, 24}, modes[i]);
    code_resize_32f(std::string(names[i]) + "_roi_roi", W, H, NppiRect{2, 3, 8, 6}, 16, 12,
                    NppiRect{1, 2, 12, 8}, modes[i]);
    code_resize_32f(std::string(names[i]) + "_roi_full", W, H, NppiRect{2, 3, 8, 6}, 16, 12,
                    NppiRect{0, 0, 16, 12}, modes[i]);
    code_resize_32f(std::string(names[i]) + "_full_dstsub", W, H, NppiRect{0, 0, W, H}, 20, 20,
                    NppiRect{3, 2, 12, 10}, modes[i]);
  }
  code_resize_32f("nn_roi_beyond", W, H, NppiRect{8, 0, 12, 12}, 16, 16, NppiRect{0, 0, 16, 16}, kInterpNN);
}

// Error paths of nppiResize (nppiResize_8u_C1R_Ctx).
TEST_F(NppiResizeFamilyProbeDiff, Contract_8u_C1) {
  const char *fname = "nppiResize_8u_C1R_Ctx";
  struct Cfg {
    const char *key;
    NppiSize srcSize;
    int sStep;
    NppiRect srcROI;
    NppiSize dstSize;
    int dStep;
    NppiRect dstROI;
    int interp;
    bool nullSrc, nullDst;
  };
  const NppiSize good{8, 8};
  const int gs = align_step(8, 1), gd = align_step(8, 1);
  const NppiRect gsr{0, 0, 8, 8};
  const NppiRect gdr{0, 0, 8, 8};
  const Cfg cfgs[] = {
      {"null_src", good, gs, gsr, good, gd, gdr, kInterpNN, true, false},
      {"null_dst", good, gs, gsr, good, gd, gdr, kInterpNN, false, true},
      {"step_src", good, 2, gsr, good, gd, gdr, kInterpNN, false, false},
      {"step_dst", good, gs, gsr, good, 2, gdr, kInterpNN, false, false},
      {"src_zero_w", NppiSize{0, 8}, gs, gsr, good, gd, gdr, kInterpNN, false, false},
      {"src_one_w", NppiSize{1, 8}, gs, NppiRect{0, 0, 1, 8}, good, gd, gdr, kInterpNN, false, false},
      {"src_neg", NppiSize{-4, 8}, gs, gsr, good, gd, gdr, kInterpNN, false, false},
      {"src_roi_zero_w", good, gs, NppiRect{0, 0, 0, 8}, good, gd, gdr, kInterpNN, false, false},
      {"src_roi_neg_w", good, gs, NppiRect{0, 0, -2, 8}, good, gd, gdr, kInterpNN, false, false},
      {"src_roi_no_intersect", good, gs, NppiRect{20, 0, 4, 4}, good, gd, gdr, kInterpNN, false, false},
      {"src_roi_partial", good, gs, NppiRect{4, 0, 8, 8}, good, gd, gdr, kInterpNN, false, false},
      {"dst_zero_w", good, gs, gsr, good, gd, NppiRect{0, 0, 0, 8}, kInterpNN, false, false},
      {"dst_neg_w", good, gs, gsr, good, gd, NppiRect{0, 0, -1, 8}, kInterpNN, false, false},
      {"dst_size_zero_w", good, gs, gsr, NppiSize{0, 8}, gd, gdr, kInterpNN, false, false},
      {"dst_size_neg", good, gs, gsr, NppiSize{-2, 8}, gd, gdr, kInterpNN, false, false},
      {"interp_bad", good, gs, gsr, good, gd, gdr, 3, false, false},
      {"interp_super_up", good, gs, gsr, NppiSize{16, 16}, align_step(16, 1), NppiRect{0, 0, 16, 16}, kInterpSuper,
       false, false},
      {"order_null_and_step", good, 2, gsr, good, gd, gdr, kInterpNN, true, false},
      {"order_step_and_interp", good, 2, gsr, good, gd, gdr, 3, false, false},
      {"order_stepdst_and_interp", good, gs, gsr, good, 2, gdr, 3, false, false},
      {"order_dstneg_and_interp", good, gs, gsr, good, gd, NppiRect{0, 0, -1, 8}, 3, false, false},
      {"order_nointersect_and_interp", good, gs, NppiRect{20, 0, 4, 4}, good, gd, gdr, 3, false, false},
      {"order_srcneg_and_roineg", NppiSize{-4, 8}, gs, NppiRect{0, 0, -2, 8}, good, gd, gdr, kInterpNN, false, false},
  };
  for (const auto &cfg : cfgs) {
    const size_t se = (size_t)(cfg.srcSize.height > 0 ? cfg.srcSize.height : 1) * 16;
    const size_t de = (size_t)(cfg.dstSize.height > 0 ? cfg.dstSize.height : 1) * 16;
    DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
    s.copyFromHost(make_uniform<Npp8u>(se, 900));
    d.copyFromHost(std::vector<Npp8u>(de, 0xEE));
    clear_cuda_error();
    NppStatus st = nppiResize_8u_C1R_Ctx(cfg.nullSrc ? nullptr : s.get(), cfg.sStep, cfg.srcSize, cfg.srcROI,
                                         cfg.nullDst ? nullptr : d.get(), cfg.dStep, cfg.dstSize, cfg.dstROI,
                                         cfg.interp, current_ctx());
    Case c(std::string("nppiResizeProbe/resize_contract/") + cfg.key, fname);
    c.status(st);
    c.device_output<Npp8u>(d.get(), de, ValueKind::U8);
    c.finish();
  }
}

// ===== nppiResizeBatch ====================================================
namespace {
struct BatchImages {
  std::vector<std::vector<Npp8u>> host;
  std::vector<DeviceMemory<Npp8u> *> dev;
};

// Simple batch of 3 packed 8u C1 images, each 10x9, with a per-image pattern.
BatchImages make_batch_u8(int n, int w, int h, int step, uint64_t seed) {
  BatchImages b;
  for (int i = 0; i < n; ++i) {
    std::vector<Npp8u> v((size_t)h * step, 0x55);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) v[(size_t)y * step + x] = (Npp8u)((y * w + x + i * 37) & 0xFF);
    b.host.push_back(v);
    auto &d = arena<Npp8u>((size_t)h * step);
    d.copyFromHost(v);
    b.dev.push_back(&d);
  }
  (void)seed;
  return b;
}
} // namespace

class NppiResizeBatchProbeDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiResizeBatchProbeDiff, Basic_8u_C1) {
  const int n = 3, W = 10, H = 9, sStep = align_step(W, 1), dW = 20, dH = 18, dStep = align_step(dW, 1);
  BatchImages src = make_batch_u8(n, W, H, sStep, 1);
  std::vector<std::vector<Npp8u>> hd;
  std::vector<DeviceMemory<Npp8u> *> dev;
  for (int i = 0; i < n; ++i) {
    hd.push_back(std::vector<Npp8u>((size_t)dH * dStep, 0xEE));
    auto &d = arena<Npp8u>((size_t)dH * dStep);
    d.copyFromHost(hd.back());
    dev.push_back(&d);
  }
  std::vector<NppiResizeBatchCXR> list(n);
  for (int i = 0; i < n; ++i) {
    list[i].pSrc = src.dev[i]->get();
    list[i].nSrcStep = sStep;
    list[i].pDst = dev[i]->get();
    list[i].nDstStep = dStep;
  }
  DeviceMemory<NppiResizeBatchCXR> &dl = arena<NppiResizeBatchCXR>(n);
  dl.copyFromHost(list);
  clear_cuda_error();
  NppStatus st = nppiResizeBatch_8u_C1R_Ctx(NppiSize{W, H}, NppiRect{0, 0, W, H}, NppiSize{dW, dH},
                                            NppiRect{0, 0, dW, dH}, NPPI_INTER_LINEAR, dl.get(), n, current_ctx());
  Case c("nppiResizeProbe/batch/basic_8u_c1", "nppiResizeBatch_8u_C1R_Ctx");
  for (int i = 0; i < n; ++i) c.input(src.host[i].data(), src.host[i].size());
  c.status(st);
  for (int i = 0; i < n; ++i) c.device_output<Npp8u>(dev[i]->get(), (size_t)dH * dStep, ValueKind::U8);
  c.finish();
}

TEST_F(NppiResizeBatchProbeDiff, Advanced_8u_C1) {
  const int n = 3, W = 12, H = 10, sStep = align_step(W, 1), dW = 24, dH = 20, dStep = align_step(dW, 1);
  BatchImages src = make_batch_u8(n, W, H, sStep, 2);
  std::vector<std::vector<Npp8u>> hd;
  std::vector<DeviceMemory<Npp8u> *> dev;
  for (int i = 0; i < n; ++i) {
    hd.push_back(std::vector<Npp8u>((size_t)dH * dStep, 0xEE));
    auto &d = arena<Npp8u>((size_t)dH * dStep);
    d.copyFromHost(hd.back());
    dev.push_back(&d);
  }
  std::vector<NppiImageDescriptor> sdesc(n), ddesc(n);
  std::vector<NppiResizeBatchROI_Advanced> rois(n);
  for (int i = 0; i < n; ++i) {
    sdesc[i].pData = src.dev[i]->get();
    sdesc[i].nStep = sStep;
    sdesc[i].oSize = NppiSize{W, H};
    ddesc[i].pData = dev[i]->get();
    ddesc[i].nStep = dStep;
    ddesc[i].oSize = NppiSize{dW, dH};
    // per-image variable ROIs: index 1 uses a sub-rectangle and a different
    // destination area so the per-image ROI mapping is exercised.
    if (i == 1) {
      rois[i].oSrcRectROI = NppiRect{2, 1, 8, 6};
      rois[i].oDstRectROI = NppiRect{3, 2, 16, 12};
    } else {
      rois[i].oSrcRectROI = NppiRect{0, 0, W, H};
      rois[i].oDstRectROI = NppiRect{0, 0, dW, dH};
    }
  }
  DeviceMemory<NppiImageDescriptor> &ds = arena<NppiImageDescriptor>(n);
  DeviceMemory<NppiImageDescriptor> &dd = arena<NppiImageDescriptor>(n);
  DeviceMemory<NppiResizeBatchROI_Advanced> &dr = arena<NppiResizeBatchROI_Advanced>(n);
  ds.copyFromHost(sdesc);
  dd.copyFromHost(ddesc);
  dr.copyFromHost(rois);
  clear_cuda_error();
  NppStatus st = nppiResizeBatch_8u_C1R_Advanced_Ctx(dW, dH, ds.get(), dd.get(), dr.get(), n, NPPI_INTER_LINEAR,
                                                     current_ctx());
  Case c("nppiResizeProbe/batch/advanced_8u_c1", "nppiResizeBatch_8u_C1R_Advanced_Ctx");
  for (int i = 0; i < n; ++i) c.input(src.host[i].data(), src.host[i].size());
  c.status(st);
  for (int i = 0; i < n; ++i) c.device_output<Npp8u>(dev[i]->get(), (size_t)dH * dStep, ValueKind::U8);
  c.finish();
}

// ===== Ramp probes: the output value IS the effective source coordinate ====
// A source ramp (value = x) is reproduced exactly by any interpolating filter,
// so the recorded output reveals the sampled coordinate (and the filter
// normalisation) without any inverse fitting.
namespace {
template <typename T>
void ramp_sqr(const std::string &key, const char *fname, int srcW, int srcH, int dstW, int dstH, double fx, double fy,
              double sx, double sy, int interp, int chans, const std::vector<T> &pixels) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcW * chans * es, es);
  const int dStep = align_step(dstW * chans * es, es);
  const size_t se = (size_t)srcH * sStep / es;
  const size_t de = (size_t)dstH * dStep / es;
  std::vector<T> hs(se, (T)0);
  for (int y = 0; y < srcH; ++y)
    for (int x = 0; x < srcW * chans; ++x) hs[(size_t)y * (sStep / es) + x] = pixels[(size_t)y * srcW * chans + x];
  std::vector<T> hd(de, (T)0);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp32f>) {
    st = nppiResizeSqrPixel_32f_C1R_Ctx(s.get(), NppiSize{srcW, srcH}, sStep, NppiRect{0, 0, srcW, srcH}, d.get(),
                                        dStep, NppiRect{0, 0, dstW, dstH}, fx, fy, sx, sy, interp, current_ctx());
  } else if constexpr (std::is_same_v<T, Npp64f>) {
    st = nppiResizeSqrPixel_64f_C1R_Ctx(s.get(), NppiSize{srcW, srcH}, sStep, NppiRect{0, 0, srcW, srcH}, d.get(),
                                        dStep, NppiRect{0, 0, dstW, dstH}, fx, fy, sx, sy, interp, current_ctx());
  } else {
    st = nppiResizeSqrPixel_16u_C1R_Ctx(s.get(), NppiSize{srcW, srcH}, sStep, NppiRect{0, 0, srcW, srcH}, d.get(),
                                        dStep, NppiRect{0, 0, dstW, dstH}, fx, fy, sx, sy, interp, current_ctx());
  }
  Case c("nppiResizeProbe/ramp/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  if constexpr (std::is_same_v<T, Npp32f>) {
    c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-6, 1e-6);
  } else if constexpr (std::is_same_v<T, Npp64f>) {
    c.device_output<Npp64f>(d.get(), de, ValueKind::F64, 1e-12, 1e-12);
  } else {
    c.device_output<Npp16u>(d.get(), de, ValueKind::U16);
  }
  c.finish();
}
} // namespace

TEST_F(NppiResizeProbeDiff, Ramp_32f_C1) {
  const int W = 16, H = 16;
  // value 1.0 everywhere is useless; the ramp value is the row/column index,
  // carried per channel slot: channel 0 = x ramp, then a second run = y ramp.
  struct SH {
    double sx, sy;
    const char *name;
  };
  const SH shs[] = {{0.25, 0.0, "sx025"},  {0.75, 0.0, "sx075"},  {-0.25, 0.0, "sxm025"},
                    {-0.75, 0.0, "sxm075"},{0.0, 0.25, "sy025"},  {0.0, 0.75, "sy075"},
                    {0.3, 0.6, "sxy0306"}, {0.5, 0.5, "sxy0505"}};
  const int modes[] = {kInterpNN, kInterpLinear, kInterpCubic, kInterpLanczos};
  const char *names[] = {"nn", "linear", "cubic", "lanczos"};
  for (int m = 0; m < 4; ++m) {
    for (const auto &s : shs) {
      // x ramp: value(x,y) = y*1024 + x so both coordinates are recoverable
      std::vector<Npp32f> rampx(W * H);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) rampx[(size_t)y * W + x] = (Npp32f)(y * 1024 + x);
      ramp_sqr<Npp32f>(std::string(names[m]) + "_" + s.name + "_f1", "nppiResizeSqrPixel_32f_C1R_Ctx", W, H, W, H, 1.0,
                       1.0, s.sx, s.sy, modes[m], 1, rampx);
      // x-only ramp: value = x (vertical blending then keeps the value)
      std::vector<Npp32f> xonly(W * H), yonly(W * H);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          xonly[(size_t)y * W + x] = (Npp32f)x;
          yonly[(size_t)y * W + x] = (Npp32f)y;
        }
      ramp_sqr<Npp32f>(std::string(names[m]) + "_" + s.name + "_f1_xonly", "nppiResizeSqrPixel_32f_C1R_Ctx", W, H, W, H,
                       1.0, 1.0, s.sx, s.sy, modes[m], 1, xonly);
      ramp_sqr<Npp32f>(std::string(names[m]) + "_" + s.name + "_f1_yonly", "nppiResizeSqrPixel_32f_C1R_Ctx", W, H, W, H,
                       1.0, 1.0, s.sx, s.sy, modes[m], 1, yonly);
    }
    // non-unit factors with a shifted phase
    std::vector<Npp32f> ramp2(32 * 32);
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) ramp2[(size_t)y * 32 + x] = (Npp32f)(y * 1024 + x);
    ramp_sqr<Npp32f>(std::string(names[m]) + "_f2_sx075", "nppiResizeSqrPixel_32f_C1R_Ctx", 32, 32, 64, 64, 2.0, 2.0,
                     0.75, 0.75, modes[m], 1, ramp2);
    ramp_sqr<Npp32f>(std::string(names[m]) + "_f15_sx075", "nppiResizeSqrPixel_32f_C1R_Ctx", 32, 32, 48, 48, 1.5, 1.5,
                     0.75, 0.75, modes[m], 1, ramp2);
    ramp_sqr<Npp32f>(std::string(names[m]) + "_fr07_sx075", "nppiResizeSqrPixel_32f_C1R_Ctx", 32, 32, 23, 23, 0.7,
                     0.7, 0.75, 0.75, modes[m], 1, ramp2);
  }
}

TEST_F(NppiResizeProbeDiff, Ramp_Other_Dtypes) {
  const int W = 16, H = 16;
  std::vector<Npp16u> r16(W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) r16[(size_t)y * W + x] = (Npp16u)(y * 1024 + x);
  std::vector<Npp64f> r64(W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) r64[(size_t)y * W + x] = (Npp64f)(y * 1024.0 + x);
  const double shs[] = {0.25, 0.75, -0.25, -0.75, 0.5};
  for (double s : shs) {
    const std::string tag = std::to_string(s);
    ramp_sqr<Npp16u>("16u_nn_s" + tag + "_f1", "nppiResizeSqrPixel_16u_C1R_Ctx", W, H, W, H, 1.0, 1.0, s, s,
                     kInterpNN, 1, r16);
    ramp_sqr<Npp16u>("16u_linear_s" + tag + "_f1", "nppiResizeSqrPixel_16u_C1R_Ctx", W, H, W, H, 1.0, 1.0, s, s,
                     kInterpLinear, 1, r16);
    ramp_sqr<Npp64f>("64f_nn_s" + tag + "_f1", "nppiResizeSqrPixel_64f_C1R_Ctx", W, H, W, H, 1.0, 1.0, s, s,
                     kInterpNN, 1, r64);
    ramp_sqr<Npp64f>("64f_linear_s" + tag + "_f1", "nppiResizeSqrPixel_64f_C1R_Ctx", W, H, W, H, 1.0, 1.0, s, s,
                     kInterpLinear, 1, r64);
  }
}
