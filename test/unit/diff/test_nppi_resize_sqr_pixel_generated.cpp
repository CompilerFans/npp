// Differential tests for package g4 family nppiResizeSqrPixel: every dtype
// (8u/16u/16s/32f/64f) and layout (C1/C3/C4/AC4 packed, P3/P4 planar) plus
// nppiResizeSqrPixel_8u_C1R_Advanced.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiResizeSqrPixelDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build-mpp/src \
//     ./build-mpp/diff_tests --gtest_filter='NppiResizeSqrPixelDiff.*'
//
// The destination is always pre-filled with a poison value that cannot occur
// in the source data, so the recorded golden pins down the documented "pixels
// whose source coordinates fall outside oSrcROI are not written" clip rule:
// a poison entry in the golden means NVIDIA skipped the write.
//
// The *_decode cases fill the source with coordinate-coded values
// (c * KC + y * KY + x, distinct per pixel/channel) so that the golden for
// NPPI_INTER_NN decodes the exact coordinate mapping, rounding and clip rule.

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

// Step aligned up to 16 bytes: safe for every dtype/layout combination.
int align_step(int step, int es) { return ((step + 15) / 16) * 16; }

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp64f>) return ValueKind::F64;
  else return ValueKind::Bytes;
}

// Encode a pixel coordinate into a value that is unique per (channel, x, y)
// and never equal to the poison fill.  8u decode cases therefore keep the
// image within 8x8 C1 (or 8x8 AC4 with the tightest encoding below).
template <typename T> T coord_code(int c, int y, int x) {
  if constexpr (sizeof(T) == 1) {
    return static_cast<T>(c * 64 + y * 8 + x);  // 8x8, 4 channels max
  } else if constexpr (sizeof(T) == 2) {
    return static_cast<T>(c * 4096 + y * 64 + x);  // 64x64, 4 channels max
  } else {
    return static_cast<T>(c * 1048576.0 + y * 1024.0 + x);
  }
}

// Source image with coordinate-coded pixels inside the width and a pad fill
// (distinct from both the codes and the destination poison) in the row tails.
template <typename T>
std::vector<T> coord_image(int step_bytes, int w, int h, int chans, int fill) {
  const int row = step_bytes / (int)sizeof(T);
  std::vector<T> v((size_t)h * row, static_cast<T>(fill));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < chans; ++c) {
        v[(size_t)y * row + (size_t)x * chans + c] = coord_code<T>(c, y, x);
      }
    }
  }
  return v;
}

// Out-of-place packed ResizeSqrPixel case.
//   decode=true fills the source with coordinate codes, otherwise with
//   deterministic pseudo-random data; the destination is always poisoned.
template <typename T, typename F>
void sqr_packed(const char *fname, const std::string &key, int srcW, int srcH, NppiRect srcROI, int dstW, int dstH,
                NppiRect dstROI, double fx, double fy, double sx, double sy, int interp, F f, uint64_t seed,
                T fill, bool decode, int chans) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcW * chans * es, es);
  const int dStep = align_step(dstW * chans * es, es);
  const size_t se = (size_t)srcH * sStep / es;
  const size_t de = (size_t)dstH * dStep / es;
  std::vector<T> hs = decode ? coord_image<T>(sStep, srcW, srcH, chans, (T)0x77) : make_uniform<T>(se, seed + 11);
  std::vector<T> hd(de, fill);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), NppiSize{srcW, srcH}, sStep, srcROI, d.get(), dStep, dstROI, fx, fy, sx, sy, interp,
                   current_ctx());
  Case c("nppiResizeSqrPixel/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// Out-of-place planar ResizeSqrPixel case (P3/P4).  Each plane is a separate
// coordinate-coded (or random) image; every plane is captured in the golden.
template <typename T, typename F>
void sqr_planar(const char *fname, const std::string &key, int planes, int srcW, int srcH, NppiRect srcROI, int dstW,
                int dstH, NppiRect dstROI, double fx, double fy, double sx, double sy, int interp, F f, uint64_t seed,
                T fill, bool decode) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcW * es, es);
  const int dStep = align_step(dstW * es, es);
  const size_t se = (size_t)srcH * sStep / es;
  const size_t de = (size_t)dstH * dStep / es;
  const T *srcPtrs[4] = {nullptr, nullptr, nullptr, nullptr};
  T *dstPtrs[4] = {nullptr, nullptr, nullptr, nullptr};
  std::vector<std::vector<T>> hplanes;
  Case c("nppiResizeSqrPixel/" + key, fname);
  for (int p = 0; p < planes; ++p) {
    hplanes.push_back(decode ? coord_image<T>(sStep, srcW, srcH, 1, (T)0x77) : make_uniform<T>(se, seed + 11 + p));
    DeviceMemory<T> &s = arena<T>(se);
    DeviceMemory<T> &d = arena<T>(de);
    std::vector<T> hd(de, fill);
    s.copyFromHost(hplanes.back());
    d.copyFromHost(hd);
    srcPtrs[p] = s.get();
    dstPtrs[p] = d.get();
    c.input(hplanes.back().data(), hplanes.back().size() * sizeof(T));
  }
  clear_cuda_error();
  NppStatus st = f(srcPtrs, NppiSize{srcW, srcH}, sStep, srcROI, dstPtrs, dStep, dstROI, fx, fy, sx, sy, interp,
                   current_ctx());
  c.status(st);
  for (int p = 0; p < planes; ++p) {
    c.device_output<T>(dstPtrs[p], de, out_kind<T>(), 1e-5, 1e-6);
  }
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

}  // namespace

class NppiResizeSqrPixelDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== NN coordinate-mapping decode (8u C1) ================================
TEST_F(NppiResizeSqrPixelDiff, NN_Decode_8u_C1) {
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_up2_decode", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 100,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_down2_decode", 16, 16, NppiRect{0, 0, 16, 16}, 8, 8,
                    NppiRect{0, 0, 8, 8}, 0.5, 0.5, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 101,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_up3_decode", 8, 8, NppiRect{0, 0, 8, 8}, 24, 24,
                    NppiRect{0, 0, 24, 24}, 3.0, 3.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 102,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_frac_decode", 8, 8, NppiRect{0, 0, 8, 8}, 12, 12,
                    NppiRect{0, 0, 12, 12}, 1.5, 1.5, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 103,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_shift_decode", 8, 8, NppiRect{0, 0, 8, 8}, 8, 8,
                    NppiRect{0, 0, 8, 8}, 1.0, 1.0, 1.0, 1.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 104,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_srcroi_decode", 10, 10, NppiRect{2, 3, 5, 4}, 10, 8,
                    NppiRect{0, 0, 10, 8}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 105,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_dstroi_decode", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{3, 2, 6, 5}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 106,
                    (Npp8u)0xEE, true, 1);
  // Extreme shift: every destination pixel maps outside the source ROI, so a
  // conforming implementation returns success and writes nothing.
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nn_allclip", 8, 8, NppiRect{0, 0, 8, 8}, 8, 8,
                    NppiRect{0, 0, 8, 8}, 1.0, 1.0, 40.0, 40.0, kInterpNN, &nppiResizeSqrPixel_8u_C1R_Ctx, 107,
                    (Npp8u)0xEE, true, 1);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/linear_up2_decode", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_8u_C1R_Ctx, 108,
                    (Npp8u)0xEE, true, 1);
}

// ===== interpolation-mode matrix ============================================
TEST_F(NppiResizeSqrPixelDiff, Interp_32f_C1) {
  const int W = 12, H = 9;
  const int modes[] = {kInterpNN, kInterpLinear, kInterpCubic, kInterpBSpline, kInterpCatmullRom, kInterpB05C03,
                       kInterpLanczos};
  const char *names[] = {"nn", "linear", "cubic", "bspline", "catmullrom", "b05c03", "lanczos"};
  for (int i = 0; i < 7; ++i) {
    sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C1R_Ctx", std::string("32f_C1/") + names[i] + "_up2", W, H,
                       NppiRect{0, 0, W, H}, W * 2, H * 2, NppiRect{0, 0, W * 2, H * 2}, 2.0, 2.0, 0.0, 0.0, modes[i],
                       &nppiResizeSqrPixel_32f_C1R_Ctx, 200 + i, (Npp32f)-1e30f, false, 1);
    sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C1R_Ctx", std::string("32f_C1/") + names[i] + "_down2", W * 2, H * 2,
                       NppiRect{0, 0, W * 2, H * 2}, W, H, NppiRect{0, 0, W, H}, 0.5, 0.5, 0.0, 0.0, modes[i],
                       &nppiResizeSqrPixel_32f_C1R_Ctx, 210 + i, (Npp32f)-1e30f, false, 1);
  }
  // SUPER only accepts downscaling.
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C1R_Ctx", "32f_C1/super_down2", W * 2, H * 2,
                     NppiRect{0, 0, W * 2, H * 2}, W, H, NppiRect{0, 0, W, H}, 0.5, 0.5, 0.0, 0.0, kInterpSuper,
                     &nppiResizeSqrPixel_32f_C1R_Ctx, 220, (Npp32f)-1e30f, false, 1);
}

TEST_F(NppiResizeSqrPixelDiff, Interp_8u_C1) {
  const int W = 11, H = 7;
  const int modes[] = {kInterpNN, kInterpLinear, kInterpCubic, kInterpBSpline, kInterpCatmullRom, kInterpB05C03,
                       kInterpLanczos};
  const char *names[] = {"nn", "linear", "cubic", "bspline", "catmullrom", "b05c03", "lanczos"};
  for (int i = 0; i < 7; ++i) {
    sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", std::string("8u_C1/") + names[i] + "_up2", W, H,
                      NppiRect{0, 0, W, H}, W * 2, H * 2, NppiRect{0, 0, W * 2, H * 2}, 2.0, 2.0, 0.0, 0.0, modes[i],
                      &nppiResizeSqrPixel_8u_C1R_Ctx, 300 + i, (Npp8u)0xEE, false, 1);
    sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", std::string("8u_C1/") + names[i] + "_down2", W * 2, H * 2,
                      NppiRect{0, 0, W * 2, H * 2}, W, H, NppiRect{0, 0, W, H}, 0.5, 0.5, 0.0, 0.0, modes[i],
                      &nppiResizeSqrPixel_8u_C1R_Ctx, 310 + i, (Npp8u)0xEE, false, 1);
  }
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/super_down2", W * 2, H * 2, NppiRect{0, 0, W * 2, H * 2},
                    W, H, NppiRect{0, 0, W, H}, 0.5, 0.5, 0.0, 0.0, kInterpSuper, &nppiResizeSqrPixel_8u_C1R_Ctx, 320,
                    (Npp8u)0xEE, false, 1);
}

TEST_F(NppiResizeSqrPixelDiff, Interp_OtherDtypes_C1) {
  // 16u
  sqr_packed<Npp16u>("nppiResizeSqrPixel_16u_C1R_Ctx", "16u_C1/nn_up2", 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_16u_C1R_Ctx, 330,
                     (Npp16u)0xEEEE, true, 1);
  sqr_packed<Npp16u>("nppiResizeSqrPixel_16u_C1R_Ctx", "16u_C1/linear_down2", 18, 12, NppiRect{0, 0, 18, 12}, 9, 6,
                     NppiRect{0, 0, 9, 6}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_16u_C1R_Ctx, 331,
                     (Npp16u)0xEEEE, false, 1);
  sqr_packed<Npp16u>("nppiResizeSqrPixel_16u_C1R_Ctx", "16u_C1/cubic_up2", 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpCubic, &nppiResizeSqrPixel_16u_C1R_Ctx, 332,
                     (Npp16u)0xEEEE, false, 1);
  // 16s
  sqr_packed<Npp16s>("nppiResizeSqrPixel_16s_C1R_Ctx", "16s_C1/nn_up2", 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_16s_C1R_Ctx, 333,
                     (Npp16s)-21846, true, 1);
  sqr_packed<Npp16s>("nppiResizeSqrPixel_16s_C1R_Ctx", "16s_C1/linear_down2", 18, 12, NppiRect{0, 0, 18, 12}, 9, 6,
                     NppiRect{0, 0, 9, 6}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_16s_C1R_Ctx, 334,
                     (Npp16s)-21846, false, 1);
  sqr_packed<Npp16s>("nppiResizeSqrPixel_16s_C1R_Ctx", "16s_C1/cubic_up2", 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpCubic, &nppiResizeSqrPixel_16s_C1R_Ctx, 335,
                     (Npp16s)-21846, false, 1);
  // 64f
  sqr_packed<Npp64f>("nppiResizeSqrPixel_64f_C1R_Ctx", "64f_C1/nn_up2", 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_64f_C1R_Ctx, 336,
                     -1e300, true, 1);
  sqr_packed<Npp64f>("nppiResizeSqrPixel_64f_C1R_Ctx", "64f_C1/linear_down2", 18, 12, NppiRect{0, 0, 18, 12}, 9, 6,
                     NppiRect{0, 0, 9, 6}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_64f_C1R_Ctx, 337,
                     -1e300, false, 1);
  sqr_packed<Npp64f>("nppiResizeSqrPixel_64f_C1R_Ctx", "64f_C1/lanczos_up2", 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpLanczos, &nppiResizeSqrPixel_64f_C1R_Ctx, 338,
                     -1e300, false, 1);
}

// ===== channel/layout coverage ==============================================
TEST_F(NppiResizeSqrPixelDiff, Channels_32f) {
  // C3
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C3R_Ctx", "32f_C3/nn_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_32f_C3R_Ctx, 400,
                     (Npp32f)-1e30f, false, 3);
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C3R_Ctx", "32f_C3/linear_down2", 14, 10, NppiRect{0, 0, 14, 10}, 7, 5,
                     NppiRect{0, 0, 7, 5}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_32f_C3R_Ctx, 401,
                     (Npp32f)-1e30f, false, 3);
  // C4
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C4R_Ctx", "32f_C4/nn_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_32f_C4R_Ctx, 402,
                     (Npp32f)-1e30f, false, 4);
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_C4R_Ctx", "32f_C4/linear_down2", 14, 10, NppiRect{0, 0, 14, 10}, 7, 5,
                     NppiRect{0, 0, 7, 5}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_32f_C4R_Ctx, 403,
                     (Npp32f)-1e30f, false, 4);
  // AC4: alpha behavior pinned by the poison fill in all four channels.
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_AC4R_Ctx", "32f_AC4/nn_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_32f_AC4R_Ctx, 404,
                     (Npp32f)-1e30f, false, 4);
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_AC4R_Ctx", "32f_AC4/linear_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_32f_AC4R_Ctx, 405,
                     (Npp32f)-1e30f, false, 4);
  sqr_packed<Npp32f>("nppiResizeSqrPixel_32f_AC4R_Ctx", "32f_AC4/cubic_down2", 14, 10, NppiRect{0, 0, 14, 10}, 7, 5,
                     NppiRect{0, 0, 7, 5}, 0.5, 0.5, 0.0, 0.0, kInterpCubic, &nppiResizeSqrPixel_32f_AC4R_Ctx, 406,
                     (Npp32f)-1e30f, false, 4);
  // planar P3
  sqr_planar<Npp32f>("nppiResizeSqrPixel_32f_P3R_Ctx", "32f_P3/nn_up2", 3, 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_32f_P3R_Ctx, 407,
                     (Npp32f)-1e30f, false);
  sqr_planar<Npp32f>("nppiResizeSqrPixel_32f_P3R_Ctx", "32f_P3/linear_down2", 3, 14, 10, NppiRect{0, 0, 14, 10}, 7, 5,
                     NppiRect{0, 0, 7, 5}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_32f_P3R_Ctx, 408,
                     (Npp32f)-1e30f, false);
  // planar P4
  sqr_planar<Npp32f>("nppiResizeSqrPixel_32f_P4R_Ctx", "32f_P4/nn_up2", 4, 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_32f_P4R_Ctx, 409,
                     (Npp32f)-1e30f, false);
  sqr_planar<Npp32f>("nppiResizeSqrPixel_32f_P4R_Ctx", "32f_P4/linear_down2", 4, 14, 10, NppiRect{0, 0, 14, 10}, 7, 5,
                     NppiRect{0, 0, 7, 5}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_32f_P4R_Ctx, 410,
                     (Npp32f)-1e30f, false);
}

TEST_F(NppiResizeSqrPixelDiff, Channels_8u_AC4_And_Existing) {
  // AC4 alpha rule for the pre-existing 8u path.
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_AC4R_Ctx", "8u_AC4/nn_up2", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_AC4R_Ctx, 420,
                    (Npp8u)0xEE, true, 4);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_AC4R_Ctx", "8u_AC4/linear_up2", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_8u_AC4R_Ctx, 421,
                    (Npp8u)0xEE, false, 4);
  // planar
  sqr_planar<Npp8u>("nppiResizeSqrPixel_8u_P3R_Ctx", "8u_P3/nn_up2", 3, 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_P3R_Ctx, 422,
                    (Npp8u)0xEE, true);
  sqr_planar<Npp8u>("nppiResizeSqrPixel_8u_P4R_Ctx", "8u_P4/linear_down2", 4, 16, 16, NppiRect{0, 0, 16, 16}, 8, 8,
                    NppiRect{0, 0, 8, 8}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_8u_P4R_Ctx, 423,
                    (Npp8u)0xEE, false);
  // the pre-existing 8u C3/C4 implementations, re-measured against NVIDIA
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C3R_Ctx", "8u_C3/nn_up2", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_8u_C3R_Ctx, 424,
                    (Npp8u)0xEE, true, 3);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C4R_Ctx", "8u_C4/linear_up2", 8, 8, NppiRect{0, 0, 8, 8}, 16, 16,
                    NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_8u_C4R_Ctx, 425,
                    (Npp8u)0xEE, false, 4);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C4R_Ctx", "8u_C4/identity", 5, 4, NppiRect{0, 0, 5, 4}, 5, 4,
                    NppiRect{0, 0, 5, 4}, 1.0, 1.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_8u_C4R_Ctx, 426,
                    (Npp8u)0xEE, false, 4);
}

TEST_F(NppiResizeSqrPixelDiff, Channels_Misc) {
  // 16s P3, 64f P4, 16u C3
  sqr_planar<Npp16s>("nppiResizeSqrPixel_16s_P3R_Ctx", "16s_P3/nn_up2", 3, 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_16s_P3R_Ctx, 430,
                     (Npp16s)-21846, true);
  sqr_planar<Npp16s>("nppiResizeSqrPixel_16s_P4R_Ctx", "16s_P4/linear_down2", 4, 18, 12, NppiRect{0, 0, 18, 12}, 9, 6,
                     NppiRect{0, 0, 9, 6}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_16s_P4R_Ctx, 431,
                     (Npp16s)-21846, false);
  sqr_planar<Npp16u>("nppiResizeSqrPixel_16u_P3R_Ctx", "16u_P3/nn_up2", 3, 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_16u_P3R_Ctx, 432,
                     (Npp16u)0xEEEE, true);
  sqr_planar<Npp16u>("nppiResizeSqrPixel_16u_P4R_Ctx", "16u_P4/linear_down2", 4, 18, 12, NppiRect{0, 0, 18, 12}, 9, 6,
                     NppiRect{0, 0, 9, 6}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_16u_P4R_Ctx, 433,
                     (Npp16u)0xEEEE, false);
  sqr_planar<Npp64f>("nppiResizeSqrPixel_64f_P3R_Ctx", "64f_P3/nn_up2", 3, 9, 6, NppiRect{0, 0, 9, 6}, 18, 12,
                     NppiRect{0, 0, 18, 12}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_64f_P3R_Ctx, 434,
                     -1e300, true);
  sqr_planar<Npp64f>("nppiResizeSqrPixel_64f_P4R_Ctx", "64f_P4/linear_down2", 4, 18, 12, NppiRect{0, 0, 18, 12}, 9, 6,
                     NppiRect{0, 0, 9, 6}, 0.5, 0.5, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_64f_P4R_Ctx, 435,
                     -1e300, false);
  sqr_packed<Npp16u>("nppiResizeSqrPixel_16u_C3R_Ctx", "16u_C3/nn_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_16u_C3R_Ctx, 436,
                     (Npp16u)0xEEEE, false, 3);
  sqr_packed<Npp16s>("nppiResizeSqrPixel_16s_C4R_Ctx", "16s_C4/linear_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_16s_C4R_Ctx, 437,
                     (Npp16s)-21846, false, 4);
  sqr_packed<Npp64f>("nppiResizeSqrPixel_64f_C4R_Ctx", "64f_C4/nn_down2", 14, 10, NppiRect{0, 0, 14, 10}, 7, 5,
                     NppiRect{0, 0, 7, 5}, 0.5, 0.5, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_64f_C4R_Ctx, 438,
                     -1e300, false, 4);
  sqr_packed<Npp16u>("nppiResizeSqrPixel_16u_AC4R_Ctx", "16u_AC4/nn_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN, &nppiResizeSqrPixel_16u_AC4R_Ctx, 439,
                     (Npp16u)0xEEEE, false, 4);
  sqr_packed<Npp16s>("nppiResizeSqrPixel_16s_AC4R_Ctx", "16s_AC4/linear_up2", 7, 5, NppiRect{0, 0, 7, 5}, 14, 10,
                     NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpLinear, &nppiResizeSqrPixel_16s_AC4R_Ctx, 440,
                     (Npp16s)-21846, false, 4);
  sqr_packed<Npp8u>("nppiResizeSqrPixel_8u_C1R_Ctx", "8u_C1/nonzero_shift", 8, 8, NppiRect{0, 0, 8, 8}, 8, 8,
                    NppiRect{0, 0, 8, 8}, 1.5, 1.5, 2.0, -1.0, kInterpLinear, &nppiResizeSqrPixel_8u_C1R_Ctx, 441,
                    (Npp8u)0xEE, false, 1);
}

// ===== Advanced variant (8u C1R, pBuffer, eInterpolationMode) ==============
TEST_F(NppiResizeSqrPixelDiff, Advanced_8u_C1) {
  const char *fname = "nppiResizeSqrPixel_8u_C1R_Advanced_Ctx";
  struct Cfg {
    const char *key;
    int srcW, srcH, dstW, dstH;
    double fx, fy;
    int mode;
    uint64_t seed;
    bool expect_run;
  };
  const Cfg cfgs[] = {
      {"advanced/lanczos3_up2", 8, 8, 16, 16, 2.0, 2.0, NPPI_INTER_LANCZOS3_ADVANCED, 500, true},
      {"advanced/lanczos3_down2", 16, 16, 8, 8, 0.5, 0.5, NPPI_INTER_LANCZOS3_ADVANCED, 501, true},
      {"advanced/mode_nn", 8, 8, 16, 16, 2.0, 2.0, kInterpNN, 502, false},
      {"advanced/mode_linear", 8, 8, 16, 16, 2.0, 2.0, kInterpLinear, 503, false},
      {"advanced/zero_factor", 8, 8, 16, 16, 0.0, 2.0, NPPI_INTER_LANCZOS3_ADVANCED, 504, false},
  };
  for (const auto &cfg : cfgs) {
    const int es = 1;
    const int sStep = align_step(cfg.srcW * es, es);
    const int dStep = align_step(cfg.dstW * es, es);
    const size_t se = (size_t)cfg.srcH * sStep / es;
    const size_t de = (size_t)cfg.dstH * dStep / es;
    std::vector<Npp8u> hs = make_uniform<Npp8u>(se, cfg.seed + 11);
    std::vector<Npp8u> hd(de, 0xEE);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
    s.copyFromHost(hs);
    d.copyFromHost(hd);
    // Generous buffer: 4 * dstW * srcH is what the host helper reports
    // (already measured); pad heavily so a differing NVIDIA requirement
    // cannot turn a validation probe into an out-of-bounds write.
    const size_t bufSize = 4u * (size_t)cfg.dstW * cfg.srcH + 65536u;
    DeviceMemory<Npp8u> &buf = arena<Npp8u>(bufSize);
    buf.copyFromHost(std::vector<Npp8u>(bufSize, 0));
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(
        s.get(), NppiSize{cfg.srcW, cfg.srcH}, sStep, NppiRect{0, 0, cfg.srcW, cfg.srcH}, d.get(), dStep,
        NppiRect{0, 0, cfg.dstW, cfg.dstH}, cfg.fx, cfg.fy, buf.get(), cfg.mode, current_ctx());
    Case c(std::string("nppiResizeSqrPixel/") + cfg.key, fname);
    c.input(hs.data(), hs.size() * sizeof(Npp8u));
    c.status(st);
    c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 1e-5, 1e-6);
    c.finish();
  }
  // null pBuffer with otherwise valid arguments
  {
    const int sStep = align_step(8, 1), dStep = align_step(16, 1);
    const size_t se = 8 * sStep, de = 16 * dStep;
    DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
    s.copyFromHost(make_uniform<Npp8u>(se, 511));
    d.copyFromHost(std::vector<Npp8u>(de, 0xEE));
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(
        s.get(), NppiSize{8, 8}, sStep, NppiRect{0, 0, 8, 8}, d.get(), dStep, NppiRect{0, 0, 16, 16}, 2.0, 2.0,
        nullptr, NPPI_INTER_LANCZOS3_ADVANCED, current_ctx());
    Case c("nppiResizeSqrPixel/advanced/null_buffer", fname);
    c.status(st);
    c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 1e-5, 1e-6);
    c.finish();
  }
}

// ===== error / contract paths ===============================================
// Every case fixes all-but-one parameter to a known-good configuration so the
// recorded status pins down the validation rule; ordering probes combine two
// violations.  Kept in a separate test so a crash in one of NVIDIA's probes
// cannot lose the functional goldens above.
TEST_F(NppiResizeSqrPixelDiff, Contract_8u_C1) {
  const char *fname = "nppiResizeSqrPixel_8u_C1R_Ctx";
  struct Cfg {
    const char *key;
    NppiSize srcSize;
    int sStep;
    NppiRect srcROI;
    int dStep;
    NppiRect dstROI;
    double fx, fy, sx, sy;
    int interp;
    bool nullSrc, nullDst;
  };
  const NppiSize good{8, 8};
  const int gs = align_step(8, 1);
  const NppiRect gsr{0, 0, 8, 8};
  const NppiRect gdr{0, 0, 16, 16};
  const Cfg cfgs[] = {
      {"contract/null_src", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN, true, false},
      {"contract/null_dst", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN, false, true},
      {"contract/src_size_1x8", NppiSize{1, 8}, gs, NppiRect{0, 0, 1, 8}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0,
       kInterpNN, false, false},
      {"contract/src_size_8x1", NppiSize{8, 1}, gs, NppiRect{0, 0, 8, 1}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0,
       kInterpNN, false, false},
      {"contract/src_size_zero_w", NppiSize{0, 8}, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN, false,
       false},
      {"contract/src_size_neg", NppiSize{-4, 8}, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN, false,
       false},
      {"contract/src_roi_zero_w", good, gs, NppiRect{0, 0, 0, 8}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN,
       false, false},
      {"contract/src_roi_zero_h", good, gs, NppiRect{0, 0, 8, 0}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN,
       false, false},
      {"contract/src_roi_neg_w", good, gs, NppiRect{0, 0, -2, 8}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN,
       false, false},
      {"contract/src_roi_no_intersect", good, gs, NppiRect{20, 0, 4, 4}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0,
       kInterpNN, false, false},
      {"contract/src_roi_partial_out", good, gs, NppiRect{4, 0, 8, 8}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0,
       kInterpNN, false, false},
      {"contract/dst_roi_zero_w", good, gs, gsr, align_step(16, 1), NppiRect{0, 0, 0, 16}, 2.0, 2.0, 0, 0, kInterpNN,
       false, false},
      {"contract/dst_roi_zero_h", good, gs, gsr, align_step(16, 1), NppiRect{0, 0, 16, 0}, 2.0, 2.0, 0, 0, kInterpNN,
       false, false},
      {"contract/dst_roi_neg_w", good, gs, gsr, align_step(16, 1), NppiRect{0, 0, -1, 16}, 2.0, 2.0, 0, 0, kInterpNN,
       false, false},
      {"contract/factor_zero", good, gs, gsr, align_step(16, 1), gdr, 0.0, 2.0, 0, 0, kInterpNN, false, false},
      {"contract/factor_neg", good, gs, gsr, align_step(16, 1), gdr, -1.0, 2.0, 0, 0, kInterpNN, false, false},
      {"contract/interp_undef", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, 0, false, false},
      {"contract/interp_3", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, 3, false, false},
      {"contract/interp_32", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, 32, false, false},
      {"contract/interp_15", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, 15, false, false},
      {"contract/interp_neg", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, -1, false, false},
      {"contract/super_upscale", good, gs, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpSuper, false, false},
      {"contract/step_small_src", good, 2, gsr, align_step(16, 1), gdr, 2.0, 2.0, 0, 0, kInterpNN, false, false},
      {"contract/step_small_dst", good, gs, gsr, 4, gdr, 2.0, 2.0, 0, 0, kInterpNN, false, false},
      // ordering probes (two violations at once)
      {"contract/order_null_and_factor", good, gs, gsr, align_step(16, 1), gdr, 0.0, 2.0, 0, 0, kInterpNN, true,
       false},
      {"contract/order_factor_and_interp", good, gs, gsr, align_step(16, 1), gdr, 0.0, 2.0, 0, 0, 3, false, false},
      {"contract/order_size1_and_factor", NppiSize{1, 8}, gs, NppiRect{0, 0, 1, 8}, align_step(16, 1), gdr, 0.0, 2.0,
       0, 0, kInterpNN, false, false},
      {"contract/order_nointersect_and_factor", good, gs, NppiRect{20, 0, 4, 4}, align_step(16, 1), gdr, 0.0, 2.0, 0,
       0, kInterpNN, false, false},
      {"contract/order_dstzero_and_factor", good, gs, gsr, align_step(16, 1), NppiRect{0, 0, 0, 16}, 0.0, 2.0, 0, 0,
       kInterpNN, false, false},
      {"contract/order_nointersect_and_dstzero", good, gs, NppiRect{20, 0, 4, 4}, align_step(16, 1),
       NppiRect{0, 0, 0, 16}, 2.0, 2.0, 0, 0, kInterpNN, false, false},
      {"contract/order_nointersect_and_interp", good, gs, NppiRect{20, 0, 4, 4}, align_step(16, 1), gdr, 2.0, 2.0, 0,
       0, 3, false, false},
      {"contract/order_step_and_factor", good, 2, gsr, align_step(16, 1), gdr, 0.0, 2.0, 0, 0, kInterpNN, false,
       false},
      {"contract/order_step_and_nointersect", good, 2, NppiRect{20, 0, 4, 4}, align_step(16, 1), gdr, 2.0, 2.0, 0, 0,
       kInterpNN, false, false},
      {"contract/order_interp_and_factor", good, gs, gsr, align_step(16, 1), gdr, 0.0, 2.0, 0, 0, 35, false,
       false},
  };
  for (const auto &cfg : cfgs) {
    const size_t se = (size_t)cfg.srcSize.height * cfg.sStep;
    const size_t de = (size_t)16 * cfg.dStep;
    DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
    s.copyFromHost(make_uniform<Npp8u>(se, 700));
    d.copyFromHost(std::vector<Npp8u>(de, 0xEE));
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_8u_C1R_Ctx(cfg.nullSrc ? nullptr : s.get(), cfg.srcSize, cfg.sStep, cfg.srcROI,
                                                 cfg.nullDst ? nullptr : d.get(), cfg.dStep, cfg.dstROI, cfg.fx,
                                                 cfg.fy, cfg.sx, cfg.sy, cfg.interp, current_ctx());
    Case c(std::string("nppiResizeSqrPixel/") + cfg.key, fname);
    c.status(st);
    c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 1e-5, 1e-6);
    c.finish();
  }
}

TEST_F(NppiResizeSqrPixelDiff, Contract_Planar_32f_P3) {
  // Planar null-pointer and zero-size probes.
  const int es = 4;
  const int sStep = align_step(8 * es, es);
  const int dStep = align_step(16 * es, es);
  const size_t se = 8 * sStep / es;
  const size_t de = 16 * dStep / es;
  DeviceMemory<Npp32f> &s0 = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &s1 = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &s2 = arena<Npp32f>(se);
  DeviceMemory<Npp32f> &d0 = arena<Npp32f>(de);
  DeviceMemory<Npp32f> &d1 = arena<Npp32f>(de);
  DeviceMemory<Npp32f> &d2 = arena<Npp32f>(de);
  for (auto *m : {&s0, &s1, &s2}) m->copyFromHost(make_uniform<Npp32f>(se, 800));
  for (auto *m : {&d0, &d1, &d2}) m->copyFromHost(std::vector<Npp32f>(de, -1e30f));
  const Npp32f *srcs[3] = {s0.get(), s1.get(), s2.get()};
  Npp32f *dsts[3] = {d0.get(), d1.get(), d2.get()};
  {
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_32f_P3R_Ctx(nullptr, NppiSize{8, 8}, sStep, NppiRect{0, 0, 8, 8}, dsts, dStep,
                                                  NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN,
                                                  current_ctx());
    Case c("nppiResizeSqrPixel/contract/planar_null_src", "nppiResizeSqrPixel_32f_P3R_Ctx");
    c.status(st);
    c.device_output<Npp32f>(d0.get(), de, ValueKind::F32, 1e-5, 1e-6);
    c.finish();
  }
  {
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_32f_P3R_Ctx(srcs, NppiSize{8, 8}, sStep, NppiRect{0, 0, 8, 8}, nullptr, dStep,
                                                  NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN,
                                                  current_ctx());
    Case c("nppiResizeSqrPixel/contract/planar_null_dst", "nppiResizeSqrPixel_32f_P3R_Ctx");
    c.status(st);
    c.device_output<Npp32f>(d0.get(), de, ValueKind::F32, 1e-5, 1e-6);
    c.finish();
  }
}

#if NPP_DIFF_HAS_NONCTX
// Representative re-runs through the plain (non-Ctx) entry points under the
// SAME case ids as the Ctx cases above: in the record build only the _Ctx
// symbols exist, so verify compares the forwarded result against the Ctx
// golden, which pins down that the twin forwards with the ambient context.
TEST_F(NppiResizeSqrPixelDiff, NonCtx_Repr) {
  // Reproduces sqr_packed 8u_C1/nn_up2_decode (seed 100).
  {
    const int sStep = align_step(8, 1), dStep = align_step(16, 1);
    const size_t se = 8 * sStep, de = 16 * dStep;
    std::vector<Npp8u> hs = coord_image<Npp8u>(sStep, 8, 8, 1, (Npp8u)0x77);
    std::vector<Npp8u> hd(de, 0xEE);
    DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
    s.copyFromHost(hs);
    d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_8u_C1R(s.get(), NppiSize{8, 8}, sStep, NppiRect{0, 0, 8, 8}, d.get(), dStep,
                                             NppiRect{0, 0, 16, 16}, 2.0, 2.0, 0.0, 0.0, kInterpNN);
    Case c("nppiResizeSqrPixel/8u_C1/nn_up2_decode", "nppiResizeSqrPixel_8u_C1R");
    c.status(st);
    c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 1e-5, 1e-6);
    c.finish();
  }
  // Reproduces sqr_packed 32f_AC4/linear_up2 (seed 405).
  {
    const int es = 4;
    const int sStep = align_step(7 * 4 * es, es), dStep = align_step(14 * 4 * es, es);
    const size_t se = 5 * sStep / es, de = 10 * dStep / es;
    std::vector<Npp32f> hs = make_uniform<Npp32f>(se, 405 + 11);
    std::vector<Npp32f> hd(de, -1e30f);
    DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
    DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
    s.copyFromHost(hs);
    d.copyFromHost(hd);
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_32f_AC4R(s.get(), NppiSize{7, 5}, sStep, NppiRect{0, 0, 7, 5}, d.get(), dStep,
                                               NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpLinear);
    Case c("nppiResizeSqrPixel/32f_AC4/linear_up2", "nppiResizeSqrPixel_32f_AC4R");
    c.status(st);
    c.device_output<Npp32f>(d.get(), de, ValueKind::F32, 1e-5, 1e-6);
    c.finish();
  }
  // Reproduces sqr_planar 32f_P3/nn_up2 (seed 407).
  {
    const int es = 4;
    const int sStep = align_step(7 * es, es), dStep = align_step(14 * es, es);
    const size_t se = 5 * sStep / es, de = 10 * dStep / es;
    const Npp32f *srcs[3] = {nullptr, nullptr, nullptr};
    Npp32f *dsts[3] = {nullptr, nullptr, nullptr};
    Case c("nppiResizeSqrPixel/32f_P3/nn_up2", "nppiResizeSqrPixel_32f_P3R");
    for (int p = 0; p < 3; ++p) {
      std::vector<Npp32f> hs = make_uniform<Npp32f>(se, 407 + 11 + p);
      DeviceMemory<Npp32f> &s = arena<Npp32f>(se);
      DeviceMemory<Npp32f> &d = arena<Npp32f>(de);
      std::vector<Npp32f> hd(de, -1e30f);
      s.copyFromHost(hs);
      d.copyFromHost(hd);
      srcs[p] = s.get();
      dsts[p] = d.get();
      c.input(hs.data(), hs.size() * sizeof(Npp32f));
    }
    clear_cuda_error();
    NppStatus st = nppiResizeSqrPixel_32f_P3R(srcs, NppiSize{7, 5}, sStep, NppiRect{0, 0, 7, 5}, dsts, dStep,
                                              NppiRect{0, 0, 14, 10}, 2.0, 2.0, 0.0, 0.0, kInterpNN);
    c.status(st);
    for (int p = 0; p < 3; ++p) {
      c.device_output<Npp32f>(dsts[p], de, ValueKind::F32, 1e-5, 1e-6);
    }
    c.finish();
  }
}
#endif
