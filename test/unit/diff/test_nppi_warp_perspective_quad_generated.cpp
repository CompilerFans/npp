// Differential tests for package g5: nppiWarpPerspectiveQuad
// (8u/16u/32f/32s x C1R/C3R/C4R/AC4R/P3R/P4R, Ctx and non-Ctx).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiWarpPerspectiveQuadDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiWarpPerspectiveQuadDiff.*'
//
// The family maps a quadrilateral in source image space to a quadrilateral in
// destination image space; the implementation solves the 3x3 homography from
// the two quads.  The recorded goldens pin down
//   * the vertex order convention of the two quads (all probe quads are
//     asymmetric),
//   * which direction the solved matrix is applied in (the scale2 case is an
//     exact 2x upscale under src->dst mapping, but mostly out-of-bounds
//     sampling under the inverse),
//   * the status precedence between degenerate quads, image contract errors
//     and invalid interpolation,
//   * the AC4 alpha rule (destinations are pre-filled with 0xA5).
// Null quad arrays are never passed: they are host arrays that NVIDIA reads
// without validation and the reference run would crash.

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
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else return ValueKind::Bytes;
}

template <typename T>
using QuadPackedFn = NppStatus (*)(const T *, NppiSize, int, NppiRect, const double (*)[2], T *, int, NppiRect,
                                   const double (*)[2], int, NppStreamContext);
template <typename T>
using QuadPlanarFn = NppStatus (*)(const T **, NppiSize, int, NppiRect, const double (*)[2], T **, int, NppiRect,
                                   const double (*)[2], int, NppStreamContext);

// Position-encoded packed source: channel 0 = x, channel 1 = y, further
// channels = 7.
template <typename T> std::vector<T> make_packed_positional(size_t count, int stride, int row_elems) {
  std::vector<T> v(count);
  for (size_t i = 0; i < count; ++i) {
    if (stride == 1) {
      v[i] = (T)(i % (size_t)row_elems);
      continue;
    }
    switch (i % (size_t)stride) {
    case 0: v[i] = (T)((i % (size_t)row_elems) / (size_t)stride); break;
    case 1: v[i] = (T)(i / (size_t)row_elems); break;
    default: v[i] = (T)7; break;
    }
  }
  return v;
}

// Packed quad warp case.
template <typename T, int Stride>
void quad_packed_case(const char *fname, const std::string &key, NppiSize srcSize, NppiRect srcROI, NppiSize dstSize,
                      NppiRect dstROI, const double srcQuad[4][2], const double dstQuad[4][2], int interp,
                      bool positional, QuadPackedFn<T> f, uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcSize.width * Stride * es + 8, es);
  const int dStep = align_step(dstSize.width * Stride * es + 8, es);
  const size_t se = (size_t)srcSize.height * sStep / es;
  const size_t de = (size_t)dstSize.height * dStep / es;
  std::vector<T> hs = positional ? make_packed_positional<T>(se, Stride, sStep / es)
                                 : make_uniform<T>(se, seed + 11);
  std::vector<T> hd(de, (T)0xA5);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), srcSize, sStep, srcROI, srcQuad, d.get(), dStep, dstROI, dstQuad, interp, current_ctx());
  Case c("nppiWarpPerspectiveQuad/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// Planar quad warp case.
template <typename T, int Planes>
void quad_planar_case(const char *fname, const std::string &key, NppiSize srcSize, NppiRect srcROI, NppiSize dstSize,
                      NppiRect dstROI, const double srcQuad[4][2], const double dstQuad[4][2], int interp,
                      bool positional, QuadPlanarFn<T> f, uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcSize.width * es + 8, es);
  const int dStep = align_step(dstSize.width * es + 8, es);
  const size_t se = (size_t)srcSize.height * sStep / es;
  const size_t de = (size_t)dstSize.height * dStep / es;
  std::vector<const T *> srcPtrs(Planes);
  std::vector<T *> dstPtrs(Planes);
  std::vector<std::vector<T>> hsrc(Planes);
  for (int p = 0; p < Planes; ++p) {
    hsrc[p] = positional ? make_packed_positional<T>(se, 1, sStep / es) : make_uniform<T>(se, seed + 11 + p * 7);
    DeviceMemory<T> &s = arena<T>(se);
    DeviceMemory<T> &d = arena<T>(de);
    s.copyFromHost(hsrc[p]);
    d.copyFromHost(std::vector<T>(de, (T)0xA5));
    srcPtrs[p] = s.get();
    dstPtrs[p] = d.get();
  }
  clear_cuda_error();
  NppStatus st = f(srcPtrs.data(), srcSize, sStep, srcROI, srcQuad, dstPtrs.data(), dStep, dstROI, dstQuad, interp,
                   current_ctx());
  Case c("nppiWarpPerspectiveQuad/" + key, fname);
  for (int p = 0; p < Planes; ++p) {
    c.input(hsrc[p].data(), hsrc[p].size() * sizeof(T));
  }
  c.status(st);
  for (int p = 0; p < Planes; ++p) {
    c.device_output<T>(dstPtrs[p], de, out_kind<T>(), 1e-5, 1e-6);
  }
  c.finish();
}

// Quad geometries.  Vertex order convention assumed: [0]=top-left,
// [1]=top-right, [2]=bottom-right, [3]=bottom-left.
constexpr double kQuadIdentitySrc[4][2] = {{0, 0}, {20, 0}, {20, 14}, {0, 14}};
constexpr double kQuadIdentityDst[4][2] = {{0, 0}, {20, 0}, {20, 14}, {0, 14}};
constexpr double kQuadScaleSrc[4][2] = {{0, 0}, {12, 0}, {12, 9}, {0, 9}};
constexpr double kQuadScaleDst[4][2] = {{0, 0}, {24, 0}, {24, 18}, {0, 18}};
constexpr double kQuadPerspSrc[4][2] = {{1, 1}, {19, 2}, {18, 12}, {2, 11}};
constexpr double kQuadPerspDst[4][2] = {{2, 3}, {30, 1}, {27, 20}, {5, 17}};
constexpr double kQuadTiltSrc[4][2] = {{0, 0}, {12, 0}, {12, 9}, {0, 9}};
constexpr double kQuadTiltDst[4][2] = {{3, 1}, {21, 4}, {18, 17}, {1, 12}};
constexpr double kQuadSameSrc[4][2] = {{2, 1}, {18, 3}, {16, 12}, {4, 13}};
constexpr double kQuadSameDst[4][2] = {{2, 1}, {18, 3}, {16, 12}, {4, 13}};
constexpr double kQuadDegenerate[4][2] = {{4, 4}, {4, 4}, {10, 8}, {2, 9}};
constexpr double kQuadCollinear[4][2] = {{0, 0}, {5, 5}, {10, 10}, {15, 15}};

}  // namespace

class NppiWarpPerspectiveQuadDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

#define QUAD_PACKED_CASES(TAG, DT, DTS)                                                                       \
  TEST_F(NppiWarpPerspectiveQuadDiff, Quad_##TAG##_C1) {                                                      \
    quad_packed_case<DT, 1>("nppiWarpPerspectiveQuad_" DTS "_C1R", DTS "/C1R/identity", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadIdentitySrc, \
                            kQuadIdentityDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C1R_Ctx,    \
                            700);                                                                             \
    quad_packed_case<DT, 1>("nppiWarpPerspectiveQuad_" DTS "_C1R", DTS "/C1R/scale2", NppiSize{12, 9},        \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadScaleSrc,    \
                            kQuadScaleDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C1R_Ctx, 701); \
    quad_packed_case<DT, 1>("nppiWarpPerspectiveQuad_" DTS "_C1R", DTS "/C1R/persp_nn", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C1R_Ctx, 702); \
    quad_packed_case<DT, 1>("nppiWarpPerspectiveQuad_" DTS "_C1R", DTS "/C1R/persp_linear", NppiSize{20, 14}, \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_LINEAR, true, &nppiWarpPerspectiveQuad_##TAG##_C1R_Ctx,   \
                            703);                                                                             \
    quad_packed_case<DT, 1>("nppiWarpPerspectiveQuad_" DTS "_C1R", DTS "/C1R/same_quad", NppiSize{20, 14},    \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadSameSrc,    \
                            kQuadSameDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C1R_Ctx, 704);  \
    quad_packed_case<DT, 1>("nppiWarpPerspectiveQuad_" DTS "_C1R", DTS "/C1R/rng_tilt", NppiSize{12, 9},      \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadTiltSrc,     \
                            kQuadTiltDst, NPPI_INTER_NN, false, &nppiWarpPerspectiveQuad_##TAG##_C1R_Ctx, 705); \
  }                                                                                                           \
  TEST_F(NppiWarpPerspectiveQuadDiff, Quad_##TAG##_C3) {                                                      \
    quad_packed_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_C3R", DTS "/C3R/identity", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadIdentitySrc, \
                            kQuadIdentityDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C3R_Ctx,    \
                            710);                                                                             \
    quad_packed_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_C3R", DTS "/C3R/scale2", NppiSize{12, 9},        \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadScaleSrc,    \
                            kQuadScaleDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C3R_Ctx, 711); \
    quad_packed_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_C3R", DTS "/C3R/persp_nn", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C3R_Ctx, 712); \
    quad_packed_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_C3R", DTS "/C3R/persp_linear", NppiSize{20, 14}, \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_LINEAR, true, &nppiWarpPerspectiveQuad_##TAG##_C3R_Ctx,   \
                            713);                                                                             \
  }                                                                                                           \
  TEST_F(NppiWarpPerspectiveQuadDiff, Quad_##TAG##_C4) {                                                      \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_C4R", DTS "/C4R/identity", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadIdentitySrc, \
                            kQuadIdentityDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C4R_Ctx,    \
                            720);                                                                             \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_C4R", DTS "/C4R/scale2", NppiSize{12, 9},        \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadScaleSrc,    \
                            kQuadScaleDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C4R_Ctx, 721); \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_C4R", DTS "/C4R/persp_nn", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_C4R_Ctx, 722); \
  }                                                                                                           \
  TEST_F(NppiWarpPerspectiveQuadDiff, Quad_##TAG##_AC4) {                                                     \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_AC4R", DTS "/AC4R/identity", NppiSize{20, 14},   \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadIdentitySrc, \
                            kQuadIdentityDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_AC4R_Ctx,   \
                            730);                                                                             \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_AC4R", DTS "/AC4R/scale2", NppiSize{12, 9},      \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadScaleSrc,    \
                            kQuadScaleDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_AC4R_Ctx,      \
                            731);                                                                             \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_AC4R", DTS "/AC4R/persp_nn", NppiSize{20, 14},   \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_AC4R_Ctx,      \
                            732);                                                                             \
    quad_packed_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_AC4R", DTS "/AC4R/persp_linear", NppiSize{20, 14}, \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_LINEAR, true, &nppiWarpPerspectiveQuad_##TAG##_AC4R_Ctx,  \
                            733);                                                                             \
  }                                                                                                           \
  TEST_F(NppiWarpPerspectiveQuadDiff, Quad_##TAG##_P3) {                                                      \
    quad_planar_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_P3R", DTS "/P3R/identity", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadIdentitySrc, \
                            kQuadIdentityDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_P3R_Ctx,    \
                            740);                                                                             \
    quad_planar_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_P3R", DTS "/P3R/scale2", NppiSize{12, 9},        \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadScaleSrc,    \
                            kQuadScaleDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_P3R_Ctx, 741); \
    quad_planar_case<DT, 3>("nppiWarpPerspectiveQuad_" DTS "_P3R", DTS "/P3R/persp_nn", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_P3R_Ctx, 742); \
  }                                                                                                           \
  TEST_F(NppiWarpPerspectiveQuadDiff, Quad_##TAG##_P4) {                                                      \
    quad_planar_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_P4R", DTS "/P4R/identity", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{20, 14}, NppiRect{0, 0, 20, 14}, kQuadIdentitySrc, \
                            kQuadIdentityDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_P4R_Ctx,    \
                            750);                                                                             \
    quad_planar_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_P4R", DTS "/P4R/scale2", NppiSize{12, 9},        \
                            NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18}, kQuadScaleSrc,    \
                            kQuadScaleDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_P4R_Ctx, 751); \
    quad_planar_case<DT, 4>("nppiWarpPerspectiveQuad_" DTS "_P4R", DTS "/P4R/persp_nn", NppiSize{20, 14},     \
                            NppiRect{0, 0, 20, 14}, NppiSize{32, 24}, NppiRect{0, 0, 32, 24}, kQuadPerspSrc,   \
                            kQuadPerspDst, NPPI_INTER_NN, true, &nppiWarpPerspectiveQuad_##TAG##_P4R_Ctx, 752); \
  }

QUAD_PACKED_CASES(8u, Npp8u, "8u")
QUAD_PACKED_CASES(16u, Npp16u, "16u")
QUAD_PACKED_CASES(32s, Npp32s, "32s")
QUAD_PACKED_CASES(32f, Npp32f, "32f")

// ===== contract: status precedence and error codes ==========================
TEST_F(NppiWarpPerspectiveQuadDiff, Quad_Contract) {
  const NppiSize srcSize{20, 14};
  const NppiRect srcROI{0, 0, 20, 14};
  const NppiSize dstSize{32, 24};
  const NppiRect dstROI{0, 0, 32, 24};
  const int sStep = align_step(srcSize.width + 8, 1);
  const int dStep = align_step(dstSize.width + 8, 1);
  const size_t se = (size_t)srcSize.height * sStep;
  const size_t de = (size_t)dstSize.height * dStep;
  // Oversized allocation: the src_roi_oob probe reads up to (33, 13) if the
  // implementation does not validate the ROI, which must not crash.
  DeviceMemory<Npp8u> &s = arena<Npp8u>(8000);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(std::vector<Npp8u>(8000, 3));
  d.copyFromHost(std::vector<Npp8u>(de, 0xA5));

  auto run = [&](const std::string &key, const Npp8u *psrc, NppiSize ssize, int sstep, NppiRect sroi, Npp8u *pdst,
                 int dstep, NppiRect droi, const double sq[4][2], const double dq[4][2], int interp) {
    clear_cuda_error();
    NppStatus st = nppiWarpPerspectiveQuad_8u_C1R_Ctx(psrc, ssize, sstep, sroi, sq, pdst, dstep, droi, dq, interp,
                                                      current_ctx());
    Case c("nppiWarpPerspectiveQuad/contract/" + key, "nppiWarpPerspectiveQuad_8u_C1R");
    c.status(st);
    c.finish();
  };
  // Image contract with valid quads.
  run("null_src", nullptr, srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadPerspSrc, kQuadPerspDst,
      NPPI_INTER_NN);
  run("null_dst", s.get(), srcSize, sStep, srcROI, nullptr, dStep, dstROI, kQuadPerspSrc, kQuadPerspDst,
      NPPI_INTER_NN);
  run("src_step0", s.get(), srcSize, 0, srcROI, d.get(), dStep, dstROI, kQuadPerspSrc, kQuadPerspDst, NPPI_INTER_NN);
  run("dst_step0", s.get(), srcSize, sStep, srcROI, d.get(), 0, dstROI, kQuadPerspSrc, kQuadPerspDst, NPPI_INTER_NN);
  run("src_size_empty", s.get(), NppiSize{0, 14}, sStep, srcROI, d.get(), dStep, dstROI, kQuadPerspSrc,
      kQuadPerspDst, NPPI_INTER_NN);
  run("src_roi_w0", s.get(), srcSize, sStep, NppiRect{0, 0, 0, 14}, d.get(), dStep, dstROI, kQuadPerspSrc,
      kQuadPerspDst, NPPI_INTER_NN);
  run("dst_roi_w0", s.get(), srcSize, sStep, srcROI, d.get(), dStep, NppiRect{0, 0, 0, 24}, kQuadPerspSrc,
      kQuadPerspDst, NPPI_INTER_NN);
  run("src_roi_oob", s.get(), srcSize, sStep, NppiRect{2, 2, 32, 12}, d.get(), dStep, dstROI, kQuadPerspSrc,
      kQuadPerspDst, NPPI_INTER_NN);
  run("interp_undefined", s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadPerspSrc, kQuadPerspDst,
      NPPI_INTER_UNDEFINED);
  // Degenerate quads with an otherwise valid call.
  run("src_quad_duplicate", s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadDegenerate, kQuadPerspDst,
      NPPI_INTER_NN);
  run("dst_quad_duplicate", s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadPerspSrc, kQuadDegenerate,
      NPPI_INTER_NN);
  run("src_quad_collinear", s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadCollinear, kQuadPerspDst,
      NPPI_INTER_NN);
  run("dst_quad_collinear", s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadPerspSrc, kQuadCollinear,
      NPPI_INTER_NN);
  // Status precedence probes.
  run("order_null_vs_degenerate", nullptr, srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadDegenerate,
      kQuadPerspDst, NPPI_INTER_NN);
  run("order_step_vs_degenerate", s.get(), srcSize, 0, srcROI, d.get(), dStep, dstROI, kQuadDegenerate,
      kQuadPerspDst, NPPI_INTER_NN);
  run("order_size_vs_degenerate", s.get(), NppiSize{0, 14}, sStep, srcROI, d.get(), dStep, dstROI, kQuadDegenerate,
      kQuadPerspDst, NPPI_INTER_NN);
  run("order_roi_vs_degenerate", s.get(), srcSize, sStep, NppiRect{0, 0, 0, 14}, d.get(), dStep, dstROI,
      kQuadDegenerate, kQuadPerspDst, NPPI_INTER_NN);
  run("order_interp_vs_degenerate", s.get(), srcSize, sStep, srcROI, d.get(), dStep, dstROI, kQuadDegenerate,
      kQuadPerspDst, NPPI_INTER_UNDEFINED);
  // Planar contract: null plane array and null plane with valid quads.
  {
    const size_t plElems = (size_t)dstSize.height * dStep;
    DeviceMemory<Npp8u> &p0 = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &p1 = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &p2 = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &q0 = arena<Npp8u>(plElems);
    DeviceMemory<Npp8u> &q1 = arena<Npp8u>(plElems);
    DeviceMemory<Npp8u> &q2 = arena<Npp8u>(plElems);
    for (auto *m : {&p0, &p1, &p2}) m->copyFromHost(std::vector<Npp8u>(se, 3));
    for (auto *m : {&q0, &q1, &q2}) m->copyFromHost(std::vector<Npp8u>(plElems, 0xA5));
    const Npp8u *plOk[3] = {p0.get(), p1.get(), p2.get()};
    const Npp8u *plNull[3] = {p0.get(), nullptr, p2.get()};
    Npp8u *plDst[3] = {q0.get(), q1.get(), q2.get()};
    auto runPl = [&](const std::string &key, const Npp8u **psrc, Npp8u **pdst, const double sq[4][2]) {
      clear_cuda_error();
      NppStatus st = nppiWarpPerspectiveQuad_8u_P3R_Ctx(psrc, srcSize, sStep, srcROI, sq, pdst, dStep, dstROI,
                                                        kQuadPerspDst, NPPI_INTER_NN, current_ctx());
      Case c("nppiWarpPerspectiveQuad/contract/" + key, "nppiWarpPerspectiveQuad_8u_P3R");
      c.status(st);
      c.finish();
    };
    // NOTE: null plane-array pointers and null individual planes are NOT
    // probed for the planar quad entry points: NVIDIA dereferences the plane
    // array on the host without validating it (process segfault), and a null
    // plane reaches the kernel as a device null pointer (illegal-address
    // fault).  mcMPP returns NPP_NULL_POINTER_ERROR for both.
    (void)plNull;
    runPl("pl_degenerate_quad", plOk, plDst, kQuadDegenerate);
  }
}

// ===== Non-Ctx adapters (representative _Ctx cases -> same golden) ==========
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiWarpPerspectiveQuadDiff, NonCtx_Quad) {
  // Reproduces quad_packed_case<8u,1>(C1R persp_nn, positional, seed=702).
  const int sStep = align_step(20 + 8, 1);
  const int dStep = align_step(32 + 8, 1);
  const size_t se = (size_t)14 * sStep;
  const size_t de = (size_t)24 * dStep;
  std::vector<Npp8u> hs = make_packed_positional<Npp8u>(se, 1, sStep);
  std::vector<Npp8u> hd(de, 0xA5);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiWarpPerspectiveQuad_8u_C1R(s.get(), NppiSize{20, 14}, sStep, NppiRect{0, 0, 20, 14},
                                                kQuadPerspSrc, d.get(), dStep, NppiRect{0, 0, 32, 24}, kQuadPerspDst,
                                                NPPI_INTER_NN);
  Case c("nppiWarpPerspectiveQuad/8u/C1R/persp_nn", "nppiWarpPerspectiveQuad_8u_C1R");
  c.input(hs.data(), hs.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output<Npp8u>(d.get(), de, ValueKind::U8, 1e-5, 1e-6);
  c.finish();

  // Reproduces quad_packed_case<32f,4>(AC4R scale2, positional, seed=731).
  const int sStep32 = align_step(12 * 4 * 4 + 8, 4);
  const int dStep32 = align_step(24 * 4 * 4 + 8, 4);
  const size_t se32 = (size_t)9 * sStep32 / 4;
  const size_t de32 = (size_t)18 * dStep32 / 4;
  std::vector<Npp32f> hs32 = make_packed_positional<Npp32f>(se32, 4, sStep32 / 4);
  std::vector<Npp32f> hd32(de32, (Npp32f)0xA5);
  DeviceMemory<Npp32f> &s32 = arena<Npp32f>(se32);
  DeviceMemory<Npp32f> &d32 = arena<Npp32f>(de32);
  s32.copyFromHost(hs32);
  d32.copyFromHost(hd32);
  clear_cuda_error();
  NppStatus st32 = nppiWarpPerspectiveQuad_32f_AC4R(s32.get(), NppiSize{12, 9}, sStep32, NppiRect{0, 0, 12, 9},
                                                    kQuadScaleSrc, d32.get(), dStep32, NppiRect{0, 0, 24, 18},
                                                    kQuadScaleDst, NPPI_INTER_NN);
  Case c32("nppiWarpPerspectiveQuad/32f/AC4R/scale2", "nppiWarpPerspectiveQuad_32f_AC4R");
  c32.input(hs32.data(), hs32.size() * sizeof(Npp32f));
  c32.status(st32);
  c32.device_output<Npp32f>(d32.get(), de32, ValueKind::F32, 1e-5, 1e-6);
  c32.finish();
}
#endif
