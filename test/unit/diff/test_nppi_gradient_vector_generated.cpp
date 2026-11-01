// Differential tests for package f13 family 2: gradient-vector filters
// (nppiGradientVector{Sobel,Prewitt,Scharr}Border, 48 functions).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiGradientVectorDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build-mpp/src \
//     ./build-mpp/diff_tests --gtest_filter='NppiGradientVectorDiff.*'
//
// Measured semantics (5090, NPP 13.4, tools/probes/f13_probe{4..10}.cu):
//  - only the _Ctx variants exist in the NPP 13 library; the non-ctx ABI
//    was removed, so every case here goes through a _Ctx entry point
//  - X = -(vertical response) for Sobel/Prewitt, +(vertical response) for
//    Scharr; Y = +(horizontal response); Sobel also supports the 5x5 mask
//    (Prewitt/Scharr reject it with -33)
//  - magnitude: L1 = |X|+|Y|, L2 = sqrt(X^2+Y^2); a 16s destination
//    truncates; angle = atan2(Y, X) radians (NPP evaluates it in double
//    precision; the 1-2 ulp difference to float evaluation is far below
//    the F32 comparison tolerance)
//  - the four outputs are individually optional and a null output is not
//    written; all four null -> -8
//  - C3C1R: gradients per source channel, ALL outputs come from the
//    channel with the largest X^2+Y^2 (strict >, channel 0 wins ties)
//  - validation order: null src(-8) -> all outputs null(-8) -> negROI(-6)
//    -> zeroROI(0, no launch) -> step of the FIRST non-null output
//    (too small -16 / misaligned -108) -> srcSize(-6) -> offset(-11)
//    -> border type (REPLICATE only, -14) -> norm (L1/L2, -5, checked even
//    when pDstMag is null) -> mask size(-33); the source step and the
//    steps of the non-first outputs are never validated
//  - the ROI may overhang the source image; reads clamp, all ROI pixels
//    are written
//
// Test data note: 32f sources use integer-valued pixels so that the C3C1R
// channel selection (an argmax over X^2+Y^2) is exact on both sides.  For
// 16s magnitudes the exact comparison is only safe while X^2+Y^2 < 2^23
// (there the float and double sqrt pipelines are bit-identical); the
// full-range Prewitt/Scharr cases exceed that and compare the magnitude
// with ValueKind::I16Tol and atol=1 (NPP's internal sqrt precision cannot
// be pinned down beyond a 1-ulp truncation difference).  Everything else
// compares exactly.

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

template <typename T> ValueKind kind_of();
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

// Integer-valued source data in [lo, hi]: exact convolution and exact
// C3C1R channel selection on both sides.
template <typename T> std::vector<T> gv_data(size_t count, uint64_t seed, int lo, int hi) {
  std::vector<T> v(count);
  Rng rng(seed);
  for (auto &x : v) x = static_cast<T>(rng.integer<int>(lo, hi));
  return v;
}

template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiGradientVector/" + key, fname);
  c.status(st);
  c.finish();
  // Status-only cases never capture an output, so without this sync the
  // kernels launched above (notably the ok_statusonly launches that write
  // through too-small aligned strides) can still be in flight when the
  // next case or family starts.  Observed on the RTX 5090 with NVIDIA NPP
  // 13.4: with kernels left in flight, a later cudaMalloc intermittently
  // fails with a sticky "misaligned address" error and kills every
  // following test in the process.  Retiring each launch here makes the
  // suite safe to run in any process grouping and surfaces an async fault
  // at the case that caused it instead of poisoning the next one.
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
    ADD_FAILURE() << "async CUDA fault after " << key << ": " << cudaGetErrorString(e);
}

// Output selection bitmask.
enum { OUT_X = 1, OUT_Y = 2, OUT_MAG = 4, OUT_ANGLE = 8 };
constexpr unsigned OUT_ALL = OUT_X | OUT_Y | OUT_MAG | OUT_ANGLE;

// Happy-path runner.  Junk-seeds every output buffer so untouched slack
// (and untouched buffers, for zero-ROI / null-output cases) is compared.
template <typename S, typename D, typename F>
void gv_ok(const std::string &fname, const std::string &key, int ch, int srcW, int srcH, int offX,
           int offY, int W, int H, int pad, uint64_t seed, int lo, int hi, NppiMaskSize ms,
           NppiNorm norm, unsigned present, F f) {
  const int sesz = (int)sizeof(S);
  const int desz = (int)sizeof(D);
  const int sStep = align_step(srcW * ch * sesz + pad, sesz);
  const int dStep = align_step(W * desz + pad, desz);
  const int aStep = align_step(W * 4 + pad, 4);
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  const size_t ae = (size_t)H * aStep / 4;
  std::vector<S> hs = gv_data<S>(se, seed + 11, lo, hi);
  std::vector<D> hj = make_uniform<D>(de, seed + 29);
  std::vector<Npp32f> ha = make_uniform<Npp32f>(ae, seed + 37);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &dx = arena<D>(de);
  DeviceMemory<D> &dy = arena<D>(de);
  DeviceMemory<D> &dm = arena<D>(de);
  DeviceMemory<Npp32f> &da = arena<Npp32f>(ae);
  s.copyFromHost(hs);
  dx.copyFromHost(hj);
  dy.copyFromHost(hj);
  dm.copyFromHost(hj);
  da.copyFromHost(ha);
  const S *pSrc = s.get() + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * ch;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY},
                   (present & OUT_X) ? dx.get() : nullptr, dStep,
                   (present & OUT_Y) ? dy.get() : nullptr, dStep,
                   (present & OUT_MAG) ? dm.get() : nullptr, dStep,
                   (present & OUT_ANGLE) ? da.get() : nullptr, aStep, NppiSize{W, H}, ms, norm,
                   NPP_BORDER_REPLICATE, current_ctx());
  Case c("nppiGradientVector/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  if (present & OUT_X) c.device_output<D>(dx.get(), de, kind_of<D>());
  if (present & OUT_Y) c.device_output<D>(dy.get(), de, kind_of<D>());
  if (present & OUT_MAG) c.device_output<D>(dm.get(), de, kind_of<D>());
  if (present & OUT_ANGLE) c.device_output<Npp32f>(da.get(), ae, ValueKind::F32);
  c.finish();
}

// Happy-path runner with an off-spec magnitude comparison (I16Tol, atol 1)
// for full-range 8u L2 where the sqrt rounding band reaches integer size.
template <typename S, typename D, typename F>
void gv_ok_magtol(const std::string &fname, const std::string &key, int ch, int srcW, int srcH,
                  int offX, int offY, int W, int H, int pad, uint64_t seed, int lo, int hi,
                  NppiMaskSize ms, NppiNorm norm, F f) {
  const int sesz = (int)sizeof(S);
  const int desz = (int)sizeof(D);
  const int sStep = align_step(srcW * ch * sesz + pad, sesz);
  const int dStep = align_step(W * desz + pad, desz);
  const int aStep = align_step(W * 4 + pad, 4);
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  const size_t ae = (size_t)H * aStep / 4;
  std::vector<S> hs = gv_data<S>(se, seed + 11, lo, hi);
  std::vector<D> hj = make_uniform<D>(de, seed + 29);
  std::vector<Npp32f> ha = make_uniform<Npp32f>(ae, seed + 37);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &dx = arena<D>(de);
  DeviceMemory<D> &dy = arena<D>(de);
  DeviceMemory<D> &dm = arena<D>(de);
  DeviceMemory<Npp32f> &da = arena<Npp32f>(ae);
  s.copyFromHost(hs);
  dx.copyFromHost(hj);
  dy.copyFromHost(hj);
  dm.copyFromHost(hj);
  da.copyFromHost(ha);
  const S *pSrc = s.get() + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * ch;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, dx.get(), dStep,
                   dy.get(), dStep, dm.get(), dStep, da.get(), aStep, NppiSize{W, H}, ms, norm,
                   NPP_BORDER_REPLICATE, current_ctx());
  Case c("nppiGradientVector/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(dx.get(), de, kind_of<D>());
  c.device_output<D>(dy.get(), de, kind_of<D>());
  c.device_output<D>(dm.get(), de, ValueKind::I16Tol, 0.0, 1.0);
  c.device_output<Npp32f>(da.get(), ae, ValueKind::F32);
  c.finish();
}

// Error matrix.  `base` is a raw scratch buffer; pSrc is inset by one row
// and one pixel so that the launched cases (the ok_statusonly pins below)
// read inside the allocation.  `s5x5` = family supports the 5x5 mask.
template <typename S, typename D, typename F>
void gv_errors(const std::string &fam, const std::string &suffix, int ch, S *base, D *dx, D *dy,
               D *dm, Npp32f *da, bool s5x5, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const int sstep = 16;
  S *s = base + sstep / (int)sizeof(S) + ch;  // pSrc at oSrcOffset {1, 1}
  const NppiSize sz{9, 9};
  const NppiPoint off{1, 1};
  const NppiSize roi{4, 4};
  const NppiBorderType badB = NPP_BORDER_CONSTANT;
  const NppiMaskSize badM = (NppiMaskSize)0;
  const int esz = (int)sizeof(D);
  // null pointer checks (win over everything else)
  only_status(fn, fam + "/err/nullsrc" + suffix, [&] {
    return f(nullptr, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/allnull" + suffix, [&] {
    return f(s, sstep, sz, off, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, roi, badM,
             nppiNormInf, badB, ctx);
  });
  // ROI: negative before the step check, zero before everything else
  only_status(fn, fam + "/err/negroi" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 3, dy, 16, dm, 16, da, 16, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/zeroroi" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 3, dy, 16, dm, 16, da, 16, NppiSize{0, 4}, badM, nppiNormInf,
             badB, ctx);
  });
  // first non-null output (X) step: too small / misaligned; wins over
  // srcSize, offset, border, norm and mask
  only_status(fn, fam + "/err/stepsmall" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 4 * esz - 1, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/stepodd" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 4 * esz + 1, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/srcsize0_xbad" + suffix, [&] {
    return f(s, sstep, NppiSize{0, 8}, off, dx, 3, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/offneg_xbad" + suffix, [&] {
    return f(s, sstep, sz, NppiPoint{-1, 0}, dx, 3, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  // srcSize / offset (offset must be inside the source image)
  only_status(fn, fam + "/err/srcsize" + suffix, [&] {
    return f(s, sstep, NppiSize{0, 8}, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/offneg" + suffix, [&] {
    return f(s, sstep, sz, NppiPoint{0, -1}, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/offeqsrc" + suffix, [&] {
    return f(s, sstep, sz, NppiPoint{9, 1}, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  // border type: REPLICATE only, wins over norm and mask
  only_status(fn, fam + "/err/badborder" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2,
             badB, ctx);
  });
  only_status(fn, fam + "/err/badborder_badnorm" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormInf,
             badB, ctx);
  });
  // norm: L1/L2 only, checked even when pDstMag is null
  only_status(fn, fam + "/err/badnorm" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormInf,
             NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/badnorm_magnull" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, nullptr, 0, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormInf, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/badnorm_badmask" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, badM, nppiNormInf,
             NPP_BORDER_REPLICATE, ctx);
  });
  // mask size: 3x3 always; 5x5 for Sobel only
  only_status(fn, fam + "/err/badmask" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, badM, nppiNormL2,
             NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/badmask7" + suffix, [&] {
    return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_7_X_7, nppiNormL2,
             NPP_BORDER_REPLICATE, ctx);
  });
  if (!s5x5)
    only_status(fn, fam + "/err/mask5x5_unsupported" + suffix, [&] {
      return f(s, sstep, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2,
               NPP_BORDER_REPLICATE, ctx);
    });
  // first-non-null model: with X null the Y (or Mag/Angle) step is
  // validated; later outputs' steps never are (NPP launches and writes
  // through the odd strides without complaint - probe6)
  only_status(fn, fam + "/err/xnull_ybadstep" + suffix, [&] {
    return f(s, sstep, sz, off, nullptr, 0, dy, 3, dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/magonly_badstep" + suffix, [&] {
    return f(s, sstep, sz, off, nullptr, 0, nullptr, 0, dm, 3, nullptr, 0, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/angleonly_badstep" + suffix, [&] {
    return f(s, sstep, sz, off, nullptr, 0, nullptr, 0, nullptr, 0, da, 3, roi, NPP_MASK_SIZE_3_X_3,
             nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  // status-only happy paths whose later-output strides are too small but
  // element-aligned.  An odd (element-misaligned) stride would put the
  // kernel stores on unaligned addresses, so the racy overlap of an
  // aligned-but-too-small stride is used instead; only the status is
  // compared, never the overlapping buffer content.  only_status() syncs
  // after every launch, so these cannot leave a faulting kernel in flight.
  only_status(fn, fam + "/ok_statusonly/xnull_mbadstep" + suffix, [&] {
    return f(s, sstep, sz, off, nullptr, 0, dy, 16, dm, 4 * esz - esz, da, 16, roi,
             NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/ok_statusonly/magonly_abadstep" + suffix, [&] {
    return f(s, sstep, sz, off, nullptr, 0, nullptr, 0, dm, 16, da, 4 * 4 - 4, roi,
             NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, ctx);
  });
}

// One family's full ok matrix (lo/hi = source data range).
#define GV_OK_MATRIX(FAM, S5X5, S8LO, S8HI, S16LO, S16HI, S16UHI, FHI)                               \
  do {                                                                                               \
    const std::string fam = #FAM;                                                                    \
    /* C1R / C3C1R, all outputs, L2 */                                                                \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/l2", 1, 13, 13, 1, 1, 11, 11, 3,    \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,             \
                         FAM##_8u16s_C1R_Ctx);                                                       \
    gv_ok<Npp16s, Npp32f>(fam + "_16s32f_C1R_Ctx", fam + "/16s32f/C1/l2", 1, 13, 13, 1, 1, 11, 11, 3, \
                          __LINE__, S16LO, S16HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,          \
                          FAM##_16s32f_C1R_Ctx);                                                     \
    gv_ok<Npp16u, Npp32f>(fam + "_16u32f_C1R_Ctx", fam + "/16u32f/C1/l2", 1, 13, 13, 1, 1, 11, 11, 3, \
                          __LINE__, 0, S16UHI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,             \
                          FAM##_16u32f_C1R_Ctx);                                                     \
    gv_ok<Npp32f, Npp32f>(fam + "_32f_C1R_Ctx", fam + "/32f/C1/l2", 1, 13, 13, 1, 1, 11, 11, 3,       \
                          __LINE__, 0, FHI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,                \
                          FAM##_32f_C1R_Ctx);                                                        \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C3C1R_Ctx", fam + "/8u16s/C3C1/l2", 3, 13, 13, 1, 1, 11, 11, 3, \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,             \
                         FAM##_8u16s_C3C1R_Ctx);                                                     \
    gv_ok<Npp16s, Npp32f>(fam + "_16s32f_C3C1R_Ctx", fam + "/16s32f/C3C1/l2", 3, 13, 13, 1, 1, 11,    \
                          11, 3, __LINE__, S16LO, S16HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,   \
                          FAM##_16s32f_C3C1R_Ctx);                                                   \
    gv_ok<Npp16u, Npp32f>(fam + "_16u32f_C3C1R_Ctx", fam + "/16u32f/C3C1/l2", 3, 13, 13, 1, 1, 11,    \
                          11, 3, __LINE__, 0, S16UHI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,      \
                          FAM##_16u32f_C3C1R_Ctx);                                                   \
    gv_ok<Npp32f, Npp32f>(fam + "_32f_C3C1R_Ctx", fam + "/32f/C3C1/l2", 3, 13, 13, 1, 1, 11, 11, 3,   \
                          __LINE__, 0, FHI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,                \
                          FAM##_32f_C3C1R_Ctx);                                                      \
    /* L1 norm */                                                                                     \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/l1", 1, 13, 13, 1, 1, 11, 11, 3,    \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL1, OUT_ALL,             \
                         FAM##_8u16s_C1R_Ctx);                                                       \
    gv_ok<Npp32f, Npp32f>(fam + "_32f_C1R_Ctx", fam + "/32f/C1/l1", 1, 13, 13, 1, 1, 11, 11, 3,       \
                          __LINE__, 0, FHI, NPP_MASK_SIZE_3_X_3, nppiNormL1, OUT_ALL,                \
                          FAM##_32f_C1R_Ctx);                                                        \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C3C1R_Ctx", fam + "/8u16s/C3C1/l1", 3, 13, 13, 1, 1, 11, 11, 3, \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL1, OUT_ALL,             \
                         FAM##_8u16s_C3C1R_Ctx);                                                     \
    /* optional outputs: X only / angle only / no X */                                                \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/xonly", 1, 13, 13, 1, 1, 11, 11, 3, \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_X,               \
                         FAM##_8u16s_C1R_Ctx);                                                       \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/angleonly", 1, 13, 13, 1, 1, 11,    \
                         11, 3, __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL1, OUT_ANGLE,    \
                         FAM##_8u16s_C1R_Ctx);                                                       \
    gv_ok<Npp32f, Npp32f>(fam + "_32f_C3C1R_Ctx", fam + "/32f/C3C1/nox", 3, 13, 13, 1, 1, 11, 11, 3,  \
                          __LINE__, 0, FHI, NPP_MASK_SIZE_3_X_3, nppiNormL2,                         \
                          OUT_Y | OUT_MAG | OUT_ANGLE, FAM##_32f_C3C1R_Ctx);                         \
    /* corner (offset 0) and overhanging ROI */                                                       \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/corner", 1, 9, 9, 0, 0, 9, 9, 3,    \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,             \
                         FAM##_8u16s_C1R_Ctx);                                                       \
    gv_ok<Npp32f, Npp32f>(fam + "_32f_C1R_Ctx", fam + "/32f/C1/overhang", 1, 9, 9, 5, 5, 5, 5, 3,     \
                          __LINE__, 0, FHI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,                \
                          FAM##_32f_C1R_Ctx);                                                        \
    /* zero ROI: no launch, buffers untouched (junk-seeded) */                                        \
    gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/zeroroi", 1, 9, 9, 1, 1, 0, 4, 3,   \
                         __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3, nppiNormL2, OUT_ALL,             \
                         FAM##_8u16s_C1R_Ctx);                                                       \
    if (S5X5) {                                                                                       \
      /* Sobel-only 5x5; the small 8u range keeps both sqrt pipelines exact */                         \
      gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/l2_5x5_small", 1, 13, 13, 1, 1,   \
                           11, 11, 3, __LINE__, 0, 31, NPP_MASK_SIZE_5_X_5, nppiNormL2, OUT_ALL,     \
                           FAM##_8u16s_C1R_Ctx);                                                     \
      gv_ok<Npp8u, Npp16s>(fam + "_8u16s_C3C1R_Ctx", fam + "/8u16s/C3C1/l1_5x5", 3, 13, 13, 1, 1, 11, \
                           11, 3, __LINE__, S8LO, S8HI, NPP_MASK_SIZE_5_X_5, nppiNormL1, OUT_ALL,    \
                           FAM##_8u16s_C3C1R_Ctx);                                                   \
      gv_ok<Npp32f, Npp32f>(fam + "_32f_C1R_Ctx", fam + "/32f/C1/l2_5x5", 1, 13, 13, 1, 1, 11, 11, 3, \
                            __LINE__, 0, FHI, NPP_MASK_SIZE_5_X_5, nppiNormL2, OUT_ALL,              \
                            FAM##_32f_C1R_Ctx);                                                      \
      gv_ok<Npp32f, Npp32f>(fam + "_32f_C3C1R_Ctx", fam + "/32f/C3C1/l2_5x5", 3, 13, 13, 1, 1, 11,    \
                            11, 3, __LINE__, 0, FHI, NPP_MASK_SIZE_5_X_5, nppiNormL2, OUT_ALL,       \
                            FAM##_32f_C3C1R_Ctx);                                                    \
    } else {                                                                                          \
      /* full-range 8u L2: magnitude compared with off-by-one slack */                                 \
      gv_ok_magtol<Npp8u, Npp16s>(fam + "_8u16s_C1R_Ctx", fam + "/8u16s/C1/l2_magtol", 1, 13, 13, 1,  \
                                  1, 11, 11, 3, __LINE__, S8LO, S8HI, NPP_MASK_SIZE_3_X_3,           \
                                  nppiNormL2, FAM##_8u16s_C1R_Ctx);                                  \
    }                                                                                                 \
    /* error matrix on two representative variants */                                                 \
    gv_errors<Npp8u, Npp16s>(fam, "_8u16s_C1R_Ctx", 1, arena<Npp8u>(256).get(),                       \
                             arena<Npp16s>(256).get(), arena<Npp16s>(256).get(),                     \
                             arena<Npp16s>(256).get(), arena<Npp32f>(256).get(), S5X5,               \
                             FAM##_8u16s_C1R_Ctx);                                                   \
    gv_errors<Npp32f, Npp32f>(fam, "_32f_C1R_Ctx", 1, arena<Npp32f>(256).get(),                       \
                              arena<Npp32f>(256).get(), arena<Npp32f>(256).get(),                    \
                              arena<Npp32f>(256).get(), arena<Npp32f>(256).get(), S5X5,              \
                              FAM##_32f_C1R_Ctx);                                                    \
  } while (0)

}  // namespace

class NppiGradientVectorDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// Data ranges: 8u/16u/32f cover their full (integer-valued) range; 16s
// sources use the full signed range.
TEST_F(NppiGradientVectorDiff, Sobel) {
  GV_OK_MATRIX(nppiGradientVectorSobelBorder, true, 0, 255, -32768, 32767, 65535, 24);
}
TEST_F(NppiGradientVectorDiff, Prewitt) {
  GV_OK_MATRIX(nppiGradientVectorPrewittBorder, false, 0, 255, -32768, 32767, 65535, 24);
}
TEST_F(NppiGradientVectorDiff, Scharr) {
  GV_OK_MATRIX(nppiGradientVectorScharrBorder, false, 0, 255, -32768, 32767, 65535, 24);
}
