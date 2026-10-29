// Differential tests for the nppiCrossCorr* and nppiSqrDistance*_Norm
// families (package p1.8): Full/Same/Valid cross correlation (plain, Norm,
// NormLevel, NormLevelAdvanced) and normalized squared distance.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiCrossCorrDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiCrossCorrDiff.*'
//
// Measured semantics (5090, NPP 13.4):
//   * dst geometry: Full (Ws+tw-1)x(Hs+th-1) window origin (c-(tw-1), r-(th-1));
//     Same WsxHs origin (c-tw/2, r-th/2) (integer division); Valid
//     (Ws-tw+1)x(Hs-th+1) origin (c, r).  Out-of-bounds source samples
//     contribute zero (zero padding).
//   * Norm: gamma = Rst / sqrt(Rss*Rtt), 0/0 -> NaN.
//   * NormLevel (ZNCC): zero-variance windows (varW*varT == 0) write a
//     sentinel (FLT_MAX for <=4-byte dst, DBL_MAX for 64f dst) to every dst
//     pixel of the image, not just the affected windows.
//   * SqrDistance_Norm: sigma = sum((w-t)^2) / sqrt(Rss*Rtt); zero src or
//     zero tpl -> +inf, both zero -> NaN.
//   * 8u Sfs store: trunc_toward_zero(value) scaled by 2^(-nScaleFactor)
//     (truncation BEFORE scaling), then clamped to [0,255]; NaN -> 0,
//     +/-inf -> 0/255.
// Test data uses small integers so every per-window sum is exact in float as
// well as in double; the values are kept away from Sfs truncation boundaries
// so the goldens never depend on NVIDIA's internal accumulation precision.
// Destinations are pre-filled with a 0x7F byte pattern so unwritten pixels
// are deterministic (and any deviation shows up as a real diff).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
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

// Fresh 4 MB device scratch per call (NormLevel device buffer, Advanced
// scratch buffer, ...).
inline Npp8u *scratch() { return arena<Npp8u>(4 << 20).get(); }

enum class CCMode { Full, Same, Valid };

inline NppiSize dst_size(CCMode mode, NppiSize src, NppiSize tpl) {
  switch (mode) {
  case CCMode::Full: return NppiSize{src.width + tpl.width - 1, src.height + tpl.height - 1};
  case CCMode::Same: return NppiSize{src.width, src.height};
  default: return NppiSize{src.width - tpl.width + 1, src.height - tpl.height + 1};
  }
}

template <typename D> ValueKind dst_kind() {
  if constexpr (sizeof(D) == 1) return ValueKind::U8;
  else if constexpr (sizeof(D) == 4) return ValueKind::F32;
  else return ValueKind::F64;
}

// Small integers: 8u/32f/64f in [1,20], 8s in [-20,20], 16u in [1,1000].
template <typename T> std::vector<T> make_small(size_t count, uint64_t seed) {
  std::vector<T> data(count);
  Rng rng(seed);
  for (auto &v : data) {
    if constexpr (std::is_same_v<T, Npp8s>)
      v = static_cast<T>(rng.integer<int>(-20, 20));
    else if constexpr (std::is_same_v<T, Npp16u>)
      v = static_cast<T>(1 + rng.integer<uint32_t>(0, 999));
    else
      v = static_cast<T>(1 + rng.integer<uint32_t>(0, 19));
  }
  return data;
}

// Row-padded host image: `pad` extra bytes per row (rounded up to the element
// size), row tails filled with a fixed 0xC3 pattern to prove only the ROI is
// read.  `vals` holds one scalar per pixel, replicated across channels.
template <typename T>
std::vector<T> pad_image(int width, int height, int ch, int pad, const std::vector<T> &vals) {
  const int se = (int)sizeof(T);
  const int step = align_step(width * ch * se + pad, se);
  T tail{};
  std::memset(&tail, 0xC3, se);
  std::vector<T> img((size_t)height * (size_t)(step / se), tail);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < ch; ++c)
        img[(size_t)y * (step / se) + (size_t)(x * ch + c)] = vals[(size_t)y * width + x];
  return img;
}

template <typename T> DeviceMemory<T> &put_image(const std::vector<T> &h) {
  DeviceMemory<T> &d = arena<T>(h.empty() ? 1 : h.size());
  if (!h.empty()) d.copyFromHost(h);
  return d;
}

// Per-case knobs (defaults exercise the happy path).
template <typename S> struct Knobs {
  int srcStepOv = -1;  // pass this step instead of the valid one
  int tplStepOv = -1;
  int dstStepOv = -1;
  int srcPad = 8;      // row padding of the host images
  int tplPad = 4;
  int dstPad = 8;
  bool nullSrc = false;
  bool nullTpl = false;
  bool nullDst = false;
  double rtol = 1e-5;
  double atol = 1e-6;
  const std::vector<S> *srcData = nullptr;  // override the generated scalars
  const std::vector<S> *tplData = nullptr;
};

// Generic runner: builds deterministic images, sentinel-fills the dst, calls
// `call` with raw pointers/steps and captures the dst.  For non-NO_ERROR
// outcomes the harness compares only the status (error goldens skip output
// comparison), so the same runner serves the contract tests.
template <typename S, typename D>
void corr_case(const char *fname, const std::string &id, CCMode mode, NppiSize srcRoi, NppiSize tplRoi, int ch,
               uint64_t seed, const std::function<NppStatus(const S *, int, const S *, int, D *, int)> &call,
               const Knobs<S> &k = Knobs<S>()) {
  const int se = (int)sizeof(S), de = (int)sizeof(D);
  // Negative ROI dimensions never reach the kernels; build 1-pixel dummies so
  // the allocations stay valid (zero dimensions keep zero-width images).
  const int iw = srcRoi.width >= 0 ? srcRoi.width : 1;
  const int ih = srcRoi.height >= 0 ? srcRoi.height : 1;
  const int tw = tplRoi.width >= 0 ? tplRoi.width : 1;
  const int th = tplRoi.height >= 0 ? tplRoi.height : 1;
  const std::vector<S> hsVals = k.srcData ? *k.srcData : make_small<S>((size_t)iw * ih, seed);
  const std::vector<S> htVals = k.tplData ? *k.tplData : make_small<S>((size_t)tw * th, seed ^ 0x9E3779B97F4A7C15ULL);
  const std::vector<S> hs = pad_image<S>(iw, ih, ch, k.srcPad, hsVals);
  const std::vector<S> ht = pad_image<S>(tw, th, ch, k.tplPad, htVals);
  const int srcStep = align_step(iw * ch * se + k.srcPad, se);
  const int tplStep = align_step(tw * ch * se + k.tplPad, se);
  const NppiSize ds = dst_size(mode, srcRoi, tplRoi);
  const int dw = ds.width > 0 ? ds.width : 0;
  const int dh = ds.height > 0 ? ds.height : 0;
  const int dstStep = align_step(dw * ch * de + k.dstPad, de);
  DeviceMemory<S> &s = put_image(hs);
  DeviceMemory<S> &t = put_image(ht);
  D fill{};
  std::memset(&fill, 0x7F, sizeof(D));
  const size_t dElems = (size_t)dh * (size_t)(dstStep / de);
  const std::vector<D> hd(dElems, fill);
  DeviceMemory<D> &d = put_image(hd);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = call(k.nullSrc ? nullptr : s.get(), k.srcStepOv >= 0 ? k.srcStepOv : srcStep,
                      k.nullTpl ? nullptr : t.get(), k.tplStepOv >= 0 ? k.tplStepOv : tplStep,
                      k.nullDst ? nullptr : d.get(), k.dstStepOv >= 0 ? k.dstStepOv : dstStep);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.input(ht.data(), ht.size() * sizeof(S));
  c.status(st);
  if (!k.nullDst) c.device_output(d.get(), dElems, dst_kind<D>(), k.rtol, k.atol);
  c.finish();
}

// ------------------------- typed entry points ------------------------------
// Plain / Norm / SqrDistance_Norm without scale factor or device buffer.
template <typename S, typename D>
using FnPlain = NppStatus (*)(const S *, int, NppiSize, const S *, int, NppiSize, D *, int, NppStreamContext);

template <typename S, typename D>
void corr_plain(const char *fname, const std::string &id, CCMode mode, NppiSize srcRoi, NppiSize tplRoi, int ch,
                uint64_t seed, FnPlain<S, D> f, const Knobs<S> &k = Knobs<S>()) {
  corr_case<S, D>(fname, id, mode, srcRoi, tplRoi, ch, seed,
                  [f, srcRoi, tplRoi](const S *s, int ss, const S *t, int ts, D *d, int ds) {
                    return f(s, ss, srcRoi, t, ts, tplRoi, d, ds, current_context());
                  },
                  k);
}

// Norm / SqrDistance_Norm with 8u output and scale factor.
template <typename S>
using FnSfs = NppStatus (*)(const S *, int, NppiSize, const S *, int, NppiSize, Npp8u *, int, int, NppStreamContext);

template <typename S>
void corr_sfs(const char *fname, const std::string &id, CCMode mode, NppiSize srcRoi, NppiSize tplRoi, int ch,
              int sf, uint64_t seed, FnSfs<S> f, const Knobs<S> &k = Knobs<S>()) {
  corr_case<S, Npp8u>(fname, id, mode, srcRoi, tplRoi, ch, seed,
                      [f, srcRoi, tplRoi, sf](const S *s, int ss, const S *t, int ts, Npp8u *d, int ds) {
                        return f(s, ss, srcRoi, t, ts, tplRoi, d, ds, sf, current_context());
                      },
                      k);
}

// NormLevel without scale factor (device buffer).
template <typename S, typename D>
using FnNl = NppStatus (*)(const S *, int, NppiSize, const S *, int, NppiSize, D *, int, Npp8u *, NppStreamContext);

template <typename S, typename D>
void corr_nl(const char *fname, const std::string &id, CCMode mode, NppiSize srcRoi, NppiSize tplRoi, int ch,
             uint64_t seed, FnNl<S, D> f, Npp8u *buf, const Knobs<S> &k = Knobs<S>()) {
  corr_case<S, D>(fname, id, mode, srcRoi, tplRoi, ch, seed,
                  [f, srcRoi, tplRoi, buf](const S *s, int ss, const S *t, int ts, D *d, int ds) {
                    return f(s, ss, srcRoi, t, ts, tplRoi, d, ds, buf, current_context());
                  },
                  k);
}

// NormLevel with 8u output and scale factor.
template <typename S>
using FnSfsNl =
    NppStatus (*)(const S *, int, NppiSize, const S *, int, NppiSize, Npp8u *, int, int, Npp8u *, NppStreamContext);

template <typename S>
void corr_sfsnl(const char *fname, const std::string &id, CCMode mode, NppiSize srcRoi, NppiSize tplRoi, int ch,
                int sf, uint64_t seed, FnSfsNl<S> f, Npp8u *buf, const Knobs<S> &k = Knobs<S>()) {
  corr_case<S, Npp8u>(fname, id, mode, srcRoi, tplRoi, ch, seed,
                      [f, srcRoi, tplRoi, sf, buf](const S *s, int ss, const S *t, int ts, Npp8u *d, int ds) {
                        return f(s, ss, srcRoi, t, ts, tplRoi, d, ds, sf, buf, current_context());
                      },
                      k);
}

// NormLevelAdvanced (device buffer + advanced scratch buffer).
template <typename S, typename D>
using FnAdv =
    NppStatus (*)(const S *, int, NppiSize, const S *, int, NppiSize, D *, int, Npp8u *, Npp8u *, NppStreamContext);

template <typename S, typename D>
void corr_adv(const char *fname, const std::string &id, CCMode mode, NppiSize srcRoi, NppiSize tplRoi, int ch,
              uint64_t seed, FnAdv<S, D> f, Npp8u *dev, Npp8u *adv, const Knobs<S> &k = Knobs<S>()) {
  corr_case<S, D>(fname, id, mode, srcRoi, tplRoi, ch, seed,
                  [f, srcRoi, tplRoi, dev, adv](const S *s, int ss, const S *t, int ts, D *d, int ds) {
                    return f(s, ss, srcRoi, t, ts, tplRoi, d, ds, dev, adv, current_context());
                  },
                  k);
}

}  // namespace

class NppiCrossCorrDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ CrossCorrFull_Norm ============================
TEST_F(NppiCrossCorrDiff, Full_Norm) {
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/full/norm/32f/C1R_17x13_t3x3",
                             CCMode::Full, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x401,
                             &nppiCrossCorrFull_Norm_32f_C1R_Ctx);
  corr_plain<Npp8u, Npp32f>("nppiCrossCorrFull_Norm_8u32f_C3R_Ctx", "xcorr/full/norm/8u32f/C3R_8x6_t2x2",
                            CCMode::Full, NppiSize{8, 6}, NppiSize{2, 2}, 3, 0x402,
                            &nppiCrossCorrFull_Norm_8u32f_C3R_Ctx);
  corr_plain<Npp8s, Npp32f>("nppiCrossCorrFull_Norm_8s32f_AC4R_Ctx", "xcorr/full/norm/8s32f/AC4R_6x5_t2x2",
                            CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 4, 0x403,
                            &nppiCrossCorrFull_Norm_8s32f_AC4R_Ctx);
  corr_plain<Npp16u, Npp32f>("nppiCrossCorrFull_Norm_16u32f_C4R_Ctx", "xcorr/full/norm/16u32f/C4R_17x13_t3x3",
                             CCMode::Full, NppiSize{17, 13}, NppiSize{3, 3}, 4, 0x404,
                             &nppiCrossCorrFull_Norm_16u32f_C4R_Ctx);
  Knobs<Npp64f> k64;
  k64.rtol = k64.atol = 1e-9;
  corr_plain<Npp64f, Npp64f>("nppiCrossCorrFull_Norm_64f_C1R_Ctx", "xcorr/full/norm/64f/C1R_9x7_t3x3", CCMode::Full,
                             NppiSize{9, 7}, NppiSize{3, 3}, 1, 0x405, &nppiCrossCorrFull_Norm_64f_C1R_Ctx, k64);
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/full/norm/32f/C1R_1x1_t1x1", CCMode::Full,
                             NppiSize{1, 1}, NppiSize{1, 1}, 1, 0x406, &nppiCrossCorrFull_Norm_32f_C1R_Ctx);
  // 8s source with a 1x1 template: gamma in {-1, +1, NaN(0/0)} per pixel.
  corr_plain<Npp8s, Npp32f>("nppiCrossCorrFull_Norm_8s32f_C1R_Ctx", "xcorr/full/norm/8s32f/C1R_6x5_t1x1",
                            CCMode::Full, NppiSize{6, 5}, NppiSize{1, 1}, 1, 0x407,
                            &nppiCrossCorrFull_Norm_8s32f_C1R_Ctx);
  // 8u Sfs: trunc(value)*2^(-sf) clamped to [0,255]; fractional gamma -> 0.
  corr_sfs<Npp8u>("nppiCrossCorrFull_Norm_8u_C1RSfs_Ctx", "xcorr/full/norm/8u/C1RSfs_sf0_6x5_t2x2", CCMode::Full,
                  NppiSize{6, 5}, NppiSize{2, 2}, 1, 0, 0xA05, &nppiCrossCorrFull_Norm_8u_C1RSfs_Ctx);
  corr_sfs<Npp8u>("nppiCrossCorrFull_Norm_8u_C3RSfs_Ctx", "xcorr/full/norm/8u/C3RSfs_sf2_6x5_t2x2", CCMode::Full,
                  NppiSize{6, 5}, NppiSize{2, 2}, 3, 2, 0xA07, &nppiCrossCorrFull_Norm_8u_C3RSfs_Ctx);
}

// ============================ CrossCorrSame_Norm ============================
TEST_F(NppiCrossCorrDiff, Same_Norm) {
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrSame_Norm_32f_C1R_Ctx", "xcorr/same/norm/32f/C1R_17x13_t3x3",
                             CCMode::Same, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x411,
                             &nppiCrossCorrSame_Norm_32f_C1R_Ctx);
  corr_plain<Npp8u, Npp32f>("nppiCrossCorrSame_Norm_8u32f_C3R_Ctx", "xcorr/same/norm/8u32f/C3R_8x6_t2x2",
                            CCMode::Same, NppiSize{8, 6}, NppiSize{2, 2}, 3, 0x412,
                            &nppiCrossCorrSame_Norm_8u32f_C3R_Ctx);
  corr_plain<Npp16u, Npp32f>("nppiCrossCorrSame_Norm_16u32f_AC4R_Ctx", "xcorr/same/norm/16u32f/AC4R_6x5_t2x2",
                             CCMode::Same, NppiSize{6, 5}, NppiSize{2, 2}, 4, 0x413,
                             &nppiCrossCorrSame_Norm_16u32f_AC4R_Ctx);
  corr_plain<Npp8s, Npp32f>("nppiCrossCorrSame_Norm_8s32f_C1R_Ctx", "xcorr/same/norm/8s32f/C1R_9x7_t3x3",
                            CCMode::Same, NppiSize{9, 7}, NppiSize{3, 3}, 1, 0x414,
                            &nppiCrossCorrSame_Norm_8s32f_C1R_Ctx);
  Knobs<Npp64f> k64;
  k64.rtol = k64.atol = 1e-9;
  corr_plain<Npp64f, Npp64f>("nppiCrossCorrSame_Norm_64f_C3R_Ctx", "xcorr/same/norm/64f/C3R_9x7_t3x3", CCMode::Same,
                             NppiSize{9, 7}, NppiSize{3, 3}, 3, 0x415, &nppiCrossCorrSame_Norm_64f_C3R_Ctx, k64);
  // Even 4x4 template: window origin (c-2, r-2) (integer division).
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrSame_Norm_32f_C1R_Ctx", "xcorr/same/norm/32f/C1R_9x7_t4x4", CCMode::Same,
                             NppiSize{9, 7}, NppiSize{4, 4}, 1, 0x416, &nppiCrossCorrSame_Norm_32f_C1R_Ctx);
  corr_sfs<Npp8u>("nppiCrossCorrSame_Norm_8u_C1RSfs_Ctx", "xcorr/same/norm/8u/C1RSfs_sfm8_6x5_t2x2", CCMode::Same,
                  NppiSize{6, 5}, NppiSize{2, 2}, 1, -8, 0xA11, &nppiCrossCorrSame_Norm_8u_C1RSfs_Ctx);
}

// ============================ CrossCorrValid_Norm ===========================
TEST_F(NppiCrossCorrDiff, Valid_Norm) {
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/valid/norm/32f/C1R_17x13_t3x3",
                             CCMode::Valid, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x421,
                             &nppiCrossCorrValid_Norm_32f_C1R_Ctx);
  corr_plain<Npp8u, Npp32f>("nppiCrossCorrValid_Norm_8u32f_C4R_Ctx", "xcorr/valid/norm/8u32f/C4R_8x6_t3x3",
                            CCMode::Valid, NppiSize{8, 6}, NppiSize{3, 3}, 4, 0x422,
                            &nppiCrossCorrValid_Norm_8u32f_C4R_Ctx);
  corr_plain<Npp8s, Npp32f>("nppiCrossCorrValid_Norm_8s32f_C1R_Ctx", "xcorr/valid/norm/8s32f/C1R_6x5_t2x2",
                            CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x423,
                            &nppiCrossCorrValid_Norm_8s32f_C1R_Ctx);
  corr_plain<Npp16u, Npp32f>("nppiCrossCorrValid_Norm_16u32f_C3R_Ctx", "xcorr/valid/norm/16u32f/C3R_6x5_t2x2",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 3, 0x424,
                             &nppiCrossCorrValid_Norm_16u32f_C3R_Ctx);
  Knobs<Npp64f> k64;
  k64.rtol = k64.atol = 1e-9;
  corr_plain<Npp64f, Npp64f>("nppiCrossCorrValid_Norm_64f_C1R_Ctx", "xcorr/valid/norm/64f/C1R_9x7_t3x3",
                             CCMode::Valid, NppiSize{9, 7}, NppiSize{3, 3}, 1, 0x425,
                             &nppiCrossCorrValid_Norm_64f_C1R_Ctx, k64);
  // Exact template match: gamma = 1 at dst(0,0) exactly (Rst = Rss = Rtt),
  // fractional elsewhere -> Sfs output 1 at sf 0 and 255 at sf -8.
  const std::vector<Npp8u> exactSrc{3, 17, 9, 14, 6, 11, 2, 19, 15, 8, 12, 5, 1, 16, 7, 10};
  const std::vector<Npp8u> exactTpl{3, 17, 6, 11};
  Knobs<Npp8u> kExact;
  kExact.srcData = &exactSrc;
  kExact.tplData = &exactTpl;
  corr_sfs<Npp8u>("nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx", "xcorr/valid/norm/8u/C1RSfs_exact_sf0", CCMode::Valid,
                  NppiSize{4, 4}, NppiSize{2, 2}, 1, 0, 0xA21, &nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx, kExact);
  corr_sfs<Npp8u>("nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx", "xcorr/valid/norm/8u/C1RSfs_exact_sfm8", CCMode::Valid,
                  NppiSize{4, 4}, NppiSize{2, 2}, 1, -8, 0xA21, &nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx, kExact);
}

// ============================ CrossCorrValid (plain) ========================
TEST_F(NppiCrossCorrDiff, Valid_Plain) {
  corr_plain<Npp8u, Npp32f>("nppiCrossCorrValid_8u32f_C1R_Ctx", "xcorr/valid/plain/8u32f/C1R_8x6_t2x2",
                            CCMode::Valid, NppiSize{8, 6}, NppiSize{2, 2}, 1, 0x431,
                            &nppiCrossCorrValid_8u32f_C1R_Ctx);
  corr_plain<Npp8s, Npp32f>("nppiCrossCorrValid_8s32f_C1R_Ctx", "xcorr/valid/plain/8s32f/C1R_6x5_t2x2",
                            CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x432,
                            &nppiCrossCorrValid_8s32f_C1R_Ctx);
  corr_plain<Npp16u, Npp32f>("nppiCrossCorrValid_16u32f_C1R_Ctx", "xcorr/valid/plain/16u32f/C1R_8x6_t3x3",
                             CCMode::Valid, NppiSize{8, 6}, NppiSize{3, 3}, 1, 0x433,
                             &nppiCrossCorrValid_16u32f_C1R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_32f_C1R_Ctx", "xcorr/valid/plain/32f/C1R_17x13_t3x3",
                             CCMode::Valid, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x434,
                             &nppiCrossCorrValid_32f_C1R_Ctx);
  Knobs<Npp64f> k64;
  k64.rtol = k64.atol = 1e-9;
  corr_plain<Npp64f, Npp64f>("nppiCrossCorrValid_64f_C1R_Ctx", "xcorr/valid/plain/64f/C1R_9x7_t3x3", CCMode::Valid,
                             NppiSize{9, 7}, NppiSize{3, 3}, 1, 0x435, &nppiCrossCorrValid_64f_C1R_Ctx, k64);
  // Template equal to the source: single 1x1 dst window.
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_32f_C1R_Ctx", "xcorr/valid/plain/32f/C1R_5x4_t5x4", CCMode::Valid,
                             NppiSize{5, 4}, NppiSize{5, 4}, 1, 0x436, &nppiCrossCorrValid_32f_C1R_Ctx);
}

// ============================ NormLevel =====================================
TEST_F(NppiCrossCorrDiff, NormLevel) {
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevel_32f_C1R_Ctx", "xcorr/full/nl/32f/C1R_17x13_t3x3",
                          CCMode::Full, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x441,
                          &nppiCrossCorrFull_NormLevel_32f_C1R_Ctx, scratch());
  corr_nl<Npp8u, Npp32f>("nppiCrossCorrFull_NormLevel_8u32f_C3R_Ctx", "xcorr/full/nl/8u32f/C3R_6x5_t2x2",
                         CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 3, 0x442,
                         &nppiCrossCorrFull_NormLevel_8u32f_C3R_Ctx, scratch());
  Knobs<Npp64f> k64;
  k64.rtol = k64.atol = 1e-9;
  corr_nl<Npp64f, Npp64f>("nppiCrossCorrFull_NormLevel_64f_C1R_Ctx", "xcorr/full/nl/64f/C1R_9x7_t3x3", CCMode::Full,
                          NppiSize{9, 7}, NppiSize{3, 3}, 1, 0x443, &nppiCrossCorrFull_NormLevel_64f_C1R_Ctx,
                          scratch(), k64);
  corr_nl<Npp16u, Npp32f>("nppiCrossCorrSame_NormLevel_16u32f_C4R_Ctx", "xcorr/same/nl/16u32f/C4R_8x6_t2x2",
                          CCMode::Same, NppiSize{8, 6}, NppiSize{2, 2}, 4, 0x444,
                          &nppiCrossCorrSame_NormLevel_16u32f_C4R_Ctx, scratch());
  // Fractional (incl. negative) ZNCC gamma -> Sfs output 0 at any sf.
  corr_sfsnl<Npp8u>("nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx", "xcorr/same/nl/8u/C1RSfs_sfm8_6x5_t2x2",
                    CCMode::Same, NppiSize{6, 5}, NppiSize{2, 2}, 1, -8, 0x445,
                    &nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx, scratch());
  corr_nl<Npp8s, Npp32f>("nppiCrossCorrValid_NormLevel_8s32f_AC4R_Ctx", "xcorr/valid/nl/8s32f/AC4R_6x5_t2x2",
                         CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 4, 0x446,
                         &nppiCrossCorrValid_NormLevel_8s32f_AC4R_Ctx, scratch());
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrValid_NormLevel_32f_C1R_Ctx", "xcorr/valid/nl/32f/C1R_17x13_t3x3",
                          CCMode::Valid, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x447,
                          &nppiCrossCorrValid_NormLevel_32f_C1R_Ctx, scratch());
}

// ============================ NormLevelAdvanced =============================
TEST_F(NppiCrossCorrDiff, NormLevelAdvanced) {
  corr_adv<Npp8u, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_8u32f_C1R_Ctx", "xcorr/full/adv/8u32f/C1R_6x5_t2x2",
                          CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x451,
                          &nppiCrossCorrFull_NormLevelAdvanced_8u32f_C1R_Ctx, scratch(), scratch());
  corr_adv<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_32f_C3R_Ctx", "xcorr/full/adv/32f/C3R_8x6_t2x2",
                           CCMode::Full, NppiSize{8, 6}, NppiSize{2, 2}, 3, 0x452,
                           &nppiCrossCorrFull_NormLevelAdvanced_32f_C3R_Ctx, scratch(), scratch());
  Knobs<Npp64f> k64;
  k64.rtol = k64.atol = 1e-9;
  corr_adv<Npp64f, Npp64f>("nppiCrossCorrFull_NormLevelAdvanced_64f_C1R_Ctx", "xcorr/full/adv/64f/C1R_9x7_t3x3",
                           CCMode::Full, NppiSize{9, 7}, NppiSize{3, 3}, 1, 0x453,
                           &nppiCrossCorrFull_NormLevelAdvanced_64f_C1R_Ctx, scratch(), scratch(), k64);
  corr_adv<Npp16u, Npp32f>("nppiCrossCorrSame_NormLevelAdvanced_16u32f_C1R_Ctx", "xcorr/same/adv/16u32f/C1R_8x6_t2x2",
                           CCMode::Same, NppiSize{8, 6}, NppiSize{2, 2}, 1, 0x454,
                           &nppiCrossCorrSame_NormLevelAdvanced_16u32f_C1R_Ctx, scratch(), scratch());
  corr_adv<Npp8s, Npp32f>("nppiCrossCorrSame_NormLevelAdvanced_8s32f_C3R_Ctx", "xcorr/same/adv/8s32f/C3R_6x5_t2x2",
                          CCMode::Same, NppiSize{6, 5}, NppiSize{2, 2}, 3, 0x455,
                          &nppiCrossCorrSame_NormLevelAdvanced_8s32f_C3R_Ctx, scratch(), scratch());
  corr_adv<Npp32f, Npp32f>("nppiCrossCorrValid_NormLevelAdvanced_32f_C1R_Ctx", "xcorr/valid/adv/32f/C1R_17x13_t3x3",
                           CCMode::Valid, NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x456,
                           &nppiCrossCorrValid_NormLevelAdvanced_32f_C1R_Ctx, scratch(), scratch());
  corr_adv<Npp8u, Npp32f>("nppiCrossCorrValid_NormLevelAdvanced_8u32f_C4R_Ctx", "xcorr/valid/adv/8u32f/C4R_8x6_t3x3",
                          CCMode::Valid, NppiSize{8, 6}, NppiSize{3, 3}, 4, 0x457,
                          &nppiCrossCorrValid_NormLevelAdvanced_8u32f_C4R_Ctx, scratch(), scratch());
  corr_adv<Npp64f, Npp64f>("nppiCrossCorrValid_NormLevelAdvanced_64f_C3R_Ctx", "xcorr/valid/adv/64f/C3R_9x7_t3x3",
                           CCMode::Valid, NppiSize{9, 7}, NppiSize{3, 3}, 3, 0x458,
                           &nppiCrossCorrValid_NormLevelAdvanced_64f_C3R_Ctx, scratch(), scratch(), k64);
}

// ============================ SqrDistance_Norm ==============================
TEST_F(NppiCrossCorrDiff, SqrDistance_Norm) {
  corr_sfs<Npp8u>("nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx", "xcorr/full/sd/8u/C1RSfs_sf0_6x5_t2x2", CCMode::Full,
                  NppiSize{6, 5}, NppiSize{2, 2}, 1, 0, 0xA01, &nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx);
  corr_plain<Npp16u, Npp32f>("nppiSqrDistanceFull_Norm_16u32f_C3R_Ctx", "xcorr/full/sd/16u32f/C3R_6x5_t2x2",
                             CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 3, 0x462,
                             &nppiSqrDistanceFull_Norm_16u32f_C3R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiSqrDistanceFull_Norm_32f_AC4R_Ctx", "xcorr/full/sd/32f/AC4R_6x5_t2x2",
                             CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 4, 0x463,
                             &nppiSqrDistanceFull_Norm_32f_AC4R_Ctx);
  corr_sfs<Npp8u>("nppiSqrDistanceSame_Norm_8u_C1RSfs_Ctx", "xcorr/same/sd/8u/C1RSfs_sfm2_6x5_t2x2", CCMode::Same,
                  NppiSize{6, 5}, NppiSize{2, 2}, 1, -2, 0xA09, &nppiSqrDistanceSame_Norm_8u_C1RSfs_Ctx);
  corr_plain<Npp8u, Npp32f>("nppiSqrDistanceSame_Norm_8u32f_C4R_Ctx", "xcorr/same/sd/8u32f/C4R_8x6_t2x2",
                            CCMode::Same, NppiSize{8, 6}, NppiSize{2, 2}, 4, 0x465,
                            &nppiSqrDistanceSame_Norm_8u32f_C4R_Ctx);
  corr_sfs<Npp8u>("nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx", "xcorr/valid/sd/8u/C1RSfs_sf0_8x6_t3x3", CCMode::Valid,
                  NppiSize{8, 6}, NppiSize{3, 3}, 1, 0, 0xA03, &nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx);
  corr_plain<Npp8s, Npp32f>("nppiSqrDistanceValid_Norm_8s32f_C1R_Ctx", "xcorr/valid/sd/8s32f/C1R_6x5_t2x2",
                            CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x467,
                            &nppiSqrDistanceValid_Norm_8s32f_C1R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiSqrDistanceValid_Norm_32f_C3R_Ctx", "xcorr/valid/sd/32f/C3R_9x7_t3x3",
                             CCMode::Valid, NppiSize{9, 7}, NppiSize{3, 3}, 3, 0x468,
                             &nppiSqrDistanceValid_Norm_32f_C3R_Ctx);
}

// ==================== zero variance / zero denominator ======================
// NormLevel zero-variance windows write a whole-image sentinel (FLT_MAX /
// DBL_MAX / 255); Norm 0/0 -> NaN; SqrDistance zero src or tpl -> +inf, both
// zero -> NaN (8u Sfs maps NaN -> 0 and +inf -> 255).
TEST_F(NppiCrossCorrDiff, ZeroVariance_And_ZeroDenominator) {
  const std::vector<Npp32f> constSrcF(30, 5.0f);
  Knobs<Npp32f> kSrcF;
  kSrcF.srcData = &constSrcF;
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevel_32f_C1R_Ctx", "xcorr/full/nl/32f/C1R_constsrc_6x5_t2x2",
                          CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x471,
                          &nppiCrossCorrFull_NormLevel_32f_C1R_Ctx, scratch(), kSrcF);
  const std::vector<Npp64f> constTplD(4, 7.0);
  Knobs<Npp64f> kTplD;
  kTplD.tplData = &constTplD;
  kTplD.rtol = kTplD.atol = 1e-9;
  corr_nl<Npp64f, Npp64f>("nppiCrossCorrFull_NormLevel_64f_C1R_Ctx", "xcorr/full/nl/64f/C1R_consttpl_6x5_t2x2",
                          CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x472,
                          &nppiCrossCorrFull_NormLevel_64f_C1R_Ctx, scratch(), kTplD);
  const std::vector<Npp8u> constSrc8(30, 5);
  Knobs<Npp8u> kSrc8;
  kSrc8.srcData = &constSrc8;
  corr_sfsnl<Npp8u>("nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx", "xcorr/same/nl/8u/C1RSfs_constsrc_sf0_6x5_t2x2",
                    CCMode::Same, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0, 0x473,
                    &nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx, scratch(), kSrc8);
  const std::vector<Npp16u> constTpl16(4, 7);
  Knobs<Npp16u> kTpl16;
  kTpl16.tplData = &constTpl16;
  corr_nl<Npp16u, Npp32f>("nppiCrossCorrValid_NormLevel_16u32f_C1R_Ctx",
                          "xcorr/valid/nl/16u32f/C1R_consttpl_6x5_t2x2", CCMode::Valid, NppiSize{6, 5},
                          NppiSize{2, 2}, 1, 0x474, &nppiCrossCorrValid_NormLevel_16u32f_C1R_Ctx, scratch(), kTpl16);

  const std::vector<Npp32f> zeroSrcF(30, 0.0f), zeroTplF(4, 0.0f);
  Knobs<Npp32f> kzSrc;
  kzSrc.srcData = &zeroSrcF;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/full/norm/32f/C1R_zerosrc_6x5_t2x2",
                             CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x475,
                             &nppiCrossCorrFull_Norm_32f_C1R_Ctx, kzSrc);
  Knobs<Npp32f> kzTpl;
  kzTpl.tplData = &zeroTplF;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrSame_Norm_32f_C1R_Ctx", "xcorr/same/norm/32f/C1R_zerotpl_6x5_t2x2",
                             CCMode::Same, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x476,
                             &nppiCrossCorrSame_Norm_32f_C1R_Ctx, kzTpl);
  corr_plain<Npp32f, Npp32f>("nppiSqrDistanceValid_Norm_32f_C1R_Ctx", "xcorr/valid/sd/32f/C1R_zerosrc_6x5_t2x2",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x477,
                             &nppiSqrDistanceValid_Norm_32f_C1R_Ctx, kzSrc);
  corr_plain<Npp32f, Npp32f>("nppiSqrDistanceValid_Norm_32f_C1R_Ctx", "xcorr/valid/sd/32f/C1R_zerotpl_6x5_t2x2",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x478,
                             &nppiSqrDistanceValid_Norm_32f_C1R_Ctx, kzTpl);
  Knobs<Npp32f> kzBoth;
  kzBoth.srcData = &zeroSrcF;
  kzBoth.tplData = &zeroTplF;
  corr_plain<Npp32f, Npp32f>("nppiSqrDistanceValid_Norm_32f_C1R_Ctx", "xcorr/valid/sd/32f/C1R_bothzero_6x5_t2x2",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x479,
                             &nppiSqrDistanceValid_Norm_32f_C1R_Ctx, kzBoth);
  const std::vector<Npp8u> zeroSrc8(30, 0), zeroTpl8(4, 0);
  Knobs<Npp8u> kzSrc8;
  kzSrc8.srcData = &zeroSrc8;
  corr_sfs<Npp8u>("nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx", "xcorr/valid/sd/8u/C1RSfs_zerosrc_sf0_6x5_t2x2",
                  CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0, 0x47A,
                  &nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx, kzSrc8);
  Knobs<Npp8u> kzBoth8;
  kzBoth8.srcData = &zeroSrc8;
  kzBoth8.tplData = &zeroTpl8;
  corr_sfs<Npp8u>("nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx", "xcorr/valid/sd/8u/C1RSfs_bothzero_sf0_6x5_t2x2",
                  CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0, 0x47B,
                  &nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx, kzBoth8);
}

// ======================== contracts: null / ROI =============================
// Measured order (5090): null src/tpl -> null src ROI (zero: no-op,
// negative: -6) -> null dst -> src step -> [Valid: empty dst no-op] -> dst
// step -> tpl step -> [Full/Same: tpl larger than src -> -6].
TEST_F(NppiCrossCorrDiff, Contract_NullAndROI) {
  Knobs<Npp32f> kNullSrc;
  kNullSrc.nullSrc = true;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/nullsrc", CCMode::Full,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x501, &nppiCrossCorrFull_Norm_32f_C1R_Ctx, kNullSrc);
  Knobs<Npp32f> kNullTpl;
  kNullTpl.nullTpl = true;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrSame_Norm_32f_C1R_Ctx", "xcorr/contract/nulltpl", CCMode::Same,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x502, &nppiCrossCorrSame_Norm_32f_C1R_Ctx, kNullTpl);
  Knobs<Npp32f> kNullDst;
  kNullDst.nullDst = true;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/nulldst", CCMode::Valid,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x503, &nppiCrossCorrValid_Norm_32f_C1R_Ctx, kNullDst);
  // null src wins over a negative src ROI.
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/nullsrc_negroi", CCMode::Full,
                             NppiSize{-1, 5}, NppiSize{2, 2}, 1, 0x504, &nppiCrossCorrFull_Norm_32f_C1R_Ctx, kNullSrc);
  // null dst wins over a bad src step.
  Knobs<Npp32f> kNullDstStep;
  kNullDstStep.nullDst = true;
  kNullDstStep.srcStepOv = 1;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/nulldst_badsrcstep", CCMode::Full,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x505, &nppiCrossCorrFull_Norm_32f_C1R_Ctx,
                             kNullDstStep);
  // Zero src/tpl ROI: NO_ERROR, no writes (dst keeps the sentinel pattern).
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/zero_roi_src", CCMode::Full,
                             NppiSize{0, 5}, NppiSize{2, 2}, 1, 0x506, &nppiCrossCorrFull_Norm_32f_C1R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrSame_Norm_32f_C1R_Ctx", "xcorr/contract/zero_roi_tpl", CCMode::Same,
                             NppiSize{6, 5}, NppiSize{0, 0}, 1, 0x507, &nppiCrossCorrSame_Norm_32f_C1R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/zero_roi_src_valid",
                             CCMode::Valid, NppiSize{0, 5}, NppiSize{2, 2}, 1, 0x508,
                             &nppiCrossCorrValid_Norm_32f_C1R_Ctx);
  // Negative src/tpl ROI -> -6 (before the step checks).
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/neg_roi_src", CCMode::Valid,
                             NppiSize{-1, 5}, NppiSize{2, 2}, 1, 0x509, &nppiCrossCorrValid_Norm_32f_C1R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/neg_roi_tpl", CCMode::Full,
                             NppiSize{6, 5}, NppiSize{4, -3}, 1, 0x50A, &nppiCrossCorrFull_Norm_32f_C1R_Ctx);
  Knobs<Npp32f> kNegRoiStep;
  kNegRoiStep.srcStepOv = 1;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/negroi_beats_step", CCMode::Full,
                             NppiSize{-1, 5}, NppiSize{2, 2}, 1, 0x50B, &nppiCrossCorrFull_Norm_32f_C1R_Ctx,
                             kNegRoiStep);
  // Full/Same: template larger than the source -> -6; Valid: empty dst ->
  // NO_ERROR without writes (dst/tpl steps are not validated in that case).
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/full_tpl_gt_src", CCMode::Full,
                             NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x50C, &nppiCrossCorrFull_Norm_32f_C1R_Ctx);
  corr_plain<Npp32f, Npp32f>("nppiSqrDistanceSame_Norm_32f_C1R_Ctx", "xcorr/contract/same_tpl_gt_src", CCMode::Same,
                             NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x50D, &nppiSqrDistanceSame_Norm_32f_C1R_Ctx);
  Knobs<Npp32f> kEmptySteps;
  kEmptySteps.dstStepOv = 3;  // misaligned for 32f - must NOT be validated
  kEmptySteps.tplStepOv = 3;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/valid_tpl_gt_src", CCMode::Valid,
                             NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x50E, &nppiCrossCorrValid_Norm_32f_C1R_Ctx,
                             kEmptySteps);
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/valid_tpl_taller_src",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 7}, 1, 0x50F,
                             &nppiCrossCorrValid_Norm_32f_C1R_Ctx, kEmptySteps);
  // Valid empty dst happens after the src step check.
  Knobs<Npp32f> kEmptySrcStep;
  kEmptySrcStep.srcStepOv = 1;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/valid_empty_badsrcstep",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x510,
                             &nppiCrossCorrValid_Norm_32f_C1R_Ctx, kEmptySrcStep);
}

// ========================== contracts: steps ================================
// Step checks: (unsigned)step < width*ch*sizeof(S) -> -16 (size), step %
// sizeof(S) -> -108 (alignment); size is checked before alignment; 8u has no
// alignment requirement; src before dst before tpl; the Full/Same tpl>src
// size error comes last.
TEST_F(NppiCrossCorrDiff, Contract_Steps) {
  Knobs<Npp8u> kSrcSmall8;
  kSrcSmall8.srcStepOv = 5;  // src 6x5 C1R needs >= 6
  corr_plain<Npp8u, Npp32f>("nppiCrossCorrFull_Norm_8u32f_C1R_Ctx", "xcorr/contract/src_step_small_8u", CCMode::Full,
                            NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x521, &nppiCrossCorrFull_Norm_8u32f_C1R_Ctx,
                            kSrcSmall8);
  Knobs<Npp16u> kSrcMis16;
  kSrcMis16.srcStepOv = 13;  // src 6x5 C1R needs >= 12, 13 % 2 != 0
  corr_nl<Npp16u, Npp32f>("nppiCrossCorrValid_NormLevel_16u32f_C1R_Ctx", "xcorr/contract/src_step_misalign_16u",
                          CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x522,
                          &nppiCrossCorrValid_NormLevel_16u32f_C1R_Ctx, scratch(), kSrcMis16);
  Knobs<Npp16u> kSrcBoth16;
  kSrcBoth16.srcStepOv = 11;  // too small AND misaligned -> -16, not -108
  corr_nl<Npp16u, Npp32f>("nppiCrossCorrValid_NormLevel_16u32f_C1R_Ctx", "xcorr/contract/src_step_size_beats_align",
                          CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x523,
                          &nppiCrossCorrValid_NormLevel_16u32f_C1R_Ctx, scratch(), kSrcBoth16);
  Knobs<Npp32f> kDstSmall;
  kDstSmall.dstStepOv = 20;  // Same dst 6x5 C1R needs >= 24
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrSame_Norm_32f_C1R_Ctx", "xcorr/contract/dst_step_small", CCMode::Same,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x524, &nppiCrossCorrSame_Norm_32f_C1R_Ctx, kDstSmall);
  Knobs<Npp32f> kDstMis;
  kDstMis.dstStepOv = 30;  // Full dst 7x6 C1R needs >= 28, 30 % 4 != 0
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevel_32f_C1R_Ctx", "xcorr/contract/dst_step_misalign", CCMode::Full,
                          NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x525, &nppiCrossCorrFull_NormLevel_32f_C1R_Ctx,
                          scratch(), kDstMis);
  Knobs<Npp16u> kTplSmall;
  kTplSmall.tplStepOv = 3;  // tpl 2x2 C1R needs >= 4
  corr_plain<Npp16u, Npp32f>("nppiSqrDistanceValid_Norm_16u32f_C1R_Ctx", "xcorr/contract/tpl_step_small",
                             CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x526,
                             &nppiSqrDistanceValid_Norm_16u32f_C1R_Ctx, kTplSmall);
  Knobs<Npp16u> kTplMis;
  kTplMis.tplStepOv = 5;  // >= 4 but 5 % 2 != 0
  corr_plain<Npp16u, Npp32f>("nppiSqrDistanceFull_Norm_16u32f_C1R_Ctx", "xcorr/contract/tpl_step_misalign",
                             CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x527,
                             &nppiSqrDistanceFull_Norm_16u32f_C1R_Ctx, kTplMis);
  // dst step is checked before the tpl step: dst misalign (-108) wins over
  // the tpl size error (-16).
  Knobs<Npp32f> kDstBeatsTpl;
  kDstBeatsTpl.dstStepOv = 26;  // Valid dst 5x4 C1R: >= 20, 26 % 4 != 0
  kDstBeatsTpl.tplStepOv = 3;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrValid_Norm_32f_C1R_Ctx", "xcorr/contract/dst_beats_tpl", CCMode::Valid,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x528, &nppiCrossCorrValid_Norm_32f_C1R_Ctx,
                             kDstBeatsTpl);
  // src step is checked before the dst step.
  Knobs<Npp32f> kSrcBeatsDst;
  kSrcBeatsDst.srcStepOv = 3;
  kSrcBeatsDst.dstStepOv = 26;
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/src_beats_dst", CCMode::Full,
                             NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x529, &nppiCrossCorrFull_Norm_32f_C1R_Ctx,
                             kSrcBeatsDst);
  // The Full/Same tpl>src size error comes after the dst step check.
  Knobs<Npp32f> kTplSizeLast;
  kTplSizeLast.dstStepOv = 50;  // Full dst 12x9 C1R: >= 48, 50 % 4 != 0
  corr_plain<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R_Ctx", "xcorr/contract/tplsize_after_steps", CCMode::Full,
                             NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x52A, &nppiCrossCorrFull_Norm_32f_C1R_Ctx,
                             kTplSizeLast);
  // 8u has no alignment requirement: odd steps are accepted.
  Knobs<Npp8u> kOdd8;
  kOdd8.srcPad = 7;  // srcStep = 6 + 7 = 13 (odd)
  kOdd8.tplPad = 3;  // tplStep = 2 + 3 = 5 (odd)
  corr_plain<Npp8u, Npp32f>("nppiCrossCorrFull_Norm_8u32f_C1R_Ctx", "xcorr/contract/8u_odd_steps", CCMode::Full,
                            NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x52B, &nppiCrossCorrFull_Norm_8u32f_C1R_Ctx, kOdd8);
}

// ========================= contracts: buffers ===============================
// NormLevel (non-Advanced) validates pDeviceBuffer LAST (a tpl>src size error
// wins); NormLevelAdvanced validates both buffers FIRST (a null buffer wins
// over ROI/step errors).
TEST_F(NppiCrossCorrDiff, Contract_Buffers) {
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevel_32f_C1R_Ctx", "xcorr/contract/nl_nullbuf", CCMode::Full,
                          NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x541, &nppiCrossCorrFull_NormLevel_32f_C1R_Ctx,
                          nullptr);
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevel_32f_C1R_Ctx", "xcorr/contract/nl_nullbuf_tplbig",
                          CCMode::Full, NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x542,
                          &nppiCrossCorrFull_NormLevel_32f_C1R_Ctx, nullptr);
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrValid_NormLevel_32f_C1R_Ctx", "xcorr/contract/nl_nullbuf_valid",
                          CCMode::Valid, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x543,
                          &nppiCrossCorrValid_NormLevel_32f_C1R_Ctx, nullptr);
  corr_adv<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx", "xcorr/contract/adv_nulldev",
                           CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x544,
                           &nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx, nullptr, scratch());
  corr_adv<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx", "xcorr/contract/adv_nullscratch",
                           CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x545,
                           &nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx, scratch(), nullptr);
  corr_adv<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx", "xcorr/contract/adv_nullscratch_tplbig",
                           CCMode::Full, NppiSize{6, 5}, NppiSize{7, 5}, 1, 0x546,
                           &nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx, scratch(), nullptr);
  Knobs<Npp32f> kAdvStep;
  kAdvStep.srcStepOv = 1;
  corr_adv<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx", "xcorr/contract/adv_nulldev_badsrcstep",
                           CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x547,
                           &nppiCrossCorrFull_NormLevelAdvanced_32f_C1R_Ctx, nullptr, scratch(), kAdvStep);
  corr_sfsnl<Npp8u>("nppiCrossCorrValid_NormLevel_8u_C1RSfs_Ctx", "xcorr/contract/nlsfs_nullbuf", CCMode::Valid,
                    NppiSize{6, 5}, NppiSize{2, 2}, 1, 0, 0x548, &nppiCrossCorrValid_NormLevel_8u_C1RSfs_Ctx,
                    nullptr);
}

// ==================== GetAdvancedScratchBufferSize ==========================
// Measured formula (5090): size = 16 + 2*dstArea*sizeof(D)*srcChannels where
// dstArea is the Full dst area ((Ws+tw-1)*(Hs+th-1)); Same and Valid use
// Ws*Hs and ignore the template completely.  Full rejects negative template
// dimensions; zero dimensions flow through the arithmetic.
TEST_F(NppiCrossCorrDiff, ScratchBufferSize_Values) {
  size_t sz = 0;
  {
    Case c("xcorr/scratchsize/full/basic", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/full/ch3", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{17, 13}, NppiSize{3, 3}, 4, 3, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/full/sz8", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 8, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/full/zerotpl", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{0, 0}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/full/zerosrc", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{0, 0}, NppiSize{2, 2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/valid/basic", "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize{8, 6}, NppiSize{3, 3}, 4, 4, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    // Valid ignores the template, even with negative dimensions.
    Case c("xcorr/scratchsize/valid/negtpl", "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, -2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/same/basic", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    // Same ignores the template, even with negative dimensions.
    Case c("xcorr/scratchsize/same/negtpl", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{-3, 2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/same/zerosrc", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{0, 0}, NppiSize{2, 2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/same/ch2", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 2, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
}

TEST_F(NppiCrossCorrDiff, ScratchBufferSize_Errors) {
  size_t sz = 0x5A5A5A5A5A5A5A5AULL;
  {
    Case c("xcorr/scratchsize/err/null_full", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 1, static_cast<size_t *>(nullptr));
    c.status(st);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/err/null_valid", "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 1, static_cast<size_t *>(nullptr));
    c.status(st);
    c.finish();
  }
  {
    // Full rejects negative template dimensions (-6); Same/Valid do not.
    Case c("xcorr/scratchsize/err/full_negtpl", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, -2}, 4, 1, &sz);
    c.status(st);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/err/negsrc", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{-4, 3}, NppiSize{2, 2}, 4, 1, &sz);
    c.status(st);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/err/size0", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 0, 1, &sz);
    c.status(st);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/err/sizeneg", "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, -4, 1, &sz);
    c.status(st);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/err/ch0", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 0, &sz);
    c.status(st);
    c.finish();
  }
  {
    Case c("xcorr/scratchsize/err/ch5", "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize{4, 3}, NppiSize{2, 2}, 4, 5, &sz);
    c.status(st);
    c.finish();
  }
}

// ======================== buffer size sufficiency ===========================
// The Advanced scratch buffer must be usable at exactly the queried size, and
// the non-Advanced NormLevel device buffer at exactly the
// nppiFullNormLevelGetBufferHostSize size (the query itself is recorded as a
// differential case).
TEST_F(NppiCrossCorrDiff, Scratch_Sufficiency) {
  size_t sz = 0;
  {
    Case c("xcorr/scratchsize/sufficiency/full_query", "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize{6, 5}, NppiSize{2, 2}, 4, 1, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  ASSERT_GT(sz, 0u);
  DeviceMemory<Npp8u> &adv = arena<Npp8u>(sz);
  corr_adv<Npp8u, Npp32f>("nppiCrossCorrFull_NormLevelAdvanced_8u32f_C1R_Ctx",
                          "xcorr/full/adv/8u32f/C1R_exactscratch_6x5_t2x2", CCMode::Full, NppiSize{6, 5},
                          NppiSize{2, 2}, 1, 0x461, &nppiCrossCorrFull_NormLevelAdvanced_8u32f_C1R_Ctx, scratch(),
                          adv.get());
  {
    Case c("xcorr/scratchsize/sufficiency/same_query", "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize");
    NppStatus st = nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize{8, 6}, NppiSize{2, 2}, 4, 3, &sz);
    c.status(st);
    c.host_output(&sz, 1, ValueKind::U64);
    c.finish();
  }
  ASSERT_GT(sz, 0u);
  DeviceMemory<Npp8u> &advSame = arena<Npp8u>(sz);
  corr_adv<Npp16u, Npp32f>("nppiCrossCorrSame_NormLevelAdvanced_16u32f_C3R_Ctx",
                           "xcorr/same/adv/16u32f/C3R_exactscratch_8x6_t2x2", CCMode::Same, NppiSize{8, 6},
                           NppiSize{2, 2}, 3, 0x462, &nppiCrossCorrSame_NormLevelAdvanced_16u32f_C3R_Ctx, scratch(),
                           advSame.get());

  size_t need = 0;
  {
    Case c("xcorr/scratchsize/sufficiency/nl_hostsize", "nppiFullNormLevelGetBufferHostSize_32f_AC4R_Ctx");
    NppStatus st = nppiFullNormLevelGetBufferHostSize_32f_AC4R_Ctx(NppiSize{6, 5}, &need, current_context());
    c.status(st);
    c.host_output(&need, 1, ValueKind::U64);
    c.finish();
  }
  ASSERT_GT(need, 0);
  DeviceMemory<Npp8u> &nlBuf = arena<Npp8u>((size_t)need);
  corr_nl<Npp32f, Npp32f>("nppiCrossCorrFull_NormLevel_32f_AC4R_Ctx",
                          "xcorr/full/nl/32f/AC4R_exactbuf_6x5_t2x2", CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2},
                          4, 0x463, &nppiCrossCorrFull_NormLevel_32f_AC4R_Ctx, nlBuf.get());
}

#if NPP_DIFF_HAS_NONCTX
// Non-Ctx entry points reuse the Ctx goldens (identical data, steps and ids).
TEST_F(NppiCrossCorrDiff, NonCtx_Reuse) {
  corr_case<Npp32f, Npp32f>("nppiCrossCorrFull_Norm_32f_C1R", "xcorr/full/norm/32f/C1R_17x13_t3x3", CCMode::Full,
                            NppiSize{17, 13}, NppiSize{3, 3}, 1, 0x401,
                            [](const Npp32f *s, int ss, const Npp32f *t, int ts, Npp32f *d, int ds) {
                              return nppiCrossCorrFull_Norm_32f_C1R(s, ss, NppiSize{17, 13}, t, ts, NppiSize{3, 3}, d,
                                                                    ds);
                            });
  corr_case<Npp16u, Npp32f>("nppiCrossCorrValid_16u32f_C1R", "xcorr/valid/plain/16u32f/C1R_8x6_t3x3", CCMode::Valid,
                            NppiSize{8, 6}, NppiSize{3, 3}, 1, 0x433,
                            [](const Npp16u *s, int ss, const Npp16u *t, int ts, Npp32f *d, int ds) {
                              return nppiCrossCorrValid_16u32f_C1R(s, ss, NppiSize{8, 6}, t, ts, NppiSize{3, 3}, d,
                                                                    ds);
                            });
  corr_case<Npp8u, Npp8u>("nppiCrossCorrSame_NormLevel_8u_C1RSfs", "xcorr/same/nl/8u/C1RSfs_sfm8_6x5_t2x2",
                          CCMode::Same, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0x445,
                          [](const Npp8u *s, int ss, const Npp8u *t, int ts, Npp8u *d, int ds) {
                            return nppiCrossCorrSame_NormLevel_8u_C1RSfs(s, ss, NppiSize{6, 5}, t, ts, NppiSize{2, 2},
                                                                         d, ds, -8, scratch());
                          });
  corr_case<Npp8u, Npp8u>("nppiSqrDistanceFull_Norm_8u_C1RSfs", "xcorr/full/sd/8u/C1RSfs_sf0_6x5_t2x2",
                          CCMode::Full, NppiSize{6, 5}, NppiSize{2, 2}, 1, 0xA01,
                          [](const Npp8u *s, int ss, const Npp8u *t, int ts, Npp8u *d, int ds) {
                            return nppiSqrDistanceFull_Norm_8u_C1RSfs(s, ss, NppiSize{6, 5}, t, ts, NppiSize{2, 2}, d,
                                                                      ds, 0);
                          });
}
#endif
