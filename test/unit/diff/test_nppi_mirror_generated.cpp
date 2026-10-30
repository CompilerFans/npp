// Differential tests for package g3: nppiMirror (16s C1/C3/C4 and every AC4
// variant) and nppiMirrorBatch (32f C1/C3/C4/AC4, R and IR).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiMirrorDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiMirrorDiff.*'
//
// The AC4 cases pre-fill the destination with 0xA5 (and the in-place/batch
// source alpha with a column-varying pattern) so that the recorded golden
// pins down whether NVIDIA moves the alpha channel with the pixel or leaves
// it untouched.
//
// Measured NVIDIA NPP 13.4 quirks covered here:
//   * every in-place variant (plain and batch) rejects odd width or height
//     with NPP_SIZE_ERROR, so success paths use even dimensions;
//   * steps must be a multiple of the largest power of two dividing the
//     pixel size (Mirror_Contract_StepAlign); C4/AC4 cases therefore use
//     pads that keep 8/16-byte step granularity;
//   * NVIDIA's batch C4/AC4 kernels fault (device misaligned address) when
//     the step is not 16-byte aligned, poisoning the CUDA context - such
//     steps are never fed to the batch functions here.

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

// Out-of-place mirror.  When ac4 is true the destination is pre-filled with
// 0xA5 (including the alpha channel) so the golden shows the alpha rule.
template <typename T>
void mirror_oop(const char *fname, const std::string &key, int width, int height, int stride, int pad, NppiAxis flip,
                bool ac4, NppStatus (*f)(const T *, int, T *, int, NppiSize, NppiAxis, NppStreamContext),
                uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * stride * es + pad, es);
  const int dStep = align_step(width * stride * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  if (ac4) {
    for (size_t i = 0; i < hd.size(); ++i) hd[i] = (T)0xA5;
  }
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, flip, current_ctx());
  Case c("nppiMirror/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// In-place mirror.
template <typename T>
void mirror_ir(const char *fname, const std::string &key, int width, int height, int stride, int pad, NppiAxis flip,
               bool ac4, NppStatus (*f)(T *, int, NppiSize, NppiAxis, NppStreamContext), uint64_t seed) {
  const int es = (int)sizeof(T);
  const int step = align_step(width * stride * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<T> hs = make_uniform<T>(ne, seed + 11);
  if (ac4) {
    // Column-varying alpha so that "alpha untouched in place" and "alpha
    // mirrored with the pixel" differ in the recorded golden.
    const int row_elems = step / es;
    for (size_t i = 0; i < hs.size(); ++i) {
      if ((i % stride) == 3) {
        int col = (int)((i % (size_t)row_elems) / (size_t)stride);
        hs[i] = (T)(col + 1);
      }
    }
  }
  DeviceMemory<T> &s = arena<T>(ne);
  s.copyFromHost(hs);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, NppiSize{width, height}, flip, current_ctx());
  Case c("nppiMirror/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(s.get(), ne, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// Shared 8u AC4 contract buffers.
DeviceMemory<Npp8u> *g_ac4_src = nullptr;
DeviceMemory<Npp8u> *g_ac4_dst = nullptr;
Npp8u *ac4_src() {
  if (!g_ac4_src) {
    g_ac4_src = new DeviceMemory<Npp8u>(256);
    g_ac4_src->copyFromHost(std::vector<Npp8u>(256, 3));
  }
  return g_ac4_src->get();
}
Npp8u *ac4_dst() {
  if (!g_ac4_dst) {
    g_ac4_dst = new DeviceMemory<Npp8u>(256);
    g_ac4_dst->copyFromHost(std::vector<Npp8u>(256, 4));
  }
  return g_ac4_dst->get();
}

}  // namespace

class NppiMirrorDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== nppiMirror_16s_* (previously missing) ===============================
TEST_F(NppiMirrorDiff, Mirror_16s_C1) {
  mirror_oop<Npp16s>("nppiMirror_16s_C1R", "16s_C1R", 17, 13, 1, 16, NPP_VERTICAL_AXIS, false,
                     &nppiMirror_16s_C1R_Ctx, 300);
  mirror_ir<Npp16s>("nppiMirror_16s_C1IR", "16s_C1IR", 17, 13, 1, 16, NPP_BOTH_AXIS, false, &nppiMirror_16s_C1IR_Ctx,
                    301);
}
TEST_F(NppiMirrorDiff, Mirror_16s_C3) {
  mirror_oop<Npp16s>("nppiMirror_16s_C3R", "16s_C3R", 17, 13, 3, 16, NPP_HORIZONTAL_AXIS, false,
                     &nppiMirror_16s_C3R_Ctx, 302);
  mirror_ir<Npp16s>("nppiMirror_16s_C3IR", "16s_C3IR", 17, 13, 3, 16, NPP_VERTICAL_AXIS, false,
                    &nppiMirror_16s_C3IR_Ctx, 303);
}
TEST_F(NppiMirrorDiff, Mirror_16s_C4) {
  mirror_oop<Npp16s>("nppiMirror_16s_C4R", "16s_C4R", 64, 32, 4, 32, NPP_BOTH_AXIS, false, &nppiMirror_16s_C4R_Ctx,
                     304);
  mirror_ir<Npp16s>("nppiMirror_16s_C4IR", "16s_C4IR", 1, 1, 4, 8, NPP_VERTICAL_AXIS, false, &nppiMirror_16s_C4IR_Ctx,
                    305);
}

// ===== nppiMirror_*_AC4R / AC4IR ============================================
TEST_F(NppiMirrorDiff, Mirror_8u_AC4) {
  mirror_oop<Npp8u>("nppiMirror_8u_AC4R", "8u_AC4R", 17, 13, 4, 16, NPP_VERTICAL_AXIS, true, &nppiMirror_8u_AC4R_Ctx,
                    310);
  mirror_ir<Npp8u>("nppiMirror_8u_AC4IR", "8u_AC4IR", 18, 14, 4, 16, NPP_HORIZONTAL_AXIS, true,
                   &nppiMirror_8u_AC4IR_Ctx, 311);
}
TEST_F(NppiMirrorDiff, Mirror_16u_AC4) {
  mirror_oop<Npp16u>("nppiMirror_16u_AC4R", "16u_AC4R", 17, 13, 4, 16, NPP_BOTH_AXIS, true, &nppiMirror_16u_AC4R_Ctx,
                     312);
  mirror_ir<Npp16u>("nppiMirror_16u_AC4IR", "16u_AC4IR", 2, 2, 4, 8, NPP_VERTICAL_AXIS, true,
                    &nppiMirror_16u_AC4IR_Ctx, 313);
}
TEST_F(NppiMirrorDiff, Mirror_16s_AC4) {
  mirror_oop<Npp16s>("nppiMirror_16s_AC4R", "16s_AC4R", 17, 13, 4, 16, NPP_HORIZONTAL_AXIS, true,
                     &nppiMirror_16s_AC4R_Ctx, 314);
  mirror_ir<Npp16s>("nppiMirror_16s_AC4IR", "16s_AC4IR", 17, 13, 4, 16, NPP_BOTH_AXIS, true,
                    &nppiMirror_16s_AC4IR_Ctx, 315);
}
TEST_F(NppiMirrorDiff, Mirror_32f_AC4) {
  mirror_oop<Npp32f>("nppiMirror_32f_AC4R", "32f_AC4R", 17, 13, 4, 16, NPP_VERTICAL_AXIS, true,
                     &nppiMirror_32f_AC4R_Ctx, 316);
  mirror_ir<Npp32f>("nppiMirror_32f_AC4IR", "32f_AC4IR", 64, 32, 4, 32, NPP_BOTH_AXIS, true,
                    &nppiMirror_32f_AC4IR_Ctx, 317);
}
TEST_F(NppiMirrorDiff, Mirror_32s_AC4) {
  mirror_oop<Npp32s>("nppiMirror_32s_AC4R", "32s_AC4R", 17, 13, 4, 16, NPP_BOTH_AXIS, true, &nppiMirror_32s_AC4R_Ctx,
                     318);
  mirror_ir<Npp32s>("nppiMirror_32s_AC4IR", "32s_AC4IR", 2, 2, 4, 48, NPP_HORIZONTAL_AXIS, true,
                    &nppiMirror_32s_AC4IR_Ctx, 319);
}

// ===== flip axis coverage ===================================================
TEST_F(NppiMirrorDiff, Mirror_Flips) {
  mirror_oop<Npp16s>("nppiMirror_16s_C1R", "16s_C1R/flipH", 17, 13, 1, 16, NPP_HORIZONTAL_AXIS, false,
                     &nppiMirror_16s_C1R_Ctx, 320);
  mirror_oop<Npp16s>("nppiMirror_16s_C1R", "16s_C1R/flipV", 17, 13, 1, 16, NPP_VERTICAL_AXIS, false,
                     &nppiMirror_16s_C1R_Ctx, 321);
  mirror_oop<Npp16s>("nppiMirror_16s_C1R", "16s_C1R/flipB", 17, 13, 1, 16, NPP_BOTH_AXIS, false,
                     &nppiMirror_16s_C1R_Ctx, 322);
  mirror_oop<Npp16s>("nppiMirror_16s_C1R", "16s_C1R/1x1", 1, 1, 1, 8, NPP_BOTH_AXIS, false, &nppiMirror_16s_C1R_Ctx,
                     323);
  mirror_ir<Npp16s>("nppiMirror_16s_C1IR", "16s_C1IR/odd", 5, 7, 1, 8, NPP_BOTH_AXIS, false, &nppiMirror_16s_C1IR_Ctx,
                    324);
}

// ===== in-place success paths: even width and height ========================
// NVIDIA NPP 13.4 rejects odd dimensions for every in-place mirror variant, so
// the success path is only reachable with even sizes (see executor header).
TEST_F(NppiMirrorDiff, Mirror_IR_Even) {
  mirror_ir<Npp16s>("nppiMirror_16s_C1IR", "16s_C1IR/even", 18, 14, 1, 16, NPP_BOTH_AXIS, false,
                    &nppiMirror_16s_C1IR_Ctx, 330);
  mirror_ir<Npp16s>("nppiMirror_16s_C3IR", "16s_C3IR/even", 34, 22, 3, 16, NPP_VERTICAL_AXIS, false,
                    &nppiMirror_16s_C3IR_Ctx, 331);
  mirror_ir<Npp16s>("nppiMirror_16s_C4IR", "16s_C4IR/even", 2, 2, 4, 8, NPP_BOTH_AXIS, false,
                    &nppiMirror_16s_C4IR_Ctx, 332);
  mirror_ir<Npp8u>("nppiMirror_8u_C1IR", "8u_C1IR/even", 18, 14, 1, 16, NPP_HORIZONTAL_AXIS, false,
                   &nppiMirror_8u_C1IR_Ctx, 333);
  mirror_ir<Npp16u>("nppiMirror_16u_C3IR", "16u_C3IR/even", 34, 22, 3, 16, NPP_VERTICAL_AXIS, false,
                    &nppiMirror_16u_C3IR_Ctx, 334);
  mirror_ir<Npp32f>("nppiMirror_32f_C4IR", "32f_C4IR/even", 2, 2, 4, 48, NPP_BOTH_AXIS, false,
                    &nppiMirror_32f_C4IR_Ctx, 335);
  mirror_ir<Npp32s>("nppiMirror_32s_C1IR", "32s_C1IR/even", 18, 14, 1, 16, NPP_HORIZONTAL_AXIS, false,
                    &nppiMirror_32s_C1IR_Ctx, 336);
}

// ===== contract: empty/negative ROI, null pointer, bad flip, short step =====
TEST_F(NppiMirrorDiff, Mirror_Contract) {
  DeviceMemory<Npp16s> &s = arena<Npp16s>(256);
  DeviceMemory<Npp16s> &d = arena<Npp16s>(256);
  s.copyFromHost(std::vector<Npp16s>(256, 7));
  d.copyFromHost(std::vector<Npp16s>(256, 9));
  auto run = [&](const std::string &key, NppiSize roi, NppiAxis flip, Npp16s *dst, int srcStep) {
    clear_cuda_error();
    NppStatus st = nppiMirror_16s_C1R_Ctx(s.get(), srcStep, dst, 64, roi, flip, current_ctx());
    Case c("nppiMirror/contract/" + key, "nppiMirror_16s_C1R");
    c.status(st);
    c.finish();
  };
  run("empty", NppiSize{0, 5}, NPP_VERTICAL_AXIS, d.get(), 64);
  run("negative", NppiSize{4, -1}, NPP_VERTICAL_AXIS, d.get(), 64);
  run("nulldst", NppiSize{4, 5}, NPP_VERTICAL_AXIS, nullptr, 64);
  run("badflip", NppiSize{4, 5}, (NppiAxis)7, d.get(), 64);
  run("shortsrcstep", NppiSize{4, 5}, NPP_VERTICAL_AXIS, d.get(), 4);
  clear_cuda_error();
  {
    NppStatus st = nppiMirror_16s_C1IR_Ctx(s.get(), 4, NppiSize{4, 5}, NPP_VERTICAL_AXIS, current_ctx());
    Case c("nppiMirror/contract/shortstep_ir", "nppiMirror_16s_C1IR");
    c.status(st);
    c.finish();
  }
  clear_cuda_error();
  {
    NppStatus st = nppiMirror_16s_C1IR_Ctx(nullptr, 64, NppiSize{4, 5}, NPP_VERTICAL_AXIS, current_ctx());
    Case c("nppiMirror/contract/nullsrc_ir", "nppiMirror_16s_C1IR");
    c.status(st);
    c.finish();
  }
  clear_cuda_error();
  {
    NppStatus st = nppiMirror_8u_AC4R_Ctx(ac4_src(), 4, ac4_dst(), 64, NppiSize{2, 2}, NPP_VERTICAL_AXIS,
                                         current_ctx());
    Case c("nppiMirror/contract/ac4_shortstep", "nppiMirror_8u_AC4R");
    c.status(st);
    c.finish();
  }
  clear_cuda_error();
  {
    NppStatus st = nppiMirror_8u_AC4IR_Ctx(nullptr, 64, NppiSize{2, 2}, NPP_VERTICAL_AXIS, current_ctx());
    Case c("nppiMirror/contract/ac4_null_ir", "nppiMirror_8u_AC4IR");
    c.status(st);
    c.finish();
  }
}

// Empty ROI for the remaining shapes (out-of-place height 0, in-place 0 x n).
TEST_F(NppiMirrorDiff, Mirror_Contract_EmptyRoi) {
  DeviceMemory<Npp16s> &s = arena<Npp16s>(256);
  DeviceMemory<Npp16s> &d = arena<Npp16s>(256);
  s.copyFromHost(std::vector<Npp16s>(256, 7));
  d.copyFromHost(std::vector<Npp16s>(256, 9));
  auto run = [&](const std::string &key, NppiSize roi, Npp16s *dst) {
    clear_cuda_error();
    NppStatus st = nppiMirror_16s_C1R_Ctx(s.get(), 64, dst, 64, roi, NPP_VERTICAL_AXIS, current_ctx());
    Case c("nppiMirror/contract/" + key, "nppiMirror_16s_C1R");
    c.status(st);
    c.finish();
  };
  run("emptyh", NppiSize{5, 0}, d.get());
  run("emptyboth", NppiSize{0, 0}, d.get());
  clear_cuda_error();
  {
    NppStatus st = nppiMirror_16s_C1IR_Ctx(s.get(), 64, NppiSize{0, 6}, NPP_VERTICAL_AXIS, current_ctx());
    Case c("nppiMirror/contract/empty_ir", "nppiMirror_16s_C1IR");
    c.status(st);
    c.finish();
  }
  clear_cuda_error();
  {
    NppStatus st = nppiMirror_16s_C1IR_Ctx(s.get(), 64, NppiSize{0, 5}, NPP_VERTICAL_AXIS, current_ctx());
    Case c("nppiMirror/contract/empty_ir_odd", "nppiMirror_16s_C1IR");
    c.status(st);
    c.finish();
  }
}

// NVIDIA NPP 13.4 requires each step to be a multiple of the largest power of
// two dividing the pixel size (8u C3 -> 1, 32f C3 -> 4, 16s C4 -> 8,
// 32f C4/AC4 -> 16) and returns NPP_NOT_EVEN_STEP_ERROR otherwise.  A 32f C3
// step does NOT need 12-byte granularity, so 256 (256 % 12 == 4) is accepted.
// The check fires after the empty-ROI early-out and before the null-pointer,
// in-place-parity and flip checks.
TEST_F(NppiMirrorDiff, Mirror_Contract_StepAlign) {
  DeviceMemory<Npp32f> &s = arena<Npp32f>(4096);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(4096);
  s.copyFromHost(make_uniform<Npp32f>(4096, 501));
  d.copyFromHost(make_uniform<Npp32f>(4096, 502));
  DeviceMemory<Npp16s> &s16 = arena<Npp16s>(2048);
  DeviceMemory<Npp16s> &d16 = arena<Npp16s>(2048);
  s16.copyFromHost(make_uniform<Npp16s>(2048, 503));
  d16.copyFromHost(make_uniform<Npp16s>(2048, 504));
  DeviceMemory<Npp8u> &s8 = arena<Npp8u>(2048);
  DeviceMemory<Npp8u> &d8 = arena<Npp8u>(2048);
  s8.copyFromHost(make_uniform<Npp8u>(2048, 505));
  d8.copyFromHost(make_uniform<Npp8u>(2048, 506));
  auto run = [&](const std::string &key, NppStatus st, const char *fn) {
    Case c("nppiMirror/contract/stepalign/" + key, fn);
    c.status(st);
    c.finish();
  };
  // 32f C4 (pixel 16): step 284 (284 % 16 == 12).
  clear_cuda_error();
  run("c4_32f_mis", nppiMirror_32f_C4R_Ctx(s.get(), 284, d.get(), 284, NppiSize{17, 13}, NPP_VERTICAL_AXIS,
                                           current_ctx()),
      "nppiMirror_32f_C4R");
  clear_cuda_error();
  run("c4_32f_dstonly", nppiMirror_32f_C4R_Ctx(s.get(), 288, d.get(), 284, NppiSize{17, 13}, NPP_VERTICAL_AXIS,
                                               current_ctx()),
      "nppiMirror_32f_C4R");
  // 32f C3 (pixel 12, 4-byte granularity): 250 fails, 256 is accepted.
  clear_cuda_error();
  run("c3_32f_mis", nppiMirror_32f_C3R_Ctx(s.get(), 250, d.get(), 250, NppiSize{20, 13}, NPP_VERTICAL_AXIS,
                                           current_ctx()),
      "nppiMirror_32f_C3R");
  clear_cuda_error();
  run("c3_32f_ok", nppiMirror_32f_C3R_Ctx(s.get(), 256, d.get(), 256, NppiSize{20, 13}, NPP_VERTICAL_AXIS,
                                          current_ctx()),
      "nppiMirror_32f_C3R");
  // 8u AC4 (pixel 4): step 66 (66 % 4 == 2).
  clear_cuda_error();
  run("ac4_8u_mis", nppiMirror_8u_AC4R_Ctx(s8.get(), 66, d8.get(), 66, NppiSize{16, 13}, NPP_VERTICAL_AXIS,
                                           current_ctx()),
      "nppiMirror_8u_AC4R");
  // 16s C4 (pixel 8): step 132 (132 % 8 == 4).
  clear_cuda_error();
  run("c4_16s_mis", nppiMirror_16s_C4R_Ctx(s16.get(), 132, d16.get(), 132, NppiSize{16, 13}, NPP_VERTICAL_AXIS,
                                           current_ctx()),
      "nppiMirror_16s_C4R");
  // Misaligned step + even in-place dims: alignment wins over the parity rule.
  clear_cuda_error();
  run("c4ir_32f_mis", nppiMirror_32f_C4IR_Ctx(s.get(), 260, NppiSize{16, 12}, NPP_VERTICAL_AXIS, current_ctx()),
      "nppiMirror_32f_C4IR");
  // Misaligned step beats the null-pointer check.
  clear_cuda_error();
  run("c4_32f_mis_null", nppiMirror_32f_C4R_Ctx(s.get(), 284, nullptr, 284, NppiSize{17, 13}, NPP_VERTICAL_AXIS,
                                                current_ctx()),
      "nppiMirror_32f_C4R");
  // Empty ROI beats the alignment check.
  clear_cuda_error();
  run("c4_32f_mis_empty", nppiMirror_32f_C4R_Ctx(s.get(), 284, d.get(), 284, NppiSize{17, 0}, NPP_VERTICAL_AXIS,
                                                 current_ctx()),
      "nppiMirror_32f_C4R");
  // Misaligned step beats the flip check.
  clear_cuda_error();
  run("c4_32f_mis_badflip",
      nppiMirror_32f_C4R_Ctx(s.get(), 284, d.get(), 284, NppiSize{17, 13}, (NppiAxis)7, current_ctx()),
      "nppiMirror_32f_C4R");
}

// ===== nppiMirrorBatch_32f_* ===============================================
// R variants get distinct src/dst buffers (no aliasing); IR variants point
// pSrc/pDst at the same buffer.
void mirror_batch(const char *fname, const std::string &key, int width, int height, int stride, int pad, NppiAxis flip,
                  bool ac4, bool inplace, NppStatus (*f)(NppiSize, NppiAxis, NppiMirrorBatchCXR *, int, NppStreamContext),
                  uint64_t seed) {
  const int es = 4;
  const int step = align_step(width * stride * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  const int nBatch = 3;
  const int row_elems = step / es;
  std::vector<std::vector<Npp32f>> host(nBatch), hdst(nBatch);
  std::vector<DeviceMemory<Npp32f> *> srcs(nBatch), dsts(nBatch);
  for (int i = 0; i < nBatch; ++i) {
    host[i] = make_uniform<Npp32f>(ne, seed + 11 + (uint64_t)i * 7);
    if (ac4) {
      // Column-varying alpha pins down whether the batch AC4 variants move
      // the alpha channel or leave it in place.
      for (size_t p = 0; p < host[i].size(); ++p) {
        if ((p % stride) == 3) host[i][p] = (float)(1 + (int)((p % (size_t)row_elems) / (size_t)stride));
      }
    }
    hdst[i] = make_uniform<Npp32f>(ne, seed + 29 + (uint64_t)i * 7);
    srcs[i] = new DeviceMemory<Npp32f>(ne);
    dsts[i] = new DeviceMemory<Npp32f>(ne);
    srcs[i]->copyFromHost(host[i]);
    dsts[i]->copyFromHost(hdst[i]);
  }
  std::vector<NppiMirrorBatchCXR> list(nBatch);
  for (int i = 0; i < nBatch; ++i) {
    list[i].pSrc = srcs[i]->get();
    list[i].nSrcStep = step;
    list[i].pDst = inplace ? srcs[i]->get() : dsts[i]->get();
    list[i].nDstStep = step;
  }
  DeviceMemory<NppiMirrorBatchCXR> &dl = arena<NppiMirrorBatchCXR>(nBatch);
  dl.copyFromHost(list);
  clear_cuda_error();
  NppStatus st = f(NppiSize{width, height}, flip, dl.get(), nBatch, current_ctx());
  for (int i = 0; i < nBatch; ++i) {
    Npp32f *out = inplace ? srcs[i]->get() : dsts[i]->get();
    Case c("nppiMirrorBatch/" + key + "/out" + std::to_string(i), fname);
    // The inputs belong to the output cases: an input-only Case has no status
    // and the verifier rejects status-less cases.
    c.input(host[i].data(), host[i].size() * sizeof(Npp32f));
    c.input(hdst[i].data(), hdst[i].size() * sizeof(Npp32f));
    c.status(st);
    c.device_output<Npp32f>(out, ne, ValueKind::F32, 1e-5, 1e-6);
    c.finish();
  }
}

TEST_F(NppiMirrorDiff, MirrorBatch_C1) {
  mirror_batch("nppiMirrorBatch_32f_C1R", "C1R", 17, 13, 1, 16, NPP_VERTICAL_AXIS, false, false,
               &nppiMirrorBatch_32f_C1R_Ctx, 400);
  mirror_batch("nppiMirrorBatch_32f_C1IR", "C1IR", 17, 13, 1, 16, NPP_BOTH_AXIS, false, true,
               &nppiMirrorBatch_32f_C1IR_Ctx, 401);
}
TEST_F(NppiMirrorDiff, MirrorBatch_C3) {
  mirror_batch("nppiMirrorBatch_32f_C3R", "C3R", 17, 13, 3, 16, NPP_HORIZONTAL_AXIS, false, false,
               &nppiMirrorBatch_32f_C3R_Ctx, 402);
  mirror_batch("nppiMirrorBatch_32f_C3IR", "C3IR", 1, 1, 3, 8, NPP_VERTICAL_AXIS, false, true,
               &nppiMirrorBatch_32f_C3IR_Ctx, 403);
}
TEST_F(NppiMirrorDiff, MirrorBatch_C4) {
  mirror_batch("nppiMirrorBatch_32f_C4R", "C4R", 17, 13, 4, 16, NPP_BOTH_AXIS, false, false,
               &nppiMirrorBatch_32f_C4R_Ctx, 404);
  mirror_batch("nppiMirrorBatch_32f_C4IR", "C4IR", 5, 7, 4, 8, NPP_VERTICAL_AXIS, false, true,
               &nppiMirrorBatch_32f_C4IR_Ctx, 405);
}
TEST_F(NppiMirrorDiff, MirrorBatch_AC4) {
  mirror_batch("nppiMirrorBatch_32f_AC4R", "AC4R", 17, 13, 4, 16, NPP_VERTICAL_AXIS, true, false,
               &nppiMirrorBatch_32f_AC4R_Ctx, 406);
  mirror_batch("nppiMirrorBatch_32f_AC4IR", "AC4IR", 17, 13, 4, 16, NPP_BOTH_AXIS, true, true,
               &nppiMirrorBatch_32f_AC4IR_Ctx, 407);
}

// In-place batch success paths require even dimensions (see executor header).
TEST_F(NppiMirrorDiff, MirrorBatch_IR_Even) {
  mirror_batch("nppiMirrorBatch_32f_C1IR", "C1IR/even", 16, 12, 1, 16, NPP_BOTH_AXIS, false, true,
               &nppiMirrorBatch_32f_C1IR_Ctx, 408);
  mirror_batch("nppiMirrorBatch_32f_C3IR", "C3IR/even", 18, 14, 3, 16, NPP_VERTICAL_AXIS, false, true,
               &nppiMirrorBatch_32f_C3IR_Ctx, 409);
  // pad 48 keeps the step 16-byte aligned: NVIDIA's batch C4/AC4 kernels
  // fault with a device misaligned-address error otherwise (poisoning the
  // CUDA context), which would make this case unrecordable.
  mirror_batch("nppiMirrorBatch_32f_C4IR", "C4IR/even", 2, 2, 4, 48, NPP_HORIZONTAL_AXIS, false, true,
               &nppiMirrorBatch_32f_C4IR_Ctx, 410);
  mirror_batch("nppiMirrorBatch_32f_AC4IR", "AC4IR/even", 18, 14, 4, 16, NPP_BOTH_AXIS, true, true,
               &nppiMirrorBatch_32f_AC4IR_Ctx, 411);
}
// Batch contract cases are separate TEST_Fs: NVIDIA NPP 13.4 raises SIGFPE
// (integer divide by zero, host side) for oSizeROI.width == 0, which aborts
// the whole process, so that combination is deliberately NOT recorded here.
// nBatchSize == 0 merely returns NPP_BAD_ARGUMENT_ERROR and is safe.
namespace {
void mirror_batch_contract(const std::string &key, NppiSize roi, NppiAxis flip, int nBatch, bool null_list) {
  const int step = 17 * 4 * 4;
  const size_t ne = (size_t)13 * step / 4;
  DeviceMemory<Npp32f> &b = arena<Npp32f>(ne);
  b.copyFromHost(make_uniform<Npp32f>(ne, 99));
  std::vector<NppiMirrorBatchCXR> list(3);
  for (auto &e : list) {
    e.pSrc = b.get();
    e.nSrcStep = step;
    e.pDst = b.get();
    e.nDstStep = step;
  }
  DeviceMemory<NppiMirrorBatchCXR> &dl = arena<NppiMirrorBatchCXR>(3);
  dl.copyFromHost(list);
  clear_cuda_error();
  NppStatus st = nppiMirrorBatch_32f_C1R_Ctx(roi, flip, null_list ? nullptr : dl.get(), nBatch, current_ctx());
  Case c("nppiMirrorBatch/contract/" + key, "nppiMirrorBatch_32f_C1R");
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(NppiMirrorDiff, MirrorBatch_Contract_Negative) {
  mirror_batch_contract("negative", NppiSize{4, -1}, NPP_VERTICAL_AXIS, 3, false);
}
TEST_F(NppiMirrorDiff, MirrorBatch_Contract_OneImage) {
  mirror_batch_contract("oneimage", NppiSize{4, 5}, NPP_VERTICAL_AXIS, 1, false);
}
TEST_F(NppiMirrorDiff, MirrorBatch_Contract_TwoImages) {
  mirror_batch_contract("twoimages", NppiSize{4, 5}, NPP_VERTICAL_AXIS, 2, false);
}
TEST_F(NppiMirrorDiff, MirrorBatch_Contract_BadFlip) {
  mirror_batch_contract("badflip", NppiSize{4, 5}, (NppiAxis)7, 3, false);
}
TEST_F(NppiMirrorDiff, MirrorBatch_Contract_NullList) {
  mirror_batch_contract("nulllist", NppiSize{4, 5}, NPP_VERTICAL_AXIS, 3, true);
}
TEST_F(NppiMirrorDiff, MirrorBatch_Contract_Zero) {
  mirror_batch_contract("zero", NppiSize{4, 5}, NPP_VERTICAL_AXIS, 0, false);
}

// ===== Non-Ctx adapters (representative _Ctx cases -> same golden) =========
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiMirrorDiff, NonCtx_Mirror) {
  // Reproduces mirror_oop(16s_C1R, 17,13,1,16, VERTICAL, seed=300).
  const int es = 2, stride = 1, pad = 16, width = 17, height = 13;
  const int step = ((width * stride * es + pad + es - 1) / es) * es;
  const size_t ne = (size_t)height * step / es;
  std::vector<Npp16s> hs = make_uniform<Npp16s>(ne, 300 + 11);
  std::vector<Npp16s> hd = make_uniform<Npp16s>(ne, 300 + 29);
  DeviceMemory<Npp16s> &s = arena<Npp16s>(ne);
  DeviceMemory<Npp16s> &d = arena<Npp16s>(ne);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiMirror_16s_C1R(s.get(), step, d.get(), step, NppiSize{width, height}, NPP_VERTICAL_AXIS);
  Case c("nppiMirror/16s_C1R", "nppiMirror_16s_C1R");
  c.status(st);
  c.device_output<Npp16s>(d.get(), ne, ValueKind::I16, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppiMirrorDiff, NonCtx_MirrorAC4) {
  // Reproduces mirror_oop(8u_AC4R, 17,13,4,16, VERTICAL, ac4, seed=310).
  const int es = 1, stride = 4, pad = 16, width = 17, height = 13;
  const int step = ((width * stride * es + pad + es - 1) / es) * es;
  const size_t ne = (size_t)height * step / es;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(ne, 310 + 11);
  std::vector<Npp8u> hd(ne, 0xA5);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(ne);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(ne);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiMirror_8u_AC4R(s.get(), step, d.get(), step, NppiSize{width, height}, NPP_VERTICAL_AXIS);
  Case c("nppiMirror/8u_AC4R", "nppiMirror_8u_AC4R");
  c.status(st);
  c.device_output<Npp8u>(d.get(), ne, ValueKind::U8, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppiMirrorDiff, NonCtx_MirrorBatch) {
  // Reproduces mirror_batch(C1R, 17,13,1,16, VERTICAL, seed=400) with nBatch=3.
  const int es = 4, stride = 1, pad = 16, width = 17, height = 13;
  const int step = ((width * stride * es + pad + es - 1) / es) * es;
  const size_t ne = (size_t)height * step / es;
  const int nBatch = 3;
  std::vector<DeviceMemory<Npp32f> *> srcs(nBatch), dsts(nBatch);
  std::vector<NppiMirrorBatchCXR> list(nBatch);
  for (int i = 0; i < nBatch; ++i) {
    std::vector<Npp32f> hs = make_uniform<Npp32f>(ne, 400 + 11 + (uint64_t)i * 7);
    std::vector<Npp32f> hd = make_uniform<Npp32f>(ne, 400 + 29 + (uint64_t)i * 7);
    srcs[i] = new DeviceMemory<Npp32f>(ne);
    dsts[i] = new DeviceMemory<Npp32f>(ne);
    srcs[i]->copyFromHost(hs);
    dsts[i]->copyFromHost(hd);
    list[i].pSrc = srcs[i]->get();
    list[i].nSrcStep = step;
    list[i].pDst = dsts[i]->get();
    list[i].nDstStep = step;
  }
  DeviceMemory<NppiMirrorBatchCXR> &dl = arena<NppiMirrorBatchCXR>(nBatch);
  dl.copyFromHost(list);
  clear_cuda_error();
  NppStatus st = nppiMirrorBatch_32f_C1R(NppiSize{width, height}, NPP_VERTICAL_AXIS, dl.get(), nBatch);
  for (int i = 0; i < nBatch; ++i) {
    Case c("nppiMirrorBatch/C1R/out" + std::to_string(i), "nppiMirrorBatch_32f_C1R");
    c.status(st);
    c.device_output<Npp32f>(dsts[i]->get(), ne, ValueKind::F32, 1e-5, 1e-6);
    c.finish();
  }
}
#endif
