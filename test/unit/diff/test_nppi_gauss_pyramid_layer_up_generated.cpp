// Differential tests for nppiFilterGaussPyramidLayerUpBorder.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiGaussPyramidUpDiff.*'
//
// The reference tap phase is host-driven and not reproduced for every rate.
// These cases pin the measured geometry that this executor does reproduce:
// delta kernels at rate 2 (nearest-neighbour mapping src = floor(dst/rate)),
// unit-sum kernels on constant images, and the status contract (shared with
// LayerDown: null -8, size -6, offset -11, border -14, taps -33, rate -5,
// empty ROI no-op, short dst step -16).
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

template <typename T>
std::vector<T> makeSource(int sw, int sh, int pad, int ch, bool constant) {
  const int stride = sw + pad;
  std::vector<T> src((size_t)stride * sh * ch);
  for (int y = 0; y < sh; ++y) {
    for (int x = 0; x < stride; ++x) {
      for (int c = 0; c < ch; ++c) {
        src[((size_t)y * stride + x) * ch + c] =
            constant ? (T)7 : (T)(x * 10 + y + c * 3);
      }
    }
  }
  return src;
}

template <typename T, int CH>
void up_case(const std::string &id, const char *fname, int sw, int sh, int dw, int dh, int pad, float rate,
             int taps, const std::vector<float> &kernel, int offX, int offY, NppiBorderType border,
             const std::vector<T> &src) {
  DeviceMemory<T> dsrc(src.size());
  dsrc.copyFromHost(src);
  DeviceMemory<T> dst((size_t)dw * dh * CH);
  std::vector<T> init((size_t)dw * dh * CH, (T)0xCD);
  dst.copyFromHost(init);

  NppStreamContext ctx = current_context();
  const int srcStep = (sw + pad) * CH * (int)sizeof(T);
  const int dstStep = dw * CH * (int)sizeof(T);
  NppStatus st;
  if constexpr (std::is_same<T, Npp8u>::value) {
    if constexpr (CH == 1) {
      st = nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), srcStep, {sw, sh}, {offX, offY}, dst.get(),
                                                          dstStep, {dw, dh}, rate, taps, kernel.data(), border, ctx);
    } else {
      st = nppiFilterGaussPyramidLayerUpBorder_8u_C3R_Ctx(dsrc.get(), srcStep, {sw, sh}, {offX, offY}, dst.get(),
                                                          dstStep, {dw, dh}, rate, taps, kernel.data(), border, ctx);
    }
  } else if constexpr (std::is_same<T, Npp16u>::value) {
    if constexpr (CH == 1) {
      st = nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx(dsrc.get(), srcStep, {sw, sh}, {offX, offY}, dst.get(),
                                                           dstStep, {dw, dh}, rate, taps, kernel.data(), border,
                                                           ctx);
    } else {
      st = nppiFilterGaussPyramidLayerUpBorder_16u_C3R_Ctx(dsrc.get(), srcStep, {sw, sh}, {offX, offY}, dst.get(),
                                                           dstStep, {dw, dh}, rate, taps, kernel.data(), border,
                                                           ctx);
    }
  } else {
    if constexpr (CH == 1) {
      st = nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx(dsrc.get(), srcStep, {sw, sh}, {offX, offY}, dst.get(),
                                                           dstStep, {dw, dh}, rate, taps, kernel.data(), border,
                                                           ctx);
    } else {
      st = nppiFilterGaussPyramidLayerUpBorder_32f_C3R_Ctx(dsrc.get(), srcStep, {sw, sh}, {offX, offY}, dst.get(),
                                                           dstStep, {dw, dh}, rate, taps, kernel.data(), border,
                                                           ctx);
    }
  }
  clear_cuda_error();

  Case c(id, fname);
  c.status(st);
  c.input(src.data(), src.size() * sizeof(T));
  if (st == NPP_NO_ERROR) {
    std::vector<T> host((size_t)dw * dh * CH);
    dst.copyToHost(host);
    c.host_output(host.data(), host.size() * sizeof(T), ValueKind::Bytes);
  }
  c.finish();
}

const std::vector<float> kDelta = {0.0f, 1.0f, 0.0f};
const std::vector<float> kGauss = {0.25f, 0.5f, 0.25f};

}  // namespace

TEST(NppiGaussPyramidUpDiff, Values) {
  // 8x8 source at rate 2 -> 16x16 destination; delta kernel exposes the
  // nearest-neighbour upsampling map.
  up_case<Npp8u, 1>("s2/PyramidUp/8u_C1_rate2_delta", "nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx", 8, 8, 16,
                    16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp8u>(8, 8, 2, 1, false));
  up_case<Npp8u, 3>("s2/PyramidUp/8u_C3_rate2_delta", "nppiFilterGaussPyramidLayerUpBorder_8u_C3R_Ctx", 8, 8, 16,
                    16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp8u>(8, 8, 2, 3, false));
  up_case<Npp16u, 1>("s2/PyramidUp/16u_C1_rate2_delta", "nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx", 8, 8,
                     16, 16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp16u>(8, 8, 2, 1, false));
  up_case<Npp16u, 3>("s2/PyramidUp/16u_C3_rate2_delta", "nppiFilterGaussPyramidLayerUpBorder_16u_C3R_Ctx", 8, 8,
                     16, 16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp16u>(8, 8, 2, 3, false));
  up_case<Npp32f, 1>("s2/PyramidUp/32f_C1_rate2_delta", "nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx", 8, 8,
                     16, 16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp32f>(8, 8, 2, 1, false));
  up_case<Npp32f, 3>("s2/PyramidUp/32f_C3_rate2_delta", "nppiFilterGaussPyramidLayerUpBorder_32f_C3R_Ctx", 8, 8,
                     16, 16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp32f>(8, 8, 2, 3, false));
  // Mirror geometry and a unit-sum Gaussian on a constant image.  (A non-zero
  // oSrcOffset requires pSrc to sit inside a larger source image, so it is
  // covered by the status cases only.)
  up_case<Npp8u, 1>("s2/PyramidUp/8u_C1_rate2_mirror", "nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx", 8, 8, 16,
                    16, 2, 2.0f, 3, kDelta, 0, 0, NPP_BORDER_MIRROR, makeSource<Npp8u>(8, 8, 2, 1, false));
  up_case<Npp8u, 1>("s2/PyramidUp/8u_C1_const_gauss", "nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx", 8, 8, 16,
                    16, 2, 2.0f, 3, kGauss, 0, 0, NPP_BORDER_REPLICATE, makeSource<Npp8u>(8, 8, 2, 1, true));
}

TEST(NppiGaussPyramidUpDiff, Contract) {
  const int sw = 8, sh = 8, dw = 16, dh = 16;
  std::vector<Npp8u> src = makeSource<Npp8u>(sw, sh, 0, 1, false);
  DeviceMemory<Npp8u> dsrc(src.size());
  dsrc.copyFromHost(src);
  DeviceMemory<Npp8u> dst((size_t)dw * dh);
  NppStreamContext ctx = current_context();
  const char *fname = "nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx";

  status_case("s2/PyramidUp/null_src", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(nullptr, sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/null_dst", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, nullptr, dw,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/null_kernel", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 3, nullptr, NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/srcsize_zero", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {0, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/roi_negative", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {-1, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/offset_negative", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {-1, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/offset_too_big", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {8, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/border_bad", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(), NPP_BORDER_CONSTANT,
                                                             ctx));
  status_case("s2/PyramidUp/taps_one", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 1, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/taps_100", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 2.0f, 100, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/rate_one", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 1.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/rate_too_big", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {dw, dh}, 10.1f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/roi_zero", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw,
                                                             {0, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  status_case("s2/PyramidUp/step_short", fname,
              nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(dsrc.get(), sw, {sw, sh}, {0, 0}, dst.get(), dw - 1,
                                                             {dw, dh}, 2.0f, 3, kDelta.data(),
                                                             NPP_BORDER_REPLICATE, ctx));
  clear_cuda_error();
}
