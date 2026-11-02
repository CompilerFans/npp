// Differential tests for the c3 batch color-conversion families
// (nppi_color_conversion.h):
//
//   nppi{YCbCr420,YUV420}To{BGR,RGB}Batch_8u_P3C3R
//   nppi{YCbCr422,YUV422}To{BGR,RGB}Batch_8u_P3C3R
//   nppi{YCbCr,YUV}To{BGR,RGB}Batch_8u_{C3R,P3C3R}
//
// Every family has a "single ROI" and an "_Advanced" (per destination image
// ROI) entry point, both in _Ctx and plain form.  The tests drive the batch
// descriptors from device memory, exactly like the reference suite.
//
// Semantics recorded from NVIDIA NPP 13.4 (RTX 5090):
//   * The batch entry points validate the list pointers, nBatchSize, the
//     per-image data pointers, steps and ROIs; the probe test records the
//     exact status for each illegal input.
//   * The plain variant applies oSizeROI to every pair; the _Advanced
//     variant takes each pair's ROI from pDstBatchList[i].oSize and
//     oMaxSizeROI is an upper bound for those ROIs.
//   * The per-image work matches the corresponding single-image conversion,
//     so the conversion contract is the one documented by the
//     test_nppi_c3_*_generated.cpp suites.
//   * Sampling constraints apply per ROI (even width for 4:2:2, even width
//     and height for 4:2:0).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiC3Batch*Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiC3Batch*Diff.*'

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

NppiImageDescriptor descOf(const DeviceMemory<Npp8u> &m, int step, int w, int h) {
  NppiImageDescriptor d{};
  d.pData = const_cast<Npp8u *>(m.get());
  d.nStep = step;
  d.oSize = NppiSize{w, h};
  return d;
}

struct P3Image {
  std::unique_ptr<DeviceMemory<Npp8u>> y, cb, cr, dst;
  int yStep = 0, cStep = 0, dStep = 0;
  int roiW = 0, roiH = 0;
  int bufH = 0; // allocated destination/texture rows (roiH is the ROI height)
};

// One planar (P3C3R) batch: three lists of `nb` source descriptors plus the
// destination descriptor list.
struct PlanarBatch {
  std::vector<P3Image> images;
  DeviceMemory<NppiImageDescriptor> lists[3];
  DeviceMemory<NppiImageDescriptor> dstList;
  const NppiImageDescriptor *ptr[3];

  PlanarBatch(int nb, int w, int h, int chromaDiv, bool advanced, uint64_t seed)
      : lists{DeviceMemory<NppiImageDescriptor>(nb), DeviceMemory<NppiImageDescriptor>(nb),
              DeviceMemory<NppiImageDescriptor>(nb)},
        dstList(nb) {
    std::vector<NppiImageDescriptor> host[3];
    for (int p = 0; p < 3; ++p) {
      host[p].resize(nb);
    }
    std::vector<NppiImageDescriptor> dst(nb);
    const int cw = w / chromaDiv;
    for (int i = 0; i < nb; ++i) {
      P3Image img;
      img.yStep = w + 5 + i;
      img.cStep = cw + 8;
      img.dStep = 3 * w + 8 + 2 * i;
      img.roiW = (advanced && i > 0) ? w - 2 * i : w;
      img.roiH = (advanced && i > 0) ? h - 2 * i : h;
      img.bufH = h + 8;
      img.y = std::make_unique<DeviceMemory<Npp8u>>((size_t)img.yStep * (h + 8));
      img.y->copyFromHost(make_uniform<Npp8u>((size_t)img.yStep * (h + 8), seed + 13 * i));
      const size_t chromaBytes = (size_t)(cw + 8) * (h + 8);
      img.cb = std::make_unique<DeviceMemory<Npp8u>>(chromaBytes);
      img.cb->copyFromHost(make_uniform<Npp8u>(chromaBytes, seed + 29 + 13 * i));
      img.cr = std::make_unique<DeviceMemory<Npp8u>>(chromaBytes);
      img.cr->copyFromHost(make_uniform<Npp8u>(chromaBytes, seed + 43 + 13 * i));
      img.dst = std::make_unique<DeviceMemory<Npp8u>>((size_t)img.dStep * (h + 8));
      img.dst->copyFromHost(make_uniform<Npp8u>((size_t)img.dStep * (h + 8), seed + 71 + 17 * i));
      host[0][i] = descOf(*img.y, img.yStep, w, h);
      host[1][i] = descOf(*img.cb, img.cStep, w, h);
      host[2][i] = descOf(*img.cr, img.cStep, w, h);
      dst[i] = descOf(*img.dst, img.dStep, img.roiW, img.roiH);
      images.push_back(std::move(img));
    }
    for (int p = 0; p < 3; ++p) {
      lists[p].copyFromHost(host[p]);
      ptr[p] = lists[p].get();
    }
    dstList.copyFromHost(dst);
  }
};

// One packed (C3R) batch.
struct PackedBatch {
  std::vector<std::unique_ptr<DeviceMemory<Npp8u>>> src, dst;
  std::vector<int> srcStep, dstStep;
  DeviceMemory<NppiImageDescriptor> srcList, dstList;

  PackedBatch(int nb, int w, int h, bool advanced, uint64_t seed) : srcList(nb), dstList(nb) {
    std::vector<NppiImageDescriptor> host(nb), dstDesc(nb);
    for (int i = 0; i < nb; ++i) {
      const int ss = 3 * w + 6 + i;
      const int ds = 3 * w + 8 + i;
      const int rw = (advanced && i > 0) ? w - 2 * i : w;
      const int rh = (advanced && i > 0) ? h - 2 * i : h;
      auto s = std::make_unique<DeviceMemory<Npp8u>>((size_t)ss * (h + 8));
      s->copyFromHost(make_uniform<Npp8u>((size_t)ss * (h + 8), seed + 13 * i));
      auto d = std::make_unique<DeviceMemory<Npp8u>>((size_t)ds * (h + 8));
      d->copyFromHost(make_uniform<Npp8u>((size_t)ds * (h + 8), seed + 71 + 17 * i));
      host[i] = descOf(*s, ss, w, h);
      dstDesc[i] = descOf(*d, ds, rw, rh);
      srcStep.push_back(ss);
      dstStep.push_back(ds);
      src.push_back(std::move(s));
      dst.push_back(std::move(d));
    }
    srcList.copyFromHost(host);
    dstList.copyFromHost(dstDesc);
  }
};

void captureP3(const std::string &id, const char *fname, NppStatus st, const PlanarBatch &b) {
  Case c(id, fname);
  c.status(st);
  for (const auto &img : b.images) {
    c.device_output<Npp8u>(img.dst->get(), (size_t)img.dStep * img.bufH, ValueKind::U8);
  }
  c.finish();
}

void capturePacked(const std::string &id, const char *fname, NppStatus st, const PackedBatch &b, int h) {
  Case c(id, fname);
  c.status(st);
  for (size_t i = 0; i < b.dst.size(); ++i) {
    c.device_output<Npp8u>(b.dst[i]->get(), (size_t)b.dstStep[i] * (h + 8), ValueKind::U8);
  }
  c.finish();
}

using P3Fn = NppStatus (*)(const NppiImageDescriptor *const *, NppiImageDescriptor *, int, NppiSize, NppStreamContext);
using PackedFn = NppStatus (*)(const NppiImageDescriptor *, NppiImageDescriptor *, int, NppiSize, NppStreamContext);

void runP3(const char *fname, const std::string &key, P3Fn f, int w, int h, int chromaDiv, bool advanced,
           uint64_t seed, const std::string &idPrefix = "nppiC3Batch/") {
  const int nb = 3;
  PlanarBatch b(nb, w, h, chromaDiv, advanced, seed);
  clear_cuda_error();
  NppStatus st = f(b.ptr, b.dstList.get(), nb, NppiSize{w, h}, current_ctx());
  captureP3(idPrefix + key, fname, st, b);
}

void runPacked(const char *fname, const std::string &key, PackedFn f, int w, int h, bool advanced, uint64_t seed) {
  const int nb = 3;
  PackedBatch b(nb, w, h, advanced, seed);
  clear_cuda_error();
  NppStatus st = f(b.srcList.get(), b.dstList.get(), nb, NppiSize{w, h}, current_ctx());
  capturePacked("nppiC3Batch/" + key, fname, st, b, h);
}

} // namespace

TEST(NppiC3BatchStdDiff, YCbCr420) {
  runP3("nppiYCbCr420ToBGRBatch_8u_P3C3R_Ctx", "ycbcr420bgr/16x12", nppiYCbCr420ToBGRBatch_8u_P3C3R_Ctx, 16,
        12, 2, false, 0xB301);
  runP3("nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced_Ctx", "ycbcr420bgr/adv", nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB302);
  runP3("nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx", "ycbcr420rgb/16x12", nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx, 16,
        12, 2, false, 0xB303);
  runP3("nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx", "ycbcr420rgb/adv", nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB304);
}

TEST(NppiC3BatchStdDiff, YUV420) {
  runP3("nppiYUV420ToBGRBatch_8u_P3C3R_Ctx", "yuv420bgr/16x12", nppiYUV420ToBGRBatch_8u_P3C3R_Ctx, 16, 12,
        2, false, 0xB305);
  runP3("nppiYUV420ToBGRBatch_8u_P3C3R_Advanced_Ctx", "yuv420bgr/adv", nppiYUV420ToBGRBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB306);
  runP3("nppiYUV420ToRGBBatch_8u_P3C3R_Ctx", "yuv420rgb/16x12", nppiYUV420ToRGBBatch_8u_P3C3R_Ctx, 16, 12,
        2, false, 0xB307);
  runP3("nppiYUV420ToRGBBatch_8u_P3C3R_Advanced_Ctx", "yuv420rgb/adv", nppiYUV420ToRGBBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB308);
}

TEST(NppiC3BatchStdDiff, YCbCr422) {
  runP3("nppiYCbCr422ToBGRBatch_8u_P3C3R_Ctx", "ycbcr422bgr/16x12", nppiYCbCr422ToBGRBatch_8u_P3C3R_Ctx, 16,
        12, 2, false, 0xB309);
  runP3("nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced_Ctx", "ycbcr422bgr/adv", nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB30A);
  runP3("nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx", "ycbcr422rgb/16x12", nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx, 16,
        12, 2, false, 0xB30B);
  runP3("nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced_Ctx", "ycbcr422rgb/adv", nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB30C);
}

TEST(NppiC3BatchStdDiff, YUV422) {
  runP3("nppiYUV422ToBGRBatch_8u_P3C3R_Ctx", "yuv422bgr/16x12", nppiYUV422ToBGRBatch_8u_P3C3R_Ctx, 16, 12,
        2, false, 0xB30D);
  runP3("nppiYUV422ToBGRBatch_8u_P3C3R_Advanced_Ctx", "yuv422bgr/adv", nppiYUV422ToBGRBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB30E);
  runP3("nppiYUV422ToRGBBatch_8u_P3C3R_Ctx", "yuv422rgb/16x12", nppiYUV422ToRGBBatch_8u_P3C3R_Ctx, 16, 12,
        2, false, 0xB30F);
  runP3("nppiYUV422ToRGBBatch_8u_P3C3R_Advanced_Ctx", "yuv422rgb/adv", nppiYUV422ToRGBBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 2, true, 0xB310);
}

TEST(NppiC3BatchStdDiff, PackedFamilies) {
  runPacked("nppiYCbCrToBGRBatch_8u_C3R_Ctx", "ycbcrbgr/c3/16x12", nppiYCbCrToBGRBatch_8u_C3R_Ctx, 16, 12,
            false, 0xB311);
  runPacked("nppiYCbCrToBGRBatch_8u_C3R_Advanced_Ctx", "ycbcrbgr/c3/adv",
            nppiYCbCrToBGRBatch_8u_C3R_Advanced_Ctx, 16, 12, true, 0xB312);
  runPacked("nppiYCbCrToRGBBatch_8u_C3R_Ctx", "ycbcrrgb/c3/16x12", nppiYCbCrToRGBBatch_8u_C3R_Ctx, 16, 12,
            false, 0xB313);
  runPacked("nppiYCbCrToRGBBatch_8u_C3R_Advanced_Ctx", "ycbcrrgb/c3/adv",
            nppiYCbCrToRGBBatch_8u_C3R_Advanced_Ctx, 16, 12, true, 0xB314);
  runPacked("nppiYUVToBGRBatch_8u_C3R_Ctx", "yuvbgr/c3/16x12", nppiYUVToBGRBatch_8u_C3R_Ctx, 16, 12, false,
            0xB315);
  runPacked("nppiYUVToBGRBatch_8u_C3R_Advanced_Ctx", "yuvbgr/c3/adv", nppiYUVToBGRBatch_8u_C3R_Advanced_Ctx, 16,
            12, true, 0xB316);
  runPacked("nppiYUVToRGBBatch_8u_C3R_Ctx", "yuvrgb/c3/16x12", nppiYUVToRGBBatch_8u_C3R_Ctx, 16, 12, false,
            0xB317);
  runPacked("nppiYUVToRGBBatch_8u_C3R_Advanced_Ctx", "yuvrgb/c3/adv", nppiYUVToRGBBatch_8u_C3R_Advanced_Ctx, 16,
            12, true, 0xB318);
}

TEST(NppiC3BatchStdDiff, PlanarFamilies) {
  runP3("nppiYCbCrToBGRBatch_8u_P3C3R_Ctx", "ycbcrbgr/p3/16x12", nppiYCbCrToBGRBatch_8u_P3C3R_Ctx, 16, 12,
        1, false, 0xB319);
  runP3("nppiYCbCrToBGRBatch_8u_P3C3R_Advanced_Ctx", "ycbcrbgr/p3/adv", nppiYCbCrToBGRBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 1, true, 0xB31A);
  runP3("nppiYCbCrToRGBBatch_8u_P3C3R_Ctx", "ycbcrrgb/p3/16x12", nppiYCbCrToRGBBatch_8u_P3C3R_Ctx, 16, 12,
        1, false, 0xB31B);
  runP3("nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx", "ycbcrrgb/p3/adv", nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx,
        16, 12, 1, true, 0xB31C);
  runP3("nppiYUVToBGRBatch_8u_P3C3R_Ctx", "yuvbgr/p3/16x12", nppiYUVToBGRBatch_8u_P3C3R_Ctx, 16, 12, 1,
        false, 0xB31D);
  runP3("nppiYUVToBGRBatch_8u_P3C3R_Advanced_Ctx", "yuvbgr/p3/adv", nppiYUVToBGRBatch_8u_P3C3R_Advanced_Ctx, 16,
        12, 1, true, 0xB31E);
  runP3("nppiYUVToRGBBatch_8u_P3C3R_Ctx", "yuvrgb/p3/16x12", nppiYUVToRGBBatch_8u_P3C3R_Ctx, 16, 12, 1,
        false, 0xB31F);
  runP3("nppiYUVToRGBBatch_8u_P3C3R_Advanced_Ctx", "yuvrgb/p3/adv", nppiYUVToRGBBatch_8u_P3C3R_Advanced_Ctx, 16,
        12, 1, true, 0xB320);
}

TEST(NppiC3BatchDiff, ErrorPaths) {
  const int w = 16, h = 12, nb = 3;
  PlanarBatch b(nb, w, h, 2, false, 0xB340);
  PackedBatch pk(nb, w, h, false, 0xB341);
  NppStreamContext ctx = current_ctx();

  std::vector<NppiImageDescriptor> l0, l1, l2, ld;
  b.lists[0].copyToHost(l0);
  b.lists[1].copyToHost(l1);
  b.lists[2].copyToHost(l2);
  b.dstList.copyToHost(ld);

  auto upload = [](const std::vector<NppiImageDescriptor> &v) {
    auto m = std::make_unique<DeviceMemory<NppiImageDescriptor>>(v.size());
    m->copyFromHost(v);
    return m;
  };

  auto probe = [&](const std::string &tag, const char *fname, auto &&call) {
    clear_cuda_error();
    NppStatus st = call();
    Case c("nppiC3Batch/err/" + tag, fname);
    c.status(st);
    c.finish();
  };

  const NppiImageDescriptor *ok[3] = {b.ptr[0], b.ptr[1], b.ptr[2]};
  const NppiSize roi{w, h};

  // The reference dereferences the batch list pointers without validating
  // them (a null list segfaults), so there is nothing to record for those
  // paths; the implementation still returns NPP_NULL_POINTER_ERROR there.
  probe("batch_zero", "nppiYCbCr420ToRGBBatch_8u_P3C3R",
        [&] { return nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(ok, b.dstList.get(), 0, roi, ctx); });
  probe("batch_one", "nppiYCbCr420ToRGBBatch_8u_P3C3R",
        [&] { return nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(ok, b.dstList.get(), 1, roi, ctx); });
  probe("batch_neg", "nppiYCbCr420ToRGBBatch_8u_P3C3R",
        [&] { return nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(ok, b.dstList.get(), -1, roi, ctx); });
  // Zero and negative ROIs crash the reference batch path (no ROI validation
  // before the kernel launch), so there is no status to record.
  probe("odd_width_420", "nppiYCbCr420ToRGBBatch_8u_P3C3R",
        [&] { return nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(ok, b.dstList.get(), nb, NppiSize{w - 1, h}, ctx); });
  probe("odd_height_420", "nppiYCbCr420ToRGBBatch_8u_P3C3R",
        [&] { return nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(ok, b.dstList.get(), nb, NppiSize{w, h - 1}, ctx); });
  probe("odd_width_422", "nppiYCbCr422ToRGBBatch_8u_P3C3R",
        [&] { return nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx(ok, b.dstList.get(), nb, NppiSize{w - 1, h}, ctx); });
  // A null per-image pData is not validated before launch: the reference
  // returns NPP_NO_ERROR and then faults the kernel.  Our implementation
  // returns NPP_NULL_POINTER_ERROR instead, so this path is not recorded.

  // The reference does not validate per-image steps or descriptor image sizes
  // (it returns NPP_NO_ERROR for nStep == 0 and for an oSize smaller than the
  // ROI).  Those paths write through an invalid step, so they are not part of
  // the recorded contract; MPP validates them and returns NPP_STEP_ERROR /
  // NPP_SIZE_ERROR before touching memory.
  probe("adv_max_small", "nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced",
        [&] {
          return nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx(ok, b.dstList.get(), nb, NppiSize{8, 8}, ctx);
        });
  probe("adv_batch_one", "nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced",
        [&] {
          return nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx(ok, b.dstList.get(), 1, roi, ctx);
        });

  // Packed (C3R) list validation.
  probe("c3_batch_one", "nppiYCbCrToRGBBatch_8u_C3R",
        [&] { return nppiYCbCrToRGBBatch_8u_C3R_Ctx(pk.srcList.get(), pk.dstList.get(), 1, roi, ctx); });
}

TEST(NppiC3BatchDiff, OddExtents) {
  // Odd extents are processed up to the largest even rectangle and report
  // NPP_DOUBLE_SIZE_WARNING (the recorded goldens pin the converted extent).
  runP3("nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx", "ycbcr420rgb/odd15x11",
        nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx, 15, 11, 2, false, 0xB330);
  runP3("nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx", "ycbcr420rgb/odd16x11",
        nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx, 16, 11, 2, false, 0xB331);
  runP3("nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx", "ycbcr422rgb/odd15x12",
        nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx, 15, 12, 2, false, 0xB332);
  runP3("nppiYUV422ToRGBBatch_8u_P3C3R_Ctx", "yuv422rgb/odd15x12",
        nppiYUV422ToRGBBatch_8u_P3C3R_Ctx, 15, 12, 2, false, 0xB333);
}

// The planar batch entry points read every source plane with its own
// descriptor nStep (the single-image P3C3R entry points use one step for all
// planes, which is why the batch executor has dedicated three-step kernels).
TEST(NppiC3BatchDiff, PerPlaneSteps) {
  const int w = 16, h = 12, nb = 2;
  const int sy = w + 5, sc = w + 9, sr = w + 13;
  const int dy = 3 * w + 8;
  auto y0 = std::make_unique<DeviceMemory<Npp8u>>((size_t)sy * (h + 4));
  auto c0 = std::make_unique<DeviceMemory<Npp8u>>((size_t)sc * (h + 4));
  auto r0 = std::make_unique<DeviceMemory<Npp8u>>((size_t)sr * (h + 4));
  auto y1 = std::make_unique<DeviceMemory<Npp8u>>((size_t)sy * (h + 4));
  auto c1 = std::make_unique<DeviceMemory<Npp8u>>((size_t)sc * (h + 4));
  auto r1 = std::make_unique<DeviceMemory<Npp8u>>((size_t)sr * (h + 4));
  y0->copyFromHost(make_uniform<Npp8u>((size_t)sy * (h + 4), 0xC001));
  c0->copyFromHost(make_uniform<Npp8u>((size_t)sc * (h + 4), 0xC002));
  r0->copyFromHost(make_uniform<Npp8u>((size_t)sr * (h + 4), 0xC003));
  y1->copyFromHost(make_uniform<Npp8u>((size_t)sy * (h + 4), 0xC004));
  c1->copyFromHost(make_uniform<Npp8u>((size_t)sc * (h + 4), 0xC005));
  r1->copyFromHost(make_uniform<Npp8u>((size_t)sr * (h + 4), 0xC006));
  DeviceMemory<NppiImageDescriptor> ly(nb), lc(nb), lr(nb), ld(nb);
  std::vector<NppiImageDescriptor> vy(nb), vc(nb), vr(nb), vd(nb);
  vy[0] = {y0->get(), sy, {w, h}};
  vc[0] = {c0->get(), sc, {w, h}};
  vr[0] = {r0->get(), sr, {w, h}};
  vy[1] = {y1->get(), sy, {w, h}};
  vc[1] = {c1->get(), sc, {w, h}};
  vr[1] = {r1->get(), sr, {w, h}};
  DeviceMemory<Npp8u> db((size_t)dy * (h + 4)), d1((size_t)dy * (h + 4)), d2((size_t)dy * (h + 4)),
      d3((size_t)dy * (h + 4));
  db.copyFromHost(make_uniform<Npp8u>((size_t)dy * (h + 4), 0xC010));
  d1.copyFromHost(make_uniform<Npp8u>((size_t)dy * (h + 4), 0xC011));
  d2.copyFromHost(make_uniform<Npp8u>((size_t)dy * (h + 4), 0xC012));
  d3.copyFromHost(make_uniform<Npp8u>((size_t)dy * (h + 4), 0xC013));
  vd[0] = {db.get(), dy, {w, h}};
  vd[1] = {db.get(), dy, {w, h}};
  ly.copyFromHost(vy); lc.copyFromHost(vc); lr.copyFromHost(vr); ld.copyFromHost(vd);
  const NppiImageDescriptor *lp[3] = {ly.get(), lc.get(), lr.get()};
  const Npp8u *sp[3] = {y0->get(), c0->get(), r0->get()};
  NppStreamContext ctx = current_ctx();
  NppStatus st = nppiYCbCrToRGBBatch_8u_P3C3R_Ctx(lp, ld.get(), nb, {w, h}, ctx);
  NppStatus s1 = nppiYCbCrToRGB_8u_P3C3R_Ctx(sp, sy, d1.get(), dy, {w, h}, ctx);
  NppStatus s2 = nppiYCbCrToRGB_8u_P3C3R_Ctx(sp, sc, d2.get(), dy, {w, h}, ctx);
  NppStatus s3 = nppiYCbCrToRGB_8u_P3C3R_Ctx(sp, sr, d3.get(), dy, {w, h}, ctx);
  Case c("nppiC3Batch/perplanestep", "nppiYCbCrToRGBBatch_8u_P3C3R_Ctx");
  c.status(st);
  c.status(s1);
  c.status(s2);
  c.status(s3);
  c.device_output<Npp8u>(db.get(), (size_t)dy * (h + 4), ValueKind::U8);
  c.device_output<Npp8u>(d1.get(), (size_t)dy * (h + 4), ValueKind::U8);
  c.device_output<Npp8u>(d2.get(), (size_t)dy * (h + 4), ValueKind::U8);
  c.device_output<Npp8u>(d3.get(), (size_t)dy * (h + 4), ValueKind::U8);
  c.finish();
}
