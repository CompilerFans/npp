// Differential tests for package f13 family 4 (part):
// nppiCompressMarkerLabelsUFBatch_32u_C1IR(+Ctx) and the Advanced form.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiCompressMarkerLabelsDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiCompressMarkerLabelsDiff.*'
//
// Measured semantics (5090, NPP 13.4, probes f13_probe32..36):
//  - the compressed labels are renumbered from 1 with no gaps; pNewMaxLabelID
//    is the largest label after compression (== the number of distinct labels)
//  - the non-Advanced form applies the common oSizeROI to every image and
//    ignores the per-image descriptor oSize/nStep; the Advanced form uses the
//    per-image oSize
//  - a batch size of 1 is accepted (the header says > 1, NPP does not enforce
//    it); a negative batch/ROI is NPP_SIZE_ERROR and a zero one surfaces as
//    NPP_CUDA_KERNEL_EXECUTION_ERROR
//  - null list/buffer/newMax pointers are NPP_NULL_POINTER_ERROR
//  - a non-positive nPerImageBufferSize is NPP_CUDA_KERNEL_EXECUTION_ERROR

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <string>
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

// NPP's CompressMarkerLabelsGetBufferSize is 65556 + 8 * labelCount.  The diff
// tests size the per-image buffers explicitly so record and verify see the
// same descriptor values.
int buffer_size(int pixels) { return 65556 + 8 * pixels; }

struct Batch {
  std::vector<char *> data;
  std::vector<int> steps;
  std::vector<NppiSize> sizes;
};

// Renders a region pattern into an 8u source image.  The compress entry points
// are only defined for real nppiLabelMarkersUF output, so every value case runs
// label markers first and compresses the labels it produced.
std::vector<Npp8u> make_source(int W, int H, int images) {
  std::vector<Npp8u> v((size_t)W * H * images);
  for (int img = 0; img < images; ++img)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        v[(size_t)img * W * H + y * W + x] =
            (Npp8u)((((x + img) / 3) * 2 + ((y + img) / 2) * 3) % 4 * 60);
  return v;
}

template <typename F>
void value_case(const std::string &fn, const std::string &key, int W, int H,
                const std::vector<NppiSize> &bufSizes, const std::vector<NppiSize> &descSizes,
                NppiSize roi, bool advanced, F f) {
  const int nImg = (int)bufSizes.size();
  DeviceMemory<Npp8u> &lmbuf = arena<Npp8u>(1 << 20);
  const int bufBytes = buffer_size(W * H);
  DeviceMemory<Npp8u> &bufs = arena<Npp8u>((size_t)bufBytes * nImg);
  std::vector<DeviceMemory<Npp32u> *> labels;
  std::vector<NppiImageDescriptor> himg;
  std::vector<NppiBufferDescriptor> hbuf;
  std::vector<Npp32u> nmHost((size_t)nImg, 0xAAAA5555u);
  for (int i = 0; i < nImg; ++i) {
    const int iw = bufSizes[i].width, ih = bufSizes[i].height;
    auto host = make_source(iw, ih, 1);
    DeviceMemory<Npp8u> &src = arena<Npp8u>((size_t)iw * ih);
    src.copyFromHost(host);
    labels.push_back(&arena<Npp32u>((size_t)iw * ih));
    labels.back()->copyFromHost(std::vector<Npp32u>((size_t)iw * ih, 0));
    clear_cuda_error();
    // Real UF labels over the buffer ROI (the only documented input for the
    // compress entry points).
    NppStatus ls = nppiLabelMarkersUF_8u32u_C1R_Ctx(src.get(), iw, labels.back()->get(), iw * 4,
                                                    bufSizes[i], nppiNormL1, lmbuf.get(),
                                                    current_ctx());
    ASSERT_EQ(ls, NPP_NO_ERROR) << "label markers failed for image " << i;
    himg.push_back({(char *)labels.back()->get(), iw * 4, descSizes[i]});
    hbuf.push_back({(char *)bufs.get() + (size_t)bufBytes * i, bufBytes});
  }
  DeviceMemory<Npp32u> &nm = arena<Npp32u>((size_t)nImg);
  nm.copyFromHost(nmHost);
  DeviceMemory<NppiImageDescriptor> &dimg = arena<NppiImageDescriptor>((size_t)nImg);
  DeviceMemory<NppiBufferDescriptor> &dbuf = arena<NppiBufferDescriptor>((size_t)nImg);
  dimg.copyFromHost(himg);
  dbuf.copyFromHost(hbuf);
  clear_cuda_error();
  NppStatus st = f(dimg.get(), dbuf.get(), nm.get(), nImg, roi, advanced, current_ctx());
  Case c(fn + "/" + key, fn);
  c.status(st);
  for (int i = 0; i < nImg; ++i) {
    c.device_output<Npp32u>(labels[i]->get(), (size_t)bufSizes[i].width * bufSizes[i].height,
                            ValueKind::U32, 0.0, 0.0);
  }
  c.device_output<Npp32u>(nm.get(), (size_t)nImg, ValueKind::U32, 0.0, 0.0);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) ADD_FAILURE() << "async CUDA fault after " << key << ": "
                                      << cudaGetErrorString(e);
}

using BatchFn = NppStatus (*)(NppiImageDescriptor *, NppiBufferDescriptor *, unsigned int *, int,
                              NppiSize, int, NppStreamContext);

template <typename F> void status_case(const std::string &key, NppiSize roi, int batch,
                                       int bufParam, int variant, F f) {
  const int W = 10, H = 7, nImg = 2;
  DeviceMemory<Npp32u> &d = arena<Npp32u>((size_t)W * H * nImg);
  d.copyFromHost(std::vector<Npp32u>((size_t)W * H * nImg, 1));
  const int bufBytes = buffer_size(W * H);
  DeviceMemory<Npp8u> &bufs = arena<Npp8u>((size_t)bufBytes * nImg);
  DeviceMemory<Npp32u> &nm = arena<Npp32u>(nImg);
  std::vector<NppiImageDescriptor> himg = {
      {(char *)d.get(), W * 4, {W, H}},
      {(char *)d.get() + (size_t)W * H * 4, W * 4, {W, H}}};
  std::vector<NppiBufferDescriptor> hbuf = {{(char *)bufs.get(), bufBytes},
                                            {(char *)bufs.get() + bufBytes, bufBytes}};
  DeviceMemory<NppiImageDescriptor> &dimg = arena<NppiImageDescriptor>(nImg);
  DeviceMemory<NppiBufferDescriptor> &dbuf = arena<NppiBufferDescriptor>(nImg);
  dimg.copyFromHost(himg);
  dbuf.copyFromHost(hbuf);
  clear_cuda_error();
  NppStatus st = f(variant == 0 ? nullptr : dimg.get(), variant == 1 ? nullptr : dbuf.get(),
                   variant == 2 ? nullptr : nm.get(), batch, roi, bufParam, current_ctx());
  Case c("nppiCompressMarkerLabelsUFBatch_32u_C1IR/" + key, "nppiCompressMarkerLabelsUFBatch_32u_C1IR");
  c.status(st);
  c.finish();
  // A failed NPP call may still enqueue an invalid kernel; do not propagate
  // that CUDA error to the next case.
  cudaGetLastError();
}

}  // namespace

TEST(NppiCompressMarkerLabelsDiff, Values) {
  const int W = 14, H = 10;
  for (int advanced = 0; advanced < 2; ++advanced) {
    const bool adv = advanced != 0;
    const std::string n = adv ? "adv" : "plain";
    auto call = [](NppiImageDescriptor *s, NppiBufferDescriptor *b, unsigned int *nm, int nb,
                   NppiSize roi, bool advanced, NppStreamContext ctx) {
      return advanced ? nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(
                           s, b, nm, nb, roi, roi.width * roi.height, ctx)
                     : nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(s, b, nm, nb, roi,
                                                                    roi.width * roi.height, ctx);
    };
    value_case("nppiCompressMarkerLabelsUFBatch_32u_C1IR", n + "/same-roi", W, H,
               {{W, H}, {W, H}}, {{W, H}, {W, H}}, {W, H}, adv, call);
    if (!adv) {
      // The plain form ignores the per-image descriptor ROI and step.
      value_case("nppiCompressMarkerLabelsUFBatch_32u_C1IR", n + "/ignored-desc-roi", W, H,
                 {{W, H}, {W, H}}, {{W, H}, {W - 4, H - 3}}, {W, H}, adv, call);
    } else {
      // The Advanced form honours per-image descriptor ROIs.
      value_case("nppiCompressMarkerLabelsUFBatch_32u_C1IR", n + "/mixed-roi", W, H,
                 {{W, H}, {W - 4, H - 3}}, {{W, H}, {W - 4, H - 3}}, {W, H}, adv, call);
    }
    // Batch size 1 is accepted (NPP leaves the second new-max entry alone).
    value_case("nppiCompressMarkerLabelsUFBatch_32u_C1IR", n + "/batch1", W, H, {{W, H}},
               {{W, H}}, {W, H}, adv, call);
  }
}

TEST(NppiCompressMarkerLabelsDiff, Errors) {
  const int W = 10, H = 7;
  for (int advanced = 0; advanced < 2; ++advanced) {
    const bool adv = advanced != 0;
    const std::string n = adv ? "/adv" : "/plain";
    auto run = [&](const std::string &key, NppiSize roi, int batch, int bufParam, int variant) {
      status_case(n + "/" + key, roi, batch, bufParam, variant,
                  [adv](NppiImageDescriptor *s, NppiBufferDescriptor *b, unsigned int *nm, int nb,
                        NppiSize r, int buf, NppStreamContext ctx) {
                    return adv ? nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(s, b, nm, nb, r,
                                                                                       buf, ctx)
                               : nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(s, b, nm, nb, r, buf,
                                                                              ctx);
                  });
    };
    run("null-list", {W, H}, 2, buffer_size(W * H), 0);
    run("null-buffers", {W, H}, 2, buffer_size(W * H), 1);
    run("null-newmax", {W, H}, 2, buffer_size(W * H), 2);
    run("roi-negative", {-1, H}, 2, buffer_size(W * H), 3);
    run("roi-zero", {0, H}, 2, buffer_size(W * H), 3);
    run("batch-negative", {W, H}, -1, buffer_size(W * H), 3);
    run("batch-zero", {W, H}, 0, buffer_size(W * H), 3);
    run("bufparam-zero", {W, H}, 2, 0, 3);
    run("bufparam-negative", {W, H}, 2, -8, 3);
  }
}
