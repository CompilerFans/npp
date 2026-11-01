// Differential tests for the package-s2 quality / error metric families:
// QualityIndex (+GetBufferHostSize), CountInRange (+GBHS), MSE/PSNR (+GBHS)
// and the MSE/PSNR batch forms (+GBHS).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiQualityMetricsDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiQualityMetricsDiff.*'
//
// Semantics pinned by the 5090 probes (tools/5090_probes_s2/s2_probe*.cu):
//   * QualityIndex: Q = 4 cov ma mb / ((ma^2+mb^2)(va+vb)) with n-1
//     denominators; C3/AC4 keep the channel-0 means and (va+vb) but a
//     per-channel covariance; a constant channel -> NPP_QUALITY_INDEX_ERROR
//     and no output is written; 1x1 -> NaN.
//   * CountInRange: inclusive host bounds, device int counts, alpha ignored,
//     reversed bounds -> NPP_RANGE_ERROR.
//   * MSE/PSNR: per-channel 8u metrics; PSNR = 10 log10(255^2/MSE).
//   * Buffer-size helpers: exact 5090 values (recorded as u64).

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

constexpr float kSentinel = -1.0e30f;

template <typename T> void prefill(DeviceMemory<T> &mem, T value) {
  std::vector<T> host(mem.size(), value);
  mem.copyFromHost(host);
}

using SizeFn = NppStatus (*)(NppiSize, SIZE_TYPE *, NppStreamContext);

SIZE_TYPE query_size(SizeFn fn, NppiSize roi) {
  SIZE_TYPE bytes = 0;
  EXPECT_EQ(fn(roi, &bytes, current_context()), NPP_NO_ERROR);
  return bytes;
}

void capture_size(const std::string &id, const char *fname, SizeFn fn, NppiSize roi) {
  SIZE_TYPE bytes = (SIZE_TYPE)-1;
  NppStatus st = fn(roi, &bytes, current_context());
  Case c(id, fname);
  c.status(st);
  if (st == NPP_NO_ERROR && roi.width > 0 && roi.height > 0) {
    uint64_t v = (uint64_t)bytes;
    c.host_output(&v, 1, ValueKind::U64);
  }
  c.finish();
}

// Non-Ctx form of a buffer-size helper (no stream context parameter).
using SizeFnPlain = NppStatus (*)(NppiSize, SIZE_TYPE *);

void capture_size(const std::string &id, const char *fname, SizeFnPlain fn, NppiSize roi) {
  SIZE_TYPE bytes = (SIZE_TYPE)-1;
  NppStatus st = fn(roi, &bytes);
  Case c(id, fname);
  c.status(st);
  if (st == NPP_NO_ERROR && roi.width > 0 && roi.height > 0) {
    uint64_t v = (uint64_t)bytes;
    c.host_output(&v, 1, ValueKind::U64);
  }
  c.finish();
}

// Builds a padded image of h rows; returns the host buffer and the byte step.
template <typename T> int make_step(int w, int ch, int pad) {
  const int se = (int)sizeof(T);
  return ((w * ch * se + pad + se - 1) / se) * se;
}

template <typename T> std::vector<T> make_image(int w, int h, int ch, int step, uint64_t seed) {
  const int se = (int)sizeof(T);
  std::vector<T> img((size_t)step / se * h);
  std::vector<T> rnd = make_uniform<T>(img.size(), seed);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w * ch; ++x) {
      img[(size_t)y * (step / se) + x] = rnd[(size_t)y * (step / se) + x];
    }
  }
  return img;
}

// ---- QualityIndex ----------------------------------------------------------
template <typename T>
void qi_case(const char *fname, const std::string &id, int w, int h, int ch, bool alpha, int pad, uint64_t seed,
             NppStatus (*f)(const T *, int, const T *, int, NppiSize, Npp32f *, Npp8u *, NppStreamContext), SizeFn szfn) {
  const int step = make_step<T>(w, ch, pad);
  std::vector<T> a = make_image<T>(w, h, ch, step, seed);
  std::vector<T> b = make_image<T>(w, h, ch, step, seed ^ 0x5DEECE66DULL);
  const int nout = alpha ? ch - 1 : ch;
  DeviceMemory<T> da(a.size()), db(b.size());
  DeviceMemory<Npp32f> out(nout);
  da.copyFromHost(a);
  db.copyFromHost(b);
  prefill(out, kSentinel);
  SIZE_TYPE bytes = query_size(szfn, NppiSize{w, h});
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(da.get(), step, db.get(), step, NppiSize{w, h}, out.get(), buf.get(), current_context());
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(T));
  c.input(b.data(), b.size() * sizeof(T));
  c.status(st);
  c.device_output(out.get(), (size_t)nout, ValueKind::F32, 2e-4, 2e-4);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T>
void qi_case_nc(const char *fname, const std::string &id, int w, int h, int ch, bool alpha, int pad, uint64_t seed,
                NppStatus (*f)(const T *, int, const T *, int, NppiSize, Npp32f *, Npp8u *)) {
  const int step = make_step<T>(w, ch, pad);
  std::vector<T> a = make_image<T>(w, h, ch, step, seed);
  std::vector<T> b = make_image<T>(w, h, ch, step, seed ^ 0x5DEECE66DULL);
  const int nout = alpha ? ch - 1 : ch;
  DeviceMemory<T> da(a.size()), db(b.size());
  DeviceMemory<Npp32f> out(nout);
  da.copyFromHost(a);
  db.copyFromHost(b);
  prefill(out, kSentinel);
  DeviceMemory<Npp8u> buf(1 << 20);
  clear_cuda_error();
  NppStatus st = f(da.get(), step, db.get(), step, NppiSize{w, h}, out.get(), buf.get());
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(T));
  c.input(b.data(), b.size() * sizeof(T));
  c.status(st);
  c.device_output(out.get(), (size_t)nout, ValueKind::F32, 2e-4, 2e-4);
  c.finish();
}
#endif

void qi_error_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

// ---- CountInRange ----------------------------------------------------------
template <typename T>
void cir1_case(const char *fname, const std::string &id, int w, int h, int pad, uint64_t seed, T lo, T hi,
               NppStatus (*f)(const T *, int, NppiSize, int *, T, T, Npp8u *, NppStreamContext), SizeFn szfn) {
  const int step = make_step<T>(w, 1, pad);
  std::vector<T> img = make_image<T>(w, h, 1, step, seed);
  // Pin the inclusive-bound behaviour: plant the bounds in the first pixels.
  img[0] = lo;
  img[1] = hi;
  DeviceMemory<T> d(img.size());
  DeviceMemory<int> cnt(1);
  d.copyFromHost(img);
  cnt.copyFromHost(std::vector<int>{-12345});
  SIZE_TYPE bytes = query_size(szfn, NppiSize{w, h});
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(d.get(), step, NppiSize{w, h}, cnt.get(), lo, hi, buf.get(), current_context());
  Case c(id, fname);
  c.input(img.data(), img.size() * sizeof(T));
  c.status(st);
  c.device_output(cnt.get(), 1, ValueKind::I32);
  c.finish();
}

template <typename T>
void cirn_case(const char *fname, const std::string &id, int w, int h, int ch, bool alpha, int pad, uint64_t seed,
               NppStatus (*f)(const T *, int, NppiSize, int *, T *, T *, Npp8u *, NppStreamContext),
               SizeFn szfn) {
  const int step = make_step<T>(w, ch, pad);
  std::vector<T> img = make_image<T>(w, h, ch, step, seed);
  const int nch = alpha ? ch - 1 : ch;
  std::vector<T> lo(nch), hi(nch);
  for (int k = 0; k < nch; ++k) {
    if constexpr (std::is_floating_point_v<T>) {
      lo[k] = (T)(-0.5 - 0.25 * k);
      hi[k] = (T)(0.5 + 0.25 * k);
    } else {
      lo[k] = (T)(32 + 16 * k);
      hi[k] = (T)(200 - 16 * k);
    }
  }
  DeviceMemory<T> d(img.size()), dlo(nch), dhi(nch);
  DeviceMemory<int> cnt(nch);
  d.copyFromHost(img);
  dlo.copyFromHost(lo);
  dhi.copyFromHost(hi);
  cnt.copyFromHost(std::vector<int>(nch, -12345));
  SIZE_TYPE bytes = query_size(szfn, NppiSize{w, h});
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(d.get(), step, NppiSize{w, h}, cnt.get(), lo.data(), hi.data(), buf.get(), current_context());
  Case c(id, fname);
  c.input(img.data(), img.size() * sizeof(T));
  c.status(st);
  c.device_output(cnt.get(), (size_t)nch, ValueKind::I32);
  c.finish();
}

// ---- MSE / PSNR ------------------------------------------------------------
template <typename T>
void mse_case(const char *fname, const std::string &id, int w, int h, int ch, int pad, uint64_t seed, bool identical,
              bool psnr,
              NppStatus (*f)(const T *, int, const T *, int, NppiSize, Npp32f *, Npp8u *, NppStreamContext),
              SizeFn szfn) {
  const int step = make_step<T>(w, ch, pad);
  std::vector<T> a = make_image<T>(w, h, ch, step, seed);
  std::vector<T> b = identical ? a : make_image<T>(w, h, ch, step, seed ^ 0x9E3779B97F4A7C15ULL);
  DeviceMemory<T> da(a.size()), db(b.size());
  DeviceMemory<Npp32f> out(ch);
  da.copyFromHost(a);
  db.copyFromHost(b);
  prefill(out, kSentinel);
  SIZE_TYPE bytes = query_size(szfn, NppiSize{w, h});
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(da.get(), step, db.get(), step, NppiSize{w, h}, out.get(), buf.get(), current_context());
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(T));
  c.input(b.data(), b.size() * sizeof(T));
  c.status(st);
  c.device_output(out.get(), (size_t)ch, ValueKind::F32, 1e-5, 1e-5);
  (void)psnr;
  c.finish();
}

// ---- batch metrics --------------------------------------------------------
struct Batch {
  std::vector<Npp8u> pool;
  std::vector<NppiImageDescriptor> d1, d2;
};

Batch make_batch(const std::vector<NppiSize> &sizes, int ch, uint64_t seed) {
  Batch b;
  size_t total = 0;
  for (auto s : sizes) total += (size_t)s.width * s.height * ch * sizeof(Npp8u) * 2;
  b.pool.resize(total + 16);
  size_t off = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int w = sizes[i].width, h = sizes[i].height;
    const int step = w * ch;
    NppiImageDescriptor d1{}, d2{};
    std::vector<Npp8u> img = make_image<Npp8u>(w, h, ch, step, seed + i);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w * ch; ++x) b.pool[off + (size_t)y * step + x] = img[(size_t)y * step + x];
    d1.pData = reinterpret_cast<void *>(off);
    d1.nStep = step;
    d1.oSize = NppiSize{w, h};
    off += (size_t)h * step;
    img = make_image<Npp8u>(w, h, ch, step, seed + 100 + i);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w * ch; ++x) b.pool[off + (size_t)y * step + x] = img[(size_t)y * step + x];
    d2.pData = reinterpret_cast<void *>(off);
    d2.nStep = step;
    d2.oSize = NppiSize{w, h};
    off += (size_t)h * step;
    b.d1.push_back(d1);
    b.d2.push_back(d2);
  }
  return b;
}

template <typename Fn>
void batch_case(const char *fname, const std::string &id, const std::vector<NppiSize> &sizes, int ch, NppiSize roi,
                bool advanced, uint64_t seed, Fn call) {
  Batch b = make_batch(sizes, ch, seed);
  DeviceMemory<Npp8u> pool(b.pool.size());
  pool.copyFromHost(b.pool);
  // Rebase descriptor data pointers into the device pool.
  std::vector<NppiImageDescriptor> d1 = b.d1, d2 = b.d2;
  for (size_t i = 0; i < d1.size(); ++i) {
    d1[i].pData = (Npp8u *)pool.get() + (uintptr_t)b.d1[i].pData;
    d2[i].pData = (Npp8u *)pool.get() + (uintptr_t)b.d2[i].pData;
  }
  DeviceMemory<NppiImageDescriptor> dd1(d1.size()), dd2(d2.size());
  dd1.copyFromHost(d1);
  dd2.copyFromHost(d2);
  const int n = (int)sizes.size();
  DeviceMemory<Npp32f> out((size_t)n * ch);
  prefill(out, kSentinel);
  // NPP validates the per-image buffer descriptor list even though the sizes
  // are advisory; hand out one generously sized scratch slice per image.
  const size_t scratchPerImage = (size_t)16 << 20;
  DeviceMemory<Npp8u> scratch(scratchPerImage * (size_t)n);
  std::vector<NppiBufferDescriptor> bufs((size_t)n);
  for (int i = 0; i < n; ++i) {
    bufs[i].pData = scratch.get() + scratchPerImage * (size_t)i;
    bufs[i].nBufferSize = (int)scratchPerImage;
  }
  DeviceMemory<NppiBufferDescriptor> dbufs(bufs.size());
  dbufs.copyFromHost(bufs);
  clear_cuda_error();
  NppStatus st = call(dd1.get(), dd2.get(), n, roi, out.get(), dbufs.get(), advanced);
  Case c(id, fname);
  c.input(b.pool.data(), b.pool.size());
  c.status(st);
  if (st == NPP_NO_ERROR) {
    c.device_output(out.get(), (size_t)n * ch, ValueKind::F32, 1e-5, 1e-5);
  }
  c.finish();
}

}  // namespace

class NppiQualityMetricsDiff : public NppTestBase {};

// ============================ QualityIndex =================================
TEST_F(NppiQualityMetricsDiff, QI_C1_8u) {
  qi_case<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "s2/QI/8u_C1", 17, 13, 1, false, 0, 0x101,
                 &nppiQualityIndex_8u32f_C1R_Ctx, &nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_C1_16u_pad) {
  qi_case<Npp16u>("nppiQualityIndex_16u32f_C1R_Ctx", "s2/QI/16u_C1_pad", 33, 7, 1, false, 6, 0x102,
                  &nppiQualityIndex_16u32f_C1R_Ctx, &nppiQualityIndexGetBufferHostSize_16u32f_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_C1_32f) {
  qi_case<Npp32f>("nppiQualityIndex_32f_C1R_Ctx", "s2/QI/32f_C1", 31, 19, 1, false, 12, 0x103,
                  &nppiQualityIndex_32f_C1R_Ctx, &nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_C3_8u) {
  qi_case<Npp8u>("nppiQualityIndex_8u32f_C3R_Ctx", "s2/QI/8u_C3", 19, 11, 3, false, 0, 0x104,
                 &nppiQualityIndex_8u32f_C3R_Ctx, &nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_C3_16u) {
  qi_case<Npp16u>("nppiQualityIndex_16u32f_C3R_Ctx", "s2/QI/16u_C3", 23, 9, 3, false, 10, 0x105,
                  &nppiQualityIndex_16u32f_C3R_Ctx, &nppiQualityIndexGetBufferHostSize_16u32f_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_C3_32f) {
  qi_case<Npp32f>("nppiQualityIndex_32f_C3R_Ctx", "s2/QI/32f_C3", 21, 15, 3, false, 4, 0x106,
                  &nppiQualityIndex_32f_C3R_Ctx, &nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_AC4_8u) {
  qi_case<Npp8u>("nppiQualityIndex_8u32f_AC4R_Ctx", "s2/QI/8u_AC4", 13, 17, 4, true, 8, 0x107,
                 &nppiQualityIndex_8u32f_AC4R_Ctx, &nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_AC4_32f) {
  qi_case<Npp32f>("nppiQualityIndex_32f_AC4R_Ctx", "s2/QI/32f_AC4", 17, 5, 4, true, 0, 0x108,
                  &nppiQualityIndex_32f_AC4R_Ctx, &nppiQualityIndexGetBufferHostSize_32f_AC4R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_1x1_nan) {
  qi_case<Npp8u>("nppiQualityIndex_8u32f_C1R_Ctx", "s2/QI/1x1", 1, 1, 1, false, 0, 0x109,
                 &nppiQualityIndex_8u32f_C1R_Ctx, &nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, QI_Const_Error) {
  // Constant source -> NPP_QUALITY_INDEX_ERROR and no output written.
  std::vector<Npp8u> a(16 * 4, 7), b(16 * 4, 9);
  DeviceMemory<Npp8u> da(a.size()), db(b.size());
  DeviceMemory<Npp32f> out(1);
  da.copyFromHost(a);
  db.copyFromHost(b);
  prefill(out, kSentinel);
  DeviceMemory<Npp8u> buf(1 << 20);
  NppStatus st =
      nppiQualityIndex_8u32f_C1R_Ctx(da.get(), 16, db.get(), 16, NppiSize{16, 4}, out.get(), buf.get(), current_context());
  qi_error_case("s2/QI/const", "nppiQualityIndex_8u32f_C1R_Ctx", st);
}
TEST_F(NppiQualityMetricsDiff, QI_Errors) {
  NppStreamContext ctx = current_context();
  DeviceMemory<Npp8u> a(64), b(64);
  DeviceMemory<Npp32f> out(4);
  DeviceMemory<Npp8u> buf(1 << 16);
  a.copyFromHost(std::vector<Npp8u>(64, 3));
  b.copyFromHost(std::vector<Npp8u>(64, 5));
  qi_error_case("s2/QI/err/null1", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(nullptr, 8, b.get(), 8, NppiSize{8, 8}, out.get(), buf.get(), ctx));
  qi_error_case("s2/QI/err/null2", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(a.get(), 8, nullptr, 8, NppiSize{8, 8}, out.get(), buf.get(), ctx));
  qi_error_case("s2/QI/err/nullout", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(a.get(), 8, b.get(), 8, NppiSize{8, 8}, nullptr, buf.get(), ctx));
  qi_error_case("s2/QI/err/nullbuf", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(a.get(), 8, b.get(), 8, NppiSize{8, 8}, out.get(), nullptr, ctx));
  qi_error_case("s2/QI/err/zero", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(a.get(), 8, b.get(), 8, NppiSize{0, 8}, out.get(), buf.get(), ctx));
  qi_error_case("s2/QI/err/neg", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(a.get(), 8, b.get(), 8, NppiSize{-1, 8}, out.get(), buf.get(), ctx));
  qi_error_case("s2/QI/err/step", "nppiQualityIndex_8u32f_C1R_Ctx",
                nppiQualityIndex_8u32f_C1R_Ctx(a.get(), 4, b.get(), 8, NppiSize{8, 8}, out.get(), buf.get(), ctx));
}
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiQualityMetricsDiff, QI_C1_8u_Plain) {
  qi_case_nc<Npp8u>("nppiQualityIndex_8u32f_C1R", "s2/QI/8u_C1", 17, 13, 1, false, 0, 0x101,
                    &nppiQualityIndex_8u32f_C1R);
}
#endif

TEST_F(NppiQualityMetricsDiff, QI_GBHS) {
  struct R { int w, h; };
  const R rois[] = {{1, 1}, {2, 3}, {17, 13}, {32, 32}, {64, 64}, {256, 64}, {640, 480}, {4096, 1}, {1, 4096}};
  for (auto r : rois) {
    const std::string base = std::to_string(r.w) + "x" + std::to_string(r.h);
    capture_size("s2/QI/gbhs/8u_C1/" + base, "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx",
                 &nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/QI/gbhs/8u_C3/" + base, "nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx",
                 &nppiQualityIndexGetBufferHostSize_8u32f_C3R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/QI/gbhs/8u_AC4/" + base, "nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx",
                 &nppiQualityIndexGetBufferHostSize_8u32f_AC4R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/QI/gbhs/32f_C1/" + base, "nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx",
                 &nppiQualityIndexGetBufferHostSize_32f_C1R_Ctx, NppiSize{r.w, r.h});
  }
  capture_size("s2/QI/gbhs/zero", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx",
               &nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx, NppiSize{0, 8});
  capture_size("s2/QI/gbhs/neg", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx",
               &nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx, NppiSize{-1, 8});
  {
    SIZE_TYPE bytes = 0;
    NppStatus st = nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(NppiSize{8, 8}, (SIZE_TYPE *)nullptr, current_context());
    Case c("s2/QI/gbhs/null", "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx");
    c.status(st);
    c.finish();
  }
}
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiQualityMetricsDiff, QI_GBHS_Plain) {
  capture_size("s2/QI/gbhs/8u_C1/17x13", "nppiQualityIndexGetBufferHostSize_8u32f_C1R",
               &nppiQualityIndexGetBufferHostSize_8u32f_C1R, NppiSize{17, 13});
}
#endif

// ============================ CountInRange =================================
TEST_F(NppiQualityMetricsDiff, CIR_8u_C1) {
  cir1_case<Npp8u>("nppiCountInRange_8u_C1R_Ctx", "s2/CIR/8u_C1", 23, 9, 0, 0x201, 32, 200,
                   &nppiCountInRange_8u_C1R_Ctx, &nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_8u_C1_pad) {
  cir1_case<Npp8u>("nppiCountInRange_8u_C1R_Ctx", "s2/CIR/8u_C1_pad", 31, 7, 5, 0x202, 64, 128,
                   &nppiCountInRange_8u_C1R_Ctx, &nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_32f_C1) {
  cir1_case<Npp32f>("nppiCountInRange_32f_C1R_Ctx", "s2/CIR/32f_C1", 19, 5, 0, 0x203, -0.5f, 0.5f,
                    &nppiCountInRange_32f_C1R_Ctx, &nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_8u_C3) {
  cirn_case<Npp8u>("nppiCountInRange_8u_C3R_Ctx", "s2/CIR/8u_C3", 17, 11, 3, false, 4, 0x204,
                   &nppiCountInRange_8u_C3R_Ctx, &nppiCountInRangeGetBufferHostSize_8u_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_8u_AC4) {
  cirn_case<Npp8u>("nppiCountInRange_8u_AC4R_Ctx", "s2/CIR/8u_AC4", 15, 13, 4, true, 0, 0x205,
                   &nppiCountInRange_8u_AC4R_Ctx, &nppiCountInRangeGetBufferHostSize_8u_AC4R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_32f_C3) {
  cirn_case<Npp32f>("nppiCountInRange_32f_C3R_Ctx", "s2/CIR/32f_C3", 21, 7, 3, false, 8, 0x206,
                    &nppiCountInRange_32f_C3R_Ctx, &nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_32f_AC4) {
  cirn_case<Npp32f>("nppiCountInRange_32f_AC4R_Ctx", "s2/CIR/32f_AC4", 11, 9, 4, true, 0, 0x207,
                    &nppiCountInRange_32f_AC4R_Ctx, &nppiCountInRangeGetBufferHostSize_32f_AC4R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, CIR_Errors) {
  NppStreamContext ctx = current_context();
  DeviceMemory<Npp8u> a(64);
  DeviceMemory<int> cnt(3);
  DeviceMemory<Npp8u> buf(1 << 16);
  Npp8u lo3[3] = {10, 0, 200}, hi3[3] = {100, 10, 255};
  Npp8u lo[3] = {10, 0, 200};
  qi_error_case("s2/CIR/err/null", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(nullptr, 8, NppiSize{8, 8}, cnt.get(), 0, 255, buf.get(), ctx));
  qi_error_case("s2/CIR/err/zero", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(a.get(), 8, NppiSize{0, 8}, cnt.get(), 0, 255, buf.get(), ctx));
  qi_error_case("s2/CIR/err/neg", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(a.get(), 8, NppiSize{-1, 8}, cnt.get(), 0, 255, buf.get(), ctx));
  qi_error_case("s2/CIR/err/rev", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(a.get(), 8, NppiSize{8, 8}, cnt.get(), 200, 100, buf.get(), ctx));
  qi_error_case("s2/CIR/err/eq", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(a.get(), 8, NppiSize{8, 8}, cnt.get(), 100, 100, buf.get(), ctx));
  qi_error_case("s2/CIR/err/step", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(a.get(), 4, NppiSize{8, 8}, cnt.get(), 0, 255, buf.get(), ctx));
  qi_error_case("s2/CIR/err/c3rev", "nppiCountInRange_8u_C3R_Ctx",
                nppiCountInRange_8u_C3R_Ctx(a.get(), 24, NppiSize{8, 8}, cnt.get(), hi3, lo, buf.get(), ctx));
  qi_error_case("s2/CIR/err/nullbuf", "nppiCountInRange_8u_C1R_Ctx",
                nppiCountInRange_8u_C1R_Ctx(a.get(), 8, NppiSize{8, 8}, cnt.get(), 0, 255, nullptr, ctx));
  qi_error_case("s2/CIR/err/nullbuf_c3", "nppiCountInRange_8u_C3R_Ctx",
                nppiCountInRange_8u_C3R_Ctx(a.get(), 24, NppiSize{8, 8}, cnt.get(), lo3, hi3, nullptr, ctx));
}
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiQualityMetricsDiff, CIR_8u_C1_Plain) {
  // Replay the Ctx case so the non-Ctx adapter is checked against the same
  // golden (NPP 13 no longer ships non-Ctx entry points to record from).
  const int w = 23, h = 9, step = w;
  std::vector<Npp8u> img = make_image<Npp8u>(w, h, 1, step, 0x201);
  img[0] = 32;
  img[1] = 200;
  DeviceMemory<Npp8u> d(img.size());
  DeviceMemory<int> cnt(1);
  DeviceMemory<Npp8u> buf(1 << 16);
  d.copyFromHost(img);
  cnt.copyFromHost(std::vector<int>{-12345});
  NppStatus st = nppiCountInRange_8u_C1R(d.get(), step, NppiSize{w, h}, cnt.get(), 32, 200, buf.get());
  Case c("s2/CIR/8u_C1", "nppiCountInRange_8u_C1R");
  c.input(img.data(), img.size());
  c.status(st);
  c.device_output(cnt.get(), 1, ValueKind::I32);
  c.finish();
}
#endif

TEST_F(NppiQualityMetricsDiff, CIR_GBHS) {
  struct R { int w, h; };
  const R rois[] = {{1, 1}, {2, 3}, {17, 13}, {32, 32}, {256, 64}, {640, 480}, {8, 127}, {8, 128}, {8, 129},
                    {8, 300}, {4096, 1}, {1, 4096}};
  for (auto r : rois) {
    const std::string base = std::to_string(r.w) + "x" + std::to_string(r.h);
    capture_size("s2/CIR/gbhs/8u_C1/" + base, "nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx",
                 &nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/CIR/gbhs/8u_C3/" + base, "nppiCountInRangeGetBufferHostSize_8u_C3R_Ctx",
                 &nppiCountInRangeGetBufferHostSize_8u_C3R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/CIR/gbhs/8u_AC4/" + base, "nppiCountInRangeGetBufferHostSize_8u_AC4R_Ctx",
                 &nppiCountInRangeGetBufferHostSize_8u_AC4R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/CIR/gbhs/32f_C1/" + base, "nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx",
                 &nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/CIR/gbhs/32f_C3/" + base, "nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx",
                 &nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/CIR/gbhs/32f_AC4/" + base, "nppiCountInRangeGetBufferHostSize_32f_AC4R_Ctx",
                 &nppiCountInRangeGetBufferHostSize_32f_AC4R_Ctx, NppiSize{r.w, r.h});
  }
  capture_size("s2/CIR/gbhs/zero", "nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx",
               &nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx, NppiSize{0, 8});
  capture_size("s2/CIR/gbhs/neg", "nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx",
               &nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx, NppiSize{-1, 8});
}

// ============================ MSE / PSNR ===================================
TEST_F(NppiQualityMetricsDiff, MSE_8u_C1) {
  mse_case<Npp8u>("nppiMSE_8u_C1R_Ctx", "s2/MSE/8u_C1", 32, 16, 1, 0, 0x301, false, false, &nppiMSE_8u_C1R_Ctx,
                  &nppiMSEGetBufferHostSize_8u_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, MSE_8u_C1_identical) {
  mse_case<Npp8u>("nppiMSE_8u_C1R_Ctx", "s2/MSE/8u_C1_identical", 16, 8, 1, 0, 0x302, true, false,
                  &nppiMSE_8u_C1R_Ctx, &nppiMSEGetBufferHostSize_8u_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, MSE_8u_C3_pad) {
  mse_case<Npp8u>("nppiMSE_8u_C3R_Ctx", "s2/MSE/8u_C3_pad", 21, 13, 3, 7, 0x303, false, false, &nppiMSE_8u_C3R_Ctx,
                  &nppiMSEGetBufferHostSize_8u_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, PSNR_8u_C1) {
  mse_case<Npp8u>("nppiPSNR_8u_C1R_Ctx", "s2/PSNR/8u_C1", 32, 16, 1, 0, 0x304, false, true, &nppiPSNR_8u_C1R_Ctx,
                  &nppiPSNRGetBufferHostSize_8u_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, PSNR_8u_C1_identical) {
  mse_case<Npp8u>("nppiPSNR_8u_C1R_Ctx", "s2/PSNR/8u_C1_identical", 16, 8, 1, 0, 0x305, true, true,
                  &nppiPSNR_8u_C1R_Ctx, &nppiPSNRGetBufferHostSize_8u_C1R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, PSNR_8u_C3) {
  mse_case<Npp8u>("nppiPSNR_8u_C3R_Ctx", "s2/PSNR/8u_C3", 17, 9, 3, 0, 0x306, false, true, &nppiPSNR_8u_C3R_Ctx,
                  &nppiPSNRGetBufferHostSize_8u_C3R_Ctx);
}
TEST_F(NppiQualityMetricsDiff, MSE_Errors) {
  NppStreamContext ctx = current_context();
  DeviceMemory<Npp8u> a(256), b(256);
  DeviceMemory<Npp32f> out(3);
  DeviceMemory<Npp8u> buf(1 << 16);
  qi_error_case("s2/MSE/err/null", "nppiMSE_8u_C1R_Ctx",
                nppiMSE_8u_C1R_Ctx(nullptr, 16, b.get(), 16, NppiSize{16, 16}, out.get(), buf.get(), ctx));
  qi_error_case("s2/MSE/err/zero", "nppiMSE_8u_C1R_Ctx",
                nppiMSE_8u_C1R_Ctx(a.get(), 16, b.get(), 16, NppiSize{0, 16}, out.get(), buf.get(), ctx));
  qi_error_case("s2/MSE/err/neg", "nppiMSE_8u_C1R_Ctx",
                nppiMSE_8u_C1R_Ctx(a.get(), 16, b.get(), 16, NppiSize{-1, 16}, out.get(), buf.get(), ctx));
  qi_error_case("s2/MSE/err/step", "nppiMSE_8u_C1R_Ctx",
                nppiMSE_8u_C1R_Ctx(a.get(), 4, b.get(), 16, NppiSize{16, 16}, out.get(), buf.get(), ctx));
  qi_error_case("s2/MSE/err/nullbuf", "nppiMSE_8u_C1R_Ctx",
                nppiMSE_8u_C1R_Ctx(a.get(), 16, b.get(), 16, NppiSize{16, 16}, out.get(), nullptr, ctx));
  qi_error_case("s2/PSNR/err/nullbuf", "nppiPSNR_8u_C1R_Ctx",
                nppiPSNR_8u_C1R_Ctx(a.get(), 16, b.get(), 16, NppiSize{16, 16}, out.get(), nullptr, ctx));
}
TEST_F(NppiQualityMetricsDiff, MSE_GBHS) {
  struct R { int w, h; };
  const R rois[] = {{1, 1}, {2, 3}, {17, 13}, {32, 32}, {256, 64}, {640, 480}, {8, 127}, {8, 128}, {8, 129},
                    {8, 8192}, {4096, 1}, {1, 4096}};
  for (auto r : rois) {
    const std::string base = std::to_string(r.w) + "x" + std::to_string(r.h);
    capture_size("s2/MSE/gbhs/8u_C1/" + base, "nppiMSEGetBufferHostSize_8u_C1R_Ctx",
                 &nppiMSEGetBufferHostSize_8u_C1R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/MSE/gbhs/8u_C3/" + base, "nppiMSEGetBufferHostSize_8u_C3R_Ctx",
                 &nppiMSEGetBufferHostSize_8u_C3R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/PSNR/gbhs/8u_C1/" + base, "nppiPSNRGetBufferHostSize_8u_C1R_Ctx",
                 &nppiPSNRGetBufferHostSize_8u_C1R_Ctx, NppiSize{r.w, r.h});
  }
}

// ============================ batch metrics ================================
TEST_F(NppiQualityMetricsDiff, MSEBatch_C1) {
  batch_case("nppiMSEBatch_8u_C1R_Ctx", "s2/MSEBatch/8u_C1", {{16, 8}, {16, 8}, {16, 8}}, 1, NppiSize{16, 8},
             false, 0x401,
             [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                NppiBufferDescriptor *bufs, bool adv) {
               (void)adv;
               return nppiMSEBatch_8u_C1R_Ctx(a, b, n, roi, out, bufs, current_context());
             });
}
TEST_F(NppiQualityMetricsDiff, MSEBatch_C1_Advanced) {
  batch_case("nppiMSEBatch_8u_C1R_Advanced_Ctx", "s2/MSEBatch/8u_C1_adv", {{16, 8}, {24, 12}, {32, 16}}, 1,
             NppiSize{32, 16}, true, 0x402,
             [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                NppiBufferDescriptor *bufs, bool adv) {
               (void)adv;
               return nppiMSEBatch_8u_C1R_Advanced_Ctx(a, b, n, roi, out, bufs, current_context());
             });
}
TEST_F(NppiQualityMetricsDiff, MSEBatch_C3) {
  batch_case("nppiMSEBatch_8u_C3R_Ctx", "s2/MSEBatch/8u_C3", {{16, 8}, {16, 8}}, 3, NppiSize{16, 8}, false, 0x403,
             [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                NppiBufferDescriptor *bufs, bool adv) {
               (void)adv;
               return nppiMSEBatch_8u_C3R_Ctx(a, b, n, roi, out, bufs, current_context());
             });
}
TEST_F(NppiQualityMetricsDiff, PSNRBatch_C1) {
  batch_case("nppiPSNRBatch_8u_C1R_Ctx", "s2/PSNRBatch/8u_C1", {{16, 8}, {16, 8}, {16, 8}}, 1, NppiSize{16, 8},
             false, 0x404,
             [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                NppiBufferDescriptor *bufs, bool adv) {
               (void)adv;
               return nppiPSNRBatch_8u_C1R_Ctx(a, b, n, roi, out, bufs, current_context());
             });
}
TEST_F(NppiQualityMetricsDiff, PSNRBatch_C1_Advanced) {
  batch_case("nppiPSNRBatch_8u_C1R_Advanced_Ctx", "s2/PSNRBatch/8u_C1_adv", {{16, 8}, {24, 12}}, 1, NppiSize{24, 12},
             true, 0x405,
             [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                NppiBufferDescriptor *bufs, bool adv) {
               (void)adv;
               return nppiPSNRBatch_8u_C1R_Advanced_Ctx(a, b, n, roi, out, bufs, current_context());
             });
}
TEST_F(NppiQualityMetricsDiff, Batch_Errors) {
  NppStreamContext ctx = current_context();
  DeviceMemory<NppiImageDescriptor> d(3);
  DeviceMemory<Npp32f> out(3);
  DeviceMemory<NppiBufferDescriptor> bufs(3);
  Case c1("s2/MSEBatch/err/null1", "nppiMSEBatch_8u_C1R_Ctx");
  c1.status(nppiMSEBatch_8u_C1R_Ctx(nullptr, d.get(), 3, NppiSize{16, 8}, out.get(), bufs.get(), ctx));
  c1.finish();
  Case c2("s2/MSEBatch/err/n0", "nppiMSEBatch_8u_C1R_Ctx");
  c2.status(nppiMSEBatch_8u_C1R_Ctx(d.get(), d.get(), 0, NppiSize{16, 8}, out.get(), bufs.get(), ctx));
  c2.finish();
  Case c3("s2/MSEBatch/err/negroi", "nppiMSEBatch_8u_C1R_Ctx");
  c3.status(nppiMSEBatch_8u_C1R_Ctx(d.get(), d.get(), 3, NppiSize{-1, 8}, out.get(), bufs.get(), ctx));
  c3.finish();
}
TEST_F(NppiQualityMetricsDiff, Batch_GBHS) {
  struct R { int w, h; };
  const R rois[] = {{1, 1}, {2, 3}, {17, 13}, {32, 32}, {128, 128}, {129, 8}, {8, 129}, {256, 256}, {1024, 768},
                    {4096, 1}, {1, 4096}};
  for (auto r : rois) {
    const std::string base = std::to_string(r.w) + "x" + std::to_string(r.h);
    capture_size("s2/MSEBatch/gbhs/8u_C1/" + base, "nppiMSEBatchGetBufferHostSize_8u_C1R_Ctx",
                 &nppiMSEBatchGetBufferHostSize_8u_C1R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/MSEBatch/gbhs/8u_C3/" + base, "nppiMSEBatchGetBufferHostSize_8u_C3R_Ctx",
                 &nppiMSEBatchGetBufferHostSize_8u_C3R_Ctx, NppiSize{r.w, r.h});
    capture_size("s2/PSNRBatch/gbhs/8u_C1/" + base, "nppiPSNRBatchGetBufferHostSize_8u_C1R_Ctx",
                 &nppiPSNRBatchGetBufferHostSize_8u_C1R_Ctx, NppiSize{r.w, r.h});
  }
}
