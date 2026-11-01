// Differential tests for nppiRectStdDev (32f, 32s_C1RSfs), the
// nppiEvenLevelsHost_32s helper and the SSIM / WMSSSIM families.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiRectSsimDiff.*'
//
// Contracts pinned by the 5090 probes:
//   * RectStdDev: integral-image box statistics; dst(x,y) covers the window
//     at (x + rect.x, y + rect.y) of size rect; 32s_C1RSfs truncates the square
//     root and applies an arithmetic shift (nScaleFactor < 0 shifts left).
//   * SSIM: 11x11 Gaussian sigma=1.5 clamp-border maps, C1 = 6.5025,
//     C2 = 58.5225; output = mean over all pixels of l*cs, one float per
//     channel.
//   * WMSSSIM: 5 levels, per-level mean of cs^w (levels 0..3) and (l*cs)^w
//     (level 4), product of the five means; ROI must be at least 16x16.
//   * The SSIM-family GetBufferHostSize helpers are only checked for status
//     and sufficiency (mcMPP reports a conservative working set).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <cmath>
#include <string>
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

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

// ---------------------------------------------------------------------------
// RectStdDev
// ---------------------------------------------------------------------------
// Builds integral and squared-integral images (zero first row/column) and runs
// one RectStdDev case through `call`.
template <typename S, typename Q, typename D, typename Fn>
void rect_case(const char *fname, const std::string &id, int img_w, int img_h, int ow, int oh, int pad, NppiRect rect,
               Fn call) {
  const int iw = img_w + 1, ih = img_h + 1;
  std::vector<double> img((size_t)img_w * img_h);
  for (int y = 0; y < img_h; ++y)
    for (int x = 0; x < img_w; ++x) img[(size_t)y * img_w + x] = (double)((x * 5 + y * 3) % 7);
  const int sse = (int)sizeof(S), qse = (int)sizeof(Q), dse = (int)sizeof(D);
  const int srcStep = ((iw * sse + pad + sse - 1) / sse) * sse;
  const int sqrStep = ((iw * qse + pad + qse - 1) / qse) * qse;
  const int stepElems = srcStep / sse;
  std::vector<S> I((size_t)stepElems * ih);
  std::vector<Q> Sq((size_t)(sqrStep / qse) * ih);
  std::vector<double> intSum((size_t)iw * ih, 0.0), sqrSum((size_t)iw * ih, 0.0);
  for (int y = 1; y < ih; ++y)
    for (int x = 1; x < iw; ++x) {
      const double v = img[(size_t)(y - 1) * img_w + (x - 1)], q = v * v;
      intSum[(size_t)y * iw + x] = v + intSum[(size_t)(y - 1) * iw + x] + intSum[(size_t)y * iw + x - 1] -
                                   intSum[(size_t)(y - 1) * iw + x - 1];
      sqrSum[(size_t)y * iw + x] = q + sqrSum[(size_t)(y - 1) * iw + x] + sqrSum[(size_t)y * iw + x - 1] -
                                   sqrSum[(size_t)(y - 1) * iw + x - 1];
    }
  for (int y = 0; y < ih; ++y)
    for (int x = 0; x < iw; ++x) {
      I[(size_t)y * stepElems + x] = (S)intSum[(size_t)y * iw + x];
      Sq[(size_t)y * (sqrStep / qse) + x] = (Q)sqrSum[(size_t)y * iw + x];
    }
  DeviceMemory<S> dI(I.size());
  DeviceMemory<Q> dS(Sq.size());
  DeviceMemory<D> dD((size_t)ow * oh);
  dI.copyFromHost(I);
  dS.copyFromHost(Sq);
  prefill(dD, (D)-12345);
  clear_cuda_error();
  NppStatus st = call(dI.get(), srcStep, dS.get(), sqrStep, dD.get(), ow * dse, NppiSize{ow, oh}, rect);
  Case c(id, fname);
  c.status(st);
  if (st == NPP_NO_ERROR) {
    if constexpr (std::is_same_v<D, Npp32f>) {
      c.device_output(dD.get(), (size_t)ow * oh, ValueKind::F32, 2e-4, 2e-4);
    } else {
      c.device_output(dD.get(), (size_t)ow * oh, ValueKind::I32);
    }
  }
  c.finish();
}

}  // namespace

class NppiRectSsimDiff : public NppTestBase {};

// ------------------------------ RectStdDev ---------------------------------
TEST_F(NppiRectSsimDiff, Rect_32f) {
  const NppiRect rect{1, 1, 3, 3};
  rect_case<Npp32f, Npp64f, Npp32f>("nppiRectStdDev_32f_C1R_Ctx", "s2/Rect/32f", 12, 12, 8, 8, 0, rect,
                                    [&](const Npp32f *s, int ss, const Npp64f *q, int qs, Npp32f *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32f_C1R_Ctx(s, ss, q, qs, d, ds, roi, r,
                                                                        current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_32f_pad) {
  const NppiRect rect{0, 0, 2, 2};
  rect_case<Npp32f, Npp64f, Npp32f>("nppiRectStdDev_32f_C1R_Ctx", "s2/Rect/32f_pad", 10, 8, 8, 6, 12, rect,
                                    [&](const Npp32f *s, int ss, const Npp64f *q, int qs, Npp32f *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32f_C1R_Ctx(s, ss, q, qs, d, ds, roi, r,
                                                                        current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_32f_wide) {
  const NppiRect rect{2, 3, 4, 4};
  rect_case<Npp32f, Npp64f, Npp32f>("nppiRectStdDev_32f_C1R_Ctx", "s2/Rect/32f_wide", 16, 16, 8, 8, 0, rect,
                                    [&](const Npp32f *s, int ss, const Npp64f *q, int qs, Npp32f *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32f_C1R_Ctx(s, ss, q, qs, d, ds, roi, r,
                                                                        current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_32s_Sfs0) {
  const NppiRect rect{1, 1, 3, 3};
  rect_case<Npp32s, Npp32s, Npp32s>("nppiRectStdDev_32s_C1RSfs_Ctx", "s2/Rect/32s_sfs0", 12, 12, 8, 8, 0, rect,
                                    [&](const Npp32s *s, int ss, const Npp32s *q, int qs, Npp32s *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32s_C1RSfs_Ctx(s, ss, q, qs, d, ds, roi, r, 0,
                                                                           current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_32s_Sfs1) {
  const NppiRect rect{1, 1, 3, 3};
  rect_case<Npp32s, Npp32s, Npp32s>("nppiRectStdDev_32s_C1RSfs_Ctx", "s2/Rect/32s_sfs1", 12, 12, 8, 8, 0, rect,
                                    [&](const Npp32s *s, int ss, const Npp32s *q, int qs, Npp32s *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32s_C1RSfs_Ctx(s, ss, q, qs, d, ds, roi, r, 1,
                                                                           current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_32s_Sfs2) {
  const NppiRect rect{1, 1, 3, 3};
  rect_case<Npp32s, Npp32s, Npp32s>("nppiRectStdDev_32s_C1RSfs_Ctx", "s2/Rect/32s_sfs2", 12, 12, 8, 8, 0, rect,
                                    [&](const Npp32s *s, int ss, const Npp32s *q, int qs, Npp32s *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32s_C1RSfs_Ctx(s, ss, q, qs, d, ds, roi, r, 2,
                                                                           current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_32s_pad) {
  const NppiRect rect{0, 1, 2, 3};
  rect_case<Npp32s, Npp32s, Npp32s>("nppiRectStdDev_32s_C1RSfs_Ctx", "s2/Rect/32s_pad", 9, 9, 6, 5, 8, rect,
                                    [&](const Npp32s *s, int ss, const Npp32s *q, int qs, Npp32s *d, int ds,
                                        NppiSize roi, NppiRect r) {
                                      return nppiRectStdDev_32s_C1RSfs_Ctx(s, ss, q, qs, d, ds, roi, r, 0,
                                                                           current_context());
                                    });
}
TEST_F(NppiRectSsimDiff, Rect_Errors) {
  NppStreamContext ctx = current_context();
  DeviceMemory<Npp32f> s(256 << 4);
  DeviceMemory<Npp64f> q(256 << 4);
  DeviceMemory<Npp32f> d(64);
  status_case("s2/Rect/err/null", "nppiRectStdDev_32f_C1R_Ctx",
              nppiRectStdDev_32f_C1R_Ctx(nullptr, 64, q.get(), 64, d.get(), 32, NppiSize{8, 8}, NppiRect{0, 0, 2, 2},
                                         ctx));
  status_case("s2/Rect/err/zero", "nppiRectStdDev_32f_C1R_Ctx",
              nppiRectStdDev_32f_C1R_Ctx(s.get(), 64, q.get(), 64, d.get(), 32, NppiSize{0, 8}, NppiRect{0, 0, 2, 2},
                                         ctx));
  status_case("s2/Rect/err/neg", "nppiRectStdDev_32f_C1R_Ctx",
              nppiRectStdDev_32f_C1R_Ctx(s.get(), 64, q.get(), 64, d.get(), 32, NppiSize{-1, 8}, NppiRect{0, 0, 2, 2},
                                         ctx));
  status_case("s2/Rect/err/badrect", "nppiRectStdDev_32f_C1R_Ctx",
              nppiRectStdDev_32f_C1R_Ctx(s.get(), 64, q.get(), 64, d.get(), 32, NppiSize{8, 8}, NppiRect{0, 0, 0, 2},
                                         ctx));
  status_case("s2/Rect/err/negrect", "nppiRectStdDev_32f_C1R_Ctx",
              nppiRectStdDev_32f_C1R_Ctx(s.get(), 64, q.get(), 64, d.get(), 32, NppiSize{8, 8}, NppiRect{-1, 0, 2, 2},
                                         ctx));
  status_case("s2/Rect/err/step", "nppiRectStdDev_32f_C1R_Ctx",
              nppiRectStdDev_32f_C1R_Ctx(s.get(), 32, q.get(), 64, d.get(), 32, NppiSize{8, 8}, NppiRect{1, 1, 3, 3},
                                         ctx));
}

// ---------------------------- EvenLevelsHost ------------------------------
TEST_F(NppiRectSsimDiff, EvenLevels) {
  struct C { int n; Npp32s lo, hi; };
  const C cases[] = {{3, 64, 192}, {5, 0, 256}, {2, 5, 5}, {6, 0, 10}, {257, 0, 256}, {9, 0, 8}};
  for (auto k : cases) {
    std::vector<Npp32s> lv((size_t)k.n, -777);
    NppStatus st = nppiEvenLevelsHost_32s(lv.data(), k.n, k.lo, k.hi);
    Case c("s2/EvenLevels/n" + std::to_string(k.n) + "_" + std::to_string(k.lo) + "_" + std::to_string(k.hi),
           "nppiEvenLevelsHost_32s");
    c.status(st);
    if (st == NPP_NO_ERROR) c.host_output(lv.data(), lv.size(), ValueKind::I32);
    c.finish();
  }
  {
    Npp32s lv[4] = {-1, -1, -1, -1};
    status_case("s2/EvenLevels/n1", "nppiEvenLevelsHost_32s",
                nppiEvenLevelsHost_32s(lv, 1, 0, 255));
  }
}

// ------------------------------- SSIM -------------------------------------
namespace {
template <typename Fn>
void ssim_case(const char *fname, const std::string &id, int w, int h, int ch, int pad, uint64_t seed, bool identical,
               Fn call) {
  std::vector<Npp8u> a((size_t)w * h * ch), b((size_t)w * h * ch);
  Rng rng(seed);
  for (auto &v : a) v = (Npp8u)rng.integer<unsigned>(0, 255);
  if (identical) {
    b = a;
  } else {
    Rng rng2(seed ^ 0x1234567ULL);
    for (auto &v : b) v = (Npp8u)rng2.integer<unsigned>(0, 255);
  }
  DeviceMemory<Npp8u> da(a.size()), db(b.size());
  DeviceMemory<Npp32f> out((size_t)ch);
  da.copyFromHost(a);
  db.copyFromHost(b);
  prefill(out, kSentinel);
  clear_cuda_error();
  NppStatus st = call(da.get(), w * ch, db.get(), w * ch, NppiSize{w, h}, out.get(), pad);
  Case c(id, fname);
  c.input(a.data(), a.size());
  c.input(b.data(), b.size());
  c.status(st);
  if (st == NPP_NO_ERROR) c.device_output(out.get(), (size_t)ch, ValueKind::F32, 5e-4, 5e-4);
  c.finish();
}

template <typename Fn>
void ssim_batch_case(const char *fname, const std::string &id, const std::vector<NppiSize> &sizes, int ch,
                     NppiSize roi, bool advanced, uint64_t seed, Fn call) {
  std::vector<Npp8u> pool;
  std::vector<NppiImageDescriptor> d1(sizes.size()), d2(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int w = sizes[i].width, h = sizes[i].height;
    const size_t off1 = pool.size();
    Rng rng(seed + i);
    for (size_t k = 0; k < (size_t)w * h * ch; ++k) pool.push_back((Npp8u)rng.integer<unsigned>(0, 255));
    const size_t off2 = pool.size();
    Rng rng2(seed + 100 + i);
    for (size_t k = 0; k < (size_t)w * h * ch; ++k) pool.push_back((Npp8u)rng2.integer<unsigned>(0, 255));
    d1[i].pData = (void *)(uintptr_t)off1;
    d1[i].nStep = w * ch;
    d1[i].oSize = NppiSize{w, h};
    d2[i].pData = (void *)(uintptr_t)off2;
    d2[i].nStep = w * ch;
    d2[i].oSize = NppiSize{w, h};
  }
  DeviceMemory<Npp8u> dpool(pool.size());
  dpool.copyFromHost(pool);
  for (size_t i = 0; i < sizes.size(); ++i) {
    d1[i].pData = dpool.get() + (uintptr_t)d1[i].pData;
    d2[i].pData = dpool.get() + (uintptr_t)d2[i].pData;
  }
  DeviceMemory<NppiImageDescriptor> dd1(d1.size()), dd2(d2.size());
  dd1.copyFromHost(d1);
  dd2.copyFromHost(d2);
  const size_t scratchPerImage = (size_t)64 << 20;
  DeviceMemory<Npp8u> scratch(scratchPerImage * sizes.size());
  std::vector<NppiBufferDescriptor> bufs(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    bufs[i].pData = scratch.get() + scratchPerImage * i;
    bufs[i].nBufferSize = (int)scratchPerImage;
  }
  DeviceMemory<NppiBufferDescriptor> dbufs(bufs.size());
  dbufs.copyFromHost(bufs);
  DeviceMemory<Npp32f> out((size_t)sizes.size() * ch);
  prefill(out, kSentinel);
  clear_cuda_error();
  NppStatus st = call(dd1.get(), dd2.get(), (int)sizes.size(), roi, out.get(), dbufs.get(), advanced);
  Case c(id, fname);
  c.input(pool.data(), pool.size());
  c.status(st);
  if (st == NPP_NO_ERROR) c.device_output(out.get(), (size_t)sizes.size() * ch, ValueKind::F32, 5e-4, 5e-4);
  c.finish();
}
}  // namespace

TEST_F(NppiRectSsimDiff, SSIM_C1) {
  ssim_case("nppiSSIM_8u_C1R_Ctx", "s2/SSIM/8u_C1", 32, 24, 1, 0, 0x601, false,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiSSIM_8u_C1R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, SSIM_C1_pad) {
  ssim_case("nppiSSIM_8u_C1R_Ctx", "s2/SSIM/8u_C1_pad", 31, 17, 1, 6, 0x602, false,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiSSIM_8u_C1R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, SSIM_C1_identical) {
  ssim_case("nppiSSIM_8u_C1R_Ctx", "s2/SSIM/8u_C1_identical", 64, 64, 1, 0, 0x603, true,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiSSIM_8u_C1R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, SSIM_C3) {
  ssim_case("nppiSSIM_8u_C3R_Ctx", "s2/SSIM/8u_C3", 48, 40, 3, 0, 0x604, false,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiSSIM_8u_C3R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, SSIM_Errors) {
  NppStreamContext ctx = current_context();
  DeviceMemory<Npp8u> a(256 * 3), b(256 * 3);
  DeviceMemory<Npp32f> out(3);
  DeviceMemory<Npp8u> buf((size_t)1 << 20);
  status_case("s2/SSIM/err/null", "nppiSSIM_8u_C1R_Ctx",
              nppiSSIM_8u_C1R_Ctx(nullptr, 16, b.get(), 16, NppiSize{16, 16}, out.get(), buf.get(), ctx));
  status_case("s2/SSIM/err/neg", "nppiSSIM_8u_C1R_Ctx",
              nppiSSIM_8u_C1R_Ctx(a.get(), 16, b.get(), 16, NppiSize{-1, 16}, out.get(), buf.get(), ctx));
  status_case("s2/SSIM/err/step", "nppiSSIM_8u_C1R_Ctx",
              nppiSSIM_8u_C1R_Ctx(a.get(), 4, b.get(), 16, NppiSize{16, 16}, out.get(), buf.get(), ctx));
}

TEST_F(NppiRectSsimDiff, WMSSSIM_C1) {
  ssim_case("nppiWMSSSIM_8u_C1R_Ctx", "s2/WMSSSIM/8u_C1", 32, 32, 1, 0, 0x611, false,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiWMSSSIM_8u_C1R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, WMSSSIM_C1_16) {
  ssim_case("nppiWMSSSIM_8u_C1R_Ctx", "s2/WMSSSIM/8u_C1_16", 16, 16, 1, 0, 0x612, false,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiWMSSSIM_8u_C1R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, WMSSSIM_C1_identical) {
  ssim_case("nppiWMSSSIM_8u_C1R_Ctx", "s2/WMSSSIM/8u_C1_identical", 64, 64, 1, 0, 0x613, true,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiWMSSSIM_8u_C1R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, WMSSSIM_C3) {
  ssim_case("nppiWMSSSIM_8u_C3R_Ctx", "s2/WMSSSIM/8u_C3", 24, 24, 3, 0, 0x614, false,
            [](const Npp8u *a, int sa, const Npp8u *b, int sb, NppiSize roi, Npp32f *out, int) {
              DeviceMemory<Npp8u> buf((size_t)1 << 20);
              return nppiWMSSSIM_8u_C3R_Ctx(a, sa, b, sb, roi, out, buf.get(), current_context());
            });
}
TEST_F(NppiRectSsimDiff, WMSSSIM_Small) {
  NppStreamContext ctx = current_context();
  DeviceMemory<Npp8u> a(15 * 15), b(15 * 15);
  DeviceMemory<Npp32f> out(1);
  DeviceMemory<Npp8u> buf((size_t)1 << 20);
  status_case("s2/WMSSSIM/err/small", "nppiWMSSSIM_8u_C1R_Ctx",
              nppiWMSSSIM_8u_C1R_Ctx(a.get(), 15, b.get(), 15, NppiSize{15, 15}, out.get(), buf.get(), ctx));
}

TEST_F(NppiRectSsimDiff, SSIMBatch_C1) {
  ssim_batch_case("nppiSSIMBatch_8u_C1R_Ctx", "s2/SSIMBatch/8u_C1", {{16, 16}, {16, 16}}, 1, NppiSize{16, 16}, false,
                  0x621,
                  [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                     NppiBufferDescriptor *bufs, bool) {
                    return nppiSSIMBatch_8u_C1R_Ctx(a, b, n, roi, out, bufs, current_context());
                  });
}
TEST_F(NppiRectSsimDiff, WMSSSIMBatch_C1_Advanced) {
  ssim_batch_case("nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx", "s2/WMSSSIMBatch/8u_C1_adv", {{16, 16}, {32, 32}}, 1,
                  NppiSize{32, 32}, true, 0x631,
                  [](const NppiImageDescriptor *a, const NppiImageDescriptor *b, int n, NppiSize roi, Npp32f *out,
                     NppiBufferDescriptor *bufs, bool) {
                    return nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx(a, b, n, roi, out, bufs, current_context());
                  });
}

// ------------------------- buffer-size sufficiency -------------------------
TEST_F(NppiRectSsimDiff, GBHS_Status) {
  struct R { int w, h; };
  const R rois[] = {{16, 16}, {32, 32}, {64, 64}, {256, 128}, {1, 1}, {15, 15}, {0, 8}, {-1, 8}};
  using SizeFn = NppStatus (*)(NppiSize, SIZE_TYPE *, NppStreamContext);
  struct Ent { const char *fname; SizeFn fn; };
  const Ent ents[] = {
      {"nppiSSIMGetBufferHostSize_8u_C1R_Ctx", &nppiSSIMGetBufferHostSize_8u_C1R_Ctx},
      {"nppiSSIMGetBufferHostSize_8u_C3R_Ctx", &nppiSSIMGetBufferHostSize_8u_C3R_Ctx},
      {"nppiSSIMBatchGetBufferHostSize_8u_C1R_Ctx", &nppiSSIMBatchGetBufferHostSize_8u_C1R_Ctx},
      {"nppiSSIMBatchGetBufferHostSize_8u_C3R_Ctx", &nppiSSIMBatchGetBufferHostSize_8u_C3R_Ctx},
      {"nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx", &nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx},
      {"nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx", &nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx},
      {"nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx", &nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx},
      {"nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx", &nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx},
      {"nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx", &nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx},
  };
  for (const auto &e : ents) {
    for (auto r : rois) {
      SIZE_TYPE bytes = 0;
      NppStatus st = e.fn(NppiSize{r.w, r.h}, &bytes, current_context());
      Case c("s2/SSIM/gbhs/" + std::string(e.fname) + "/" + std::to_string(r.w) + "x" + std::to_string(r.h),
             e.fname);
      c.status(st);
      c.finish();
    }
  }
}

TEST_F(NppiRectSsimDiff, GBHS_Sufficiency) {
  // The reported size must be enough for the corresponding call on both the
  // reference and the mcMPP implementation.
  SIZE_TYPE bytes = 0;
  NppStatus st = nppiSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize{32, 32}, &bytes, current_context());
  ASSERT_EQ(st, NPP_NO_ERROR);
  DeviceMemory<Npp8u> a(32 * 32), b(32 * 32);
  DeviceMemory<Npp32f> out(1);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  {
    std::vector<Npp8u> ha(32 * 32), hb(32 * 32);
    for (size_t i = 0; i < ha.size(); ++i) {
      ha[i] = (Npp8u)((i * 7 + 3) & 0xff);
      hb[i] = (Npp8u)((i * 11 + 5) & 0xff);
    }
    a.copyFromHost(ha);
    b.copyFromHost(hb);
  }
  NppStatus call = nppiSSIM_8u_C1R_Ctx(a.get(), 32, b.get(), 32, NppiSize{32, 32}, out.get(), buf.get(),
                                       current_context());
  Case c("s2/SSIM/gbhs/sufficiency", "nppiSSIM_8u_C1R_Ctx");
  c.status(call);
  if (call == NPP_NO_ERROR) c.device_output(out.get(), 1, ValueKind::F32, 5e-4, 5e-4);
  c.finish();
}
