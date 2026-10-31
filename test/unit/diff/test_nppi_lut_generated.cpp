// Differential tests for package c1: NPPI point look-up-table families
// (nppiLUT, nppiLUT_Linear, nppiLUT_Cubic).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiLut*Diff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiLut*Diff.*'
//
// The 8-bit "response" cases feed the full 0..255 ramp through a small
// mapping so the recorded golden is the complete transfer function of the
// family; the 16-bit / float cases probe range ends, midpoints and
// out-of-range inputs.  Contract cases pin the status codes for ROI / step /
// level-count violations.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &dev_buf(const std::vector<T> &h) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> arena;
  arena.push_back(std::make_unique<DeviceMemory<T>>(h.size()));
  DeviceMemory<T> &d = *arena.back();
  if (!h.empty()) d.copyFromHost(h);
  return d;
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename T> std::vector<T> ramp(int count, double lo, double hi) {
  std::vector<T> v(count);
  for (int i = 0; i < count; ++i) v[i] = static_cast<T>(lo + (hi - lo) * i / (count > 1 ? count - 1 : 1));
  return v;
}

std::vector<Npp8u> ramp8u(int count) {
  std::vector<Npp8u> v(count);
  for (int i = 0; i < count; ++i) v[i] = static_cast<Npp8u>(i * 255 / (count > 1 ? count - 1 : 1));
  return v;
}

std::vector<Npp16u> ramp16u(int count) {
  std::vector<Npp16u> v(count);
  for (int i = 0; i < count; ++i) v[i] = static_cast<Npp16u>(i * 65535 / (count > 1 ? count - 1 : 1));
  return v;
}

std::vector<Npp16s> ramp16s(int count) {
  std::vector<Npp16s> v(count);
  for (int i = 0; i < count; ++i) v[i] = static_cast<Npp16s>(-32768 + i * 65535 / (count > 1 ? count - 1 : 1));
  return v;
}

template <typename T> std::vector<T> concat(const std::vector<T> &a, const std::vector<T> &b) {
  std::vector<T> out = a;
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

// ---- C1 point-LUT runner (src/dst separate buffers) --------------------
// f(devSrc, sStep, devDst, dStep, roi, devValues, devLevels, nLevels, ctx)
template <typename Src, typename Val, typename F>
void lut_c1(const char *fname, const std::string &key, int width, int height, int pad,
            const std::vector<Src> &hs, const std::vector<Val> &values, const std::vector<Val> &levels,
            ValueKind kind, F &&f) {
  const int es = (int)sizeof(Src);
  const int step = align_step(width * es + pad, es);
  const size_t n = (size_t)height * step / es;
  std::vector<Src> hs_full(hs);
  hs_full.resize(n);
  std::vector<Src> hd = make_uniform<Src>(n, 4242);
  DeviceMemory<Src> &s = dev_buf(hs_full);
  DeviceMemory<Src> &d = dev_buf(hd);
  DeviceMemory<Val> &v = dev_buf(values);
  DeviceMemory<Val> &l = dev_buf(levels);
  clear_cuda_error();
  NppStatus st = f(s.get(), step, d.get(), step, NppiSize{width, height}, v.get(), l.get(),
                   (int)levels.size(), current_ctx());
  Case c("nppiLUT/" + key, fname);
  c.input(hs_full.data(), hs_full.size() * sizeof(Src));
  c.status(st);
  c.device_output<Src>(d.get(), n, kind);
  c.finish();
}

// ---- C1 in-place point-LUT runner --------------------------------------
template <typename Src, typename Val, typename F>
void lut_c1i(const char *fname, const std::string &key, int width, int height, int pad,
             const std::vector<Src> &hs, const std::vector<Val> &values, const std::vector<Val> &levels,
             ValueKind kind, F &&f) {
  const int es = (int)sizeof(Src);
  const int step = align_step(width * es + pad, es);
  const size_t n = (size_t)height * step / es;
  std::vector<Src> hd(hs);
  hd.resize(n);
  DeviceMemory<Src> &d = dev_buf(hd);
  DeviceMemory<Val> &v = dev_buf(values);
  DeviceMemory<Val> &l = dev_buf(levels);
  clear_cuda_error();
  NppStatus st = f(d.get(), step, NppiSize{width, height}, v.get(), l.get(), (int)levels.size(), current_ctx());
  Case c("nppiLUT/" + key, fname);
  c.input(hd.data(), hd.size() * sizeof(Src));
  c.status(st);
  c.device_output<Src>(d.get(), n, kind);
  c.finish();
}

// ---- multi-channel point-LUT runner (C3/C4/AC4) ------------------------
template <typename Src, typename Val, int NTab, typename F>
void lut_multi(const char *fname, const std::string &key, int width, int height, int pad, int ch,
               const std::vector<Src> &hs, const std::vector<std::vector<Val>> &values,
               const std::vector<std::vector<Val>> &levels, ValueKind kind, F &&f) {
  const int es = (int)sizeof(Src);
  const int px = ch * es;
  const int step = align_step(width * px + pad, px);
  const size_t n = (size_t)height * step / es;
  std::vector<Src> hs_full(hs);
  hs_full.resize(n);
  std::vector<Src> hd = make_uniform<Src>(n, 5151);
  DeviceMemory<Src> &s = dev_buf(hs_full);
  DeviceMemory<Src> &d = dev_buf(hd);
  DeviceMemory<Val> *vdev[NTab];
  DeviceMemory<Val> *ldev[NTab];
  const Val *pv[NTab];
  const Val *pl[NTab];
  int nl[NTab];
  for (int i = 0; i < NTab; ++i) {
    vdev[i] = &dev_buf(values[i]);
    ldev[i] = &dev_buf(levels[i]);
    pv[i] = vdev[i]->get();
    pl[i] = ldev[i]->get();
    nl[i] = (int)levels[i].size();
  }
  clear_cuda_error();
  NppStatus st = f(s.get(), step, d.get(), step, NppiSize{width, height}, pv, pl, nl, current_ctx());
  Case c("nppiLUT/" + key, fname);
  c.input(hs_full.data(), hs_full.size() * sizeof(Src));
  c.status(st);
  c.device_output<Src>(d.get(), n, kind);
  c.finish();
}

// ---- multi-channel in-place point-LUT runner ---------------------------
template <typename Src, typename Val, int NTab, typename F>
void lut_multi_i(const char *fname, const std::string &key, int width, int height, int pad, int ch,
                 const std::vector<Src> &hs, const std::vector<std::vector<Val>> &values,
                 const std::vector<std::vector<Val>> &levels, ValueKind kind, F &&f) {
  const int es = (int)sizeof(Src);
  const int px = ch * es;
  const int step = align_step(width * px + pad, px);
  const size_t n = (size_t)height * step / es;
  std::vector<Src> hd(hs);
  hd.resize(n);
  DeviceMemory<Src> &d = dev_buf(hd);
  DeviceMemory<Val> *vdev[NTab];
  DeviceMemory<Val> *ldev[NTab];
  const Val *pv[NTab];
  const Val *pl[NTab];
  int nl[NTab];
  for (int i = 0; i < NTab; ++i) {
    vdev[i] = &dev_buf(values[i]);
    ldev[i] = &dev_buf(levels[i]);
    pv[i] = vdev[i]->get();
    pl[i] = ldev[i]->get();
    nl[i] = (int)levels[i].size();
  }
  clear_cuda_error();
  NppStatus st = f(d.get(), step, NppiSize{width, height}, pv, pl, nl, current_ctx());
  Case c("nppiLUT/" + key, fname);
  c.input(hd.data(), hd.size() * sizeof(Src));
  c.status(st);
  c.device_output<Src>(d.get(), n, kind);
  c.finish();
}

}  // namespace

class NppiLutDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== nppiLUT: full 8-bit transfer functions ============================
TEST_F(NppiLutDiff, Lut_8u_C1_Response) {
  // below first level / exact levels / between / above last level
  lut_c1<Npp8u, Npp32s>("nppiLUT_8u_C1R", "8u/C1/response", 256, 2, 7, ramp8u(512),
                        {10, 74, 138}, {64, 128, 192}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // unsorted level list (NPP is probed for its search behavior)
  lut_c1<Npp8u, Npp32s>("nppiLUT_8u_C1R", "8u/C1/unsorted", 256, 2, 3, ramp8u(512), {7, 8, 9},
                        {100, 50, 200}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // out-of-range output values (saturation probe)
  lut_c1<Npp8u, Npp32s>("nppiLUT_8u_C1R", "8u/C1/satur", 256, 2, 5, ramp8u(512), {-30, 300},
                        {0, 255}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
}

TEST_F(NppiLutDiff, Lut_8u_C1_Random) {
  lut_c1<Npp8u, Npp32s>("nppiLUT_8u_C1R", "8u/C1/random", 17, 13, 5, make_uniform<Npp8u>(17 * 13, 100),
                        {1, 2, 3, 4}, {10, 100, 200, 240}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
}

TEST_F(NppiLutDiff, Lut_8u_MultiChannel) {
  auto rnd = make_uniform<Npp8u>(17 * 13 * 4, 200);
  lut_multi<Npp8u, Npp32s, 3>("nppiLUT_8u_C3R", "8u/C3/random", 17, 13, 5, 3, rnd,
                              {{1, 2, 3}, {4, 5}, {6, 7, 8, 9}},
                              {{10, 100}, {20, 200}, {30, 150, 250}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_8u_C3R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
  lut_multi<Npp8u, Npp32s, 4>("nppiLUT_8u_C4R", "8u/C4/random", 17, 13, 7, 4, rnd,
                              {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                              {{10, 200}, {20, 210}, {30, 220}, {40, 230}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_8u_C4R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
  lut_multi<Npp8u, Npp32s, 3>("nppiLUT_8u_AC4R", "8u/AC4/random", 17, 13, 3, 4, rnd,
                              {{1, 2}, {3, 4}, {5, 6}},
                              {{10, 200}, {20, 210}, {30, 220}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_8u_AC4R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
}

TEST_F(NppiLutDiff, Lut_8u_InPlace) {
  lut_c1i<Npp8u, Npp32s>("nppiLUT_8u_C1IR", "8u/C1IR/random", 17, 13, 5, make_uniform<Npp8u>(17 * 13, 210),
                         {1, 2, 3, 4}, {10, 100, 200, 240}, ValueKind::U8,
                         [](Npp8u *sd, int ss, NppiSize o, const Npp32s *v, const Npp32s *l, int n,
                            NppStreamContext c) {
                           return nppiLUT_8u_C1IR_Ctx(sd, ss, o, v, l, n, c);
                         });
  lut_multi_i<Npp8u, Npp32s, 3>("nppiLUT_8u_C3IR", "8u/C3IR/random", 17, 13, 5, 3,
                                make_uniform<Npp8u>(17 * 13 * 3, 211),
                                {{1, 2, 3}, {4, 5}, {6, 7, 8, 9}},
                                {{10, 100}, {20, 200}, {30, 150, 250}}, ValueKind::U8,
                                [](Npp8u *sd, int ss, NppiSize o, const Npp32s *v[], const Npp32s *l[], int n[],
                                   NppStreamContext c) {
                                  return nppiLUT_8u_C3IR_Ctx(sd, ss, o, v, l, n, c);
                                });
  lut_multi_i<Npp8u, Npp32s, 3>("nppiLUT_8u_AC4IR", "8u/AC4IR/random", 17, 13, 3, 4,
                                make_uniform<Npp8u>(17 * 13 * 4, 212),
                                {{1, 2}, {3, 4}, {5, 6}},
                                {{10, 200}, {20, 210}, {30, 220}}, ValueKind::U8,
                                [](Npp8u *sd, int ss, NppiSize o, const Npp32s *v[], const Npp32s *l[], int n[],
                                   NppStreamContext c) {
                                  return nppiLUT_8u_AC4IR_Ctx(sd, ss, o, v, l, n, c);
                                });
  lut_multi_i<Npp8u, Npp32s, 4>("nppiLUT_8u_C4IR", "8u/C4IR/random", 17, 13, 3, 4,
                                make_uniform<Npp8u>(17 * 13 * 4, 213),
                                {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                                {{10, 200}, {20, 210}, {30, 220}, {40, 230}}, ValueKind::U8,
                                [](Npp8u *sd, int ss, NppiSize o, const Npp32s *v[], const Npp32s *l[], int n[],
                                   NppStreamContext c) {
                                  return nppiLUT_8u_C4IR_Ctx(sd, ss, o, v, l, n, c);
                                });
}

// ===== nppiLUT: 16-bit / float responses ================================
TEST_F(NppiLutDiff, Lut_16u_16s_32f_C1) {
  lut_c1<Npp16u, Npp32s>("nppiLUT_16u_C1R", "16u/C1/response", 256, 2, 9, ramp16u(512), {1, 2, 3},
                         {1000, 20000, 60000}, ValueKind::U16,
                         [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_16u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  lut_c1<Npp16s, Npp32s>("nppiLUT_16s_C1R", "16s/C1/response", 256, 2, 11, ramp16s(512), {-1, 0, 1},
                         {-100, 0, 100}, ValueKind::I16,
                         [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_16s_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  const std::vector<Npp32f> fs = {-1.0f, -0.5f, 0.0f,  0.25f, 0.5f,  1.0f,  1.5f,
                                  2.0f,  2.5f,  3.0f,  4.0f,  100.0f, -100.0f, 0.5f};
  lut_c1<Npp32f, Npp32f>("nppiLUT_32f_C1R", "32f/C1/response", 14, 1, 5, fs, {10.0f, 20.0f},
                         {0.5f, 2.5f}, ValueKind::F32,
                         [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *v, const Npp32f *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_32f_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
}

TEST_F(NppiLutDiff, Lut_16u_16s_32f_Multi) {
  lut_multi<Npp16u, Npp32s, 4>("nppiLUT_16u_C4R", "16u/C4/random", 17, 13, 6, 4, make_uniform<Npp16u>(17 * 13 * 4, 300),
                               {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                               {{1000, 60000}, {2000, 50000}, {3000, 40000}, {400, 30000}}, ValueKind::U16,
                               [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v[],
                                  const Npp32s *l[], int n[], NppStreamContext c) {
                                 return nppiLUT_16u_C4R_Ctx(s, ss, d, ds, o, v, l, n, c);
                               });
  lut_multi_i<Npp16s, Npp32s, 3>("nppiLUT_16s_AC4IR", "16s/AC4IR/random", 17, 13, 6, 4,
                                 make_uniform<Npp16s>(17 * 13 * 4, 301),
                                 {{-5, -4}, {-3, -2}, {-1, 0}},
                                 {{-1000, 1000}, {-500, 500}, {-100, 100}}, ValueKind::I16,
                                 [](Npp16s *sd, int ss, NppiSize o, const Npp32s *v[], const Npp32s *l[], int n[],
                                    NppStreamContext c) {
                                   return nppiLUT_16s_AC4IR_Ctx(sd, ss, o, v, l, n, c);
                                 });
  lut_multi<Npp32f, Npp32f, 3>("nppiLUT_32f_C3R", "32f/C3/random", 17, 13, 6, 3, make_uniform<Npp32f>(17 * 13 * 3, 302),
                               {{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}},
                               {{-1.f, 1.f}, {-2.f, 2.f}, {-3.f, 3.f}}, ValueKind::F32,
                               [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *v[],
                                  const Npp32f *l[], int n[], NppStreamContext c) {
                                 return nppiLUT_32f_C3R_Ctx(s, ss, d, ds, o, v, l, n, c);
                               });
  lut_c1i<Npp16u, Npp32s>("nppiLUT_16u_C1IR", "16u/C1IR/random", 17, 13, 6, make_uniform<Npp16u>(17 * 13, 303),
                          {1, 2, 3, 4}, {100, 40000, 60000, 65000}, ValueKind::U16,
                          [](Npp16u *sd, int ss, NppiSize o, const Npp32s *v, const Npp32s *l, int n,
                             NppStreamContext c) {
                            return nppiLUT_16u_C1IR_Ctx(sd, ss, o, v, l, n, c);
                          });
}

TEST_F(NppiLutDiff, Lut_MaxLevels) {
  // 256 levels (8u maximum): identity mapping.
  std::vector<Npp32s> lv(256), vv(256);
  for (int i = 0; i < 256; ++i) {
    lv[i] = i;
    vv[i] = 255 - i;
  }
  lut_c1<Npp8u, Npp32s>("nppiLUT_8u_C1R", "8u/C1/maxlevels", 256, 2, 3, ramp8u(512), vv, lv, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // 1024 levels (16-bit maximum): identity mapping over 0..65472.
  std::vector<Npp32s> lv16(1024), vv16(1024);
  for (int i = 0; i < 1024; ++i) {
    lv16[i] = i * 64;
    vv16[i] = 65535 - i * 64;
  }
  lut_c1<Npp16u, Npp32s>("nppiLUT_16u_C1R", "16u/C1/maxlevels", 256, 2, 3, ramp16u(512), vv16, lv16, ValueKind::U16,
                         [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_16u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  // 256 levels with a duplicated tail: distinguishes the 8-bit fast path
  // (dst = values[src], levels ignored) from the binary-search path
  // (x >= levels[255] = 200 would pass through).
  std::vector<Npp32s> lvd(256), vvd(256);
  for (int i = 0; i < 256; ++i) {
    lvd[i] = i < 200 ? i : 200;
    vvd[i] = (i * 3 + 1) & 0xFF;
  }
  lut_c1<Npp8u, Npp32s>("nppiLUT_8u_C1R", "8u/C1/maxlevelsdup", 256, 2, 3, ramp8u(512), vvd, lvd, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
}

// ===== nppiLUT: contract (status codes) ==================================
TEST_F(NppiLutDiff, Lut_Contract) {
  const std::vector<Npp32s> big(2048, 5);
  auto run8 = [&](const std::string &key, const std::vector<Npp32s> &values, const std::vector<Npp32s> &levels,
                  int nLevels, NppiSize roi, int sStep, int dStep, bool nullSrc, bool nullDst) {
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 711);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 712);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    DeviceMemory<Npp32s> &v = dev_buf(values.empty() ? big : values);
    DeviceMemory<Npp32s> &l = dev_buf(levels.empty() ? big : levels);
    clear_cuda_error();
    NppStatus st = nppiLUT_8u_C1R_Ctx(nullSrc ? nullptr : s.get(), sStep, nullDst ? nullptr : d.get(), dStep, roi,
                                     v.get(), l.get(), nLevels, current_ctx());
    Case c("nppiLUT/contract/" + key, "nppiLUT_8u_C1R");
    c.status(st);
    c.finish();
  };
  const std::vector<Npp32s> okv = {1, 2, 3}, okl = {10, 100, 200};
  run8("levels1", okv, okl, 1, NppiSize{16, 4}, 16, 16, false, false);
  run8("levels0", okv, okl, 0, NppiSize{16, 4}, 16, 16, false, false);
  run8("levelsNeg", okv, okl, -3, NppiSize{16, 4}, 16, 16, false, false);
  run8("levels257", std::vector<Npp32s>(257, 3), std::vector<Npp32s>(257, 3), 257, NppiSize{16, 4}, 16, 16, false,
       false);
  run8("emptyRoi", okv, okl, 3, NppiSize{0, 4}, 16, 16, false, false);
  run8("negRoi", okv, okl, 3, NppiSize{-4, 4}, 16, 16, false, false);
  run8("nullRoi", okv, okl, 3, NppiSize{0, 0}, 16, 16, false, false);
  run8("shortSrcStep", okv, okl, 3, NppiSize{16, 1}, 8, 16, false, false);
  run8("shortDstStep", okv, okl, 3, NppiSize{16, 1}, 16, 8, false, false);
  run8("nullSrc", okv, okl, 3, NppiSize{16, 4}, 16, 16, true, false);
  run8("nullDst", okv, okl, 3, NppiSize{16, 4}, 16, 16, false, true);
  // C3 with one bad channel level count
  {
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 713);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 714);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    const Npp32s *pv[3], *pl[3];
    int nl[3] = {2, 1, 2};
    for (int i = 0; i < 3; ++i) {
      pv[i] = dev_buf(std::vector<Npp32s>({1, 2})).get();
      pl[i] = dev_buf(std::vector<Npp32s>({10, 200})).get();
    }
    clear_cuda_error();
    NppStatus st = nppiLUT_8u_C3R_Ctx(s.get(), 64, d.get(), 64, NppiSize{16, 4}, pv, pl, nl, current_ctx());
    Case c("nppiLUT/contract/c3levels1", "nppiLUT_8u_C3R");
    c.status(st);
    c.finish();
  }
}

// ===== nppiLUT_Linear ====================================================
class NppiLutLinearDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiLutLinearDiff, Linear_8u_C1_Response) {
  lut_c1<Npp8u, Npp32s>("nppiLUT_Linear_8u_C1R", "linear/8u/C1/response", 256, 2, 7, ramp8u(512),
                        {10, 74, 138, 202, 250}, {0, 64, 128, 192, 255}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_Linear_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // extrapolation / clamping outside the level range
  lut_c1<Npp8u, Npp32s>("nppiLUT_Linear_8u_C1R", "linear/8u/C1/extrap", 256, 2, 3, ramp8u(512), {10, 240},
                        {50, 200}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_Linear_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // saturating outputs
  lut_c1<Npp8u, Npp32s>("nppiLUT_Linear_8u_C1R", "linear/8u/C1/satur", 256, 2, 5, ramp8u(512), {-40, 300},
                        {0, 255}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_Linear_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
}

TEST_F(NppiLutLinearDiff, Linear_16u_16s_32f) {
  lut_c1<Npp16u, Npp32s>("nppiLUT_Linear_16u_C1R", "linear/16u/C1/response", 256, 2, 9, ramp16u(512), {5, 60000},
                         {1000, 60000}, ValueKind::U16,
                         [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_Linear_16u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  lut_c1<Npp16s, Npp32s>("nppiLUT_Linear_16s_C1R", "linear/16s/C1/response", 256, 2, 11, ramp16s(512), {-3000, 3000},
                         {-10000, 10000}, ValueKind::I16,
                         [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_Linear_16s_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  const std::vector<Npp32f> fs = {-1.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 100.0f, -100.0f};
  lut_c1<Npp32f, Npp32f>("nppiLUT_Linear_32f_C1R", "linear/32f/C1/response", 10, 1, 5, fs, {10.0f, 20.0f},
                         {0.5f, 2.5f}, ValueKind::F32,
                         [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *v, const Npp32f *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_Linear_32f_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
}

TEST_F(NppiLutLinearDiff, Linear_MultiChannel) {
  auto rnd = make_uniform<Npp8u>(17 * 13 * 4, 400);
  lut_multi<Npp8u, Npp32s, 3>("nppiLUT_Linear_8u_C3R", "linear/8u/C3/random", 17, 13, 5, 3, rnd,
                              {{1, 9, 100}, {2, 8}, {3, 7, 200}},
                              {{10, 100, 240}, {20, 220}, {30, 250}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_Linear_8u_C3R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
  lut_multi<Npp8u, Npp32s, 3>("nppiLUT_Linear_8u_AC4R", "linear/8u/AC4/random", 17, 13, 3, 4, rnd,
                              {{1, 9, 100}, {2, 8}, {3, 7, 200}},
                              {{10, 100, 240}, {20, 220}, {30, 250}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_Linear_8u_AC4R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
  lut_c1i<Npp8u, Npp32s>("nppiLUT_Linear_8u_C1IR", "linear/8u/C1IR/random", 17, 13, 5,
                         make_uniform<Npp8u>(17 * 13, 401), {1, 9, 100}, {10, 100, 240}, ValueKind::U8,
                         [](Npp8u *sd, int ss, NppiSize o, const Npp32s *v, const Npp32s *l, int n,
                            NppStreamContext c) {
                           return nppiLUT_Linear_8u_C1IR_Ctx(sd, ss, o, v, l, n, c);
                         });
  lut_multi_i<Npp16u, Npp32s, 4>("nppiLUT_Linear_16u_C4IR", "linear/16u/C4IR/random", 17, 13, 6, 4,
                                 make_uniform<Npp16u>(17 * 13 * 4, 402),
                                 {{1, 900, 2700}, {2, 800}, {3, 700}, {4, 600}},
                                 {{1000, 20000, 60000}, {2000, 30000}, {3000, 40000}, {400, 50000}}, ValueKind::U16,
                                 [](Npp16u *sd, int ss, NppiSize o, const Npp32s *v[], const Npp32s *l[], int n[],
                                    NppStreamContext c) {
                                   return nppiLUT_Linear_16u_C4IR_Ctx(sd, ss, o, v, l, n, c);
                                 });
  lut_multi_i<Npp32f, Npp32f, 3>("nppiLUT_Linear_32f_AC4IR", "linear/32f/AC4IR/random", 17, 13, 6, 4,
                                 make_uniform<Npp32f>(17 * 13 * 4, 403),
                                 {{1.f, 9.f}, {2.f, 8.f}, {3.f, 7.f}},
                                 {{-1.f, 1.f}, {-2.f, 2.f}, {-3.f, 3.f}}, ValueKind::F32,
                                 [](Npp32f *sd, int ss, NppiSize o, const Npp32f *v[], const Npp32f *l[], int n[],
                                    NppStreamContext c) {
                                   return nppiLUT_Linear_32f_AC4IR_Ctx(sd, ss, o, v, l, n, c);
                                 });
}

TEST_F(NppiLutLinearDiff, Linear_Contract) {
  const std::vector<Npp32s> okv = {1, 2, 3}, okl = {10, 100, 200};
  auto run8 = [&](const std::string &key, const std::vector<Npp32s> &values, const std::vector<Npp32s> &levels,
                  int nLevels, NppiSize roi) {
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 721);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 722);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    DeviceMemory<Npp32s> &v = dev_buf(values);
    DeviceMemory<Npp32s> &l = dev_buf(levels);
    clear_cuda_error();
    NppStatus st = nppiLUT_Linear_8u_C1R_Ctx(s.get(), 16, d.get(), 16, roi, v.get(), l.get(), nLevels, current_ctx());
    Case c("nppiLUT/linear/contract/" + key, "nppiLUT_Linear_8u_C1R");
    c.status(st);
    c.finish();
  };
  run8("levels1", okv, okl, 1, NppiSize{16, 4});
  run8("levels257", std::vector<Npp32s>(257, 3), std::vector<Npp32s>(257, 3), 257, NppiSize{16, 4});
  run8("emptyRoi", okv, okl, 3, NppiSize{0, 4});
  run8("negRoi", okv, okl, 3, NppiSize{-4, 4});
  {
    // 16-bit limit is 1024 levels
    std::vector<Npp16u> hs = make_uniform<Npp16u>(4096, 723);
    std::vector<Npp16u> hd = make_uniform<Npp16u>(4096, 724);
    DeviceMemory<Npp16u> &s = dev_buf(hs);
    DeviceMemory<Npp16u> &d = dev_buf(hd);
    std::vector<Npp32s> lv(1025), vv(1025, 7);
    for (int i = 0; i < 1025; ++i) lv[i] = i * 64;
    DeviceMemory<Npp32s> &v = dev_buf(vv);
    DeviceMemory<Npp32s> &l = dev_buf(lv);
    clear_cuda_error();
    NppStatus st = nppiLUT_Linear_16u_C1R_Ctx(s.get(), 32, d.get(), 32, NppiSize{16, 4}, v.get(), l.get(), 1025,
                                              current_ctx());
    Case c("nppiLUT/linear/contract/levels1025", "nppiLUT_Linear_16u_C1R");
    c.status(st);
    c.finish();
  }
}

// ===== nppiLUT_Cubic =====================================================
class NppiLutCubicDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiLutCubicDiff, Cubic_8u_C1_Response) {
  lut_c1<Npp8u, Npp32s>("nppiLUT_Cubic_8u_C1R", "cubic/8u/C1/response", 256, 2, 7, ramp8u(512),
                        {10, 74, 138, 202, 250}, {0, 64, 128, 192, 255}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_Cubic_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // two-point segment (cubic behavior at the range ends)
  lut_c1<Npp8u, Npp32s>("nppiLUT_Cubic_8u_C1R", "cubic/8u/C1/extrap", 256, 2, 3, ramp8u(512), {10, 240},
                        {50, 200}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_Cubic_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
  // alternating values: cubic overshoot / undershoot
  lut_c1<Npp8u, Npp32s>("nppiLUT_Cubic_8u_C1R", "cubic/8u/C1/overshoot", 256, 2, 5, ramp8u(512),
                        {0, 255, 0, 255}, {50, 100, 150, 200}, ValueKind::U8,
                        [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                           int n, NppStreamContext c) {
                          return nppiLUT_Cubic_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                        });
}

TEST_F(NppiLutCubicDiff, Cubic_16u_16s_32f) {
  lut_c1<Npp16u, Npp32s>("nppiLUT_Cubic_16u_C1R", "cubic/16u/C1/response", 256, 2, 9, ramp16u(512), {5, 60000},
                         {1000, 60000}, ValueKind::U16,
                         [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_Cubic_16u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  lut_c1<Npp16s, Npp32s>("nppiLUT_Cubic_16s_C1R", "cubic/16s/C1/response", 256, 2, 11, ramp16s(512), {-3000, 3000},
                         {-10000, 10000}, ValueKind::I16,
                         [](const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize o, const Npp32s *v, const Npp32s *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_Cubic_16s_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
  const std::vector<Npp32f> fs = {-1.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 100.0f, -100.0f};
  lut_c1<Npp32f, Npp32f>("nppiLUT_Cubic_32f_C1R", "cubic/32f/C1/response", 10, 1, 5, fs, {10.0f, 20.0f},
                         {0.5f, 2.5f}, ValueKind::F32,
                         [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize o, const Npp32f *v, const Npp32f *l,
                            int n, NppStreamContext c) {
                           return nppiLUT_Cubic_32f_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
}

TEST_F(NppiLutCubicDiff, Cubic_MultiChannel) {
  auto rnd = make_uniform<Npp8u>(17 * 13 * 4, 500);
  lut_multi<Npp8u, Npp32s, 3>("nppiLUT_Cubic_8u_C3R", "cubic/8u/C3/random", 17, 13, 5, 3, rnd,
                              {{1, 9, 100}, {2, 8}, {3, 7, 200}},
                              {{10, 100, 240}, {20, 220}, {30, 250}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_Cubic_8u_C3R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
  lut_multi<Npp8u, Npp32s, 3>("nppiLUT_Cubic_8u_AC4R", "cubic/8u/AC4/random", 17, 13, 3, 4, rnd,
                              {{1, 9, 100}, {2, 8}, {3, 7, 200}},
                              {{10, 100, 240}, {20, 220}, {30, 250}}, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v[],
                                 const Npp32s *l[], int n[], NppStreamContext c) {
                                return nppiLUT_Cubic_8u_AC4R_Ctx(s, ss, d, ds, o, v, l, n, c);
                              });
  lut_c1i<Npp8u, Npp32s>("nppiLUT_Cubic_8u_C1IR", "cubic/8u/C1IR/random", 17, 13, 5,
                         make_uniform<Npp8u>(17 * 13, 501), {1, 9, 100}, {10, 100, 240}, ValueKind::U8,
                         [](Npp8u *sd, int ss, NppiSize o, const Npp32s *v, const Npp32s *l, int n,
                            NppStreamContext c) {
                           return nppiLUT_Cubic_8u_C1IR_Ctx(sd, ss, o, v, l, n, c);
                         });
}

TEST_F(NppiLutCubicDiff, Cubic_Contract) {
  const std::vector<Npp32s> okv = {1, 2, 3}, okl = {10, 100, 200};
  std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 731);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 732);
  DeviceMemory<Npp8u> &s = dev_buf(hs);
  DeviceMemory<Npp8u> &d = dev_buf(hd);
  DeviceMemory<Npp32s> &v = dev_buf(okv);
  DeviceMemory<Npp32s> &l = dev_buf(okl);
  clear_cuda_error();
  NppStatus st = nppiLUT_Cubic_8u_C1R_Ctx(s.get(), 16, d.get(), 16, NppiSize{16, 4}, v.get(), l.get(), 1,
                                          current_ctx());
  Case c("nppiLUT/cubic/contract/levels1", "nppiLUT_Cubic_8u_C1R");
  c.status(st);
  c.finish();
  clear_cuda_error();
  st = nppiLUT_Cubic_8u_C1R_Ctx(s.get(), 16, d.get(), 16, NppiSize{-4, 4}, v.get(), l.get(), 3, current_ctx());
  Case c2("nppiLUT/cubic/contract/negRoi", "nppiLUT_Cubic_8u_C1R");
  c2.status(st);
  c2.finish();
}

// ===== full-ramp probes: complete transfer curves with no ramp gaps =====
// (the 512-pixel ramp above skips a few input values per row; these feed
// src[i] == i so every representable input is recorded exactly once)
TEST_F(NppiLutLinearDiff, Linear_FullRamp) {
  std::vector<Npp8u> full(256);
  for (int i = 0; i < 256; ++i) full[i] = static_cast<Npp8u>(i);
  auto run = [&](const std::string &key, std::vector<Npp32s> values, std::vector<Npp32s> levels) {
    lut_c1<Npp8u, Npp32s>("nppiLUT_Linear_8u_C1R", "linear/8u/C1/" + key, 256, 1, 7, full, values, levels,
                          ValueKind::U8,
                          [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v,
                             const Npp32s *l, int n, NppStreamContext c) {
                            return nppiLUT_Linear_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                          });
  };
  run("fullneg", {240, 10}, {50, 200});          // negative slope: floor vs trunc
  run("fullwrap", {-40, 300}, {0, 255});         // negative outputs -> wrap
  run("full3", {5, 250, 3}, {0, 100, 255});      // mixed slopes, odd deltas
  run("fullunsorted", {7, 8, 9}, {100, 50, 200}); // unsorted levels
  // 16-bit full curve: src[i] == i, 65536 wide
  std::vector<Npp16u> f16(65536);
  for (int i = 0; i < 65536; ++i) f16[i] = static_cast<Npp16u>(i);
  lut_c1<Npp16u, Npp32s>("nppiLUT_Linear_16u_C1R", "linear/16u/C1/full", 65536, 1, 5, f16,
                         {11, 40001, 20003, 65529}, {0, 16384, 49152, 65535}, ValueKind::U16,
                         [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v,
                            const Npp32s *l, int n, NppStreamContext c) {
                           return nppiLUT_Linear_16u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
}

TEST_F(NppiLutCubicDiff, Cubic_FullRamp) {
  std::vector<Npp8u> full(256);
  for (int i = 0; i < 256; ++i) full[i] = static_cast<Npp8u>(i);
  auto run = [&](const std::string &key, std::vector<Npp32s> values, std::vector<Npp32s> levels) {
    lut_c1<Npp8u, Npp32s>("nppiLUT_Cubic_8u_C1R", "cubic/8u/C1/" + key, 256, 1, 7, full, values, levels,
                          ValueKind::U8,
                          [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp32s *v,
                             const Npp32s *l, int n, NppStreamContext c) {
                            return nppiLUT_Cubic_8u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                          });
  };
  run("fullresp", {10, 74, 138, 202, 250}, {0, 64, 128, 192, 255});
  run("fullover", {0, 255, 0, 255}, {50, 100, 150, 200});
  run("fullnonuni", {0, 255, 10, 251}, {0, 10, 240, 255}); // non-uniform spacing
  run("full3", {5, 250, 3}, {0, 100, 255});
  run("full2", {10, 240}, {50, 200});
  run("fullunsorted", {7, 8, 9}, {100, 50, 200});
  // 16-bit full curve: src[i] == i, 65536 wide
  std::vector<Npp16u> f16(65536);
  for (int i = 0; i < 65536; ++i) f16[i] = static_cast<Npp16u>(i);
  lut_c1<Npp16u, Npp32s>("nppiLUT_Cubic_16u_C1R", "cubic/16u/C1/full", 65536, 1, 5, f16,
                         {10, 20000, 40000, 60000, 65530}, {0, 16384, 32768, 49152, 65535}, ValueKind::U16,
                         [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp32s *v,
                            const Npp32s *l, int n, NppStreamContext c) {
                           return nppiLUT_Cubic_16u_C1R_Ctx(s, ss, d, ds, o, v, l, n, c);
                         });
}

// height-0 ROI (non-empty width): does the kernel still launch?
TEST_F(NppiLutDiff, Lut_EmptyRoi_H0) {
  const std::vector<Npp32s> okv = {1, 2, 3}, okl = {10, 100, 200};
  std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 715);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 716);
  DeviceMemory<Npp8u> &s = dev_buf(hs);
  DeviceMemory<Npp8u> &d = dev_buf(hd);
  DeviceMemory<Npp32s> &v = dev_buf(okv);
  DeviceMemory<Npp32s> &l = dev_buf(okl);
  NppStreamContext ctx = current_ctx();
  clear_cuda_error();
  NppStatus st = nppiLUT_8u_C1R_Ctx(s.get(), 16, d.get(), 16, NppiSize{16, 0}, v.get(), l.get(), 3, ctx);
  Case c("nppiLUT/contract/h0", "nppiLUT_8u_C1R");
  c.status(st);
  c.finish();
  clear_cuda_error();
  st = nppiLUT_Linear_8u_C1R_Ctx(s.get(), 16, d.get(), 16, NppiSize{16, 0}, v.get(), l.get(), 3, ctx);
  Case c2("nppiLUT/linear/contract/h0", "nppiLUT_Linear_8u_C1R");
  c2.status(st);
  c2.finish();
  clear_cuda_error();
  st = nppiLUT_Cubic_8u_C1R_Ctx(s.get(), 16, d.get(), 16, NppiSize{16, 0}, v.get(), l.get(), 3, ctx);
  Case c3("nppiLUT/cubic/contract/h0", "nppiLUT_Cubic_8u_C1R");
  c3.status(st);
  c3.finish();
}
