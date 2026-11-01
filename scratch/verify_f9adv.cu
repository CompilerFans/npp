// Direct comparison of the mcMPP nppiFilterBoxBorderAdvanced executor against
// NVIDIA NPP 13.4 (RTX 5090).  Value comparisons are restricted to the
// geometry where NVIDIA implements the documented replicate-box contract
// (oSizeROI.width == oSrcSize.width, oSrcOffset.x == 0); status comparisons
// cover the full measured validation matrix.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>
#include "nppi_filter_window_executor.h"

extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

static NppStreamContext g_ctx;
static int g_fail = 0, g_cases = 0;

static bool same_status(NppStatus a, NppStatus b) {
  if (a == b) return true;
  return ((int)a == -9999 && (int)b == -14) || ((int)a == -14 && (int)b == -9999);
}

template <class T> struct Dev {
  T *p = nullptr;
  explicit Dev(size_t n) { cudaMalloc(&p, (n ? n : 1) * sizeof(T)); cudaMemset(p, 0, (n ? n : 1) * sizeof(T)); }
  ~Dev() { cudaFree(p); }
};

static std::vector<float> rnd(size_t n, unsigned seed, bool ints, int esize) {
  std::vector<float> v(n);
  unsigned s = seed ? seed : 1;
  for (auto &x : v) {
    s = s * 1664525u + 1013904223u;
    x = (float)((int)(s >> 8) % 61 - 30) / 8.0f;
    if (ints) {
      int lo = esize == 1 ? 0 : (esize == 2 ? (int)-3000 : 0);
      int hi = esize == 1 ? 255 : (esize == 2 ? 3000 : 0);
      x = (float)(lo + (int)(s % (unsigned)(hi - lo + 1)));
    }
  }
  return v;
}

template <class T> static std::vector<T> to_typed(const std::vector<float> &f) {
  std::vector<T> v(f.size());
  for (size_t i = 0; i < f.size(); ++i) v[i] = (T)f[i];
  return v;
}

template <class T> static bool same(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) return (double)a == (double)b;
  else return a == b;
}

// reference + mcMPP calls for every instantiation
struct Call8u { static const char *tag() { return "8u_C1R"; } };
#define ADV_CALL(SUF, TY, CH)                                                                   \
  struct Adv_##SUF##_##CH {                                                                     \
    using T = TY;                                                                               \
    static constexpr int Ch = CH;                                                               \
    static const char *tag() { return #SUF "_" #CH; }                                            \
    static NppStatus nv(const T *s, int ss, NppiSize sz, NppiPoint o, T *d, int ds, NppiSize r,  \
                        NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *pb) {                  \
      return nppiFilterBoxBorderAdvanced_##SUF##_C##CH##R_Ctx(s, ss, sz, o, d, ds, r, m, a, b,   \
                                                              pb, g_ctx);                        \
    }                                                                                            \
    static NppStatus mp(const T *s, int ss, NppiSize sz, NppiPoint o, T *d, int ds, NppiSize r,  \
                        NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *pb) {                  \
      return nppi::filter::filterBoxBorderAdvanced<T, CH>(s, ss, sz, o, d, ds, r, m, a, b, pb,   \
                                                          g_ctx.hStream);                        \
    }                                                                                            \
  };
ADV_CALL(8u, Npp8u, 1) ADV_CALL(8u, Npp8u, 3) ADV_CALL(8u, Npp8u, 4)
ADV_CALL(16u, Npp16u, 1) ADV_CALL(16u, Npp16u, 3) ADV_CALL(16u, Npp16u, 4)
ADV_CALL(16s, Npp16s, 1) ADV_CALL(16s, Npp16s, 3) ADV_CALL(16s, Npp16s, 4)
ADV_CALL(32f, Npp32f, 1) ADV_CALL(32f, Npp32f, 3) ADV_CALL(32f, Npp32f, 4)
ADV_CALL(64f, Npp64f, 1)

template <class A>
static void run(const char *name, int W, int H, NppiSize mask, NppiPoint anchor, NppiPoint off,
                unsigned seed, bool compare_values) {
  using T = typename A::T;
  constexpr int CH = A::Ch;
  const int M = 8;
  const int px = CH * (int)sizeof(T);
  const int stride = ((W + 2 * M) * px + 15) / 16 * 16;
  const size_t sn = (size_t)(H + 2 * M) * (stride / sizeof(T)) + 64;
  const int dstride = ((W * px) + 15) / 16 * 16;
  const size_t dn = (size_t)(dstride / sizeof(T)) * (H + 2) + 16;
  auto hs = to_typed<T>(rnd(sn, seed, !std::is_floating_point_v<T>, (int)sizeof(T)));
  Dev<T> s(sn), d1(dn), d2(dn);
  cudaMemcpy(s.p, hs.data(), sn * sizeof(T), cudaMemcpyHostToDevice);
  std::vector<T> fill(dn, (T)171);
  cudaMemcpy(d1.p, fill.data(), dn * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d2.p, fill.data(), dn * sizeof(T), cudaMemcpyHostToDevice);
  const size_t bufBytes = (size_t)(W * H * CH + 2) * 8;
  Dev<Npp8u> pb(bufBytes);
  const T *ps = s.p + (size_t)(M + off.y) * (stride / sizeof(T)) + (size_t)(M + off.x) * CH;
  NppStatus r1 = A::nv(ps, stride, NppiSize{W, H}, off, d1.p, dstride, NppiSize{W, H}, mask,
                       anchor, NPP_BORDER_REPLICATE, pb.p);
  NppStatus r2 = A::mp(ps, stride, NppiSize{W, H}, off, d2.p, dstride, NppiSize{W, H}, mask,
                       anchor, NPP_BORDER_REPLICATE, pb.p);
  ++g_cases;
  if (!same_status(r1, r2)) {
    ++g_fail;
    printf("FAIL %s %s status ref=%d ours=%d (W%d H%d m%dx%d a%d,%d off%d,%d)\n", name, A::tag(),
           (int)r1, (int)r2, W, H, mask.width, mask.height, anchor.x, anchor.y, off.x, off.y);
    return;
  }
  if (r1 != NPP_NO_ERROR || !compare_values) return;
  std::vector<T> h1(dn), h2(dn);
  cudaMemcpy(h1.data(), d1.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), d2.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  int bad = 0, first = -1;
  for (size_t i = 0; i < dn; ++i)
    if (!same(h1[i], h2[i])) {
      ++bad;
      if (first < 0) first = (int)i;
    }
  if (bad) {
    ++g_fail;
    printf("FAIL %s %s values %d/%zu (W%d H%d m%dx%d a%d,%d off%d,%d) first=%d ref=%g ours=%g\n",
           name, A::tag(), bad, dn, W, H, mask.width, mask.height, anchor.x, anchor.y, off.x,
           off.y, first, (double)h1[first], (double)h2[first]);
  }
}

template <class A> static void family() {
  // documented-use geometry: ROI == source, no x offset
  const NppiSize masks[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {5, 3}, {7, 7}, {9, 5},
                            {3, 3}, {15, 15}};
  const NppiPoint anchors[] = {{0, 0}, {1, 1}, {1, 1}, {2, 2}, {2, 2}, {2, 1}, {3, 3}, {4, 2},
                               {5, 5}, {7, 7}};
  int idx = 0;
  for (int i = 0; i < 10; ++i) {
    run<A>("clean13x11", 13, 11, masks[i], anchors[i], NppiPoint{0, 0}, 0xA000u + idx++, true);
    run<A>("clean8x6", 8, 6, masks[i], anchors[i], NppiPoint{0, 0}, 0xB000u + idx++, true);
  }
  run<A>("clean32x20", 32, 20, NppiSize{7, 5}, NppiPoint{3, 2}, NppiPoint{0, 0}, 0xC001u, true);
  run<A>("clean7x5", 7, 5, NppiSize{3, 3}, NppiPoint{1, 1}, NppiPoint{0, 0}, 0xC002u, true);
  run<A>("clean31x9", 31, 9, NppiSize{5, 5}, NppiPoint{2, 2}, NppiPoint{0, 0}, 0xC003u, true);
  // y-only source offset (accepted by the reference and still documented)
  run<A>("offy2", 10, 8, NppiSize{5, 5}, NppiPoint{2, 2}, NppiPoint{0, 2}, 0xC004u, false);
  // beyond the clean geometry: statuses only
  run<A>("offx2", 13, 11, NppiSize{3, 3}, NppiPoint{1, 1}, NppiPoint{2, 0}, 0xC005u, false);
  run<A>("roi_gt_src", 13, 11, NppiSize{3, 3}, NppiPoint{1, 1}, NppiPoint{0, 0}, 0xC006u, false);
}

template <class A> static void validation() {
  using T = typename A::T;
  constexpr int CH = A::Ch;
  const int es = (int)sizeof(T) * CH;
  Dev<T> s(1 << 14), d(1 << 14);
  Dev<Npp8u> pb(1 << 14);
  const NppiSize src{16, 12}, m3{3, 3};
  const NppiPoint a1{1, 1};
  struct Case { const char *name; NppiSize roi; NppiSize mask; NppiPoint anc; int ds; int ss;
                NppiBorderType bt; bool nullsrc, nulldst, nullbuf; NppiSize ssize; NppiPoint off; };
  const Case cases[] = {
      {"base", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"nullsrc", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, true, false, false, src, {0, 0}},
      {"nulldst", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, true, false, src, {0, 0}},
      {"nullbuf", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, true, src, {0, 0}},
      {"roinegw", {-3, 7}, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"roinegh", {7, -3}, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"roiempty", {0, 0}, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"roizerow", {0, 7}, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"roizeroh", {7, 0}, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"maskzero", src, {0, 0}, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"maskneg", src, {-3, 3}, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"anchorbad", src, m3, {5, 5}, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"dstepsml", src, m3, a1, 8, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"dstep0", src, m3, a1, 0, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"dstepneg", src, m3, a1, -16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"sstepsml", src, m3, a1, 16 * es, 8, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"mirror", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_MIRROR, false, false, false, src, {0, 0}},
      {"constb", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_CONSTANT, false, false, false, src, {0, 0}},
      {"wrap", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_WRAP, false, false, false, src, {0, 0}},
      {"srcsz0", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, {0, 0}, {0, 0}},
      {"offoob", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {16, 0}},
      {"offneg", src, m3, a1, 16 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, -1}},
      {"roigt", {32, 32}, m3, a1, 32 * es, 32 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"oddw", {7, 5}, m3, a1, 6 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"oddw_ok", {7, 5}, m3, a1, 7 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"evenw", {6, 5}, m3, a1, 2 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"evenw_ok", {6, 5}, m3, a1, 3 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"div4", {8, 5}, m3, a1, es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"div4_ok", {8, 5}, m3, a1, 2 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"unaligned", {8, 5}, m3, a1, 2 * es + es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
      {"stepneg_ok", {8, 5}, m3, a1, -2 * es, 16 * es, NPP_BORDER_REPLICATE, false, false, false, src, {0, 0}},
  };
  for (const auto &c : cases) {
    ++g_cases;
    NppStatus r1 = A::nv(c.nullsrc ? nullptr : s.p, c.ss, c.ssize, c.off, c.nulldst ? nullptr : d.p,
                         c.ds, c.roi, c.mask, c.anc, c.bt, c.nullbuf ? nullptr : pb.p);
    NppStatus r2 = A::mp(c.nullsrc ? nullptr : s.p, c.ss, c.ssize, c.off, c.nulldst ? nullptr : d.p,
                         c.ds, c.roi, c.mask, c.anc, c.bt, c.nullbuf ? nullptr : pb.p);
    if (!same_status(r1, r2)) {
      ++g_fail;
      printf("FAIL validation %s %s ref=%d ours=%d\n", A::tag(), c.name, (int)r1, (int)r2);
    }
  }
}

int main() {
  nppGetStreamContext(&g_ctx);
#define RUN(A)                                                                                  \
  family<A>();                                                                                  \
  validation<A>();
  RUN(Adv_8u_1) RUN(Adv_8u_3) RUN(Adv_8u_4)
  RUN(Adv_16u_1) RUN(Adv_16u_3) RUN(Adv_16u_4)
  RUN(Adv_16s_1) RUN(Adv_16s_3) RUN(Adv_16s_4)
  RUN(Adv_32f_1) RUN(Adv_32f_3) RUN(Adv_32f_4)
  RUN(Adv_64f_1)
  printf("verify_f9adv: %d cases, %d failures\n", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
