// Direct comparison of the mcMPP window executor (nppi_filter_window_executor)
// against NVIDIA NPP 13.4 on the RTX 5090.  The executor kernels are plain
// CUDA, so they can be linked next to libnppif and exercised here even though
// the full MPP build is not available in this worktree.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
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
  // mcMPP headers spell this status -9999 while NVIDIA NPP 13 uses -14
  return ((int)a == -9999 && (int)b == -14) || ((int)a == -14 && (int)b == -9999);
}

template <class T>
struct Dev {
  T *p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) { cudaMalloc(&p, (n ? n : 1) * sizeof(T)); cudaMemset(p, 0, (n ? n : 1) * sizeof(T)); }
  ~Dev() { cudaFree(p); }
  void upload(const std::vector<T> &h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
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

// ---------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------
template <class T>
static bool same(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) return std::fabs((double)a - (double)b) <= 1e-4 * (1 + std::fabs((double)b));
  else return a == b;
}

template <class T>
static void cmp_dst(const char *tag, const T *ref, const T *got, size_t n, const char *extra) {
  ++g_cases;
  for (size_t i = 0; i < n; ++i) {
    if (!same(ref[i], got[i])) {
      ++g_fail;
      printf("FAIL %-42s %s elem %zu ref=%g got=%g\n", tag, extra, i, (double)ref[i], (double)got[i]);
      return;
    }
  }
}

template <class T>
static std::vector<T> to_typed(const std::vector<float> &f) {
  std::vector<T> v(f.size());
  for (size_t i = 0; i < f.size(); ++i) {
    if constexpr (std::is_same_v<T, Npp64f>) v[i] = (T)f[i];
    else if constexpr (std::is_floating_point_v<T>) v[i] = (T)f[i];
    else v[i] = (T)(long long)f[i];
  }
  return v;
}

// geometry used by the value comparisons
static const int W = 9, H = 7, M = 10;
static const int SRCW = 17, SRCH = 13;
static const int OFFX = 3, OFFY = 2;

template <class T, int CH>
static void run_plain(const char *tag, NppStatus (*ref_fn)(const T *, int, T *, int, NppiSize, NppiSize, NppiPoint, NppStreamContext),
                      NppStatus (*our_fn)(const T *, int, T *, int, NppiSize, NppiSize, NppiPoint, cudaStream_t),
                      NppiSize mask, NppiPoint anchor, unsigned seed) {
  const int px = CH * (int)sizeof(T);
  const int stride = ((W + 2 * M) * px + 15) / 16 * 16;
  const size_t sn = (size_t)(H + 2 * M) * (stride / sizeof(T)) + 4;
  const int dstride = ((W * px) + 15) / 16 * 16;
  const size_t dn = (size_t)H * (dstride / sizeof(T)) + 4;
  auto hs = to_typed<T>(rnd(sn, seed, !std::is_floating_point_v<T>, (int)sizeof(T)));
  Dev<T> s(sn), d1(dn), d2(dn);
  s.upload(hs);
  std::vector<T> fill(dn, (T)171);
  d1.upload(fill);
  d2.upload(fill);
  const T *ps = s.p + (size_t)M * (stride / sizeof(T)) + (size_t)M * CH;
  NppStatus r1 = ref_fn(ps, stride, d1.p, dstride, NppiSize{W, H}, mask, anchor, g_ctx);
  NppStatus r2 = our_fn(ps, stride, d2.p, dstride, NppiSize{W, H}, mask, anchor, g_ctx.hStream);
  if (!same_status(r1, r2)) {
    ++g_fail; ++g_cases;
    printf("FAIL %-42s status ref=%d ours=%d\n", tag, (int)r1, (int)r2);
    return;
  }
  std::vector<T> h1(dn), h2(dn);
  cudaMemcpy(h1.data(), d1.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), d2.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  char extra[64];
  snprintf(extra, sizeof(extra), "mask%dx%d a%d,%d", mask.width, mask.height, anchor.x, anchor.y);
  if (r1 == NPP_NO_ERROR) cmp_dst(tag, h1.data(), h2.data(), dn, extra);
  // extra status checks: empty ROI and mask larger than the ROI
  {
    ++g_cases;
    const NppStatus a = ref_fn(ps, stride, d1.p, dstride, NppiSize{0, H}, mask, anchor, g_ctx);
    const NppStatus b = our_fn(ps, stride, d2.p, dstride, NppiSize{0, H}, mask, anchor, g_ctx.hStream);
    if (!same_status(a, b)) { ++g_fail; printf("FAIL %-42s empty-roi status ref=%d ours=%d\n", tag, (int)a, (int)b); }
    ++g_cases;
    const NppStatus c1 = ref_fn(ps, stride, d1.p, dstride, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{4, 2}, g_ctx);
    const NppStatus c2 = our_fn(ps, stride, d2.p, dstride, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{4, 2}, g_ctx.hStream);
    if (!same_status(c1, c2)) { ++g_fail; printf("FAIL %-42s mask>roi status ref=%d ours=%d\n", tag, (int)c1, (int)c2); }
  }
}

template <class T, int CH>
static void run_border(const char *tag, NppStatus (*ref_fn)(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, NppiBorderType, NppStreamContext),
                       NppStatus (*our_fn)(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, NppiBorderType, cudaStream_t),
                       NppiSize mask, NppiPoint anchor, NppiBorderType bt, NppiPoint off, unsigned seed) {
  const int px = CH * (int)sizeof(T);
  const int stride = ((SRCW + 2 * M) * px + 15) / 16 * 16;
  const size_t sn = (size_t)(SRCH + 2 * M) * (stride / sizeof(T)) + 4;
  const int dstride = ((W * px) + 15) / 16 * 16;
  const size_t dn = (size_t)H * (dstride / sizeof(T)) + 4;
  auto hs = to_typed<T>(rnd(sn, seed, !std::is_floating_point_v<T>, (int)sizeof(T)));
  Dev<T> s(sn), d1(dn), d2(dn);
  s.upload(hs);
  std::vector<T> fill(dn, (T)171);
  d1.upload(fill); d2.upload(fill);
  const T *ps = s.p + (size_t)M * (stride / sizeof(T)) + (size_t)M * CH;
  NppStatus r1 = ref_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d1.p, dstride, NppiSize{W, H}, mask, anchor, bt, g_ctx);
  NppStatus r2 = our_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d2.p, dstride, NppiSize{W, H}, mask, anchor, bt, g_ctx.hStream);
  if (!same_status(r1, r2)) {
    ++g_fail; ++g_cases;
    printf("FAIL %-42s status ref=%d ours=%d (border %d mask%dx%d)\n", tag, (int)r1, (int)r2, (int)bt, mask.width, mask.height);
    return;
  }
  std::vector<T> h1(dn), h2(dn);
  cudaMemcpy(h1.data(), d1.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), d2.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  char extra[80];
  snprintf(extra, sizeof(extra), "border%d mask%dx%d a%d,%d off%d,%d", (int)bt, mask.width, mask.height, anchor.x, anchor.y, off.x, off.y);
  if (r1 == NPP_NO_ERROR) cmp_dst(tag, h1.data(), h2.data(), dn, extra);
  {
    ++g_cases;
    const NppStatus a = ref_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d1.p, dstride, NppiSize{0, H}, mask, anchor, bt, g_ctx);
    const NppStatus b = our_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d2.p, dstride, NppiSize{0, H}, mask, anchor, bt, g_ctx.hStream);
    if (!same_status(a, b)) { ++g_fail; printf("FAIL %-42s empty-roi status ref=%d ours=%d\n", tag, (int)a, (int)b); }
    ++g_cases;
    const NppStatus c1 = ref_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d1.p, dstride, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{4, 2}, bt, g_ctx);
    const NppStatus c2 = our_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d2.p, dstride, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{4, 2}, bt, g_ctx.hStream);
    if (!same_status(c1, c2)) { ++g_fail; printf("FAIL %-42s mask>roi status ref=%d ours=%d\n", tag, (int)c1, (int)c2); }
  }
}


template <class T, int CH>
static void run_plain_med(const char *tag, NppStatus (*ref_fn)(const T *, int, T *, int, NppiSize, NppiSize, NppiPoint, Npp8u *, NppStreamContext),
                          NppStatus (*our_fn)(const T *, int, T *, int, NppiSize, NppiSize, NppiPoint, Npp8u *, cudaStream_t),
                          NppiSize mask, NppiPoint anchor, unsigned seed) {
  const int px = CH * (int)sizeof(T);
  const int stride = ((W + 2 * M) * px + 15) / 16 * 16;
  const size_t sn = (size_t)(H + 2 * M) * (stride / sizeof(T)) + 4;
  const int dstride = ((W * px) + 15) / 16 * 16;
  const size_t dn = (size_t)H * (dstride / sizeof(T)) + 4;
  auto hs = to_typed<T>(rnd(sn, seed, !std::is_floating_point_v<T>, (int)sizeof(T)));
  Dev<T> s(sn), d1(dn), d2(dn);
  s.upload(hs);
  std::vector<T> fill(dn, (T)171);
  d1.upload(fill); d2.upload(fill);
  const T *ps = s.p + (size_t)M * (stride / sizeof(T)) + (size_t)M * CH;
  NppStatus r1 = ref_fn(ps, stride, d1.p, dstride, NppiSize{W, H}, mask, anchor, nullptr, g_ctx);
  NppStatus r2 = our_fn(ps, stride, d2.p, dstride, NppiSize{W, H}, mask, anchor, nullptr, g_ctx.hStream);
  ++g_cases;
  if (!same_status(r1, r2)) { ++g_fail; printf("FAIL %s status ref=%d ours=%d (mask%dx%d)\n", tag, (int)r1, (int)r2, mask.width, mask.height); return; }
  if (r1 != NPP_NO_ERROR) return;
  std::vector<T> h1(dn), h2(dn);
  cudaMemcpy(h1.data(), d1.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), d2.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  char extra[64];
  snprintf(extra, sizeof(extra), "mask%dx%d a%d,%d", mask.width, mask.height, anchor.x, anchor.y);
  cmp_dst(tag, h1.data(), h2.data(), dn, extra);
  if (!std::is_same_v<T, Npp32f>) {
    for (size_t k = 0; k < dn; ++k) {
      if (!same(h1[k], h2[k])) {
        const int perRow = dstride / (int)sizeof(T);
        const int yy = (int)(k / perRow), xx = (int)(k % perRow);
        std::vector<T> win;
        for (int j = 0; j < mask.height; ++j)
          for (int i = 0; i < mask.width; ++i) {
            int sx = xx + i - anchor.x, sy = yy + j - anchor.y;
            win.push_back(hs[(size_t)(sy + M) * (stride / sizeof(T)) + (size_t)(sx + M) * CH]);
          }
        std::sort(win.begin(), win.end());
        printf("   MID median at (%d,%d) ref=%ld ours=%ld win:", xx, yy, (long)h1[k], (long)h2[k]);
        for (auto v : win) printf(" %ld", (long)v);
        printf("\n");
        break;
      }
    }
  }
}

template <class T, int CH>
static void run_border_med(const char *tag, NppStatus (*ref_fn)(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, Npp8u *, NppiBorderType, NppStreamContext),
                           NppStatus (*our_fn)(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize, NppiSize, NppiPoint, Npp8u *, NppiBorderType, cudaStream_t),
                           NppiSize mask, NppiPoint anchor, NppiBorderType bt, NppiPoint off, unsigned seed) {
  const int px = CH * (int)sizeof(T);
  const int stride = ((SRCW + 2 * M) * px + 15) / 16 * 16;
  const size_t sn = (size_t)(SRCH + 2 * M) * (stride / sizeof(T)) + 4;
  const int dstride = ((W * px) + 15) / 16 * 16;
  const size_t dn = (size_t)H * (dstride / sizeof(T)) + 4;
  auto hs = to_typed<T>(rnd(sn, seed, !std::is_floating_point_v<T>, (int)sizeof(T)));
  Dev<T> s(sn), d1(dn), d2(dn);
  s.upload(hs);
  std::vector<T> fill(dn, (T)171);
  d1.upload(fill); d2.upload(fill);
  const T *ps = s.p + (size_t)M * (stride / sizeof(T)) + (size_t)M * CH;
  NppStatus r1 = ref_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d1.p, dstride, NppiSize{W, H}, mask, anchor, nullptr, bt, g_ctx);
  NppStatus r2 = our_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d2.p, dstride, NppiSize{W, H}, mask, anchor, nullptr, bt, g_ctx.hStream);
  ++g_cases;
  if (!same_status(r1, r2)) { ++g_fail; printf("FAIL %s status ref=%d ours=%d (border%d mask%dx%d)\n", tag, (int)r1, (int)r2, (int)bt, mask.width, mask.height); return; }
  std::vector<T> h1(dn), h2(dn);
  cudaMemcpy(h1.data(), d1.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), d2.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  char extra[80];
  snprintf(extra, sizeof(extra), "border%d mask%dx%d a%d,%d off%d,%d", (int)bt, mask.width, mask.height, anchor.x, anchor.y, off.x, off.y);
  if (r1 == NPP_NO_ERROR) cmp_dst(tag, h1.data(), h2.data(), dn, extra);
  {
    ++g_cases;
    const NppStatus a = ref_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d1.p, dstride, NppiSize{0, H}, mask, anchor, nullptr, bt, g_ctx);
    const NppStatus b = our_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d2.p, dstride, NppiSize{0, H}, mask, anchor, nullptr, bt, g_ctx.hStream);
    if (!same_status(a, b)) { ++g_fail; printf("FAIL %-42s empty-roi status ref=%d ours=%d\n", tag, (int)a, (int)b); }
    ++g_cases;
    const NppStatus c1 = ref_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d1.p, dstride, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{4, 2}, nullptr, bt, g_ctx);
    const NppStatus c2 = our_fn(ps, stride, NppiSize{SRCW, SRCH}, off, d2.p, dstride, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{4, 2}, nullptr, bt, g_ctx.hStream);
    if (!same_status(c1, c2)) { ++g_fail; printf("FAIL %-42s mask>roi status ref=%d ours=%d\n", tag, (int)c1, (int)c2); }
  }
}

#define DECL_WRAP(T, FN, SIG)                                                                  \
  static NppStatus our_##FN(SIG) { return nppi::filter::FN; }

// hand-written thin wrappers (templates cannot be passed directly per variant)
#define OUR_BOX(T, TAG, CH, AC4)                                                               \
  static NppStatus our_box_##TAG(const T *s, int ss, T *d, int ds, NppiSize r, NppiSize m,     \
                                 NppiPoint a, cudaStream_t st) {                               \
    return nppi::filter::filterBox<T, CH, AC4>(s, ss, d, ds, r, m, a, st);                     \
  }
#define OUR_BOXB(T, TAG, CH, AC4)                                                              \
  static NppStatus our_boxb_##TAG(const T *s, int ss, NppiSize sz, NppiPoint o, T *d, int ds,  \
                                  NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,       \
                                  cudaStream_t st) {                                           \
    return nppi::filter::filterBoxBorder<T, CH, AC4>(s, ss, sz, o, d, ds, r, m, a, b, st);     \
  }
#define OUR_MIN(T, TAG, CH, AC4)                                                               \
  static NppStatus our_min_##TAG(const T *s, int ss, T *d, int ds, NppiSize r, NppiSize m,     \
                                 NppiPoint a, cudaStream_t st) {                               \
    return nppi::filter::filterMinMax<T, CH, AC4, false>(s, ss, d, ds, r, m, a, st);           \
  }
#define OUR_MAX(T, TAG, CH, AC4)                                                               \
  static NppStatus our_max_##TAG(const T *s, int ss, T *d, int ds, NppiSize r, NppiSize m,     \
                                 NppiPoint a, cudaStream_t st) {                               \
    return nppi::filter::filterMinMax<T, CH, AC4, true>(s, ss, d, ds, r, m, a, st);            \
  }
#define OUR_MINB(T, TAG, CH, AC4)                                                              \
  static NppStatus our_minb_##TAG(const T *s, int ss, NppiSize sz, NppiPoint o, T *d, int ds,  \
                                  NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,       \
                                  cudaStream_t st) {                                           \
    return nppi::filter::filterMinMaxBorder<T, CH, AC4, false>(s, ss, sz, o, d, ds, r, m, a, b, st); \
  }
#define OUR_MAXB(T, TAG, CH, AC4)                                                              \
  static NppStatus our_maxb_##TAG(const T *s, int ss, NppiSize sz, NppiPoint o, T *d, int ds,  \
                                  NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b,       \
                                  cudaStream_t st) {                                           \
    return nppi::filter::filterMinMaxBorder<T, CH, AC4, true>(s, ss, sz, o, d, ds, r, m, a, b, st); \
  }
#define OUR_MED(T, TAG, CH, AC4)                                                               \
  static NppStatus our_med_##TAG(const T *s, int ss, T *d, int ds, NppiSize r, NppiSize m,     \
                                 NppiPoint a, Npp8u *buf, cudaStream_t st) {                   \
    return nppi::filter::filterMedian<T, CH, AC4>(s, ss, d, ds, r, m, a, buf, st);             \
  }
#define OUR_MEDB(T, TAG, CH, AC4)                                                              \
  static NppStatus our_medb_##TAG(const T *s, int ss, NppiSize sz, NppiPoint o, T *d, int ds,  \
                                  NppiSize r, NppiSize m, NppiPoint a, Npp8u *buf,             \
                                  NppiBorderType b, cudaStream_t st) {                          \
    return nppi::filter::filterMedianBorder<T, CH, AC4>(s, ss, sz, o, d, ds, r, m, a, buf, b, st); \
  }

#define ALL_VARIANTS(FN)                                                                       \
  FN(Npp8u, 8u_C1R, Npp8u, 1, false)                                                           \
  FN(Npp8u, 8u_C3R, Npp8u, 3, false)                                                           \
  FN(Npp8u, 8u_C4R, Npp8u, 4, false)                                                           \
  FN(Npp8u, 8u_AC4R, Npp8u, 4, true)                                                           \
  FN(Npp16u, 16u_C1R, Npp16u, 1, false)                                                        \
  FN(Npp16u, 16u_C3R, Npp16u, 3, false)                                                        \
  FN(Npp16u, 16u_C4R, Npp16u, 4, false)                                                        \
  FN(Npp16u, 16u_AC4R, Npp16u, 4, true)                                                        \
  FN(Npp16s, 16s_C1R, Npp16s, 1, false)                                                        \
  FN(Npp16s, 16s_C3R, Npp16s, 3, false)                                                        \
  FN(Npp16s, 16s_C4R, Npp16s, 4, false)                                                        \
  FN(Npp16s, 16s_AC4R, Npp16s, 4, true)                                                        \
  FN(Npp32f, 32f_C1R, Npp32f, 1, false)                                                        \
  FN(Npp32f, 32f_C3R, Npp32f, 3, false)                                                        \
  FN(Npp32f, 32f_C4R, Npp32f, 4, false)                                                        \
  FN(Npp32f, 32f_AC4R, Npp32f, 4, true)


#define VB_BOX(T, TAG, CT, CH, AC4) OUR_BOX(T, TAG, CH, AC4)
#define VB_BOXB(T, TAG, CT, CH, AC4) OUR_BOXB(T, TAG, CH, AC4)
#define VB_MIN(T, TAG, CT, CH, AC4) OUR_MIN(T, TAG, CH, AC4)
#define VB_MAX(T, TAG, CT, CH, AC4) OUR_MAX(T, TAG, CH, AC4)
#define VB_MINB(T, TAG, CT, CH, AC4) OUR_MINB(T, TAG, CH, AC4)
#define VB_MAXB(T, TAG, CT, CH, AC4) OUR_MAXB(T, TAG, CH, AC4)
#define VB_MED(T, TAG, CT, CH, AC4) OUR_MED(T, TAG, CH, AC4)
#define VB_MEDB(T, TAG, CT, CH, AC4) OUR_MEDB(T, TAG, CH, AC4)

ALL_VARIANTS(VB_BOX)
ALL_VARIANTS(VB_BOXB)
ALL_VARIANTS(VB_MIN)
ALL_VARIANTS(VB_MAX)
ALL_VARIANTS(VB_MINB)
ALL_VARIANTS(VB_MAXB)
ALL_VARIANTS(VB_MED)
ALL_VARIANTS(VB_MEDB)
OUR_BOX(Npp64f, 64f_C1R, 1, false)

#define RUN_BOX(T, TAG, CT, CH, AC4)                                                           \
  {                                                                                            \
    run_plain<CT, CH>(("box/" #TAG), nppiFilterBox_##TAG##_Ctx, our_box_##TAG, NppiSize{3, 3}, \
                      NppiPoint{1, 1}, 0x11 + CH);                                             \
    run_plain<CT, CH>(("box/" #TAG), nppiFilterBox_##TAG##_Ctx, our_box_##TAG, NppiSize{5, 3}, \
                      NppiPoint{2, 1}, 0x22 + CH);                                             \
    run_plain<CT, CH>(("box/" #TAG), nppiFilterBox_##TAG##_Ctx, our_box_##TAG, NppiSize{2, 2}, \
                      NppiPoint{1, 0}, 0x33 + CH);                                             \
  }
#define RUN_BOXB(T, TAG, CT, CH, AC4)                                                          \
  {                                                                                            \
    run_border<CT, CH>(("boxborder/" #TAG), nppiFilterBoxBorder_##TAG##_Ctx, our_boxb_##TAG,   \
                       NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, NppiPoint{OFFX, OFFY}, 0x44 + CH); \
    run_border<CT, CH>(("boxborder/" #TAG), nppiFilterBoxBorder_##TAG##_Ctx, our_boxb_##TAG,   \
                       NppiSize{4, 2}, NppiPoint{3, 0}, NPP_BORDER_MIRROR, NppiPoint{0, 0}, 0x55 + CH); \
    run_border<CT, CH>(("boxborder/" #TAG), nppiFilterBoxBorder_##TAG##_Ctx, our_boxb_##TAG,   \
                       NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)5, NppiPoint{SRCW - 1, SRCH - 1}, 0x66 + CH); \
    run_border<CT, CH>(("boxborder/" #TAG), nppiFilterBoxBorder_##TAG##_Ctx, our_boxb_##TAG,   \
                       NppiSize{5, 5}, NppiPoint{2, 2}, NPP_BORDER_MIRROR, NppiPoint{1, 2}, 0x77 + CH); \
  }
#define RUN_MIN(T, TAG, CT, CH, AC4)                                                           \
  {                                                                                            \
    run_plain<CT, CH>(("min/" #TAG), nppiFilterMin_##TAG##_Ctx, our_min_##TAG, NppiSize{3, 3}, \
                      NppiPoint{1, 1}, 0x77 + CH);                                             \
    run_plain<CT, CH>(("min/" #TAG), nppiFilterMin_##TAG##_Ctx, our_min_##TAG, NppiSize{2, 2}, \
                      NppiPoint{1, 0}, 0x88 + CH);                                             \
  }
#define RUN_MAX(T, TAG, CT, CH, AC4)                                                           \
  {                                                                                            \
    run_plain<CT, CH>(("max/" #TAG), nppiFilterMax_##TAG##_Ctx, our_max_##TAG, NppiSize{3, 3}, \
                      NppiPoint{0, 0}, 0x99 + CH);                                             \
    run_plain<CT, CH>(("max/" #TAG), nppiFilterMax_##TAG##_Ctx, our_max_##TAG, NppiSize{5, 3}, \
                      NppiPoint{4, 2}, 0xAA + CH);                                             \
  }
#define RUN_MINB(T, TAG, CT, CH, AC4)                                                          \
  run_border<CT, CH>(("minborder/" #TAG), nppiFilterMinBorder_##TAG##_Ctx, our_minb_##TAG,     \
                     NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, NppiPoint{OFFX, OFFY}, 0xBB + CH);
#define RUN_MAXB(T, TAG, CT, CH, AC4)                                                          \
  run_border<CT, CH>(("maxborder/" #TAG), nppiFilterMaxBorder_##TAG##_Ctx, our_maxb_##TAG,     \
                     NppiSize{5, 3}, NppiPoint{2, 1}, NPP_BORDER_REPLICATE, NppiPoint{0, 0}, 0xCC + CH);
#define RUN_MED(T, TAG, CT, CH, AC4)                                                           \
  {                                                                                            \
    run_plain_med<CT, CH>(("median/" #TAG), nppiFilterMedian_##TAG##_Ctx, our_med_##TAG,           \
                      NppiSize{3, 3}, NppiPoint{1, 1}, 0xDD + CH);                             \
    run_plain_med<CT, CH>(("median/" #TAG), nppiFilterMedian_##TAG##_Ctx, our_med_##TAG,           \
                      NppiSize{5, 3}, NppiPoint{2, 1}, 0xEE + CH);                             \
    run_plain_med<CT, CH>(("median/" #TAG), nppiFilterMedian_##TAG##_Ctx, our_med_##TAG,           \
                      NppiSize{5, 5}, NppiPoint{2, 2}, 0xFF + CH);                             \
  }
#define RUN_MEDB(T, TAG, CT, CH, AC4)                                                          \
  {                                                                                            \
    run_border_med<CT, CH>(("medianborder/" #TAG), nppiFilterMedianBorder_##TAG##_Ctx, our_medb_##TAG, \
                       NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, NppiPoint{OFFX, OFFY}, 0x121 + CH); \
    run_border_med<CT, CH>(("medianborder/" #TAG), nppiFilterMedianBorder_##TAG##_Ctx, our_medb_##TAG, \
                       NppiSize{3, 5}, NppiPoint{1, 2}, NPP_BORDER_MIRROR, NppiPoint{0, 0}, 0x131 + CH); \
  }

static void values() {
  ALL_VARIANTS(RUN_BOX)
  ALL_VARIANTS(RUN_BOXB)
  ALL_VARIANTS(RUN_MIN)
  ALL_VARIANTS(RUN_MAX)
  ALL_VARIANTS(RUN_MINB)
  ALL_VARIANTS(RUN_MAXB)
  ALL_VARIANTS(RUN_MED)
  ALL_VARIANTS(RUN_MEDB)
  RUN_BOX(Npp64f, 64f_C1R, Npp64f, 1, false)
}

// buffered median (16x16 mask) for C1 variants
template <class T, int CH, class RefF, class OurF, class BufRef, class BufOur>
static void run_median_buffered(const char *tag, RefF ref_fn, OurF our_fn, BufRef buf_ref,
                                BufOur buf_our, unsigned seed) {
  const int px = CH * (int)sizeof(T);
  const int stride = ((W + 2 * M) * px + 15) / 16 * 16;
  const size_t sn = (size_t)(H + 2 * M) * (stride / sizeof(T)) + 4;
  const int dstride = ((W * px) + 15) / 16 * 16;
  const size_t dn = (size_t)H * (dstride / sizeof(T)) + 4;
  auto hs = to_typed<T>(rnd(sn, seed, !std::is_floating_point_v<T>, (int)sizeof(T)));
  Dev<T> s(sn), d1(dn), d2(dn);
  s.upload(hs);
  std::vector<T> fill(dn, (T)171);
  d1.upload(fill); d2.upload(fill);
  NppiSize mask{16, 16};
  Npp32u b1 = 0, b2 = 0;
  buf_ref(NppiSize{W, H}, mask, &b1, g_ctx);
  buf_our(NppiSize{W, H}, mask, &b2);
  if (b1 != b2) { ++g_fail; ++g_cases; printf("FAIL %s bufsize ref=%u ours=%u\n", tag, b1, b2); return; }
  Dev<Npp8u> buf(b1 ? b1 : 1);
  const T *ps = s.p + (size_t)M * (stride / sizeof(T)) + (size_t)M * CH;
  NppStatus r1 = ref_fn(ps, stride, d1.p, dstride, NppiSize{W, H}, mask, NppiPoint{8, 8}, buf.p, g_ctx);
  NppStatus r2 = our_fn(ps, stride, d2.p, dstride, NppiSize{W, H}, mask, NppiPoint{8, 8}, buf.p, g_ctx.hStream);
  if (!same_status(r1, r2)) { ++g_fail; ++g_cases; printf("FAIL %s status ref=%d ours=%d\n", tag, (int)r1, (int)r2); return; }
  std::vector<T> h1(dn), h2(dn);
  cudaMemcpy(h1.data(), d1.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2.data(), d2.p, dn * sizeof(T), cudaMemcpyDeviceToHost);
  cmp_dst(tag, h1.data(), h2.data(), dn, "buffered16x16");
}

#define BUF_OUR(T, CH)                                                                         \
  [](NppiSize r, NppiSize m, Npp32u *p) {                                                      \
    return nppi::filter::medianGetBufferSize<T, CH>(r, m, p);                                  \
  }

static void buffered() {
  run_median_buffered<Npp8u, 1>("median/8uC1", nppiFilterMedian_8u_C1R_Ctx, our_med_8u_C1R,
                                nppiFilterMedianGetBufferSize_8u_C1R_Ctx, BUF_OUR(Npp8u, 1), 0x201);
  run_median_buffered<Npp16u, 1>("median/16uC1", nppiFilterMedian_16u_C1R_Ctx, our_med_16u_C1R,
                                 nppiFilterMedianGetBufferSize_16u_C1R_Ctx, BUF_OUR(Npp16u, 1), 0x202);
  run_median_buffered<Npp16s, 1>("median/16sC1", nppiFilterMedian_16s_C1R_Ctx, our_med_16s_C1R,
                                 nppiFilterMedianGetBufferSize_16s_C1R_Ctx, BUF_OUR(Npp16s, 1), 0x203);
  run_median_buffered<Npp32f, 1>("median/32fC1", nppiFilterMedian_32f_C1R_Ctx, our_med_32f_C1R,
                                 nppiFilterMedianGetBufferSize_32f_C1R_Ctx, BUF_OUR(Npp32f, 1), 0x204);
}

// ---------------------------------------------------------------------------
// GetBufferSize grid comparison (all configs, masks 1..40, ROI 7x7)
// ---------------------------------------------------------------------------
typedef NppStatus (*PlainBufFn)(NppiSize, NppiSize, Npp32u *, NppStreamContext);
typedef NppStatus (*BorderBufFn)(NppiSize, NppiSize, Npp32u *, NppiBorderType, NppStreamContext);
typedef NppStatus (*HostBufFn)(NppiSize, NppiSize, Npp32u *);

#define BFN(T, TAG) nppiFilterMedianGetBufferSize_##TAG##_Ctx
#define BBFN(T, TAG) nppiFilterMedianBorderGetBufferSize_##TAG##_Ctx
static const PlainBufFn kPlain[16] = {
    BFN(Npp8u, 8u_C1R), BFN(Npp8u, 8u_C3R), BFN(Npp8u, 8u_C4R), BFN(Npp8u, 8u_AC4R),
    BFN(Npp16u, 16u_C1R), BFN(Npp16u, 16u_C3R), BFN(Npp16u, 16u_C4R), BFN(Npp16u, 16u_AC4R),
    BFN(Npp16s, 16s_C1R), BFN(Npp16s, 16s_C3R), BFN(Npp16s, 16s_C4R), BFN(Npp16s, 16s_AC4R),
    BFN(Npp32f, 32f_C1R), BFN(Npp32f, 32f_C3R), BFN(Npp32f, 32f_C4R), BFN(Npp32f, 32f_AC4R)};
static const BorderBufFn kBorder[16] = {
    BBFN(Npp8u, 8u_C1R), BBFN(Npp8u, 8u_C3R), BBFN(Npp8u, 8u_C4R), BBFN(Npp8u, 8u_AC4R),
    BBFN(Npp16u, 16u_C1R), BBFN(Npp16u, 16u_C3R), BBFN(Npp16u, 16u_C4R), BBFN(Npp16u, 16u_AC4R),
    BBFN(Npp16s, 16s_C1R), BBFN(Npp16s, 16s_C3R), BBFN(Npp16s, 16s_C4R), BBFN(Npp16s, 16s_AC4R),
    BBFN(Npp32f, 32f_C1R), BBFN(Npp32f, 32f_C3R), BBFN(Npp32f, 32f_C4R), BBFN(Npp32f, 32f_AC4R)};
static const char *kName[16] = {"8uC1", "8uC3", "8uC4", "8uAC4", "16uC1", "16uC3", "16uC4", "16uAC4",
                                "16sC1", "16sC3", "16sC4", "16sAC4", "32fC1", "32fC3", "32fC4", "32fAC4"};

static void bufsize() {
  typedef NppStatus (*BorderHostFn)(NppiSize, NppiSize, Npp32u *, NppiBorderType);
  HostBufFn ours[16] = {
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp8u, 1>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp8u, 3>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp8u, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp8u, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16u, 1>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16u, 3>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16u, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16u, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16s, 1>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16s, 3>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16s, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp16s, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp32f, 1>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp32f, 3>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp32f, 4>(r, m, p); },
      [](NppiSize r, NppiSize m, Npp32u *p) { return nppi::filter::medianGetBufferSize<Npp32f, 4>(r, m, p); }};
  BorderHostFn ours_border[16] = {
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp8u, 1>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp8u, 3>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp8u, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp8u, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16u, 1>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16u, 3>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16u, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16u, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16s, 1>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16s, 3>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16s, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp16s, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp32f, 1>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp32f, 3>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp32f, 4>(r, m, p, b); },
      [](NppiSize r, NppiSize m, Npp32u *p, NppiBorderType b) { return nppi::filter::medianBorderGetBufferSize<Npp32f, 4>(r, m, p, b); }};
  for (int k = 0; k < 16; ++k) {
    for (int h = 1; h <= 40; ++h)
      for (int w = 1; w <= 40; ++w) {
        Npp32u a = 0xDEADBEEF, b = 0xDEADBEEF;
        NppStatus s1 = kPlain[k](NppiSize{7, 7}, NppiSize{w, h}, &a, g_ctx);
        NppStatus s2 = ours[k](NppiSize{7, 7}, NppiSize{w, h}, &b);
        ++g_cases;
        if (s1 != s2 || a != b) {
          ++g_fail;
          if (g_fail < 30)
            printf("FAIL bufsize plain %s %dx%d ref=%u(%d) ours=%u(%d)\n", kName[k], w, h, a, (int)s1, b, (int)s2);
        }
      }
    for (int h = 1; h <= 40; ++h)
      for (int w = 1; w <= 40; ++w) {
        Npp32u a = 0xDEADBEEF, b = 0xDEADBEEF;
        NppStatus s1 = kBorder[k](NppiSize{7, 7}, NppiSize{w, h}, &a, NPP_BORDER_REPLICATE, g_ctx);
        NppStatus s2 = ours_border[k](NppiSize{7, 7}, NppiSize{w, h}, &b, NPP_BORDER_REPLICATE);
        ++g_cases;
        if (s1 != s2 || a != b) {
          ++g_fail;
          if (g_fail < 30)
            printf("FAIL bufsize border %s %dx%d ref=%u(%d) ours=%u(%d)\n", kName[k], w, h, a, (int)s1, b, (int)s2);
        }
      }
  }
  // error paths
  const NppiSize bad[][2] = {{{-2, 5}, {3, 3}}, {{7, 7}, {-3, 3}}, {{7, 7}, {3, -3}}, {{0, 5}, {3, 3}}, {{7, 7}, {0, 0}}};
  for (int k = 0; k < 16; ++k) {
    for (auto &c : bad) {
      Npp32u a = 0xDEADBEEF, b = 0xDEADBEEF;
      NppStatus s1 = kPlain[k](c[0], c[1], &a, g_ctx);
      NppStatus s2 = ours[k](c[0], c[1], &b);
      ++g_cases;
      if (s1 != s2 || a != b) { ++g_fail; printf("FAIL bufsize err %s ref=%u(%d) ours=%u(%d)\n", kName[k], a, (int)s1, b, (int)s2); }
    }
  }
}

int main() {
  nppGetStreamContext(&g_ctx);
  if (g_ctx.hStream == 0) { /* keep default stream */ }
  values();
  buffered();
  bufsize();
  printf("verify_f9: %d cases, %d failures\n", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
