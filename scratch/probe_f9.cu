// NVIDIA NPP 13.4 (RTX 5090) semantics probe for package f9 families:
// FilterBox, FilterBoxBorder, FilterBoxBorderAdvanced, FilterMax/Min(+Border),
// FilterMedian(+Border, +GetBufferSize).
// Sections selected by argv[1]; "all" runs everything in order.
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"

// NPP 13 removed the managed-stream API from the public headers, but libnppc
// still exports the symbols.
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppSetStream(cudaStream_t hStream);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
unsigned int nppGetStreamNumSMs(void);
unsigned int nppGetStreamMaxThreadsPerSM(void);
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------- utilities
static NppStreamContext g_ctx;
static void init_ctx() {
  NppStatus st = nppGetStreamContext(&g_ctx);
  printf("nppGetStreamContext -> %d\n", (int)st);
  fflush(stdout);
}

static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_ERROR: return "NPP_ERROR";
    case NPP_BAD_ARGUMENT_ERROR: return "BAD_ARGUMENT";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_NULL_POINTER_ERROR: return "NULL_POINTER";
    case NPP_OUT_OFF_RANGE_ERROR: return "OUT_OFF_RANGE";
    case NPP_DATA_TYPE_ERROR: return "DATA_TYPE";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOT_EVEN_STEP";
    case NPP_MASK_SIZE_ERROR: return "MASK_SIZE";
    case NPP_ANCHOR_ERROR: return "ANCHOR";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOT_SUPPORTED_MODE";
    case NPP_DIVISOR_ERROR: return "DIVISOR";
    case NPP_RECTANGLE_ERROR: return "RECTANGLE";
    case NPP_NO_MEMORY_ERROR: return "NO_MEMORY";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR: return "CUDA_KERNEL";
    case NPP_MEMORY_ALLOCATION_ERR: return "MEM_ALLOC";
    case NPP_ERROR_RESERVED: return "RESERVED";
    case NPP_NO_OPERATION_WARNING: return "NO_OP_WARN";
    case NPP_TEXTURE_BIND_ERROR: return "TEXTURE_BIND";
    default: return "OTHER";
  }
}

#define CK(what, call)                                                        \
  do {                                                                        \
    NppStatus st_ = (call);                                                   \
    printf("%-46s %4d %s\n", what, (int)st_, sn(st_));                        \
    fflush(stdout);                                                           \
  } while (0)

struct DevBuf {
  void *p = nullptr;
  size_t n = 0;
  ~DevBuf() { if (p) cudaFree(p); }
  void alloc(size_t bytes) {
    if (p) cudaFree(p);
    p = nullptr;
    if (bytes) cudaMalloc(&p, bytes);
    n = bytes;
  }
  template <class T> void put(const std::vector<T> &h) {
    alloc(h.size() * sizeof(T));
    if (p) cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  template <class T> std::vector<T> get(size_t count) {
    std::vector<T> h(count ? count : n / sizeof(T));
    if (p && !h.empty()) cudaMemcpy(h.data(), p, h.size() * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

// ------------------------------------------------------------- test patterns
template <class T> T pat(int i, int j, uint64_t seed);
template <> Npp8u pat<Npp8u>(int i, int j, uint64_t s) {
  return (Npp8u)((i * 29 + j * 47 + (int)(s & 63) * 5) % 251);
}
template <> Npp16u pat<Npp16u>(int i, int j, uint64_t s) {
  return (Npp16u)((i * 1013 + j * 3571 + (int)(s & 1023)) % 65521);
}
template <> Npp16s pat<Npp16s>(int i, int j, uint64_t s) {
  return (Npp16s)(((i * 1013 + j * 3571 + (int)(s & 1023)) % 65521) - 32760);
}
template <> Npp32f pat<Npp32f>(int i, int j, uint64_t s) {
  return (float)(((i * 7 + j * 13 + (int)(s & 255)) % 64) * 0.25) - 8.0f;
}
template <> Npp64f pat<Npp64f>(int i, int j, uint64_t s) {
  return (double)(((i * 7 + j * 13 + (int)(s & 255)) % 64) * 0.25) - 8.0;
}

// ------------------------------------------------------------------- scenes
// Source scene: AW x AH allocation, image/ROI origin at (ox, oy).
static const int AW = 20, AH = 20;

template <class T>
struct Scene {
  std::vector<T> h;
  DevBuf d;
  int ox, oy;
  int stepBytes;
  Scene(int ox_, int oy_, uint64_t seed) : h(AW * AH), ox(ox_), oy(oy_), stepBytes(AW * (int)sizeof(T)) {
    for (int j = 0; j < AH; ++j)
      for (int i = 0; i < AW; ++i) h[j * AW + i] = pat<T>(i, j, seed);
    d.put(h);
  }
  const T *src() const { return (const T *)d.p + (size_t)oy * AW + ox; }
  // raw allocation read at pSrc-relative (sx, sy)
  T raw(int sx, int sy) const { return h[(size_t)(oy + sy) * AW + (ox + sx)]; }
};

// Destination buffer with canary margins.
template <class T>
struct Dst {
  static const int M = 4;  // margin
  int W, H;
  int stepBytes;
  DevBuf d;
  std::vector<T> canary;  // expected untouched pixels
  Dst(int W_, int H_) : W(W_), H(H_), stepBytes((W_ + M) * (int)sizeof(T)) {
    int fw = W + M, fh = H + M;
    canary.assign(fw * fh, T(0xAB));
    d.put(canary);
  }
  std::vector<T> take() {
    std::vector<T> v = d.get<T>(0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) canary[y * (W + M) + x] = v[y * (W + M) + x];
    return v;
  }
  bool canaryIntact() {
    std::vector<T> v = d.get<T>(0);
    bool ok = true;
    for (int y = 0; y < H + M; ++y)
      for (int x = 0; x < W + M; ++x)
        if ((x >= W || y >= H) && v[(size_t)y * (W + M) + x] != canary[(size_t)y * (W + M) + x]) ok = false;
    if (ok) return true;
    // dump the first clobbered positions (frame coords: x + y*(W+M))
    int fw = W + M;
    printf("  canary clobber at:");
    int shown = 0;
    for (int y = 0; y < H + M && shown < 16; ++y)
      for (int x = 0; x < fw && shown < 16; ++x) {
        size_t k = (size_t)y * fw + x;
        if (v[k] != canary[k] && !(x < W && y < H)) {
          printf(" (%d,%d)=%d", x, y, (int)v[k]);
          ++shown;
        }
      }
    printf("\n");
    return false;
  }
  std::vector<T> region() {
    std::vector<T> v = d.get<T>(0);
    std::vector<T> r(W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) r[y * W + x] = v[y * (W + M) + x];
    return r;
  }
  void printRegion(const char *tag) {
    std::vector<T> r = region();
    printf("%s:\n", tag);
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        if constexpr (std::is_floating_point_v<T>) printf("%g ", (double)r[y * W + x]);
        else printf("%d ", (int)r[y * W + x]);
      }
      printf("\n");
    }
  }
};

// ------------------------------------------------------------------- models
enum Mode { M_ZERO, M_REPL, M_RAW };
enum DivMode { D_AREA, D_COUNT };
enum RoundMode { R_TRUNC, R_FLOOR, R_HALF_UP, R_HALF_EVEN, R_FLOAT, R_DBL };

static const char *modeName[] = {"zero", "repl", "raw"};
static const char *divName[] = {"area", "count"};
static const char *roundName[] = {"trunc", "floor", "halfup", "halfeven", "f32", "f64"};

// Collect the sample values for one dst pixel.  The source image (for border
// modes) has size srcW x srcH and its origin sits at pSrc - (offX, offY);
// for non-border scenes pass srcW=roiW etc with off 0.
template <class T>
static void samples(const Scene<T> &sc, int x, int y, int mw, int mh, int ax, int ay, int srcW,
                    int srcH, int offX, int offY, Mode m, std::vector<T> &out) {
  out.clear();
  for (int j = 0; j < mh; ++j) {
    for (int i = 0; i < mw; ++i) {
      int sx = x + i - ax, sy = y + j - ay;
      if (m == M_RAW) {
        out.push_back(sc.raw(sx, sy));
        continue;
      }
      int ix = sx + offX, iy = sy + offY;
      if (m == M_REPL) {
        ix = std::max(0, std::min(srcW - 1, ix));
        iy = std::max(0, std::min(srcH - 1, iy));
      } else if (ix < 0 || ix >= srcW || iy < 0 || iy >= srcH) {
        continue;  // M_ZERO: skip
      }
      out.push_back(sc.raw(ix - offX, iy - offY));
    }
  }
}

static int64_t idiv(int64_t a, int64_t b, RoundMode r) {
  switch (r) {
    case R_TRUNC: return a / b;
    case R_FLOOR: {
      int64_t q = a / b, rem = a % b;
      return (rem != 0 && ((rem < 0) != (b < 0))) ? q - 1 : q;
    }
    case R_HALF_UP: return (int64_t)std::floor((long double)a / b + 0.5L);
    case R_HALF_EVEN: {
      long double q = (long double)a / b;
      long double fl = std::floor(q);
      long double diff = q - fl;
      long double r2 = (diff > 0.5L) ? fl + 1 : (diff < 0.5L) ? fl : ((int64_t)fl % 2 == 0) ? fl : fl + 1;
      return (int64_t)r2;
    }
    default: return a / b;
  }
}

template <class T>
static T clampval(T v, long double lo, long double hi) {
  if constexpr (std::is_floating_point_v<T>) return v;
  else {
    if (v < (T)lo) return (T)lo;
    if (v > (T)hi) return (T)hi;
    return v;
  }
}

// Compute the model output image for a "sum/area" (box) filter.
template <class T>
static std::vector<T> boxModel(const Scene<T> &sc, int roiW, int roiH, int mw, int mh, int ax, int ay,
                               int srcW, int srcH, int offX, int offY, Mode m, DivMode dv, RoundMode r) {
  std::vector<T> out(roiW * roiH);
  std::vector<T> s;
  for (int y = 0; y < roiH; ++y)
    for (int x = 0; x < roiW; ++x) {
      samples(sc, x, y, mw, mh, ax, ay, srcW, srcH, offX, offY, m, s);
      T v{};
      if (s.empty()) {
        v = T(0);
      } else if constexpr (std::is_floating_point_v<T>) {
        if (r == R_FLOAT) {
          float acc = 0.f;
          for (T t : s) acc = acc + (float)t;
          v = (T)(acc / (float)(dv == D_AREA ? mw * mh : (int)s.size()));
        } else {
          double acc = 0.0;
          for (T t : s) acc = acc + (double)t;
          v = (T)(acc / (double)(dv == D_AREA ? mw * mh : (int)s.size()));
        }
      } else {
        int64_t acc = 0;
        for (T t : s) acc += (int64_t)t;
        int64_t den = dv == D_AREA ? (int64_t)mw * mh : (int64_t)s.size();
        int64_t q = den ? idiv(acc, den, r) : 0;
        v = clampval<T>((long double)q, (long double)std::numeric_limits<T>::min(),
                        (long double)std::numeric_limits<T>::max());
      }
      out[y * roiW + x] = v;
    }
  return out;
}

template <class T>
static std::vector<T> minmaxModel(const Scene<T> &sc, int roiW, int roiH, int mw, int mh, int ax,
                                  int ay, int srcW, int srcH, int offX, int offY, Mode m, bool isMax,
                                  T emptyVal) {
  std::vector<T> out(roiW * roiH);
  std::vector<T> s;
  for (int y = 0; y < roiH; ++y)
    for (int x = 0; x < roiW; ++x) {
      samples(sc, x, y, mw, mh, ax, ay, srcW, srcH, offX, offY, m, s);
      T v = emptyVal;
      if (!s.empty()) {
        v = s[0];
        for (T t : s) {
          if (isMax ? (t > v) : (t < v)) v = t;
        }
      }
      out[y * roiW + x] = v;
    }
  return out;
}

template <class T>
static std::vector<T> medianModel(const Scene<T> &sc, int roiW, int roiH, int mw, int mh, int ax,
                                  int ay, int srcW, int srcH, int offX, int offY, Mode m,
                                  int pick /*0=lower mid,1=upper mid*/) {
  std::vector<T> out(roiW * roiH);
  std::vector<T> s;
  for (int y = 0; y < roiH; ++y)
    for (int x = 0; x < roiW; ++x) {
      samples(sc, x, y, mw, mh, ax, ay, srcW, srcH, offX, offY, m, s);
      if (s.empty()) {
        out[y * roiW + x] = T(0);
        continue;
      }
      std::stable_sort(s.begin(), s.end());
      size_t idx = pick == 0 ? (s.size() - 1) / 2 : s.size() / 2;
      out[y * roiW + x] = s[idx];
    }
  return out;
}

template <class T>
static void report(const char *tag, const std::vector<T> &model, const std::vector<T> &got) {
  if (model == got) {
    printf("%-34s MATCH\n", tag);
    return;
  }
  size_t k = 0;
  while (k < model.size() && model[k] == got[k]) ++k;
  printf("%-34s diff@%zu model=%s got=%s\n", tag, k,
         std::to_string((double)model[k]).c_str(), std::to_string((double)got[k]).c_str());
}

// ============================================================================
// S1: nppiFilterBox value semantics (non-border)
// ============================================================================
template <class T>
static void boxValues(const char *tag, uint64_t seed, int mw, int mh, int ax, int ay) {
  printf("== S1 box %s mask %dx%d anchor %d,%d ==\n", tag, mw, mh, ax, ay);
  Scene<T> sc(8, 8, seed);
  const int W = 6, H = 6;
  Dst<T> dst(W, H);
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp8u>)
    st = nppiFilterBox_8u_C1R_Ctx(sc.src(), sc.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes,
                                  NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16u>)
    st = nppiFilterBox_16u_C1R_Ctx(sc.src(), sc.stepBytes, (Npp16u *)dst.d.p, dst.stepBytes,
                                   NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16s>)
    st = nppiFilterBox_16s_C1R_Ctx(sc.src(), sc.stepBytes, (Npp16s *)dst.d.p, dst.stepBytes,
                                   NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  else if constexpr (std::is_same_v<T, Npp32f>)
    st = nppiFilterBox_32f_C1R_Ctx(sc.src(), sc.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes,
                                   NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  else
    st = nppiFilterBox_64f_C1R_Ctx(sc.src(), sc.stepBytes, (Npp64f *)dst.d.p, dst.stepBytes,
                                   NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  printf("status %d %s canary=%s\n", (int)st, sn(st), dst.canaryIntact() ? "intact" : "CLOBBERED");
  if (st != NPP_NO_ERROR) return;
  std::vector<T> got = dst.region();
  char nm[64];
  if constexpr (std::is_floating_point_v<T>) {
    for (int m = 0; m < 3; ++m)
      for (int dv = 0; dv < 2; ++dv)
        for (int r = 4; r < 6; ++r) {
          snprintf(nm, 64, "%s/%s/%s", modeName[m], divName[dv], roundName[r]);
          report(nm, boxModel(sc, W, H, mw, mh, ax, ay, W, H, 0, 0, (Mode)m, (DivMode)dv, (RoundMode)r), got);
        }
  } else {
    for (int m = 0; m < 3; ++m)
      for (int dv = 0; dv < 2; ++dv)
        for (int r = 0; r < 4; ++r) {
          snprintf(nm, 64, "%s/%s/%s", modeName[m], divName[dv], roundName[r]);
          report(nm, boxModel(sc, W, H, mw, mh, ax, ay, W, H, 0, 0, (Mode)m, (DivMode)dv, (RoundMode)r), got);
        }
  }
  dst.printRegion("dst");
}

// ============================================================================
// S2: nppiFilterBoxBorder: supported border modes + replicate geometry
// ============================================================================
static void boxBorderModes() {
  printf("== S2 nppiFilterBoxBorder_8u_C1R border modes ==\n");
  Scene<Npp8u> sc(8, 8, 0x1001);
  const int W = 7, H = 7;
  Dst<Npp8u> dst(W, H);
  NppiSize srcSize{8, 8};
  NppiPoint off{2, 3};
  const int types[] = {-1, 0, 1, 2, 3, 4, 5, 7};
  const char *tn[] = {"neg1", "UNDEF", "CONST", "REPL", "WRAP", "MIRROR", "5", "7"};
  for (int k = 0; k < 8; ++k) {
    Dst<Npp8u> d2(W, H);
    NppStatus st = nppiFilterBoxBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, srcSize, off,
                                                  (Npp8u *)d2.d.p, d2.stepBytes, NppiSize{W, H},
                                                  NppiSize{3, 3}, NppiPoint{1, 1},
                                                  (NppiBorderType)types[k], g_ctx);
    printf("border %-7s -> %4d %s\n", tn[k], (int)st, sn(st));
    fflush(stdout);
  }
}

static void boxBorderValues() {
  printf("== S2b nppiFilterBoxBorder_8u_C1R replicate values ==\n");
  Scene<Npp8u> sc(8, 8, 0x1002);
  const int W = 7, H = 7;
  const int sw = 8, sh = 8, offx = 2, offy = 3;
  Dst<Npp8u> dst(W, H);
  NppStatus st = nppiFilterBoxBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{sw, sh},
                                                NppiPoint{offx, offy}, (Npp8u *)dst.d.p, dst.stepBytes,
                                                NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                NPP_BORDER_REPLICATE, g_ctx);
  printf("status %d %s canary=%s\n", (int)st, sn(st), dst.canaryIntact() ? "intact" : "CLOBBERED");
  if (st != NPP_NO_ERROR) return;
  std::vector<Npp8u> got = dst.region();
  char nm[64];
  for (int m = 0; m < 3; ++m)
    for (int dv = 0; dv < 2; ++dv)
      for (int r = 0; r < 4; ++r) {
        snprintf(nm, 64, "%s/%s/%s", modeName[m], divName[dv], roundName[r]);
        report(nm, boxModel(sc, W, H, 3, 3, 1, 1, sw, sh, offx, offy, (Mode)m, (DivMode)dv, (RoundMode)r), got);
      }
  dst.printRegion("dst");
  // 5x5 mask, non-square
  Dst<Npp8u> dst2(W, H);
  st = nppiFilterBoxBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy},
                                      (Npp8u *)dst2.d.p, dst2.stepBytes, NppiSize{W, H},
                                      NppiSize{5, 3}, NppiPoint{2, 1}, NPP_BORDER_REPLICATE, g_ctx);
  printf("mask 5x3 anchor 2,1 status %d %s\n", (int)st, sn(st));
  if (st == NPP_NO_ERROR) {
    std::vector<Npp8u> got2 = dst2.region();
    for (int m = 0; m < 3; ++m)
      for (int dv = 0; dv < 2; ++dv)
        for (int r = 0; r < 4; ++r) {
          snprintf(nm, 64, "m5x3 %s/%s/%s", modeName[m], divName[dv], roundName[r]);
          report(nm, boxModel(sc, W, H, 5, 3, 2, 1, sw, sh, offx, offy, (Mode)m, (DivMode)dv, (RoundMode)r), got2);
        }
  }
}

// ============================================================================
// S3: nppiFilterMax / Min values (non-border)
// ============================================================================
template <class T>
static void minmaxValues(const char *tag, uint64_t seed, bool isMax, int mw, int mh, int ax, int ay,
                         T emptyVal) {
  printf("== S3 %s %s mask %dx%d anchor %d,%d ==\n", isMax ? "max" : "min", tag, mw, mh, ax, ay);
  Scene<T> sc(8, 8, seed);
  const int W = 6, H = 6;
  Dst<T> dst(W, H);
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp8u>) {
    st = isMax ? nppiFilterMax_8u_C1R_Ctx(sc.src(), sc.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes,
                                          NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx)
               : nppiFilterMin_8u_C1R_Ctx(sc.src(), sc.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes,
                                          NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  } else {
    st = isMax ? nppiFilterMax_32f_C1R_Ctx(sc.src(), sc.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes,
                                           NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx)
               : nppiFilterMin_32f_C1R_Ctx(sc.src(), sc.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes,
                                           NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  }
  printf("status %d %s canary=%s\n", (int)st, sn(st), dst.canaryIntact() ? "intact" : "CLOBBERED");
  if (st != NPP_NO_ERROR) return;
  std::vector<T> got = dst.region();
  char nm[64];
  for (int m = 0; m < 3; ++m) {
    snprintf(nm, 64, "%s/empty=0", modeName[m]);
    report(nm, minmaxModel(sc, W, H, mw, mh, ax, ay, W, H, 0, 0, (Mode)m, isMax, T(0)), got);
    snprintf(nm, 64, "%s/empty=lo", modeName[m]);
    report(nm, minmaxModel(sc, W, H, mw, mh, ax, ay, W, H, 0, 0, (Mode)m, isMax,
                           isMax ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max()),
           got);
  }
  dst.printRegion("dst");
}

// ============================================================================
// S4: nppiFilterMaxBorder / MinBorder
// ============================================================================
static void minmaxBorderModes() {
  printf("== S4 nppiFilterMaxBorder/MinBorder_8u_C1R border modes ==\n");
  Scene<Npp8u> sc(8, 8, 0x2001);
  const int W = 7, H = 7;
  const int types[] = {0, 1, 2, 3, 4, 7};
  const char *tn[] = {"UNDEF", "CONST", "REPL", "WRAP", "MIRROR", "7"};
  for (int k = 0; k < 6; ++k) {
    Dst<Npp8u> d2(W, H);
    NppStatus st = nppiFilterMaxBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{8, 8},
                                                  NppiPoint{2, 3}, (Npp8u *)d2.d.p, d2.stepBytes,
                                                  NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                  (NppiBorderType)types[k], g_ctx);
    printf("max  border %-7s -> %4d %s\n", tn[k], (int)st, sn(st));
    Dst<Npp8u> d3(W, H);
    st = nppiFilterMinBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{8, 8}, NppiPoint{2, 3},
                                        (Npp8u *)d3.d.p, d3.stepBytes, NppiSize{W, H}, NppiSize{3, 3},
                                        NppiPoint{1, 1}, (NppiBorderType)types[k], g_ctx);
    printf("min  border %-7s -> %4d %s\n", tn[k], (int)st, sn(st));
    fflush(stdout);
  }
}

static void minmaxBorderValues() {
  printf("== S4b nppiFilterMaxBorder/MinBorder_8u_C1R replicate values ==\n");
  Scene<Npp8u> sc(8, 8, 0x2002);
  const int W = 7, H = 7, sw = 8, sh = 8, offx = 2, offy = 3;
  for (int isMax = 0; isMax < 2; ++isMax) {
    Dst<Npp8u> dst(W, H);
    NppStatus st = isMax ? nppiFilterMaxBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{sw, sh},
                                                          NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
                                                          dst.stepBytes, NppiSize{W, H}, NppiSize{3, 3},
                                                          NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx)
                         : nppiFilterMinBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{sw, sh},
                                                          NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
                                                          dst.stepBytes, NppiSize{W, H}, NppiSize{3, 3},
                                                          NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
    printf("%s status %d %s canary=%s\n", isMax ? "max" : "min", (int)st, sn(st),
           dst.canaryIntact() ? "intact" : "CLOBBERED");
    if (st != NPP_NO_ERROR) continue;
    std::vector<Npp8u> got = dst.region();
    char nm[64];
    for (int m = 0; m < 3; ++m) {
      snprintf(nm, 64, "%s", modeName[m]);
      report(nm, minmaxModel(sc, W, H, 3, 3, 1, 1, sw, sh, offx, offy, (Mode)m, isMax != 0, (Npp8u)0), got);
    }
  }
}

// ============================================================================
// S5: nppiFilterMedian + GetBufferSize
// ============================================================================
static void medianBufferSizes() {
  printf("== S5 nppiFilterMedianGetBufferSize_8u_C1R grid ==\n");
  const int ws[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 127, 128, 256, 257};
  const int hs[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 127, 128, 256, 257};
  const int ms[][2] = {{1, 1}, {2, 2}, {3, 3}, {5, 5}, {4, 7}, {8, 8}, {16, 16}, {17, 17}, {3, 5}};
  for (int k = 0; k < 9; ++k) {
    printf("mask %dx%d:", ms[k][0], ms[k][1]);
    for (int i = 0; i < 18; ++i) {
      Npp32u sz = 0;
      NppStatus st = nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{ws[i], hs[i]},
                                                              NppiSize{ms[k][0], ms[k][1]}, &sz, g_ctx);
      printf(" %dx%d=%u(%d)", ws[i], hs[i], sz, (int)st);
    }
    printf("\n");
    fflush(stdout);
  }
  printf("-- other dtypes (ROI 100x100) --\n");
  Npp32u sz = 0;
  printf("16u m3x3 %d %u\n", (int)nppiFilterMedianGetBufferSize_16u_C1R_Ctx(NppiSize{100, 100}, NppiSize{3, 3}, &sz, g_ctx), sz);
  printf("16s m3x3 %d %u\n", (int)nppiFilterMedianGetBufferSize_16s_C1R_Ctx(NppiSize{100, 100}, NppiSize{3, 3}, &sz, g_ctx), sz);
  printf("32f m3x3 %d %u\n", (int)nppiFilterMedianGetBufferSize_32f_C1R_Ctx(NppiSize{100, 100}, NppiSize{3, 3}, &sz, g_ctx), sz);
  printf("8u  m5x5 %d %u\n", (int)nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{100, 100}, NppiSize{5, 5}, &sz, g_ctx), sz);
  printf("8u  m1x1 %d %u\n", (int)nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{100, 100}, NppiSize{1, 1}, &sz, g_ctx), sz);
  printf("8u  m2x2 %d %u\n", (int)nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{100, 100}, NppiSize{2, 2}, &sz, g_ctx), sz);
  printf("8u  m4x4 %d %u\n", (int)nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{100, 100}, NppiSize{4, 4}, &sz, g_ctx), sz);
  printf("8u  m16x16 %d %u\n", (int)nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{100, 100}, NppiSize{16, 16}, &sz, g_ctx), sz);
  fflush(stdout);
}

template <class T>
static void medianValuesT(const char *tag, uint64_t seed, int mw, int mh, int ax, int ay);

static void medianValues() {
  printf("== S5b nppiFilterMedian_8u_C1R values ==\n");
  medianValuesT<Npp8u>("8u", 0x3001, 3, 3, 1, 1);
  medianValuesT<Npp8u>("8u", 0x3002, 5, 5, 2, 2);
  medianValuesT<Npp8u>("8u", 0x3003, 2, 2, 1, 1);
  medianValuesT<Npp8u>("8u", 0x3004, 3, 1, 2, 0);
}

template <class T>
static void medianValuesT(const char *tag, uint64_t seed, int mw, int mh, int ax, int ay) {
  printf("-- median %s mask %dx%d anchor %d,%d --\n", tag, mw, mh, ax, ay);
  Scene<T> sc(8, 8, seed);
  const int W = 6, H = 6;
  Dst<T> dst(W, H);
  Npp32u bsz = 0;
  NppStatus bs;
  if constexpr (std::is_same_v<T, Npp8u>)
    bs = nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{W, H}, NppiSize{mw, mh}, &bsz, g_ctx);
  else
    bs = nppiFilterMedianGetBufferSize_16s_C1R_Ctx(NppiSize{W, H}, NppiSize{mw, mh}, &bsz, g_ctx);
  printf("bufsize %u (%d)\n", bsz, (int)bs);
  DevBuf buf;
  buf.alloc(bsz ? bsz : 4);
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp8u>)
    st = nppiFilterMedian_8u_C1R_Ctx(sc.src(), sc.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes,
                                     NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay},
                                     (Npp8u *)buf.p, g_ctx);
  else
    st = nppiFilterMedian_16s_C1R_Ctx(sc.src(), sc.stepBytes, (Npp16s *)dst.d.p, dst.stepBytes,
                                      NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay},
                                      (Npp8u *)buf.p, g_ctx);
  printf("status %d %s canary=%s\n", (int)st, sn(st), dst.canaryIntact() ? "intact" : "CLOBBERED");
  if (st != NPP_NO_ERROR) return;
  std::vector<T> got = dst.region();
  char nm[64];
  for (int m = 0; m < 3; ++m)
    for (int p = 0; p < 2; ++p) {
      snprintf(nm, 64, "%s/pick=%s", modeName[m], p ? "upper" : "lower");
      report(nm, medianModel(sc, W, H, mw, mh, ax, ay, W, H, 0, 0, (Mode)m, p), got);
    }
  dst.printRegion("dst");
}

// ============================================================================
// S6: nppiFilterMedianBorder + BorderGetBufferSize
// ============================================================================
static void medianBorderModes() {
  printf("== S6 nppiFilterMedianBorder_8u_C1R border modes + bufsize ==\n");
  Scene<Npp8u> sc(8, 8, 0x4001);
  const int W = 7, H = 7;
  const int types[] = {0, 1, 2, 3, 4, 7};
  const char *tn[] = {"UNDEF", "CONST", "REPL", "WRAP", "MIRROR", "7"};
  for (int k = 0; k < 6; ++k) {
    Npp32u sz = 0;
    NppStatus bs = nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{W, H}, NppiSize{3, 3}, &sz,
                                                                  (NppiBorderType)types[k], g_ctx);
    printf("border %-7s bufsize %u (%d %s)", tn[k], sz, (int)bs, sn(bs));
    if (bs == NPP_NO_ERROR) {
      DevBuf buf;
      buf.alloc(sz ? sz : 4);
      Dst<Npp8u> d2(W, H);
      NppStatus st = nppiFilterMedianBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{8, 8},
                                                       NppiPoint{2, 3}, (Npp8u *)d2.d.p, d2.stepBytes,
                                                       NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                       (Npp8u *)buf.p, (NppiBorderType)types[k], g_ctx);
      printf(" run %d %s", (int)st, sn(st));
    }
    printf("\n");
    fflush(stdout);
  }
  printf("-- MedianBorderGetBufferSize 8u C1 grid (REPL) --\n");
  const int ws[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 127, 128, 256, 257};
  const int ms[][2] = {{1, 1}, {2, 2}, {3, 3}, {5, 5}, {4, 7}, {8, 8}, {16, 16}, {17, 17}, {3, 5}};
  for (int k = 0; k < 9; ++k) {
    printf("mask %dx%d:", ms[k][0], ms[k][1]);
    for (int i = 0; i < 18; ++i) {
      Npp32u sz = 0;
      NppStatus st = nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{ws[i], ws[i]},
                                                                    NppiSize{ms[k][0], ms[k][1]}, &sz,
                                                                    NPP_BORDER_REPLICATE, g_ctx);
      printf(" %dx%d=%u(%d)", ws[i], ws[i], sz, (int)st);
    }
    printf("\n");
    fflush(stdout);
  }
}

static void medianBorderValues() {
  printf("== S6b nppiFilterMedianBorder_8u_C1R replicate values ==\n");
  Scene<Npp8u> sc(8, 8, 0x4002);
  const int W = 7, H = 7, sw = 8, sh = 8, offx = 2, offy = 3;
  Npp32u sz = 0;
  NppStatus bs = nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{W, H}, NppiSize{3, 3}, &sz,
                                                                NPP_BORDER_REPLICATE, g_ctx);
  printf("bufsize %u (%d)\n", sz, (int)bs);
  DevBuf buf;
  buf.alloc(sz ? sz : 4);
  Dst<Npp8u> dst(W, H);
  NppStatus st = nppiFilterMedianBorder_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{sw, sh},
                                                   NppiPoint{offx, offy}, (Npp8u *)dst.d.p, dst.stepBytes,
                                                   NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                   (Npp8u *)buf.p, NPP_BORDER_REPLICATE, g_ctx);
  printf("status %d %s canary=%s\n", (int)st, sn(st), dst.canaryIntact() ? "intact" : "CLOBBERED");
  if (st != NPP_NO_ERROR) return;
  std::vector<Npp8u> got = dst.region();
  char nm[64];
  for (int m = 0; m < 3; ++m)
    for (int p = 0; p < 2; ++p) {
      snprintf(nm, 64, "%s/pick=%s", modeName[m], p ? "upper" : "lower");
      report(nm, medianModel(sc, W, H, 3, 3, 1, 1, sw, sh, offx, offy, (Mode)m, p), got);
    }
  dst.printRegion("dst");
}

// ============================================================================
// S7: nppiFilterBoxBorderAdvanced
// ============================================================================
static void advancedSizes() {
  printf("== S7 nppiFilterBoxBorderAdvancedGetDeviceBufferSize grid ==\n");
  const int ws[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 127, 128, 256, 257, 512};
  for (int ch = 1; ch <= 4; ++ch) {
    printf("ch%d:", ch);
    for (int i = 0; i < 19; ++i) {
      int sz = -1;
      NppStatus st = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{ws[i], ws[i]}, ch, &sz);
      printf(" %d=%d(%d)", ws[i], sz, (int)st);
    }
    printf("\n");
    fflush(stdout);
  }
  printf("ch64_1:");
  for (int i = 0; i < 19; ++i) {
    int sz = -1;
    NppStatus st = nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{ws[i], ws[i]}, 1, &sz);
    printf(" %d=%d(%d)", ws[i], sz, (int)st);
  }
  printf("\n");
  fflush(stdout);
}

static void advancedModes() {
  printf("== S7b nppiFilterBoxBorderAdvanced_8u_C1R border modes ==\n");
  Scene<Npp8u> sc(8, 8, 0x5001);
  const int W = 7, H = 7;
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  printf("bufsize %d\n", bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  const int types[] = {0, 1, 2, 3, 4, 7};
  const char *tn[] = {"UNDEF", "CONST", "REPL", "WRAP", "MIRROR", "7"};
  for (int k = 0; k < 6; ++k) {
    Dst<Npp8u> d2(W, H);
    NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{8, 8},
                                                          NppiPoint{2, 3}, (Npp8u *)d2.d.p, d2.stepBytes,
                                                          NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                          (NppiBorderType)types[k], (Npp8u *)buf.p, g_ctx);
    printf("border %-7s -> %4d %s\n", tn[k], (int)st, sn(st));
    fflush(stdout);
  }
}

static void advancedValues() {
  printf("== S7c nppiFilterBoxBorderAdvanced_8u_C1R values ==\n");
  Scene<Npp8u> sc(8, 8, 0x5002);
  const int W = 7, H = 7, sw = 8, sh = 8, offx = 2, offy = 3;
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  Dst<Npp8u> dst(W, H);
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{sw, sh},
                                                        NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
                                                        dst.stepBytes, NppiSize{W, H}, NppiSize{3, 3},
                                                        NppiPoint{1, 1}, NPP_BORDER_REPLICATE,
                                                        (Npp8u *)buf.p, g_ctx);
  printf("status %d %s canary=%s\n", (int)st, sn(st), dst.canaryIntact() ? "intact" : "CLOBBERED");
  if (st != NPP_NO_ERROR) return;
  std::vector<Npp8u> got = dst.region();
  char nm[64];
  for (int m = 0; m < 3; ++m)
    for (int dv = 0; dv < 2; ++dv)
      for (int r = 0; r < 4; ++r) {
        snprintf(nm, 64, "%s/%s/%s", modeName[m], divName[dv], roundName[r]);
        report(nm, boxModel(sc, W, H, 3, 3, 1, 1, sw, sh, offx, offy, (Mode)m, (DivMode)dv, (RoundMode)r), got);
      }
  dst.printRegion("dst");
  // 32f and 64f variants
  {
    Scene<Npp32f> scf(8, 8, 0x5003);
    int bszf = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bszf);
    DevBuf buff;
    buff.alloc((size_t)bszf);
    Dst<Npp32f> dstf(W, H);
    NppStatus stf = nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(
        scf.src(), scf.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp32f *)dstf.d.p,
        dstf.stepBytes, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE,
        (Npp8u *)buff.p, g_ctx);
    printf("32f status %d %s\n", (int)stf, sn(stf));
    if (stf == NPP_NO_ERROR) {
      std::vector<Npp32f> gotf = dstf.region();
      for (int m = 0; m < 3; ++m)
        for (int dv = 0; dv < 2; ++dv)
          for (int r = 4; r < 6; ++r) {
            snprintf(nm, 64, "32f %s/%s/%s", modeName[m], divName[dv], roundName[r]);
            report(nm, boxModel(scf, W, H, 3, 3, 1, 1, sw, sh, offx, offy, (Mode)m, (DivMode)dv,
                                (RoundMode)r), gotf);
          }
      dstf.printRegion("dst32f");
    }
  }
  {
    Scene<Npp64f> scd(8, 8, 0x5004);
    int bszd = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{W, H}, 1, &bszd);
    DevBuf bufd;
    bufd.alloc((size_t)bszd);
    Dst<Npp64f> dstd(W, H);
    NppStatus std2 = nppiFilterBoxBorderAdvanced_64f_C1R_Ctx(
        scd.src(), scd.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp64f *)dstd.d.p,
        dstd.stepBytes, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE,
        (Npp8u *)bufd.p, g_ctx);
    printf("64f status %d %s\n", (int)std2, sn(std2));
    if (std2 == NPP_NO_ERROR) {
      std::vector<Npp64f> gotd = dstd.region();
      for (int m = 0; m < 3; ++m)
        for (int dv = 0; dv < 2; ++dv) {
          snprintf(nm, 64, "64f %s/%s", modeName[m], divName[dv]);
          report(nm, boxModel(scd, W, H, 3, 3, 1, 1, sw, sh, offx, offy, (Mode)m, (DivMode)dv, R_DBL),
                 gotd);
        }
      dstd.printRegion("dst64f");
    }
  }
}

// ============================================================================
// S8: validation order / error statuses
// ============================================================================
template <class T>
static void valNonBorder(const char *fam, uint64_t seed) {
  printf("== S8 validation %s ==\n", fam);
  Scene<T> sc(8, 8, seed);
  const int W = 7, H = 7, es = (int)sizeof(T);
  Dst<T> dst(W, H);
  const T *ps = sc.src();
  T *pd = (T *)dst.d.p;
  int step = sc.stepBytes, dstep = dst.stepBytes;
  NppiSize roi{W, H}, m3{3, 3}, mbad{0, 3}, mneg{-3, 3};
  NppiPoint a1{1, 1}, a3{3, 1}, an{ -1, 1}, ay{1, 3};
  NppStatus (*fn)(const T *, int, T *, int, NppiSize, NppiSize, NppiPoint, NppStreamContext);
  if constexpr (std::is_same_v<T, Npp8u>) fn = nppiFilterBox_8u_C1R_Ctx;
  else if constexpr (std::is_same_v<T, Npp16u>) fn = nppiFilterBox_16u_C1R_Ctx;
  else if constexpr (std::is_same_v<T, Npp16s>) fn = nppiFilterBox_16s_C1R_Ctx;
  else if constexpr (std::is_same_v<T, Npp32f>) fn = nppiFilterBox_32f_C1R_Ctx;
  else fn = nppiFilterBox_64f_C1R_Ctx;
  CK("base", fn(ps, step, pd, dstep, roi, m3, a1, g_ctx));
  CK("null_src", fn(nullptr, step, pd, dstep, roi, m3, a1, g_ctx));
  CK("null_dst", fn(ps, step, nullptr, dstep, roi, m3, a1, g_ctx));
  CK("roi_neg_w", fn(ps, step, pd, dstep, NppiSize{-1, H}, m3, a1, g_ctx));
  CK("roi_neg_h", fn(ps, step, pd, dstep, NppiSize{W, -1}, m3, a1, g_ctx));
  CK("roi_zero_w", fn(ps, step, pd, dstep, NppiSize{0, H}, m3, a1, g_ctx));
  CK("roi_zero_h", fn(ps, step, pd, dstep, NppiSize{W, 0}, m3, a1, g_ctx));
  CK("roi_zero_both", fn(ps, step, pd, dstep, NppiSize{0, 0}, m3, a1, g_ctx));
  CK("mask_zero", fn(ps, step, pd, dstep, roi, NppiSize{0, 0}, a1, g_ctx));
  CK("mask_zero_w", fn(ps, step, pd, dstep, roi, mbad, NppiPoint{0, 1}, g_ctx));
  CK("mask_zero_h", fn(ps, step, pd, dstep, roi, NppiSize{3, 0}, NppiPoint{1, 0}, g_ctx));
  CK("mask_neg", fn(ps, step, pd, dstep, roi, mneg, a1, g_ctx));
  CK("anchor_x_gt", fn(ps, step, pd, dstep, roi, m3, a3, g_ctx));
  CK("anchor_x_neg", fn(ps, step, pd, dstep, roi, m3, an, g_ctx));
  CK("anchor_y_gt", fn(ps, step, pd, dstep, roi, m3, ay, g_ctx));
  CK("dststep_small", fn(ps, step, pd, (W - 1) * es, roi, m3, a1, g_ctx));
  CK("dststep_zero", fn(ps, step, pd, 0, roi, m3, a1, g_ctx));
  CK("dststep_neg", fn(ps, step, pd, -dstep, roi, m3, a1, g_ctx));
  CK("srcstep_zero", fn(ps, 0, pd, dstep, roi, m3, a1, g_ctx));
  CK("srcstep_neg", fn(ps, -step, pd, dstep, roi, m3, a1, g_ctx));
  CK("prec_null_vs_negroi", fn(nullptr, step, pd, dstep, NppiSize{-1, H}, m3, a1, g_ctx));
  CK("prec_negroi_vs_mask", fn(ps, step, pd, dstep, NppiSize{-1, H}, mbad, NppiPoint{0, 1}, g_ctx));
  CK("prec_zeroroi_vs_mask", fn(ps, step, pd, dstep, NppiSize{0, H}, mbad, NppiPoint{0, 1}, g_ctx));
  CK("prec_mask_vs_anchor", fn(ps, step, pd, dstep, roi, mbad, a3, g_ctx));
  CK("prec_mask_vs_step", fn(ps, step, pd, (W - 1) * es, roi, mbad, NppiPoint{0, 1}, g_ctx));
  CK("prec_anchor_vs_step", fn(ps, step, pd, (W - 1) * es, roi, m3, a3, g_ctx));
  CK("prec_negroi_vs_step", fn(ps, step, pd, (W - 1) * es, NppiSize{-1, H}, m3, a1, g_ctx));
  CK("prec_zeroroi_vs_step", fn(ps, step, pd, (W - 1) * es, NppiSize{0, H}, m3, a1, g_ctx));
}

static void valMinMaxNonBorder() {
  printf("== S8 validation FilterMax/Min_8u_C1R ==\n");
  Scene<Npp8u> sc(8, 8, 0x6001);
  const int W = 7, H = 7;
  Dst<Npp8u> dst(W, H);
  const Npp8u *ps = sc.src();
  Npp8u *pd = (Npp8u *)dst.d.p;
  int dstep = dst.stepBytes;
  NppiSize roi{W, H}, m3{3, 3}, mbad{0, 3};
  NppiPoint a1{1, 1}, a3{3, 1};
  CK("max base", nppiFilterMax_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a1, g_ctx));
  CK("max null_src", nppiFilterMax_8u_C1R_Ctx(nullptr, sc.stepBytes, pd, dstep, roi, m3, a1, g_ctx));
  CK("max roi_neg", nppiFilterMax_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{-1, H}, m3, a1, g_ctx));
  CK("max roi_zero", nppiFilterMax_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{0, H}, m3, a1, g_ctx));
  CK("max mask_zero", nppiFilterMax_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, NppiSize{0, 0}, a1, g_ctx));
  CK("max anchor_x_gt", nppiFilterMax_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a3, g_ctx));
  CK("max dststep_small", nppiFilterMax_8u_C1R_Ctx(ps, sc.stepBytes, pd, W - 1, roi, m3, a1, g_ctx));
  CK("min base", nppiFilterMin_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a1, g_ctx));
  CK("min mask_zero", nppiFilterMin_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, NppiSize{0, 0}, a1, g_ctx));
  CK("min anchor_x_gt", nppiFilterMin_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a3, g_ctx));
}

template <class T>
static void valBorder(const char *fam, uint64_t seed, int which /*0 box,1 max,2 min,3 adv,4 median*/) {
  printf("== S8 validation %s ==\n", fam);
  Scene<T> sc(8, 8, seed);
  const int W = 7, H = 7, es = (int)sizeof(T);
  Dst<T> dst(W, H);
  const T *ps = sc.src();
  T *pd = (T *)dst.d.p;
  int dstep = dst.stepBytes;
  NppiSize roi{W, H}, sz{8, 8}, m3{3, 3}, mbad{0, 3};
  NppiPoint a1{1, 1}, a3{3, 1};
  NppiPoint off{2, 3}, offb{8, 3}, offn{-1, 3};
  Npp32u bsz = 0;
  DevBuf buf;
  if (which == 4) {
    nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(roi, m3, &bsz, NPP_BORDER_REPLICATE, g_ctx);
  } else {
    int b = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(roi, 1, &b);
    bsz = (Npp32u)b;
  }
  buf.alloc(bsz ? bsz : 16);
  Npp8u *pb = (Npp8u *)buf.p;

  auto boxb = [&](const T *s, int ss, NppiSize z, NppiPoint o, T *d, int ds, NppiSize r, NppiSize m,
                  NppiPoint a, NppiBorderType b, Npp8u *p) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if (which == 0) return nppiFilterBoxBorder_8u_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
      if (which == 1) return nppiFilterMaxBorder_8u_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
      if (which == 2) return nppiFilterMinBorder_8u_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
      if (which == 3) return nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, p, g_ctx);
      return nppiFilterMedianBorder_8u_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, p, b, g_ctx);
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if (which == 0) return nppiFilterBoxBorder_16s_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
      if (which == 1) return nppiFilterMaxBorder_16s_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
      if (which == 2) return nppiFilterMinBorder_16s_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if (which == 0) return nppiFilterBoxBorder_32f_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
      if (which == 1) return nppiFilterMaxBorder_32f_C1R_Ctx(s, ss, z, o, d, ds, r, m, a, b, g_ctx);
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  };
  CK("base", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("null_src", boxb(nullptr, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("null_dst", boxb(ps, sc.stepBytes, sz, off, nullptr, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("roi_neg_w", boxb(ps, sc.stepBytes, sz, off, pd, dstep, NppiSize{-1, H}, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("roi_zero_w", boxb(ps, sc.stepBytes, sz, off, pd, dstep, NppiSize{0, H}, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("roi_zero_both", boxb(ps, sc.stepBytes, sz, off, pd, dstep, NppiSize{0, 0}, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("border_const", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_CONSTANT, pb));
  CK("border_wrap", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_WRAP, pb));
  CK("border_mirror", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_MIRROR, pb));
  CK("border_undef", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_UNDEFINED, pb));
  CK("srcsize_zero_w", boxb(ps, sc.stepBytes, NppiSize{0, 8}, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("srcsize_zero_h", boxb(ps, sc.stepBytes, NppiSize{8, 0}, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("srcsize_neg", boxb(ps, sc.stepBytes, NppiSize{-1, 8}, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("offset_x_oob", boxb(ps, sc.stepBytes, sz, offb, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("offset_neg", boxb(ps, sc.stepBytes, sz, offn, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("mask_zero", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, pb));
  CK("anchor_x_gt", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a3, NPP_BORDER_REPLICATE, pb));
  CK("dststep_small", boxb(ps, sc.stepBytes, sz, off, pd, (W - 1) * es, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("dststep_zero", boxb(ps, sc.stepBytes, sz, off, pd, 0, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("dststep_neg", boxb(ps, sc.stepBytes, sz, off, pd, -dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("prec_negroi_vs_border", boxb(ps, sc.stepBytes, sz, off, pd, dstep, NppiSize{-1, H}, m3, a1, NPP_BORDER_CONSTANT, pb));
  CK("prec_zeroroi_vs_border", boxb(ps, sc.stepBytes, sz, off, pd, dstep, NppiSize{0, H}, m3, a1, NPP_BORDER_CONSTANT, pb));
  CK("prec_border_vs_srcsize", boxb(ps, sc.stepBytes, NppiSize{0, 8}, off, pd, dstep, roi, m3, a1, NPP_BORDER_CONSTANT, pb));
  CK("prec_border_vs_offset", boxb(ps, sc.stepBytes, sz, offb, pd, dstep, roi, m3, a1, NPP_BORDER_CONSTANT, pb));
  CK("prec_srcsize_vs_border", boxb(ps, sc.stepBytes, NppiSize{0, 8}, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("prec_offset_vs_step", boxb(ps, sc.stepBytes, sz, offb, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("prec_mask_vs_border", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, NppiSize{0, 0}, a1, NPP_BORDER_CONSTANT, pb));
  CK("prec_mask_vs_step", boxb(ps, sc.stepBytes, sz, off, pd, (W - 1) * es, roi, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, pb));
  CK("prec_border_vs_step", boxb(ps, sc.stepBytes, sz, off, pd, (W - 1) * es, roi, m3, a1, NPP_BORDER_CONSTANT, pb));
  CK("prec_zeroroi_vs_step", boxb(ps, sc.stepBytes, sz, off, pd, (W - 1) * es, NppiSize{0, H}, m3, a1, NPP_BORDER_REPLICATE, pb));
  CK("prec_negroi_vs_null", boxb(nullptr, sc.stepBytes, sz, off, pd, dstep, NppiSize{-1, H}, m3, a1, NPP_BORDER_REPLICATE, pb));
  if (which == 3) CK("adv null_buf", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, nullptr));
  if (which == 4) CK("med null_buf", boxb(ps, sc.stepBytes, sz, off, pd, dstep, roi, m3, a1, NPP_BORDER_REPLICATE, nullptr));
}

static void valMedianNonBorder() {
  printf("== S8 validation FilterMedian_8u_C1R ==\n");
  Scene<Npp8u> sc(8, 8, 0x7001);
  const int W = 7, H = 7;
  Dst<Npp8u> dst(W, H);
  const Npp8u *ps = sc.src();
  Npp8u *pd = (Npp8u *)dst.d.p;
  int dstep = dst.stepBytes;
  NppiSize roi{W, H}, m3{3, 3};
  NppiPoint a1{1, 1}, a3{3, 1};
  Npp32u bsz = 0;
  nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, m3, &bsz, g_ctx);
  DevBuf buf;
  buf.alloc(bsz);
  Npp8u *pb = (Npp8u *)buf.p;
  CK("base", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a1, pb, g_ctx));
  CK("null_src", nppiFilterMedian_8u_C1R_Ctx(nullptr, sc.stepBytes, pd, dstep, roi, m3, a1, pb, g_ctx));
  CK("null_dst", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, nullptr, dstep, roi, m3, a1, pb, g_ctx));
  CK("null_buf", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a1, nullptr, g_ctx));
  CK("roi_neg", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{-1, H}, m3, a1, pb, g_ctx));
  CK("roi_zero", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{0, H}, m3, a1, pb, g_ctx));
  CK("mask_zero", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, NppiSize{0, 0}, a1, pb, g_ctx));
  CK("mask_17", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, NppiSize{17, 17}, NppiPoint{8, 8}, pb, g_ctx));
  CK("anchor_x_gt", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, m3, a3, pb, g_ctx));
  CK("dststep_small", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, W - 1, roi, m3, a1, pb, g_ctx));
    CK("prec_null_vs_negroi", nppiFilterMedian_8u_C1R_Ctx(nullptr, sc.stepBytes, pd, dstep, NppiSize{-1, H}, m3, a1, pb, g_ctx));
  CK("prec_negroi_vs_buf", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{-1, H}, m3, a1, nullptr, g_ctx));
  CK("prec_zeroroi_vs_buf", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{0, H}, m3, a1, nullptr, g_ctx));
  CK("prec_zeroroi_vs_mask", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, NppiSize{0, H}, NppiSize{0, 0}, a1, pb, g_ctx));
  CK("prec_mask_vs_buf", nppiFilterMedian_8u_C1R_Ctx(ps, sc.stepBytes, pd, dstep, roi, NppiSize{0, 0}, a1, nullptr, g_ctx));
  printf("-- GetBufferSize validation --\n");
  Npp32u sz = 0;
  CK("bs null_size", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, m3, nullptr, g_ctx));
  CK("bs roi_neg", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{-1, H}, m3, &sz, g_ctx));
  CK("bs roi_zero", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{0, H}, m3, &sz, g_ctx));
  CK("bs mask_zero", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, NppiSize{0, 0}, &sz, g_ctx));
  CK("bs mask_neg", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, NppiSize{-3, 3}, &sz, g_ctx));
  CK("bs mask_17", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, NppiSize{17, 17}, &sz, g_ctx));
  CK("bs mask_33", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, NppiSize{33, 33}, &sz, g_ctx));
  CK("bs mask_area_1025", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, NppiSize{1025, 1}, &sz, g_ctx));
  CK("bs mask_area_1024", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, NppiSize{1024, 1}, &sz, g_ctx));
  CK("bs border null_size", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(roi, m3, nullptr, NPP_BORDER_REPLICATE, g_ctx));
  CK("bs border bad_border", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(roi, m3, &sz, NPP_BORDER_CONSTANT, g_ctx));
  CK("bs border roi_neg", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{-1, H}, m3, &sz, NPP_BORDER_REPLICATE, g_ctx));
  CK("bs border roi_zero", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{0, H}, m3, &sz, NPP_BORDER_REPLICATE, g_ctx));
  CK("bs border mask_zero", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(roi, NppiSize{0, 0}, &sz, NPP_BORDER_REPLICATE, g_ctx));
}

// ============================================================================
// S9: channels: C4 computation vs AC4 alpha preservation
// ============================================================================
template <class T>
static void chanProbe(const char *tag, uint64_t seed) {
  printf("== S9 %s channels ==\n", tag);
  // 4-channel scene: allocation AW x AH of 4-channel pixels
  std::vector<T> h(AW * AH * 4);
  for (int j = 0; j < AH; ++j)
    for (int i = 0; i < AW; ++i)
      for (int c = 0; c < 4; ++c) {
        T v = pat<T>(i * 4 + c, j, seed);
        if (c == 3) v = (T)0xEE;  // distinctive alpha
        h[(j * AW + i) * 4 + c] = v;
      }
  DevBuf src;
  src.put(h);
  const int W = 6, H = 6, ox = 8, oy = 8;
  int step = AW * 4 * (int)sizeof(T);
  const T *ps = (const T *)src.p + (size_t)oy * AW * 4 + ox * 4;
  // dst
  const int fw = W + 4, fh = H + 4;
  std::vector<T> hd(fw * fh * 4, T(0xAB));
  DevBuf dst;
  dst.put(hd);
  int dstep = fw * 4 * (int)sizeof(T);
  NppStatus st = NPP_NO_ERROR;
  if constexpr (std::is_same_v<T, Npp8u>) {
    st = nppiFilterBox_8u_C4R_Ctx(ps, step, (Npp8u *)dst.p, dstep, NppiSize{W, H}, NppiSize{3, 3},
                                  NppiPoint{1, 1}, g_ctx);
    printf("C4 box %d %s\n", (int)st, sn(st));
    st = nppiFilterBox_8u_AC4R_Ctx(ps, step, (Npp8u *)dst.p, dstep, NppiSize{W, H}, NppiSize{3, 3},
                                   NppiPoint{1, 1}, g_ctx);
    printf("AC4 box %d %s\n", (int)st, sn(st));
  }
  std::vector<T> got = dst.get<T>(fw * fh * 4);
  // print alpha column of first row region and a computed channel sample
  printf("dst[0][*] c0..c3 of first 3 px:");
  for (int x = 0; x < 3; ++x)
    for (int c = 0; c < 4; ++c) printf(" %d", (int)got[(x)*4 + c]);
  printf("\n");
  fflush(stdout);
  // C4/AC4 border variants + median + advanced
  if constexpr (std::is_same_v<T, Npp8u>) {
    DevBuf dst2;
    dst2.put(hd);
    st = nppiFilterBoxBorder_8u_AC4R_Ctx(ps, step, NppiSize{8, 8}, NppiPoint{2, 3}, (Npp8u *)dst2.p,
                                         dstep, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                         NPP_BORDER_REPLICATE, g_ctx);
    printf("AC4 boxborder %d %s\n", (int)st, sn(st));
    std::vector<Npp8u> g2 = dst2.get<Npp8u>(fw * fh * 4);
    printf("border dst[0][*]:");
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 4; ++c) printf(" %d", (int)g2[(x)*4 + c]);
    printf("\n");
    Npp32u bsz = 0;
    nppiFilterMedianGetBufferSize_8u_AC4R_Ctx(NppiSize{W, H}, NppiSize{3, 3}, &bsz, g_ctx);
    printf("median AC4 bufsize %u\n", bsz);
    DevBuf buf;
    buf.alloc(bsz);
    DevBuf dst3;
    dst3.put(hd);
    st = nppiFilterMedian_8u_AC4R_Ctx(ps, step, (Npp8u *)dst3.p, dstep, NppiSize{W, H}, NppiSize{3, 3},
                                       NppiPoint{1, 1}, (Npp8u *)buf.p, g_ctx);
    printf("AC4 median %d %s\n", (int)st, sn(st));
    std::vector<Npp8u> g3 = dst3.get<Npp8u>(fw * fh * 4);
    printf("median dst[0][*]:");
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 4; ++c) printf(" %d", (int)g3[(x)*4 + c]);
    printf("\n");
    st = nppiFilterMaxBorder_8u_AC4R_Ctx(ps, step, NppiSize{8, 8}, NppiPoint{2, 3}, (Npp8u *)dst3.p,
                                         dstep, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                         NPP_BORDER_REPLICATE, g_ctx);
    printf("AC4 maxborder %d %s\n", (int)st, sn(st));
    std::vector<Npp8u> g4 = dst3.get<Npp8u>(fw * fh * 4);
    printf("maxborder dst[0][*]:");
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 4; ++c) printf(" %d", (int)g4[(x)*4 + c]);
    printf("\n");
    fflush(stdout);
  }
}

// ============================================================================
// S10: advanced box minimal buffer size probing (run LAST: may corrupt)
// ============================================================================
static void advancedBufferProbe() {
  printf("== S10 advanced box minimal buffer ==\n");
  Scene<Npp8u> sc(8, 8, 0x8001);
  const int W = 7, H = 7;
  Dst<Npp8u> ref(W, H);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf big;
  big.alloc((size_t)bsz);
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(sc.src(), sc.stepBytes, NppiSize{8, 8},
                                                        NppiPoint{2, 3}, (Npp8u *)ref.d.p, ref.stepBytes,
                                                        NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1},
                                                        NPP_BORDER_REPLICATE, (Npp8u *)big.p, g_ctx);
  printf("ref run %d %s bufsize %d\n", (int)st, sn(st), bsz);
  std::vector<Npp8u> refv = ref.region();
  const size_t tries[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576};
  for (size_t k = 0; k < sizeof(tries) / sizeof(tries[0]); ++k) {
    size_t n = tries[k];
    if (n > (size_t)bsz) break;
    // canary: allocate n + 1024, fill canary region, run with pBuf
    DevBuf buf;
    buf.alloc(n + 1024);
    std::vector<Npp8u> can(n + 1024, 0xAB);
    cudaMemcpy(buf.p, can.data(), can.size(), cudaMemcpyHostToDevice);
    Dst<Npp8u> d(W, H);
    cudaError_t ce0 = cudaGetLastError();
    NppStatus s2 = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
        sc.src(), sc.stepBytes, NppiSize{8, 8}, NppiPoint{2, 3}, (Npp8u *)d.d.p, d.stepBytes,
        NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx);
    std::vector<Npp8u> after = buf.get<Npp8u>(n + 1024);
    size_t clobbered = 0, firstClob = 0;
    for (size_t i = 0; i < after.size(); ++i)
      if (after[i] != 0xAB) {
        if (!clobbered) firstClob = i;
        ++clobbered;
      }
    bool same = d.region() == refv;
    printf("buf %8zu -> %4d %s dst_same=%d clobbered=%zu first@%zu (cuda %d)\n", n, (int)s2, sn(s2),
           (int)same, clobbered, firstClob, (int)ce0);
    fflush(stdout);
  }
  // does it write beyond the dst ROI?
  printf("dst canary after ref run: %s\n", ref.canaryIntact() ? "intact" : "CLOBBERED");
}

int main(int argc, char **argv) {
  cudaFree(0);
  init_ctx();
  const char *sec = argc > 1 ? argv[1] : "all";
  std::string s = sec;
  if (s == "all" || s == "box") {
    boxValues<Npp8u>("8u", 0x11, 3, 3, 1, 1);
    boxValues<Npp8u>("8u", 0x12, 2, 2, 1, 1);
    boxValues<Npp8u>("8u", 0x13, 3, 1, 2, 0);
    boxValues<Npp16u>("16u", 0x14, 3, 3, 1, 1);
    boxValues<Npp16s>("16s", 0x15, 3, 3, 1, 1);
    boxValues<Npp16s>("16s", 0x16, 2, 1, 1, 0);
    boxValues<Npp32f>("32f", 0x17, 3, 3, 1, 1);
    boxValues<Npp64f>("64f", 0x18, 3, 3, 1, 1);
    boxValues<Npp8u>("8u", 0x19, 3, 3, 3, 3);  // anchor==maskDim, zero-valid-sample pixels
    boxValues<Npp8u>("8u", 0x1A, 3, 3, 0, 0);  // anchor 0,0
  }
  if (s == "all" || s == "boxborder") {
    boxBorderModes();
    boxBorderValues();
  }
  if (s == "all" || s == "minmax") {
    minmaxValues<Npp8u>("8u", 0x21, true, 3, 3, 1, 1, (Npp8u)0);
    minmaxValues<Npp8u>("8u", 0x22, false, 3, 3, 1, 1, (Npp8u)255);
    minmaxValues<Npp8u>("8u", 0x23, true, 3, 1, 3, 0, (Npp8u)0);
    minmaxValues<Npp8u>("8u", 0x24, false, 3, 1, 3, 0, (Npp8u)255);
    minmaxValues<Npp32f>("32f", 0x25, true, 3, 3, 1, 1, -3.4e38f);
    minmaxValues<Npp32f>("32f", 0x26, false, 3, 3, 1, 1, 3.4e38f);
  }
  if (s == "all" || s == "minmaxborder") {
    minmaxBorderModes();
    minmaxBorderValues();
  }
  if (s == "all" || s == "median") {
    medianBufferSizes();
    medianValues();
  }
  if (s == "all" || s == "medianborder") {
    medianBorderModes();
    medianBorderValues();
  }
  if (s == "all" || s == "advanced") {
    advancedSizes();
    advancedModes();
    advancedValues();
  }
  if (s == "all" || s == "val") {
    valNonBorder<Npp8u>("FilterBox_8u_C1R", 0x91);
    valNonBorder<Npp16s>("FilterBox_16s_C1R", 0x92);
    valNonBorder<Npp32f>("FilterBox_32f_C1R", 0x93);
    valNonBorder<Npp64f>("FilterBox_64f_C1R", 0x94);
    valMinMaxNonBorder();
    valBorder<Npp8u>("FilterBoxBorder_8u_C1R", 0x95, 0);
    valBorder<Npp16s>("FilterBoxBorder_16s_C1R", 0x96, 0);
    valBorder<Npp8u>("FilterMaxBorder_8u_C1R", 0x97, 1);
    valBorder<Npp8u>("FilterMinBorder_8u_C1R", 0x98, 2);
    valBorder<Npp8u>("FilterBoxBorderAdvanced_8u_C1R", 0x99, 3);
    valBorder<Npp8u>("FilterMedianBorder_8u_C1R", 0x9A, 4);
    valMedianNonBorder();
  }
  if (s == "all" || s == "chan") {
    chanProbe<Npp8u>("box/median 8u", 0xA1);
  }
  if (s == "all" || s == "bufprobe") {
    advancedBufferProbe();
  }
  printf("== probe done ==\n");
  return 0;
}
