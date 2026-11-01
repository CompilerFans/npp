// f9e: remaining unknowns for the f9 window-filter package.
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppSetStream(cudaStream_t hStream);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
  case NPP_NO_ERROR: return "OK";
  case NPP_NULL_POINTER_ERROR: return "NULL";
  case NPP_SIZE_ERROR: return "SIZE";
  case NPP_STEP_ERROR: return "STEP";
  case NPP_NOT_EVEN_STEP_ERROR: return "NOT_EVEN_STEP";
  case NPP_MASK_SIZE_ERROR: return "MASK_SIZE";
  case NPP_ANCHOR_ERROR: return "ANCHOR";
  case NPP_OUT_OFF_RANGE_ERROR: return "OUT_OFF_RANGE";
  case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOT_SUPPORTED_MODE";
  default: return "other";
  }
}

struct Dev {
  void *p = nullptr;
  size_t bytes = 0;
  ~Dev() { if (p) cudaFree(p); }
  void alloc(size_t n) { bytes = n; cudaMalloc(&p, n); cudaMemset(p, 0, n); }
};

template <class T>
struct Img {
  Dev d;
  int W, H, margin, step;
  T *src;
  Img(int w, int h, int m = 8) : W(w), H(h), margin(m) {
    step = (W + 2 * margin) * (int)sizeof(T);
    d.alloc((size_t)step * (H + 2 * margin));
    src = (T *)((char *)d.p + (size_t)margin * step + margin * sizeof(T));
  }
  void set(int i, int j, T v) {
    cudaMemcpy(src + (size_t)j * (step / (int)sizeof(T)) + i, &v, sizeof(T), cudaMemcpyHostToDevice);
  }
  void fill_host(const std::vector<T> &h) {
    cudaMemcpy2D(src, step, h.data(), (size_t)W * sizeof(T), (size_t)W * sizeof(T), H, cudaMemcpyHostToDevice);
  }
};

template <class T>
struct Dst {
  Dev d;
  int W, H, step;
  T *p;
  Dst(int w, int h) : W(w), H(h) {
    step = W * (int)sizeof(T);
    d.alloc((size_t)step * H);
    p = (T *)d.p;
  }
  std::vector<T> get() {
    std::vector<T> o((size_t)W * H);
    cudaMemcpy2D(o.data(), (size_t)W * sizeof(T), p, step, (size_t)W * sizeof(T), H, cudaMemcpyDeviceToHost);
    return o;
  }
};

typedef NppStatus (*PlainBufFn)(NppiSize, NppiSize, Npp32u *, NppStreamContext);
typedef NppStatus (*BorderBufFn)(NppiSize, NppiSize, Npp32u *, NppiBorderType, NppStreamContext);
#define BFN(dt, lay) nppiFilterMedianGetBufferSize_##dt##_##lay##R_Ctx
#define BBFN(dt, lay) nppiFilterMedianBorderGetBufferSize_##dt##_##lay##R_Ctx
static const PlainBufFn kPlain[16] = {
    BFN(8u, C1), BFN(8u, C3), BFN(8u, C4), BFN(8u, AC4),
    BFN(16u, C1), BFN(16u, C3), BFN(16u, C4), BFN(16u, AC4),
    BFN(16s, C1), BFN(16s, C3), BFN(16s, C4), BFN(16s, AC4),
    BFN(32f, C1), BFN(32f, C3), BFN(32f, C4), BFN(32f, AC4)};
static const BorderBufFn kBorder[16] = {
    BBFN(8u, C1), BBFN(8u, C3), BBFN(8u, C4), BBFN(8u, AC4),
    BBFN(16u, C1), BBFN(16u, C3), BBFN(16u, C4), BBFN(16u, AC4),
    BBFN(16s, C1), BBFN(16s, C3), BBFN(16s, C4), BBFN(16s, AC4),
    BBFN(32f, C1), BBFN(32f, C3), BBFN(32f, C4), BBFN(32f, AC4)};
static const char *kName[16] = {"8uC1", "8uC3", "8uC4", "8uAC4", "16uC1", "16uC3",
                                "16uC4", "16uAC4", "16sC1", "16sC3", "16sC4",
                                "16sAC4", "32fC1", "32fC3", "32fC4", "32fAC4"};

static void grid() {
  for (int k = 0; k < 16; ++k) {
    printf("P %s\n", kName[k]);
    for (int h = 1; h <= 40; ++h) {
      char row[48];
      for (int w = 1; w <= 40; ++w) {
        Npp32u sz = 0xDEADBEEF;
        NppStatus st = kPlain[k](NppiSize{w, h}, NppiSize{w, h}, &sz, g_ctx);
        row[w - 1] = (st == NPP_NO_ERROR && sz > 0) ? '#' : (st == NPP_NO_ERROR ? '.' : 'E');
      }
      row[40] = 0;
      printf("h %2d %s\n", h, row);
    }
  }
  for (int k = 0; k < 16; ++k) {
    printf("B %s\n", kName[k]);
    for (int h = 1; h <= 40; ++h) {
      char row[48];
      for (int w = 1; w <= 40; ++w) {
        Npp32u sz = 0xDEADBEEF;
        NppStatus st = kBorder[k](NppiSize{w, h}, NppiSize{w, h}, &sz, NPP_BORDER_REPLICATE, g_ctx);
        row[w - 1] = (st == NPP_NO_ERROR && sz > 0) ? '#' : (st == NPP_NO_ERROR ? '.' : 'E');
      }
      row[40] = 0;
      printf("h %2d %s\n", h, row);
    }
  }
}


static void grid_roi(int rw, int rh) {
  for (int k = 0; k < 16; ++k) {
    printf("P %s roi=%dx%d\n", kName[k], rw, rh);
    for (int h = 1; h <= 40; ++h) {
      char row[48];
      for (int w = 1; w <= 40; ++w) {
        Npp32u sz = 0xDEADBEEF;
        NppStatus st = kPlain[k](NppiSize{rw, rh}, NppiSize{w, h}, &sz, g_ctx);
        row[w - 1] = (st == NPP_NO_ERROR && sz > 0) ? '#' : (st == NPP_NO_ERROR ? '.' : 'E');
      }
      row[40] = 0;
      printf("h %2d %s\n", h, row);
    }
  }
}


static void roi_vary() {
  int masks[][2] = {{38,5},{25,6},{16,16},{9,9},{11,11},{8,8},{4,4}};
  for (auto &m : masks) {
    printf("P 8uC1 mask=%dx%d\n", m[0], m[1]);
    for (int rh = 1; rh <= 40; ++rh) {
      char row[48];
      for (int rw = 1; rw <= 40; ++rw) {
        Npp32u sz = 0xDEADBEEF;
        NppStatus st = kPlain[0](NppiSize{rw, rh}, NppiSize{m[0], m[1]}, &sz, g_ctx);
        row[rw - 1] = (st == NPP_NO_ERROR && sz > 0) ? '#' : (st == NPP_NO_ERROR ? '.' : 'E');
      }
      row[40] = 0;
      printf("h %2d %s\n", rh, row);
    }
  }
}

static void edge() {
  struct Case { const char *n; NppiSize roi; NppiSize m; };
  Case cs[] = {
      {"roi0", {0, 5}, {3, 3}},        {"roi_neg", {-2, 5}, {3, 3}},
      {"mask0", {5, 5}, {0, 0}},       {"mask_w0", {5, 5}, {0, 3}},
      {"mask_neg", {5, 5}, {-3, 3}},   {"mask_neg_h", {5, 5}, {3, -3}},
      {"mask_gt_roi", {4, 4}, {9, 2}}, {"mask_gt_roi_h", {4, 4}, {2, 9}},
      {"big", {100, 100}, {7, 5}},     {"roi1_mask1", {1, 1}, {1, 1}},
  };
  for (auto &c : cs) {
    Npp32u s1 = 0xDEADBEEF;
    NppStatus st = kPlain[0](c.roi, c.m, &s1, g_ctx);
    Npp32u s2 = 0xDEADBEEF;
    NppStatus st2 = kBorder[0](c.roi, c.m, &s2, NPP_BORDER_REPLICATE, g_ctx);
    printf("%-14s plain=%4d %-18s val=%u  border=%4d %-18s val=%u\n", c.n, (int)st, sn(st),
           s1, (int)st2, sn(st2), s2);
  }
  NppiBorderType bs[] = {NPP_BORDER_UNDEFINED, NPP_BORDER_CONSTANT, NPP_BORDER_REPLICATE,
                        NPP_BORDER_WRAP, NPP_BORDER_MIRROR, (NppiBorderType)5,
                        (NppiBorderType)7, (NppiBorderType)-1};
  for (auto b : bs) {
    Npp32u s = 0xDEADBEEF;
    NppStatus st = kBorder[0](NppiSize{7, 7}, NppiSize{12, 12}, &s, b, g_ctx);
    printf("border type %2d -> %4d %-18s val=%u\n", (int)b, (int)st, sn(st), s);
  }
  for (int k = 0; k < 16; ++k) {
    Npp32u s = 0;
    NppStatus st = kPlain[k](NppiSize{7, 7}, NppiSize{12, 12}, &s, g_ctx);
    Npp32u b = 0;
    NppStatus stb = kBorder[k](NppiSize{7, 7}, NppiSize{12, 12}, &b, NPP_BORDER_REPLICATE, g_ctx);
    printf("scale %-7s 12x12 plain=%u(%d) border=%u(%d)\n", kName[k], s, (int)st, b, (int)stb);
  }
}

enum Mode { M_RAW, M_REPL, M_MIRROR, M_MIRROR_EDGE, M_ZERO };

template <class T>
static T raw_get(const Img<T> &im, int i, int j) {
  T v;
  cudaMemcpy(&v, im.src + (size_t)j * (im.step / (int)sizeof(T)) + i, sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}
static int reflect_noedge(int v, int n) {
  if (n <= 1) return 0;
  int p = 2 * (n - 1);
  v %= p;
  if (v < 0) v += p;
  return v < n ? v : p - v;
}
static int reflect_edge(int v, int n) {
  if (n <= 1) return 0;
  int p = 2 * n;
  v %= p;
  if (v < 0) v += p;
  return v < n ? v : p - 1 - v;
}

template <class T, class Op>
static std::vector<T> model(const Img<T> &im, int roiW, int roiH, int mw, int mh, int ax, int ay,
                            int offx, int offy, int sw, int sh, Mode m, Op op) {
  std::vector<T> out((size_t)roiW * roiH);
  for (int y = 0; y < roiH; ++y)
    for (int x = 0; x < roiW; ++x) {
      std::vector<T> s;
      s.reserve((size_t)mw * mh);
      for (int j = 0; j < mh; ++j)
        for (int i = 0; i < mw; ++i) {
          int sx = x + i - ax, sy = y + j - ay;
          if (m == M_RAW) { s.push_back(raw_get(im, sx, sy)); continue; }
          int ix = sx + offx, iy = sy + offy;
          if (m == M_REPL) {
            ix = std::clamp(ix, 0, sw - 1);
            iy = std::clamp(iy, 0, sh - 1);
          } else if (m == M_MIRROR) {
            ix = reflect_noedge(ix, sw);
            iy = reflect_noedge(iy, sh);
          } else if (m == M_MIRROR_EDGE) {
            ix = reflect_edge(ix, sw);
            iy = reflect_edge(iy, sh);
          } else if (ix < 0 || ix >= sw || iy < 0 || iy >= sh) {
            continue;
          }
          s.push_back(raw_get(im, ix - offx, iy - offy));
        }
      out[(size_t)y * roiW + x] = op(s);
    }
  return out;
}

template <class T> static T opp_max(const std::vector<T> &v) { T r = v[0]; for (T x : v) r = std::max(r, x); return r; }
template <class T> static T opp_min(const std::vector<T> &v) { T r = v[0]; for (T x : v) r = std::min(r, x); return r; }

template <class T>
static void minmax_values(const char *tag, bool ismax, int mw, int mh, int ax, int ay) {
  const int W = 6, H = 6;
  Img<T> im(W, H);
  std::vector<T> h((size_t)W * H);
  for (int j = 0; j < H; ++j)
    for (int i = 0; i < W; ++i) h[j * W + i] = (T)(3 + ((i * 7 + j * 13) % 29));
  im.fill_host(h);
  Dst<T> dst(W, H);
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp8u>) {
    st = ismax ? nppiFilterMax_8u_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx)
               : nppiFilterMin_8u_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    st = ismax ? nppiFilterMax_16u_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx)
               : nppiFilterMin_16u_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    st = ismax ? nppiFilterMax_16s_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx)
               : nppiFilterMin_16s_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  } else {
    st = ismax ? nppiFilterMax_32f_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx)
               : nppiFilterMin_32f_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, g_ctx);
  }
  std::vector<T> got = dst.get();
  auto ref = ismax ? model(im, W, H, mw, mh, ax, ay, 0, 0, W, H, M_RAW, opp_max<T>)
                   : model(im, W, H, mw, mh, ax, ay, 0, 0, W, H, M_RAW, opp_min<T>);
  int bad = 0;
  for (size_t i = 0; i < got.size(); ++i) if (got[i] != ref[i]) ++bad;
  printf("%-4s %-3s m%dx%d a%d,%d: %d %s mism=%d", tag, ismax ? "max" : "min", mw, mh, ax, ay,
         (int)st, sn(st), bad);
  if (bad) printf(" got0=%g ref0=%g", (double)got[0], (double)ref[0]);
  printf("\n");
}

template <class T>
static void minmax_border_values(const char *tag, bool ismax) {
  const int W = 6, H = 6, SW = 9, SH = 8, offx = 2, offy = 1;
  Img<T> im(SW, SH);
  std::vector<T> h((size_t)SW * SH);
  for (int j = 0; j < SH; ++j)
    for (int i = 0; i < SW; ++i) h[j * SW + i] = (T)(3 + ((i * 7 + j * 13) % 29));
  im.fill_host(h);
  for (int mode = 0; mode < 5; ++mode) {
    int bt = mode == 0 ? NPP_BORDER_REPLICATE : mode == 1 ? NPP_BORDER_CONSTANT
             : mode == 2 ? NPP_BORDER_WRAP : mode == 3 ? NPP_BORDER_MIRROR : 5;
    Dst<T> dst(W, H);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp8u>) {
      st = ismax ? nppiFilterMaxBorder_8u_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx)
                 : nppiFilterMinBorder_8u_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx);
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      st = ismax ? nppiFilterMaxBorder_16u_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx)
                 : nppiFilterMinBorder_16u_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx);
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      st = ismax ? nppiFilterMaxBorder_16s_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx)
                 : nppiFilterMinBorder_16s_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx);
    } else {
      st = ismax ? nppiFilterMaxBorder_32f_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx)
                 : nppiFilterMinBorder_32f_C1R_Ctx(im.src, im.step, NppiSize{SW, SH}, NppiPoint{offx, offy}, dst.p, dst.step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)bt, g_ctx);
    }
    std::vector<T> got = dst.get();
    int bad = -1;
    Mode mm = mode == 0 ? M_REPL : mode == 1 ? M_ZERO : mode == 2 ? M_RAW : mode == 3 ? M_MIRROR : M_MIRROR_EDGE;
    if (st == NPP_NO_ERROR) {
      auto ref = ismax ? model(im, W, H, 3, 3, 1, 1, offx, offy, SW, SH, mm, opp_max<T>)
                       : model(im, W, H, 3, 3, 1, 1, offx, offy, SW, SH, mm, opp_min<T>);
      bad = 0;
      for (size_t i = 0; i < got.size(); ++i) if (got[i] != ref[i]) ++bad;
      if (bad) printf("    got0=%g ref0=%g\n", (double)got[0], (double)ref[0]);
    }
    printf("B %-4s %-3s mode%d: %d %s mism=%d\n", tag, ismax ? "max" : "min", bt, (int)st, sn(st), bad);
  }
}

template <class T>
static void minmax_mask_gt_roi(const char *tag) {
  Img<T> im(12, 12);
  std::vector<T> h(144, (T)5);
  im.fill_host(h);
  Dst<T> dst(4, 4);
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp8u>) st = nppiFilterMax_8u_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16u>) st = nppiFilterMax_16u_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16s>) st = nppiFilterMax_16s_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, g_ctx);
  else st = nppiFilterMax_32f_C1R_Ctx(im.src, im.step, dst.p, dst.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, g_ctx);
  printf("plain mask>roi %-4s max: %d %s\n", tag, (int)st, sn(st));
  Dst<T> d2(4, 4);
  if constexpr (std::is_same_v<T, Npp8u>) st = nppiFilterMedian_8u_C1R_Ctx(im.src, im.step, d2.p, d2.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16u>) st = nppiFilterMedian_16u_C1R_Ctx(im.src, im.step, d2.p, d2.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16s>) st = nppiFilterMedian_16s_C1R_Ctx(im.src, im.step, d2.p, d2.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, g_ctx);
  else st = nppiFilterMedian_32f_C1R_Ctx(im.src, im.step, d2.p, d2.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, g_ctx);
  printf("plain mask>roi %-4s median: %d %s\n", tag, (int)st, sn(st));
  Dst<T> d3(4, 4);
  if constexpr (std::is_same_v<T, Npp8u>) st = nppiFilterMedianBorder_8u_C1R_Ctx(im.src, im.step, NppiSize{12, 12}, NppiPoint{0, 0}, d3.p, d3.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16u>) st = nppiFilterMedianBorder_16u_C1R_Ctx(im.src, im.step, NppiSize{12, 12}, NppiPoint{0, 0}, d3.p, d3.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16s>) st = nppiFilterMedianBorder_16s_C1R_Ctx(im.src, im.step, NppiSize{12, 12}, NppiPoint{0, 0}, d3.p, d3.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
  else st = nppiFilterMedianBorder_32f_C1R_Ctx(im.src, im.step, NppiSize{12, 12}, NppiPoint{0, 0}, d3.p, d3.step, NppiSize{4, 4}, NppiSize{9, 5}, NppiPoint{2, 2}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
  printf("border mask>roi %-4s median: %d %s\n", tag, (int)st, sn(st));
}

static void medianb_even() {
  {
    Img<Npp8u> im(10, 10);
    std::vector<Npp8u> h(100);
    for (int j = 0; j < 10; ++j) for (int i = 0; i < 10; ++i) h[j * 10 + i] = (Npp8u)(2 * (i + j));
    im.fill_host(h);
    Dst<Npp8u> dst(6, 6);
    NppStatus st = nppiFilterMedianBorder_8u_C1R_Ctx(im.src, im.step, NppiSize{10, 10}, NppiPoint{3, 3}, dst.p, dst.step, NppiSize{6, 6}, NppiSize{1, 2}, NppiPoint{0, 0}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
    auto got = dst.get();
    printf("medianb even 8u: %d %s", (int)st, sn(st));
    if (st == NPP_NO_ERROR) { printf(" row0:"); for (int x = 0; x < 6; ++x) printf(" %d", got[x]); }
    printf("\n");
  }
  {
    Img<Npp16s> im(10, 10);
    std::vector<Npp16s> h(100);
    for (int j = 0; j < 10; ++j) for (int i = 0; i < 10; ++i) h[j * 10 + i] = (Npp16s)(i - j - 5);
    im.fill_host(h);
    Dst<Npp16s> dst(6, 6);
    NppStatus st = nppiFilterMedianBorder_16s_C1R_Ctx(im.src, im.step, NppiSize{10, 10}, NppiPoint{3, 3}, dst.p, dst.step, NppiSize{6, 6}, NppiSize{1, 2}, NppiPoint{0, 0}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
    auto got = dst.get();
    printf("medianb even 16s: %d %s", (int)st, sn(st));
    if (st == NPP_NO_ERROR) { printf(" row0:"); for (int x = 0; x < 6; ++x) printf(" %d", got[x]); }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  nppGetStreamContext(&g_ctx);
  std::string sec = argc > 1 ? argv[1] : "all";
  if (sec == "buf" || sec == "all") grid();
  if (sec == "roi7") grid_roi(7, 7);
  if (sec == "roiv") roi_vary();
  if (sec == "roi32") grid_roi(32, 32);
  if (sec == "roi64x16") grid_roi(64, 16);
  if (sec == "edge" || sec == "all") edge();
  if (sec == "vals" || sec == "all") {
    minmax_values<Npp8u>("8u", true, 3, 3, 1, 1);
    minmax_values<Npp8u>("8u", false, 3, 3, 1, 1);
    minmax_values<Npp8u>("8u", true, 2, 2, 1, 1);
    minmax_values<Npp8u>("8u", false, 3, 1, 3, 0);
    minmax_values<Npp16u>("16u", true, 3, 3, 0, 0);
    minmax_values<Npp16u>("16u", false, 3, 3, 2, 2);
    minmax_values<Npp16s>("16s", true, 3, 3, 1, 1);
    minmax_values<Npp16s>("16s", false, 2, 2, 1, 1);
    minmax_values<Npp32f>("32f", true, 3, 3, 1, 1);
    minmax_values<Npp32f>("32f", false, 3, 1, 0, 0);
    minmax_border_values<Npp8u>("8u", true);
    minmax_border_values<Npp8u>("8u", false);
    minmax_border_values<Npp16u>("16u", true);
    minmax_border_values<Npp16u>("16u", false);
    minmax_border_values<Npp16s>("16s", true);
    minmax_border_values<Npp16s>("16s", false);
    minmax_border_values<Npp32f>("32f", true);
    minmax_border_values<Npp32f>("32f", false);
    minmax_mask_gt_roi<Npp8u>("8u");
    minmax_mask_gt_roi<Npp16u>("16u");
    minmax_mask_gt_roi<Npp16s>("16s");
    minmax_mask_gt_roi<Npp32f>("32f");
  }
  if (sec == "misc" || sec == "all") medianb_even();
  fflush(stdout);
  return 0;
}
