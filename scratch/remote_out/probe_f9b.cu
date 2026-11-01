// NVIDIA NPP 13.4 (RTX 5090) follow-up semantics probe for package f9.
// Sections by argv[1]: advimp, advstep, mirror, median, bufsz, chan, align, valid
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"

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
#include <type_traits>
#include <vector>

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
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOT_EVEN_STEP";
    case NPP_MASK_SIZE_ERROR: return "MASK_SIZE";
    case NPP_ANCHOR_ERROR: return "ANCHOR";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOT_SUPPORTED_MODE";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR: return "CUDA_KERNEL";
    default: return "OTHER";
  }
}

#define CK(what, call)                                                        \
  do {                                                                        \
    NppStatus st_ = (call);                                                   \
    printf("%-46s %4d %s\n", what, (int)st_, sn(st_));                        \
    fflush(stdout);                                                           \
  } while (0)

// like CK but aborts the section if the CUDA context got poisoned
#define CKL(what, call)                                                       \
  do {                                                                        \
    NppStatus st_ = (call);                                                   \
    printf("%-46s %4d %s\n", what, (int)st_, sn(st_));                        \
    fflush(stdout);                                                           \
    if (st_ == NPP_CUDA_KERNEL_EXECUTION_ERROR) {                             \
      printf("!! CUDA context poisoned, aborting section\n");                 \
      return;                                                                 \
    }                                                                         \
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

// ---------------------------------------------------------------------------
// Generic image: allocation AW x AH (single channel for now), image content
// set by a lambda; pSrc points into the allocation at (px, py).
// ---------------------------------------------------------------------------
static const int AW = 40, AH = 40;

template <class T>
struct Img {
  std::vector<T> h;
  DevBuf d;
  int px, py, aw, ah;
  int stepBytes;
  Img(int px_, int py_, int aw_ = AW, int ah_ = AH)
      : h((size_t)aw_ * ah_, T(0)), px(px_), py(py_), aw(aw_), ah(ah_),
        stepBytes(aw_ * (int)sizeof(T)) {}
  void put() { d.put(h); }
  const T *src() const { return (const T *)d.p + (size_t)py * aw + px; }
  T &alloc_at(int ax, int ay) { return h[(size_t)ay * aw + ax]; }
};

// interleaved multi-channel source: image S x S (C channels) at alloc (8,8)
template <class T>
struct MultiSrc {
  std::vector<T> h;
  DevBuf d;
  int aw, ah, C, stepBytes;
  template <class F>
  MultiSrc(int S, int C_, F fill) : aw(S + 16), ah(S + 16), C(C_) {
    h.assign((size_t)aw * ah * C, T(0));
    for (int j = 0; j < S; ++j)
      for (int i = 0; i < S; ++i)
        for (int c = 0; c < C; ++c)
          h[((size_t)(8 + j) * aw + (8 + i)) * C + c] = fill(i, j, c);
    stepBytes = aw * C * (int)sizeof(T);
  }
  void put() { d.put(h); }
  const T *src() const { return (const T *)d.p + ((size_t)8 * aw + 8) * C; }
};

// interleaved multi-channel dst with canary padding
template <class T>
struct MultiDst {
  int W, H, C, stepBytes;
  DevBuf d;
  MultiDst(int W_, int H_, int C_)
      : W(W_), H(H_), C(C_), stepBytes((W_ * C_ + 4) * (int)sizeof(T)) {
    std::vector<T> c(((size_t)stepBytes / sizeof(T)) * (H + 8), T(0xAB));
    d.put(c);
  }
  T *p() { return (T *)d.p; }
  std::vector<T> region() {
    std::vector<T> v = d.get<T>(0);
    std::vector<T> r((size_t)W * H * C);
    size_t rowElems = stepBytes / sizeof(T);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W * C; ++x) r[(size_t)y * W * C + x] = v[y * rowElems + x];
    return r;
  }
  bool canaryIntact() {
    std::vector<T> v = d.get<T>(0);
    size_t rowElems = stepBytes / sizeof(T);
    for (int y = 0; y < H; ++y)
      for (int x = W * C; x < (int)rowElems; ++x)
        if (v[y * rowElems + x] != T(0xAB)) return false;
    for (size_t k = rowElems * H; k < v.size(); ++k)
      if (v[k] != T(0xAB)) return false;
    return true;
  }
  void print(const char *tag) {
    std::vector<T> r = region();
    printf("%s (canary %s):\n", tag, canaryIntact() ? "intact" : "CLOBBERED");
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        printf("(");
        for (int c = 0; c < C; ++c) {
          if constexpr (std::is_floating_point_v<T>)
            printf(c ? ",%g" : "%g", (double)r[((size_t)y * W + x) * C + c]);
          else
            printf(c ? ",%d" : "%d", (int)r[((size_t)y * W + x) * C + c]);
        }
        printf(") ");
      }
      printf("\n");
    }
  }
};

// Dst with canary margins; stepBytes = (W + margin) * es
template <class T>
struct Dst {
  static const int M = 4;
  int W, H, stepBytes;
  DevBuf d;
  Dst(int W_, int H_, int step_override = 0)
      : W(W_), H(H_), stepBytes(step_override ? step_override : (W_ + M) * (int)sizeof(T)) {
    std::vector<T> c(((size_t)stepBytes / sizeof(T)) * (H + M), T(0xAB));
    d.put(c);
  }
  std::vector<T> region() {
    std::vector<T> v = d.get<T>(0);
    std::vector<T> r(W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) r[y * W + x] = v[(size_t)y * (stepBytes / sizeof(T)) + x];
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

// ===========================================================================
// advimp: FilterBoxBorderAdvanced impulse response (8u C1)
// ===========================================================================
static void adv_imp_case(const char *tag, int offx, int offy, int sw, int sh, int W, int H,
                         int mw, int mh, int ax, int ay, int ix, int iy, Npp8u iv,
                         int bufMul = 1, bool canaryBuf = false) {
  printf("-- %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) impulse=(%d,%d)=%d --\n",
         tag, offx, offy, sw, sh, W, H, mw, mh, ax, ay, ix, iy, (int)iv);
  // image origin at alloc (8,8); pSrc at image origin + offset
  Img<Npp8u> im(8 + offx, 8 + offy);
  // image pixels: image (i,j) at alloc (8+i, 8+j)
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i) im.alloc_at(8 + i, 8 + j) = 0;
  im.alloc_at(8 + ix, 8 + iy) = iv;
  im.put();
  Dst<Npp8u> dst(W, H);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  size_t balloc = (size_t)bsz * bufMul + (canaryBuf ? 4096 : 0);
  DevBuf buf;
  buf.alloc(balloc);
  if (canaryBuf) {
    std::vector<Npp8u> tail(4096, 0xCD);
    if (buf.p) cudaMemcpy((char *)buf.p + (size_t)bsz * bufMul, tail.data(), 4096, cudaMemcpyHostToDevice);
  }
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
      dst.stepBytes, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE,
      (Npp8u *)buf.p, g_ctx);
  printf("status %d %s (bufsz=%d)\n", (int)st, sn(st), bsz);
  if (st != NPP_NO_ERROR) return;
  dst.printRegion("dst");
  if (canaryBuf && buf.p) {
    std::vector<Npp8u> tail = buf.get<Npp8u>(0);
    std::vector<Npp8u> tailc(4096);
    cudaMemcpy(tailc.data(), (char *)buf.p + (size_t)bsz * bufMul, 4096, cudaMemcpyDeviceToHost);
    bool intact = true;
    for (int k = 0; k < 4096; ++k)
      if (tailc[k] != 0xCD) { intact = false; break; }
    printf("buffer tail canary: %s\n", intact ? "intact" : "CLOBBERED");
  }
  fflush(stdout);
}

static void advimp() {
  printf("== advimp: advanced impulse response ==\n");
  // G10: everything interior (image 20x20, off 3,3, roi 8x8, impulse center)
  adv_imp_case("G10 interior", 3, 3, 20, 20, 8, 8, 3, 3, 1, 1, 6, 6, 200);
  // G1: roi right edge past image (2+7 > 8), impulse interior
  adv_imp_case("G1 roi-past-right", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5, 200);
  // G2: impulse at roi left column (image col = offx)
  adv_imp_case("G2 impulse-left-col", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 2, 5, 200);
  // G3: impulse one left of roi start
  adv_imp_case("G3 impulse-off-left", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 1, 5, 200);
  // G4: offset 0,0 (roi at image origin)
  adv_imp_case("G4 off0", 0, 0, 16, 16, 7, 7, 3, 3, 1, 1, 0, 0, 200);
  // G5: image smaller than roi+offset
  adv_imp_case("G5 src-smaller", 2, 3, 6, 6, 7, 7, 3, 3, 1, 1, 3, 3, 200);
  // G6: 5x3 mask
  adv_imp_case("G6 mask5x3", 2, 3, 16, 16, 7, 7, 5, 3, 2, 1, 5, 5, 200);
  // G7: 1x1 mask
  adv_imp_case("G7 mask1x1", 2, 3, 16, 16, 7, 7, 1, 1, 0, 0, 5, 5, 200);
  // G8: anchor 0,0
  adv_imp_case("G8 anchor00", 2, 3, 16, 16, 7, 7, 3, 3, 0, 0, 5, 5, 200);
  // G9: 4x buffer + canary tail
  adv_imp_case("G9 bufx4+canary", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5, 200, 4, true);
  // G11: two impulses to decode x=0 (far from other edges)
  {
    printf("-- G11 two-impulse --\n");
    Img<Npp8u> im(8 + 2, 8 + 3);
    for (int j = 0; j < 16; ++j)
      for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = 0;
    im.alloc_at(8 + 1, 8 + 5) = 200;   // image (1,5): left of roi
    im.alloc_at(8 + 12, 8 + 5) = 90;   // image (12,5): right of roi end (roi cols 2..8)
    im.put();
    Dst<Npp8u> dst(7, 7);
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{7, 7}, 1, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
        im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, dst.stepBytes,
        NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx);
    printf("status %d %s\n", (int)st, sn(st));
    if (st == NPP_NO_ERROR) dst.printRegion("dst");
  }
  fflush(stdout);
}

// 32f impulse: exact weights (impulse 1.0 -> dst = k/denominator)
static void adv_imp32_case(const char *tag, int offx, int offy, int sw, int sh, int W, int H,
                           int mw, int mh, int ax, int ay, int ix, int iy) {
  printf("-- %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) impulse=(%d,%d) --\n",
         tag, offx, offy, sw, sh, W, H, mw, mh, ax, ay, ix, iy);
  Img<Npp32f> im(8 + offx, 8 + offy);
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i) im.alloc_at(8 + i, 8 + j) = 0.f;
  im.alloc_at(8 + ix, 8 + iy) = 1.f;
  im.put();
  Dst<Npp32f> dst(W, H);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  NppStatus st = nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp32f *)dst.d.p,
      dst.stepBytes, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE,
      (Npp8u *)buf.p, g_ctx);
  printf("status %d %s (bufsz=%d)\n", (int)st, sn(st), bsz);
  if (st != NPP_NO_ERROR) return;
  std::vector<Npp32f> r = dst.region();
  printf("dst (raw *36, *9):\n");
  for (int y = 0; y < H; ++y) {
    printf("  ");
    for (int x = 0; x < W; ++x) {
      double v = r[y * W + x];
      printf("[%9.4g %6.3g] ", v * 36.0, v * 9.0);
    }
    printf("\n");
  }
  fflush(stdout);
}

// full-gradient advanced 8u for whole-ROI model comparison
static void adv_grad_case(const char *tag, int offx, int offy, int sw, int sh, int W, int H,
                          int mw, int mh, int ax, int ay) {
  printf("-- %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) --\n", tag, offx, offy, sw,
         sh, W, H, mw, mh, ax, ay);
  Img<Npp8u> im(8 + offx, 8 + offy);
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((3 * i + 5 * j) % 251);
  im.put();
  Dst<Npp8u> dst(W, H);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
      dst.stepBytes, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE,
      (Npp8u *)buf.p, g_ctx);
  printf("status %d %s\n", (int)st, sn(st));
  if (st == NPP_NO_ERROR) dst.printRegion("dst");
  fflush(stdout);
}

static void advimp32() {
  printf("== advimp32: advanced 32f impulse response ==\n");
  adv_imp32_case("G10f interior", 3, 3, 20, 20, 8, 8, 3, 3, 1, 1, 6, 6);
  adv_imp32_case("G1f roi-past-right", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5);
  adv_imp32_case("G2f impulse-left-col", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 2, 5);
  adv_imp32_case("G3f impulse-off-left", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 1, 5);
  adv_imp32_case("G4f impulse-right-col", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 8, 5);
  adv_imp32_case("G5f impulse-bottom-row", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 9);
  adv_imp32_case("G6f impulse-top-row", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 3);
  adv_imp32_case("G12f off0 impulse(1,5)", 0, 0, 20, 20, 7, 7, 3, 3, 1, 1, 1, 5);
  adv_imp32_case("G13f off0 impulse(5,1)", 0, 0, 20, 20, 7, 7, 3, 3, 1, 1, 5, 1);
  adv_grad_case("G18 grad off0", 0, 0, 16, 16, 7, 7, 3, 3, 1, 1);
  adv_grad_case("G19 grad off23", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1);
  adv_grad_case("G20 grad mask5x3 off23", 2, 3, 16, 16, 7, 7, 5, 3, 2, 1);
}

// ===========================================================================
// advstep: advanced validation/step matrix + bufsize non-square/channels
// ===========================================================================
template <class T>
static void adv_val_dtype(const char *tag) {
  printf("== advstep %s validation ==\n", tag);
  Img<T> im(8 + 2, 8 + 3);
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = T(7);
  im.put();
  const int W = 7, H = 7, es = (int)sizeof(T);
  Dst<T> dst(W, H);
  int bsz = 0;
  if constexpr (std::is_same_v<T, Npp64f>)
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{W, H}, 1, &bsz);
  else
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc(bsz < (1 << 26) ? (size_t)(1 << 26) : (size_t)bsz);
  NppiSize sz{16, 16}, roi{W, H}, m3{3, 3};
  NppiPoint a1{1, 1}, off{2, 3};
  auto call = [&](NppiSize r, NppiPoint o, int ds, NppiSize m, NppiPoint a, NppiBorderType b, void *pb) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp8u>)
      return nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(im.src(), im.stepBytes, sz, o, (Npp8u *)dst.d.p, ds, r, m, a, b, (Npp8u *)pb, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16u>)
      return nppiFilterBoxBorderAdvanced_16u_C1R_Ctx(im.src(), im.stepBytes, sz, o, (Npp16u *)dst.d.p, ds, r, m, a, b, (Npp8u *)pb, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return nppiFilterBoxBorderAdvanced_16s_C1R_Ctx(im.src(), im.stepBytes, sz, o, (Npp16s *)dst.d.p, ds, r, m, a, b, (Npp8u *)pb, g_ctx);
    else if constexpr (std::is_same_v<T, Npp32f>)
      return nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(im.src(), im.stepBytes, sz, o, (Npp32f *)dst.d.p, ds, r, m, a, b, (Npp8u *)pb, g_ctx);
    else
      return nppiFilterBoxBorderAdvanced_64f_C1R_Ctx(im.src(), im.stepBytes, sz, o, (Npp64f *)dst.d.p, ds, r, m, a, b, (Npp8u *)pb, g_ctx);
  };
  CKL("roi_zero_w dstep_ok", call(NppiSize{0, H}, off, dst.stepBytes, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  const int steps[] = {0, 1, 2, 3, 5, 7, 11, 16, 28, 32};
  for (int k = 0; k < 10; ++k) {
    char nm[48];
    snprintf(nm, 48, "roi0x7 dstep=%d", steps[k]);
    CKL(nm, call(NppiSize{0, 7}, off, steps[k] * 1, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  }
  CKL("roi7x0 dstep=11", call(NppiSize{7, 0}, off, 11, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  CKL("roi1x7 dstep=11", call(NppiSize{1, 7}, off, 11, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  CKL("roi0x1 dstep=11", call(NppiSize{0, 1}, off, 11, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  CKL("roi0x0 dstep=11", call(NppiSize{0, 0}, off, 11, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  CKL("roi0x7 mask1x1", call(NppiSize{0, 7}, off, 11, NppiSize{1, 1}, NppiPoint{0, 0}, NPP_BORDER_REPLICATE, buf.p));
  CKL("border_mirror", call(roi, off, dst.stepBytes, m3, a1, NPP_BORDER_MIRROR, buf.p));
  CKL("border_5", call(roi, off, dst.stepBytes, m3, a1, (NppiBorderType)5, buf.p));
  CKL("null_buf", call(roi, off, dst.stepBytes, m3, a1, NPP_BORDER_REPLICATE, nullptr));
  CKL("mask_zero", call(roi, off, dst.stepBytes, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, buf.p));
  CKL("mask_neg", call(roi, off, dst.stepBytes, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, buf.p));
  CKL("anchor_gt", call(roi, off, dst.stepBytes, m3, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, buf.p));
  // alignment: roi 4 wide so rowBytes=4*es; steps that are >= rowBytes but odd
  for (int k = 0; k < 5; ++k) {
    char nm[48];
    int s = 4 * es + k;
    snprintf(nm, 48, "align roi4x3 dstep=%d", s);
    CKL(nm, call(NppiSize{4, 3}, off, s, m3, a1, NPP_BORDER_REPLICATE, buf.p));
  }
  // non-square bufsize
  printf("-- adv bufsize non-square %s --\n", tag);
  const int rois[][2] = {{7, 3}, {3, 7}, {16, 5}, {5, 16}, {100, 7}, {7, 100}, {1, 1}, {0, 7}, {7, 0}, {-1, 7}, {7, -1}};
  for (int k = 0; k < 11; ++k) {
    int s1 = -1;
    NppStatus st1 = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{rois[k][0], rois[k][1]}, 1, &s1);
    if constexpr (std::is_same_v<T, Npp64f>) {
      (void)st1; (void)s1;
    } else { (void)st1; (void)s1; }
    printf("roi %dx%d ch1: %d (%d)", rois[k][0], rois[k][1], s1, (int)st1);
    if constexpr (!std::is_same_v<T, Npp64f>) {
      int s3 = -1;
      nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{rois[k][0], rois[k][1]}, 3, &s3);
      int s4 = -1;
      nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{rois[k][0], rois[k][1]}, 4, &s4);
      printf(" ch3: %d ch4: %d", s3, s4);
    } else {
      int s3 = -1;
      nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{rois[k][0], rois[k][1]}, 3, &s3);
      int s4 = -1;
      nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{rois[k][0], rois[k][1]}, 4, &s4);
      printf(" 64ch3: %d 64ch4: %d", s3, s4);
    }
    printf("\n");
  }
  fflush(stdout);
}

static void advstep() {
  printf("== advstep: channels/neg validation of GetDeviceBufferSize ==\n");
  const int chs[] = {-1, 0, 1, 2, 3, 4, 5, 6, 8, 16};
  for (int k = 0; k < 10; ++k) {
    int s = -1;
    NppStatus st = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{7, 5}, chs[k], &s);
    int s64 = -1;
    NppStatus st64 = nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{7, 5}, chs[k], &s64);
    printf("ch %3d: %d (%d)   _64: %d (%d)\n", chs[k], s, (int)st, s64, (int)st64);
  }
  // null hpBufferSize SEGFAULTS NVIDIA 13.4 host code (verified) - not callable
  adv_val_dtype<Npp8u>("8u");
  adv_val_dtype<Npp16u>("16u");
  adv_val_dtype<Npp16s>("16s");
  adv_val_dtype<Npp32f>("32f");
  adv_val_dtype<Npp64f>("64f");
}

// ===========================================================================
// risky: negative-mask launches (may poison the CUDA context; run LAST,
// in its own process invocation)
// ===========================================================================
template <class T>
static void risky_maskneg(const char *tag) {
  printf("== risky mask_neg %s ==\n", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  NppiSize r7{7, 7}, s16{16, 16};
  NppiPoint a1{1, 1}, off{2, 3};
  {
    Dst<T> dst(7, 7);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterBoxBorder_16u_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp16u *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterBoxBorder_16s_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp16s *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, g_ctx);
    else
      st = nppiFilterBoxBorder_32f_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp32f *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, g_ctx);
    printf("%-12s %-12s %4d %s\n", tag, "boxborder", (int)st, sn(st));
  }
  {
    Dst<T> dst(7, 7);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterMax_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterMax_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, g_ctx);
    else
      st = nppiFilterMax_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, g_ctx);
    printf("%-12s %-12s %4d %s\n", tag, "max", (int)st, sn(st));
  }
  {
    Dst<T> dst(7, 7);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterMin_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterMin_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, g_ctx);
    else
      st = nppiFilterMin_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, g_ctx);
    printf("%-12s %-12s %4d %s\n", tag, "min", (int)st, sn(st));
  }
  {
    Dst<T> dst(7, 7);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterMaxBorder_16u_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp16u *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterMaxBorder_16s_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp16s *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, g_ctx);
    else
      st = nppiFilterMaxBorder_32f_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp32f *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, NPP_BORDER_REPLICATE, g_ctx);
    printf("%-12s %-12s %4d %s\n", tag, "maxborder", (int)st, sn(st));
  }
  {
    Dst<T> dst(7, 7);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterMedian_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, nullptr, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterMedian_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, nullptr, g_ctx);
    else
      st = nppiFilterMedian_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, nullptr, g_ctx);
    printf("%-12s %-12s %4d %s\n", tag, "median", (int)st, sn(st));
  }
  {
    Dst<T> dst(7, 7);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterMedianBorder_16u_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp16u *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, nullptr, NPP_BORDER_REPLICATE, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterMedianBorder_16s_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp16s *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, nullptr, NPP_BORDER_REPLICATE, g_ctx);
    else
      st = nppiFilterMedianBorder_32f_C1R_Ctx(im.src(), im.stepBytes, s16, off, (Npp32f *)dst.d.p, dst.stepBytes, r7, NppiSize{-3, 3}, a1, nullptr, NPP_BORDER_REPLICATE, g_ctx);
    printf("%-12s %-12s %4d %s\n", tag, "medborder", (int)st, sn(st));
  }
  fflush(stdout);
}

static void risky() {
  printf("== risky: negative masks ==\n");
  risky_maskneg<Npp16u>("16u");
  risky_maskneg<Npp16s>("16s");
  risky_maskneg<Npp32f>("32f");
  fflush(stdout);
}


// ===========================================================================
// mirror: BoxBorder MIRROR/MIRROR_REFL semantics (8u) + mode acceptance
// ===========================================================================
static void boxborder_mirror_case(const char *tag, NppiBorderType b, int offx, int offy) {
  Img<Npp8u> im(8 + offx, 8 + offy);
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((i + 16 * j) & 0xFF);
  im.put();
  Dst<Npp8u> dst(7, 7);
  char nm[64];
  snprintf(nm, 64, "%s off=(%d,%d)", tag, offx, offy);
  NppStatus st = nppiFilterBoxBorder_8u_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
      dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, b, g_ctx);
  printf("%s: %d %s\n", nm, (int)st, sn(st));
  if (st == NPP_NO_ERROR) dst.printRegion(nm);
  fflush(stdout);
}

static void medianborder_mirror_case(const char *tag, NppiBorderType b, int offx, int offy,
                                     bool vertical) {
  // src = x (horizontal probe) or src = y (vertical probe); mask 1x3 / 3x1
  Img<Npp8u> im(8 + offx, 8 + offy);
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i)
      im.alloc_at(8 + i, 8 + j) = (Npp8u)(vertical ? j : i);
  im.put();
  Dst<Npp8u> dst(7, 1);
  NppiSize mask = vertical ? NppiSize{1, 3} : NppiSize{3, 1};
  NppiPoint anc = vertical ? NppiPoint{0, 1} : NppiPoint{1, 0};
  NppStatus st = nppiFilterMedianBorder_8u_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
      dst.stepBytes, NppiSize{7, 1}, mask, anc, nullptr, b, g_ctx);
  printf("%s: %d %s\n", tag, (int)st, sn(st));
  if (st == NPP_NO_ERROR) dst.printRegion(tag);
  fflush(stdout);
}

template <class T>
static void border_accept_box(const char *tag) {
  printf("== mirror %s boxborder border acceptance ==\n", tag);
  Img<T> im(8 + 2, 8 + 3);
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = T(7);
  im.put();
  Dst<T> dst(7, 7);
  const int modes[] = {0, 1, 2, 3, 4, 5, 7};
  for (int k = 0; k < 7; ++k) {
    char nm[32];
    snprintf(nm, 32, "border=%d", modes[k]);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp8u>)
      st = nppiFilterBoxBorder_8u_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)modes[k], g_ctx);
    else if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterBoxBorder_16u_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp16u *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)modes[k], g_ctx);
    else
      st = nppiFilterBoxBorder_32f_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp32f *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, (NppiBorderType)modes[k], g_ctx);
    printf("%-10s %4d %s\n", nm, (int)st, sn(st));
  }
  fflush(stdout);
}

template <class T>
static void border_accept_median(const char *tag) {
  printf("== mirror %s medianborder border acceptance ==\n", tag);
  Img<T> im(8 + 2, 8 + 3);
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = T(7);
  im.put();
  Dst<T> dst(7, 7);
  const int modes[] = {0, 1, 2, 3, 4, 5, 7};
  for (int k = 0; k < 7; ++k) {
    char nm[32];
    snprintf(nm, 32, "border=%d", modes[k]);
    NppStatus st;
    if constexpr (std::is_same_v<T, Npp8u>)
      st = nppiFilterMedianBorder_8u_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, (NppiBorderType)modes[k], g_ctx);
    else if constexpr (std::is_same_v<T, Npp16u>)
      st = nppiFilterMedianBorder_16u_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp16u *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, (NppiBorderType)modes[k], g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      st = nppiFilterMedianBorder_16s_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp16s *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, (NppiBorderType)modes[k], g_ctx);
    else
      st = nppiFilterMedianBorder_32f_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp32f *)dst.d.p, dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, (NppiBorderType)modes[k], g_ctx);
    printf("%-10s %4d %s\n", nm, (int)st, sn(st));
  }
  fflush(stdout);
}

static void mirror() {
  printf("== mirror: MIRROR(4) vs MIRROR_REFL(5) ==\n");
  boxborder_mirror_case("boxborder MIRROR off00", NPP_BORDER_MIRROR, 0, 0);
  boxborder_mirror_case("boxborder MIRROR_REFL off00", (NppiBorderType)5, 0, 0);
  boxborder_mirror_case("boxborder MIRROR off99", NPP_BORDER_MIRROR, 9, 9);
  boxborder_mirror_case("boxborder MIRROR_REFL off99", (NppiBorderType)5, 9, 9);
  boxborder_mirror_case("boxborder MIRROR off1010", NPP_BORDER_MIRROR, 10, 10);
  boxborder_mirror_case("boxborder MIRROR_REFL off1010", (NppiBorderType)5, 10, 10);
  medianborder_mirror_case("medborder MIRROR off150 h", NPP_BORDER_MIRROR, 15, 0, false);
  medianborder_mirror_case("medborder MIRROR_REFL off150 h", (NppiBorderType)5, 15, 0, false);
  medianborder_mirror_case("medborder MIRROR off015 v", NPP_BORDER_MIRROR, 0, 15, true);
  medianborder_mirror_case("medborder MIRROR_REFL off015 v", (NppiBorderType)5, 0, 15, true);
  border_accept_box<Npp8u>("8u");
  border_accept_box<Npp16u>("16u");
  border_accept_box<Npp32f>("32f");
  border_accept_median<Npp8u>("8u");
  border_accept_median<Npp16u>("16u");
  border_accept_median<Npp16s>("16s");
  border_accept_median<Npp32f>("32f");
}

// ===========================================================================
// median: even-count rounding, 2x2 anchor geometry, big masks, C3
// ===========================================================================
template <class T>
static void med_even_rounding(const char *tag) {
  printf("== median %s even-count (mask 1x2, anchor 0,0) ==\n", tag);
  // src[x] = x + 1 (x=0..19); window at dst x covers src x..x+1
  Img<T> im(8, 8);
  for (int j = 0; j < 20; ++j)
    for (int i = 0; i < 20; ++i) {
      int v = (j == 0) ? i + 1 : 0;
      if constexpr (std::is_same_v<T, Npp16u>)
        im.alloc_at(8 + i, 8 + j) = (Npp16u)(40001 + v - 1);
      else if constexpr (std::is_same_v<T, Npp16s>)
        im.alloc_at(8 + i, 8 + j) = (Npp16s)(v - 9);
      else
        im.alloc_at(8 + i, 8 + j) = T(v);
    }
  im.put();
  Dst<T> dst(14, 1);
  NppStatus st;
  if constexpr (std::is_same_v<T, Npp8u>)
    st = nppiFilterMedian_8u_C1R_Ctx(im.src(), im.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{14, 1}, NppiSize{1, 2}, NppiPoint{0, 0}, nullptr, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16u>)
    st = nppiFilterMedian_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, dst.stepBytes, NppiSize{14, 1}, NppiSize{1, 2}, NppiPoint{0, 0}, nullptr, g_ctx);
  else if constexpr (std::is_same_v<T, Npp16s>)
    st = nppiFilterMedian_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, dst.stepBytes, NppiSize{14, 1}, NppiSize{1, 2}, NppiPoint{0, 0}, nullptr, g_ctx);
  else
    st = nppiFilterMedian_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, dst.stepBytes, NppiSize{14, 1}, NppiSize{1, 2}, NppiPoint{0, 0}, nullptr, g_ctx);
  printf("status %d %s\n", (int)st, sn(st));
  if (st == NPP_NO_ERROR) dst.printRegion("dst");
  fflush(stdout);
}

static void median() {
  printf("== median details ==\n");
  med_even_rounding<Npp8u>("8u");
  med_even_rounding<Npp16u>("16u");
  med_even_rounding<Npp16s>("16s");
  med_even_rounding<Npp32f>("32f");

  // 2x2 anchor geometry: src = 10*y + x, image 20x20, ROI 6x6 at (0,0)
  printf("-- median 2x2 anchor geometry (src=10y+x) --\n");
  {
    Img<Npp8u> im(8, 8);
    for (int j = 0; j < 20; ++j)
      for (int i = 0; i < 20; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)(10 * j + i);
    im.put();
    const int anchors[][2] = {{0, 0}, {1, 1}, {2, 2}, {2, 0}, {0, 2}};
    for (int k = 0; k < 5; ++k) {
      Dst<Npp8u> dst(6, 6);
      NppStatus st = nppiFilterMedian_8u_C1R_Ctx(
          im.src(), im.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{6, 6},
          NppiSize{2, 2}, NppiPoint{anchors[k][0], anchors[k][1]}, nullptr, g_ctx);
      printf("anchor (%d,%d): %d %s\n", anchors[k][0], anchors[k][1], (int)st, sn(st));
      if (st == NPP_NO_ERROR) dst.printRegion("dst");
    }
    // anchor beyond mask (3,1), mask 3x3
    Dst<Npp8u> dst(6, 6);
    NppStatus st = nppiFilterMedian_8u_C1R_Ctx(
        im.src(), im.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{6, 6},
        NppiSize{3, 3}, NppiPoint{3, 1}, nullptr, g_ctx);
    printf("anchor (3,1) mask 3x3: %d %s\n", (int)st, sn(st));
    if (st == NPP_NO_ERROR) dst.printRegion("dst");
  }
  fflush(stdout);

  // big masks, buffer via GetBufferSize (SAFE)
  printf("-- median big masks (128x128 src, ROI 16x16 interior) --\n");
  {
    const int S = 128;
    Img<Npp8u> im(8, 8, S + 16, S + 16);
    for (int j = 0; j < S; ++j)
      for (int i = 0; i < S; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((3 * i + 5 * j) % 251);
    im.put();
    const int masks[][2] = {{9, 9}, {15, 15}, {16, 16}, {17, 17}};
    for (int k = 0; k < 4; ++k) {
      NppiSize m{masks[k][0], masks[k][1]};
      Npp32u bsz = 0;
      nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, m, &bsz, g_ctx);
      DevBuf buf;
      buf.alloc(bsz ? bsz : 16);
      Dst<Npp8u> dst(16, 16);
      NppStatus st = nppiFilterMedian_8u_C1R_Ctx(
          im.src(), im.stepBytes, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{16, 16}, m,
          NppiPoint{m.width / 2, m.height / 2}, (Npp8u *)buf.p, g_ctx);
      printf("mask %dx%d bufsize=%u status %d %s\n", m.width, m.height, bsz, (int)st, sn(st));
      if (st == NPP_NO_ERROR) dst.printRegion("dst");
      fflush(stdout);
    }
    // same masks through MedianBorder (replicate), ROI interior so identical values expected
    for (int k = 0; k < 4; ++k) {
      NppiSize m{masks[k][0], masks[k][1]};
      Npp32u bsz = 0;
      nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, m, &bsz, NPP_BORDER_REPLICATE, g_ctx);
      DevBuf buf;
      buf.alloc(bsz ? bsz : 16);
      Dst<Npp8u> dst(16, 16);
      NppStatus st = nppiFilterMedianBorder_8u_C1R_Ctx(
          im.src(), im.stepBytes, NppiSize{S, S}, NppiPoint{10, 10}, (Npp8u *)dst.d.p,
          dst.stepBytes, NppiSize{16, 16}, m, NppiPoint{m.width / 2, m.height / 2},
          (Npp8u *)buf.p, NPP_BORDER_REPLICATE, g_ctx);
      printf("border mask %dx%d bufsize=%u status %d %s\n", m.width, m.height, bsz, (int)st, sn(st));
      if (st == NPP_NO_ERROR) dst.printRegion("dst");
      fflush(stdout);
    }
  }
  fflush(stdout);

  // C3 per-channel independence
  printf("-- median C3 per-channel --\n");
  {
    MultiSrc<Npp8u> ms(20, 3, [](int i, int j, int c) -> Npp8u {
      if (c == 0) return (Npp8u)((i * 7 + j * 3) % 251);
      if (c == 1) return (Npp8u)((i * 2 + j * 11) % 241);
      return (Npp8u)((i * 13 + j * 5) % 239);
    });
    ms.put();
    MultiDst<Npp8u> dst(7, 7, 3);
    NppStatus st = nppiFilterMedian_8u_C3R_Ctx(
        ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3},
        NppiPoint{1, 1}, nullptr, g_ctx);
    printf("status %d %s\n", (int)st, sn(st));
    if (st == NPP_NO_ERROR) dst.print("dst");
  }
  fflush(stdout);
}

// ===========================================================================
// bufsz: Median/MedianBorder GetBufferSize grid + validation (host-only)
// ===========================================================================
#define CAT2_(a, b) a##b
#define CAT_(a, b) CAT2_(a, b)
#define MEDBS(dt, lay) CAT_(CAT_(nppiFilterMedianGetBufferSize_, dt), CAT_(lay, _Ctx))
#define MBBS(dt, lay) CAT_(CAT_(nppiFilterMedianBorderGetBufferSize_, dt), CAT_(lay, _Ctx))

#define DEF_MEDBS(dt)                                                          \
  static void medbs_##dt() {                                                   \
    printf("== bufsz median " #dt " ==\n");                                    \
    const int masks[][2] = {{1, 1},  {1, 2},  {2, 1},  {1, 3},  {3, 1},  {2, 2}, \
                            {3, 3},  {3, 4},  {4, 3},  {4, 4},  {4, 5},  {5, 4}, \
                            {4, 7},  {7, 4},  {5, 5},  {5, 7},  {7, 5},  {6, 8}, \
                            {8, 6},  {7, 7},  {8, 8},  {9, 8},  {8, 9},  {9, 9}, \
                            {10, 10}, {11, 11}, {12, 12}, {13, 13}, {14, 14},   \
                            {15, 15}, {16, 16}, {17, 17}, {18, 18}, {20, 20},   \
                            {24, 24}, {25, 25}, {32, 32}, {33, 33},             \
                            {4, 16}, {16, 4}, {4, 17}, {17, 4}, {2, 32},        \
                            {32, 2}, {1, 64}, {64, 1}, {1, 65}, {65, 1},        \
                            {2, 33}, {33, 2}, {3, 22}, {22, 3}, {5, 13},        \
                            {13, 5}, {1, 128}, {128, 1}, {1, 129}, {129, 1},    \
                            {6, 11}, {11, 6}, {10, 12}, {12, 10}, {15, 16},     \
                            {16, 15}, {9, 16}, {16, 9}};                        \
    for (size_t k = 0; k < sizeof(masks) / sizeof(masks[0]); ++k) {            \
      NppiSize m{masks[k][0], masks[k][1]};                                    \
      Npp32u s1 = 0xdeadbeef, s3 = 0xdeadbeef, s4 = 0xdeadbeef, sa = 0xdeadbeef, sb = 0xdeadbeef, sbm = 0xdeadbeef; \
      NppStatus st1 = MEDBS(dt, _C1R)(NppiSize{16, 16}, m, &s1, g_ctx);        \
      NppStatus st3 = MEDBS(dt, _C3R)(NppiSize{16, 16}, m, &s3, g_ctx);        \
      NppStatus st4 = MEDBS(dt, _C4R)(NppiSize{16, 16}, m, &s4, g_ctx);        \
      NppStatus sta = MEDBS(dt, _AC4R)(NppiSize{16, 16}, m, &sa, g_ctx);       \
      NppStatus stb = MBBS(dt, _C1R)(NppiSize{16, 16}, m, &sb, NPP_BORDER_REPLICATE, g_ctx); \
      NppStatus stbm = MBBS(dt, _C1R)(NppiSize{16, 16}, m, &sbm, NPP_BORDER_MIRROR, g_ctx); \
      printf("mask %2dx%-2d: C1=%-9u(%d) C3=%-9u(%d) C4=%-9u(%d) AC4=%-9u(%d) BC1=%-9u(%d) BC1m=%u(%d)\n", \
             m.width, m.height, s1, (int)st1, s3, (int)st3, s4, (int)st4, sa,  \
             (int)sta, sb, (int)stb, sbm, (int)stbm);                          \
    }                                                                          \
    fflush(stdout);                                                            \
  }
DEF_MEDBS(8u)
DEF_MEDBS(16u)
DEF_MEDBS(16s)
DEF_MEDBS(32f)

static void bufsz() {
  printf("== bufsz ==\n");
  medbs_8u();
  medbs_16u();
  medbs_16s();
  medbs_32f();

  // ROI dependence for large masks
  printf("-- bufsz ROI dependence (8u C1) --\n");
  const int rois[][2] = {{1, 1}, {2, 2}, {4, 4}, {7, 7}, {8, 8}, {9, 9}, {16, 16}, {17, 17},
                         {32, 32}, {64, 64}, {128, 128}, {256, 256}, {257, 257}, {512, 512},
                         {0, 7}, {7, 0}, {0, 0}, {-1, 7}, {7, -1}, {-3, -3}, {1, 0}, {0, 1}};
  const int mks[][2] = {{16, 16}, {17, 17}, {3, 3}, {8, 8}};
  for (size_t r = 0; r < sizeof(rois) / sizeof(rois[0]); ++r) {
    printf("roi %4dx%-4d:", rois[r][0], rois[r][1]);
    for (size_t k = 0; k < sizeof(mks) / sizeof(mks[0]); ++k) {
      Npp32u s = 0xdeadbeef;
      NppStatus st = nppiFilterMedianGetBufferSize_8u_C1R_Ctx(
          NppiSize{rois[r][0], rois[r][1]}, NppiSize{mks[k][0], mks[k][1]}, &s, g_ctx);
      printf("  %2dx%-2d=%-9u(%d)", mks[k][0], mks[k][1], s, (int)st);
    }
    printf("\n");
  }
  fflush(stdout);

  // GetBufferSize validation
  printf("-- bufsz validation (8u C1 / 32f C1) --\n");
  {
    Npp32u s = 0xdeadbeef;
    // null hpBufferSize crashes nppiFilterMedianGetBufferSize (verified) - not callable
    CK("8u mask 0x0", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{0, 0}, &s, g_ctx));
    CK("8u mask -3x3", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{-3, 3}, &s, g_ctx));
    CK("8u mask 3x-3", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, -3}, &s, g_ctx));
    CK("8u mask 0x3", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{0, 3}, &s, g_ctx));
    CK("8u mask 3x0", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, 0}, &s, g_ctx));
    CK("8u roi 0x7", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{0, 7}, NppiSize{3, 3}, &s, g_ctx));
    CK("8u roi neg", nppiFilterMedianGetBufferSize_8u_C1R_Ctx(NppiSize{-3, 7}, NppiSize{3, 3}, &s, g_ctx));
    // null hpBufferSize crashes nppiFilterMedianGetBufferSize_32f too - not callable
    CK("32f mask 0x0", nppiFilterMedianGetBufferSize_32f_C1R_Ctx(NppiSize{16, 16}, NppiSize{0, 0}, &s, g_ctx));
    CK("32f mask neg", nppiFilterMedianGetBufferSize_32f_C1R_Ctx(NppiSize{16, 16}, NppiSize{-3, 3}, &s, g_ctx));
    CK("32f roi 0x7", nppiFilterMedianGetBufferSize_32f_C1R_Ctx(NppiSize{0, 7}, NppiSize{3, 3}, &s, g_ctx));
    CK("32f roi neg", nppiFilterMedianGetBufferSize_32f_C1R_Ctx(NppiSize{-3, 7}, NppiSize{3, 3}, &s, g_ctx));
    CK("16u mask neg", nppiFilterMedianGetBufferSize_16u_C1R_Ctx(NppiSize{16, 16}, NppiSize{-3, 3}, &s, g_ctx));
    CK("16s mask neg", nppiFilterMedianGetBufferSize_16s_C1R_Ctx(NppiSize{16, 16}, NppiSize{-3, 3}, &s, g_ctx));
    // null hpBufferSize crashes nppiFilterMedianBorderGetBufferSize (verified) - not callable
    CK("B8u mask 0x0", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{0, 0}, &s, NPP_BORDER_REPLICATE, g_ctx));
    CK("B8u mask neg", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{-3, 3}, &s, NPP_BORDER_REPLICATE, g_ctx));
    CK("B8u roi neg", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{-3, 7}, NppiSize{3, 3}, &s, NPP_BORDER_REPLICATE, g_ctx));
    CK("B8u border 0", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, 3}, &s, (NppiBorderType)0, g_ctx));
    CK("B8u border 1", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, 3}, &s, (NppiBorderType)1, g_ctx));
    CK("B8u border 3", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, 3}, &s, (NppiBorderType)3, g_ctx));
    CK("B8u border 7", nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, 3}, &s, (NppiBorderType)7, g_ctx));
    CK("B32f border 0", nppiFilterMedianBorderGetBufferSize_32f_C1R_Ctx(NppiSize{16, 16}, NppiSize{3, 3}, &s, (NppiBorderType)0, g_ctx));
    CK("B32f mask neg", nppiFilterMedianBorderGetBufferSize_32f_C1R_Ctx(NppiSize{16, 16}, NppiSize{-3, 3}, &s, NPP_BORDER_REPLICATE, g_ctx));
  }
  fflush(stdout);
}

// ===========================================================================
// chan: C4 vs AC4 alpha handling for box/max/min/median (+ C3)
// ===========================================================================
static void chan() {
  printf("== chan: C4/AC4 alpha semantics ==\n");
  auto fill = [](int i, int j, int c) -> Npp8u {
    if (c == 0) return (Npp8u)((7 * i + 3 * j) % 251);
    if (c == 1) return (Npp8u)((2 * i + 11 * j) % 241);
    if (c == 2) return (Npp8u)((13 * i + 5 * j) % 239);
    return (Npp8u)((i + j) & 0xFF);  // alpha varies
  };
  MultiSrc<Npp8u> ms(20, 4, fill);
  ms.put();
  const char *fams[] = {"box", "max", "min", "median"};
  for (int f = 0; f < 4; ++f) {
    for (int lay = 0; lay < 2; ++lay) {  // 0=C4, 1=AC4
      for (int bord = 0; bord < 2; ++bord) {  // 0=plain, 1=border repl
        char tag[64];
        snprintf(tag, 64, "%s 8u %s %s", fams[f], lay ? "AC4" : "C4",
                 bord ? "border" : "plain");
        MultiDst<Npp8u> dst(7, 7, 4);
        NppStatus st;
        if (f == 0 && !lay && !bord)
          st = nppiFilterBox_8u_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
        else if (f == 0 && lay && !bord)
          st = nppiFilterBox_8u_AC4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
        else if (f == 0 && !lay && bord)
          st = nppiFilterBoxBorder_8u_C4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
        else if (f == 0 && lay && bord)
          st = nppiFilterBoxBorder_8u_AC4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
        else if (f == 1 && !lay && !bord)
          st = nppiFilterMax_8u_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
        else if (f == 1 && lay && !bord)
          st = nppiFilterMax_8u_AC4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
        else if (f == 1 && !lay && bord)
          st = nppiFilterMaxBorder_8u_C4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
        else if (f == 1 && lay && bord)
          st = nppiFilterMaxBorder_8u_AC4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
        else if (f == 2 && !lay && !bord)
          st = nppiFilterMin_8u_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
        else if (f == 2 && lay && !bord)
          st = nppiFilterMin_8u_AC4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
        else if (f == 2 && !lay && bord)
          st = nppiFilterMinBorder_8u_C4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
        else if (f == 2 && lay && bord)
          st = nppiFilterMinBorder_8u_AC4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
        else if (f == 3 && !lay && !bord)
          st = nppiFilterMedian_8u_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, g_ctx);
        else if (f == 3 && lay && !bord)
          st = nppiFilterMedian_8u_AC4R_Ctx(ms.src(), ms.stepBytes, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, g_ctx);
        else if (f == 3 && !lay && bord)
          st = nppiFilterMedianBorder_8u_C4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
        else
          st = nppiFilterMedianBorder_8u_AC4R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes, NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, nullptr, NPP_BORDER_REPLICATE, g_ctx);
        printf("%s: %d %s\n", tag, (int)st, sn(st));
        if (st == NPP_NO_ERROR) dst.print(tag);
        fflush(stdout);
      }
    }
  }
  // box C3 plain (all channels filtered)
  {
    MultiSrc<Npp8u> ms3(20, 3, fill);
    ms3.put();
    MultiDst<Npp8u> dst(7, 7, 3);
    NppStatus st = nppiFilterBox_8u_C3R_Ctx(ms3.src(), ms3.stepBytes, dst.p(), dst.stepBytes,
                                            NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, g_ctx);
    printf("box 8u C3 plain: %d %s\n", (int)st, sn(st));
    if (st == NPP_NO_ERROR) dst.print("box 8u C3 plain");
  }
  // boxborder 16s AC4 spot check
  {
    MultiSrc<Npp16s> ms4(20, 4, [](int i, int j, int c) -> Npp16s {
      if (c == 0) return (Npp16s)((3 * i + 5 * j) % 251);
      if (c == 1) return (Npp16s)((i * 2 + j * 11) % 241);
      if (c == 2) return (Npp16s)((i * 13 + j * 5) % 239);
      return (Npp16s)((i + j) & 0xFF);
    });
    ms4.put();
    MultiDst<Npp16s> dst(7, 7, 4);
    NppStatus st = nppiFilterBoxBorder_16s_AC4R_Ctx(
        ms4.src(), ms4.stepBytes, NppiSize{20, 20}, NppiPoint{0, 0}, dst.p(), dst.stepBytes,
        NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, g_ctx);
    printf("boxborder 16s AC4: %d %s\n", (int)st, sn(st));
    if (st == NPP_NO_ERROR) dst.print("boxborder 16s AC4");
  }
  fflush(stdout);
}

// ===========================================================================
// align: dst/src step parity and element-multiple checks
// ===========================================================================
static void align() {
  printf("== align: step checks (mask 1x1, identity) ==\n");
  // 8u C1: ROI 1x1, rowBytes 1
  {
    Img<Npp8u> im(10, 11);
    im.h.assign(40 * 40, 7);
    im.put();
    for (int ds = 1; ds <= 4; ++ds) {
      Dst<Npp8u> dst(1, 1);
      char nm[32];
      snprintf(nm, 32, "box 8u C1 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_8u_C1R_Ctx(im.src(), im.stepBytes, (Npp8u *)dst.d.p, ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
    // srcstep odd
    Dst<Npp8u> dst(1, 1);
    CK("box 8u C1 srcstep=39", nppiFilterBox_8u_C1R_Ctx(im.src(), 39, (Npp8u *)dst.d.p, dst.stepBytes, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
  }
  // 16u C1: ROI 1x1, rowBytes 2
  {
    Img<Npp16u> im(10, 11);
    im.h.assign(40 * 40, 7);
    im.put();
    for (int ds = 1; ds <= 5; ++ds) {
      Dst<Npp16u> dst(1, 1);
      char nm[32];
      snprintf(nm, 32, "box 16u C1 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
    Dst<Npp16u> dst(1, 1);
    CK("box 16u C1 srcstep=39", nppiFilterBox_16u_C1R_Ctx(im.src(), 39, (Npp16u *)dst.d.p, dst.stepBytes, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    // ROI 3x3 with odd dstep (rowBytes 6): dstep 7 -> rows overlap but valid unsigned
    Dst<Npp16u> dst3(3, 3);
    CK("box 16u C1 roi3x3 dstep=7", nppiFilterBox_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst3.d.p, 7, NppiSize{3, 3}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
  }
  // 32f C1: ROI 1x1, rowBytes 4
  {
    Img<Npp32f> im(10, 11);
    im.h.assign(40 * 40, 7.f);
    im.put();
    for (int ds = 1; ds <= 8; ++ds) {
      Dst<Npp32f> dst(1, 1);
      char nm[32];
      snprintf(nm, 32, "box 32f C1 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
  }
  // 8u C3: ROI 1x1, rowBytes 3
  {
    MultiSrc<Npp8u> ms(20, 3, [](int, int, int) -> Npp8u { return 7; });
    ms.put();
    for (int ds = 1; ds <= 6; ++ds) {
      MultiDst<Npp8u> dst(1, 1, 3);
      char nm[32];
      snprintf(nm, 32, "box 8u C3 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_8u_C3R_Ctx(ms.src(), ms.stepBytes, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
  }
  // 16u C3: rowBytes 6
  {
    MultiSrc<Npp16u> ms(20, 3, [](int, int, int) -> Npp16u { return 7; });
    ms.put();
    for (int ds = 5; ds <= 10; ++ds) {
      MultiDst<Npp16u> dst(1, 1, 3);
      char nm[32];
      snprintf(nm, 32, "box 16u C3 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_16u_C3R_Ctx(ms.src(), ms.stepBytes, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
  }
  // 32f C3: rowBytes 12
  {
    MultiSrc<Npp32f> ms(20, 3, [](int, int, int) -> Npp32f { return 7.f; });
    ms.put();
    for (int ds = 11; ds <= 16; ++ds) {
      MultiDst<Npp32f> dst(1, 1, 3);
      char nm[32];
      snprintf(nm, 32, "box 32f C3 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_32f_C3R_Ctx(ms.src(), ms.stepBytes, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
  }
  // 8u C4 / AC4: rowBytes 4
  {
    MultiSrc<Npp8u> ms(20, 4, [](int, int, int) -> Npp8u { return 7; });
    ms.put();
    for (int ds = 3; ds <= 8; ++ds) {
      MultiDst<Npp8u> dst(1, 1, 4);
      char nm[40];
      snprintf(nm, 40, "box 8u C4 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterBox_8u_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
      snprintf(nm, 40, "median 8u C4 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterMedian_8u_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, nullptr, g_ctx));
    }
  }
  // 16s C4: rowBytes 8 (max)
  {
    MultiSrc<Npp16s> ms(20, 4, [](int, int, int) -> Npp16s { return 7; });
    ms.put();
    for (int ds = 7; ds <= 12; ++ds) {
      MultiDst<Npp16s> dst(1, 1, 4);
      char nm[40];
      snprintf(nm, 40, "max 16s C4 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterMax_16s_C4R_Ctx(ms.src(), ms.stepBytes, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, g_ctx));
    }
  }
  // medianborder 32f C3: rowBytes 12
  {
    MultiSrc<Npp32f> ms(20, 3, [](int, int, int) -> Npp32f { return 7.f; });
    ms.put();
    for (int ds = 11; ds <= 16; ++ds) {
      MultiDst<Npp32f> dst(1, 1, 3);
      char nm[48];
      snprintf(nm, 48, "medborder 32f C3 roi1x1 dstep=%d", ds);
      CK(nm, nppiFilterMedianBorder_32f_C3R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{2, 3}, dst.p(), ds, NppiSize{1, 1}, NppiSize{1, 1}, NppiPoint{0, 0}, nullptr, NPP_BORDER_REPLICATE, g_ctx));
    }
  }
  // advanced 8u C1: ROI 4x3 rowBytes 4
  {
    Img<Npp8u> im(10, 11);
    im.h.assign(40 * 40, 7);
    im.put();
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{4, 3}, 1, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    for (int ds = 3; ds <= 9; ++ds) {
      Dst<Npp8u> dst(4, 3);
      char nm[40];
      snprintf(nm, 40, "adv 8u C1 roi4x3 dstep=%d", ds);
      CK(nm, nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, ds, NppiSize{4, 3}, NppiSize{1, 1}, NppiPoint{0, 0}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx));
    }
  }
  // advanced 32f C3: ROI 2x2 rowBytes 24
  {
    MultiSrc<Npp32f> ms(20, 3, [](int, int, int) -> Npp32f { return 7.f; });
    ms.put();
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{2, 2}, 3, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    for (int ds = 23; ds <= 28; ++ds) {
      MultiDst<Npp32f> dst(2, 2, 3);
      char nm[40];
      snprintf(nm, 40, "adv 32f C3 roi2x2 dstep=%d", ds);
      CK(nm, nppiFilterBoxBorderAdvanced_32f_C3R_Ctx(ms.src(), ms.stepBytes, NppiSize{20, 20}, NppiPoint{2, 3}, dst.p(), ds, NppiSize{2, 2}, NppiSize{1, 1}, NppiPoint{0, 0}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx));
    }
  }
  fflush(stdout);
}

// ===========================================================================
// valid: per-dtype validation matrices (16u/16s/32f); negative-mask launches
// are in the separate "risky" section.
// ===========================================================================
template <class T>
static void val_box(const char *tag) {
  printf("== valid box %s ==\n", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  Dst<T> dst(7, 7);
  const int es = (int)sizeof(T), rowB = 7 * es;
  auto call = [&](NppiSize roi, NppiSize m, NppiPoint a, int ds, int ss) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp16u>)
      return nppiFilterBox_16u_C1R_Ctx(im.src(), ss, (Npp16u *)dst.d.p, ds, roi, m, a, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return nppiFilterBox_16s_C1R_Ctx(im.src(), ss, (Npp16s *)dst.d.p, ds, roi, m, a, g_ctx);
    else
      return nppiFilterBox_32f_C1R_Ctx(im.src(), ss, (Npp32f *)dst.d.p, ds, roi, m, a, g_ctx);
  };
  NppiSize r7{7, 7}, m3{3, 3};
  NppiPoint a1{1, 1};
  CK("baseline", call(r7, m3, a1, dst.stepBytes, im.stepBytes));
  CK("roi_zero_w", call(NppiSize{0, 7}, m3, a1, dst.stepBytes, im.stepBytes));
  CK("roi_zero_h", call(NppiSize{7, 0}, m3, a1, dst.stepBytes, im.stepBytes));
  CK("roi_zero_both", call(NppiSize{0, 0}, m3, a1, dst.stepBytes, im.stepBytes));
  CK("roi_neg", call(NppiSize{-3, 7}, m3, a1, dst.stepBytes, im.stepBytes));
  CK("mask_zero", call(r7, NppiSize{0, 0}, a1, dst.stepBytes, im.stepBytes));
  CK("mask_neg", call(r7, NppiSize{-3, 3}, a1, dst.stepBytes, im.stepBytes));
  CK("mask_w0", call(r7, NppiSize{0, 3}, a1, dst.stepBytes, im.stepBytes));
  CK("dstep_small", call(r7, m3, a1, rowB - 1, im.stepBytes));
  CK("dstep_zero", call(r7, m3, a1, 0, im.stepBytes));
  CK("dstep_neg roi1x1", call(NppiSize{1, 1}, m3, a1, -1, im.stepBytes));
  CK("sstep_zero", call(r7, m3, a1, dst.stepBytes, 0));
  CK("sstep_neg", call(r7, m3, a1, dst.stepBytes, -im.stepBytes));
  CK("anchor_gt", call(r7, m3, NppiPoint{3, 1}, dst.stepBytes, im.stepBytes));
  CK("prec_zeroroi_vs_step", call(NppiSize{0, 7}, m3, a1, 0, im.stepBytes));
  CK("prec_maskzero_vs_step", call(r7, NppiSize{0, 0}, a1, 0, im.stepBytes));
  CK("prec_maskneg_vs_zeroroi", call(NppiSize{0, 7}, NppiSize{-3, 3}, a1, dst.stepBytes, im.stepBytes));
  fflush(stdout);
}

template <class T>
static void val_boxborder(const char *tag) {
  printf("== valid boxborder %s ==\n", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  Dst<T> dst(7, 7);
  const int es = (int)sizeof(T), rowB = 7 * es;
  auto call = [&](NppiSize roi, NppiPoint off, NppiSize ss, NppiSize m, NppiPoint a,
                  NppiBorderType b, int ds) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp16u>)
      return nppiFilterBoxBorder_16u_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16u *)dst.d.p, ds, roi, m, a, b, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return nppiFilterBoxBorder_16s_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16s *)dst.d.p, ds, roi, m, a, b, g_ctx);
    else
      return nppiFilterBoxBorder_32f_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp32f *)dst.d.p, ds, roi, m, a, b, g_ctx);
  };
  NppiSize r7{7, 7}, s16{16, 16}, m3{3, 3};
  NppiPoint a1{1, 1}, off{2, 3};
  CK("baseline", call(r7, off, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("roi_zero_w", call(NppiSize{0, 7}, off, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("roi_zero_h", call(NppiSize{7, 0}, off, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("roi_zero_both", call(NppiSize{0, 0}, off, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("border_undef", call(r7, off, s16, m3, a1, NPP_BORDER_UNDEFINED, dst.stepBytes));
  CK("border_const", call(r7, off, s16, m3, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("border_wrap", call(r7, off, s16, m3, a1, NPP_BORDER_WRAP, dst.stepBytes));
  CK("border_mirror", call(r7, off, s16, m3, a1, NPP_BORDER_MIRROR, dst.stepBytes));
  CK("border_5", call(r7, off, s16, m3, a1, (NppiBorderType)5, dst.stepBytes));
  CK("srcsize_zero", call(r7, off, NppiSize{0, 0}, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("srcsize_neg", call(r7, off, NppiSize{-3, 7}, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_oob", call(r7, NppiPoint{20, 3}, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_neg", call(r7, NppiPoint{-1, 3}, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_gt_src", call(r7, NppiPoint{2, 13}, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("mask_zero", call(r7, off, s16, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("dstep_small", call(r7, off, s16, m3, a1, NPP_BORDER_REPLICATE, rowB - 1));
  CK("dstep_zero", call(r7, off, s16, m3, a1, NPP_BORDER_REPLICATE, 0));
  CK("dstep_neg roi1x1", call(NppiSize{1, 1}, off, s16, m3, a1, NPP_BORDER_REPLICATE, -1));
  CK("anchor_gt", call(r7, off, s16, m3, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("prec_zeroroi_vs_border", call(NppiSize{0, 7}, off, s16, m3, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("prec_border_vs_step", call(r7, off, s16, m3, a1, NPP_BORDER_CONSTANT, 0));
  CK("prec_border_vs_offset", call(r7, NppiPoint{20, 3}, s16, m3, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("prec_offset_vs_step", call(r7, NppiPoint{20, 3}, s16, m3, a1, NPP_BORDER_REPLICATE, 0));
  CK("prec_mask_vs_step", call(r7, off, s16, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, 0));
  fflush(stdout);
}

template <class T>
static void val_minmax(const char *tag, bool ismin) {
  printf("== valid %s %s ==\n", ismin ? "min" : "max", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  Dst<T> dst(7, 7);
  const int es = (int)sizeof(T), rowB = 7 * es;
  auto call = [&](NppiSize roi, NppiSize m, NppiPoint a, int ds) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp16u>)
      return ismin ? nppiFilterMin_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, ds, roi, m, a, g_ctx)
                   : nppiFilterMax_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, ds, roi, m, a, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return ismin ? nppiFilterMin_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, ds, roi, m, a, g_ctx)
                   : nppiFilterMax_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, ds, roi, m, a, g_ctx);
    else
      return ismin ? nppiFilterMin_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, ds, roi, m, a, g_ctx)
                   : nppiFilterMax_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, ds, roi, m, a, g_ctx);
  };
  NppiSize r7{7, 7}, m3{3, 3};
  NppiPoint a1{1, 1};
  CK("baseline", call(r7, m3, a1, dst.stepBytes));
  CK("roi_zero_w", call(NppiSize{0, 7}, m3, a1, dst.stepBytes));
  CK("roi_zero_h", call(NppiSize{7, 0}, m3, a1, dst.stepBytes));
  CK("roi_zero_both", call(NppiSize{0, 0}, m3, a1, dst.stepBytes));
  CK("mask_zero", call(r7, NppiSize{0, 0}, a1, dst.stepBytes));
  CK("mask_w0", call(r7, NppiSize{0, 3}, a1, dst.stepBytes));
  CK("dstep_small", call(r7, m3, a1, rowB - 1));
  CK("dstep_zero", call(r7, m3, a1, 0));
  CK("dstep_neg roi1x1", call(NppiSize{1, 1}, m3, a1, -1));
  CK("anchor_gt", call(r7, m3, NppiPoint{3, 1}, dst.stepBytes));
  CK("prec_zeroroi_vs_step", call(NppiSize{0, 7}, m3, a1, 0));
  CK("prec_maskzero_vs_step", call(r7, NppiSize{0, 0}, a1, 0));
  fflush(stdout);
}

template <class T>
static void val_minmaxborder(const char *tag, bool ismin) {
  printf("== valid %s %s ==\n", ismin ? "minborder" : "maxborder", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  Dst<T> dst(7, 7);
  const int es = (int)sizeof(T), rowB = 7 * es;
  auto call = [&](NppiSize roi, NppiPoint off, NppiSize ss, NppiSize m, NppiPoint a,
                  NppiBorderType b, int ds) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp16u>)
      return ismin ? nppiFilterMinBorder_16u_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16u *)dst.d.p, ds, roi, m, a, b, g_ctx)
                   : nppiFilterMaxBorder_16u_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16u *)dst.d.p, ds, roi, m, a, b, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return ismin ? nppiFilterMinBorder_16s_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16s *)dst.d.p, ds, roi, m, a, b, g_ctx)
                   : nppiFilterMaxBorder_16s_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16s *)dst.d.p, ds, roi, m, a, b, g_ctx);
    else
      return ismin ? nppiFilterMinBorder_32f_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp32f *)dst.d.p, ds, roi, m, a, b, g_ctx)
                   : nppiFilterMaxBorder_32f_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp32f *)dst.d.p, ds, roi, m, a, b, g_ctx);
  };
  NppiSize r7{7, 7}, s16{16, 16}, m3{3, 3};
  NppiPoint a1{1, 1}, off{2, 3};
  CK("baseline", call(r7, off, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("border_undef", call(r7, off, s16, m3, a1, NPP_BORDER_UNDEFINED, dst.stepBytes));
  CK("border_mirror", call(r7, off, s16, m3, a1, NPP_BORDER_MIRROR, dst.stepBytes));
  CK("border_5", call(r7, off, s16, m3, a1, (NppiBorderType)5, dst.stepBytes));
  CK("roi_zero_w", call(NppiSize{0, 7}, off, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("roi_zero + badborder", call(NppiSize{0, 7}, off, s16, m3, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("srcsize_zero", call(r7, off, NppiSize{0, 0}, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("srcsize_neg", call(r7, off, NppiSize{-3, 7}, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_oob", call(r7, NppiPoint{20, 3}, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_neg", call(r7, NppiPoint{-1, 3}, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_gt_src", call(r7, NppiPoint{2, 13}, s16, m3, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("mask_zero", call(r7, off, s16, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("mask_zero + badborder", call(r7, off, s16, NppiSize{0, 0}, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("dstep_small", call(r7, off, s16, m3, a1, NPP_BORDER_REPLICATE, rowB - 1));
  CK("dstep_zero", call(r7, off, s16, m3, a1, NPP_BORDER_REPLICATE, 0));
  CK("dstep_neg roi1x1", call(NppiSize{1, 1}, off, s16, m3, a1, NPP_BORDER_REPLICATE, -1));
  CK("anchor_gt", call(r7, off, s16, m3, NppiPoint{3, 1}, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("prec_zeroroi_vs_border", call(NppiSize{0, 7}, off, s16, m3, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("prec_border_vs_step", call(r7, off, s16, m3, a1, NPP_BORDER_CONSTANT, 0));
  CK("prec_border_vs_offset", call(r7, NppiPoint{20, 3}, s16, m3, a1, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("prec_mask_vs_step", call(r7, off, s16, NppiSize{0, 0}, a1, NPP_BORDER_REPLICATE, 0));
  CK("prec_offset_vs_step", call(r7, NppiPoint{20, 3}, s16, m3, a1, NPP_BORDER_REPLICATE, 0));
  fflush(stdout);
}

template <class T>
static void val_median(const char *tag) {
  printf("== valid median %s ==\n", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  Dst<T> dst(7, 7);
  const int es = (int)sizeof(T), rowB = 7 * es;
  auto call = [&](NppiSize roi, NppiSize m, NppiPoint a, int ds, void *pb) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp16u>)
      return nppiFilterMedian_16u_C1R_Ctx(im.src(), im.stepBytes, (Npp16u *)dst.d.p, ds, roi, m, a, (Npp8u *)pb, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return nppiFilterMedian_16s_C1R_Ctx(im.src(), im.stepBytes, (Npp16s *)dst.d.p, ds, roi, m, a, (Npp8u *)pb, g_ctx);
    else
      return nppiFilterMedian_32f_C1R_Ctx(im.src(), im.stepBytes, (Npp32f *)dst.d.p, ds, roi, m, a, (Npp8u *)pb, g_ctx);
  };
  NppiSize r7{7, 7}, m3{3, 3};
  NppiPoint a1{1, 1};
  CK("baseline", call(r7, m3, a1, dst.stepBytes, nullptr));
  CK("roi_zero_w", call(NppiSize{0, 7}, m3, a1, dst.stepBytes, nullptr));
  CK("roi_zero_h", call(NppiSize{7, 0}, m3, a1, dst.stepBytes, nullptr));
  CK("roi_zero_both", call(NppiSize{0, 0}, m3, a1, dst.stepBytes, nullptr));
  CK("roi_neg", call(NppiSize{-3, 7}, m3, a1, dst.stepBytes, nullptr));
  CK("mask_zero", call(r7, NppiSize{0, 0}, a1, dst.stepBytes, nullptr));
  CK("mask_w0", call(r7, NppiSize{0, 3}, a1, dst.stepBytes, nullptr));
  CK("dstep_small", call(r7, m3, a1, rowB - 1, nullptr));
  CK("dstep_zero", call(r7, m3, a1, 0, nullptr));
  CK("dstep_neg roi1x1", call(NppiSize{1, 1}, m3, a1, -1, nullptr));
  CK("null_buf", call(r7, m3, a1, dst.stepBytes, nullptr));
  CK("anchor_gt", call(r7, m3, NppiPoint{3, 1}, dst.stepBytes, nullptr));
  CK("prec_zeroroi_vs_step", call(NppiSize{0, 7}, m3, a1, 0, nullptr));
  CK("prec_maskzero_vs_step", call(r7, NppiSize{0, 0}, a1, 0, nullptr));
  fflush(stdout);
}

template <class T>
static void val_medianborder(const char *tag) {
  printf("== valid medianborder %s ==\n", tag);
  Img<T> im(10, 11);
  im.h.assign((size_t)40 * 40, T(7));
  im.put();
  Dst<T> dst(7, 7);
  const int es = (int)sizeof(T), rowB = 7 * es;
  auto call = [&](NppiSize roi, NppiPoint off, NppiSize ss, NppiSize m, NppiPoint a, void *pb,
                  NppiBorderType b, int ds) -> NppStatus {
    if constexpr (std::is_same_v<T, Npp16u>)
      return nppiFilterMedianBorder_16u_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16u *)dst.d.p, ds, roi, m, a, (Npp8u *)pb, b, g_ctx);
    else if constexpr (std::is_same_v<T, Npp16s>)
      return nppiFilterMedianBorder_16s_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp16s *)dst.d.p, ds, roi, m, a, (Npp8u *)pb, b, g_ctx);
    else
      return nppiFilterMedianBorder_32f_C1R_Ctx(im.src(), im.stepBytes, ss, off, (Npp32f *)dst.d.p, ds, roi, m, a, (Npp8u *)pb, b, g_ctx);
  };
  NppiSize r7{7, 7}, s16{16, 16}, m3{3, 3};
  NppiPoint a1{1, 1}, off{2, 3};
  CK("baseline", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("border_undef", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_UNDEFINED, dst.stepBytes));
  CK("border_mirror", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_MIRROR, dst.stepBytes));
  CK("border_5", call(r7, off, s16, m3, a1, nullptr, (NppiBorderType)5, dst.stepBytes));
  CK("roi_zero_w", call(NppiSize{0, 7}, off, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("roi_zero + badborder", call(NppiSize{0, 7}, off, s16, m3, a1, nullptr, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("srcsize_zero", call(r7, off, NppiSize{0, 0}, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("srcsize_neg", call(r7, off, NppiSize{-3, 7}, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_oob", call(r7, NppiPoint{20, 3}, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_neg", call(r7, NppiPoint{-1, 3}, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("offset_gt_src", call(r7, NppiPoint{2, 13}, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("mask_zero", call(r7, off, s16, NppiSize{0, 0}, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("mask_zero + badborder", call(r7, off, s16, NppiSize{0, 0}, a1, nullptr, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("dstep_small", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, rowB - 1));
  CK("dstep_zero", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, 0));
  CK("dstep_neg roi1x1", call(NppiSize{1, 1}, off, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, -1));
  CK("null_buf", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("anchor_gt", call(r7, off, s16, m3, NppiPoint{3, 1}, nullptr, NPP_BORDER_REPLICATE, dst.stepBytes));
  CK("prec_zeroroi_vs_border", call(NppiSize{0, 7}, off, s16, m3, a1, nullptr, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("prec_border_vs_step", call(r7, off, s16, m3, a1, nullptr, NPP_BORDER_CONSTANT, 0));
  CK("prec_border_vs_offset", call(r7, NppiPoint{20, 3}, s16, m3, a1, nullptr, NPP_BORDER_CONSTANT, dst.stepBytes));
  CK("prec_mask_vs_step", call(r7, off, s16, NppiSize{0, 0}, a1, nullptr, NPP_BORDER_REPLICATE, 0));
  CK("prec_mask_vs_badborder", call(r7, off, s16, NppiSize{0, 0}, a1, nullptr, NPP_BORDER_CONSTANT, dst.stepBytes));
  fflush(stdout);
}

static void valid() {
  printf("== valid: per-dtype validation ==\n");
  val_box<Npp16u>("16u");
  val_box<Npp16s>("16s");
  val_box<Npp32f>("32f");
  val_boxborder<Npp16u>("16u");
  val_boxborder<Npp32f>("32f");
  val_minmax<Npp16u>("16u", false);
  val_minmax<Npp16s>("16s", false);
  val_minmax<Npp32f>("32f", false);
  val_minmax<Npp16u>("16u", true);
  val_minmaxborder<Npp16u>("16u", false);
  val_minmaxborder<Npp16s>("16s", false);
  val_minmaxborder<Npp32f>("32f", false);
  val_minmaxborder<Npp16u>("16u", true);
  val_median<Npp16u>("16u");
  val_median<Npp16s>("16s");
  val_median<Npp32f>("32f");
  val_medianborder<Npp16u>("16u");
  val_medianborder<Npp16s>("16s");
  val_medianborder<Npp32f>("32f");
  fflush(stdout);
}

int main(int argc, char **argv) {
  init_ctx();
  const char *sec = argc > 1 ? argv[1] : "all";
  if (!strcmp(sec, "advimp") || !strcmp(sec, "all")) advimp();
  if (!strcmp(sec, "advimp32") || !strcmp(sec, "all")) advimp32();
  if (!strcmp(sec, "advstep") || !strcmp(sec, "all")) advstep();
  if (!strcmp(sec, "mirror") || !strcmp(sec, "all")) mirror();
  if (!strcmp(sec, "median") || !strcmp(sec, "all")) median();
  if (!strcmp(sec, "bufsz") || !strcmp(sec, "all")) bufsz();
  if (!strcmp(sec, "chan") || !strcmp(sec, "all")) chan();
  if (!strcmp(sec, "align") || !strcmp(sec, "all")) align();
  if (!strcmp(sec, "valid") || !strcmp(sec, "all")) valid();
  if (!strcmp(sec, "risky")) risky();
  printf("== probe_f9b done ==\n");
  return 0;
}
