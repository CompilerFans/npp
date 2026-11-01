// NVIDIA NPP 13.4 (RTX 5090) probe: decode nppiFilterBoxBorderAdvanced.
// Usage: probe_f9m [bufsizes|values|buffers]
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <utility>

static NppStreamContext g_ctx;

static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
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

struct DevBuf {
  void *p = nullptr;
  ~DevBuf() { if (p) cudaFree(p); }
  void alloc(size_t b) { if (p) cudaFree(p); p = nullptr; if (b) cudaMalloc(&p, b); }
};

template <class T>
struct Img {
  std::vector<T> h;
  DevBuf d;
  int px, py, aw, ah, stepBytes;
  Img(int px_, int py_, int aw_, int ah_)
      : h((size_t)aw_ * ah_, T(0)), px(px_), py(py_), aw(aw_), ah(ah_),
        stepBytes(aw_ * (int)sizeof(T)) {}
  void put() {
    d.alloc(h.size() * sizeof(T));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  const T *src() const { return (const T *)d.p + (size_t)py * aw + px; }
  T &at(int i, int j) { return h[(size_t)j * aw + i]; }
};

static int clampv(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Case { int offx, offy, sw, sh, W, H, mw, mh, ax, ay; };

static std::vector<int> host_model(const std::vector<Npp8u> &h, int aw, int px, int py,
                                   const Case &c) {
  std::vector<int> r((size_t)c.W * c.H, -1);
  for (int y = 0; y < c.H; ++y)
    for (int x = 0; x < c.W; ++x) {
      int s = 0;
      for (int j = 0; j < c.mh; ++j)
        for (int i = 0; i < c.mw; ++i) {
          int sx = clampv(c.offx + x - c.ax + i, 0, c.sw - 1);
          int sy = clampv(c.offy + y - c.ay + j, 0, c.sh - 1);
          s += (int)h[(size_t)(py + sy) * aw + px + sx];
        }
      r[(size_t)y * c.W + x] = s / (c.mw * c.mh);
    }
  return r;
}

static void one(const Case &c, int ch, bool nullbuf, int prefill, bool print_dst) {
  Img<Npp8u> im(16 + c.offx, 16 + c.offy, c.sw + 64, c.sh + 64);
  for (int j = 0; j < c.sh; ++j)
    for (int i = 0; i < c.sw; ++i) im.at(16 + i, 16 + j) = (Npp8u)((3 * i + 5 * j) % 251);
  im.put();
  int bsz = -12345;
  NppStatus bst = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{c.W, c.H}, ch, &bsz);
  DevBuf buf;
  buf.alloc((size_t)(bsz > 0 ? bsz : 0));
  if (buf.p && prefill >= 0) cudaMemset(buf.p, prefill, (size_t)bsz);

  int dw = c.W * ch, dstep = dw;
  DevBuf dst;
  dst.alloc((size_t)dw * c.H);
  cudaMemset(dst.p, 0x61, (size_t)dw * c.H);

  NppStatus st;
  if (ch == 1)
    st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(im.src(), im.stepBytes, NppiSize{c.sw, c.sh},
        NppiPoint{c.offx, c.offy}, (Npp8u *)dst.p, dstep, NppiSize{c.W, c.H},
        NppiSize{c.mw, c.mh}, NppiPoint{c.ax, c.ay}, NPP_BORDER_REPLICATE,
        nullbuf ? nullptr : (Npp8u *)buf.p, g_ctx);
  else if (ch == 3)
    st = nppiFilterBoxBorderAdvanced_8u_C3R_Ctx(im.src(), im.stepBytes, NppiSize{c.sw, c.sh},
        NppiPoint{c.offx, c.offy}, (Npp8u *)dst.p, dstep, NppiSize{c.W, c.H},
        NppiSize{c.mw, c.mh}, NppiPoint{c.ax, c.ay}, NPP_BORDER_REPLICATE,
        nullbuf ? nullptr : (Npp8u *)buf.p, g_ctx);
  else
    st = nppiFilterBoxBorderAdvanced_8u_C4R_Ctx(im.src(), im.stepBytes, NppiSize{c.sw, c.sh},
        NppiPoint{c.offx, c.offy}, (Npp8u *)dst.p, dstep, NppiSize{c.W, c.H},
        NppiSize{c.mw, c.mh}, NppiPoint{c.ax, c.ay}, NPP_BORDER_REPLICATE,
        nullbuf ? nullptr : (Npp8u *)buf.p, g_ctx);

  std::vector<Npp8u> out((size_t)dw * c.H, 0);
  cudaMemcpy(out.data(), dst.p, out.size(), cudaMemcpyDeviceToHost);
  // compare model against channel 0 only
  std::vector<int> exp;
  if (ch == 1) exp = host_model(im.h, im.aw, 16, 16, c);
  int mism = 0;
  std::string ex;
  if (ch == 1) {
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x) {
        int g = out[(size_t)y * dw + x], e = exp[(size_t)y * c.W + x];
        if (g != e) {
          ++mism;
          if (mism <= 3) { char b[64]; snprintf(b, sizeof b, " (%d,%d)%d/%d", x, y, g, e); ex += b; }
        }
      }
  }
  printf("  ch%d buffsz=%d(bst=%s) st=%s mism=%d/%d%s\n", ch, bsz, sn(bst), sn(st), mism,
         c.W * c.H, ex.c_str());
  if (print_dst) {
    for (int y = 0; y < c.H && y < 12; ++y) {
      printf("    ");
      for (int x = 0; x < c.W && x < 12; ++x) printf("%4d", (int)out[(size_t)y * dw + x]);
      printf("\n");
    }
  }
  fflush(stdout);
}

static void bufsizes() {
  printf("== GetDeviceBufferSize ==\n");
  for (int ch = 1; ch <= 4; ++ch) {
    int ws[] = {1, 2, 3, 7, 8, 16, 17, 32, 64, 65, 100, 128, 129, 256};
    for (int wi = 0; wi < 14; ++wi) {
      int W = ws[wi], a = -1, b = -1;
      NppStatus s1 = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, W}, ch, &a);
      NppStatus s2 = nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{W, W}, ch, &b);
      printf("  ch%d %dx%d: %s %d | 64:%s %d\n", ch, W, W, sn(s1), a, sn(s2), b);
    }
  }
  std::pair<int,int> whs[] = {{7,3},{3,7},{64,16},{16,64},{1,64}};
  for (int ci = 1; ci <= 3; ci += 2)
    for (int i = 0; i < 5; ++i) {
      int a = -1;
      NppStatus s = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{whs[i].first, whs[i].second}, ci, &a);
      printf("  ch%d %dx%d: %s %d\n", ci, whs[i].first, whs[i].second, sn(s), a);
    }
  fflush(stdout);
}

static void values() {
  printf("== values: gradient image vs replicate box model ==\n");
  struct V { const char *tag; Case c; };
  V v[] = {
      {"roi7 off0 m3", {0, 0, 16, 16, 7, 7, 3, 3, 1, 1}},
      {"roi7 off(2,3) m3", {2, 3, 16, 16, 7, 7, 3, 3, 1, 1}},
      {"roi8 off(3,3) m3", {3, 3, 20, 20, 8, 8, 3, 3, 1, 1}},
      {"roi16 off(4,4) m3", {4, 4, 36, 36, 16, 16, 3, 3, 1, 1}},
      {"roi32 off(8,8) m3", {8, 8, 64, 64, 32, 32, 3, 3, 1, 1}},
      {"roi48 off(8,8) m3", {8, 8, 64, 64, 48, 48, 3, 3, 1, 1}},
      {"roi64 off(8,8) m3", {8, 8, 96, 96, 64, 64, 3, 3, 1, 1}},
      {"roi64 off(8,8) m5", {8, 8, 96, 96, 64, 64, 5, 5, 2, 2}},
      {"roi32 off(8,8) m1", {8, 8, 64, 64, 32, 32, 1, 1, 0, 0}},
      {"roi32 off(8,8) m5x3", {8, 8, 64, 64, 32, 32, 5, 3, 2, 1}},
      {"roi32 off(8,8) m15", {8, 8, 64, 64, 32, 32, 15, 15, 7, 7}},
      {"roi5 off(8,8) m7", {8, 8, 32, 32, 5, 5, 7, 7, 3, 3}},
      {"roi32 off(0,0) m3", {0, 0, 32, 32, 32, 32, 3, 3, 1, 1}},
      {"roi32 sw=W m3", {0, 0, 32, 32, 32, 32, 3, 3, 1, 1}},
      {"roi60x30 off(8,8) m3", {8, 8, 96, 96, 60, 30, 3, 3, 1, 1}},
  };
  for (int i = 0; i < 15; ++i) {
    const V &e = v[i];
    printf("-- %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) --\n", e.tag,
           e.c.offx, e.c.offy, e.c.sw, e.c.sh, e.c.W, e.c.H, e.c.mw, e.c.mh, e.c.ax, e.c.ay);
    one(e.c, 1, false, 0, false);
    one(e.c, 3, false, 0, false);
    one(e.c, 4, false, 0, false);
  }
  fflush(stdout);
}

static void buffers() {
  printf("== buffer dependence (roi32 off(8,8) m3) ==\n");
  Case c = {8, 8, 64, 64, 32, 32, 3, 3, 1, 1};
  int pfs[] = {0, 0xff, 0x7f};
  for (int i = 0; i < 3; ++i) {
    printf("-- prefill %02x --\n", pfs[i]);
    one(c, 1, false, pfs[i], true);
  }
  printf("-- null buffer --\n");
  one(c, 1, true, -1, true);
  fflush(stdout);
}

int main(int argc, char **argv) {
  nppGetStreamContext(&g_ctx);
  std::string what = argc > 1 ? argv[1] : "bufsizes";
  if (what == "bufsizes") bufsizes();
  else if (what == "values") values();
  else if (what == "buffers") buffers();
  else printf("unknown\n");
  return 0;
}
