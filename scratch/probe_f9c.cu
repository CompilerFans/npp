// NVIDIA NPP 13.4 (RTX 5090) advanced-box investigation probe (package f9).
// Sections by argv[1]: big, rmap, stab, gradchk
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"

extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppSetStream(cudaStream_t hStream);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
  size_t n = 0;
  ~DevBuf() { if (p) cudaFree(p); }
  void alloc(size_t bytes) {
    if (p) cudaFree(p);
    p = nullptr;
    if (bytes) cudaMalloc(&p, bytes);
    n = bytes;
  }
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
    if (d.p) cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  const T *src() const { return (const T *)d.p + (size_t)py * aw + px; }
  T &alloc_at(int ax, int ay) { return h[(size_t)ay * aw + ax]; }
};

template <class T>
struct Dst {
  DevBuf d;
  int W, H, stepBytes;
  std::vector<T> canary;
  Dst(int W_, int H_) : W(W_), H(H_), stepBytes(W_ * (int)sizeof(T)) {
    canary.assign((size_t)W * H, T(0xAB));
  }
  void fill(unsigned char v) {
    d.alloc((size_t)W * H * sizeof(T));
    if (d.p) cudaMemset(d.p, v, (size_t)W * H * sizeof(T));
  }
  void put() {
    d.alloc((size_t)W * H * sizeof(T));
    if (d.p) cudaMemcpy(d.p, canary.data(), canary.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  std::vector<T> region() {
    std::vector<T> r((size_t)W * H);
    if (d.p) cudaMemcpy(r.data(), d.p, r.size() * sizeof(T), cudaMemcpyDeviceToHost);
    return r;
  }
  void print(const char *tag) {
    std::vector<T> r = region();
    printf("%s:\n", tag);
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("%4d ", (int)r[(size_t)y * W + x]);
      printf("\n");
    }
  }
};

static int rep(int i, int lo, int hi) { return i < lo ? lo : (i > hi ? hi : i); }

// host plain-replicate box model over a host image
static int host_box(const std::vector<Npp8u> &h, int aw, int px, int py, int sw, int sh,
                    int offx, int offy, int x, int y, int mw, int mh, int ax, int ay) {
  int s = 0;
  for (int dx = 0; dx < mw; ++dx)
    for (int dy = 0; dy < mh; ++dy) {
      int sx = rep(offx + x - ax + dx, 0, sw - 1);
      int sy = rep(offy + y - ay + dy, 0, sh - 1);
      s += (int)h[(size_t)(py + sy) * aw + px + sx];
    }
  return s / (mw * mh);
}

// gradient sanity case: compares advanced output to plain-replicate host model
static void big_case(const char *tag, int offx, int offy, int sw, int sh, int W, int H,
                     int mw, int mh, int ax, int ay) {
  printf("-- %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) --\n", tag, offx, offy,
         sw, sh, W, H, mw, mh, ax, ay);
  Img<Npp8u> im(8 + offx, 8 + offy, sw + 16, sh + 16);
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((3 * i + 5 * j) % 251);
  im.put();
  Dst<Npp8u> dst(W, H);
  dst.fill(0x00);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  cudaMemset(buf.p, 0x00, (size_t)bsz);
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
      dst.stepBytes, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE,
      (Npp8u *)buf.p, g_ctx);
  printf("status %d %s (bufsz=%d)\n", (int)st, sn(st));
  if (st != NPP_NO_ERROR) return;
  std::vector<Npp8u> r = dst.region();
  int mism = 0;
  std::string ex;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int e = host_box(im.h, im.aw, 8, 8, sw, sh, offx, offy, x, y, mw, mh, ax, ay);
      int g = (int)r[(size_t)y * W + x];
      if (g != e) {
        ++mism;
        if (mism <= 4 && ex.size() < 200) {
          char b[96];
          snprintf(b, sizeof b, " (%d,%d)got=%d exp=%d", x, y, g, e);
          ex += b;
        }
      }
    }
  printf("mismatches %d/%d%s\n", mism, W * H, ex.c_str());
  fflush(stdout);
}

// interior impulse landing test
static void imp_case(const char *tag, int offx, int offy, int sw, int sh, int W, int H,
                     int mw, int mh, int ax, int ay, int ix, int iy, int val, int prefill) {
  printf("-- %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) imp=(%d,%d)=%d pf=%02x --\n",
         tag, offx, offy, sw, sh, W, H, mw, mh, ax, ay, ix, iy, val, prefill);
  Img<Npp8u> im(8 + offx, 8 + offy, sw + 16, sh + 16);
  im.alloc_at(8 + ix, 8 + iy) = (Npp8u)val;
  im.put();
  Dst<Npp8u> dst(W, H);
  dst.fill(0x00);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  cudaMemset(buf.p, prefill, (size_t)bsz);
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
      im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
      dst.stepBytes, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE,
      (Npp8u *)buf.p, g_ctx);
  printf("status %d %s (bufsz=%d)\n", (int)st, sn(st));
  if (st != NPP_NO_ERROR) return;
  std::vector<Npp8u> r = dst.region();
  std::string s;
  int nz = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (r[(size_t)y * W + x]) {
        ++nz;
        if (s.size() < 400) {
          char b[64];
          snprintf(b, sizeof b, " (%d,%d)=%d", x, y, (int)r[(size_t)y * W + x]);
          s += b;
        }
      }
  printf("nonzero %d:%s\n", nz, s.c_str());
  fflush(stdout);
}

static void big() {
  printf("== big: advanced vs plain-replicate model, gradient image ==\n");
  big_case("B1 roi7 off0", 0, 0, 16, 16, 7, 7, 3, 3, 1, 1);
  big_case("B2 roi7 off23", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1);
  big_case("B3 roi8 off33", 3, 3, 20, 20, 8, 8, 3, 3, 1, 1);
  big_case("B4 roi9 off44", 4, 4, 24, 24, 9, 9, 3, 3, 1, 1);
  big_case("B5 roi12 off44", 4, 4, 32, 32, 12, 12, 3, 3, 1, 1);
  big_case("B6 roi16 off44", 4, 4, 36, 36, 16, 16, 3, 3, 1, 1);
  big_case("B7 roi16 off88", 8, 8, 48, 48, 16, 16, 3, 3, 1, 1);
  big_case("B8 roi32 off88", 8, 8, 64, 64, 32, 32, 3, 3, 1, 1);
  big_case("B9 roi48 off88", 8, 8, 64, 64, 48, 48, 3, 3, 1, 1);
  big_case("B10 roi64 off88", 8, 8, 96, 96, 64, 64, 3, 3, 1, 1);
  big_case("B11 roi64 m5x5", 8, 8, 96, 96, 64, 64, 5, 5, 2, 2);
  big_case("B12 roi60x30", 8, 8, 96, 96, 60, 30, 3, 3, 1, 1);
  big_case("B13 roi7 off23 m5x3", 2, 3, 16, 16, 7, 7, 5, 3, 2, 1);
  printf("== big: interior impulse landing ==\n");
  imp_case("I1 roi7 off23", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5, 90, 0x00);
  imp_case("I2 roi16 off44", 4, 4, 36, 36, 16, 16, 3, 3, 1, 1, 11, 11, 90, 0x00);
  imp_case("I3 roi32 off88", 8, 8, 64, 64, 32, 32, 3, 3, 1, 1, 23, 23, 90, 0x00);
  imp_case("I4 roi48 off88", 8, 8, 64, 64, 48, 48, 3, 3, 1, 1, 31, 31, 90, 0x00);
  imp_case("I5 roi64 off88", 8, 8, 96, 96, 64, 64, 3, 3, 1, 1, 39, 39, 90, 0x00);
  imp_case("I6 roi64 m5x5", 8, 8, 96, 96, 64, 64, 5, 5, 2, 2, 39, 39, 250, 0x00);
  imp_case("I7 roi16 off0", 0, 0, 36, 36, 16, 16, 3, 3, 1, 1, 8, 8, 90, 0x00);
}

// full single-impulse response map
static void rmap_case(const char *tag, int offx, int offy, int sw, int sh, int W, int H,
                      int mw, int mh, int ax, int ay, int val, int prefill) {
  printf("== rmap %s: off=(%d,%d) src=%dx%d roi=%dx%d mask=%dx%d a=(%d,%d) val=%d pf=%02x ==\n",
         tag, offx, offy, sw, sh, W, H, mw, mh, ax, ay, val, prefill);
  Img<Npp8u> im(8 + offx, 8 + offy, sw + 16, sh + 16);
  Dst<Npp8u> dst(W, H);
  int bsz = 0;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  DevBuf buf;
  buf.alloc((size_t)bsz);
  int nz_cases = 0;
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i) {
      memset(im.h.data(), 0, im.h.size() * sizeof(Npp8u));
      im.alloc_at(8 + i, 8 + j) = (Npp8u)val;
      im.put();
      dst.fill(0x00);
      cudaMemset(buf.p, prefill, (size_t)bsz);
      NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
          im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
          dst.stepBytes, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE,
          (Npp8u *)buf.p, g_ctx);
      if (st != NPP_NO_ERROR) {
        printf("  imp(%d,%d): status %d\n", i, j, (int)st);
        continue;
      }
      std::vector<Npp8u> r = dst.region();
      std::string s;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          if (r[(size_t)y * W + x]) {
            char b[64];
            snprintf(b, sizeof b, "(%d,%d)=%d ", x, y, (int)r[(size_t)y * W + x]);
            s += b;
          }
      if (!s.empty()) {
        ++nz_cases;
        printf("  imp(%d,%d): %s\n", i, j, s.c_str());
      }
    }
  printf("  landing impulses: %d/%d\n", nz_cases, sw * sh);
  fflush(stdout);
}

static void rmap() {
  rmap_case("R1 off23", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 90, 0x00);
  rmap_case("R2 off0", 0, 0, 16, 16, 7, 7, 3, 3, 1, 1, 90, 0x00);
  rmap_case("R3 off23 pfFF", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 90, 0xFF);
}

static void stab() {
  printf("== stab: buffer-state dependence ==\n");
  // same case, different prefills
  imp_case("S1 pf00", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5, 90, 0x00);
  imp_case("S2 pfFF", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5, 90, 0xFF);
  imp_case("S3 pf5A", 2, 3, 16, 16, 7, 7, 3, 3, 1, 1, 5, 5, 90, 0x5A);
  // gradient case with different prefills (full print)
  for (int pf = 0; pf < 256; pf += 85) {
    printf("-- G pf=%02x --\n", pf);
    Img<Npp8u> im(8 + 2, 8 + 3, 32, 32);
    for (int j = 0; j < 16; ++j)
      for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((3 * i + 5 * j) % 251);
    im.put();
    Dst<Npp8u> dst(7, 7);
    dst.fill(0x00);
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{7, 7}, 1, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    cudaMemset(buf.p, pf, (size_t)bsz);
    NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
        im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, dst.stepBytes,
        NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx);
    printf("status %d %s\n", (int)st, sn(st));
    if (st == NPP_NO_ERROR) dst.print("dst");
  }
  // linearity: two impulses vs sum of singles (both pf00)
  {
    int offx = 2, offy = 3, sw = 16, sh = 16, W = 7, H = 7;
    Img<Npp8u> im(8 + offx, 8 + offy, sw + 16, sh + 16);
    Dst<Npp8u> dst(W, H);
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    std::vector<std::vector<Npp8u>> singles;
    int pos[2][2] = {{5, 5}, {9, 2}};
    for (int k = 0; k < 2; ++k) {
      memset(im.h.data(), 0, im.h.size() * sizeof(Npp8u));
      im.alloc_at(8 + pos[k][0], 8 + pos[k][1]) = 90;
      im.put();
      dst.fill(0x00);
      cudaMemset(buf.p, 0x00, (size_t)bsz);
      nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
          im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
          dst.stepBytes, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE,
          (Npp8u *)buf.p, g_ctx);
      singles.push_back(dst.region());
    }
    memset(im.h.data(), 0, im.h.size() * sizeof(Npp8u));
    im.alloc_at(8 + pos[0][0], 8 + pos[0][1]) = 90;
    im.alloc_at(8 + pos[1][0], 8 + pos[1][1]) = 90;
    im.put();
    dst.fill(0x00);
    cudaMemset(buf.p, 0x00, (size_t)bsz);
    NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
        im.src(), im.stepBytes, NppiSize{sw, sh}, NppiPoint{offx, offy}, (Npp8u *)dst.d.p,
        dst.stepBytes, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE,
        (Npp8u *)buf.p, g_ctx);
    printf("-- linearity: status %d --\n", (int)st);
    if (st == NPP_NO_ERROR) {
      std::vector<Npp8u> both = dst.region();
      int mism = 0;
      for (size_t k = 0; k < both.size(); ++k) {
        int sum = (int)singles[0][k] + (int)singles[1][k];
        if (both[k] != (Npp8u)sum) {
          ++mism;
          if (mism <= 6) printf("  (%zu): both=%d sum=%d\n", k, (int)both[k], sum);
        }
      }
      printf("  linearity mismatches: %d/%zu\n", mism, both.size());
    }
  }
  fflush(stdout);
}

static void gradchk() {
  printf("== gradchk: repeat determinism ==\n");
  // run the SAME gradient case 4 times, print each dst; compare
  std::vector<std::vector<Npp8u>> outs;
  for (int rep_i = 0; rep_i < 4; ++rep_i) {
    Img<Npp8u> im(8 + 2, 8 + 3, 32, 32);
    for (int j = 0; j < 16; ++j)
      for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((3 * i + 5 * j) % 251);
    im.put();
    Dst<Npp8u> dst(7, 7);
    dst.fill(0x00);
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{7, 7}, 1, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    cudaMemset(buf.p, 0x00, (size_t)bsz);
    NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
        im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, dst.stepBytes,
        NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx);
    if (st != NPP_NO_ERROR) { printf("  rep %d status %d\n", rep_i, (int)st); return; }
    outs.push_back(dst.region());
  }
  int mism = 0;
  for (size_t k = 0; k < outs[0].size(); ++k)
    for (int r = 1; r < 4; ++r)
      if (outs[r][k] != outs[0][k]) ++mism;
  printf("  determinism mismatches across 4 reps: %d\n", mism);
  // now run WITHOUT cudaMemset on the buffer (fresh malloc, no prefill)
  std::vector<std::vector<Npp8u>> outs2;
  for (int rep_i = 0; rep_i < 4; ++rep_i) {
    Img<Npp8u> im(8 + 2, 8 + 3, 32, 32);
    for (int j = 0; j < 16; ++j)
      for (int i = 0; i < 16; ++i) im.alloc_at(8 + i, 8 + j) = (Npp8u)((3 * i + 5 * j) % 251);
    im.put();
    Dst<Npp8u> dst(7, 7);
    dst.fill(0x00);
    int bsz = 0;
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{7, 7}, 1, &bsz);
    DevBuf buf;
    buf.alloc((size_t)bsz);
    NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
        im.src(), im.stepBytes, NppiSize{16, 16}, NppiPoint{2, 3}, (Npp8u *)dst.d.p, dst.stepBytes,
        NppiSize{7, 7}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, (Npp8u *)buf.p, g_ctx);
    if (st != NPP_NO_ERROR) { printf("  norep %d status %d\n", rep_i, (int)st); return; }
    outs2.push_back(dst.region());
  }
  mism = 0;
  for (size_t k = 0; k < outs2[0].size(); ++k) {
    if (outs2[0][k] != outs[0][k]) {
      ++mism;
      if (mism <= 6) printf("  pf00-vs-noprefill (%zu): %d vs %d\n", k, (int)outs2[0][k], (int)outs[0][k]);
    }
    for (int r = 1; r < 4; ++r)
      if (outs2[r][k] != outs2[0][k]) { ++mism; break; }
  }
  printf("  noprefill vs pf00 / self-mismatches: %d\n", mism);
  fflush(stdout);
}

int main(int argc, char **argv) {
  init_ctx();
  const char *sec = argc > 1 ? argv[1] : "all";
  if (!strcmp(sec, "big") || !strcmp(sec, "all")) big();
  if (!strcmp(sec, "rmap") || !strcmp(sec, "all")) rmap();
  if (!strcmp(sec, "stab") || !strcmp(sec, "all")) stab();
  if (!strcmp(sec, "gradchk") || !strcmp(sec, "all")) gradchk();
  printf("== probe_f9c done ==\n");
  return 0;
}
