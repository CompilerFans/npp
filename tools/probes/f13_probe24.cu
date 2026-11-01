// f13 probe 24: pyramid DOWN for 8u/16u rounding, C3R layout and the full
// validation matrix (the 32f arithmetic model was settled in probes 15/16).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static NppStreamContext ctx() { NppStreamContext c{}; c.hStream = 0; return c; }

static void run8u(const char *tag, int W, int H, const Npp8u *h, int step, int dw, int dh,
                  float rate, int taps, const float *k, NppiBorderType bt, NppiPoint off,
                  NppiSize srcSize) {
  Npp8u *d, *o;
  cudaMalloc(&d, (size_t)step * H);
  cudaMalloc(&o, (size_t)(dw + 2) * (dh + 2) * 4 * 4);
  cudaMemcpy(d, h, (size_t)step * H, cudaMemcpyHostToDevice);
  cudaMemset(o, 0xEE, (size_t)(dw + 2) * (dh + 2) * 4 * 4);
  NppStreamContext c = ctx();
  NppStatus st = nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
      d, step, srcSize, off, o, dw + 1, NppiSize{dw, dh}, rate, taps, k, bt, c);
  printf("%-26s st=%-4s:", tag, es(st));
  if (st == 0) {
    std::vector<Npp8u> v((size_t)(dw + 1) * dh);
    cudaMemcpy(v.data(), o, v.size(), cudaMemcpyDeviceToHost);
    for (int y = 0; y < dh && y < 4; ++y) {
      printf(" |");
      for (int x = 0; x < dw && x < 12; ++x) printf(" %3d", (int)v[y * (dw + 1) + x]);
    }
  }
  printf("\n");
  cudaFree(d); cudaFree(o);
}

static void run16u(const char *tag, int W, int H, const Npp16u *h, int step, int dw, int dh,
                   float rate, int taps, const float *k, NppiBorderType bt) {
  Npp16u *d, *o;
  cudaMalloc(&d, (size_t)step * H);
  cudaMalloc(&o, (size_t)(dw + 1) * dh * 2);
  cudaMemcpy(d, h, (size_t)step * H, cudaMemcpyHostToDevice);
  cudaMemset(o, 0xEE, (size_t)(dw + 1) * dh * 2);
  NppStreamContext c = ctx();
  NppStatus st = nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
      d, step, NppiSize{W, H}, NppiPoint{0, 0}, o, (dw + 1) * 2, NppiSize{dw, dh}, rate, taps, k,
      bt, c);
  printf("%-26s st=%-4s:", tag, es(st));
  if (st == 0) {
    std::vector<Npp16u> v((size_t)(dw + 1) * dh);
    cudaMemcpy(v.data(), o, v.size() * 2, cudaMemcpyDeviceToHost);
    for (int y = 0; y < dh && y < 4; ++y) {
      printf(" |");
      for (int x = 0; x < dw && x < 12; ++x) printf(" %5d", (int)v[y * (dw + 1) + x]);
    }
  }
  printf("\n");
  cudaFree(d); cudaFree(o);
}

static void run32f(const char *tag, int W, int H, const Npp32f *h, int step, int dw, int dh,
                   float rate, int taps, const float *k, NppiBorderType bt) {
  Npp32f *d, *o;
  cudaMalloc(&d, (size_t)step * H);
  cudaMalloc(&o, (size_t)(dw + 1) * dh * 4);
  cudaMemcpy(d, h, (size_t)step * H, cudaMemcpyHostToDevice);
  cudaMemset(o, 0xEE, (size_t)(dw + 1) * dh * 4);
  NppStreamContext c = ctx();
  NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
      d, step, NppiSize{W, H}, NppiPoint{0, 0}, o, (dw + 1) * 4, NppiSize{dw, dh}, rate, taps, k,
      bt, c);
  printf("%-26s st=%-4s:", tag, es(st));
  if (st == 0) {
    std::vector<Npp32f> v((size_t)(dw + 1) * dh);
    cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
    for (int y = 0; y < dh && y < 4; ++y) {
      printf(" |");
      for (int x = 0; x < dw && x < 12; ++x) printf(" %9.5g", v[y * (dw + 1) + x]);
    }
  }
  printf("\n");
  cudaFree(d); cudaFree(o);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  const float k3[3] = {1.f / 3, 1.f / 3, 1.f / 3};
  const float k125[3] = {0.25f, 0.5f, 0.25f};
  const float k5[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};

  if (part == 1) {
    // 8u / 16u / 32f on the same fractional-friendly source, rate 1.5 and 2.5
    const int W = 12, H = 8, S8 = 16, S16 = 24, S32 = 48;
    std::vector<Npp8u> h8((size_t)S8 * H);
    std::vector<Npp16u> h16((size_t)S16 * H / 2);
    std::vector<Npp32f> h32((size_t)S32 * H / 4);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int v = (x * 7 + y * 13) % 61;
        h8[y * S8 + x] = (Npp8u)v;
        h16[y * (S16 / 2) + x] = (Npp16u)(v * 37);
        h32[y * (S32 / 4) + x] = (Npp32f)(v * 37);
      }
    NppiSize d15{};
    // rate 1.5: dst 8x6 ; rate 2.5: dst 5x4 (from the helper sweep)
    run32f("32f k3 r1.5", W, H, h32.data(), S32, 8, 6, 1.5f, 3, k3, NPP_BORDER_REPLICATE);
    run8u("8u  k3 r1.5", W, H, h8.data(), S8, 8, 6, 1.5f, 3, k3, NPP_BORDER_REPLICATE,
          NppiPoint{0, 0}, NppiSize{W, H});
    run16u("16u k3 r1.5", W, H, h16.data(), S16, 8, 6, 1.5f, 3, k3, NPP_BORDER_REPLICATE);
    run32f("32f k125 r1.5", W, H, h32.data(), S32, 8, 6, 1.5f, 3, k125, NPP_BORDER_REPLICATE);
    run8u("8u  k125 r1.5", W, H, h8.data(), S8, 8, 6, 1.5f, 3, k125, NPP_BORDER_REPLICATE,
          NppiPoint{0, 0}, NppiSize{W, H});
    run16u("16u k125 r1.5", W, H, h16.data(), S16, 8, 6, 1.5f, 3, k125, NPP_BORDER_REPLICATE);
    run32f("32f k5 r2.5", W, H, h32.data(), S32, 5, 4, 2.5f, 5, k5, NPP_BORDER_REPLICATE);
    run8u("8u  k5 r2.5", W, H, h8.data(), S8, 5, 4, 2.5f, 5, k5, NPP_BORDER_REPLICATE,
          NppiPoint{0, 0}, NppiSize{W, H});
    run16u("16u k5 r2.5", W, H, h16.data(), S16, 5, 4, 2.5f, 5, k5, NPP_BORDER_REPLICATE);
  }

  if (part == 2) {
    // C3R: interleaved RGB. Per-channel independent filtering?
    const int W = 12, H = 8, STEP = 12 * 3;
    std::vector<Npp8u> h((size_t)STEP * H, 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        h[y * STEP + x * 3 + 0] = (Npp8u)((x * 7 + y * 13) % 61);
        h[y * STEP + x * 3 + 1] = (Npp8u)((x * 3 + y * 5) % 61);
        h[y * STEP + x * 3 + 2] = (Npp8u)((x + 2 * y) % 61);
      }
    Npp8u *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, (size_t)9 * 6 * 3 * 2);
    cudaMemcpy(d, h.data(), (size_t)STEP * H, cudaMemcpyHostToDevice);
    cudaMemset(o, 0xEE, (size_t)9 * 6 * 3 * 2);
    NppStreamContext c = ctx();
    NppStatus st = nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx(
        d, STEP, NppiSize{W, H}, NppiPoint{0, 0}, o, 9 * 3, NppiSize{8, 6}, 1.5f, 3, k125,
        NPP_BORDER_REPLICATE, c);
    printf("8u C3R st=%s\n", es(st));
    if (st == 0) {
      std::vector<Npp8u> v((size_t)9 * 6 * 3);
      cudaMemcpy(v.data(), o, v.size(), cudaMemcpyDeviceToHost);
      for (int y = 0; y < 2; ++y) {
        printf("  y%d:", y);
        for (int x = 0; x < 8; ++x)
          printf(" (%d,%d,%d)", v[(y * 9 + x) * 3], v[(y * 9 + x) * 3 + 1], v[(y * 9 + x) * 3 + 2]);
        printf("\n");
      }
    }
    cudaFree(d); cudaFree(o);
  }

  if (part == 3) {
    // validation matrix on the 8u C1R variant
    const int W = 12, H = 8, STEP = 16;
    std::vector<Npp8u> h((size_t)STEP * H, 3);
    Npp8u *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 4096);
    cudaMemcpy(d, h.data(), (size_t)STEP * H, cudaMemcpyHostToDevice);
    NppStreamContext c = ctx();
    struct Case { const char *n; const Npp8u *s; Npp8u *dst; int sstep, dstep; NppiSize ss, roi;
                  NppiPoint off; float rate; int taps; const float *k; NppiBorderType bt; };
    Case cs[] = {
      {"ok", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"null src", nullptr, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"null dst", d, nullptr, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"null kern", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, nullptr, NPP_BORDER_REPLICATE},
      {"srcstep 0", d, o, 0, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"srcstep 8", d, o, 8, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"dststep 4", d, o, STEP, 4, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"dststep 8", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"roi 0x0", d, o, STEP, 8, {W, H}, {0, 0}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"roi -1", d, o, STEP, 8, {W, H}, {-1, 4}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"srcsize 0", d, o, STEP, 8, {0, 0}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"off neg", d, o, STEP, 8, {W, H}, {8, 6}, {-1, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"off big", d, o, STEP, 8, {W, H}, {8, 6}, {4, 4}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"roi>src", d, o, STEP, 16, {W, H}, {14, 9}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_REPLICATE},
      {"bord const", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_CONSTANT},
      {"taps 5 k3ptr", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 5, k3, NPP_BORDER_REPLICATE},
      {"taps 1", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 1, k3, NPP_BORDER_REPLICATE},
      {"taps 65", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 65, k3, NPP_BORDER_REPLICATE},
      {"rate 1.0001", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.0001f, 3, k3, NPP_BORDER_REPLICATE},
      {"mirror off0", d, o, STEP, 8, {W, H}, {8, 6}, {0, 0}, 1.5f, 3, k3, NPP_BORDER_MIRROR},
    };
    for (auto &cs2 : cs) {
      cudaError_t e = cudaMemset(o, 0xEE, 4096);
      (void)e;
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
          cs2.s, cs2.sstep, cs2.ss, cs2.off, cs2.dst, cs2.dstep, cs2.roi, cs2.rate, cs2.taps,
          cs2.k, cs2.bt, c);
      printf("%-14s st=%s", cs2.n, es(st));
      cudaError_t se = cudaDeviceSynchronize();
      printf(" sync=%s\n", cudaGetErrorString(se));
    }
    cudaFree(d); cudaFree(o);
  }
  return 0;
}
