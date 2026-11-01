// f13 probe 26: BilateralGaussBorder / ThresholdAdaptiveBoxBorder /
// HarrisCornersBorder value and status probes.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static void dump8(const char *tag, const std::vector<Npp8u> &v, int W, int H, int step) {
  printf("%s", tag);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) printf(" %d", (int)v[y * step + x]);
  printf("\n");
}

static void dump32(const char *tag, const std::vector<Npp32f> &v, int W, int H, int step) {
  printf("%s", tag);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) printf(" %.9g", v[y * step + x]);
  printf("\n");
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  NppStreamContext c{};
  c.hStream = 0;

  if (part == 1) {
    // bilateral 8u C1 on a small crafted image
    const int W = 8, H = 8;
    std::vector<Npp8u> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[y * W + x] = (Npp8u)((x * 11 + y * 23) % 97);
    dump8("src", h, W, H, W);
    Npp8u *d, *o;
    cudaMalloc(&d, W * H);
    cudaMalloc(&o, W * H);
    cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
    struct C { const char *n; int r, step; float vs, ps; NppiBorderType bt; };
    C cs[] = {{"r1 s1 v10 p2", 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
              {"r2 s1 v10 p2", 2, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
              {"r1 s2 v10 p2", 1, 2, 10.f, 2.f, NPP_BORDER_REPLICATE},
              {"r1 s1 v100 p2", 1, 1, 100.f, 2.f, NPP_BORDER_REPLICATE},
              {"r2 s1 v50 p5", 2, 1, 50.f, 5.f, NPP_BORDER_MIRROR}};
    for (auto &x : cs) {
      cudaMemset(o, 0, W * H);
      NppStatus st = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(
          d, W, {W, H}, {0, 0}, o, W, {W, H}, x.r, x.step, x.vs, x.ps, x.bt, c);
      std::vector<Npp8u> v(W * H);
      cudaMemcpy(v.data(), o, W * H, cudaMemcpyDeviceToHost);
      char tag[64];
      snprintf(tag, 64, "bi %s st=%s:", x.n, es(st));
      if (st == 0) dump8(tag, v, W, H, W); else printf("%s\n", tag);
    }
    // 32f C1 with fractional values
    std::vector<Npp32f> hf((size_t)W * H);
    for (int i = 0; i < W * H; ++i) hf[i] = (float)((i * 13) % 53) * 0.25f;
    dump32("src32", hf, W, H, W);
    Npp32f *df, *of;
    cudaMalloc(&df, W * H * 4);
    cudaMalloc(&of, W * H * 4);
    cudaMemcpy(df, hf.data(), W * H * 4, cudaMemcpyHostToDevice);
    cudaMemset(of, 0, W * H * 4);
    NppStatus st = nppiFilterBilateralGaussBorder_32f_C1R_Ctx(df, W * 4, {W, H}, {0, 0}, of,
                                                              W * 4, {W, H}, 1, 1, 10.f, 2.f,
                                                              NPP_BORDER_REPLICATE, c);
    std::vector<Npp32f> vf(W * H);
    cudaMemcpy(vf.data(), of, W * H * 4, cudaMemcpyDeviceToHost);
    char tag[64];
    snprintf(tag, 64, "bi32 r1 s1 st=%s:", es(st));
    if (st == 0) dump32(tag, vf, W, H, W); else printf("%s\n", tag);
    cudaFree(d);
    cudaFree(o);
    cudaFree(df);
    cudaFree(of);
  }

  if (part == 2) {
    // adaptive threshold box
    const int W = 8, H = 8;
    std::vector<Npp8u> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[y * W + x] = (Npp8u)((x * 31 + y * 7) % 101);
    dump8("src", h, W, H, W);
    Npp8u *d, *o;
    cudaMalloc(&d, W * H);
    cudaMalloc(&o, W * H);
    cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
    struct C { const char *n; NppiSize m; float delta; Npp8u gt, le; NppiBorderType bt; };
    C cs[] = {{"m3 d0 gt1 le0", {3, 3}, 0.f, 1, 0, NPP_BORDER_REPLICATE},
              {"m3 d10 gt200 le7", {3, 3}, 10.f, 200, 7, NPP_BORDER_REPLICATE},
              {"m5 d0 gt1 le0", {5, 5}, 0.f, 1, 0, NPP_BORDER_REPLICATE},
              {"m4 d0 gt1 le0", {4, 4}, 0.f, 1, 0, NPP_BORDER_REPLICATE},
              {"m3 d0.5 gt1 le0", {3, 3}, 0.5f, 1, 0, NPP_BORDER_REPLICATE},
              {"m3 d0 gt1 le0 mirror", {3, 3}, 0.f, 1, 0, NPP_BORDER_MIRROR}};
    for (auto &x : cs) {
      cudaMemset(o, 0, W * H);
      NppStatus st = nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(
          d, W, {W, H}, {0, 0}, o, W, {W, H}, x.m, x.delta, x.gt, x.le, x.bt, c);
      std::vector<Npp8u> v(W * H);
      cudaMemcpy(v.data(), o, W * H, cudaMemcpyDeviceToHost);
      char tag[64];
      snprintf(tag, 64, "at %s st=%s:", x.n, es(st));
      if (st == 0) dump8(tag, v, W, H, W); else printf("%s\n", tag);
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 3) {
    // Harris on impulses / borders
    const int W = 12, H = 12;
    std::vector<Npp8u> h((size_t)W * H, 0);
    // a bright square corner at (5,5)
    for (int y = 5; y < 9; ++y)
      for (int x = 5; x < 9; ++x) h[y * W + x] = 200;
    dump8("src", h, W, H, W);
    Npp8u *d;
    Npp32f *o;
    cudaMalloc(&d, W * H);
    cudaMalloc(&o, W * H * 4);
    cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
    Npp8u *buf;
    cudaMalloc(&buf, 1 << 16);
    struct C { const char *n; NppiDifferentialKernel f; NppiMaskSize m, a; float k, sc;
               NppiBorderType bt; };
    C cs[] = {{"sobel3 avg3 k.04 s1", NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f, NPP_BORDER_REPLICATE},
              {"sobel3 avg5 k.04 s1", NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, 0.04f, 1.f, NPP_BORDER_REPLICATE},
              {"scharr3 avg3 k.04 s2", NPP_FILTER_SCHARR, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 2.f, NPP_BORDER_REPLICATE},
              {"sobel5 avg3 k.04 s1", NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f, NPP_BORDER_REPLICATE},
              {"sobel3 avg3 k.04 s1 mirror", NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f, NPP_BORDER_MIRROR}};
    for (auto &x : cs) {
      cudaMemset(o, 0xEE, W * H * 4);
      NppStatus st = nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(d, W, {W, H}, {0, 0}, o, W * 4,
                                                                 {W, H}, x.f, x.m, x.a, x.k, x.sc,
                                                                 x.bt, buf, c);
      std::vector<Npp32f> v(W * H);
      cudaMemcpy(v.data(), o, W * H * 4, cudaMemcpyDeviceToHost);
      char tag[80];
      snprintf(tag, 80, "hc %s st=%s:", x.n, es(st));
      if (st == 0) {
        printf("%s", tag);
        for (int y = 0; y < H; ++y)
          for (int xx = 0; xx < W; ++xx)
            if (v[y * W + xx] != 0.0f && v[y * W + xx] != -0.0f) printf(" (%d,%d)=%.6g", xx, y, v[y * W + xx]);
        printf("\n");
      } else {
        printf("%s\n", tag);
      }
    }
    cudaFree(d);
    cudaFree(o);
    cudaFree(buf);
  }
  return 0;
}
