// f13 probe 15: nppiFilterGaussPyramidLayerDownBorder semantics.
// - DstROI helper formula across rates/sizes
// - identity-kernel sampling map (which source coordinate feeds each dst pixel)
// - arithmetic/rounding with a real 5-tap kernel on 8u/16u/32f
// - status: rate range, taps, border, steps, null
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static void try_dst(int w, int h, float rate) {
  NppiSize d{};
  NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(w, h, &d, rate);
  printf("down dstroi %3dx%3d rate %-6.3f st=%-4s dst=%dx%d\n", w, h, rate, es(st), d.width, d.height);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;

  if (part == 1) {
    // dst ROI formula for down
    try_dst(16, 16, 2.0f);
    try_dst(16, 16, 1.5f);
    try_dst(17, 17, 2.0f);
    try_dst(17, 17, 2.5f);
    try_dst(15, 15, 3.0f);
    try_dst(100, 33, 2.0f);
    try_dst(100, 33, 1.25f);
    try_dst(8, 8, 4.0f);
    try_dst(8, 8, 1.1f);
    try_dst(8, 8, 10.0f);
    try_dst(8, 8, 1.0f);
    try_dst(8, 8, 10.5f);
    try_dst(8, 8, 0.5f);
    try_dst(8, 8, -2.0f);
    try_dst(8, 8, 0.0f);
    try_dst(0, 8, 2.0f);
    try_dst(8, 0, 2.0f);
    try_dst(-8, 8, 2.0f);
    { NppStatus st = nppiGetFilterGaussPyramidLayerDownBorderDstROI(8, 8, nullptr, 2.0f);
      printf("down dstroi null st=%s\n", es(st)); }
  }

  if (part == 2) {
    // identity kernel sampling map, 32f, rate 2: src(x,y) = x + 100*y
    const int W = 16, H = 16;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d; cudaMalloc(&d, W * H * 4); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.0f, 1.0f, 0.0f};
    Npp32f *o; cudaMalloc(&o, 4096); cudaMemset(o, 0xEE, 4096);
    float rates[] = {2.0f, 1.5f, 2.5f, 3.0f, 1.25f, 1.1f, 4.0f};
    for (float r : rates) {
      cudaMemset(o, 0xEE, 4096);
      NppiSize dst{}; nppiGetFilterGaussPyramidLayerDownBorderDstROI(W, H, &dst, r);
      NppStreamContext c{}; c.hStream = 0;
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, r, 3, kern,
          NPP_BORDER_REPLICATE, c);
      printf("rate %-5.2f st=%-4s dst=%dx%d:", r, es(st), dst.width, dst.height);
      if (st == 0) {
        std::vector<Npp32f> v((size_t)dst.width * dst.height);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        for (int y = 0; y < std::min(dst.height, 4); ++y) {
          printf(" |");
          for (int x = 0; x < std::min(dst.width, 6); ++x) printf(" %.4g", v[y * dst.width + x]);
        }
      }
      printf("\n");
    }
  }

  if (part == 3) {
    // arithmetic: 5-tap kernel, rate 2, 32f and 8u; src = ramp + delta
    const int W = 16, H = 16;
    const float k5[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};  // sums to 16, NOT 1
    std::vector<Npp32f> hf((size_t)W * H);
    std::vector<Npp8u> h8((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
      float v = (float)((x * 7 + y * 13) % 251) + ((x + y) % 2 ? 0.5f : 0.0f);
      hf[y * W + x] = v; h8[y * W + x] = (Npp8u)((int)v % 256);
    }
    Npp32f *df; cudaMalloc(&df, 4096); cudaMemcpy(df, hf.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp8u *d8; cudaMalloc(&d8, 4096); cudaMemcpy(d8, h8.data(), W * H, cudaMemcpyHostToDevice);
    Npp32f kern[5]; for (int i = 0; i < 5; ++i) kern[i] = k5[i] / 16.0f;
    Npp32f *of; cudaMalloc(&of, 4096);
    Npp8u *o8; cudaMalloc(&o8, 4096);
    NppiSize dst{}; nppiGetFilterGaussPyramidLayerDownBorderDstROI(W, H, &dst, 2.0f);
    NppStreamContext c{}; c.hStream = 0;
    cudaMemset(of, 0xEE, 4096);
    NppStatus stf = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        df, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, of, dst.width * 4, dst, 2.0f, 5, kern,
        NPP_BORDER_REPLICATE, c);
    cudaMemset(o8, 0xEE, 4096);
    NppStatus st8 = nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
        d8, W, NppiSize{W, H}, NppiPoint{0, 0}, o8, dst.width, dst, 2.0f, 5, kern,
        NPP_BORDER_REPLICATE, c);
    printf("32f st=%s 8u st=%s dst=%dx%d\n", es(stf), es(st8), dst.width, dst.height);
    std::vector<Npp32f> vf((size_t)dst.width * dst.height);
    std::vector<Npp8u> v8((size_t)dst.width * dst.height);
    cudaMemcpy(vf.data(), of, vf.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(v8.data(), o8, v8.size(), cudaMemcpyDeviceToHost);
    for (int y = 0; y < dst.height; ++y) {
      printf("y%d:", y);
      for (int x = 0; x < dst.width; ++x)
        printf(" %g/%d", vf[y * dst.width + x], (int)v8[y * dst.width + x]);
      printf("\n");
    }
  }

  if (part == 4) {
    // statuses: bad rate, bad taps, border types, mirror border values, steps
    const int W = 16, H = 16;
    std::vector<Npp32f> h((size_t)W * H, 1.0f);
    h[0] = 9.0f;  // corner marker for mirror probing
    Npp32f *d; cudaMalloc(&d, 4096); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {0.25f, 0.5f, 0.25f};
    Npp32f *o; cudaMalloc(&o, 4096);
    NppiSize dst{}; nppiGetFilterGaussPyramidLayerDownBorderDstROI(W, H, &dst, 2.0f);
    NppStreamContext c{}; c.hStream = 0;
    struct { float rate; int taps; NppiBorderType bt; int step; const char *tag; } cases[] = {
      {2.0f, 3, NPP_BORDER_REPLICATE, 32, "ok"},
      {1.0f, 3, NPP_BORDER_REPLICATE, 32, "rate1.0"},
      {0.9f, 3, NPP_BORDER_REPLICATE, 32, "rate0.9"},
      {10.1f, 3, NPP_BORDER_REPLICATE, 32, "rate10.1"},
      {10.0f, 3, NPP_BORDER_REPLICATE, 32, "rate10"},
      {-2.0f, 3, NPP_BORDER_REPLICATE, 32, "rateneg"},
      {2.0f, 2, NPP_BORDER_REPLICATE, 32, "taps2"},
      {2.0f, 0, NPP_BORDER_REPLICATE, 32, "taps0"},
      {2.0f, -3, NPP_BORDER_REPLICATE, 32, "taps-3"},
      {2.0f, 4, NPP_BORDER_REPLICATE, 32, "taps4even"},
      {2.0f, 101, NPP_BORDER_REPLICATE, 32, "taps101"},
      {2.0f, 3, NPP_BORDER_MIRROR, 32, "mirror"},
      {2.0f, 3, NPP_BORDER_CONSTANT, 32, "const"},
      {2.0f, 3, NPP_BORDER_REPLICATE, 16, "step16"},
      {2.0f, 3, NPP_BORDER_REPLICATE, 24, "step24"},
      {2.0f, 3, NPP_BORDER_REPLICATE, 31, "step31"},
      {2.0f, 3, NPP_BORDER_MIRROR, 16, "mirror step16"},
    };
    for (auto &t : cases) {
      cudaMemset(o, 0xEE, 4096);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, t.step, dst, t.rate, t.taps,
          t.taps > 0 && t.taps < 100 ? kern : kern, t.bt, c);
      printf("%-14s st=%-4s", t.tag, es(st));
      if (st == 0) {
        float v0, v1; cudaMemcpy(&v0, o, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&v1, (char *)o + 4, 4, cudaMemcpyDeviceToHost);
        printf(" dst0=%.6g dst1=%.6g", v0, v1);
      }
      printf("\n");
    }
    // null kernel / null dst / null src
    NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, 32, dst, 2.0f, 3, nullptr,
        NPP_BORDER_REPLICATE, c);
    printf("nullkern st=%s\n", es(st));
    st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, nullptr, 32, dst, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    printf("nulldst st=%s\n", es(st));
    st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        nullptr, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, 32, dst, 2.0f, 3, kern,
        NPP_BORDER_REPLICATE, c);
    printf("nullsrc st=%s\n", es(st));
  }

  if (part == 5) {
    // mirror border values: 16x16 corner marker, kernel [0,1,0]... use full
    // 3-tap so border matters; src row 0 = 9 only at x=0
    const int W = 8, H = 8;
    std::vector<Npp32f> h((size_t)W * H, 0.0f);
    h[0] = 9.0f;
    Npp32f *d; cudaMalloc(&d, 4096); cudaMemcpy(d, h.data(), W * H * 4, cudaMemcpyHostToDevice);
    Npp32f kern[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
    Npp32f *o; cudaMalloc(&o, 4096);
    NppiSize dst{}; nppiGetFilterGaussPyramidLayerDownBorderDstROI(W, H, &dst, 2.0f);
    NppStreamContext c{}; c.hStream = 0;
    for (int bt = 0; bt < 2; ++bt) {
      cudaMemset(o, 0xEE, 4096);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, W * 4, NppiSize{W, H}, NppiPoint{0, 0}, o, dst.width * 4, dst, 2.0f, 3, kern,
          bt ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE, c);
      printf("%s st=%s:", bt ? "mirror" : "repl", es(st));
      std::vector<Npp32f> v((size_t)dst.width * dst.height);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < v.size(); ++i) printf(" %.5g", v[i]);
      printf("\n");
    }
  }
  return 0;
}
