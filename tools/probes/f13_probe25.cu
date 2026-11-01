// f13 probe 25: down pyramid border value, ordering and offset questions.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static void dump(const char *tag, int W, int H, int px, int py, float val, float rate, int taps,
                 const float *k, NppiBorderType bt, NppiPoint off, NppiSize ss, NppiSize roi) {
  std::vector<Npp32f> h((size_t)W * H, 0.0f);
  if (px >= 0) h[(size_t)py * W + px] = val;
  Npp32f *d, *o;
  cudaMalloc(&d, (size_t)W * H * 4);
  cudaMalloc(&o, 1 << 20);
  cudaMemcpy(d, h.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
  cudaMemset(o, 0, 1 << 20);
  NppStreamContext c{};
  c.hStream = 0;
  NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
      d, W * 4, ss, off, o, roi.width * 4, roi, rate, taps, k, bt, c);
  printf("%-30s st=%-4s:", tag, es(st));
  if (st == 0) {
    std::vector<Npp32f> v((size_t)roi.width * roi.height);
    cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < v.size() && i < 10; ++i) printf(" %.6g", v[i]);
  }
  printf("\n");
  cudaFree(d);
  cudaFree(o);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  const float k3[3] = {1.f / 3, 1.f / 3, 1.f / 3};
  const float k125[3] = {0.25f, 0.5f, 0.25f};

  if (part == 1) {
    dump("8x8 d9(0,0) k3 r2", 8, 8, 0, 0, 9, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {0, 0},
         {8, 8}, {4, 4});
    dump("8x8 d1(0,0) k3 r2", 8, 8, 0, 0, 1, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {0, 0},
         {8, 8}, {4, 4});
    dump("8x8 d9(0,0) k125 r2", 8, 8, 0, 0, 9, 2.0f, 3, k125, NPP_BORDER_REPLICATE, {0, 0},
         {8, 8}, {4, 4});
    dump("12x12 d9(0,0) k3 r2", 12, 12, 0, 0, 9, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {0, 0},
         {12, 12}, {6, 6});
    dump("12x12 d9(0,0) k125 r2", 12, 12, 0, 0, 9, 2.0f, 3, k125, NPP_BORDER_REPLICATE, {0, 0},
         {12, 12}, {6, 6});
    dump("12x12 d9(2,2) k3 r2", 12, 12, 2, 2, 9, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {0, 0},
         {12, 12}, {6, 6});
    dump("12x12 d1(0,0) k3 r2", 12, 12, 0, 0, 1, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {0, 0},
         {12, 12}, {6, 6});
    dump("8x8 d9(0,0) k3 r2 mirror", 8, 8, 0, 0, 9, 2.0f, 3, k3, NPP_BORDER_MIRROR, {0, 0},
         {8, 8}, {4, 4});
    dump("12x12 d9(0,0) k3 r2 mirror", 12, 12, 0, 0, 9, 2.0f, 3, k3, NPP_BORDER_MIRROR, {0, 0},
         {12, 12}, {6, 6});
  }

  if (part == 2) {
    // validation ordering: exactly two faults at a time
    const int W = 12, H = 8, STEP = 48;
    std::vector<Npp32f> h((size_t)W * H, 3.0f);
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 4096);
    cudaMemcpy(d, h.data(), (size_t)STEP * H, cudaMemcpyHostToDevice);
    NppStreamContext c{};
    c.hStream = 0;
    struct C { const char *n; int ss; int ds; NppiSize ss2; NppiPoint off; float rate; int taps;
               const float *k; NppiBorderType bt; NppiSize roi; };
    C cs[] = {
      {"zeroROI+badrate", STEP, 32, {W, H}, {0, 0}, 1.0f, 3, k3, NPP_BORDER_REPLICATE, {0, 0}},
      {"zeroROI+badtaps", STEP, 32, {W, H}, {0, 0}, 2.0f, 1, k3, NPP_BORDER_REPLICATE, {0, 0}},
      {"zeroROI+badborder", STEP, 32, {W, H}, {0, 0}, 2.0f, 3, k3, NPP_BORDER_CONSTANT, {0, 0}},
      {"badrate+badtaps", STEP, 32, {W, H}, {0, 0}, 1.0f, 1, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"badrate+badborder", STEP, 32, {W, H}, {0, 0}, 1.0f, 3, k3, NPP_BORDER_CONSTANT, {8, 6}},
      {"badtaps+badborder", STEP, 32, {W, H}, {0, 0}, 2.0f, 1, k3, NPP_BORDER_CONSTANT, {8, 6}},
      {"badborder+badstep", STEP, 4, {W, H}, {0, 0}, 2.0f, 3, k3, NPP_BORDER_CONSTANT, {8, 6}},
      {"badrate+badstep", STEP, 4, {W, H}, {0, 0}, 1.0f, 3, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"badtaps+badstep", STEP, 4, {W, H}, {0, 0}, 2.0f, 1, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"srcsize0+badrate", STEP, 32, {0, 0}, {0, 0}, 1.0f, 3, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"offneg+badrate", STEP, 32, {W, H}, {-1, 0}, 1.0f, 3, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"offneg+badborder", STEP, 32, {W, H}, {-1, 0}, 2.0f, 3, k3, NPP_BORDER_CONSTANT, {8, 6}},
      {"offeq+badrate", STEP, 32, {W, H}, {W, 0}, 1.0f, 3, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"offeq+badborder", STEP, 32, {W, H}, {W, 0}, 2.0f, 3, k3, NPP_BORDER_CONSTANT, {8, 6}},
      {"offeq ok", STEP, 32, {W, H}, {W - 1, H - 1}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {8, 6}},
      {"negROI+offneg", STEP, 32, {W, H}, {-1, 0}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, {-1, 6}},
    };
    for (auto &x : cs) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, x.ss, x.ss2, x.off, o, x.ds, x.roi, x.rate, x.taps, x.k, x.bt, c);
      printf("%-20s st=%s\n", x.n, es(st));
    }
    // taps sweep
    int taps[] = {3, 4, 5, 6, 7, 8, 9, 15, 63, 65, 127, 129};
    for (int t : taps) {
      std::vector<Npp32f> kk(t > 0 ? t : 1, 1.0f / (t > 0 ? t : 1));
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, STEP, {W, H}, {0, 0}, o, 32, {8, 6}, 2.0f, t, kk.data(), NPP_BORDER_REPLICATE, c);
      printf("taps %-4d st=%s\n", t, es(st));
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 3) {
    // offset semantics: ROI origin shifts the sampling?
    const int W = 12, H = 8, STEP = 48;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 1 << 20);
    cudaMemcpy(d, h.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    NppStreamContext c{};
    c.hStream = 0;
    struct C { const char *n; NppiPoint off; NppiSize ss; } cs[] = {
      {"off0 src12", {0, 0}, {W, H}},
      {"off2 src12", {2, 2}, {W, H}},
      {"off2 src10", {2, 2}, {10, 6}},
      {"off0 src10", {0, 0}, {10, 6}},
    };
    for (auto &x : cs) {
      cudaMemset(o, 0, 1 << 20);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, STEP, x.ss, x.off, o, 4 * 6, {4, 4}, 2.0f, 3, k125, NPP_BORDER_REPLICATE, c);
      printf("%-12s st=%s:", x.n, es(st));
      if (st == 0) {
        std::vector<Npp32f> v(24);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 8; ++i) printf(" %.5g", v[i]);
      }
      printf("\n");
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 4) {
    // full dumps for the offset question: ramp src = x + 100y, k125, rate 2
    const int W = 12, H = 8, STEP = 48;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 1 << 20);
    cudaMemcpy(d, h.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    NppStreamContext c{};
    c.hStream = 0;
    struct C { const char *n; NppiPoint off; NppiSize ss; NppiSize roi; } cs[] = {
      {"off0 12", {0, 0}, {W, H}, {4, 4}},
      {"off2 12", {2, 2}, {W, H}, {4, 4}},
      {"off2 roi0", {2, 2}, {W, H}, {0, 0}},
      {"off2 6x6", {2, 2}, {W, H}, {6, 6}},
      {"off4 12", {4, 4}, {W, H}, {4, 4}},
      {"off0 roi6", {0, 0}, {W, H}, {6, 6}},
    };
    for (auto &x : cs) {
      cudaMemset(o, 0, 1 << 20);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, STEP, x.ss, x.off, o, 8 * 4, x.roi, 2.0f, 3, k125, NPP_BORDER_REPLICATE, c);
      printf("%-10s st=%s:\n", x.n, es(st));
      if (st == 0) {
        std::vector<Npp32f> v(8 * 8);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        int rw = x.roi.width > 0 ? x.roi.width : 1;
        for (int y = 0; y < x.roi.height; ++y) {
          printf("   ");
          for (int xx = 0; xx < rw; ++xx) printf(" %8.4g", v[y * 8 + xx]);
          printf("\n");
        }
      }
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 5) {
    // MIRROR on a ramp and on border deltas
    const int W = 8, H = 8, STEP = 32;
    std::vector<Npp32f> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[y * W + x] = (float)(x + 100 * y);
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 1 << 20);
    cudaMemcpy(d, h.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    NppStreamContext c{};
    c.hStream = 0;
    for (int bt = 0; bt < 2; ++bt) {
      cudaMemset(o, 0, 1 << 20);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, STEP, {W, H}, {0, 0}, o, 4 * 4, {4, 4}, 2.0f, 3, k125,
          bt ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE, c);
      printf("%s ramp st=%s:", bt ? "mirror" : "repl", es(st));
      if (st == 0) {
        std::vector<Npp32f> v(16);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 16; ++i) printf(" %.4g", v[i]);
      }
      printf("\n");
    }
    // right/bottom edge delta
    for (int px : {6, 7}) {
      std::vector<Npp32f> hh((size_t)W * H, 0.0f);
      hh[(size_t)3 * W + px] = 1.0f;
      cudaMemcpy(d, hh.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
      for (int bt = 0; bt < 2; ++bt) {
        cudaMemset(o, 0, 1 << 20);
        NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
            d, STEP, {W, H}, {0, 0}, o, 4 * 4, {4, 4}, 2.0f, 3, k3,
            bt ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE, c);
        printf("px%d %s st=%s:", px, bt ? "mirror" : "repl", es(st));
        if (st == 0) {
          std::vector<Npp32f> v(16);
          cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
          for (int i = 0; i < 16; ++i) printf(" %.4g", v[i]);
        }
        printf("\n");
      }
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 6) {
    // taps upper bound + even taps behaviour
    const int W = 12, H = 8, STEP = 48;
    std::vector<Npp32f> h((size_t)W * H, 1.0f);
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 4096);
    cudaMemcpy(d, h.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    NppStreamContext c{};
    c.hStream = 0;
    for (int t : {64, 65, 66, 67, 96, 97, 100, 101, 120, 121, 125, 126}) {
      std::vector<Npp32f> kk(t > 0 ? t : 1, 1.0f / (t > 0 ? t : 1));
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, STEP, {W, H}, {0, 0}, o, 32, {8, 6}, 2.0f, t, kk.data(), NPP_BORDER_REPLICATE, c);
      printf("taps %-4d st=%s\n", t, es(st));
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 7) {
    // offset: delta probes with x-only / y-only offsets to see the read pattern
    const int W = 16, H = 12, STEP = 64;
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 1 << 20);
    NppStreamContext c{};
    c.hStream = 0;
    struct C { const char *n; int px, py; NppiPoint off; } cs[] = {
      {"off(2,0) delta(3,0)", 3, 0, {2, 0}},
      {"off(2,0) delta(2,0)", 2, 0, {2, 0}},
      {"off(0,2) delta(0,3)", 0, 3, {0, 2}},
      {"off(2,2) delta(3,3)", 3, 3, {2, 2}},
      {"off(0,0) delta(3,3)", 3, 3, {0, 0}},
    };
    for (auto &x : cs) {
      std::vector<Npp32f> h((size_t)W * H, 0.0f);
      h[(size_t)x.py * W + x.px] = 1.0f;
      cudaMemcpy(d, h.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
      cudaMemset(o, 0, 1 << 20);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          d, STEP, {W, H}, x.off, o, 8 * 4, {6, 6}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
      printf("%-22s st=%s:", x.n, es(st));
      if (st == 0) {
        std::vector<Npp32f> v(48);
        cudaMemcpy(v.data(), o, 8 * 6 * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 48; ++i)
          if (v[i] != 0.0f) printf(" (%d,%d)=%.4g", i % 8, i / 8, v[i]);
      }
      printf("\n");
    }
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 8) {
    // dst step alignment, zero-ROI step skip, taps 98/99, C3R step rules
    const int W = 12, H = 8;
    std::vector<Npp32f> hf((size_t)48 * H, 1.0f);
    std::vector<Npp8u> h8((size_t)16 * H, 3);
    Npp32f *df, *of;
    Npp8u *d8, *o8;
    cudaMalloc(&df, (size_t)48 * H);
    cudaMalloc(&of, 1 << 16);
    cudaMalloc(&d8, (size_t)16 * H);
    cudaMalloc(&o8, 1 << 16);
    cudaMemcpy(df, hf.data(), (size_t)48 * H, cudaMemcpyHostToDevice);
    cudaMemcpy(d8, h8.data(), (size_t)16 * H, cudaMemcpyHostToDevice);
    NppStreamContext c{};
    c.hStream = 0;
    struct C { const char *n; int ds; NppiSize roi; } cs[] = {
      {"32f w6 ds24", 24, {6, 6}},   {"32f w6 ds26", 26, {6, 6}},
      {"32f w6 ds28", 28, {6, 6}},   {"32f w6 ds23", 23, {6, 6}},
      {"32f zeroROI ds4", 4, {0, 0}}, {"32f w0x6 ds4", 4, {0, 6}},
      {"32f w6x0 ds4", 4, {6, 0}},
    };
    for (auto &x : cs) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          df, 48, {W, H}, {0, 0}, of, x.ds, x.roi, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
      printf("%-18s st=%s\n", x.n, es(st));
    }
    for (int t : {98, 99}) {
      std::vector<Npp32f> kk(t, 1.0f / t);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          df, 48, {W, H}, {0, 0}, of, 32, {6, 6}, 2.0f, t, kk.data(), NPP_BORDER_REPLICATE, c);
      printf("taps %-4d st=%s\n", t, es(st));
    }
    // 8u C3R dst step: misaligned (not a multiple of 3), too small
    struct D { const char *n; int ds; } ds3[] = {
      {"8u C3R w4 ds12", 12}, {"8u C3R w4 ds14", 14}, {"8u C3R w4 ds13", 13}, {"8u C3R w4 ds11", 11},
    };
    for (auto &x : ds3) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx(
          d8, 16, {12, 8}, {0, 0}, o8, x.ds, {4, 6}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
      printf("%-18s st=%s\n", x.n, es(st));
    }
    // 8u C1R dst step 0 with zero ROI
    NppStatus st = nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(
        d8, 16, {12, 8}, {0, 0}, o8, 0, {0, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
    printf("8u zeroROI ds0 st=%s\n", es(st));
    cudaFree(df);
    cudaFree(of);
    cudaFree(d8);
    cudaFree(o8);
  }

  if (part == 9) {
    // correct NPP convention: pSrc = image + offset, samples pSrc-relative
    const int W = 12, H = 8, STEP = 48;
    Npp32f *d, *o;
    cudaMalloc(&d, (size_t)STEP * H);
    cudaMalloc(&o, 1 << 16);
    NppStreamContext c{};
    c.hStream = 0;
    std::vector<Npp32f> ramp((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) ramp[y * W + x] = (float)(x + 100 * y);
    cudaMemcpy(d, ramp.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    for (int off : {0, 2, 4}) {
      const Npp32f *ps = d + (size_t)off * W + off;
      cudaMemset(o, 0, 1 << 16);
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
          ps, STEP, {W, H}, {off, off}, o, 8 * 4, {4, 4}, 2.0f, 3, k125, NPP_BORDER_REPLICATE, c);
      printf("ramp off%d st=%s:", off, es(st));
      if (st == 0) {
        std::vector<Npp32f> v(32);
        cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 32; ++i) printf(" %.4g", v[i]);
      }
      printf("\n");
    }
    // delta at image (3,3), pSrc = image+2
    std::vector<Npp32f> dl((size_t)W * H, 0.0f);
    dl[(size_t)3 * W + 3] = 1.0f;
    cudaMemcpy(d, dl.data(), (size_t)W * H * 4, cudaMemcpyHostToDevice);
    const Npp32f *ps = d + (size_t)2 * W + 2;
    cudaMemset(o, 0, 1 << 16);
    NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx(
        ps, STEP, {W, H}, {2, 2}, o, 8 * 4, {4, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
    printf("delta(3,3) psrc+2 st=%s:", es(st));
    if (st == 0) {
      std::vector<Npp32f> v(32);
      cudaMemcpy(v.data(), o, v.size() * 4, cudaMemcpyDeviceToHost);
      for (int i = 0; i < 32; ++i)
        if (v[i] != 0.0f) printf(" (%d,%d)=%.4g", i % 8, i / 8, v[i]);
    }
    printf("\n");
    cudaFree(d);
    cudaFree(o);
  }

  if (part == 10) {
    // alignment rules per destination dtype / channel count
    const int W = 12, H = 8;
    Npp32f *d32, *o32;
    Npp16u *d16, *o16;
    cudaMalloc(&d32, (size_t)W * H * 3 * 4);
    cudaMalloc(&o32, 1 << 16);
    cudaMalloc(&d16, (size_t)W * H * 3 * 2);
    cudaMalloc(&o16, 1 << 16);
    cudaMemset(d32, 1, (size_t)W * H * 3 * 4);
    cudaMemset(d16, 1, (size_t)W * H * 3 * 2);
    NppStreamContext c{};
    c.hStream = 0;
    for (int ds : {48, 50, 52, 51}) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx(
          d32, W * 3 * 4, {W, H}, {0, 0}, o32, ds, {4, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
      printf("32f C3 ds%-3d st=%s\n", ds, es(st));
    }
    for (int ds : {12, 13, 14, 16}) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx(
          d16, W * 3 * 2, {W, H}, {0, 0}, o16, ds, {6, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
      printf("16u C1 ds%-3d st=%s\n", ds, es(st));
    }
    for (int ds : {24, 25, 26, 27}) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx(
          d16, W * 3 * 2, {W, H}, {0, 0}, o16, ds, {4, 4}, 2.0f, 3, k3, NPP_BORDER_REPLICATE, c);
      printf("16u C3 ds%-3d st=%s\n", ds, es(st));
    }
    for (int ds : {12, 13, 14}) {
      NppStatus st = nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx(
          (const Npp8u *)d16, W * 3 * 2, {W, H}, {0, 0}, (Npp8u *)o16, ds, {4, 4}, 2.0f, 3, k3,
          NPP_BORDER_REPLICATE, c);
      printf("8u C3 ds%-3d st=%s\n", ds, es(st));
    }
    cudaFree(d32);
    cudaFree(o32);
    cudaFree(d16);
    cudaFree(o16);
  }
  return 0;
}
// appended parts 4-6 (see main dispatcher below)
