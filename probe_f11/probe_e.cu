// Probe E: unsharp semantics WITH a proper device scratch buffer.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("CUDAERR %s\n", cudaGetErrorString(e_)); exit(1);} } while(0)

static NppStreamContext mkctx() {
  NppStreamContext c{};
  c.hStream = 0;
  int dev = 0; CK(cudaGetDevice(&dev)); c.nCudaDeviceId = dev;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  c.nMultiProcessorCount = prop.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  c.nSharedMemPerBlock = prop.sharedMemPerBlock;
  return c;
}

static uint64_t g_seed = 88172645463325252ULL;
static float rnd01() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return (float)((g_seed >> 40) & 0xFFFFFF) / 16777215.0f; }

static Npp8u *g_buf = nullptr;   // 1 MiB scratch
static Npp8u *g_probe = nullptr; // second buffer used to inspect g_buf contents

static void dump_buf(const char *tag, int want_floats) {
  int n = want_floats;
  float *h = (float*)malloc(n * 4);
  CK(cudaMemcpy(h, g_buf, n * 4, cudaMemcpyDeviceToHost));
  int side = (int)sqrtf((float)n);
  printf("BUF %s k=%d\n", tag, side);
  for (int y = 0; y < side; ++y) { printf("  B");
    for (int x = 0; x < side; ++x) printf(" %.8g", h[y * side + x]);
    printf("\n"); }
  free(h);
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  NppStreamContext c = mkctx();
  CK(cudaMalloc(&g_buf, 1 << 20));
  CK(cudaMalloc(&g_probe, 1 << 20));

  // ---------------- Part A: scratch buffer contents (kernel taps)
  printf("== buf contents\n");
  {
    const int W = 16, H = 16;
    Npp32f *ds, *dd;
    CK(cudaMalloc(&ds, W * H * 4)); CK(cudaMalloc(&dd, W * H * 4));
    for (int i = 0; i < W * H; ++i) { float v = 100.0f + i; CK(cudaMemcpy(ds + i, &v, 4, cudaMemcpyHostToDevice)); }
    struct Combo { const char *tag; float r, s; };
    Combo combos[] = {{"r2_s1", 2.0f, 1.0f}, {"r1_s1", 1.0f, 1.0f}, {"r4_s1", 4.0f, 1.0f},
                      {"r2_s2", 2.0f, 2.0f}, {"r2_s05", 2.0f, 0.5f}, {"r2_s5", 2.0f, 5.0f},
                      {"r0_s1", 0.0f, 1.0f}, {"r0_s2", 0.0f, 2.0f}, {"r0_s065", 0.0f, 0.65f},
                      {"r0_s5", 0.0f, 5.0f}, {"r0_s10", 0.0f, 10.0f}};
    for (auto &cb : combos) {
      int need = 0;
      NppStatus gst = nppiFilterUnsharpGetBufferSize_32f_C1R(cb.r, cb.s, &need);
      printf("BUF combo %s need=%d gbs_status=%d\n", cb.tag, need, (int)gst);
      if (need <= 0 || (long long)need * 4 > (1 << 20) / 2) { printf("BUF skip\n"); continue; }
      CK(cudaMemset(g_buf, 0xCC, 1 << 20));
      NppStatus st = nppiFilterUnsharpBorder_32f_C1R_Ctx(ds, W * 4, NppiPoint{0, 0}, dd, W * 4,
          NppiSize{W, H}, cb.r, cb.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      printf("BUF combo %s status=%d\n", cb.tag, (int)st);
      if (st == 0) {
        dump_buf(cb.tag, need / 4);
        // how many bytes of the 1MiB buffer were touched?
        CK(cudaMemset(g_probe, 0x00, 1 << 20));
        // copy g_buf to host and count non-0xCC prefix length
        unsigned char *h = (unsigned char*)malloc(1 << 20);
        CK(cudaMemcpy(h, g_buf, 1 << 20, cudaMemcpyDeviceToHost));
        int last = -1;
        for (int i = 0; i < (1 << 20); ++i) if (h[i] != 0xCC) last = i;
        printf("BUF combo %s last_touched_byte=%d (need=%d)\n", cb.tag, last, need);
        free(h);
      }
    }
    cudaFree(ds); cudaFree(dd);
  }

  // ---------------- Part B: 32f C1 delta outputs (Gaussian + offset semantics)
  printf("== delta 32f C1\n");
  {
    const int BW = 31, BH = 31, W = 21, H = 21, OFF = 5;
    Npp32f hs[BW * BH]; Npp32f hd[W * H];
    Npp32f *ds, *dd; CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
    auto run = [&](const char *tag, int delta_pos, NppiPoint off, float rad, float sig, float w, float thr,
                   NppiBorderType bord, int print) {
      for (int i = 0; i < BW * BH; ++i) hs[i] = 0.0f;
      if (delta_pos >= 0) hs[delta_pos] = 1.0f;
      for (int i = 0; i < W * H; ++i) hd[i] = -777.0f;
      CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
      NppStatus st = nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * sizeof(Npp32f), off, dd,
          W * sizeof(Npp32f), NppiSize{W, H}, rad, sig, w, thr, bord, g_buf, mkctx());
      CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
      printf("USH %s off=(%d,%d) r=%g s=%g w=%g thr=%g bord=%d status=%d\n", tag, off.x, off.y, rad, sig, w, thr, (int)bord, (int)st);
      if (print && st == 0) for (int y = 0; y < H; ++y) { printf("  R"); for (int x = 0; x < W; ++x) printf(" %.9g", hd[y * W + x]); printf("\n"); }
      return st;
    };
    run("delta_r2s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r1s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 1.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r4s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 4.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r2s2", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 2.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r0s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 0.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r0s2", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 0.0f, 2.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    // border types near edge
    run("edge_rep", OFF + 0 + 0 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("edge_wra", OFF + 0 + 0 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_WRAP, 0);
    run("edge_mir", OFF + 0 + 0 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_MIRROR, 0);
    run("edge_con", OFF + 0 + 0 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_CONSTANT, 0);
    // offset semantics
    run("off_zero", OFF + 0 + 0 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("off_two",  OFF + 0 + 0 * BW, NppiPoint{2, 2}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("off_neg",  OFF + 0 + 0 * BW, NppiPoint{-2, -2}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("off_big",  OFF + 0 + 0 * BW, NppiPoint{50, 50}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 0);
    // weights
    run("w0",   OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 0.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("w2",   OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 2.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("wneg", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, -1.5f, 0.0f, NPP_BORDER_REPLICATE, 1);
    cudaFree(ds); cudaFree(dd);
  }

  // ---------------- Part C: threshold semantics
  printf("== threshold 32f C1\n");
  {
    const int BW = 31, BH = 31, W = 21, H = 21, OFF = 5;
    Npp32f *ds, *dd; CK(cudaMalloc(&ds, sizeof(Npp32f) * BW * BH)); CK(cudaMalloc(&dd, sizeof(Npp32f) * W * H));
    Npp32f base[BW * BH], out0[W * H], outT[W * H], outTp[W * H], out2[W * H];
    for (int i = 0; i < BW * BH; ++i) base[i] = 0.0f;
    base[(OFF + 10) + (OFF + 10) * BW] = 1.0f;
    CK(cudaMemcpy(ds, base, sizeof(base), cudaMemcpyHostToDevice));
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * 4, NppiPoint{0, 0}, dd, W * 4,
        NppiSize{W, H}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    CK(cudaMemcpy(out0, dd, sizeof(out0), cudaMemcpyDeviceToHost));
    float hp = out0[12 * W + 10];
    float thr = fabsf(hp);
    float thrP = nextafterf(thr, 1e30f);
    printf("THR hpval=%.9g thr=%.9g thrP=%.9g\n", hp, thr, thrP);
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * 4, NppiPoint{0, 0}, dd, W * 4,
        NppiSize{W, H}, 2.0f, 1.0f, 1.0f, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
    CK(cudaMemcpy(outT, dd, sizeof(outT), cudaMemcpyDeviceToHost));
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * 4, NppiPoint{0, 0}, dd, W * 4,
        NppiSize{W, H}, 2.0f, 1.0f, 1.0f, thrP, NPP_BORDER_REPLICATE, g_buf, mkctx());
    CK(cudaMemcpy(outTp, dd, sizeof(outTp), cudaMemcpyDeviceToHost));
    printf("THR applied_at_eq=%d applied_at_gt=%d\n", outT[12 * W + 10] == hp ? 1 : 0, outTp[12 * W + 10] == hp ? 1 : 0);
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * 4, NppiPoint{0, 0}, dd, W * 4,
        NppiSize{W, H}, 2.0f, 1.0f, 2.0f, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
    CK(cudaMemcpy(out2, dd, sizeof(out2), cudaMemcpyDeviceToHost));
    printf("THR w2_out=%.9g (2*hp=%.9g, 0 means |HP|<thr pre-weight)\n", out2[12 * W + 10], 2 * hp);
    cudaFree(ds); cudaFree(dd);
  }

  // ---------------- Part D: 8u/16u/16s/32f random (rounding/saturation) + AC4
  printf("== random\n");
  {
    const int BW = 24, BH = 12, W = 16, H = 8, OFF = 4;
    NppStatus st;
    Npp8u hs8[BW * BH * 4], hd8[W * H * 4]; Npp16u hs16u[BW * BH * 4], hd16u[W * H * 4];
    Npp16s hs16s[BW * BH * 4], hd16s[W * H * 4]; Npp32f hs32[BW * BH * 4], hd32[W * H * 4];
    g_seed = 88172645463325252ULL;
    for (int i = 0; i < BW * BH * 4; ++i) { hs8[i] = (Npp8u)(rnd01()*255.9f); hs16u[i] = (Npp16u)(rnd01()*65535.9f);
      hs16s[i] = (Npp16s)(rnd01()*65535.9f - 32768.0f); hs32[i] = rnd01()*255.0f; }
    Npp8u *d8, *o8; Npp16u *d16u, *o16u; Npp16s *d16s, *o16s; Npp32f *d32, *o32;
    CK(cudaMalloc(&d8, sizeof(hs8))); CK(cudaMalloc(&o8, sizeof(hd8)));
    CK(cudaMalloc(&d16u, sizeof(hs16u))); CK(cudaMalloc(&o16u, sizeof(hd16u)));
    CK(cudaMalloc(&d16s, sizeof(hs16s))); CK(cudaMalloc(&o16s, sizeof(hd16s)));
    CK(cudaMalloc(&d32, sizeof(hs32))); CK(cudaMalloc(&o32, sizeof(hd32)));
    CK(cudaMemcpy(d8, hs8, sizeof(hs8), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d16u, hs16u, sizeof(hs16u), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d16s, hs16s, sizeof(hs16s), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d32, hs32, sizeof(hs32), cudaMemcpyHostToDevice));
    auto dump8 = [&](const char *tag, const Npp8u *p, int ch, int stepb, const char *what) {
      printf("%s %s\n", tag, what);
      for (int y = 0; y < (what[0]=='S'?BH:H); ++y) { printf("  D");
        for (int x = 0; x < (what[0]=='S'?BW:W) * ch; ++x) printf(" %d", (int)((const Npp8u*)((const char*)p + y*stepb))[x]);
        printf("\n"); }
    };
    auto dump16 = [&](const char *tag, const void *p, int ch, int stepb, const char *what) {
      printf("%s %s\n", tag, what);
      for (int y = 0; y < (what[0]=='S'?BH:H); ++y) { printf("  D");
        for (int x = 0; x < (what[0]=='S'?BW:W) * ch; ++x) printf(" %d", (int)((const Npp16u*)((const char*)p + y*stepb))[x]);
        printf("\n"); }
    };
    auto dump32 = [&](const char *tag, const Npp32f *p, int ch, int stepb, const char *what) {
      printf("%s %s\n", tag, what);
      for (int y = 0; y < (what[0]=='S'?BH:H); ++y) { printf("  D");
        for (int x = 0; x < (what[0]=='S'?BW:W) * ch; ++x) printf(" %.9g", ((const Npp32f*)((const char*)p + y*stepb))[x]);
        printf("\n"); }
    };
    st = nppiFilterUnsharpBorder_8u_C1R_Ctx(d8 + (OFF*BW+OFF), BW, NppiPoint{0,0}, o8, W, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH8U_C1R status=%d\n", (int)st);
    dump8("8u_C1R", hs8 + (OFF*BW+OFF), 1, BW, "SRC");
    CK(cudaMemcpy(hd8, o8, W * H, cudaMemcpyDeviceToHost));
    dump8("8u_C1R", hd8, 1, W, "DST");
    st = nppiFilterUnsharpBorder_8u_C3R_Ctx(d8 + (OFF*BW+OFF)*3, BW*3, NppiPoint{0,0}, o8, W*3, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH8U_C3R status=%d\n", (int)st);
    dump8("8u_C3R", hs8 + (OFF*BW+OFF)*3, 3, BW*3, "SRC");
    CK(cudaMemcpy(hd8, o8, W * H * 3, cudaMemcpyDeviceToHost));
    dump8("8u_C3R", hd8, 3, W*3, "DST");
    for (int y = 0; y < BH; ++y) for (int x = 0; x < BW; ++x) hs8[(y*BW+x)*4+3] = (Npp8u)((x*13+y*7)&0xFF);
    CK(cudaMemcpy(d8, hs8, sizeof(hs8), cudaMemcpyHostToDevice));
    memset(hd8, 0xEE, sizeof(hd8));
    CK(cudaMemcpy(o8, hd8, W*H*4, cudaMemcpyHostToDevice));
    st = nppiFilterUnsharpBorder_8u_AC4R_Ctx(d8 + (OFF*BW+OFF)*4, BW*4, NppiPoint{0,0}, o8, W*4, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH8U_AC4R status=%d\n", (int)st);
    dump8("8u_AC4R", hs8 + (OFF*BW+OFF)*4, 4, BW*4, "SRC");
    CK(cudaMemcpy(hd8, o8, W * H * 4, cudaMemcpyDeviceToHost));
    dump8("8u_AC4R", hd8, 4, W*4, "DST");
    st = nppiFilterUnsharpBorder_16u_C1R_Ctx(d16u + (OFF*BW+OFF), BW*2, NppiPoint{0,0}, o16u, W*2, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH16U_C1R status=%d\n", (int)st);
    dump16("16u_C1R", hs16u + (OFF*BW+OFF), 1, BW*2, "SRC");
    CK(cudaMemcpy(hd16u, o16u, W * H * 2, cudaMemcpyDeviceToHost));
    dump16("16u_C1R", hd16u, 1, W*2, "DST");
    st = nppiFilterUnsharpBorder_16s_C1R_Ctx(d16s + (OFF*BW+OFF), BW*2, NppiPoint{0,0}, o16s, W*2, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH16S_C1R status=%d\n", (int)st);
    dump16("16s_C1R", hs16s + (OFF*BW+OFF), 1, BW*2, "SRC");
    CK(cudaMemcpy(hd16s, o16s, W * H * 2, cudaMemcpyDeviceToHost));
    dump16("16s_C1R", hd16s, 1, W*2, "DST");
    st = nppiFilterUnsharpBorder_32f_C1R_Ctx(d32 + (OFF*BW+OFF), BW*4, NppiPoint{0,0}, o32, W*4, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH32F_C1R status=%d\n", (int)st);
    dump32("32f_C1R", hs32 + (OFF*BW+OFF), 1, BW*4, "SRC");
    CK(cudaMemcpy(hd32, o32, W * H * 4, cudaMemcpyDeviceToHost));
    dump32("32f_C1R", hd32, 1, W*4, "DST");
    st = nppiFilterUnsharpBorder_8u_C1R_Ctx(d8 + (OFF*BW+OFF), BW, NppiPoint{0,0}, o8, W, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 8.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH8U_C1R_thr8 status=%d\n", (int)st);
    CK(cudaMemcpy(hd8, o8, W * H, cudaMemcpyDeviceToHost));
    dump8("8u_C1R_thr8", hd8, 1, W, "DST");
    // 32f AC4 alpha preservation check
    st = nppiFilterUnsharpBorder_32f_AC4R_Ctx(d32 + (OFF*BW+OFF)*4, BW*16, NppiPoint{0,0}, o32, W*16, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    printf("USH32F_AC4R status=%d\n", (int)st);
    dump32("32f_AC4R", hs32 + (OFF*BW+OFF)*4, 4, BW*16, "SRC");
    CK(cudaMemcpy(hd32, o32, W * H * 16, cudaMemcpyDeviceToHost));
    dump32("32f_AC4R", hd32, 4, W*16, "DST");
    cudaFree(d8); cudaFree(o8); cudaFree(d16u); cudaFree(o16u); cudaFree(d16s); cudaFree(o16s); cudaFree(d32); cudaFree(o32);
  }
  return 0;
}
