// Probe C: Unsharp semantics + GetBufferSize formula.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>

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

template <class T> struct Lim { static T lo(); static T hi(); };
template <> struct Lim<Npp8u>  { static Npp8u lo() { return 0; } static Npp8u hi() { return 255; } };
template <> struct Lim<Npp16u> { static Npp16u lo() { return 0; } static Npp16u hi() { return 65535; } };
template <> struct Lim<Npp16s> { static Npp16s lo() { return -32768; } static Npp16s hi() { return 32767; } };
template <> struct Lim<Npp32f> { static Npp32f lo() { return -1e30f; } static Npp32f hi() { return 1e30f; } };

static const char *bn(NppiBorderType b) { return b==NPP_BORDER_REPLICATE?"REP":b==NPP_BORDER_CONSTANT?"CON":b==NPP_BORDER_WRAP?"WRA":b==NPP_BORDER_MIRROR?"MIR":"?"; }

// ---- GetBufferSize dispatch
static int gbs(const char *v, float r, float s, NppStatus *st) {
  int n = -1;
  if (!strcmp(v,"8u_C1R")) *st = nppiFilterUnsharpGetBufferSize_8u_C1R(r,s,&n);
  else if (!strcmp(v,"8u_C3R")) *st = nppiFilterUnsharpGetBufferSize_8u_C3R(r,s,&n);
  else if (!strcmp(v,"8u_C4R")) *st = nppiFilterUnsharpGetBufferSize_8u_C4R(r,s,&n);
  else if (!strcmp(v,"8u_AC4R")) *st = nppiFilterUnsharpGetBufferSize_8u_AC4R(r,s,&n);
  else if (!strcmp(v,"16u_C1R")) *st = nppiFilterUnsharpGetBufferSize_16u_C1R(r,s,&n);
  else if (!strcmp(v,"16u_C3R")) *st = nppiFilterUnsharpGetBufferSize_16u_C3R(r,s,&n);
  else if (!strcmp(v,"16u_C4R")) *st = nppiFilterUnsharpGetBufferSize_16u_C4R(r,s,&n);
  else if (!strcmp(v,"16u_AC4R")) *st = nppiFilterUnsharpGetBufferSize_16u_AC4R(r,s,&n);
  else if (!strcmp(v,"16s_C1R")) *st = nppiFilterUnsharpGetBufferSize_16s_C1R(r,s,&n);
  else if (!strcmp(v,"16s_C3R")) *st = nppiFilterUnsharpGetBufferSize_16s_C3R(r,s,&n);
  else if (!strcmp(v,"16s_C4R")) *st = nppiFilterUnsharpGetBufferSize_16s_C4R(r,s,&n);
  else if (!strcmp(v,"16s_AC4R")) *st = nppiFilterUnsharpGetBufferSize_16s_AC4R(r,s,&n);
  else if (!strcmp(v,"32f_C1R")) *st = nppiFilterUnsharpGetBufferSize_32f_C1R(r,s,&n);
  else if (!strcmp(v,"32f_C3R")) *st = nppiFilterUnsharpGetBufferSize_32f_C3R(r,s,&n);
  else if (!strcmp(v,"32f_C4R")) *st = nppiFilterUnsharpGetBufferSize_32f_C4R(r,s,&n);
  else if (!strcmp(v,"32f_AC4R")) *st = nppiFilterUnsharpGetBufferSize_32f_AC4R(r,s,&n);
  return n;
}

static void gbs_null(const char *v, float r, float s) {
  NppStatus st;
  if (!strcmp(v,"8u_C1R")) st = nppiFilterUnsharpGetBufferSize_8u_C1R(r,s,nullptr);
  else if (!strcmp(v,"32f_C1R")) st = nppiFilterUnsharpGetBufferSize_32f_C1R(r,s,nullptr);
  else st = NPP_ERROR;
  printf("GBSNULL %s r=%g s=%g status=%d\n", v, r, s, (int)st);
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  NppStreamContext c = mkctx();
  // ---------------- Part 1: GetBufferSize grid
  printf("== GBS grid\n");
  const char *vars[16] = {"8u_C1R","8u_C3R","8u_C4R","8u_AC4R","16u_C1R","16u_C3R","16u_C4R","16u_AC4R",
                          "16s_C1R","16s_C3R","16s_C4R","16s_AC4R","32f_C1R","32f_C3R","32f_C4R","32f_AC4R"};
  float rads[] = {0.1f, 0.5f, 0.9f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 5.0f, 7.5f, 10.0f};
  float sigs[] = {0.1f, 0.5f, 0.9f, 1.0f, 2.0f, 5.0f};
  for (int i = 0; i < 16; ++i)
    for (float r : rads)
      for (float s : sigs) {
        NppStatus st; int n = gbs(vars[i], r, s, &st);
        printf("GBS %s r=%.6g s=%.6g status=%d size=%d\n", vars[i], r, s, (int)st, n);
      }
  // NOTE: GetBufferSize with NULL out pointer SEGFAULTS on NPP 13.4 (measured) - not probed further.

  // ---------------- Part 2: 32f C1 delta (Gaussian extraction + border support)
  printf("== unsharp 32f C1 delta\n");
  {
    const int BW = 31, BH = 31, W = 21, H = 21, OFF = 5;
    Npp32f hs[BW * BH]; Npp32f hd[W * H];
    Npp32f *ds, *dd; CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
    auto run = [&](const char *tag, int delta_pos, NppiPoint off, float rad, float sig, float w, float thr,
                   NppiBorderType bord, int print) {
      for (int i = 0; i < BW * BH; ++i) hs[i] = 0.0f;
      if (delta_pos >= 0) hs[(delta_pos / BW) * BW + delta_pos % BW] = 1.0f;
      for (int i = 0; i < W * H; ++i) hd[i] = -777.0f;
      CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
      NppStatus st = nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * sizeof(Npp32f), off, dd,
          W * sizeof(Npp32f), NppiSize{W, H}, rad, sig, w, thr, bord, nullptr, mkctx());
      CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
      printf("USH %s off=(%d,%d) r=%g s=%g w=%g thr=%g bord=%s status=%d\n", tag, off.x, off.y, rad, sig, w, thr, bn(bord), (int)st);
      if (print) for (int y = 0; y < H; ++y) { printf("  R"); for (int x = 0; x < W; ++x) printf(" %.7g", hd[y * W + x]); printf("\n"); }
      return st;
    };
    // delta at ROI center; extract G = 2*delta - Result
    run("delta_r2s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r1s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 1.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r4s1", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 4.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("delta_r2s2", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 2.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    // radius vs sigma independence: r4 s1 vs r2 s1 should differ if radius matters
    // border types on a delta near the edge
    run("edge_rep",  OFF + 0 + 0 * BW,  NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("edge_con",  OFF + 0 + 0 * BW,  NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_CONSTANT, 1);
    run("edge_wra",  OFF + 0 + 0 * BW,  NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_WRAP, 1);
    run("edge_mir",  OFF + 0 + 0 * BW,  NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_MIRROR, 1);
    run("edge_none", OFF + 0 + 0 * BW,  NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, (NppiBorderType)0, 0);
    // offset semantics: same buffer, different oSrcOffset
    run("off_zero",  OFF + 0 + 0 * BW,  NppiPoint{0, 0}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("off_two",   OFF + 0 + 0 * BW,  NppiPoint{2, 2}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("off_neg",   OFF + 0 + 0 * BW,  NppiPoint{-2, -2}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 0);
    run("off_big",   OFF + 0 + 0 * BW,  NppiPoint{50, 50}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, 0);
    // weight: 0 and 2
    run("w0",  OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 0.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("w2",  OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, 2.0f, 0.0f, NPP_BORDER_REPLICATE, 1);
    run("wneg", OFF + 10 + 10 * BW, NppiPoint{0, 0}, 2.0f, 1.0f, -1.5f, 0.0f, NPP_BORDER_REPLICATE, 1);
    cudaFree(ds); cudaFree(dd);
  }

  // ---------------- Part 3: threshold semantics (32f C1)
  printf("== unsharp threshold 32f C1\n");
  {
    const int BW = 31, BH = 31, W = 21, H = 21, OFF = 5;
    Npp32f *ds, *dd; CK(cudaMalloc(&ds, sizeof(Npp32f) * BW * BH)); CK(cudaMalloc(&dd, sizeof(Npp32f) * W * H));
    Npp32f base[BW * BH], out0[W * H], outT[W * H], outTp[W * H];
    for (int i = 0; i < BW * BH; ++i) base[i] = 0.0f;
    base[(OFF + 10) + (OFF + 10) * BW] = 1.0f;
    CK(cudaMemcpy(ds, base, sizeof(base), cudaMemcpyHostToDevice));
    // reference: thr = 0, w = 1 -> out = HP everywhere (since image is 0 off-center)
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * sizeof(Npp32f), NppiPoint{0,0}, dd,
        W * sizeof(Npp32f), NppiSize{W,H}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    CK(cudaMemcpy(out0, dd, sizeof(out0), cudaMemcpyDeviceToHost));
    // pick pixel (10,12): its HP value is out0[12*W+10] (image==0 there)
    float hp = out0[12 * W + 10];
    float thr = fabsf(hp);
    float thrP = nextafterf(thr, 1e30f);
    printf("THR hpval=%.9g thr=%.9g thrP=%.9g\n", hp, thr, thrP);
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * sizeof(Npp32f), NppiPoint{0,0}, dd,
        W * sizeof(Npp32f), NppiSize{W,H}, 2.0f, 1.0f, 1.0f, thr, NPP_BORDER_REPLICATE, nullptr, mkctx());
    CK(cudaMemcpy(outT, dd, sizeof(outT), cudaMemcpyDeviceToHost));
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * sizeof(Npp32f), NppiPoint{0,0}, dd,
        W * sizeof(Npp32f), NppiSize{W,H}, 2.0f, 1.0f, 1.0f, thrP, NPP_BORDER_REPLICATE, nullptr, mkctx());
    CK(cudaMemcpy(outTp, dd, sizeof(outTp), cudaMemcpyDeviceToHost));
    printf("THR applied_at_eq=%d applied_at_gt=%d\n", outT[12*W+10] == hp ? 1 : 0, outTp[12*W+10] == hp ? 1 : 0);
    // does threshold apply to HP or w*HP? w=2, threshold = |hp| (pre-weight value)
    float out2[W * H];
    nppiFilterUnsharpBorder_32f_C1R_Ctx(ds + OFF * BW + OFF, BW * sizeof(Npp32f), NppiPoint{0,0}, dd,
        W * sizeof(Npp32f), NppiSize{W,H}, 2.0f, 1.0f, 2.0f, fabsf(hp), NPP_BORDER_REPLICATE, nullptr, mkctx());
    CK(cudaMemcpy(out2, dd, sizeof(out2), cudaMemcpyDeviceToHost));
    printf("THR w2_pre=%.9g (would be 0 if |HP|<thr pre-weight) w2_post=%d\n", out2[12*W+10], (int)(out2[12*W+10] == 2*hp));
    cudaFree(ds); cudaFree(dd);
  }

  // ---------------- Part 4: 8u/16u/16s/32f random + AC4 alpha
  printf("== unsharp random\n");
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
    auto dump16 = [&](const char *tag, const Npp16u *p, int ch, int stepb, const char *what) {
      printf("%s %s\n", tag, what);
      for (int y = 0; y < (what[0]=='S'?BH:H); ++y) { printf("  D");
        for (int x = 0; x < (what[0]=='S'?BW:W) * ch; ++x) printf(" %d", (int)((const Npp16u*)((const char*)p + y*stepb))[x]);
        printf("\n"); }
    };
    auto dump32 = [&](const char *tag, const Npp32f *p, int ch, int stepb, const char *what) {
      printf("%s %s\n", tag, what);
      for (int y = 0; y < (what[0]=='S'?BH:H); ++y) { printf("  D");
        for (int x = 0; x < (what[0]=='S'?BW:W) * ch; ++x) printf(" %.8g", ((const Npp32f*)((const char*)p + y*stepb))[x]);
        printf("\n"); }
    };
    // 8u C1
    st = nppiFilterUnsharpBorder_8u_C1R_Ctx(d8 + (OFF*BW+OFF), BW, NppiPoint{0,0}, o8, W, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH8U_C1R status=%d\n", (int)st);
    dump8("8u_C1R", hs8 + (OFF*BW+OFF), 1, BW, "SRC");
    CK(cudaMemcpy(hd8, o8, W * H, cudaMemcpyDeviceToHost));
    dump8("8u_C1R", hd8, 1, W, "DST");
    // 8u C3
    st = nppiFilterUnsharpBorder_8u_C3R_Ctx(d8 + (OFF*BW+OFF)*3, BW*3, NppiPoint{0,0}, o8, W*3, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH8U_C3R status=%d\n", (int)st);
    dump8("8u_C3R", hs8 + (OFF*BW+OFF)*3, 3, BW*3, "SRC");
    CK(cudaMemcpy(hd8, o8, W * H * 3, cudaMemcpyDeviceToHost));
    dump8("8u_C3R", hd8, 3, W*3, "DST");
    // 8u AC4 with distinctive alpha
    for (int y = 0; y < BH; ++y) for (int x = 0; x < BW; ++x) hs8[(y*BW+x)*4+3] = (Npp8u)((x*13+y*7)&0xFF);
    CK(cudaMemcpy(d8, hs8, sizeof(hs8), cudaMemcpyHostToDevice));
    memset(hd8, 0xEE, sizeof(hd8));
    CK(cudaMemcpy(o8, hd8, W*H*4, cudaMemcpyHostToDevice));
    st = nppiFilterUnsharpBorder_8u_AC4R_Ctx(d8 + (OFF*BW+OFF)*4, BW*4, NppiPoint{0,0}, o8, W*4, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH8U_AC4R status=%d\n", (int)st);
    dump8("8u_AC4R", hs8 + (OFF*BW+OFF)*4, 4, BW*4, "SRC");
    CK(cudaMemcpy(hd8, o8, W * H * 4, cudaMemcpyDeviceToHost));
    dump8("8u_AC4R", hd8, 4, W*4, "DST");
    // 16u C1, 16s C1
    st = nppiFilterUnsharpBorder_16u_C1R_Ctx(d16u + (OFF*BW+OFF), BW*2, NppiPoint{0,0}, o16u, W*2, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH16U_C1R status=%d\n", (int)st);
    dump16("16u_C1R", hs16u + (OFF*BW+OFF), 1, BW*2, "SRC");
    CK(cudaMemcpy(hd16u, o16u, W * H * 2, cudaMemcpyDeviceToHost));
    dump16("16u_C1R", hd16u, 1, W*2, "DST");
    st = nppiFilterUnsharpBorder_16s_C1R_Ctx(d16s + (OFF*BW+OFF), BW*2, NppiPoint{0,0}, o16s, W*2, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH16S_C1R status=%d\n", (int)st);
    dump16("16s_C1R", (Npp16u*)hs16s + (OFF*BW+OFF), 1, BW*2, "SRC");
    CK(cudaMemcpy(hd16s, o16s, W * H * 2, cudaMemcpyDeviceToHost));
    dump16("16s_C1R", (Npp16u*)hd16s, 1, W*2, "DST");
    // 32f C1
    st = nppiFilterUnsharpBorder_32f_C1R_Ctx(d32 + (OFF*BW+OFF), BW*4, NppiPoint{0,0}, o32, W*4, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH32F_C1R status=%d\n", (int)st);
    dump32("32f_C1R", hs32 + (OFF*BW+OFF), 1, BW*4, "SRC");
    CK(cudaMemcpy(hd32, o32, W * H * 4, cudaMemcpyDeviceToHost));
    dump32("32f_C1R", hd32, 1, W*4, "DST");
    // 8u C1 with nonzero threshold
    st = nppiFilterUnsharpBorder_8u_C1R_Ctx(d8 + (OFF*BW+OFF), BW, NppiPoint{0,0}, o8, W, NppiSize{W,H},
        2.0f, 1.0f, 1.2f, 8.0f, NPP_BORDER_REPLICATE, nullptr, mkctx());
    printf("USH8U_C1R_thr8 status=%d\n", (int)st);
    CK(cudaMemcpy(hd8, o8, W * H, cudaMemcpyDeviceToHost));
    dump8("8u_C1R_thr8", hd8, 1, W, "DST");
    cudaFree(d8); cudaFree(o8); cudaFree(d16u); cudaFree(o16u); cudaFree(d16s); cudaFree(o16s); cudaFree(d32); cudaFree(o32);
  }

  // ---------------- Part 5: validation order / buffer
  printf("== unsharp validation\n");
  {
    const int W = 16, H = 8;
    Npp8u *ds, *dd; CK(cudaMalloc(&ds, W * H * 32)); CK(cudaMalloc(&dd, W * H * 32));
    CK(cudaMemset(ds, 100, W * H * 16));
    Npp8u *buf; CK(cudaMalloc(&buf, 1 << 20));
    NppStatus st; int need = 0;
    st = nppiFilterUnsharpGetBufferSize_8u_C1R(2.0f, 1.0f, &need);
    printf("VAL need=%d gbs_status=%d\n", need, (int)st);
    auto p = [&](const char *tag, NppStatus s) { printf("VAL %-24s %d\n", tag, (int)s); };
    p("ok", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("nullbuf", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx()));
    p("nullsrc", nppiFilterUnsharpBorder_8u_C1R_Ctx(nullptr, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("nulldst", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, nullptr, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("negroi", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{-4,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("zeroroi", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{0,0}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("zeroroi_nullbuf", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{0,0}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx()));
    p("zeroroi_badborder", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{0,0}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_CONSTANT, buf, mkctx()));
    p("badborder", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_CONSTANT, buf, mkctx()));
    p("border_none", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, (NppiBorderType)0, buf, mkctx()));
    p("bordervalid_wrap", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_MIRROR, buf, mkctx()));
    p("offneg", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{-1,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("offbig", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{40,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("stepsmall", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, 15, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("negroi_nullbuf", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{-4,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, nullptr, mkctx()));
    p("negroi_badborder", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{-4,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_CONSTANT, buf, mkctx()));
    p("badborder_nullbuf", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_CONSTANT, nullptr, mkctx()));
    p("badborder_stepsmall", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, 15, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_CONSTANT, buf, mkctx()));
    // exact-size buffer
    {
      Npp8u *bsmall; CK(cudaMalloc(&bsmall, need));
      p("bufexact", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, bsmall, mkctx()));
      cudaFree(bsmall);
    }
    { Npp8u *bsmall; CK(cudaMalloc(&bsmall, need>8?need-8:need)); 
      p("bufminus8", nppiFilterUnsharpBorder_8u_C1R_Ctx(ds, W, NppiPoint{0,0}, dd, W, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, bsmall, mkctx()));
      cudaFree(bsmall); }
    // 32f C4 step odd / small
    p("c4_stepodd", nppiFilterUnsharpBorder_32f_C4R_Ctx((Npp32f*)ds, W*4*4, NppiPoint{0,0}, (Npp32f*)dd, W*4*4+1, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    p("ac4_alpha", nppiFilterUnsharpBorder_32f_AC4R_Ctx((Npp32f*)ds, W*4*4, NppiPoint{0,0}, (Npp32f*)dd, W*4*4, NppiSize{W,H}, 2.0f,1.0f,1.2f,0.0f, NPP_BORDER_REPLICATE, buf, mkctx()));
    cudaFree(ds); cudaFree(dd); cudaFree(buf);
  }
  return 0;
}
