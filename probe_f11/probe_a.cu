// Probe A: fixed-kernel filter coefficients, rounding, saturation, raw reads.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <typeinfo>
#include <type_traits>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("CUDAERR %s\n", cudaGetErrorString(e_)); exit(1);} } while(0)

static unsigned long long g_seed = 88172645463325252ULL;
static unsigned rnd() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return (unsigned)(g_seed >> 32); }

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

// ---- family runners (plain variants) --------------------------------------
struct HighPass {
  static const char *name() { return "HighPass"; }
  static NppStatus call(const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterHighPass_8u_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterHighPass_16u_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterHighPass_16s_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterHighPass_32f_C1R_Ctx(s, ss, d, ds, r, m, c); }
};
struct LowPass {
  static const char *name() { return "LowPass"; }
  static NppStatus call(const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLowPass_8u_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLowPass_16u_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLowPass_16s_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLowPass_32f_C1R_Ctx(s, ss, d, ds, r, m, c); }
};
struct Sharpen {
  static const char *name() { return "Sharpen"; }
  static NppStatus call(const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiMaskSize, NppStreamContext c) { return nppiFilterSharpen_8u_C1R_Ctx(s, ss, d, ds, r, c); }
  static NppStatus call(const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize r, NppiMaskSize, NppStreamContext c) { return nppiFilterSharpen_16u_C1R_Ctx(s, ss, d, ds, r, c); }
  static NppStatus call(const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiMaskSize, NppStreamContext c) { return nppiFilterSharpen_16s_C1R_Ctx(s, ss, d, ds, r, c); }
  static NppStatus call(const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r, NppiMaskSize, NppStreamContext c) { return nppiFilterSharpen_32f_C1R_Ctx(s, ss, d, ds, r, c); }
};
struct Laplace {
  static const char *name() { return "Laplace"; }
  static NppStatus call(const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLaplace_8u_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp16s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLaplace_16s_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLaplace_32f_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp8u *s, int ss, Npp16s *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLaplace_8u16s_C1R_Ctx(s, ss, d, ds, r, m, c); }
  static NppStatus call(const Npp8s *s, int ss, Npp16s *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLaplace_8s16s_C1R_Ctx(s, ss, d, ds, r, m, c); }
};
struct LaplaceBorder {
  static const char *name() { return "LaplaceBorder"; }
  static NppStatus call(const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize r, NppiMaskSize m, NppStreamContext c) { return nppiFilterLaplaceBorder_8u_C1R_Ctx(s, ss, NppiSize{7,7}, NppiPoint{0,0}, d, ds, r, m, NPP_BORDER_REPLICATE, c); }
};

static const char *fmt(float v) { static char b[4][32]; static int i=0; i=(i+1)&3; snprintf(b[i],32,"%g",v); return b[i]; }

// buffer 11x11, ROI 7x7 at offset (2,2). modes: 0=delta@center(ampl), 1=rand, 2=delta@(0,0)
template <typename F, typename S, typename D>
void run_case(const char *casename, int mode, S ampl, NppiMaskSize ms, int maskTag) {
  const int BW = 11, BH = 11, W = 7, H = 7, off = 2;
  S hs[BW * BH];
  memset(hs, 0, sizeof(hs));
  if (mode == 0) { hs[5 * BW + 5] = ampl; }
  else if (mode == 2) { hs[1 * BW + 1] = ampl; }
  else { for (int i = 0; i < BW * BH; ++i) hs[i] = (S)rnd(); }
  D hd[W * H];
  // For mode 2 (delta outside ROI) use a sentinel prefill so that "read and
  // saturated to 0" is distinguishable from "not read at all".
  for (int i = 0; i < W * H; ++i) hd[i] = (D)(mode == 2 ? 100 : 0);
  S *ds; D *dd;
  CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
  CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
  (void)cudaGetLastError();
  NppStatus st = F::call(ds + off * BW + off, BW * sizeof(S), dd, W * sizeof(D), NppiSize{W, H}, ms, mkctx());
  CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
  cudaFree(ds); cudaFree(dd);
  printf("CASE %s %s mask=%d %s status=%d\n", F::name(), casename, maskTag, typeid(S).name(), (int)st);
  if (mode == 1) {
    printf("  SRC");
    for (int i = 0; i < BW * BH; ++i) {
      if constexpr (std::is_same_v<S, Npp32f>) printf(" %g", (float)hs[i]);
      else printf(" %d", (int)hs[i]);
    }
    printf("\n");
  }
  if (st == 0) {
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        if constexpr (std::is_same_v<D, Npp32f>) printf("%s ", fmt((float)hd[y * W + x]));
        else printf("%d ", (int)hd[y * W + x]);
      }
      printf("\n");
    }
  }
}

int main() {
  const NppiMaskSize m3 = NPP_MASK_SIZE_3_X_3, m5 = NPP_MASK_SIZE_5_X_5;
  // exact kernels from 32f delta responses
  run_case<HighPass, Npp32f, Npp32f>("delta", 0, 1.0f, m3, 3); run_case<HighPass, Npp32f, Npp32f>("delta", 0, 1.0f, m5, 5);
  run_case<LowPass, Npp32f, Npp32f>("delta", 0, 1.0f, m3, 3); run_case<LowPass, Npp32f, Npp32f>("delta", 0, 1.0f, m5, 5);
  run_case<Sharpen, Npp32f, Npp32f>("delta", 0, 1.0f, m3, 3);
  run_case<Laplace, Npp32f, Npp32f>("delta", 0, 1.0f, m3, 3); run_case<Laplace, Npp32f, Npp32f>("delta", 0, 1.0f, m5, 5);
  // border delta (replicate at image edge, ROI = whole 7x7 image)
  run_case<LaplaceBorder, Npp8u, Npp8u>("delta", 0, (Npp8u)1, m3, 3); run_case<LaplaceBorder, Npp8u, Npp8u>("delta", 0, (Npp8u)1, m5, 5);
  // integer rounding probes: delta with small amplitudes
  run_case<LowPass, Npp8u, Npp8u>("delta", 0, (Npp8u)4, m3, 3); run_case<LowPass, Npp8u, Npp8u>("delta", 0, (Npp8u)14, m3, 3);
  run_case<LowPass, Npp8u, Npp8u>("delta", 0, (Npp8u)12, m5, 5); run_case<LowPass, Npp8u, Npp8u>("delta", 0, (Npp8u)13, m5, 5);
  run_case<Sharpen, Npp8u, Npp8u>("delta", 0, (Npp8u)4, m3, 3); run_case<Sharpen, Npp8u, Npp8u>("delta", 0, (Npp8u)12, m3, 3);
  run_case<LowPass, Npp16s, Npp16s>("delta", 0, (Npp16s)-4, m3, 3); run_case<LowPass, Npp16s, Npp16s>("delta", 0, (Npp16s)-13, m3, 3);
  run_case<LowPass, Npp16s, Npp16s>("delta", 0, (Npp16s)13, m5, 5);
  // random images (rounding + saturation statistics)
  run_case<HighPass, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m3, 3); run_case<HighPass, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m5, 5);
  run_case<HighPass, Npp16u, Npp16u>("rand", 1, (Npp16u)0, m3, 3); run_case<HighPass, Npp16u, Npp16u>("rand", 1, (Npp16u)0, m5, 5);
  run_case<HighPass, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m3, 3); run_case<HighPass, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m5, 5);
  run_case<HighPass, Npp32f, Npp32f>("rand", 1, 0.0f, m3, 3);
  run_case<LowPass, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m3, 3); run_case<LowPass, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m5, 5);
  run_case<LowPass, Npp16u, Npp16u>("rand", 1, (Npp16u)0, m3, 3); run_case<LowPass, Npp16u, Npp16u>("rand", 1, (Npp16u)0, m5, 5);
  run_case<LowPass, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m3, 3); run_case<LowPass, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m5, 5);
  run_case<LowPass, Npp32f, Npp32f>("rand", 1, 0.0f, m3, 3);
  run_case<Sharpen, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m3, 3); run_case<Sharpen, Npp16u, Npp16u>("rand", 1, (Npp16u)0, m3, 3);
  run_case<Sharpen, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m3, 3); run_case<Sharpen, Npp32f, Npp32f>("rand", 1, 0.0f, m3, 3);
  run_case<Laplace, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m3, 3); run_case<Laplace, Npp8u, Npp8u>("rand", 1, (Npp8u)0, m5, 5);
  run_case<Laplace, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m3, 3); run_case<Laplace, Npp16s, Npp16s>("rand", 1, (Npp16s)0, m5, 5);
  run_case<Laplace, Npp32f, Npp32f>("rand", 1, 0.0f, m3, 3); run_case<Laplace, Npp32f, Npp32f>("rand", 1, 0.0f, m5, 5);
  run_case<Laplace, Npp8u, Npp16s>("rand", 1, (Npp8u)0, m3, 3); run_case<Laplace, Npp8u, Npp16s>("rand", 1, (Npp8u)0, m5, 5);
  run_case<Laplace, Npp8s, Npp16s>("rand", 1, (Npp8s)0, m3, 3); run_case<Laplace, Npp8s, Npp16s>("rand", 1, (Npp8s)0, m5, 5);
  // raw-read check: delta outside the ROI, at buffer corner
  run_case<HighPass, Npp32f, Npp32f>("outside", 2, 1000.0f, m3, 3);
  run_case<Laplace, Npp8u, Npp8u>("outside", 2, (Npp8u)200, m3, 3);
  return 0;
}
