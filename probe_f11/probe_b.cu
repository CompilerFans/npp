// Probe B: validation order / status codes for fixed-kernel filter families.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

static Npp8u *g_s8; static Npp16u *g_s16u; static Npp16s *g_s16s;
static Npp8u *g_d8; static Npp16u *g_d16u; static Npp16s *g_d16s;

static void p(const char *tag, NppStatus st) { printf("%-28s %d\n", tag, (int)st); }

int main() {
  CK(cudaMalloc(&g_s8, 4096)); CK(cudaMalloc(&g_d8, 4096));
  CK(cudaMalloc(&g_s16u, 8192)); CK(cudaMalloc(&g_d16u, 8192));
  CK(cudaMalloc(&g_s16s, 8192)); CK(cudaMalloc(&g_d16s, 8192));
  CK(cudaMemset(g_s8, 77, 4096)); CK(cudaMemset(g_d8, 55, 4096));
  CK(cudaMemset(g_s16u, 0x1111, 8192)); CK(cudaMemset(g_d16u, 0x2222, 8192));
  NppStreamContext c = mkctx();
  const NppiSize roi{4, 4};
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiMaskSize bad = (NppiMaskSize)0, m3 = NPP_MASK_SIZE_3_X_3, m5 = NPP_MASK_SIZE_5_X_5;
  const NppiBorderType rep = NPP_BORDER_REPLICATE, badb = NPP_BORDER_CONSTANT, wrap = NPP_BORDER_WRAP;

  // ---------------- plain HighPass 8u C1
  printf("== HighPass plain 8u C1\n");
  p("nullsrc", nppiFilterHighPass_8u_C1R_Ctx(nullptr, 16, g_d8, 16, roi, m3, c));
  p("nulldst", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, nullptr, 16, roi, m3, c));
  p("negroi_w", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{-1, 4}, m3, c));
  p("negroi_h", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{4, -1}, m3, c));
  p("zeroroi", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 3, NppiSize{0, 0}, m3, c));
  p("badmask", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, roi, bad, c));
  p("mask1x3", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, roi, NPP_MASK_SIZE_1_X_3, c));
  p("mask7x7", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, roi, NPP_MASK_SIZE_7_X_7, c));
  p("badmask_stepsmall", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 3, roi, bad, c));
  p("zeroroi_badmask", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 3, NppiSize{0, 0}, bad, c));
  p("zeroroi_badstep", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 3, NppiSize{0, 0}, m3, c));
  p("stepsmall", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 3, roi, m3, c));
  p("step0", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 0, g_d8, 0, roi, m3, c));
  p("stepneg", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, -16, roi, m3, c));
  p("srcstepneg", nppiFilterHighPass_8u_C1R_Ctx(g_s8, -16, g_d8, 16, roi, m3, c));

  // ---------------- plain LowPass / Sharpen / Laplace spot checks
  printf("== LowPass plain 8u C1\n");
  p("badmask", nppiFilterLowPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, roi, bad, c));
  p("negroi", nppiFilterLowPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{4, -1}, m3, c));
  p("stepsmall", nppiFilterLowPass_8u_C1R_Ctx(g_s8, 16, g_d8, 3, roi, m3, c));
  printf("== Sharpen plain 8u C1\n");
  p("negroi", nppiFilterSharpen_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{4, -1}, c));
  p("zeroroi", nppiFilterSharpen_8u_C1R_Ctx(g_s8, 16, g_d8, 3, NppiSize{0, 0}, c));
  p("stepsmall", nppiFilterSharpen_8u_C1R_Ctx(g_s8, 16, g_d8, 3, roi, c));
  printf("== Laplace plain 8u C1\n");
  p("badmask", nppiFilterLaplace_8u_C1R_Ctx(g_s8, 16, g_d8, 16, roi, bad, c));
  p("mask1x3", nppiFilterLaplace_8u_C1R_Ctx(g_s8, 16, g_d8, 16, roi, NPP_MASK_SIZE_1_X_3, c));
  p("negroi", nppiFilterLaplace_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{-1, 4}, m3, c));
  p("stepsmall", nppiFilterLaplace_8u_C1R_Ctx(g_s8, 16, g_d8, 3, roi, m3, c));
  printf("== Laplace plain 8u16s C1\n");
  p("badmask", nppiFilterLaplace_8u16s_C1R_Ctx(g_s8, 16, g_d16s, 16, roi, bad, c));
  p("stepsmall", nppiFilterLaplace_8u16s_C1R_Ctx(g_s8, 16, g_d16s, 3, roi, m3, c));
  p("stepodd", nppiFilterLaplace_8u16s_C1R_Ctx(g_s8, 16, g_d16s, 17, roi, m3, c));

  // ---------------- step alignment per dtype/channel
  printf("== step alignment (plain)\n");
  p("hp16u_c1_stepodd", nppiFilterHighPass_16u_C1R_Ctx(g_s16u, 32, g_d16u, 9, roi, m3, c));
  p("hp16u_c1_stepok", nppiFilterHighPass_16u_C1R_Ctx(g_s16u, 32, g_d16u, 8, roi, m3, c));
  p("hp8u_c3_stepodd", nppiFilterHighPass_8u_C3R_Ctx(g_s8, 16, g_d8, 13, roi, m3, c));
  p("hp8u_c4_stepodd", nppiFilterHighPass_8u_C4R_Ctx(g_s8, 16, g_d8, 17, roi, m3, c));
  p("hp8u_ac4_stepodd", nppiFilterHighPass_8u_AC4R_Ctx(g_s8, 16, g_d8, 17, roi, m3, c));
  p("hp8u_ac4_stepok", nppiFilterHighPass_8u_AC4R_Ctx(g_s8, 16, g_d8, 16, roi, m3, c));
  p("hp16s_c4_stepodd", nppiFilterHighPass_16s_C4R_Ctx(g_s16s, 32, g_d16s, 33, roi, m3, c));
  p("hp32f_c4_stepodd", nppiFilterHighPass_32f_C4R_Ctx((Npp32f*)g_s8, 64, (Npp32f*)g_d8, 65, roi, m3, c));
  p("lp16s_c3_stepodd", nppiFilterLowPass_16s_C3R_Ctx(g_s16s, 32, g_d16s, 25, roi, m3, c));

  // ---------------- border HighPass 8u C1
  printf("== HighPassBorder 8u C1\n");
  p("nullsrc", nppiFilterHighPassBorder_8u_C1R_Ctx(nullptr, 16, sz, off, g_d8, 16, roi, m3, rep, c));
  p("nulldst", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, nullptr, 16, roi, m3, rep, c));
  p("negroi", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{4, -1}, m3, rep, c));
  p("zeroroi", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 3, NppiSize{0, 0}, m3, rep, c));
  p("srcsize_w0", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 16, roi, m3, rep, c));
  p("srcsize_neg", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{-1, 5}, off, g_d8, 16, roi, m3, rep, c));
  p("offneg", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{-1, 0}, g_d8, 16, roi, m3, rep, c));
  p("offbig", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{5, 2}, g_d8, 16, roi, m3, rep, c));
  p("offbig2", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{2, 9}, g_d8, 16, roi, m3, rep, c));
  p("badborder_const", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, m3, badb, c));
  p("badborder_wrap", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, m3, wrap, c));
  p("badborder_none", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, m3, (NppiBorderType)0, c));
  p("badmask", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, bad, rep, c));
  p("badborder_badmask", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, bad, badb, c));
  p("zeroroi_badborder", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{0, 0}, m3, badb, c));
  p("zeroroi_badmask", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 3, NppiSize{0, 0}, bad, rep, c));
  p("zeroroi_offneg", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{-1, 0}, g_d8, 16, NppiSize{0, 0}, m3, rep, c));
  p("zeroroi_badstep", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 3, NppiSize{0, 0}, m3, rep, c));
  p("negroi_badborder", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{4, -1}, m3, badb, c));
  p("srcsize_badborder", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 16, roi, m3, badb, c));
  p("offneg_badborder", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{-1, 0}, g_d8, 16, roi, m3, badb, c));
  p("stepsmall", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 3, roi, m3, rep, c));
  p("badmask_stepsmall", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 3, roi, bad, rep, c));
  p("offbig_5x5", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{5, 5}, g_d8, 16, roi, m5, rep, c));

  // ---------------- border others spot checks
  printf("== LowPassBorder 8u C1\n");
  p("badmask", nppiFilterLowPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, bad, rep, c));
  p("srcsize_w0", nppiFilterLowPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 16, roi, m3, rep, c));
  p("offneg", nppiFilterLowPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{-1, 0}, g_d8, 16, roi, m3, rep, c));
  p("badborder", nppiFilterLowPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, m3, badb, c));
  printf("== SharpenBorder 8u C1\n");
  p("srcsize_w0", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 16, roi, rep, c));
  p("offneg", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{-1, 0}, g_d8, 16, roi, rep, c));
  p("badborder", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, badb, c));
  p("zeroroi", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{0, 5}, off, g_d8, 3, NppiSize{0, 0}, rep, c));
  printf("== LaplaceBorder 8u C1\n");
  p("badmask", nppiFilterLaplaceBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, bad, rep, c));
  p("badborder", nppiFilterLaplaceBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, roi, m3, badb, c));
  p("offneg", nppiFilterLaplaceBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{-1, 0}, g_d8, 16, roi, m3, rep, c));
  printf("== LaplaceBorder 8u16s C1\n");
  p("badmask", nppiFilterLaplaceBorder_8u16s_C1R_Ctx(g_s8, 16, sz, off, g_d16s, 16, roi, bad, rep, c));
  p("badborder", nppiFilterLaplaceBorder_8u16s_C1R_Ctx(g_s8, 16, sz, off, g_d16s, 16, roi, m3, badb, c));

  // ---------------- in-place (src==dst) sanity for plain
  printf("== in-place\n");
  p("hp8u_inplace", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_s8, 16, roi, m3, c));
  p("lp8u_inplace", nppiFilterLowPass_8u_C1R_Ctx(g_s8, 16, g_s8, 16, roi, m3, c));
  p("sh8u_inplace", nppiFilterSharpen_8u_C1R_Ctx(g_s8, 16, g_s8, 16, roi, c));

  // ---------------- AC4 alpha write check (HighPass 8u AC4R, C1-printed)
  printf("== AC4 alpha check\n");
  {
    const int W = 4, H = 4, CH = 4;
    Npp8u hs[W * H * CH], hd[W * H * CH];
    for (int i = 0; i < W * H * CH; ++i) { hs[i] = (Npp8u)(i * 7 % 251); hd[i] = 0xAA; }
    Npp8u *ds, *dd;
    CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
    CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
    (void)cudaGetLastError();
    NppStatus st = nppiFilterHighPass_8u_AC4R_Ctx(ds, W * CH, dd, W * CH, NppiSize{W, H}, m3, mkctx());
    CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
    cudaFree(ds); cudaFree(dd);
    printf("status=%d\n", (int)st);
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("(%d %d %d %d) ", hd[(y * W + x) * CH + 0], hd[(y * W + x) * CH + 1], hd[(y * W + x) * CH + 2], hd[(y * W + x) * CH + 3]);
      printf("\n");
    }
  }
  return 0;
}
