// Probe I: remaining unpinned validation-boundary cases for the fixed-kernel
// filter families (f11).
//  * plain masked: negroi vs mask order, null vs mask, zero-width-only ROI
//  * step alignment across dtypes/layouts (16u C1/AC4, 32f C1, 16u C1 border,
//    LaplaceBorder 8u16s) and odd source steps
//  * border masked: zero ROI vs mask with GOOD srcSize, offset == srcSize
//    boundary, 1x1 srcSize with 5x5 mask
//  * border Sharpen (unmasked): zero ROI with GOOD srcSize, offset boundary
//  * negative dst step on border (unsigned compare)
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

static Npp8u *g_s8; static Npp16u *g_s16u; static Npp8u *g_d8; static Npp16u *g_d16u;
static Npp32f *g_s32f; static Npp16s *g_d16s; static Npp32f *g_d32f;

static void p(const char *tag, NppStatus st) {
  printf("%-28s %d\n", tag, (int)st);
  fflush(stdout);
}

int main() {
  CK(cudaMalloc(&g_s8, 1 << 20)); CK(cudaMalloc(&g_d8, 1 << 20));
  CK(cudaMalloc(&g_s16u, 1 << 20)); CK(cudaMalloc(&g_d16u, 1 << 20));
  CK(cudaMalloc(&g_s32f, 1 << 20)); CK(cudaMemset(g_s32f, 0, 1 << 20));
  CK(cudaMalloc(&g_d16s, 1 << 20)); CK(cudaMalloc(&g_d32f, 1 << 20));
  CK(cudaMemset(g_s8, 77, 1 << 20)); CK(cudaMemset(g_d8, 55, 1 << 20));
  CK(cudaMemset(g_s16u, 0x11, 1 << 20)); CK(cudaMemset(g_d16u, 0x22, 1 << 20));
  NppStreamContext c = mkctx();
  const NppiSize roi{4, 4};
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiMaskSize bad = (NppiMaskSize)0, m3 = NPP_MASK_SIZE_3_X_3, m5 = NPP_MASK_SIZE_5_X_5;
  const NppiBorderType rep = NPP_BORDER_REPLICATE;

  printf("== plain HighPass 8u C1: order pins\n");
  p("negroi_h_badmask", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{4, -1}, bad, c));
  p("negroi_w_badmask", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{-1, 4}, bad, c));
  p("nullsrc_badmask", nppiFilterHighPass_8u_C1R_Ctx(nullptr, 16, g_d8, 16, roi, bad, c));
  p("nulldst_badmask", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, nullptr, 16, roi, bad, c));
  p("roizero_w", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{0, 4}, m3, c));
  p("roizero_h", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{4, 0}, m3, c));
  p("roizero_w_badmask", nppiFilterHighPass_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{0, 4}, bad, c));

  printf("== plain step alignment (16u C1 / 16u AC4 / 32f C1)\n");
  p("hp_16u_c1_stepodd", nppiFilterHighPass_16u_C1R_Ctx(g_s16u, 8, g_d16u, 9, roi, m3, c));
  p("hp_16u_c1_stepsmall", nppiFilterHighPass_16u_C1R_Ctx(g_s16u, 8, g_d16u, 7, roi, m3, c));
  p("hp_16u_ac4_stepodd", nppiFilterHighPass_16u_AC4R_Ctx(g_s16u, 32, g_d16u, 33, roi, m3, c));
  p("hp_16u_ac4_stepsmall", nppiFilterHighPass_16u_AC4R_Ctx(g_s16u, 32, g_d16u, 31, roi, m3, c));
  p("hp_32f_c1_stepodd", nppiFilterHighPass_32f_C1R_Ctx(g_s32f, 16, g_d32f, 17, roi, m3, c));
  p("hp_16u_c1_srcstepodd", nppiFilterHighPass_16u_C1R_Ctx(g_s16u, 9, g_d16u, 8, roi, m3, c));

  printf("== plain Sharpen 8u/16u C1\n");
  p("sh_roizero_w", nppiFilterSharpen_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{0, 4}, c));
  p("sh_roizero_h", nppiFilterSharpen_8u_C1R_Ctx(g_s8, 16, g_d8, 16, NppiSize{4, 0}, c));
  p("sh_16u_c1_stepodd", nppiFilterSharpen_16u_C1R_Ctx(g_s16u, 8, g_d16u, 9, roi, c));
  p("sh_16u_ac4_stepodd", nppiFilterSharpen_16u_AC4R_Ctx(g_s16u, 32, g_d16u, 33, roi, c));

  printf("== border HighPass 8u C1: zroi/mask/offset pins\n");
  p("zroi_goodsz", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{0, 0}, m3, rep, c));
  p("zroi_goodsz_badmask", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{0, 0}, bad, rep, c));
  p("zroiw_goodsz", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{0, 4}, m3, rep, c));
  p("off_eq_w", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{7, 2}, g_d8, 16, roi, m3, rep, c));
  p("off_w_minus1", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{6, 2}, g_d8, 16, roi, m3, rep, c));
  p("off_eq_h", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{2, 7}, g_d8, 16, roi, m3, rep, c));
  p("off_h_minus1", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{2, 6}, g_d8, 16, roi, m3, rep, c));
  p("srcsize1x1_m5", nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, NppiSize{1, 1}, NppiPoint{0, 0}, g_d8, 4, NppiSize{1, 1}, m5, rep, c));
  p("hp16u_b_c1_stepodd", nppiFilterHighPassBorder_16u_C1R_Ctx(g_s16u, 8, sz, off, g_d16u, 9, roi, m3, rep, c));

  printf("== border Sharpen 8u C1\n");
  p("sh_zroi_goodsz", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 16, NppiSize{0, 0}, rep, c));
  p("sh_off_eq_w", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{7, 2}, g_d8, 16, roi, rep, c));
  p("sh_off_w_minus1", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, sz, NppiPoint{6, 2}, g_d8, 16, roi, rep, c));
  p("sh_stepsmall", nppiFilterSharpenBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, 3, roi, rep, c));
  p("sh16u_b_c1_stepodd", nppiFilterSharpenBorder_16u_C1R_Ctx(g_s16u, 8, sz, off, g_d16u, 9, roi, rep, c));

  printf("== border Laplace 8u16s C1 step alignment\n");
  p("lp8u16s_b_stepodd", nppiFilterLaplaceBorder_8u16s_C1R_Ctx(g_s8, 16, sz, off, g_d16s, 9, roi, m3, rep, c));

  printf("== border negative dst step (unsigned compare)\n");
  {
    NppStatus st = nppiFilterHighPassBorder_8u_C1R_Ctx(g_s8, 16, sz, off, g_d8, -16, roi, m3, rep, c);
    cudaError_t e = cudaDeviceSynchronize();
    printf("border_stepneg %d cuda_after=%s\n", (int)st, cudaGetErrorString(e));
  }
  return 0;
}
