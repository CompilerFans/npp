// valprobe: validation-order matrix for f10 families (32f C1 entry points).
#include <npp.h>
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppGetStreamContext(NppStreamContext *p);
}
#include <nppi_filtering_functions.h>
#include <cstdio>
#include <cuda_runtime.h>
static NppStreamContext ctx() { NppStreamContext c{}; nppGetStreamContext(&c); return c; }
int main() {
  const int W = 9, H = 9;
  float *s, *d, *k;
  cudaMalloc(&s, (size_t)W * H * 4);
  cudaMalloc(&d, (size_t)W * H * 4);
  cudaMalloc(&k, 40 * 4);
  cudaMemset(s, 0x3f, (size_t)W * H * 4);
  NppStreamContext c = ctx();
  NppiSize sz{9, 9}, roi{5, 5}, zroi{0, 0}, roi0w{0, 5}, roi0h{5, 0};
  NppiPoint off{2, 2};
  float noise[4] = {0.5f, 0.5f, 0.5f, 0.5f};
  float noisebad[4] = {1.5f, 0.5f, 0.5f, 0.5f};
  float n1000[4] = {1000.f, 1000.f, 1000.f, 1000.f};
  float n1[4] = {1.f, 1.f, 1.f, 1.f};
  float n0999[4] = {0.999f, 0.999f, 0.999f, 0.999f};
  float n1001[4] = {1.001f, 1.001f, 1.001f, 1.001f};
#define P(t, x) printf("%-34s %d\n", t, (int)(x))
  P("adv zroi taps0", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, zroi, 0, k, c));
  P("adv zroi taps-1", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, zroi, -1, k, c));
  P("adv zroi taps1", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, zroi, 1, k, c));
  P("adv zroi taps2", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, zroi, 2, k, c));
  P("adv zroi taps4", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, zroi, 4, k, c));
  P("adv zroi taps5", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, zroi, 5, k, c));
  P("adv roi0w taps0", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, roi0w, 0, k, c));
  P("adv roi0h taps3", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, roi0h, 3, k, c));
  P("adv zroi taps3 badstep", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 3, zroi, 3, k, c));
  P("adv taps0 badstep", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 3, roi, 0, k, c));
  P("adv negRoi nullk", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 36, NppiSize{5, -1}, 3, nullptr, c));
  P("adv zroi badstep taps3", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 3, zroi, 3, k, c));
  P("advb zroi taps0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, 0, k, NPP_BORDER_REPLICATE, c));
  P("advb zroi taps2", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, 2, k, NPP_BORDER_REPLICATE, c));
  P("advb zroi taps-1", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, -1, k, NPP_BORDER_REPLICATE, c));
  P("advb zroi taps3", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, 3, k, NPP_BORDER_REPLICATE, c));
  P("advb zroi badborder", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, 3, k, NPP_BORDER_CONSTANT, c));
  P("advb zroi offneg", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, NppiPoint{0, -1}, d, 36, zroi, 3, k, NPP_BORDER_REPLICATE, c));
  P("advb zroi srcsize0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, NppiSize{0, 5}, off, d, 36, zroi, 3, k, NPP_BORDER_REPLICATE, c));
  P("advb srcsize0 taps0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, NppiSize{0, 5}, off, d, 36, roi, 0, k, NPP_BORDER_REPLICATE, c));
  P("advb offneg taps0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, NppiPoint{-1, 0}, d, 36, roi, 0, k, NPP_BORDER_REPLICATE, c));
  P("advb badborder taps0", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, 0, k, NPP_BORDER_CONSTANT, c));
  P("advb badborder taps-1", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, -1, k, NPP_BORDER_CONSTANT, c));
  P("advb nullk badborder", nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, 3, nullptr, NPP_BORDER_CONSTANT, c));
  P("adv taps0 smallstep", nppiFilterGaussAdvanced_32f_C1R_Ctx(s, 36, d, 19, roi, 0, k, c));
  P("gb zroi badmask", nppiFilterGaussBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, (NppiMaskSize)0, NPP_BORDER_REPLICATE, c));
  P("gb zroi srcsize0", nppiFilterGaussBorder_32f_C1R_Ctx(s, 36, NppiSize{0, 5}, off, d, 36, zroi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, c));
  P("gb srcsize0 badmask", nppiFilterGaussBorder_32f_C1R_Ctx(s, 36, NppiSize{0, 5}, off, d, 36, roi, (NppiMaskSize)0, NPP_BORDER_REPLICATE, c));
  P("gb offneg badmask", nppiFilterGaussBorder_32f_C1R_Ctx(s, 36, sz, NppiPoint{-1, 0}, d, 36, roi, (NppiMaskSize)0, NPP_BORDER_REPLICATE, c));
  P("gb badborder badmask", nppiFilterGaussBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, (NppiMaskSize)0, NPP_BORDER_CONSTANT, c));
  P("gb null mask1x3", nppiFilterGaussBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NPP_MASK_SIZE_1_X_3, NPP_BORDER_REPLICATE, c));
  P("gauss 1x3", nppiFilterGauss_32f_C1R_Ctx(s, 36, d, 36, roi, NPP_MASK_SIZE_1_X_3, c));
  P("gauss 3x1", nppiFilterGauss_32f_C1R_Ctx(s, 36, d, 36, roi, NPP_MASK_SIZE_3_X_1, c));
  P("gauss 1x5", nppiFilterGauss_32f_C1R_Ctx(s, 36, d, 36, roi, NPP_MASK_SIZE_1_X_5, c));
  P("gauss 5x1", nppiFilterGauss_32f_C1R_Ctx(s, 36, d, 36, roi, NPP_MASK_SIZE_5_X_1, c));
  P("wi zroi badnoise", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, NppiSize{3, 3}, NppiPoint{1, 1}, noisebad, NPP_BORDER_REPLICATE, c));
  P("wi zroi badmask", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, NppiSize{0, 0}, NppiPoint{0, 0}, noise, NPP_BORDER_REPLICATE, c));
  P("wi zroi badanchor", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, NppiSize{3, 3}, NppiPoint{5, 5}, noise, NPP_BORDER_REPLICATE, c));
  P("wi zroi badborder", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, zroi, NppiSize{3, 3}, NppiPoint{1, 1}, noise, NPP_BORDER_CONSTANT, c));
  P("wi zroi offneg", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, NppiPoint{-1, 0}, d, 36, zroi, NppiSize{3, 3}, NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi zroi badstep", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 3, zroi, NppiSize{3, 3}, NppiPoint{1, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi badmask smallstep", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 19, roi, NppiSize{0, 0}, NppiPoint{0, 0}, noise, NPP_BORDER_REPLICATE, c));
  P("wi badanchor smallstep", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 19, roi, NppiSize{3, 3}, NppiPoint{9, 9}, noise, NPP_BORDER_REPLICATE, c));
  P("wi badnoise smallstep", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 19, roi, NppiSize{3, 3}, NppiPoint{1, 1}, noisebad, NPP_BORDER_REPLICATE, c));
  P("wi mask0 anchor0", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{0, 1}, NppiPoint{0, 0}, noise, NPP_BORDER_REPLICATE, c));
  P("wi mask1x0", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{1, 0}, NppiPoint{0, 0}, noise, NPP_BORDER_REPLICATE, c));
  P("wi anchor=w", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{3, 1}, noise, NPP_BORDER_REPLICATE, c));
  P("wi anchor=h", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{1, 3}, noise, NPP_BORDER_REPLICATE, c));
  P("wi anchor w-1", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{2, 2}, noise, NPP_BORDER_REPLICATE, c));
  P("wi anchor neg", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{-5, -5}, noise, NPP_BORDER_REPLICATE, c));
  P("wi noise 1000", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{1, 1}, n1000, NPP_BORDER_REPLICATE, c));
  P("wi noise 1.0", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{1, 1}, n1, NPP_BORDER_REPLICATE, c));
  P("wi noise 0.999", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{1, 1}, n0999, NPP_BORDER_REPLICATE, c));
  P("wi noise 1.001", nppiFilterWienerBorder_32f_C1R_Ctx(s, 36, sz, off, d, 36, roi, NppiSize{3, 3}, NppiPoint{1, 1}, n1001, NPP_BORDER_REPLICATE, c));
  return 0;
}
