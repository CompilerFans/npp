// f13: bilateral status matrix (one process per case).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  int which = argc > 1 ? atoi(argv[1]) : 0;
  const int W = 16, H = 10;
  std::vector<Npp8u> hs((size_t)W * H, 5), hd((size_t)W * H, 0);
  Npp8u *ds = nullptr, *dd = nullptr;
  cudaMalloc(&ds, hs.size());
  cudaMalloc(&dd, hd.size());
  cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dd, hd.data(), hd.size(), cudaMemcpyHostToDevice);
  NppiSize sz{W, H}, roi{6, 6};
  NppiPoint off{0, 0};
  int radius = 1, stp = 1, sStep = W, dStep = W;
  Npp8u *ps = ds, *pd = dd;
  NppiBorderType bt = NPP_BORDER_REPLICATE;
  const char *name = "ok";
  switch (which) {
    case 1: name = "null_src"; ps = nullptr; break;
    case 2: name = "null_dst"; pd = nullptr; break;
    case 3: name = "radius0"; radius = 0; break;
    case 4: name = "radius33"; radius = 33; break;
    case 5: name = "step0"; stp = 0; break;
    case 6: name = "stepneg"; stp = -1; break;
    case 7: name = "step3"; stp = 3; break;
    case 8: name = "border_const"; bt = NPP_BORDER_CONSTANT; break;
    case 9: name = "radius0_border"; radius = 0; bt = NPP_BORDER_CONSTANT; break;
    case 10: name = "step0_border"; stp = 0; bt = NPP_BORDER_CONSTANT; break;
    case 11: name = "step_small"; dStep = 6; break;
    case 12: name = "step_unaligned"; dStep = W + 1; break;
    case 13: name = "roi_zero"; roi = {0, 6}; break;
    case 14: name = "roi_neg"; roi = {-1, 6}; break;
    case 15: name = "srcsize0"; sz = {0, H}; break;
    case 16: name = "off_eq"; off = {W, 0}; break;
    case 17: name = "off_neg"; off = {-1, 0}; break;
    case 18: name = "srcstep0"; sStep = 0; break;
    default: break;
  }
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(ps, sStep, sz, off, pd, dStep, roi,
                                                            radius, stp, 10.f, 2.f, bt, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  printf("%-16s st=%d cuda=%s\n", name, (int)st, cudaGetErrorName(e));
  if (argc > 2) {
    // second call in the same process: does the failure stick?
    NppStatus st2 = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(ds, W, {W, H}, {0, 0}, dd, W,
                                                               {6, 6}, 1, 1, 10.f, 2.f,
                                                               NPP_BORDER_CONSTANT, ctx);
    cudaError_t e2 = cudaDeviceSynchronize();
    printf("  next(border_const) st=%d cuda=%s   ok_call=%d\n", (int)st2, cudaGetErrorName(e2),
           (int)nppiFilterBilateralGaussBorder_8u_C1R_Ctx(ds, W, {W, H}, {0, 0}, dd, W, {6, 6}, 1, 1,
                                                          10.f, 2.f, NPP_BORDER_REPLICATE, ctx));
  }
  return 0;
}
