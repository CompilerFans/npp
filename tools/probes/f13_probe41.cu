// f13: single vs batch label markers (build once against NPP, once against mcMPP).
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  const int W = 12, H = 9;
  // deterministic 4-value pattern with diagonal contacts
  std::vector<Npp8u> src((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src[y * W + x] = (Npp8u)((((x * 7 + y * 5) / 3) % 4) * 60);
  Npp8u *ds = nullptr;
  Npp32u *dl = nullptr;
  cudaMalloc(&ds, src.size());
  cudaMalloc(&dl, src.size() * 4);
  cudaMemcpy(ds, src.data(), src.size(), cudaMemcpyHostToDevice);
  cudaMemset(dl, 0, src.size() * 4);
  Npp8u *lbuf = nullptr;
  cudaMalloc(&lbuf, 1 << 20);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus s1 = nppiLabelMarkersUF_8u32u_C1R_Ctx(ds, W, dl, W * 4, {W, H}, nppiNormInf, lbuf, ctx);
  std::vector<Npp32u> single((size_t)W * H);
  cudaMemcpy(single.data(), dl, single.size() * 4, cudaMemcpyDeviceToHost);

  // batch with two identical images
  Npp32u *db = nullptr;
  cudaMalloc(&db, src.size() * 4 * 2);
  cudaMemset(db, 0, src.size() * 4 * 2);
  cudaMemcpy(db, src.data(), src.size(), cudaMemcpyHostToDevice);
  cudaMemcpy((char *)db + src.size() * 4, src.data(), src.size(), cudaMemcpyHostToDevice);
  NppiImageDescriptor simg[2] = {{(char *)ds, W, {W, H}}, {(char *)ds, W, {W, H}}};
  NppiImageDescriptor dimg[2] = {{(char *)db, W * 4, {W, H}},
                                 {(char *)db + src.size() * 4, W * 4, {W, H}}};
  NppStatus s2 = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(simg, dimg, 2, {W, H}, nppiNormInf, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32u> batch((size_t)W * H * 2);
  cudaMemcpy(batch.data(), db, batch.size() * 4, cudaMemcpyDeviceToHost);
  printf("single st=%d batch st=%d cuda=%s\n", (int)s1, (int)s2, cudaGetErrorName(e));
  printf("single:");
  for (auto v : single) printf(" %u", v);
  printf("\nbatch :");
  for (size_t i = 0; i < single.size(); ++i) printf(" %u", batch[i]);
  printf("\n");
  return 0;
}
