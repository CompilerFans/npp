// Pins the MSSSIM minimum ROI size and the MSSSIM/WMSSSIM value relation.
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <vector>
extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}
int main() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  Npp8u *buf = nullptr;
  cudaMalloc((void **)&buf, 256u << 20);
  Npp32f *out = nullptr;
  cudaMalloc((void **)&out, sizeof(Npp32f));
  for (int sz = 150; sz <= 180; ++sz) {
    std::vector<Npp8u> a((size_t)sz * sz), b((size_t)sz * sz);
    for (int y = 0; y < sz; ++y)
      for (int x = 0; x < sz; ++x) {
        a[(size_t)y * sz + x] = (Npp8u)((x * 7 + y * 13 + 5) & 0xff);
        b[(size_t)y * sz + x] = (Npp8u)((x * 7 + y * 13 + 6) & 0xff);
      }
    Npp8u *da = nullptr, *db = nullptr;
    cudaMalloc(&da, a.size());
    cudaMalloc(&db, b.size());
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    NppStatus sm = nppiMSSSIM_8u_C1R_Ctx(da, sz, db, sz, NppiSize{sz, sz}, out, buf, c);
    NppStatus sw = nppiWMSSSIM_8u_C1R_Ctx(da, sz, db, sz, NppiSize{sz, sz}, out, buf, c);
    printf("sz=%3d msssim=%d wmsssim=%d\n", sz, (int)sm, (int)sw);
    cudaFree(da);
    cudaFree(db);
  }
  // non-square: minimum applies to width and height separately
  for (auto wh : {std::pair<int, int>{176, 128}, {128, 176}, {176, 176}, {200, 160}}) {
    int W = wh.first, H = wh.second;
    std::vector<Npp8u> a((size_t)W * H, 3), b((size_t)W * H, 4);
    Npp8u *da = nullptr, *db = nullptr;
    cudaMalloc(&da, a.size());
    cudaMalloc(&db, b.size());
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    NppStatus sm = nppiMSSSIM_8u_C1R_Ctx(da, W, db, W, NppiSize{W, H}, out, buf, c);
    printf("W=%d H=%d msssim=%d\n", W, H, (int)sm);
    cudaFree(da);
    cudaFree(db);
  }
  return 0;
}
