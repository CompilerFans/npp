// Probe nppiFilterHoughLine on synthetic binary images: accumulator
// quantization, output rho/theta values, ordering and the region filter.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

#define CK(x)                                                                                                      \
  do {                                                                                                             \
    cudaError_t e_ = (x);                                                                                          \
    if (e_ != cudaSuccess) {                                                                                       \
      printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__);                                                \
      return;                                                                                                      \
    }                                                                                                              \
  } while (0)

namespace {

constexpr int W = 64, H = 64;

void run(const char *name, const std::vector<Npp8u> &src, float rho, float theta, int threshold,
         int maxLines, bool region, float rhoLo, float rhoHi, float thetaLo, float thetaHi) {
  int bufferSize = 0;
  NppStatus st = nppiFilterHoughLineGetBufferSize({W, H}, {rho, theta}, maxLines, &bufferSize);
  printf("=== %s rho=%.9g theta=%.9g thr=%d max=%d getbuf=%d size=%d\n", name, rho, theta, threshold,
         maxLines, (int)st, bufferSize);
  if (st != NPP_NO_ERROR) return;
  Npp8u *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  NppPointPolar *dLines = nullptr;
  int *dCount = nullptr;
  CK(cudaMalloc(&dSrc, src.size()));
  CK(cudaMalloc(&dBuffer, bufferSize));
  CK(cudaMalloc(&dLines, (size_t)std::max(1, maxLines) * sizeof(NppPointPolar)));
  CK(cudaMalloc(&dCount, sizeof(int)));
  CK(cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dLines, 0xEE, (size_t)std::max(1, maxLines) * sizeof(NppPointPolar)));
  CK(cudaMemset(dCount, 0xEE, sizeof(int)));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  if (region) {
    NppPointPolar bounds[2] = {{rhoLo, thetaLo}, {rhoHi, thetaHi}};
    st = nppiFilterHoughLineRegion_8u32f_C1R_Ctx(dSrc, W, {W, H}, {rho, theta}, threshold, dLines, bounds,
                                                 maxLines, dCount, dBuffer, ctx);
  } else {
    st = nppiFilterHoughLine_8u32f_C1R_Ctx(dSrc, W, {W, H}, {rho, theta}, threshold, dLines, maxLines,
                                           dCount, dBuffer, ctx);
  }
  CK(cudaDeviceSynchronize());
  int count = -1;
  std::vector<NppPointPolar> lines(std::max(1, maxLines));
  CK(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(lines.data(), dLines, lines.size() * sizeof(NppPointPolar), cudaMemcpyDeviceToHost));
  printf("status=%d count=%d\n", (int)st, count);
  for (int i = 0; i < count && i < 20; ++i) {
    if (i < 20 || count <= 200) printf("  [%2d] rho=%.9g theta=%.9g\n", i, lines[i].rho, lines[i].theta);
  }
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dLines);
  cudaFree(dCount);
}

std::vector<Npp8u> makeImage(int pattern) {
  std::vector<Npp8u> src(W * H, 0);
  switch (pattern) {
  case 0:  // horizontal line y=20 over the full width
    for (int x = 0; x < W; ++x) src[20 * W + x] = 255;
    break;
  case 1:  // vertical line x=20 over the full height
    for (int y = 0; y < H; ++y) src[y * W + 20] = 255;
    break;
  case 2:  // main diagonal y=x
    for (int i = 0; i < W; ++i) src[i * W + i] = 255;
    break;
  case 3:  // two lines: y=20 and y=40
    for (int x = 0; x < W; ++x) {
      src[20 * W + x] = 255;
      src[40 * W + x] = 255;
    }
    break;
  case 4:  // single pixel
    src[20 * W + 30] = 255;
    break;
  case 6:  // horizontal line with value 128
    for (int x = 0; x < W; ++x) src[20 * W + x] = 128;
    break;
  case 7:  // y=20 full width (64 points), y=40 half width (32 points)
    for (int x = 0; x < W; ++x) src[20 * W + x] = 255;
    for (int x = 0; x < W / 2; ++x) src[40 * W + x] = 255;
    break;
  case 8:  // horizontal line with value 1
    for (int x = 0; x < W; ++x) src[20 * W + x] = 1;
    break;
  default:
    break;
  }
  return src;
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  run("empty", makeImage(5), 1.0f, 1.0f, 1, 10, false, 0, 0, 0, 0);
  run("horiz_y20", makeImage(0), 1.0f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  run("vert_x20", makeImage(1), 1.0f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  run("diag", makeImage(2), 1.0f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  run("two_lines", makeImage(3), 1.0f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  run("single_pixel", makeImage(4), 1.0f, 1.0f, 1, 10, false, 0, 0, 0, 0);
  run("horiz_rho2", makeImage(0), 2.0f, 1.0f, 20, 10, false, 0, 0, 0, 0);
  run("horiz_theta2", makeImage(0), 1.0f, 2.0f, 20, 10, false, 0, 0, 0, 0);
  run("horiz_rho05", makeImage(0), 0.5f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  run("horiz_region", makeImage(0), 1.0f, 1.0f, 50, 10, true, 25, 40, 80, 100);
  run("single_pixel_all", makeImage(4), 1.0f, 1.0f, 1, 300, false, 0, 0, 0, 0);
  run("horiz_rho2_all", makeImage(0), 2.0f, 1.0f, 50, 300, false, 0, 0, 0, 0);
  run("horiz_theta2_all", makeImage(0), 1.0f, 2.0f, 50, 300, false, 0, 0, 0, 0);
  run("horiz_both2_all", makeImage(0), 2.0f, 2.0f, 50, 300, false, 0, 0, 0, 0);
  run("vert_rho2_all", makeImage(1), 2.0f, 1.0f, 50, 300, false, 0, 0, 0, 0);
  run("single_pixel_d2_all", makeImage(4), 2.0f, 2.0f, 1, 400, false, 0, 0, 0, 0);
  run("single_pixel_t05_all", makeImage(4), 1.0f, 0.5f, 1, 800, false, 0, 0, 0, 0);
  run("horiz_t05", makeImage(0), 1.0f, 0.5f, 50, 10, false, 0, 0, 0, 0);
  run("two_lines_diff", makeImage(7), 1.0f, 1.0f, 20, 10, false, 0, 0, 0, 0);
  run("grey128", makeImage(6), 1.0f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  run("grey1", makeImage(8), 1.0f, 1.0f, 50, 10, false, 0, 0, 0, 0);
  return 0;
}
