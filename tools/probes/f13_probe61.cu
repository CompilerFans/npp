// f13: C3 Boundary fill, 4/8-way "region outline" with L1 connectivity, and 16u/32u checks.
#include <nppi.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *sname(NppStatus s) {
  static char b[32];
  if (s == NPP_NO_ERROR) return "OK";
  if (s == NPP_NULL_POINTER_ERROR) return "NULL";
  if (s == NPP_SIZE_ERROR) return "SIZE";
  if (s == NPP_STEP_ERROR) return "STEP";
  if (s == NPP_NOT_SUPPORTED_MODE_ERROR) return "NOTSUP";
  if (s == NPP_BAD_ARGUMENT_ERROR) return "BADARG";
  if (s == NPP_CUDA_KERNEL_EXECUTION_ERROR) return "KERNEL";
  snprintf(b, sizeof b, "%d", (int)s);
  return b;
}

static NppStreamContext make_ctx() {
  NppStreamContext c;
  memset(&c, 0, sizeof c);
  c.hStream = 0;
  cudaGetDevice(&c.nCudaDeviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, c.nCudaDeviceId);
  c.nMultiProcessorCount = props.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
  c.nSharedMemPerBlock = props.sharedMemPerBlock;
  c.nCudaDevAttrComputeCapabilityMajor = props.major;
  c.nCudaDevAttrComputeCapabilityMinor = props.minor;
  cudaStreamGetFlags(c.hStream, &c.nStreamFlags);
  return c;
}

static Npp8u *dev_alloc(size_t n) {
  Npp8u *p = nullptr;
  if (cudaMalloc(&p, n) != cudaSuccess) { printf("cudaMalloc failed\n"); exit(1); }
  return p;
}

int main() {
  const NppStreamContext C = make_ctx();
  const int W = 10, H = 10;
  Npp8u *buf = dev_alloc(1 << 22);
  Npp8u *D = dev_alloc((size_t)W * H * 4 + 4096);

  // A: C3 boundary on a 4x4 block of (20,20,20) in a field of (10,10,10).
  {
    std::vector<Npp8u> h((size_t)W * H * 3, 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h[i] = 10; h[i + 1] = 10; h[i + 2] = 10;
      }
    for (int y = 3; y < 7; ++y)
      for (int x = 3; x < 7; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h[i] = 20; h[i + 1] = 20; h[i + 2] = 20;
      }
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    Npp8u nv[3] = {77, 78, 79}, bv[3] = {200, 201, 202};
    NppStatus st = nppiFloodFillBoundary_8u_C3IR_Ctx(D, 3 * W, NppiPoint{4, 4}, nv, bv, nppiNormL1,
                                                     NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("A C3 boundary 4x4 -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        if (h[i] == 77) printf(" N ");
        else if (h[i] == 200) printf(" B ");
        else if (h[i] == 20) printf(" o ");
        else printf(" . ");
      }
      printf("\n");
    }
    printf("  count=%u seed=(%u,%u,%u) box=(%d,%d,%d,%d)\n", reg.nConnectedPixelCount,
           reg.aSeedPixelValue[0], reg.aSeedPixelValue[1], reg.aSeedPixelValue[2],
           reg.oBoundingBox.x, reg.oBoundingBox.y, reg.oBoundingBox.width, reg.oBoundingBox.height);
  }
  // B: C1 with L1 connectivity, diagonal chain attached to a block.
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 2; y < 5; ++y)
      for (int x = 2; x < 5; ++x) h[(size_t)y * W + x] = 5;
    h[5 * W + 5] = 5;  // diagonal to (4,4)
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFillBoundary_8u_C1IR_Ctx(D, W, NppiPoint{3, 3}, 33, 200, nppiNormL1,
                                                     NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("B C1 L1 3x3 block + diagonal, boundary=200 -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("%3d", (int)h[(size_t)y * W + x]);
      printf("\n");
    }
    printf("  count=%u\n", reg.nConnectedPixelCount);
  }
  // C: 16u boundary: same 3x3 block shape.
  {
    std::vector<Npp16u> h((size_t)W * H, 1);
    for (int y = 3; y < 6; ++y)
      for (int x = 3; x < 6; ++x) h[(size_t)y * W + x] = 500;
    Npp16u *D16 = (Npp16u *)D;
    cudaMemcpy(D16, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFillBoundary_16u_C1IR_Ctx(D16, 2 * W, NppiPoint{4, 4}, 33, 200,
                                                      nppiNormInf, NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D16, h.size() * 2, cudaMemcpyDeviceToHost);
    printf("C 16u boundary 3x3 Inf -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("%4d", (int)h[(size_t)y * W + x]);
      printf("\n");
    }
    printf("  count=%u seed=(%u)\n", reg.nConnectedPixelCount, reg.aSeedPixelValue[0]);
  }
  // D: 32u plain fill with L1 in a diagonal chain (connectivity sanity).
  {
    std::vector<Npp32u> h((size_t)W * H, 7);
    for (int i = 0; i < 5; ++i) h[(size_t)(2 + i) * W + (2 + i)] = 9;
    Npp32u *D32 = (Npp32u *)D;
    cudaMemcpy(D32, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    NppStatus st = nppiFloodFill_32u_C1IR_Ctx(D32, 4 * W, NppiPoint{2, 2}, 1, nppiNormInf,
                                              NppiSize{W, H}, nullptr, buf, C);
    cudaMemcpy(h.data(), D32, h.size() * 4, cudaMemcpyDeviceToHost);
    printf("D 32u plain Inf diagonal -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("%2u", h[(size_t)y * W + x]);
      printf("\n");
    }
    // L1 on the same picture: only the seed changes.
    std::vector<Npp32u> h2((size_t)W * H, 7);
    for (int i = 0; i < 5; ++i) h2[(size_t)(2 + i) * W + (2 + i)] = 9;
    cudaMemcpy(D32, h2.data(), h2.size() * 4, cudaMemcpyHostToDevice);
    st = nppiFloodFill_32u_C1IR_Ctx(D32, 4 * W, NppiPoint{2, 2}, 1, nppiNormL1, NppiSize{W, H},
                                    nullptr, buf, C);
    cudaMemcpy(h2.data(), D32, h2.size() * 4, cudaMemcpyDeviceToHost);
    printf("D 32u plain L1 diagonal -> %s (diag pixels: %u %u %u %u %u)\n", sname(st),
           h2[2 * W + 2], h2[3 * W + 3], h2[4 * W + 4], h2[5 * W + 5], h2[6 * W + 6]);
  }
  // E: range boundary on two in-range blobs separated by an out-of-range line.
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    for (int x = 0; x < W; ++x) h[5 * W + x] = 250;
    h[W + 1] = 6;
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppStatus st = nppiFloodFillRangeBoundary_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 4, 8, 33, 200,
                                                          nppiNormL1, NppiSize{W, H}, nullptr, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("E rangeBoundary [4,8] boundary=200 -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("%3d", (int)h[(size_t)y * W + x]);
      printf("\n");
    }
  }
  cudaFree(buf);
  cudaFree(D);
  return 0;
}
