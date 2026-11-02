// f13: C3 seed-pixel read quirk in the FloodFill families.
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

// picture: W=12,H=10, field (10,10,10), block x=[6,11], y=[2,6] value (30,40,50)
static std::vector<Npp8u> picture(int W, int H, int bx0, int bx1, int by0, int by1, int kind) {
  std::vector<Npp8u> h((size_t)W * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      size_t i = ((size_t)y * W + x) * 3;
      bool in = x >= bx0 && x <= bx1 && y >= by0 && y <= by1;
      h[i] = (Npp8u)(in ? 30 : 10);
      h[i + 1] = (Npp8u)(in ? 40 : 10);
      h[i + 2] = (Npp8u)(in ? 50 : 10);
    }
  (void)kind;
  return h;
}

static void show(const char *tag, const std::vector<Npp8u> &h, int W, int H, int x0, int x1, int y0,
                 int y1, Npp8u n0, Npp8u b0, Npp8u o0) {
  printf("%s\n", tag);
  for (int y = y0; y <= y1; ++y) {
    printf("  ");
    for (int x = x0; x <= x1; ++x) {
      size_t i = ((size_t)y * W + x) * 3;
      printf("%s", h[i] == n0 ? " N " : (h[i] == b0 ? " B " : (h[i] == o0 ? " o " : " . ")));
    }
    printf("\n");
  }
}

int main() {
  const NppStreamContext C = make_ctx();
  const int W = 12, H = 10;
  Npp8u *buf = dev_alloc(1 << 22);
  Npp8u *D = dev_alloc((size_t)W * H * 3 + 4096);
  Npp8u nv[3] = {77, 78, 79}, bv[3] = {200, 201, 202};

  // Seed (9,4): correct pixel is in the block (30,40,50); the "byte offset"
  // address y*step+x (x=9 -> byte 9 of row 4) is pixel (4,3) in the field.
  {
    std::vector<Npp8u> h = picture(W, H, 6, 11, 2, 6, 0);
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFillBoundary_8u_C3IR_Ctx(D, 3 * W, NppiPoint{9, 4}, nv, bv, nppiNormL1,
                                                     NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("C3 boundary seed(9,4) -> %s\n", sname(st));
    show("  (N=new B=boundary o=block untouched . =field)", h, W, H, 0, 11, 1, 7, 77, 200, 30);
    printf("  count=%u seed=(%u,%u,%u)\n", reg.nConnectedPixelCount, reg.aSeedPixelValue[0],
           reg.aSeedPixelValue[1], reg.aSeedPixelValue[2]);
  }
  // Seed (11,4): byte offset 4*36+11 = 155 -> pixel (4,3) hmm 155/3=51.67; use (12,4)? out of ROI.
  // Seed (10,4): byte 154 -> pixel (4,3) again (154/3 = 51.3). Use seed (6,4): 4*36+6=150 -> pixel
  // (4,2), which is inside the block.
  {
    std::vector<Npp8u> h = picture(W, H, 6, 11, 2, 6, 0);
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFillBoundary_8u_C3IR_Ctx(D, 3 * W, NppiPoint{6, 4}, nv, bv, nppiNormL1,
                                                     NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("C3 boundary seed(6,4) -> %s\n", sname(st));
    show("  ", h, W, H, 0, 11, 1, 7, 77, 200, 30);
    printf("  count=%u seed=(%u,%u,%u)\n", reg.nConnectedPixelCount, reg.aSeedPixelValue[0],
           reg.aSeedPixelValue[1], reg.aSeedPixelValue[2]);
  }
  // C1 control on the equivalent single-channel picture.
  {
    std::vector<Npp8u> h((size_t)W * H, 10);
    for (int y = 2; y <= 6; ++y)
      for (int x = 6; x <= 11; ++x) h[(size_t)y * W + x] = 30;
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFillBoundary_8u_C1IR_Ctx(D, W, NppiPoint{9, 4}, 77, 200, nppiNormL1,
                                                     NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("C1 control seed(9,4) -> %s count=%u seed=%u\n", sname(st), reg.nConnectedPixelCount,
           reg.aSeedPixelValue[0]);
    for (int y = 1; y <= 7; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) printf("%3d", (int)h[(size_t)y * W + x]);
      printf("\n");
    }
  }
  // C3 plain fill with seed (9,4) (no boundary value).
  {
    std::vector<Npp8u> h = picture(W, H, 6, 11, 2, 6, 0);
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFill_8u_C3IR_Ctx(D, 3 * W, NppiPoint{9, 4}, nv, nppiNormInf,
                                             NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("C3 plain seed(9,4) Inf -> %s\n", sname(st));
    show("  ", h, W, H, 0, 11, 1, 7, 77, 200, 30);
    printf("  count=%u seed=(%u,%u,%u)\n", reg.nConnectedPixelCount, reg.aSeedPixelValue[0],
           reg.aSeedPixelValue[1], reg.aSeedPixelValue[2]);
  }
  // C1 with an odd width (step not a multiple of anything special), seed read check.
  {
    const int W2 = 11, H2 = 8;
    std::vector<Npp8u> h((size_t)W2 * H2, 10);
    for (int y = 2; y <= 5; ++y)
      for (int x = 5; x <= 10; ++x) h[(size_t)y * W2 + x] = 30;
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = nppiFloodFillBoundary_8u_C1IR_Ctx(D, W2, NppiPoint{8, 3}, 77, 200, nppiNormL1,
                                                     NppiSize{W2, H2}, &reg, buf, C);
    printf("C1 W=11 seed(8,3) -> %s count=%u seed=%u\n", sname(st), reg.nConnectedPixelCount,
           reg.aSeedPixelValue[0]);
  }
  cudaFree(buf);
  cudaFree(D);
  return 0;
}
