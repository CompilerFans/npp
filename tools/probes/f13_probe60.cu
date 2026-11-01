// f13: pin down the FloodFillBoundary "region outline" rule and the
// Range/Gradient predicates for C3 (NPP 13.4, RTX 5090).
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

static void pr(const char *tag, const std::vector<Npp8u> &h, int w, int ht) {
  printf("%s\n", tag);
  for (int y = 0; y < ht; ++y) {
    printf("  ");
    for (int x = 0; x < w; ++x) printf("%3d", (int)h[(size_t)y * w + x]);
    printf("\n");
  }
}

static Npp8u *dev_alloc(size_t n) {
  Npp8u *p = nullptr;
  if (cudaMalloc(&p, n) != cudaSuccess) { printf("cudaMalloc failed\n"); exit(1); }
  return p;
}

// mode: 0 plain, 1 boundary, 2 range, 3 gradient
static void run8(const char *tag, std::vector<Npp8u> h, int W, int H, NppiPoint seed, int mode,
                 Npp8u newv, Npp8u bnd, Npp8u minv, Npp8u maxv, NppiNorm norm, Npp8u *D,
                 Npp8u *buf, const NppStreamContext &C, bool print = true) {
  cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
  NppiConnectedRegion reg;
  memset(&reg, 0xEE, sizeof reg);
  NppStatus st;
  const NppiSize roi{W, H};
  switch (mode) {
    case 0: st = nppiFloodFill_8u_C1IR_Ctx(D, W, seed, newv, norm, roi, &reg, buf, C); break;
    case 1: st = nppiFloodFillBoundary_8u_C1IR_Ctx(D, W, seed, newv, bnd, norm, roi, &reg, buf, C); break;
    case 2: st = nppiFloodFillRange_8u_C1IR_Ctx(D, W, seed, minv, maxv, newv, norm, roi, &reg, buf, C); break;
    default: st = nppiFloodFillGradient_8u_C1IR_Ctx(D, W, seed, minv, maxv, newv, norm, roi, &reg, buf, C); break;
  }
  cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
  printf("%s -> %s\n", tag, sname(st));
  if (print) pr("  after", h, W, H);
  printf("  region box=(%d,%d,%d,%d) count=%u seed=(%u)\n", reg.oBoundingBox.x, reg.oBoundingBox.y,
         reg.oBoundingBox.width, reg.oBoundingBox.height, reg.nConnectedPixelCount,
         reg.aSeedPixelValue[0]);
}

int main() {
  const NppStreamContext C = make_ctx();
  const int W = 10, H = 10;
  Npp8u *buf = dev_alloc(1 << 22);
  Npp8u *D = dev_alloc((size_t)W * H + 4096);

  // F: 3x3 block -> 4-neighbour ring or 8-neighbour ring?
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 3; y < 6; ++y)
      for (int x = 3; x < 6; ++x) h[(size_t)y * W + x] = 5;
    run8("F 3x3 block L1 boundary=200", h, W, H, NppiPoint{4, 4}, 1, 33, 200, 0, 0, nppiNormL1, D,
         buf, C);
    run8("F 3x3 block Inf boundary=200", h, W, H, NppiPoint{4, 4}, 1, 33, 200, 0, 0, nppiNormInf, D,
         buf, C);
  }
  // G: 1-pixel wide line, Inf connectivity
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int x = 2; x < 8; ++x) h[4 * W + x] = 5;
    run8("G 1px line Inf boundary=200", h, W, H, NppiPoint{5, 4}, 1, 33, 200, 0, 0, nppiNormInf, D,
         buf, C);
  }
  // H: region touching the ROI edge
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 6; ++x) h[(size_t)y * W + x] = 5;
    run8("H region touching ROI corner, boundary=200", h, W, H, NppiPoint{2, 2}, 1, 33, 200, 0, 0,
         nppiNormL1, D, buf, C);
  }
  // I: annulus region of 5 with a hole of 1
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 2; y < 8; ++y)
      for (int x = 2; x < 8; ++x) h[(size_t)y * W + x] = 5;
    h[5 * W + 5] = 1;
    h[4 * W + 4] = 1;
    run8("I ring-shaped region, boundary=200", h, W, H, NppiPoint{2, 2}, 1, 33, 200, 0, 0,
         nppiNormL1, D, buf, C);
  }
  // J: newValue == seedValue
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 3; y < 6; ++y)
      for (int x = 3; x < 6; ++x) h[(size_t)y * W + x] = 5;
    run8("J 3x3 new=5 (=seed)", h, W, H, NppiPoint{4, 4}, 0, 5, 0, 0, 0, nppiNormL1, D, buf, C);
  }
  // K: C3 gradient / range predicate semantics
  {
    std::vector<Npp8u> h((size_t)W * H * 3, 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h[i + 0] = (Npp8u)(10 + x);
        h[i + 1] = (Npp8u)(10 + y);
        h[i + 2] = 200;
      }
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    Npp8u nv[3] = {33, 33, 33}, mn[3] = {12, 12, 150}, mx[3] = {13, 13, 250};
    NppStatus st = nppiFloodFillRange_8u_C3IR_Ctx(D, 3 * W, NppiPoint{5, 5}, mn, mx, nv,
                                                  nppiNormL1, NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("K C3 range min=(12,12,150) max=(13,13,250) seed(5,5) -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        printf("%s", h[i] == 33 ? "x" : ".");
      }
      printf("\n");
    }
    printf("  region count=%u seed=(%u,%u,%u) box=(%d,%d,%d,%d)\n", reg.nConnectedPixelCount,
           reg.aSeedPixelValue[0], reg.aSeedPixelValue[1], reg.aSeedPixelValue[2],
           reg.oBoundingBox.x, reg.oBoundingBox.y, reg.oBoundingBox.width, reg.oBoundingBox.height);

    // gradient C3: seed +/- per-channel
    std::vector<Npp8u> h2((size_t)W * H * 3, 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h2[i + 0] = (Npp8u)(10 + x);
        h2[i + 1] = (Npp8u)(10 + y);
        h2[i + 2] = 200;
      }
    cudaMemcpy(D, h2.data(), h2.size(), cudaMemcpyHostToDevice);
    memset(&reg, 0xEE, sizeof reg);
    st = nppiFloodFillGradient_8u_C3IR_Ctx(D, 3 * W, NppiPoint{5, 5}, mn, mx, nv, nppiNormL1,
                                           NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h2.data(), D, h2.size(), cudaMemcpyDeviceToHost);
    printf("K C3 gradient min=(12,12,150) max=(13,13,250) seed(5,5) -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        printf("%s", h2[i] == 33 ? "x" : ".");
      }
      printf("\n");
    }
  }
  // L: C3 boundary semantics (all-channel equality for the ring rule?)
  {
    std::vector<Npp8u> h((size_t)W * H * 3, 0);
    for (int y = 3; y < 7; ++y)
      for (int x = 3; x < 7; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        h[i + 0] = 5;
        h[i + 1] = 6;
        h[i + 2] = 7;
      }
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    Npp8u nv[3] = {33, 34, 35}, bv[3] = {200, 201, 202};
    NppStatus st = nppiFloodFillBoundary_8u_C3IR_Ctx(D, 3 * W, NppiPoint{4, 4}, nv, bv, nppiNormL1,
                                                     NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("L C3 boundary ring test -> %s\n", sname(st));
    for (int y = 2; y < 8; ++y) {
      printf("  ");
      for (int x = 2; x < 8; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        if (h[i] == 33) printf(" N ");
        else if (h[i] == 200) printf(" B ");
        else printf(" . ");
      }
      printf("\n");
    }
    printf("  count=%u seed=(%u,%u,%u)\n", reg.nConnectedPixelCount, reg.aSeedPixelValue[0],
           reg.aSeedPixelValue[1], reg.aSeedPixelValue[2]);
  }
  cudaFree(buf);
  cudaFree(D);
  return 0;
}
