// f13: clean flood-fill behaviour probes (NPP 13.4, RTX 5090): buffer-size sweep,
// plain vs boundary fill on clean synthetic pictures.
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

static void run(const char *tag, std::vector<Npp8u> h, int W, int H, NppiPoint seed, int mode,
                Npp8u newv, Npp8u bnd, Npp8u minv, Npp8u maxv, NppiNorm norm, Npp8u *D,
                Npp8u *buf, const NppStreamContext &C) {
  cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
  NppiConnectedRegion reg;
  memset(&reg, 0xEE, sizeof reg);
  NppStatus st;
  const NppiSize roi{W, H};
  switch (mode) {
    case 0: st = nppiFloodFill_8u_C1IR_Ctx(D, W, seed, newv, norm, roi, &reg, buf, C); break;
    case 1:
      st = nppiFloodFillBoundary_8u_C1IR_Ctx(D, W, seed, newv, bnd, norm, roi, &reg, buf, C);
      break;
    case 2:
      st = nppiFloodFillRange_8u_C1IR_Ctx(D, W, seed, minv, maxv, newv, norm, roi, &reg, buf, C);
      break;
    default:
      st = nppiFloodFillGradient_8u_C1IR_Ctx(D, W, seed, minv, maxv, newv, norm, roi, &reg, buf, C);
      break;
  }
  cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
  printf("%s -> %s\n", tag, sname(st));
  pr("  after", h, W, H);
  printf("  region box=(%d,%d,%d,%d) count=%u seed=(%u,%u,%u)\n", reg.oBoundingBox.x,
         reg.oBoundingBox.y, reg.oBoundingBox.width, reg.oBoundingBox.height,
         reg.nConnectedPixelCount, reg.aSeedPixelValue[0], reg.aSeedPixelValue[1],
         reg.aSeedPixelValue[2]);
}

int main() {
  const NppStreamContext C = make_ctx();
  const int W = 10, H = 10;
  Npp8u *buf = dev_alloc(1 << 22);
  Npp8u *D = dev_alloc((size_t)W * H + 4096);

  printf("GetBufferSize sweep (a -1 means negative):\n");
  for (int w = 1; w <= 40; ++w) {
    int v = -999;
    NppStatus st = nppiFloodFillGetBufferSize(NppiSize{w, w}, &v);
    printf("  %2dx%2d -> %s %d%s", w, w, sname(st), v, (w % 4) ? " " : "\n");
  }
  int v1 = -999, v2 = -999, v3 = -999;
  nppiFloodFillGetBufferSize(NppiSize{1, 64}, &v1);
  nppiFloodFillGetBufferSize(NppiSize{64, 1}, &v2);
  nppiFloodFillGetBufferSize(NppiSize{2, 8}, &v3);
  printf("  1x64 -> %d, 64x1 -> %d, 2x8 -> %d\n", v1, v2, v3);

  // A: ring on the ROI border, interior uniform.
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    for (int i = 0; i < W; ++i) { h[i] = 200; h[(size_t)(H - 1) * W + i] = 200; }
    for (int y = 0; y < H; ++y) { h[(size_t)y * W] = 200; h[(size_t)y * W + W - 1] = 200; }
    pr("A before (border ring)", h, W, H);
    run("A border ring, boundary=200 seed(4,4) new=33", h, W, H, NppiPoint{4, 4}, 1, 33, 200, 0, 0,
        nppiNormL1, D, buf, C);
  }
  // B: ring one pixel inside the ROI, interior uniform.
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    for (int i = 1; i < W - 1; ++i) { h[1 * W + i] = 200; h[(size_t)(H - 2) * W + i] = 200; }
    for (int y = 1; y < H - 1; ++y) { h[(size_t)y * W + 1] = 200; h[(size_t)y * W + W - 2] = 200; }
    pr("B before (inner ring)", h, W, H);
    run("B inner ring, boundary=200 seed(4,4) new=33", h, W, H, NppiPoint{4, 4}, 1, 33, 200, 0, 0,
        nppiNormL1, D, buf, C);
  }
  // C: no boundary at all, plain fill of a solid rectangle inside a different field.
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 2; y < 7; ++y)
      for (int x = 2; x < 8; ++x) h[(size_t)y * W + x] = 5;
    pr("C before (rect of 5 in field 1)", h, W, H);
    run("C plain seed(4,4) new=33", h, W, H, NppiPoint{4, 4}, 0, 33, 0, 0, 0, nppiNormL1, D, buf, C);
    run("C boundary=200 (no 200 in image) seed(4,4) new=33", h, W, H, NppiPoint{4, 4}, 1, 33, 200, 0,
        0, nppiNormL1, D, buf, C);
  }
  // D: interior with two values, boundary ring.
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    for (int y = 3; y < 7; ++y)
      for (int x = 3; x < 7; ++x) h[(size_t)y * W + x] = 6;
    for (int i = 1; i < W - 1; ++i) { h[1 * W + i] = 200; h[(size_t)(H - 2) * W + i] = 200; }
    for (int y = 1; y < H - 1; ++y) { h[(size_t)y * W + 1] = 200; h[(size_t)y * W + W - 2] = 200; }
    pr("D before (ring + inner 6)", h, W, H);
    run("D boundary=200 seed(4,4)=6 new=33", h, W, H, NppiPoint{4, 4}, 1, 33, 200, 0, 0, nppiNormL1, D,
        buf, C);
  }
  // E: range fill inside the ring.
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    for (int y = 3; y < 7; ++y)
      for (int x = 3; x < 7; ++x) h[(size_t)y * W + x] = 6;
    for (int i = 1; i < W - 1; ++i) { h[1 * W + i] = 200; h[(size_t)(H - 2) * W + i] = 200; }
    for (int y = 1; y < H - 1; ++y) { h[(size_t)y * W + 1] = 200; h[(size_t)y * W + W - 2] = 200; }
    run("E range seed(4,4)=6 min=4 max=8 new=33", h, W, H, NppiPoint{4, 4}, 2, 33, 0, 4, 8,
        nppiNormL1, D, buf, C);
  }
  cudaFree(buf);
  cudaFree(D);
  return 0;
}
