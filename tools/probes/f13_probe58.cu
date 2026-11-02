// f13: reverse engineering of the nppiFloodFill* families (NPP 13.4, RTX 5090).
// Prints the filled image, the connected-region descriptor and the status codes
// for the plain / Boundary / Range / Gradient variants and their *Boundary forms.
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
  if (cudaMalloc(&p, n) != cudaSuccess) {
    printf("cudaMalloc(%zu) failed: %s\n", n, cudaGetErrorString(cudaGetLastError()));
    exit(1);
  }
  return p;
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


#define FF1(...) nppiFloodFill_8u_C1IR_Ctx(__VA_ARGS__, C)
#define FF3(...) nppiFloodFill_8u_C3IR_Ctx(__VA_ARGS__, C)
#define FFB1(...) nppiFloodFillBoundary_8u_C1IR_Ctx(__VA_ARGS__, C)
#define FFR1(...) nppiFloodFillRange_8u_C1IR_Ctx(__VA_ARGS__, C)
#define FFG1(...) nppiFloodFillGradient_8u_C1IR_Ctx(__VA_ARGS__, C)
#define FFRB1(...) nppiFloodFillRangeBoundary_8u_C1IR_Ctx(__VA_ARGS__, C)
#define FFGB1(...) nppiFloodFillGradientBoundary_8u_C1IR_Ctx(__VA_ARGS__, C)

static void report_region(const char *tag, const NppiConnectedRegion &r) {
  printf("%s region: box=(%d,%d,%d,%d) count=%u seed=(%u,%u,%u)\n", tag, r.oBoundingBox.x,
         r.oBoundingBox.y, r.oBoundingBox.width, r.oBoundingBox.height, r.nConnectedPixelCount,
         r.aSeedPixelValue[0], r.aSeedPixelValue[1], r.aSeedPixelValue[2]);
}

int main() {
  const int W = 12, H = 10;
  const NppStreamContext C = make_ctx();
  int bs = 0;
  printf("GetBufferSize 12x10 -> st=%s size=%d\n",
         sname(nppiFloodFillGetBufferSize(NppiSize{W, H}, &bs)), bs);
  int bs2 = -1;
  printf("GetBufferSize 1x1 -> st=%s size=%d\n",
         sname(nppiFloodFillGetBufferSize(NppiSize{1, 1}, &bs2)), bs2);
  int bs3 = -1;
  printf("GetBufferSize 100x64 -> st=%s size=%d\n",
         sname(nppiFloodFillGetBufferSize(NppiSize{100, 64}, &bs3)), bs3);
  printf("GetBufferSize 0x0 -> st=%s\n", sname(nppiFloodFillGetBufferSize(NppiSize{0, 0}, &bs3)));
  printf("GetBufferSize null -> st=%s\n", sname(nppiFloodFillGetBufferSize(NppiSize{W, H}, nullptr)));
  Npp8u *buf = dev_alloc(1 << 22);
  Npp8u *D = dev_alloc((size_t)W * H + 4096);

  // ---------------------------------------------------------------- plain L1/Inf
  {
    std::vector<Npp8u> h((size_t)W * H, 1);
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x) h[(size_t)y * W + x] = 10;
    h[(size_t)4 * W + 4] = 10;
    h[(size_t)5 * W + 5] = 10;
    h[(size_t)5 * W + 6] = 10;
    h[(size_t)6 * W + 6] = 10;
    h[(size_t)1 * W + 6] = 10;  // isolated
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = FF1(D, W, NppiPoint{1, 1}, 33, nppiNormL1, NppiSize{W, H},
                                        &reg, buf);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("plain L1 seed(1,1) -> %s\n", sname(st));
    pr("  after", h, W, H);
    report_region("  ", reg);

    std::vector<Npp8u> h2((size_t)W * H, 1);
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x) h2[(size_t)y * W + x] = 10;
    h2[(size_t)4 * W + 4] = 10;
    h2[(size_t)5 * W + 5] = 10;
    h2[(size_t)5 * W + 6] = 10;
    h2[(size_t)6 * W + 6] = 10;
    h2[(size_t)1 * W + 6] = 10;
    cudaMemcpy(D, h2.data(), h2.size(), cudaMemcpyHostToDevice);
    memset(&reg, 0xEE, sizeof reg);
    st = FF1(D, W, NppiPoint{1, 1}, 33, nppiNormInf, NppiSize{W, H}, &reg, buf);
    cudaMemcpy(h2.data(), D, h2.size(), cudaMemcpyDeviceToHost);
    printf("plain Inf seed(1,1) -> %s\n", sname(st));
    pr("  after", h2, W, H);
    report_region("  ", reg);
  }

  // --------------------------------------------------------------- C3 comparisons
  {
    std::vector<Npp8u> h((size_t)W * H * 3, 0);
    for (size_t i = 0; i < (size_t)W * H; ++i) {
      h[i * 3 + 0] = 10;
      h[i * 3 + 1] = 10;
      h[i * 3 + 2] = 10;
    }
    auto set = [&](int x, int y, int a, int b, int c) {
      size_t i = (size_t)y * W + x;
      h[i * 3 + 0] = (Npp8u)a;
      h[i * 3 + 1] = (Npp8u)b;
      h[i * 3 + 2] = (Npp8u)c;
    };
    set(4, 4, 11, 10, 10);
    set(4, 5, 10, 9, 10);
    set(5, 5, 10, 10, 12);
    set(3, 3, 0, 0, 0);
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    Npp8u nv[3] = {77, 78, 79};
    NppStatus st = FF3(D, 3 * W, NppiPoint{1, 1}, nv, nppiNormL1, NppiSize{W, H},
                                         &reg, buf);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("plain C3 L1 seed(1,1) -> %s\n", sname(st));
    for (int y = 0; y < H; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x)
        printf("(%3d,%3d,%3d)", (int)h[((size_t)y * W + x) * 3 + 0],
               (int)h[((size_t)y * W + x) * 3 + 1], (int)h[((size_t)y * W + x) * 3 + 2]);
      printf("\n");
    }
    report_region("  ", reg);
  }

  // -------------------------------------------------------------------- boundary
  {
    std::vector<Npp8u> h((size_t)W * H, 7);
    for (int y = 2; y <= 7; ++y) {
      h[(size_t)y * W + 2] = 99;
      h[(size_t)y * W + 8] = 99;
    }
    for (int x = 2; x <= 8; ++x) {
      h[(size_t)2 * W + x] = 99;
      h[(size_t)7 * W + x] = 99;
    }
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    NppStatus st = FFB1(D, W, NppiPoint{5, 5}, 33, 99, nppiNormL1,
                                                 NppiSize{W, H}, &reg, buf);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("boundary seed(5,5) new=33 boundary=99 -> %s\n", sname(st));
    pr("  after", h, W, H);
    report_region("  ", reg);

    std::vector<Npp8u> h2((size_t)W * H, 7);
    for (int y = 2; y <= 7; ++y) {
      h2[(size_t)y * W + 2] = 99;
      h2[(size_t)y * W + 8] = 99;
    }
    for (int x = 2; x <= 8; ++x) {
      h2[(size_t)2 * W + x] = 99;
      h2[(size_t)7 * W + x] = 99;
    }
    cudaMemcpy(D, h2.data(), h2.size(), cudaMemcpyHostToDevice);
    memset(&reg, 0xEE, sizeof reg);
    st = FFB1(D, W, NppiPoint{5, 5}, 33, 7, nppiNormL1, NppiSize{W, H},
                                       &reg, buf);
    cudaMemcpy(h2.data(), D, h2.size(), cudaMemcpyDeviceToHost);
    printf("boundary seed(5,5) new=33 boundary=7 (=interior) -> %s\n", sname(st));
    pr("  after", h2, W, H);
    report_region("  ", reg);

    std::vector<Npp8u> h3((size_t)W * H, 7);
    for (int y = 2; y <= 7; ++y) {
      h3[(size_t)y * W + 2] = 99;
      h3[(size_t)y * W + 8] = 99;
    }
    for (int x = 2; x <= 8; ++x) {
      h3[(size_t)2 * W + x] = 99;
      h3[(size_t)7 * W + x] = 99;
    }
    cudaMemcpy(D, h3.data(), h3.size(), cudaMemcpyHostToDevice);
    memset(&reg, 0xEE, sizeof reg);
    st = FFB1(D, W, NppiPoint{2, 2}, 33, 99, nppiNormL1, NppiSize{W, H},
                                       &reg, buf);
    cudaMemcpy(h3.data(), D, h3.size(), cudaMemcpyDeviceToHost);
    printf("boundary seed on ring (2,2) new=33 boundary=99 -> %s\n", sname(st));
    pr("  after", h3, W, H);
    report_region("  ", reg);
  }

  // ----------------------------------------------------------------------- range
  {
    std::vector<Npp8u> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[(size_t)y * W + x] = (Npp8u)(x + y);
    h[(size_t)5 * W + 5] = 8;
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppStatus st = FFR1(D, W, NppiPoint{5, 5}, 7, 9, 33, nppiNormL1,
                                              NppiSize{W, H}, nullptr, buf);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("range seed(5,5)=8 min=7 max=9 -> %s\n", sname(st));
    pr("  after", h, W, H);

    std::vector<Npp8u> h2((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h2[(size_t)y * W + x] = (Npp8u)(x + y);
    h2[(size_t)5 * W + 5] = 8;
    cudaMemcpy(D, h2.data(), h2.size(), cudaMemcpyHostToDevice);
    st = FFR1(D, W, NppiPoint{5, 5}, 8, 8, 33, nppiNormInf, NppiSize{W, H},
                                    nullptr, buf);
    cudaMemcpy(h2.data(), D, h2.size(), cudaMemcpyDeviceToHost);
    printf("range seed(5,5)=8 min=8 max=8 Inf -> %s\n", sname(st));
    pr("  after", h2, W, H);

    std::vector<Npp8u> h3((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h3[(size_t)y * W + x] = (Npp8u)(x + y);
    h3[(size_t)5 * W + 5] = 8;
    cudaMemcpy(D, h3.data(), h3.size(), cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    st = FFR1(D, W, NppiPoint{5, 5}, 20, 30, 33, nppiNormL1, NppiSize{W, H},
                                    &reg, buf);
    cudaMemcpy(h3.data(), D, h3.size(), cudaMemcpyDeviceToHost);
    printf("range seed(5,5)=8 min=20 max=30 (seed outside) -> %s\n", sname(st));
    pr("  after", h3, W, H);
    report_region("  ", reg);
  }

  // -------------------------------------------------------------------- gradient
  {
    std::vector<Npp8u> h((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[(size_t)y * W + x] = (Npp8u)(x + y);
    h[(size_t)5 * W + 5] = 8;
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppStatus st = FFG1(D, W, NppiPoint{5, 5}, 2, 3, 33, nppiNormL1,
                                                 NppiSize{W, H}, nullptr, buf);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("gradient seed(5,5)=8 min=2 max=3 -> %s\n", sname(st));
    pr("  after", h, W, H);

    std::vector<Npp8u> h2((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h2[(size_t)y * W + x] = (Npp8u)(x + y);
    h2[(size_t)0 * W + 0] = 1;
    cudaMemcpy(D, h2.data(), h2.size(), cudaMemcpyHostToDevice);
    st = FFG1(D, W, NppiPoint{0, 0}, 3, 2, 33, nppiNormL1, NppiSize{W, H},
                                       nullptr, buf);
    cudaMemcpy(h2.data(), D, h2.size(), cudaMemcpyDeviceToHost);
    printf("gradient seed(0,0)=1 min=3 max=2 (underflow) -> %s\n", sname(st));
    pr("  after", h2, W, H);
  }

  // ------------------------------------------------------ range/gradient + boundary
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    for (int y = 2; y <= 7; ++y) {
      h[(size_t)y * W + 2] = 200;
      h[(size_t)y * W + 8] = 200;
    }
    for (int x = 2; x <= 8; ++x) {
      h[(size_t)2 * W + x] = 200;
      h[(size_t)7 * W + x] = 200;
    }
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    NppStatus st = FFRB1(D, W, NppiPoint{5, 5}, 4, 6, 33, 200,
                                                      nppiNormL1, NppiSize{W, H}, nullptr, buf);
    cudaMemcpy(h.data(), D, h.size(), cudaMemcpyDeviceToHost);
    printf("rangeBoundary seed(5,5)=5 min=4 max=6 new=33 boundary=200 -> %s\n", sname(st));
    pr("  after", h, W, H);

    std::vector<Npp8u> h2((size_t)W * H, 5);
    for (int y = 2; y <= 7; ++y) {
      h2[(size_t)y * W + 2] = 200;
      h2[(size_t)y * W + 8] = 200;
    }
    for (int x = 2; x <= 8; ++x) {
      h2[(size_t)2 * W + x] = 200;
      h2[(size_t)7 * W + x] = 200;
    }
    cudaMemcpy(D, h2.data(), h2.size(), cudaMemcpyHostToDevice);
    st = FFGB1(D, W, NppiPoint{5, 5}, 0, 0, 33, 200, nppiNormL1,
                                               NppiSize{W, H}, nullptr, buf);
    cudaMemcpy(h2.data(), D, h2.size(), cudaMemcpyDeviceToHost);
    printf("gradientBoundary seed(5,5)=5 min=0 max=0 new=33 boundary=200 -> %s\n", sname(st));
    pr("  after", h2, W, H);
  }

  // ------------------------------------------------------------------ error matrix
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    struct T {
      const char *n;
      NppStatus s;
    };
    std::vector<T> v;
    v.push_back({"null src", FF1(nullptr, W, NppiPoint{1, 1}, 3, nppiNormL1,
                                                   NppiSize{W, H}, nullptr, buf)});
    v.push_back({"null buf", FF1(D, W, NppiPoint{1, 1}, 3, nppiNormL1,
                                                   NppiSize{W, H}, nullptr, nullptr)});
    v.push_back({"zero roi", FF1(D, W, NppiPoint{1, 1}, 3, nppiNormL1,
                                                   NppiSize{0, 0}, nullptr, buf)});
    v.push_back({"neg roi", FF1(D, W, NppiPoint{1, 1}, 3, nppiNormL1,
                                                  NppiSize{-2, 4}, nullptr, buf)});
    v.push_back({"seed x<0", FF1(D, W, NppiPoint{-1, 1}, 3, nppiNormL1,
                                                   NppiSize{W, H}, nullptr, buf)});
    v.push_back({"seed y<0", FF1(D, W, NppiPoint{1, -1}, 3, nppiNormL1,
                                                   NppiSize{W, H}, nullptr, buf)});
    v.push_back({"seed x==W", FF1(D, W, NppiPoint{W, 1}, 3, nppiNormL1,
                                                    NppiSize{W, H}, nullptr, buf)});
    v.push_back({"seed y==H", FF1(D, W, NppiPoint{1, H}, 3, nppiNormL1,
                                                    NppiSize{W, H}, nullptr, buf)});
    v.push_back({"step 0", FF1(D, 0, NppiPoint{1, 1}, 3, nppiNormL1,
                                                 NppiSize{W, H}, nullptr, buf)});
    v.push_back({"step -W", FF1(D, -W, NppiPoint{1, 1}, 3, nppiNormL1,
                                                  NppiSize{W, H}, nullptr, buf)});
    v.push_back({"step W-1", FF1(D, W - 1, NppiPoint{1, 1}, 3, nppiNormL1,
                                                   NppiSize{W, H}, nullptr, buf)});
    v.push_back({"norm L2", FF1(D, W, NppiPoint{1, 1}, 3, nppiNormL2,
                                                  NppiSize{W, H}, nullptr, buf)});
    v.push_back({"norm 7", FF1(D, W, NppiPoint{1, 1}, 3, (NppiNorm)7,
                                                 NppiSize{W, H}, nullptr, buf)});
    v.push_back({"null dst C3", FF3(nullptr, 3 * W, NppiPoint{1, 1}, nullptr,
                                                      nppiNormL1, NppiSize{W, H}, nullptr, buf)});
    for (auto &t : v) printf("err %-12s -> %s\n", t.n, sname(t.s));
    cudaDeviceSynchronize();
    printf("sync after error matrix: %s\n", cudaGetErrorString(cudaGetLastError()));
  }

  cudaFree(buf);
  cudaFree(D);
  return 0;
}
