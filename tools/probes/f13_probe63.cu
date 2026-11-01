// f13: flood fill - 16u C3 seed read, buffer sizes per channel count, status order.
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
  const int W = 12, H = 10;
  Npp8u *buf = dev_alloc(1 << 22);
  Npp8u *D = dev_alloc((size_t)W * H * 8 + 4096);

  // Buffer sizes depend on channels?
  {
    int a = -1, b = -1;
    nppiFloodFillGetBufferSize(NppiSize{W, H}, &a);
    NppiSize r{W, H};
    (void)r;
    b = a;
    printf("GetBufferSize 12x10 = %d; fit 4wh+4w+64 = %d; 12x10 other sizes:", a,
           4 * W * H + 4 * W + 64);
    int s[4];
    int h2 = 0;
    NppiSize t1{3, 7};
    nppiFloodFillGetBufferSize(t1, &h2);
    printf(" 3x7=%d (formula %d)\n", h2, 4 * 3 * 7 + 4 * 3 + 64);
    (void)s;
  }

  // 16u C3: seed read at y*step + x*2 bytes?
  {
    std::vector<Npp16u> h((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        bool in = x >= 6 && x <= 11 && y >= 2 && y <= 6;
        h[i] = (Npp16u)(in ? 300 : 100);
        h[i + 1] = (Npp16u)(in ? 400 : 100);
        h[i + 2] = (Npp16u)(in ? 500 : 100);
      }
    // seed (9,4) -> byte offset 4*(6*12) + 9*2 = 288+18 = 306 -> element 153 = (4,3) field
    Npp16u *D16 = (Npp16u *)D;
    cudaMemcpy(D16, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
    NppiConnectedRegion reg;
    memset(&reg, 0xEE, sizeof reg);
    Npp16u nv[3] = {7, 8, 9}, bv[3] = {20, 21, 22};
    NppStatus st = nppiFloodFillBoundary_16u_C3IR_Ctx(D16, 6 * W, NppiPoint{9, 4}, nv, bv,
                                                      nppiNormL1, NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h.data(), D16, h.size() * 2, cudaMemcpyDeviceToHost);
    printf("16u C3 boundary seed(9,4) -> %s count=%u seed=(%u,%u,%u)\n", sname(st),
           reg.nConnectedPixelCount, reg.aSeedPixelValue[0], reg.aSeedPixelValue[1],
           reg.aSeedPixelValue[2]);
    for (int y = 1; y <= 7; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        printf("%s", h[i] == 7 ? " N " : (h[i] == 20 ? " B " : (h[i] == 300 ? " o " : " . ")));
      }
      printf("\n");
    }
    // and the same with seed (6,4) whose byte offset lands in the block
    std::vector<Npp16u> h2((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        bool in = x >= 6 && x <= 11 && y >= 2 && y <= 6;
        h2[i] = (Npp16u)(in ? 300 : 100);
        h2[i + 1] = (Npp16u)(in ? 400 : 100);
        h2[i + 2] = (Npp16u)(in ? 500 : 100);
      }
    cudaMemcpy(D16, h2.data(), h2.size() * 2, cudaMemcpyHostToDevice);
    memset(&reg, 0xEE, sizeof reg);
    st = nppiFloodFillBoundary_16u_C3IR_Ctx(D16, 6 * W, NppiPoint{6, 4}, nv, bv, nppiNormL1,
                                            NppiSize{W, H}, &reg, buf, C);
    cudaMemcpy(h2.data(), D16, h2.size() * 2, cudaMemcpyDeviceToHost);
    printf("16u C3 boundary seed(6,4) -> %s count=%u seed=(%u,%u,%u)\n", sname(st),
           reg.nConnectedPixelCount, reg.aSeedPixelValue[0], reg.aSeedPixelValue[1],
           reg.aSeedPixelValue[2]);
    for (int y = 1; y <= 7; ++y) {
      printf("  ");
      for (int x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 3;
        printf("%s", h2[i] == 7 ? " N " : (h2[i] == 20 ? " B " : (h2[i] == 300 ? " o " : " . ")));
      }
      printf("\n");
    }
  }

  // Validation order.
  {
    std::vector<Npp8u> h((size_t)W * H, 5);
    cudaMemcpy(D, h.data(), h.size(), cudaMemcpyHostToDevice);
    struct T { const char *n; NppStatus s; };
    std::vector<T> v;
    v.push_back({"nullsrc+zeroROI", nppiFloodFill_8u_C1IR_Ctx(nullptr, W, NppiPoint{1, 1}, 3, nppiNormL1, NppiSize{0, 0}, nullptr, buf, C)});
    v.push_back({"nullbuf+zeroROI", nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 3, nppiNormL1, NppiSize{0, 0}, nullptr, nullptr, C)});
    v.push_back({"nullsrc+seedbad", nppiFloodFill_8u_C1IR_Ctx(nullptr, W, NppiPoint{-5, 1}, 3, nppiNormL1, NppiSize{W, H}, nullptr, buf, C)});
    v.push_back({"seedbad+nullbuf", nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{-5, 1}, 3, nppiNormL1, NppiSize{W, H}, nullptr, nullptr, C)});
    v.push_back({"seedbad+negROI", nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{-5, 1}, 3, nppiNormL1, NppiSize{-2, 3}, nullptr, buf, C)});
    v.push_back({"normbad+seedbad", nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{-5, 1}, 3, nppiNormL2, NppiSize{W, H}, nullptr, buf, C)});
    v.push_back({"normbad+zeroROI", nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 3, nppiNormL2, NppiSize{0, 0}, nullptr, buf, C)});
    v.push_back({"step0+seedbad", nppiFloodFill_8u_C1IR_Ctx(D, 0, NppiPoint{-5, 1}, 3, nppiNormL1, NppiSize{W, H}, nullptr, buf, C)});
    v.push_back({"step0+normbad", nppiFloodFill_8u_C1IR_Ctx(D, 0, NppiPoint{1, 1}, 3, nppiNormL2, NppiSize{W, H}, nullptr, buf, C)});
    v.push_back({"step0+zeroROI", nppiFloodFill_8u_C1IR_Ctx(D, 0, NppiPoint{1, 1}, 3, nppiNormL1, NppiSize{0, 0}, nullptr, buf, C)});
    v.push_back({"nullbuf+normbad", nppiFloodFill_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 3, nppiNormL2, NppiSize{W, H}, nullptr, nullptr, C)});
    v.push_back({"nullbuf+step0", nppiFloodFill_8u_C1IR_Ctx(D, 0, NppiPoint{1, 1}, 3, nppiNormL1, NppiSize{W, H}, nullptr, nullptr, C)});
    v.push_back({"range nullbuf", nppiFloodFillRange_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 0, 1, 3, nppiNormL1, NppiSize{W, H}, nullptr, nullptr, C)});
    v.push_back({"range min>max", nppiFloodFillRange_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 9, 2, 3, nppiNormL1, NppiSize{W, H}, nullptr, buf, C)});
    v.push_back({"grad min>max", nppiFloodFillGradient_8u_C1IR_Ctx(D, W, NppiPoint{1, 1}, 9, 2, 3, nppiNormL1, NppiSize{W, H}, nullptr, buf, C)});
    for (auto &t : v) printf("order %-18s -> %s\n", t.n, sname(t.s));
  }
  cudaFree(buf);
  cudaFree(D);
  return 0;
}
