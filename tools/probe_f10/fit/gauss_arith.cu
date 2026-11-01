// gauss_arith: pin down NPP 13.4 Gauss kernel generation and integer emission
// arithmetic for square masks 3..15.  Standalone probe (not part of the build).
#include <npp.h>
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

// Generate a Gaussian 2D kernel on the device.
//  mode 0: 1D float32 expf + float32 normalise + float32 outer product
//  mode 1: 2D float32 expf + float32 normalise
//  mode 2: 1D double exp + double normalise, cast to float
//  mode 3: 1D float32 expf, double normalise, cast
__global__ void gen_kernel(float *K, int taps, float sigma, int mode) {
  int tid = threadIdx.x;
  const int n1 = 64;
  __shared__ float w1[n1];
  int r = taps / 2;
  if (mode == 1) {
    if (tid < taps * taps) {
      int j = tid / taps, i = tid % taps;
      float dx = (float)(i - r), dy = (float)(j - r);
      K[tid] = __expf(-(dx * dx + dy * dy) / (2.f * sigma * sigma));
    }
    __syncthreads();
    if (tid == 0) {
      float s = 0.f;
      for (int k = 0; k < taps * taps; ++k) s += K[k];
      for (int k = 0; k < taps * taps; ++k) K[k] = K[k] / s;
    }
    return;
  }
  if (tid < taps) {
    float d = (float)(tid - r);
    if (mode == 0 || mode == 3) w1[tid] = __expf(-(d * d) / (2.f * sigma * sigma));
    else w1[tid] = (float)exp(-((double)d * d) / (2.0 * (double)sigma * (double)sigma));
  }
  __syncthreads();
  if (tid == 0) {
    if (mode == 0) {
      float s = 0.f; for (int k = 0; k < taps; ++k) s += w1[k];
      for (int k = 0; k < taps; ++k) w1[k] = w1[k] / s;
    } else if (mode == 2) {
      double s = 0; for (int k = 0; k < taps; ++k) s += (double)w1[k];
      for (int k = 0; k < taps; ++k) w1[k] = (float)((double)w1[k] / s);
    } else {
      double s = 0; for (int k = 0; k < taps; ++k) s += (double)w1[k];
      for (int k = 0; k < taps; ++k) w1[k] = (float)((double)w1[k] / s);
    }
  }
  __syncthreads();
  if (tid < taps * taps) {
    int j = tid / taps, i = tid % taps;
    K[tid] = w1[j] * w1[i];
  }
}

template <typename S, typename D, typename Acc, bool RowMajor>
__global__ void model_kernel(const S *src, int sstep, D *dst, int dstep, int W, int H, const float *K, int taps,
                             int r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *sb = reinterpret_cast<const char *>(src);
  Acc acc = Acc(0);
  if constexpr (RowMajor) {
    for (int j = 0; j < taps; ++j) {
      const S *row = reinterpret_cast<const S *>(sb + (long)(y + j - r) * sstep);
      for (int i = 0; i < taps; ++i) acc += Acc(K[j * taps + i]) * Acc(row[x + i - r]);
    }
  } else {
    for (int i = 0; i < taps; ++i)
      for (int j = 0; j < taps; ++j) {
        const S *row = reinterpret_cast<const S *>(sb + (long)(y + j - r) * sstep);
        acc += Acc(K[j * taps + i]) * Acc(row[x + i - r]);
      }
  }
  D v;
  if constexpr (std::is_floating_point_v<D>) {
    v = D(acc);
  } else {
    double d = double(acc);
    double hi = double(std::is_same_v<D, Npp8u> ? 255 : (std::is_same_v<D, Npp16u> ? 65535 : 32767));
    double lo = std::is_same_v<D, Npp16s> ? -32768 : 0;
    if (d > hi) d = hi;
    if (d < lo) d = lo;
    v = D(d);
  }
  *reinterpret_cast<D *>(reinterpret_cast<char *>(dst) + (long)y * dstep + (long)x * (int)sizeof(D)) = v;
}

template <typename S, typename D, typename Acc, bool RM>
static void run_model(const S *ds, int sstep, D *dd, int dstep, int W, int H, const float *dk, int taps, int r) {
  dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);
  model_kernel<S, D, Acc, RM><<<grd, blk>>>(ds, sstep, dd, dstep, W, H, dk, taps, r);
  cudaDeviceSynchronize();
}

static int taps_of(int ms) {
  if (ms == 200) return 3;
  if (ms == 201) return 5;
  return ms / 100 + 2;  // 400->6? guarded below
}

int main() {
  const int M = 20;
  const NppiSize roi{64, 48};
  const int BW = roi.width + 2 * M, BH = roi.height + 2 * M;
  NppStreamContext c = ctx();
  const int masks[] = {200, 201, 400, 500, 600, 700, 800};
  const int taps_by_mask[] = {3, 5, 7, 9, 11, 13, 15};

  std::vector<float> fdelta((size_t)BW * BH, 0.f), fout((size_t)BW * BH, -1.f);
  const int cy = BH / 2, cx = BW / 2;
  fdelta[(size_t)cy * BW + cx] = 1.f;
  float *d_fd, *d_fo;
  cudaMalloc(&d_fd, fdelta.size() * 4);
  cudaMalloc(&d_fo, fout.size() * 4);
  cudaMemcpy(d_fd, fdelta.data(), fdelta.size() * 4, cudaMemcpyHostToDevice);

  std::vector<short> i16((size_t)BW * BH);
  std::vector<unsigned short> u16((size_t)BW * BH);
  std::vector<unsigned char> u8((size_t)BW * BH);
  unsigned st = 12345;
  auto next = [&]() {
    st = st * 1103515245u + 12345u;
    return (st >> 16) & 0x7FFF;
  };
  for (size_t i = 0; i < i16.size(); ++i) {
    i16[i] = (short)((int)(next() % 4001) - 2000);
    u16[i] = (unsigned short)(next() % 60001);
    u8[i] = (unsigned char)(next() & 0xFF);
  }
  short *d_s16, *d_d16;
  unsigned short *d_u16, *d_du16;
  unsigned char *d_u8, *d_du8;
  cudaMalloc(&d_s16, i16.size() * 2); cudaMalloc(&d_d16, (size_t)roi.width * roi.height * 2);
  cudaMalloc(&d_u16, u16.size() * 2); cudaMalloc(&d_du16, (size_t)roi.width * roi.height * 2);
  cudaMalloc(&d_u8, u8.size()); cudaMalloc(&d_du8, (size_t)roi.width * roi.height);
  cudaMemcpy(d_s16, i16.data(), i16.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u16, u16.data(), u16.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u8, u8.data(), u8.size(), cudaMemcpyHostToDevice);
  short *s16p = d_s16 + (size_t)M * BW + M;
  unsigned short *u16p = d_u16 + (size_t)M * BW + M;
  unsigned char *u8p = d_u8 + (size_t)M * BW + M;

  for (int mi = 0; mi < 7; ++mi) {
    int ms = masks[mi], taps = taps_by_mask[mi], r = taps / 2;
    cudaMemcpy(d_fo, fout.data(), fout.size() * 4, cudaMemcpyDeviceToDevice);
    const int oy = roi.height / 2, ox = roi.width / 2;
    NppStatus s32 = nppiFilterGauss_32f_C1R_Ctx(
        d_fd + (size_t)(cy - oy) * BW + (cx - ox), BW * 4, d_fo, BW * 4, roi, (NppiMaskSize)ms, c);
    cudaDeviceSynchronize();
    std::vector<float> of((size_t)BW * BH, -1.f);
    cudaMemcpy(of.data(), d_fo, of.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> K2((size_t)taps * taps);
    for (int j = 0; j < taps; ++j)
      for (int i = 0; i < taps; ++i)
        K2[j * taps + i] = of[(size_t)(oy + r - j) * BW + (ox + r - i)];
    double ksum = 0;
    for (float v : K2) ksum += v;
    printf("mask=%d taps=%d status=%d Kc=%.9g sum=%.9g\n", ms, taps, (int)s32, K2[(taps / 2) * taps + taps / 2],
           ksum);

    float *d_k2;
    cudaMalloc(&d_k2, K2.size() * 4);
    cudaMemcpy(d_k2, K2.data(), K2.size() * 4, cudaMemcpyHostToDevice);
    float *d_kg;
    cudaMalloc(&d_kg, K2.size() * 4);
    float sigma = 0.6f + 0.4f * (float)r;
    for (int mode = 0; mode < 4; ++mode) {
      gen_kernel<<<1, 256>>>(d_kg, taps, sigma, mode);
      cudaDeviceSynchronize();
      std::vector<float> kg(K2.size());
      cudaMemcpy(kg.data(), d_kg, kg.size() * 4, cudaMemcpyDeviceToHost);
      int diff = 0;
      float maxd = 0;
      for (size_t k = 0; k < kg.size(); ++k) {
        float d = fabsf(kg[k] - K2[k]);
        if (d > maxd) maxd = d;
        if (kg[k] != K2[k]) ++diff;
      }
      printf("  kgen mode=%d vs npp: diff=%d maxd=%.3g\n", mode, diff, maxd);
    }

    std::vector<short> o16((size_t)roi.width * roi.height);
    std::vector<unsigned short> ou16((size_t)roi.width * roi.height);
    std::vector<unsigned char> ou8((size_t)roi.width * roi.height);
    nppiFilterGauss_16s_C1R_Ctx(s16p, BW * 2, d_d16, roi.width * 2, roi, (NppiMaskSize)ms, c);
    nppiFilterGauss_16u_C1R_Ctx(u16p, BW * 2, d_du16, roi.width * 2, roi, (NppiMaskSize)ms, c);
    nppiFilterGauss_8u_C1R_Ctx(u8p, BW, d_du8, roi.width, roi, (NppiMaskSize)ms, c);
    cudaDeviceSynchronize();
    cudaMemcpy(o16.data(), d_d16, o16.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ou16.data(), d_du16, ou16.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ou8.data(), d_du8, ou8.size(), cudaMemcpyDeviceToHost);
    (void)taps_of;

    auto compare = [](const char *tag, auto *npp, auto *mod, size_t n) {
      size_t bad = 0;
      double maxd = 0;
      for (size_t k = 0; k < n; ++k) {
        double d = fabs(double(npp[k]) - double(mod[k]));
        if (d > maxd) maxd = d;
        if (npp[k] != mod[k]) ++bad;
      }
      printf("  %-22s bad=%zu/%zu maxd=%.6g\n", tag, bad, n, maxd);
    };
    auto dumpfew = [&](const char *tag, const short *npp, const short *mod, int W, int H) {
      int shown = 0;
      for (int y = 0; y < H && shown < 4; ++y)
        for (int x = 0; x < W && shown < 4; ++x) {
          int k = y * W + x;
          if (npp[k] != mod[k]) {
            printf("    %s (%d,%d) npp=%d model=%d\n", tag, x, y, (int)npp[k], (int)mod[k]);
            ++shown;
          }
        }
    };

    short *dm16; unsigned short *dmu16; unsigned char *dmu8;
    cudaMalloc(&dm16, o16.size() * 2);
    cudaMalloc(&dmu16, ou16.size() * 2);
    cudaMalloc(&dmu8, ou8.size());
    std::vector<short> m16(o16.size());
    std::vector<unsigned short> mu16(ou16.size());
    std::vector<unsigned char> mu8(ou8.size());
    run_model<short, short, float, true>(s16p, BW * 2, dm16, roi.width * 2, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(m16.data(), dm16, o16.size() * 2, cudaMemcpyDeviceToHost);
    compare("16s float rowmajor", o16.data(), m16.data(), o16.size());
    dumpfew("16s", o16.data(), m16.data(), roi.width, roi.height);
    if (ms == 200) {
      for (int mode = 0; mode < 4; ++mode) {
        std::vector<short> alt(i16);
        unsigned s2 = 999;
        for (size_t k = 0; k < alt.size(); ++k) {
          s2 = s2 * 1103515245u + 12345u;
          int rv = (int)((s2 >> 16) & 0x1FFF);
          if (mode == 0) alt[k] = (short)(rv % 2001);
          else if (mode == 1) alt[k] = (short)(-(rv % 2001));
          else if (mode == 2) alt[k] = (short)((rv % 401) - 200);
          else alt[k] = (short)((rv % 40001) - 20000);
        }
        cudaMemcpy(d_s16, alt.data(), alt.size() * 2, cudaMemcpyHostToDevice);
        nppiFilterGauss_16s_C1R_Ctx(s16p, BW * 2, d_d16, roi.width * 2, roi, (NppiMaskSize)ms, c);
        cudaDeviceSynchronize();
        std::vector<short> o16b(o16.size());
        cudaMemcpy(o16b.data(), d_d16, o16b.size() * 2, cudaMemcpyDeviceToHost);
        run_model<short, short, float, true>(s16p, BW * 2, dm16, roi.width * 2, roi.width, roi.height, d_k2, taps, r);
        cudaMemcpy(m16.data(), dm16, o16.size() * 2, cudaMemcpyDeviceToHost);
        char tg[64];
        snprintf(tg, sizeof tg, "16s alt mode%d", mode);
        compare(tg, o16b.data(), m16.data(), o16b.size());
        dumpfew(tg, o16b.data(), m16.data(), roi.width, roi.height);
        cudaMemcpy(d_s16, i16.data(), i16.size() * 2, cudaMemcpyHostToDevice);
      }
    }
    run_model<short, short, double, true>(s16p, BW * 2, dm16, roi.width * 2, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(m16.data(), dm16, o16.size() * 2, cudaMemcpyDeviceToHost);
    compare("16s double rowmajor", o16.data(), m16.data(), o16.size());
    run_model<short, short, float, false>(s16p, BW * 2, dm16, roi.width * 2, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(m16.data(), dm16, o16.size() * 2, cudaMemcpyDeviceToHost);
    compare("16s float colmajor", o16.data(), m16.data(), o16.size());
    run_model<unsigned short, unsigned short, float, true>(u16p, BW * 2, dmu16, roi.width * 2, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(mu16.data(), dmu16, ou16.size() * 2, cudaMemcpyDeviceToHost);
    compare("16u float rowmajor", ou16.data(), mu16.data(), ou16.size());
    run_model<unsigned short, unsigned short, double, true>(u16p, BW * 2, dmu16, roi.width * 2, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(mu16.data(), dmu16, ou16.size() * 2, cudaMemcpyDeviceToHost);
    compare("16u double rowmajor", ou16.data(), mu16.data(), ou16.size());
    run_model<unsigned char, unsigned char, float, true>(u8p, BW, dmu8, roi.width, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(mu8.data(), dmu8, ou8.size(), cudaMemcpyDeviceToHost);
    compare("8u float rowmajor", ou8.data(), mu8.data(), ou8.size());
    run_model<unsigned char, unsigned char, double, true>(u8p, BW, dmu8, roi.width, roi.width, roi.height, d_k2, taps, r);
    cudaMemcpy(mu8.data(), dmu8, ou8.size(), cudaMemcpyDeviceToHost);
    compare("8u double rowmajor", ou8.data(), mu8.data(), ou8.size());
    for (int mode = 0; mode < 4; ++mode) {
      gen_kernel<<<1, 256>>>(d_kg, taps, sigma, mode);
      cudaDeviceSynchronize();
      run_model<unsigned short, unsigned short, float, true>(u16p, BW * 2, dmu16, roi.width * 2, roi.width,
                                                             roi.height, d_kg, taps, r);
      cudaMemcpy(mu16.data(), dmu16, ou16.size() * 2, cudaMemcpyDeviceToHost);
      char tg[64];
      snprintf(tg, sizeof tg, "16u gen%d float rm", mode);
      compare(tg, ou16.data(), mu16.data(), ou16.size());
      run_model<unsigned char, unsigned char, float, true>(u8p, BW, dmu8, roi.width, roi.width, roi.height,
                                                           d_kg, taps, r);
      cudaMemcpy(mu8.data(), dmu8, ou8.size(), cudaMemcpyDeviceToHost);
      snprintf(tg, sizeof tg, "8u gen%d float rm", mode);
      compare(tg, ou8.data(), mu8.data(), ou8.size());
      run_model<short, short, float, true>(s16p, BW * 2, dm16, roi.width * 2, roi.width, roi.height, d_kg, taps, r);
      cudaMemcpy(m16.data(), dm16, o16.size() * 2, cudaMemcpyDeviceToHost);
      snprintf(tg, sizeof tg, "16s gen%d float rm", mode);
      compare(tg, o16.data(), m16.data(), o16.size());
    }
    cudaFree(d_k2); cudaFree(d_kg); cudaFree(dm16); cudaFree(dmu16); cudaFree(dmu8);
  }
  return 0;
}
