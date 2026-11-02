// Probe the effective dst->src map of nppiWarpAffineQuad on the reference.
//
// A 32f source ramp f(x,y)=y (then f(x,y)=x) is warped with linear
// interpolation.  For an affine map the bilinear sample of a linear ramp is
// the sampled coordinate itself, so the dst values expose the reference's
// float map (including its rounding) without relying on SASS.
//
// Cases:
//   inner - the failing NppiWarpAffineQuadDiff.Packed_8u roi_offset quads.
//   para  - the parallelogram quads, where dst (3,11) maps exactly to the
//           source ROI's bottom edge src (8,12).
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

#define CK(x)                                                                                                      \
  do {                                                                                                             \
    cudaError_t e_ = (x);                                                                                          \
    if (e_ != cudaSuccess) {                                                                                       \
      printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__);                                                \
      return 1;                                                                                                    \
    }                                                                                                              \
  } while (0)

static unsigned bits(float f) {
  unsigned u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

struct QuadCase {
  const char *name;
  double src[4][2];
  double dst[4][2];
  NppiRect srcRoi;
  NppiRect dstRoi;
};

int main() {
  const int W = 17, H = 13, step = 84;  // bytes: 17*4 + 16 pad
  float *hs = (float *)malloc(step * H);
  float *hd = (float *)malloc(step * H);

  const QuadCase cases[2] = {
      {"inner",
       {{2, 3}, {14, 3}, {14, 10}, {2, 10}},
       {{1, 2}, {12, 2}, {12, 8}, {1, 8}},
       {2, 3, 13, 8},
       {1, 2, 12, 7}},
      {"para",
       {{0, 0}, {16, 0}, {16, 12}, {0, 12}},
       {{0, 0}, {14, 2}, {10, 12}, {-4, 10}},
       {0, 0, 17, 13},
       {0, 0, 17, 13}},
  };

  for (const QuadCase &qc : cases) {
    printf("### case %s\n", qc.name);
    for (int mode = 0; mode < 2; ++mode) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          hs[(y * step) / 4 + x] = (float)(mode == 0 ? y : x);
          hd[(y * step) / 4 + x] = -12345.0f;
        }
      }
      float *ds = nullptr;
      float *dd = nullptr;
      CK(cudaMalloc(&ds, step * H));
      CK(cudaMalloc(&dd, step * H));
      CK(cudaMemcpy(ds, hs, step * H, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, step * H, cudaMemcpyHostToDevice));

      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      NppStatus st = nppiWarpAffineQuad_32f_C1R_Ctx(ds, NppiSize{W, H}, step, qc.srcRoi, qc.src, dd, step,
                                                    qc.dstRoi, qc.dst, NPPI_INTER_LINEAR, ctx);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hd, dd, step * H, cudaMemcpyDeviceToHost));
      CK(cudaFree(ds));
      CK(cudaFree(dd));

      printf("--- mode=%d (%s ramp) status=%d\n", mode, mode == 0 ? "y" : "x", (int)st);
      for (int y = 0; y < H; ++y) {
        printf("y=%2d:", y);
        for (int x = 0; x < W; ++x) {
          float v = hd[(y * step) / 4 + x];
          if (v == -12345.0f) {
            printf(" SKIP");
          } else {
            printf(" %.9g/0x%08x", v, bits(v));
          }
        }
        printf("\n");
      }
    }
  }
  return 0;
}
