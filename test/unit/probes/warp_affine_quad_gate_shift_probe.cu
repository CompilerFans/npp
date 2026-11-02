// Probe the nppiWarpAffineQuad ROI gate precision by shifting the destination
// quad vertically by tiny amounts and reporting whether the boundary pixel is
// written.
//
// inner: dst (1,2) maps to the source ROI's lower edge y=3; the reference
//        rejects it at shift 0.
// para : dst (3,11) maps to the source ROI's upper edge y=12; the reference
//        writes it at shift 0 even though a double evaluation of the
//        float-stored map is 12.0000004.
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

struct QuadCase {
  const char *name;
  double src[4][2];
  double dst[4][2];
  NppiRect srcRoi;
  NppiRect dstRoi;
  int px, py;
};

int main() {
  const int W = 17, H = 13, step = 33;
  unsigned char hs[33 * 13], hd[33 * 13];
  for (int i = 0; i < 33 * 13; ++i) {
    hs[i] = 0x11;
    hd[i] = 0xEE;
  }
  unsigned char *ds = nullptr, *dd = nullptr;
  CK(cudaMalloc(&ds, sizeof(hs)));
  CK(cudaMalloc(&dd, sizeof(hs)));

  const QuadCase cases[3] = {
      {"inner",
       {{2, 3}, {14, 3}, {14, 10}, {2, 10}},
       {{1, 2}, {12, 2}, {12, 8}, {1, 8}},
       {2, 3, 13, 8},
       {1, 2, 12, 7},
       1,
       2},
      {"para",
       {{0, 0}, {16, 0}, {16, 12}, {0, 12}},
       {{0, 0}, {14, 2}, {10, 12}, {-4, 10}},
       {0, 0, 17, 13},
       {0, 0, 17, 13},
       3,
       11},
      {"rect",
       {{0, 0}, {16, 0}, {16, 12}, {0, 12}},
       {{1, 1}, {15, 1}, {15, 11}, {1, 11}},
       {0, 0, 17, 13},
       {0, 0, 17, 13},
       15,
       1},
  };
  const double shifts[] = {0.0,     1e-9,    5e-9,    1e-8,   2e-8,   5e-8,    1e-7,
                           2e-7,    5e-7,    1e-6,    2e-6,   5e-6,   1e-5,    1e-4,
                           -1e-9,   -5e-9,   -1e-8,   -5e-8,  -1e-7,  -2e-7,  -5e-7, -1e-6};

  for (const QuadCase &qc : cases) {
    printf("### %s (pixel %d,%d)\n", qc.name, qc.px, qc.py);
    for (double sh : shifts) {
      double dq[4][2];
      std::memcpy(dq, qc.dst, sizeof(dq));
      for (int i = 0; i < 4; ++i) {
        dq[i][1] += sh;
      }
      CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, sizeof(hs), cudaMemcpyHostToDevice));
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      NppStatus st = nppiWarpAffineQuad_8u_C1R_Ctx(ds, NppiSize{W, H}, step, qc.srcRoi, qc.src, dd, step,
                                                   qc.dstRoi, dq, NPPI_INTER_NN, ctx);
      CK(cudaDeviceSynchronize());
      unsigned char out[33 * 13];
      CK(cudaMemcpy(out, dd, sizeof(out), cudaMemcpyDeviceToHost));
      unsigned char v = out[qc.py * step + qc.px];
      printf("  shift=% .12g status=%d pixel=%s\n", sh, (int)st, v == 0xEE ? "SKIP" : "WRITE");
    }
  }
  CK(cudaFree(ds));
  CK(cudaFree(dd));
  return 0;
}
