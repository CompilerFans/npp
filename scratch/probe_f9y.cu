// Validation matrix for all 13 nppiFilterBoxBorderAdvanced instantiations.
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
#include <vector>

static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_BAD_ARGUMENT_ERROR: return "BAD_ARG";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_OUT_OFF_RANGE_ERROR: return "OUT_OFF";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOTEVEN";
    case NPP_MASK_SIZE_ERROR: return "MASK";
    case NPP_ANCHOR_ERROR: return "ANCHOR";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOTSUP";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR: return "CUDAKERN";
    default: return "OTHER";
  }
}

template <class T, int Ch>
struct Run {
  static NppStatus call(const T *src, int sstep, NppiSize ssize, NppiPoint soff, T *dst, int dstep,
                        NppiSize roi, NppiSize mask, NppiPoint anc, NppiBorderType bt, Npp8u *buf);
};
#define C1(SUF, TY) \
  template <> struct Run<TY, 1> { static NppStatus call(const TY *s, int ss, NppiSize sz, NppiPoint o, TY *d, int ds, NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p) { \
    return nppiFilterBoxBorderAdvanced_##SUF##_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, p, g_ctx); } }
#define C3(SUF, TY) \
  template <> struct Run<TY, 3> { static NppStatus call(const TY *s, int ss, NppiSize sz, NppiPoint o, TY *d, int ds, NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p) { \
    return nppiFilterBoxBorderAdvanced_##SUF##_C3R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, p, g_ctx); } }
#define C4(SUF, TY) \
  template <> struct Run<TY, 4> { static NppStatus call(const TY *s, int ss, NppiSize sz, NppiPoint o, TY *d, int ds, NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p) { \
    return nppiFilterBoxBorderAdvanced_##SUF##_C4R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, p, g_ctx); } }
C1(8u, Npp8u); C3(8u, Npp8u); C4(8u, Npp8u);
C1(16u, Npp16u); C3(16u, Npp16u); C4(16u, Npp16u);
C1(16s, Npp16s); C3(16s, Npp16s); C4(16s, Npp16s);
C1(32f, Npp32f); C3(32f, Npp32f); C4(32f, Npp32f);
template <> struct Run<Npp64f, 1> { static NppStatus call(const Npp64f *s, int ss, NppiSize sz, NppiPoint o, Npp64f *d, int ds, NppiSize r, NppiSize m, NppiPoint a, NppiBorderType b, Npp8u *p) { \
    return nppiFilterBoxBorderAdvanced_64f_C1R_Ctx(s, ss, sz, o, d, ds, r, m, a, b, p, g_ctx); } };

template <class T, int Ch>
static void matrix(const char *name) {
  const int es = (int)sizeof(T) * Ch;
  Npp8u *sb = nullptr, *db = nullptr, *bp = nullptr;
  cudaMalloc(&sb, 1 << 20); cudaMemset(sb, 1, 1 << 20);
  cudaMalloc(&db, 1 << 20); cudaMemset(db, 0, 1 << 20);
  cudaMalloc(&bp, 1 << 20); cudaMemset(bp, 0, 1 << 20);
  const T *src = (const T *)sb; T *dst = (T *)db;
  NppiSize sz{64, 64}, roi{64, 64}, mask{3, 3};
  NppiPoint off{0, 0}, anc{1, 1};
  printf("%-12s es=%2d :", name, es);
#define P(tag, ...) printf(" %s=%s", tag, sn(Run<T, Ch>::call(__VA_ARGS__)))
  P("base", src, 64 * es, sz, off, dst, 64 * es, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("nulS", nullptr, 64 * es, sz, off, dst, 64 * es, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("nulD", src, 64 * es, sz, off, nullptr, 64 * es, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("nulB", src, 64 * es, sz, off, dst, 64 * es, roi, mask, anc, NPP_BORDER_REPLICATE, nullptr);
  P("negROI", src, 64 * es, sz, off, dst, 64 * es, NppiSize{-1, 64}, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("zROI", src, 64 * es, sz, off, dst, 64 * es, NppiSize{0, 0}, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("zROIW", src, 64 * es, sz, off, dst, 64 * es, NppiSize{0, 8}, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("negMsk", src, 64 * es, sz, off, dst, 64 * es, roi, NppiSize{-1, 3}, anc, NPP_BORDER_REPLICATE, bp);
  P("ancBad", src, 64 * es, sz, off, dst, 64 * es, roi, mask, NppiPoint{5, 5}, NPP_BORDER_REPLICATE, bp);
  P("dStSml", src, 64 * es, sz, off, dst, 8, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("dStOdd", src, 64 * es, sz, off, dst, 64 * es - 1, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("dSt0", src, 64 * es, sz, off, dst, 0, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("sStSml", src, 8, sz, off, dst, 64 * es, roi, mask, anc, NPP_BORDER_REPLICATE, bp);
  P("mirror", src, 64 * es, sz, off, dst, 64 * es, roi, mask, anc, NPP_BORDER_MIRROR, bp);
  P("bigRoi", src, 64 * es, sz, off, dst, 64 * es, NppiSize{128, 128}, mask, anc, NPP_BORDER_REPLICATE, bp);
  printf("\n");
  fflush(stdout);
  cudaFree(sb); cudaFree(db); cudaFree(bp);
}

int main() {
  nppGetStreamContext(&g_ctx);
  matrix<Npp8u, 1>("8u_C1R"); matrix<Npp8u, 3>("8u_C3R"); matrix<Npp8u, 4>("8u_C4R");
  matrix<Npp16u, 1>("16u_C1R"); matrix<Npp16u, 3>("16u_C3R"); matrix<Npp16u, 4>("16u_C4R");
  matrix<Npp16s, 1>("16s_C1R"); matrix<Npp16s, 3>("16s_C3R"); matrix<Npp16s, 4>("16s_C4R");
  matrix<Npp32f, 1>("32f_C1R"); matrix<Npp32f, 3>("32f_C3R"); matrix<Npp32f, 4>("32f_C4R");
  matrix<Npp64f, 1>("64f_C1R");
  return 0;
}
