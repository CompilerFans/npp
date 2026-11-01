#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
template <class T> struct B { T*p; int step; B(int w,int h){step=w*sizeof(T);cudaMalloc(&p,(size_t)step*h);} ~B(){cudaFree(p);} };
#define CHK(tag, expr) printf("%-30s %d\n", tag, (int)(expr))
int main(){ nppGetStreamContext(&g);
  NppiPoint a1{1,1}; NppiSize m3{3,3};
  { B<Npp8u>s(32,32),d(32,32); CHK("box 8uC1 mask9x5 roi4x4", nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{4,4},NppiSize{9,5},NppiPoint{4,2},g)); }
  { B<Npp64f>s(32,32),d(32,32); CHK("box 64fC1 mask9x5 roi4x4", nppiFilterBox_64f_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{4,4},NppiSize{9,5},NppiPoint{4,2},g)); }
  { B<Npp8u>s(32,32),d(32,32); CHK("box 8uC1 mask0 roi7", nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{7,7},NppiSize{0,0},a1,g)); }
  { B<Npp8u>s(32,32),d(32,32); CHK("box 8uC1 mask8x5 roi4x4", nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{4,4},NppiSize{8,5},NppiPoint{4,2},g)); }
  { B<Npp8u>s(32,32),d(32,32); CHK("box 8uC4 mask9x5 roi4x4", nppiFilterBox_8u_C4R_Ctx(s.p,s.step,d.p,d.step,NppiSize{4,4},NppiSize{9,5},NppiPoint{4,2},g)); }
  NppiSize src{16,16}; NppiPoint off{2,3};
  { B<Npp8u>s(16,16),d(32,32); CHK("boxb 8uC1 mask9x5 roi4x4", nppiFilterBoxBorder_8u_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{4,4},NppiSize{9,5},NppiPoint{4,2},NPP_BORDER_REPLICATE,g)); }
  { B<Npp8u>s(16,16),d(32,32); CHK("boxb 8uC1 mask4x2 roi4x4", nppiFilterBoxBorder_8u_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{4,4},NppiSize{4,2},NppiPoint{1,1},NPP_BORDER_REPLICATE,g)); }
  { B<Npp16u>s(16,16),d(32,32); CHK("boxb 16uC1 mask9x5 roi4x4", nppiFilterBoxBorder_16u_C1R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{4,4},NppiSize{9,5},NppiPoint{4,2},NPP_BORDER_REPLICATE,g)); }
  { B<Npp8u>s(16,16),d(32,32); CHK("boxb 8uC4 mask9x5 roi4x4", nppiFilterBoxBorder_8u_C4R_Ctx(s.p,s.step,src,off,d.p,d.step,NppiSize{4,4},NppiSize{9,5},NppiPoint{4,2},NPP_BORDER_REPLICATE,g)); }
  return 0; }
