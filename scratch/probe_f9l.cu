#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
template <class T> struct B { T*p; int step; B(int w,int h){step=w*sizeof(T);cudaMalloc(&p,(size_t)step*h);} ~B(){cudaFree(p);} };
int main(){ nppGetStreamContext(&g);
  int ms[][2]={{1,1},{2,2},{3,3},{4,4},{5,5},{2,3},{3,2},{1,2}};
  for (auto &m:ms) {
    NppiSize mask{m[0],m[1]};
    { B<Npp8u>s(8,8),d(8,8); printf("8uC1 m%dx%d roi0w=%d roi0h=%d\n",m[0],m[1],
        (int)nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},mask,NppiPoint{0,0},g),
        (int)nppiFilterBox_8u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{7,0},mask,NppiPoint{0,0},g)); }
    { B<Npp8u>s(32,32),d(32,32); printf("8uC4 m%dx%d roi0w=%d\n",m[0],m[1],
        (int)nppiFilterBox_8u_C4R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},mask,NppiPoint{0,0},g)); }
    { B<Npp16u>s(8,8),d(8,8); printf("16uC1 m%dx%d roi0w=%d\n",m[0],m[1],
        (int)nppiFilterBox_16u_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},mask,NppiPoint{0,0},g)); }
    { B<Npp32f>s(8,8),d(8,8); printf("32fC1 m%dx%d roi0w=%d\n",m[0],m[1],
        (int)nppiFilterBox_32f_C1R_Ctx(s.p,s.step,d.p,d.step,NppiSize{0,7},mask,NppiPoint{0,0},g)); }
    { B<Npp8u>s(16,16),d(8,8); printf("boxb8uC1 m%dx%d roi0w=%d\n",m[0],m[1],
        (int)nppiFilterBoxBorder_8u_C1R_Ctx(s.p,s.step,NppiSize{13,11},NppiPoint{2,3},d.p,d.step,NppiSize{0,7},mask,NppiPoint{0,0},NPP_BORDER_REPLICATE,g)); }
    { B<Npp8u>s(32,32),d(32,32); printf("boxb8uC4 m%dx%d roi0w=%d\n",m[0],m[1],
        (int)nppiFilterBoxBorder_8u_C4R_Ctx(s.p,s.step,NppiSize{13,11},NppiPoint{2,3},d.p,d.step,NppiSize{0,7},mask,NppiPoint{0,0},NPP_BORDER_REPLICATE,g)); }
  }
  return 0; }
