#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cstdio>
#include <cuda_runtime.h>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
int main(){
  const int W=9,H=9; float *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  cudaMemset(s,0,sizeof(float)*W*H);
  NppStreamContext c=ctx();
  float good[4]={0.5f,0.5f,0.5f,0.5f}, bad[4]={1.5f,0.5f,0.5f,0.5f};
  float bad2[4]={0.5f,1.5f,0.5f,0.5f};
  float bad4[4]={0.5f,0.5f,0.5f,1.5f};
  float bad3[4]={0.5f,0.5f,1.5f,0.5f};
  NppiSize sz{9,9}, roi{5,5}, zroi{0,0};
  NppiPoint off{2,2};
#define P(t,x) printf("%-28s %d\n",t,(int)(x))
  P("mask0 badanchor",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{0,0},NppiPoint{9,9},good,NPP_BORDER_REPLICATE,c));
  P("mask0 badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{0,0},NppiPoint{0,0},bad,NPP_BORDER_REPLICATE,c));
  P("badanchor badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{9,9},bad,NPP_BORDER_REPLICATE,c));
  P("badtap c1 noise1",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad2,NPP_BORDER_REPLICATE,c));
  P("c4 noise ch4 bad",nppiFilterWienerBorder_32f_C4R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad2,NPP_BORDER_REPLICATE,c));
  P("c4 noise ch4 bad4",nppiFilterWienerBorder_32f_C4R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad4,NPP_BORDER_REPLICATE,c));
  P("c3 noise ch3 bad",nppiFilterWienerBorder_32f_C3R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad2,NPP_BORDER_REPLICATE,c));
  P("maskneg badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{-1,3},NppiPoint{0,0},bad,NPP_BORDER_REPLICATE,c));
  P("roizero badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,zroi,NppiSize{3,3},NppiPoint{1,1},bad,NPP_BORDER_REPLICATE,c));
  P("roizero badanchor",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,zroi,NppiSize{3,3},NppiPoint{9,9},good,NPP_BORDER_REPLICATE,c));
  P("anchor eq mask w",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{2,1},good,NPP_BORDER_REPLICATE,c));
  P("anchor x neg",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{-2,1},good,NPP_BORDER_REPLICATE,c));
  P("anchor y neg",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,-2},good,NPP_BORDER_REPLICATE,c));
  P("anchor eq mask h",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,3},good,NPP_BORDER_REPLICATE,c));
  P("negroi badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,NppiSize{5,-1},NppiSize{3,3},NppiPoint{1,1},bad,NPP_BORDER_REPLICATE,c));
  P("badborder badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad,NPP_BORDER_CONSTANT,c));
  P("offneg badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,NppiPoint{-1,0},d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad,NPP_BORDER_REPLICATE,c));
  P("srcsize0 badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,NppiSize{0,5},off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad,NPP_BORDER_REPLICATE,c));
  P("null badnoise",nppiFilterWienerBorder_32f_C1R_Ctx(nullptr,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad,NPP_BORDER_REPLICATE,c));
  P("ac4 noise ch4 bad",nppiFilterWienerBorder_32f_AC4R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad4,NPP_BORDER_REPLICATE,c));
  P("ac4 noise ch3 bad",nppiFilterWienerBorder_32f_AC4R_Ctx(s,36,sz,off,d,36,roi,NppiSize{3,3},NppiPoint{1,1},bad3,NPP_BORDER_REPLICATE,c));
  P("mask1x100 anchor",nppiFilterWienerBorder_32f_C1R_Ctx(s,36,sz,off,d,36,roi,NppiSize{1,100},NppiPoint{0,99},good,NPP_BORDER_REPLICATE,c));
  return 0;
}
