// wiener_shape: determine the effective statistics window of
// nppiFilterWienerBorder (32f C1) by impulse response support.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
int main(){
  const int W=17,H=17, P=8;
  float *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  std::vector<float> img((size_t)W*H,0.f), out((size_t)W*H,-1.f);
  img[(size_t)P*W+P]=30.f;
  cudaMemcpy(s,img.data(),img.size()*4,cudaMemcpyHostToDevice);
  float noise[4]={1.f,1.f,1.f,1.f};
  struct C{int mw,mh,ax,ay;const char*t;} cs[]={
    {3,3,1,1,"m33a11"},{3,3,0,0,"m33a00"},{3,3,2,2,"m33a22"},
    {2,2,0,0,"m22a00"},{2,2,1,1,"m22a11"},{4,4,1,1,"m44a11"},{4,4,2,2,"m44a22"},
    {3,5,1,2,"m35a12"},{5,3,2,1,"m53a21"},{1,3,0,1,"m13a01"},{3,1,1,0,"m31a10"},
    {1,3,0,0,"m13a00"},{3,1,0,0,"m31a00"},{1,3,0,2,"m13a02"},{3,1,2,0,"m31a20"},
    {5,5,2,2,"m55a22"},{7,7,3,3,"m77a33"},{7,7,0,0,"m77a00"},{1,1,0,0,"m11"}};
  NppStreamContext c=ctx();
  for(auto&cc:cs){
    
    std::vector<float> mark((size_t)W*H,-1.f);
    cudaMemcpy(d,mark.data(),mark.size()*4,cudaMemcpyHostToDevice);
    NppStatus st=nppiFilterWienerBorder_32f_C1R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,
      NppiSize{W,H},NppiSize{cc.mw,cc.mh},NppiPoint{cc.ax,cc.ay},noise,NPP_BORDER_REPLICATE,c);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(),d,out.size()*4,cudaMemcpyDeviceToHost);
    int cnt=0; printf("%-8s st=%d supp:",cc.t,(int)st);
    for(int dy=-7;dy<=7;++dy){int n=0; for(int dx=-7;dx<=7;++dx){int x=P+dx,y=P+dy; if(x<0||y<0||x>=W||y>=H)continue; if(out[(size_t)y*W+x]!=0.f)++n;} if(n)printf(" dy=%d:%d",dy,n); cnt+=n;}
    printf(" total=%d\n",cnt);
    // explicit offsets for small masks
    if(cc.mw<=5&&cc.mh<=5){
      printf("   D:");
      for(int dy=-3;dy<=3;++dy)for(int dx=-3;dx<=3;++dx){int x=P+dx,y=P+dy; if(x<0||y<0||x>=W||y>=H)continue; if(out[(size_t)y*W+x]!=0.f)printf(" (%d,%d)",dx,dy);}
      printf("\n");
    }
  }
  return 0;
}
