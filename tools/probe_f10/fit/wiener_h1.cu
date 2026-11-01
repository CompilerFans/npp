// wiener_h1: diagnose the h==1 Wiener statistics path.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
int main(){
  const int W=23,H=19; NppStreamContext c=ctx();
  float *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  std::vector<float> img((size_t)W*H),out((size_t)W*H);
  unsigned st=4242;
  for(size_t i=0;i<img.size();++i){st=st*1103515245u+12345u; img[i]=(float)((st>>16)%400)-200.f;}
  cudaMemcpy(s,img.data(),img.size()*4,cudaMemcpyHostToDevice);
  float noise[4]={1.f,1.f,1.f,1.f};
  std::vector<float> mk((size_t)W*H,-1.f);
  cudaMemcpy(d,mk.data(),mk.size()*4,cudaMemcpyDeviceToHost);
  NppStatus stt=nppiFilterWienerBorder_32f_C1R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
    NppiSize{3,1},NppiPoint{1,0},noise,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(),d,out.size()*4,cudaMemcpyDeviceToHost);
  printf("st=%d\n",(int)stt);
  // n=0 run: output is the variance statistic itself
  float nz0[4]={0.f,0.f,0.f,0.f};
  std::vector<float> out0((size_t)W*H,-1.f);
  cudaMemcpy(d,mk.data(),mk.size()*4,cudaMemcpyHostToDevice);
  nppiFilterWienerBorder_32f_C1R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
    NppiSize{3,1},NppiPoint{1,0},nz0,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  cudaMemcpy(out0.data(),d,out0.size()*4,cudaMemcpyDeviceToHost);
  for(int y=5;y<8;++y)for(int x=3;x<7;++x){
    float a=img[(size_t)y*W+x-1],b=img[(size_t)y*W+x],cc=img[(size_t)y*W+x+1];
    float mu=(a+b+cc)/3.f; float V=(a*a+b*b+cc*cc-3*mu*mu)/3.f;
    float o=out[(size_t)y*W+x];
    // implied V assuming dst = mu + (V-1)/V*(src-mu)
    float s0=img[(size_t)y*W+x];
    double implied = (s0-mu)/((o-mu)/ (1.0) + 1e-30); (void)implied;
    double d1=(double)(o-mu), d2=(double)(s0-mu);
    double ratio = d2!=0? d1/d2: 0;
    double Vimp = 1.0/(1.0-ratio);
    printf("(%d,%d) src=%7.2f mu=%7.3f V3=%8.2f Vnpp=%9.3f -> out=%9.4f ratio=%.5f Vimp=%.4f\n",x,y,s0,mu,V,out0[(size_t)y*W+x],o,ratio,Vimp);
  }
  return 0;
}
