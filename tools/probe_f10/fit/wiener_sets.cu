// wiener_sets: classify the mean set A and variance set B of the Wiener
// statistics by single-impulse n=0 responses at one output pixel.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
static void run_case(int w,int h,int ax,int ay,const char*tag){
  const int W=41,H=41,CX=20,CY=20; NppStreamContext c=ctx();
  float *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  std::vector<float> img((size_t)W*H),out;
  float nz[4]={0.f,0.f,0.f,0.f};
  printf("== %s (w=%d h=%d a=%d,%d) N=%d\n",tag,w,h,ax,ay,w*h);
  for(int dy=-6;dy<=6;++dy){
    printf("  dy=%2d:",dy);
    for(int dx=-6;dx<=6;++dx){
      std::fill(img.begin(),img.end(),0.f);
      img[(size_t)(CY+dy)*W+CX+dx]=1.f;
      cudaMemcpy(s,img.data(),img.size()*4,cudaMemcpyHostToDevice);
      std::vector<float> mk((size_t)W*H,-1.f);
      cudaMemcpy(d,mk.data(),mk.size()*4,cudaMemcpyHostToDevice);
      nppiFilterWienerBorder_32f_C1R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,NppiSize{W,H},
        NppiSize{w,h},NppiPoint{ax,ay},nz,NPP_BORDER_REPLICATE,c);
      cudaDeviceSynchronize();
      float v=-1.f; cudaMemcpy(&v,d+(size_t)CY*W+CX,4,cudaMemcpyDeviceToHost);
      char ch='.';
      double N=w*h;
      if(fabs(v-1.0/N)<1e-4) ch='B';
      else if(v< -1e-6) ch='A';
      else if(fabs(v)<1e-6) ch='.';
      else ch='?';
      printf("%c",ch);
    }
    printf("\n");
  }
  cudaFree(s); cudaFree(d);
}
int main(){
  run_case(3,1,1,0,"m31a10");
  run_case(3,1,0,0,"m31a00");
  run_case(1,3,0,1,"m13a01");
  run_case(3,3,1,1,"m33a11");
  run_case(5,1,2,0,"m51a20");
  return 0;
}
