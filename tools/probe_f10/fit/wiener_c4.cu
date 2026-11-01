// wiener_c4: per-channel noise semantics of the C4 / AC4 Wiener entry points.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}
int main(){
  const int W=9,H=9; NppStreamContext c=ctx();
  unsigned char *s8,*d8; cudaMalloc(&s8,(size_t)W*H*4); cudaMalloc(&d8,(size_t)W*H*4);
  short *s16,*d16; cudaMalloc(&s16,(size_t)W*H*4*2); cudaMalloc(&d16,(size_t)W*H*4*2);
  float n[4]={0.02f,0.5f,1.0f,0.03f};
  // constant 100 (8u), alpha 200
  std::vector<unsigned char> im((size_t)W*H*4);
  for(size_t i=0;i<im.size();i+=4){im[i]=100;im[i+1]=100;im[i+2]=100;im[i+3]=200;}
  cudaMemcpy(s8,im.data(),im.size(),cudaMemcpyHostToDevice);
  std::vector<unsigned char> out((size_t)W*H*4,0x77);
  cudaMemcpy(d8,out.data(),out.size(),cudaMemcpyHostToDevice);
  NppStatus st=nppiFilterWienerBorder_8u_C4R_Ctx(s8,W*4,NppiSize{W,H},NppiPoint{0,0},d8,W*4,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},n,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(),d8,out.size(),cudaMemcpyDeviceToHost);
  printf("8u C4 st=%d px=%d,%d,%d,%d\n",(int)st,out[0],out[1],out[2],out[3]);
  cudaMemcpy(d8,out.data(),out.size(),cudaMemcpyHostToDevice); // mark again
  std::vector<unsigned char> mk((size_t)W*H*4,0x77);
  cudaMemcpy(d8,mk.data(),mk.size(),cudaMemcpyHostToDevice);
  st=nppiFilterWienerBorder_8u_AC4R_Ctx(s8,W*4,NppiSize{W,H},NppiPoint{0,0},d8,W*4,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},n,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(),d8,out.size(),cudaMemcpyDeviceToHost);
  printf("8u AC4 st=%d px=%d,%d,%d,%d\n",(int)st,out[0],out[1],out[2],out[3]);
  // 16s constant 30000 / alpha 10000
  std::vector<short> im16((size_t)W*H*4);
  for(size_t i=0;i<im16.size();i+=4){im16[i]=30000;im16[i+1]=30000;im16[i+2]=30000;im16[i+3]=10000;}
  cudaMemcpy(s16,im16.data(),im16.size()*2,cudaMemcpyHostToDevice);
  std::vector<short> o16((size_t)W*H*4,-12345);
  cudaMemcpy(d16,o16.data(),o16.size()*2,cudaMemcpyHostToDevice);
  st=nppiFilterWienerBorder_16s_C4R_Ctx(s16,W*4*2,NppiSize{W,H},NppiPoint{0,0},d16,W*4*2,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},n,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  cudaMemcpy(o16.data(),d16,o16.size()*2,cudaMemcpyDeviceToHost);
  printf("16s C4 st=%d px=%d,%d,%d,%d\n",(int)st,o16[0],o16[1],o16[2],o16[3]);
  cudaMemset(d16,0xFF,(size_t)W*H*4*2);
  st=nppiFilterWienerBorder_16s_AC4R_Ctx(s16,W*4*2,NppiSize{W,H},NppiPoint{0,0},d16,W*4*2,NppiSize{W,H},
    NppiSize{3,3},NppiPoint{1,1},n,NPP_BORDER_REPLICATE,c);
  cudaDeviceSynchronize();
  cudaMemcpy(o16.data(),d16,o16.size()*2,cudaMemcpyDeviceToHost);
  printf("16s AC4 st=%d px=%d,%d,%d,%d\n",(int)st,o16[0],o16[1],o16[2],o16[3]);
  return 0;
}
