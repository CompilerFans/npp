// wiener_model: validate the reverse-engineered Wiener algorithm against NPP.
#include <npp.h>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext*); }
#include <nppi_filtering_functions.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
static NppStreamContext ctx(){NppStreamContext c{}; nppGetStreamContext(&c); return c;}

struct Wc { int w,h,ax,ay; };

// window visited by NPP's statistics loop (see probe findings)
static void wiener_model_f(const float *src,int W,int H,Wc m,const float *noise,int nch,float *dst){
  const int R = m.h>1 ? m.h-1 : 1;
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int ch=0;ch<nch;++ch){
    float sum=0,sumsq=0; int K=0;
    for(int r=0;r<R;++r){
      int dy=m.ay-r;
      int x0 = (r==0)? m.ax-(m.w-1) : m.ax-(m.w-1)+1;
      for(int dx=x0;dx<=m.ax;++dx){
        int yy=y+dy; if(yy<0)yy=0; if(yy>H-1)yy=H-1;
        int xx=x+dx; if(xx<0)xx=0; if(xx>W-1)xx=W-1;
        float v=src[((size_t)yy*W+xx)*nch+ch];
        sum+=v; sumsq+=v*v; ++K;
      }
    }
    float N=float(m.w*m.h);
    float mu=sum/N;
    float V=(sumsq - mu*mu*float(K))/N;
    float s=src[((size_t)y*W+x)*nch+ch];
    float o;
    if(noise[ch]==0.f) o=V;
    else if(V>noise[ch]) o=mu+(V-noise[ch])/V*(s-mu);
    else o=mu;
    dst[((size_t)y*W+x)*nch+ch]=o;
  }
}

int main(){
  const int W=23,H=19;
  NppStreamContext c=ctx();
  float *s,*d; cudaMalloc(&s,(size_t)W*H*4); cudaMalloc(&d,(size_t)W*H*4);
  std::vector<float> img((size_t)W*H), out((size_t)W*H), mod((size_t)W*H);
  unsigned st=777;
  for(size_t i=0;i<img.size();++i){st=st*1103515245u+12345u; img[i]=(float)((st>>16)%400)-200.f;}
  cudaMemcpy(s,img.data(),img.size()*4,cudaMemcpyHostToDevice);
  struct Case{Wc m; float noise; const char*t;} cs[]={
    {{3,3,1,1},1.f,"m33a11n1"},{{3,3,0,0},1.f,"m33a00n1"},{{3,3,2,2},.5f,"m33a22n05"},
    {{5,5,2,2},1.f,"m55a22n1"},{{4,4,1,1},1.f,"m44a11n1"},{{2,2,0,0},1.f,"m22a00n1"},
    {{3,5,1,2},1.f,"m35a12n1"},{{5,3,2,1},.25f,"m53a21n025"},{{1,3,0,1},1.f,"m13a01n1"},
    {{3,1,1,0},1.f,"m31a10n1"},{{7,7,3,3},1.f,"m77a33n1"},{{3,3,1,1},0.f,"m33a11n0"}};
  for(auto&cc:cs){
    float noise[4]={cc.noise,cc.noise,cc.noise,cc.noise};
    std::vector<float> mark((size_t)W*H,-1.f);
    cudaMemcpy(d,mark.data(),mark.size()*4,cudaMemcpyHostToDevice);
    NppStatus stt=nppiFilterWienerBorder_32f_C1R_Ctx(s,W*4,NppiSize{W,H},NppiPoint{0,0},d,W*4,
      NppiSize{W,H},NppiSize{cc.m.w,cc.m.h},NppiPoint{cc.m.ax,cc.m.ay},noise,NPP_BORDER_REPLICATE,c);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(),d,out.size()*4,cudaMemcpyDeviceToHost);
    wiener_model_f(img.data(),W,H,cc.m,noise,1,mod.data());
    double maxd=0; int bad=0;
    for(size_t i=0;i<out.size();++i){double e=fabs(out[i]-mod[i]); if(e>maxd)maxd=e; if(e>1e-3*(1+fabs(out[i])))++bad;}
    printf("32f %-10s st=%d bad=%d maxd=%.4g\n",cc.t,(int)stt,bad,maxd);
  }
  // 8u noise scaling on a constant image: V(c)=c^2*K(N-K)/N^2
  {
    const int w=9,h=9; unsigned char *s8,*d8; cudaMalloc(&s8,w*h); cudaMalloc(&d8,w*h);
    std::vector<unsigned char> im((size_t)w*h,100), o8((size_t)w*h);
    cudaMemcpy(s8,im.data(),im.size(),cudaMemcpyHostToDevice);
    printf("8u const c=100 targetV=%.3f\n", 100.0*100.0*5*4/81.0);
    for(float n=0.02f;n<=0.06f;n+=0.005f){
      float noise[4]={n,n,n,n};
      cudaMemcpy(d8,o8.data(),o8.size(),cudaMemcpyDeviceToDevice);
      std::vector<unsigned char> mark((size_t)w*h,0x77);
      cudaMemcpy(d8,mark.data(),mark.size(),cudaMemcpyHostToDevice);
      nppiFilterWienerBorder_8u_C1R_Ctx(s8,w,NppiSize{w,h},NppiPoint{0,0},d8,w,NppiSize{w,h},
        NppiSize{3,3},NppiPoint{1,1},noise,NPP_BORDER_REPLICATE,c);
      cudaDeviceSynchronize();
      cudaMemcpy(o8.data(),d8,o8.size(),cudaMemcpyDeviceToHost);
      printf("  n=%.3f out=%d (mean=%.3f)\n",n,(int)o8[40],5*100.0/9);
    }
    cudaFree(s8); cudaFree(d8);
  }
  return 0;
}
