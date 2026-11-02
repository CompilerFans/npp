// f9h: even-count median rounding per mask shape (signed / float)
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
extern "C" { cudaStream_t nppGetStream(void); NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext g;
static unsigned st_seed=1;
static float rndf(){ st_seed=st_seed*1664525u+1013904223u; return (float)((int)(st_seed>>9)%4001-2000); }
template <class T> struct Img { T*p; int step; int W,H,M; Img(int w,int h,int m):W(w),H(h),M(m){ step=(w+2*m)*sizeof(T); cudaMalloc(&p,(size_t)step*(h+2*m)); } ~Img(){cudaFree(p);} T* org(){return (T*)((char*)p+(size_t)M*step+M*sizeof(T));} };
template <class T> struct Dst { T*p; int step; int W,H; Dst(int w,int h):W(w),H(h){step=w*sizeof(T);cudaMalloc(&p,(size_t)step*h);} ~Dst(){cudaFree(p);} };

template <class T>
static void test(const char *tag, bool border, NppiBorderType bt) {
  const int SW=16,SH=16,W=6,H=6,M=6;
  Img<T> im(SW,SH,M); Dst<T> d(8,H);  // 16-byte dst rows
  const int hsW=SW+2*M, hsH=SH+2*M;
  std::vector<T> him((size_t)hsW*hsH, 0);
  for (auto &v:him) v=(T)(long long)rndf();
  std::vector<T> row(SW);
  for (int j=0;j<SH;++j) {
    for (int i=0;i<SW;++i) row[i]=(T)him[(size_t)(j+M)*hsW+i+M];
    cudaMemcpy(im.org()+(size_t)j*(im.step/sizeof(T)), row.data(), SW*sizeof(T), cudaMemcpyHostToDevice);
  }
  int masks[][2]={{1,2},{2,1},{2,2},{2,3},{3,2},{3,4},{4,3},{2,4},{4,2},{1,4},{4,1},{5,2},{2,5},{4,4},{6,6}};
  for (auto &m:masks) {
    NppiSize ms{m[0],m[1]}; NppiPoint an{m[0]/2,m[1]/2};
    NppStatus st;
    if constexpr(std::is_same_v<T,Npp16s>) {
      if (!border) st=nppiFilterMedian_16s_C1R_Ctx(im.org(),im.step,d.p,d.step,NppiSize{W,H},ms,an,nullptr,g);
      else st=nppiFilterMedianBorder_16s_C1R_Ctx(im.org(),im.step,NppiSize{SW,SH},NppiPoint{0,0},d.p,d.step,NppiSize{W,H},ms,an,nullptr,bt,g);
    } else if constexpr(std::is_same_v<T,Npp32f>) {
      if (!border) st=nppiFilterMedian_32f_C1R_Ctx(im.org(),im.step,d.p,d.step,NppiSize{W,H},ms,an,nullptr,g);
      else st=nppiFilterMedianBorder_32f_C1R_Ctx(im.org(),im.step,NppiSize{SW,SH},NppiPoint{0,0},d.p,d.step,NppiSize{W,H},ms,an,nullptr,bt,g);
    }
    std::vector<T> o((size_t)W*H); cudaMemcpy2D(o.data(),W*sizeof(T),d.p,d.step,W*sizeof(T),H,cudaMemcpyDeviceToHost);
    long long mt=0, mf=0;
    for (int y=0;y<H;++y) for(int x=0;x<W;++x){
      std::vector<T> w;
      for(int j=0;j<m[1];++j) for(int i=0;i<m[0];++i){
        int sx=x+i-an.x+M, sy=y+j-an.y+M;
        if(border){ sx=std::clamp(sx-M,0,SW-1)+M; sy=std::clamp(sy-M,0,SH-1)+M; }
        w.push_back(him[(size_t)sy*hsW+sx]);
      }
      std::sort(w.begin(),w.end()); int n=(int)w.size(); int k1=(n-1)/2,k2=n/2;
      long long a=(long long)w[k1], b=(long long)w[k2];
      long long tr = (a+b)/2; long long fl = (a+b)>>1;
      if (n%2==0){
        T got=o[(size_t)y*W+x];
        if (got!=(T)tr) ++mt;
        if (got!=(T)fl) ++mf;
        if (m[0]*m[1]<=4 && got!=(T)tr && got!=(T)fl && mt<=3)
        { printf("    elem(%d,%d) got=%ld tr=%ld fl=%ld win:",x,y,(long)got,tr,fl); for(auto v:w) printf(" %ld",(long)v); printf("\n"); }
      }
    }
    printf("%-4s %-6s mask%dx%d %s: st=%d trunc-mism=%lld floor-mism=%lld\n",tag,border?"border":"plain",m[0],m[1],
           border?(bt==NPP_BORDER_MIRROR?"mirror":"repl"):"", (int)st, mt, mf);
  }
}
int main(){ nppGetStreamContext(&g);
  test<Npp16s>("16s",false,NPP_BORDER_REPLICATE);
  test<Npp16s>("16s",true,NPP_BORDER_REPLICATE);
  test<Npp16s>("16s",true,NPP_BORDER_MIRROR);
  test<Npp32f>("32f",false,NPP_BORDER_REPLICATE);
  test<Npp32f>("32f",true,NPP_BORDER_MIRROR);
  return 0; }
