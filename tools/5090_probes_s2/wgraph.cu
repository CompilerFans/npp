// WMSSSIM pipeline introspection: capture the kernel sequence into a CUDA graph,
// dump kernel params, and replay prefixes to snapshot intermediate buffers.
#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cuda.h>

static bool readable(const void* p){
  static std::vector<std::pair<unsigned long,unsigned long>> ranges;
  static bool init=false;
  if(!init){
    std::ifstream f("/proc/self/maps");
    char line[512];
    while(f.getline(line,sizeof line)){
      unsigned long a,b; if(sscanf(line,"%lx-%lx",&a,&b)==2) ranges.push_back({a,b});
    }
    init=true;
  }
  unsigned long v=(unsigned long)p;
  for(auto&r:ranges) if(v>=r.first && v<r.second) return true;
  return false;
}

int main(int argc,char**argv){
  int w=argc>1?atoi(argv[1]):16, h=argc>2?atoi(argv[2]):16;
  int seed=argc>3?atoi(argv[3]):0;
  const char* outdir=argc>4?argv[4]:"/tmp/s2probe/snap2";
  size_t sz=(size_t)w*h;
  Npp8u *a,*b; float*o;
  cudaMalloc(&a,sz); cudaMalloc(&b,sz); cudaMalloc(&o,8);
  unsigned char* ha=(unsigned char*)malloc(sz); unsigned char* hb=(unsigned char*)malloc(sz);
  for(int y=0;y<h;y++)for(int x=0;x<w;x++){
    ha[y*w+x]=(unsigned char)((x+w*y+seed*101)&0xff);
    hb[y*w+x]=(unsigned char)((x+w*y+seed*101+90)&0xff);
  }
  cudaMemcpy(a,ha,sz,cudaMemcpyHostToDevice); cudaMemcpy(b,hb,sz,cudaMemcpyHostToDevice);
  NppiSize s{w,h}; NppStreamContext ctx; memset(&ctx,0,sizeof(ctx));
  size_t wsSize=0;
  NppStatus st=nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(s,&wsSize,ctx);
  Npp8u* pWS; cudaMalloc(&pWS,wsSize);
  cudaMemset(pWS,0xCC,wsSize);
  cudaStream_t cst; cudaStreamCreate(&cst);
  ctx.hStream=cst;
  cudaStreamBeginCapture(cst,cudaStreamCaptureModeGlobal);
  st=nppiWMSSSIM_8u_C1R_Ctx(a,w,b,w,s,o,pWS,ctx);
  cudaGraph_t graph;
  cudaError_t ce=cudaStreamEndCapture(cst,&graph);
  printf("capture: npp=%d cuda=%s\n",(int)st,cudaGetErrorString(ce));
  size_t nn=0; cudaGraphGetNodes(graph,nullptr,&nn);
  std::vector<cudaGraphNode_t> nodes(nn);
  cudaGraphGetNodes(graph,nodes.data(),&nn);
  printf("nodes=%zu\n",nn);
  CUgraph cug=(CUgraph)graph;
  std::vector<CUgraphNode> cunodes(nn);
  for(size_t i=0;i<nn;i++) cunodes[i]=(CUgraphNode)nodes[i];
  std::vector<CUDA_KERNEL_NODE_PARAMS> cnp(nn);
  std::vector<bool> isKernel(nn,false);
  for(size_t i=0;i<nn;i++){
    memset(&cnp[i],0,sizeof(CUDA_KERNEL_NODE_PARAMS));
    CUresult cr=cuGraphKernelNodeGetParams(cunodes[i],&cnp[i]);
    if(cr==CUDA_SUCCESS){
      isKernel[i]=true;
      CUfunction cf=cnp[i].func;
      size_t off=0,szp=0; int np=0;
      for(int j=0;j<32;j++){ if(cuFuncGetParamInfo(cf,j,&off,&szp)==CUDA_SUCCESS) np++; else break; }
      printf("node%02zu KERNEL func=%p grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u nparams=%d\n",
             i,(void*)cf,cnp[i].gridDimX,cnp[i].gridDimY,cnp[i].gridDimZ,
             cnp[i].blockDimX,cnp[i].blockDimY,cnp[i].blockDimZ,
             cnp[i].sharedMemBytes,np);
      void** pp=cnp[i].kernelParams;
      if(pp && np>0){
        printf("   params:");
        for(int j=0;j<np;j++){
          void* pv=pp[j];
          unsigned long long v=*(unsigned long long*)pv;
          printf(" [%d]=0x%llx(%llu)",j,v,v);
        }
        printf("\n");
      }
    } else {
      printf("node%02zu not-kernel(type-err=%d)\n",i,(int)cr);
    }
  }
  { // write machine-readable param log
    char fn[512]; snprintf(fn,sizeof fn,"%s/params.txt",outdir);
    FILE* f=fopen(fn,"w");
    for(size_t i=0;i<nn;i++){
      if(!isKernel[i]) continue;
      CUfunction cf=cnp[i].func;
      size_t off=0,szp=0; int np=0;
      for(int j=0;j<32;j++){ if(cuFuncGetParamInfo(cf,j,&off,&szp)==CUDA_SUCCESS) np++; else break; }
      fprintf(f,"node %zu func=%p grid=%u,%u block=%u,%u smem=%u nparams=%d\n",i,(void*)cf,
              cnp[i].gridDimX,cnp[i].gridDimY,cnp[i].blockDimX,cnp[i].blockDimY,cnp[i].sharedMemBytes,np);
      for(int j=0;j<np;j++){
        unsigned char* pv=(unsigned char*)cnp[i].kernelParams[j];
        fprintf(f,"  p%d size=%zu hex=",j,szp);
        size_t nb=szp>16?szp:16; if(nb>256)nb=256; for(size_t b=0;b<nb;b++) fprintf(f,"%02x",pv[b]);
        fprintf(f,"\n");
      }
    }
    fclose(f);
  }
  // replay prefixes, dump pWS
  std::vector<char> ws(wsSize);
  for(size_t k=0;k<nn;k++){
    CUgraph sg; cuGraphCreate(&sg,0);
    std::vector<CUgraphNode> nn2;
    for(size_t i=0;i<=k;i++){
      if(!isKernel[i]) continue;
      CUgraphNode n2;
      CUDA_KERNEL_NODE_PARAMS p2=cnp[i];
      if(cuGraphAddKernelNode(&n2,sg,nullptr,0,&p2)!=CUDA_SUCCESS){ printf("addnode fail %zu\n",i); continue; }
      nn2.push_back(n2);
      if(nn2.size()>1){
        cuGraphAddDependencies(sg,&nn2[nn2.size()-2],&nn2[nn2.size()-1],nullptr,1);
      }
    }
    cudaGraphExec_t ex;
    if(cudaGraphInstantiate(&ex,(cudaGraph_t)sg,nullptr,nullptr,0)==cudaSuccess){
      cudaGraphLaunch(ex,cst);
      cudaStreamSynchronize(cst);
      cudaMemcpy(ws.data(),pWS,wsSize,cudaMemcpyDeviceToHost);
      char fn[512]; snprintf(fn,sizeof fn,"%s/prefix_%02zu.bin",outdir,k);
      FILE* f=fopen(fn,"wb"); fwrite(ws.data(),1,wsSize,f); fclose(f);
      cudaGraphExecDestroy(ex);
    }
    cuGraphDestroy(sg);
  }
  // dump inputs and final o
  { std::vector<unsigned char> v(sz); cudaMemcpy(v.data(),a,sz,cudaMemcpyDeviceToHost);
    char fn[512]; snprintf(fn,sizeof fn,"%s/img_a.bin",outdir); FILE*f=fopen(fn,"wb"); fwrite(v.data(),1,sz,f); fclose(f); }
  { std::vector<unsigned char> v(sz); cudaMemcpy(v.data(),b,sz,cudaMemcpyDeviceToHost);
    char fn[512]; snprintf(fn,sizeof fn,"%s/img_b.bin",outdir); FILE*f=fopen(fn,"wb"); fwrite(v.data(),1,sz,f); fclose(f); }
  { float v; cudaMemcpy(&v,o,4,cudaMemcpyDeviceToHost);
    char fn[512]; snprintf(fn,sizeof fn,"%s/out.bin",outdir); FILE*f=fopen(fn,"wb"); fwrite(&v,4,1,f); fclose(f);
    printf("final wmsssim value=%.7f\n",v); }
  printf("pWS=%p wsSize=%zu\n",(void*)pWS,wsSize);
  return 0;
}
