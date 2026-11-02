// f13: which ROIs does nppiDistanceTransformPBAGetBufferSize accept?
#include <nppi.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int sizes[][2] = {{1,1},{2,2},{3,3},{4,4},{4,1},{1,4},{5,5},{8,8},{7,3},{12,10},{13,10},{12,11},
                    {16,16},{15,15},{32,32},{33,17},{64,64},{100,100},{101,99},{256,256},{1000,1000}};
  for (auto &s : sizes) {
    size_t bs = 0;
    NppStatus st = nppiDistanceTransformPBAGetBufferSize(NppiSize{s[0], s[1]}, &bs);
    size_t bs2 = 0;
    NppStatus st2 = nppiSignedDistanceTransformPBAGetBufferSize(NppiSize{s[0], s[1]}, &bs2);
    printf("%4dx%-4d -> PBA st=%-3d size=%-8zu | signed st=%-3d size=%zu\n", s[0], s[1], (int)st, bs,
           (int)st2, bs2);
  }
  size_t z = 0;
  printf("null -> %d\n", (int)nppiDistanceTransformPBAGetBufferSize(NppiSize{8, 8}, nullptr));
  printf("zero -> %d\n", (int)nppiDistanceTransformPBAGetBufferSize(NppiSize{0, 8}, &z));
  printf("neg  -> %d\n", (int)nppiDistanceTransformPBAGetBufferSize(NppiSize{-4, 8}, &z));
  return 0;
}
