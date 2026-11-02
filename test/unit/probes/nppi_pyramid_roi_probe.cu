// Dense 1-D probe of the pyramid destination-ROI helpers.
#include <cstdio>

#include <nppi.h>
#include <nppi_filtering_functions.h>

int main() {
  const float rates[] = {1.0001f, 1.05f, 1.1f, 1.2f, 1.25f, 1.3333334f, 1.5f,
                         1.75f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 5.0f, 7.0f, 10.0f};
  for (float r : rates) {
    printf("RATE %.9g\n", r);
    for (int w = 1; w <= 40; ++w) {
      NppiSize d = {-1, -1};
      NppStatus s1 = nppiGetFilterGaussPyramidLayerDownBorderDstROI(w, 40, &d, r);
      NppiSize mn = {-1, -1}, mx = {-1, -1};
      NppStatus s2 = nppiGetFilterGaussPyramidLayerUpBorderDstROI(w, 40, &mn, &mx, r);
      printf("%d %d %d %d %d %d %d %d %d\n", w, (int)s1, d.width, d.height, (int)s2, mn.width,
             mn.height, mx.width, mx.height);
    }
  }
  return 0;
}
