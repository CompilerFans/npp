// Generated API wrappers -- do not edit by hand.
// module: nppi_linear_transforms.h
#include "num_suite.h"

TEST(NumGen, nppiMagnitudeSqr_32fc32f_C1R_Ctx) {
  num::run_magnitude<true>(
      "nppiMagnitudeSqr_32fc32f_C1R_Ctx",
      [](const Npp32fc *a, int sa, Npp32f *d, int ds, NppiSize r, NppStreamContext c) { (void)sa; (void)ds; (void)r; (void)c; return nppiMagnitudeSqr_32fc32f_C1R_Ctx(a, sa, d, ds, r, c); });
}

TEST(NumGen, nppiMagnitude_32fc32f_C1R_Ctx) {
  num::run_magnitude<false>(
      "nppiMagnitude_32fc32f_C1R_Ctx",
      [](const Npp32fc *a, int sa, Npp32f *d, int ds, NppiSize r, NppStreamContext c) { (void)sa; (void)ds; (void)r; (void)c; return nppiMagnitude_32fc32f_C1R_Ctx(a, sa, d, ds, r, c); });
}
