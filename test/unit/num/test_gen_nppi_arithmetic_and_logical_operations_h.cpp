// Generated API wrappers -- do not edit by hand.
// module: nppi_arithmetic_and_logical_operations.h
#include "num_suite.h"

TEST(NumGen, nppiAbsDiffC_16u_C1R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::AbsDiffC>(
      "nppiAbsDiffC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiffC_16u_C1R_Ctx(a, sa, d, sd, r, scalar, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAbsDiffC_32f_C1R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 1, 1, num::Op::AbsDiffC>(
      "nppiAbsDiffC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiffC_32f_C1R_Ctx(a, sa, d, sd, r, scalar, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiAbsDiffC_8u_C1R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::AbsDiffC>(
      "nppiAbsDiffC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiffC_8u_C1R_Ctx(a, sa, d, sd, r, scalar, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAbsDiffDeviceC_16u_C1R_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 1, 1, num::Op::AbsDiffC>(
      "nppiAbsDiffDeviceC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAbsDiffDeviceC_16u_C1R_Ctx(a, sa, d, sd, r, cv, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbsDiffDeviceC_32f_C1R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 1, 1, num::Op::AbsDiffC>(
      "nppiAbsDiffDeviceC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAbsDiffDeviceC_32f_C1R_Ctx(a, sa, d, sd, r, cv, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbsDiffDeviceC_8u_C1R_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 1, 1, num::Op::AbsDiffC>(
      "nppiAbsDiffDeviceC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAbsDiffDeviceC_8u_C1R_Ctx(a, sa, d, sd, r, cv, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbsDiff_16f_C1R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 1, 1, num::Op::AbsDiff>(
      "nppiAbsDiff_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiff_16f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAbsDiff_16u_C1R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::AbsDiff>(
      "nppiAbsDiff_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiff_16u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAbsDiff_32f_C1R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 1, 1, num::Op::AbsDiff>(
      "nppiAbsDiff_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiff_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAbsDiff_8u_C1R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::AbsDiff>(
      "nppiAbsDiff_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiff_8u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAbsDiff_8u_C3R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::AbsDiff>(
      "nppiAbsDiff_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiff_8u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAbsDiff_8u_C4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::AbsDiff>(
      "nppiAbsDiff_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbsDiff_8u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAbs_16f_C1IR_Ctx) {
  num::run_unary_inplace<Npp16f, 1, 1, num::Op::Abs>(
      "nppiAbs_16f_C1IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16f_C1IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAbs_16f_C1R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 1, 1, num::Op::Abs>(
      "nppiAbs_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAbs_16f_C3IR_Ctx) {
  num::run_unary_inplace<Npp16f, 3, 3, num::Op::Abs>(
      "nppiAbs_16f_C3IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16f_C3IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAbs_16f_C3R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 3, 3, num::Op::Abs>(
      "nppiAbs_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAbs_16f_C4IR_Ctx) {
  num::run_unary_inplace<Npp16f, 4, 4, num::Op::Abs>(
      "nppiAbs_16f_C4IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16f_C4IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAbs_16f_C4R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 4, 4, num::Op::Abs>(
      "nppiAbs_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16f_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAbs_16s_AC4IR_Ctx) {
  num::run_unary_inplace<Npp16s, 4, 3, num::Op::Abs>(
      "nppiAbs_16s_AC4IR_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_AC4IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_AC4R_Ctx) {
  num::run_unary<Npp16s, Npp16s, 4, 3, num::Op::Abs>(
      "nppiAbs_16s_AC4R_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_AC4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_C1IR_Ctx) {
  num::run_unary_inplace<Npp16s, 1, 1, num::Op::Abs>(
      "nppiAbs_16s_C1IR_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_C1IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_C1R_Ctx) {
  num::run_unary<Npp16s, Npp16s, 1, 1, num::Op::Abs>(
      "nppiAbs_16s_C1R_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_C3IR_Ctx) {
  num::run_unary_inplace<Npp16s, 3, 3, num::Op::Abs>(
      "nppiAbs_16s_C3IR_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_C3IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_C3R_Ctx) {
  num::run_unary<Npp16s, Npp16s, 3, 3, num::Op::Abs>(
      "nppiAbs_16s_C3R_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_C4IR_Ctx) {
  num::run_unary_inplace<Npp16s, 4, 4, num::Op::Abs>(
      "nppiAbs_16s_C4IR_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_C4IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_16s_C4R_Ctx) {
  num::run_unary<Npp16s, Npp16s, 4, 4, num::Op::Abs>(
      "nppiAbs_16s_C4R_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_16s_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiAbs_32f_AC4IR_Ctx) {
  num::run_unary_inplace<Npp32f, 4, 3, num::Op::Abs>(
      "nppiAbs_32f_AC4IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_AC4IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_AC4R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 4, 3, num::Op::Abs>(
      "nppiAbs_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_AC4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_C1IR_Ctx) {
  num::run_unary_inplace<Npp32f, 1, 1, num::Op::Abs>(
      "nppiAbs_32f_C1IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_C1IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_C1R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 1, 1, num::Op::Abs>(
      "nppiAbs_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_C3IR_Ctx) {
  num::run_unary_inplace<Npp32f, 3, 3, num::Op::Abs>(
      "nppiAbs_32f_C3IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_C3IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_C3R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 3, 3, num::Op::Abs>(
      "nppiAbs_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_C4IR_Ctx) {
  num::run_unary_inplace<Npp32f, 4, 4, num::Op::Abs>(
      "nppiAbs_32f_C4IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_C4IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAbs_32f_C4R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 4, 4, num::Op::Abs>(
      "nppiAbs_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAbs_32f_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddC_16f_C1IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddC_16f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, false);
}

TEST(NumGen, nppiAddC_16f_C1R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, false);
}

TEST(NumGen, nppiAddC_16f_C3IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddC_16f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiAddC_16f_C3R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiAddC_16f_C4IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddC_16f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiAddC_16f_C4R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiAddC_16s_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 3, num::Op::AddC>(
      "nppiAddC_16s_AC4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16s_AC4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 3, num::Op::AddC>(
      "nppiAddC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 1, 1, num::Op::AddC>(
      "nppiAddC_16s_C1IRSfs_Ctx",
      [](const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_16s_C1RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 1, 1, num::Op::AddC>(
      "nppiAddC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_16s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 3, 3, num::Op::AddC>(
      "nppiAddC_16s_C3IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16s_C3RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 3, 3, num::Op::AddC>(
      "nppiAddC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16s_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 4, num::Op::AddC>(
      "nppiAddC_16s_C4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16s_C4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 4, num::Op::AddC>(
      "nppiAddC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16s_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 4, 3, num::Op::AddC>(
      "nppiAddC_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16sc_AC4RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 4, 3, num::Op::AddC>(
      "nppiAddC_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 1, 1, num::Op::AddC>(
      "nppiAddC_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_16sc_C1RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 1, 1, num::Op::AddC>(
      "nppiAddC_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_16sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 3, 3, num::Op::AddC>(
      "nppiAddC_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16sc_C3RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 3, 3, num::Op::AddC>(
      "nppiAddC_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::AddC>(
      "nppiAddC_16u_AC4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16u_AC4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::AddC>(
      "nppiAddC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::AddC>(
      "nppiAddC_16u_C1IRSfs_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_16u_C1RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::AddC>(
      "nppiAddC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_16u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::AddC>(
      "nppiAddC_16u_C3IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16u_C3RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::AddC>(
      "nppiAddC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::AddC>(
      "nppiAddC_16u_C4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_16u_C4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::AddC>(
      "nppiAddC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_16u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32f_AC4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 3, num::Op::AddC>(
      "nppiAddC_32f_AC4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiAddC_32f_AC4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 3, num::Op::AddC>(
      "nppiAddC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiAddC_32f_C1IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddC_32f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiAddC_32f_C1R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiAddC_32f_C3IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddC_32f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiAddC_32f_C3R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiAddC_32f_C4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddC_32f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiAddC_32f_C4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiAddC_32fc_AC4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 3, num::Op::AddC>(
      "nppiAddC_32fc_AC4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32fc_AC4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 3, num::Op::AddC>(
      "nppiAddC_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32fc_C1IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 1, 1, num::Op::AddC>(
      "nppiAddC_32fc_C1IR_Ctx",
      [](const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_32fc_C1R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 1, 1, num::Op::AddC>(
      "nppiAddC_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_32fc_C3IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 3, 3, num::Op::AddC>(
      "nppiAddC_32fc_C3IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32fc_C3R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 3, 3, num::Op::AddC>(
      "nppiAddC_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32fc_C4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 4, num::Op::AddC>(
      "nppiAddC_32fc_C4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32fc_C4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 4, num::Op::AddC>(
      "nppiAddC_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32fc_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::AddC>(
      "nppiAddC_32s_C1IRSfs_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_32s_C1RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::AddC>(
      "nppiAddC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_32s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::AddC>(
      "nppiAddC_32s_C3IRSfs_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32s_C3RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::AddC>(
      "nppiAddC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 4, 3, num::Op::AddC>(
      "nppiAddC_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32sc_AC4RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 4, 3, num::Op::AddC>(
      "nppiAddC_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 1, 1, num::Op::AddC>(
      "nppiAddC_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_32sc_C1RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 1, 1, num::Op::AddC>(
      "nppiAddC_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_32sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 3, 3, num::Op::AddC>(
      "nppiAddC_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_32sc_C3RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 3, 3, num::Op::AddC>(
      "nppiAddC_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_32sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_8u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::AddC>(
      "nppiAddC_8u_AC4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_8u_AC4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::AddC>(
      "nppiAddC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_8u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::AddC>(
      "nppiAddC_8u_C1IRSfs_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_8u_C1RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::AddC>(
      "nppiAddC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAddC_8u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::AddC>(
      "nppiAddC_8u_C3IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_8u_C3RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::AddC>(
      "nppiAddC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_8u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::AddC>(
      "nppiAddC_8u_C4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddC_8u_C4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::AddC>(
      "nppiAddC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAddC_8u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAddDeviceC_16f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_16f_C1IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddDeviceC_16f_C1R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddDeviceC_16f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_16f_C3IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddDeviceC_16f_C3R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddDeviceC_16f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_16f_C4IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddDeviceC_16f_C4R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddDeviceC_16s_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_16s_AC4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_AC4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_16s_C1IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_C1RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_16s_C3IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_C3RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_16s_C4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16s_C4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16s_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_16u_AC4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_AC4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_16u_C1IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_C1RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_16u_C3IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_C3RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_16u_C4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_16u_C4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_16u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_32f_AC4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_32f_AC4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_AC4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_AC4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_AC4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_32f_C1IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_C1R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_32f_C3IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_C3R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_32f_C4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32f_C4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiAddDeviceC_32s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_32s_C1IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_32s_C1RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_32s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_32s_C3IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_32s_C3RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_32s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_8u_AC4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_AC4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 3, num::Op::AddC>(
      "nppiAddDeviceC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_8u_C1IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_C1RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 1, 1, num::Op::AddC>(
      "nppiAddDeviceC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_8u_C3IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_C3RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 3, 3, num::Op::AddC>(
      "nppiAddDeviceC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_8u_C4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddDeviceC_8u_C4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 4, num::Op::AddC>(
      "nppiAddDeviceC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiAddDeviceC_8u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiAddProduct_16f_C1IR_Ctx) {
  num::run_add_product<Npp16f, Npp16f, Npp16f, 1, 1, false>(
      "nppiAddProduct_16f_C1IR_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiAddProduct_16f_C1IR_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiAddProduct_16u32f_C1IMR_Ctx) {
  num::run_add_product<Npp16u, Npp16u, Npp32f, 1, 1, true>(
      "nppiAddProduct_16u32f_C1IMR_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)ms; (void)sd; (void)r; (void)c; return nppiAddProduct_16u32f_C1IMR_Ctx(a, sa, b, sb, m, ms, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddProduct_16u32f_C1IR_Ctx) {
  num::run_add_product<Npp16u, Npp16u, Npp32f, 1, 1, false>(
      "nppiAddProduct_16u32f_C1IR_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiAddProduct_16u32f_C1IR_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddProduct_32f_C1IMR_Ctx) {
  num::run_add_product<Npp32f, Npp32f, Npp32f, 1, 1, true>(
      "nppiAddProduct_32f_C1IMR_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)ms; (void)sd; (void)r; (void)c; return nppiAddProduct_32f_C1IMR_Ctx(a, sa, b, sb, m, ms, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddProduct_32f_C1IR_Ctx) {
  num::run_add_product<Npp32f, Npp32f, Npp32f, 1, 1, false>(
      "nppiAddProduct_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiAddProduct_32f_C1IR_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddProduct_8u32f_C1IMR_Ctx) {
  num::run_add_product<Npp8u, Npp8u, Npp32f, 1, 1, true>(
      "nppiAddProduct_8u32f_C1IMR_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)ms; (void)sd; (void)r; (void)c; return nppiAddProduct_8u32f_C1IMR_Ctx(a, sa, b, sb, m, ms, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddProduct_8u32f_C1IR_Ctx) {
  num::run_add_product<Npp8u, Npp8u, Npp32f, 1, 1, false>(
      "nppiAddProduct_8u32f_C1IR_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiAddProduct_8u32f_C1IR_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddSquare_16u32f_C1IMR_Ctx) {
  num::run_add_square<Npp16u, Npp32f, 1, 1, true>(
      "nppiAddSquare_16u32f_C1IMR_Ctx",
      [](const Npp16u *a, int sa, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)ms; (void)sd; (void)r; (void)c; return nppiAddSquare_16u32f_C1IMR_Ctx(a, sa, m, ms, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddSquare_16u32f_C1IR_Ctx) {
  num::run_add_square<Npp16u, Npp32f, 1, 1, false>(
      "nppiAddSquare_16u32f_C1IR_Ctx",
      [](const Npp16u *a, int sa, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiAddSquare_16u32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddSquare_32f_C1IMR_Ctx) {
  num::run_add_square<Npp32f, Npp32f, 1, 1, true>(
      "nppiAddSquare_32f_C1IMR_Ctx",
      [](const Npp32f *a, int sa, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)ms; (void)sd; (void)r; (void)c; return nppiAddSquare_32f_C1IMR_Ctx(a, sa, m, ms, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddSquare_32f_C1IR_Ctx) {
  num::run_add_square<Npp32f, Npp32f, 1, 1, false>(
      "nppiAddSquare_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiAddSquare_32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddSquare_8u32f_C1IMR_Ctx) {
  num::run_add_square<Npp8u, Npp32f, 1, 1, true>(
      "nppiAddSquare_8u32f_C1IMR_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)ms; (void)sd; (void)r; (void)c; return nppiAddSquare_8u32f_C1IMR_Ctx(a, sa, m, ms, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddSquare_8u32f_C1IR_Ctx) {
  num::run_add_square<Npp8u, Npp32f, 1, 1, false>(
      "nppiAddSquare_8u32f_C1IR_Ctx",
      [](const Npp8u *a, int sa, Npp32f *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiAddSquare_8u32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddWeighted_16u32f_C1IMR_Ctx) {
  num::run_add_weighted<Npp16u, Npp32f, 1, 1, true>(
      "nppiAddWeighted_16u32f_C1IMR_Ctx",
      [](const Npp16u *a, int sa, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, float alpha, NppStreamContext c) { (void)sa; (void)ms; (void)sd; (void)r; (void)alpha; (void)c; return nppiAddWeighted_16u32f_C1IMR_Ctx(a, sa, m, ms, d, sd, r, alpha, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddWeighted_16u32f_C1IR_Ctx) {
  num::run_add_weighted<Npp16u, Npp32f, 1, 1, false>(
      "nppiAddWeighted_16u32f_C1IR_Ctx",
      [](const Npp16u *a, int sa, Npp32f *d, int sd, NppiSize r, float alpha, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)alpha; (void)c; return nppiAddWeighted_16u32f_C1IR_Ctx(a, sa, d, sd, r, alpha, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddWeighted_32f_C1IMR_Ctx) {
  num::run_add_weighted<Npp32f, Npp32f, 1, 1, true>(
      "nppiAddWeighted_32f_C1IMR_Ctx",
      [](const Npp32f *a, int sa, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, float alpha, NppStreamContext c) { (void)sa; (void)ms; (void)sd; (void)r; (void)alpha; (void)c; return nppiAddWeighted_32f_C1IMR_Ctx(a, sa, m, ms, d, sd, r, alpha, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddWeighted_32f_C1IR_Ctx) {
  num::run_add_weighted<Npp32f, Npp32f, 1, 1, false>(
      "nppiAddWeighted_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, float alpha, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)alpha; (void)c; return nppiAddWeighted_32f_C1IR_Ctx(a, sa, d, sd, r, alpha, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddWeighted_8u32f_C1IMR_Ctx) {
  num::run_add_weighted<Npp8u, Npp32f, 1, 1, true>(
      "nppiAddWeighted_8u32f_C1IMR_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *m, int ms, Npp32f *d, int sd, NppiSize r, float alpha, NppStreamContext c) { (void)sa; (void)ms; (void)sd; (void)r; (void)alpha; (void)c; return nppiAddWeighted_8u32f_C1IMR_Ctx(a, sa, m, ms, d, sd, r, alpha, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAddWeighted_8u32f_C1IR_Ctx) {
  num::run_add_weighted<Npp8u, Npp32f, 1, 1, false>(
      "nppiAddWeighted_8u32f_C1IR_Ctx",
      [](const Npp8u *a, int sa, Npp32f *d, int sd, NppiSize r, float alpha, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)alpha; (void)c; return nppiAddWeighted_8u32f_C1IR_Ctx(a, sa, d, sd, r, alpha, c); },
      {0}, 1e-05, 0.0001);
}

TEST(NumGen, nppiAdd_16f_C1IR_Ctx) {
  num::run_binary_inplace<Npp16f, 1, 1, num::Op::Add>(
      "nppiAdd_16f_C1IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAdd_16f_C1R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 1, 1, num::Op::Add>(
      "nppiAdd_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAdd_16f_C3IR_Ctx) {
  num::run_binary_inplace<Npp16f, 3, 3, num::Op::Add>(
      "nppiAdd_16f_C3IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAdd_16f_C3R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 3, 3, num::Op::Add>(
      "nppiAdd_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAdd_16f_C4IR_Ctx) {
  num::run_binary_inplace<Npp16f, 4, 4, num::Op::Add>(
      "nppiAdd_16f_C4IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAdd_16f_C4R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 4, 4, num::Op::Add>(
      "nppiAdd_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiAdd_16s_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 3, num::Op::Add>(
      "nppiAdd_16s_AC4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_AC4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 3, num::Op::Add>(
      "nppiAdd_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 1, 1, num::Op::Add>(
      "nppiAdd_16s_C1IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_C1RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 1, 1, num::Op::Add>(
      "nppiAdd_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 3, 3, num::Op::Add>(
      "nppiAdd_16s_C3IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_C3RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 3, 3, num::Op::Add>(
      "nppiAdd_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 4, num::Op::Add>(
      "nppiAdd_16s_C4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16s_C4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 4, num::Op::Add>(
      "nppiAdd_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16s_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 4, 3, num::Op::Add>(
      "nppiAdd_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16sc_AC4RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 4, 3, num::Op::Add>(
      "nppiAdd_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 1, 1, num::Op::Add>(
      "nppiAdd_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16sc_C1RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 1, 1, num::Op::Add>(
      "nppiAdd_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 3, 3, num::Op::Add>(
      "nppiAdd_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16sc_C3RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 3, 3, num::Op::Add>(
      "nppiAdd_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::Add>(
      "nppiAdd_16u_AC4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_AC4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::Add>(
      "nppiAdd_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::Add>(
      "nppiAdd_16u_C1IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_C1RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::Add>(
      "nppiAdd_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::Add>(
      "nppiAdd_16u_C3IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_C3RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::Add>(
      "nppiAdd_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::Add>(
      "nppiAdd_16u_C4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_16u_C4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::Add>(
      "nppiAdd_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_16u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32f_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 3, num::Op::Add>(
      "nppiAdd_32f_AC4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_AC4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 3, num::Op::Add>(
      "nppiAdd_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_C1IR_Ctx) {
  num::run_binary_inplace<Npp32f, 1, 1, num::Op::Add>(
      "nppiAdd_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_C1R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 1, 1, num::Op::Add>(
      "nppiAdd_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_C3IR_Ctx) {
  num::run_binary_inplace<Npp32f, 3, 3, num::Op::Add>(
      "nppiAdd_32f_C3IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_C3R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 3, 3, num::Op::Add>(
      "nppiAdd_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_C4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 4, num::Op::Add>(
      "nppiAdd_32f_C4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32f_C4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 4, num::Op::Add>(
      "nppiAdd_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiAdd_32fc_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 3, num::Op::Add>(
      "nppiAdd_32fc_AC4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_AC4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 3, num::Op::Add>(
      "nppiAdd_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_C1IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 1, 1, num::Op::Add>(
      "nppiAdd_32fc_C1IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_C1R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 1, 1, num::Op::Add>(
      "nppiAdd_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_C3IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 3, 3, num::Op::Add>(
      "nppiAdd_32fc_C3IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_C3R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 3, 3, num::Op::Add>(
      "nppiAdd_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_C4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 4, num::Op::Add>(
      "nppiAdd_32fc_C4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32fc_C4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 4, num::Op::Add>(
      "nppiAdd_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32fc_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::Add>(
      "nppiAdd_32s_C1IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32s_C1RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Add>(
      "nppiAdd_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Add>(
      "nppiAdd_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::Add>(
      "nppiAdd_32s_C3IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32s_C3RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::Add>(
      "nppiAdd_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 4, 3, num::Op::Add>(
      "nppiAdd_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32sc_AC4RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 4, 3, num::Op::Add>(
      "nppiAdd_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 1, 1, num::Op::Add>(
      "nppiAdd_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32sc_C1RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 1, 1, num::Op::Add>(
      "nppiAdd_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 3, 3, num::Op::Add>(
      "nppiAdd_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_32sc_C3RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 3, 3, num::Op::Add>(
      "nppiAdd_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_32sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::Add>(
      "nppiAdd_8u_AC4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_AC4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::Add>(
      "nppiAdd_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::Add>(
      "nppiAdd_8u_C1IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_C1RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::Add>(
      "nppiAdd_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::Add>(
      "nppiAdd_8u_C3IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_C3RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::Add>(
      "nppiAdd_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::Add>(
      "nppiAdd_8u_C4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAdd_8u_C4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::Add>(
      "nppiAdd_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAdd_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAlphaCompC_16s_C1R_Ctx) {
  num::run_alphacomp_c<Npp16s, 1, 1, false>(
      "nppiAlphaCompC_16s_C1R_Ctx",
      [](const Npp16s *s1, int ss1, Npp16s a1, const Npp16s *s2, int ss2, Npp16s a2, Npp16s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_16s_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_16u_AC4R_Ctx) {
  num::run_alphacomp_c<Npp16u, 4, 3, true>(
      "nppiAlphaCompC_16u_AC4R_Ctx",
      [](const Npp16u *s1, int ss1, Npp16u a1, const Npp16u *s2, int ss2, Npp16u a2, Npp16u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_16u_AC4R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_16u_C1R_Ctx) {
  num::run_alphacomp_c<Npp16u, 1, 1, false>(
      "nppiAlphaCompC_16u_C1R_Ctx",
      [](const Npp16u *s1, int ss1, Npp16u a1, const Npp16u *s2, int ss2, Npp16u a2, Npp16u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_16u_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_16u_C3R_Ctx) {
  num::run_alphacomp_c<Npp16u, 3, 3, false>(
      "nppiAlphaCompC_16u_C3R_Ctx",
      [](const Npp16u *s1, int ss1, Npp16u a1, const Npp16u *s2, int ss2, Npp16u a2, Npp16u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_16u_C3R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_16u_C4R_Ctx) {
  num::run_alphacomp_c<Npp16u, 4, 4, false>(
      "nppiAlphaCompC_16u_C4R_Ctx",
      [](const Npp16u *s1, int ss1, Npp16u a1, const Npp16u *s2, int ss2, Npp16u a2, Npp16u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_16u_C4R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_32f_C1R_Ctx) {
  num::run_alphacomp_c<Npp32f, 1, 1, false>(
      "nppiAlphaCompC_32f_C1R_Ctx",
      [](const Npp32f *s1, int ss1, Npp32f a1, const Npp32f *s2, int ss2, Npp32f a2, Npp32f *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_32f_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_32s_C1R_Ctx) {
  num::run_alphacomp_c<Npp32s, 1, 1, false>(
      "nppiAlphaCompC_32s_C1R_Ctx",
      [](const Npp32s *s1, int ss1, Npp32s a1, const Npp32s *s2, int ss2, Npp32s a2, Npp32s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_32s_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_32u_C1R_Ctx) {
  num::run_alphacomp_c<Npp32u, 1, 1, false>(
      "nppiAlphaCompC_32u_C1R_Ctx",
      [](const Npp32u *s1, int ss1, Npp32u a1, const Npp32u *s2, int ss2, Npp32u a2, Npp32u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_32u_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_8s_C1R_Ctx) {
  num::run_alphacomp_c<Npp8s, 1, 1, false>(
      "nppiAlphaCompC_8s_C1R_Ctx",
      [](const Npp8s *s1, int ss1, Npp8s a1, const Npp8s *s2, int ss2, Npp8s a2, Npp8s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_8s_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_8u_AC4R_Ctx) {
  num::run_alphacomp_c<Npp8u, 4, 3, true>(
      "nppiAlphaCompC_8u_AC4R_Ctx",
      [](const Npp8u *s1, int ss1, Npp8u a1, const Npp8u *s2, int ss2, Npp8u a2, Npp8u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_8u_AC4R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_8u_C1R_Ctx) {
  num::run_alphacomp_c<Npp8u, 1, 1, false>(
      "nppiAlphaCompC_8u_C1R_Ctx",
      [](const Npp8u *s1, int ss1, Npp8u a1, const Npp8u *s2, int ss2, Npp8u a2, Npp8u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_8u_C1R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_8u_C3R_Ctx) {
  num::run_alphacomp_c<Npp8u, 3, 3, false>(
      "nppiAlphaCompC_8u_C3R_Ctx",
      [](const Npp8u *s1, int ss1, Npp8u a1, const Npp8u *s2, int ss2, Npp8u a2, Npp8u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_8u_C3R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaCompC_8u_C4R_Ctx) {
  num::run_alphacomp_c<Npp8u, 4, 4, false>(
      "nppiAlphaCompC_8u_C4R_Ctx",
      [](const Npp8u *s1, int ss1, Npp8u a1, const Npp8u *s2, int ss2, Npp8u a2, Npp8u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaCompC_8u_C4R_Ctx(s1, ss1, a1, s2, ss2, a2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_16s_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp16s, 2, 1>(
      "nppiAlphaComp_16s_AC1R_Ctx",
      [](const Npp16s *s1, int ss1, const Npp16s *s2, int ss2, Npp16s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_16s_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_16u_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp16u, 2, 1>(
      "nppiAlphaComp_16u_AC1R_Ctx",
      [](const Npp16u *s1, int ss1, const Npp16u *s2, int ss2, Npp16u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_16u_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_16u_AC4R_Ctx) {
  num::run_alphacomp_pixel<Npp16u, 4, 3>(
      "nppiAlphaComp_16u_AC4R_Ctx",
      [](const Npp16u *s1, int ss1, const Npp16u *s2, int ss2, Npp16u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_16u_AC4R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_32f_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp32f, 2, 1>(
      "nppiAlphaComp_32f_AC1R_Ctx",
      [](const Npp32f *s1, int ss1, const Npp32f *s2, int ss2, Npp32f *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_32f_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_32f_AC4R_Ctx) {
  num::run_alphacomp_pixel<Npp32f, 4, 3>(
      "nppiAlphaComp_32f_AC4R_Ctx",
      [](const Npp32f *s1, int ss1, const Npp32f *s2, int ss2, Npp32f *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_32f_AC4R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_32s_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp32s, 2, 1>(
      "nppiAlphaComp_32s_AC1R_Ctx",
      [](const Npp32s *s1, int ss1, const Npp32s *s2, int ss2, Npp32s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_32s_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_32s_AC4R_Ctx) {
  num::run_alphacomp_pixel<Npp32s, 4, 3>(
      "nppiAlphaComp_32s_AC4R_Ctx",
      [](const Npp32s *s1, int ss1, const Npp32s *s2, int ss2, Npp32s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_32s_AC4R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_32u_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp32u, 2, 1>(
      "nppiAlphaComp_32u_AC1R_Ctx",
      [](const Npp32u *s1, int ss1, const Npp32u *s2, int ss2, Npp32u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_32u_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_32u_AC4R_Ctx) {
  num::run_alphacomp_pixel<Npp32u, 4, 3>(
      "nppiAlphaComp_32u_AC4R_Ctx",
      [](const Npp32u *s1, int ss1, const Npp32u *s2, int ss2, Npp32u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_32u_AC4R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_8s_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp8s, 2, 1>(
      "nppiAlphaComp_8s_AC1R_Ctx",
      [](const Npp8s *s1, int ss1, const Npp8s *s2, int ss2, Npp8s *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_8s_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_8u_AC1R_Ctx) {
  num::run_alphacomp_pixel<Npp8u, 2, 1>(
      "nppiAlphaComp_8u_AC1R_Ctx",
      [](const Npp8u *s1, int ss1, const Npp8u *s2, int ss2, Npp8u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_8u_AC1R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaComp_8u_AC4R_Ctx) {
  num::run_alphacomp_pixel<Npp8u, 4, 3>(
      "nppiAlphaComp_8u_AC4R_Ctx",
      [](const Npp8u *s1, int ss1, const Npp8u *s2, int ss2, Npp8u *d, int ds, NppiSize r, NppiAlphaOp op, NppStreamContext c) { return nppiAlphaComp_8u_AC4R_Ctx(s1, ss1, s2, ss2, d, ds, r, op, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_AC4IR_Ctx) {
  num::run_alphapremul<Npp16u, 4, 3, true, true, true>(
      "nppiAlphaPremulC_16u_AC4IR_Ctx",
      [](Npp16u a, Npp16u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_AC4IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_AC4R_Ctx) {
  num::run_alphapremul<Npp16u, 4, 3, true, true, false>(
      "nppiAlphaPremulC_16u_AC4R_Ctx",
      [](const Npp16u *s, int ss, Npp16u a, Npp16u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_AC4R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_C1IR_Ctx) {
  num::run_alphapremul<Npp16u, 1, 1, true, false, true>(
      "nppiAlphaPremulC_16u_C1IR_Ctx",
      [](Npp16u a, Npp16u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_C1IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_C1R_Ctx) {
  num::run_alphapremul<Npp16u, 1, 1, true, false, false>(
      "nppiAlphaPremulC_16u_C1R_Ctx",
      [](const Npp16u *s, int ss, Npp16u a, Npp16u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_C1R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_C3IR_Ctx) {
  num::run_alphapremul<Npp16u, 3, 3, true, false, true>(
      "nppiAlphaPremulC_16u_C3IR_Ctx",
      [](Npp16u a, Npp16u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_C3IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_C3R_Ctx) {
  num::run_alphapremul<Npp16u, 3, 3, true, false, false>(
      "nppiAlphaPremulC_16u_C3R_Ctx",
      [](const Npp16u *s, int ss, Npp16u a, Npp16u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_C3R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_C4IR_Ctx) {
  num::run_alphapremul<Npp16u, 4, 4, true, false, true>(
      "nppiAlphaPremulC_16u_C4IR_Ctx",
      [](Npp16u a, Npp16u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_C4IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_16u_C4R_Ctx) {
  num::run_alphapremul<Npp16u, 4, 4, true, false, false>(
      "nppiAlphaPremulC_16u_C4R_Ctx",
      [](const Npp16u *s, int ss, Npp16u a, Npp16u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_16u_C4R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_AC4IR_Ctx) {
  num::run_alphapremul<Npp8u, 4, 3, true, true, true>(
      "nppiAlphaPremulC_8u_AC4IR_Ctx",
      [](Npp8u a, Npp8u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_AC4IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_AC4R_Ctx) {
  num::run_alphapremul<Npp8u, 4, 3, true, true, false>(
      "nppiAlphaPremulC_8u_AC4R_Ctx",
      [](const Npp8u *s, int ss, Npp8u a, Npp8u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_AC4R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_C1IR_Ctx) {
  num::run_alphapremul<Npp8u, 1, 1, true, false, true>(
      "nppiAlphaPremulC_8u_C1IR_Ctx",
      [](Npp8u a, Npp8u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_C1IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_C1R_Ctx) {
  num::run_alphapremul<Npp8u, 1, 1, true, false, false>(
      "nppiAlphaPremulC_8u_C1R_Ctx",
      [](const Npp8u *s, int ss, Npp8u a, Npp8u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_C1R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_C3IR_Ctx) {
  num::run_alphapremul<Npp8u, 3, 3, true, false, true>(
      "nppiAlphaPremulC_8u_C3IR_Ctx",
      [](Npp8u a, Npp8u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_C3IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_C3R_Ctx) {
  num::run_alphapremul<Npp8u, 3, 3, true, false, false>(
      "nppiAlphaPremulC_8u_C3R_Ctx",
      [](const Npp8u *s, int ss, Npp8u a, Npp8u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_C3R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_C4IR_Ctx) {
  num::run_alphapremul<Npp8u, 4, 4, true, false, true>(
      "nppiAlphaPremulC_8u_C4IR_Ctx",
      [](Npp8u a, Npp8u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_C4IR_Ctx(a, sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremulC_8u_C4R_Ctx) {
  num::run_alphapremul<Npp8u, 4, 4, true, false, false>(
      "nppiAlphaPremulC_8u_C4R_Ctx",
      [](const Npp8u *s, int ss, Npp8u a, Npp8u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremulC_8u_C4R_Ctx(s, ss, a, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremul_16u_AC4IR_Ctx) {
  num::run_alphapremul<Npp16u, 4, 3, false, true, true>(
      "nppiAlphaPremul_16u_AC4IR_Ctx",
      [](Npp16u, Npp16u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremul_16u_AC4IR_Ctx(sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremul_16u_AC4R_Ctx) {
  num::run_alphapremul<Npp16u, 4, 3, false, true, false>(
      "nppiAlphaPremul_16u_AC4R_Ctx",
      [](const Npp16u *s, int ss, Npp16u, Npp16u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremul_16u_AC4R_Ctx(s, ss, d, ds, r, c); });
}

TEST(NumGen, nppiAlphaPremul_8u_AC4IR_Ctx) {
  num::run_alphapremul<Npp8u, 4, 3, false, true, true>(
      "nppiAlphaPremul_8u_AC4IR_Ctx",
      [](Npp8u, Npp8u *sd, int st, NppiSize r, NppStreamContext c) { return nppiAlphaPremul_8u_AC4IR_Ctx(sd, st, r, c); });
}

TEST(NumGen, nppiAlphaPremul_8u_AC4R_Ctx) {
  num::run_alphapremul<Npp8u, 4, 3, false, true, false>(
      "nppiAlphaPremul_8u_AC4R_Ctx",
      [](const Npp8u *s, int ss, Npp8u, Npp8u *d, int ds, NppiSize r, NppStreamContext c) { return nppiAlphaPremul_8u_AC4R_Ctx(s, ss, d, ds, r, c); });
}

TEST(NumGen, nppiAndC_16u_AC4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::AndC>(
      "nppiAndC_16u_AC4IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_16u_AC4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::AndC>(
      "nppiAndC_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_16u_C1IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::AndC>(
      "nppiAndC_16u_C1IR_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAndC_16u_C1R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::AndC>(
      "nppiAndC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAndC_16u_C3IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::AndC>(
      "nppiAndC_16u_C3IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_16u_C3R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::AndC>(
      "nppiAndC_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_16u_C4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::AndC>(
      "nppiAndC_16u_C4IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_16u_C4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::AndC>(
      "nppiAndC_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_16u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_32s_AC4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 4, 3, num::Op::AndC>(
      "nppiAndC_32s_AC4IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_32s_AC4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 4, 3, num::Op::AndC>(
      "nppiAndC_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_32s_C1IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::AndC>(
      "nppiAndC_32s_C1IR_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAndC_32s_C1R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::AndC>(
      "nppiAndC_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAndC_32s_C3IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::AndC>(
      "nppiAndC_32s_C3IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_32s_C3R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::AndC>(
      "nppiAndC_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_32s_C4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 4, 4, num::Op::AndC>(
      "nppiAndC_32s_C4IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_32s_C4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 4, 4, num::Op::AndC>(
      "nppiAndC_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_32s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_8u_AC4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::AndC>(
      "nppiAndC_8u_AC4IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_8u_AC4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::AndC>(
      "nppiAndC_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_8u_C1IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::AndC>(
      "nppiAndC_8u_C1IR_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAndC_8u_C1R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::AndC>(
      "nppiAndC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiAndC_8u_C3IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::AndC>(
      "nppiAndC_8u_C3IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_8u_C3R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::AndC>(
      "nppiAndC_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_8u_C4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::AndC>(
      "nppiAndC_8u_C4IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAndC_8u_C4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::AndC>(
      "nppiAndC_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAndC_8u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiAnd_16u_AC4IR_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::And>(
      "nppiAnd_16u_AC4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_AC4R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::And>(
      "nppiAnd_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_C1IR_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::And>(
      "nppiAnd_16u_C1IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_C1R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::And>(
      "nppiAnd_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_C3IR_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::And>(
      "nppiAnd_16u_C3IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_C3R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::And>(
      "nppiAnd_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_C4IR_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::And>(
      "nppiAnd_16u_C4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_16u_C4R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::And>(
      "nppiAnd_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_16u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 3, num::Op::And>(
      "nppiAnd_32s_AC4IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_AC4R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 3, num::Op::And>(
      "nppiAnd_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_C1IR_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::And>(
      "nppiAnd_32s_C1IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::And>(
      "nppiAnd_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_C3IR_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::And>(
      "nppiAnd_32s_C3IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_C3R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::And>(
      "nppiAnd_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_C4IR_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 4, num::Op::And>(
      "nppiAnd_32s_C4IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_32s_C4R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 4, num::Op::And>(
      "nppiAnd_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_32s_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_AC4IR_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::And>(
      "nppiAnd_8u_AC4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_AC4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::And>(
      "nppiAnd_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_C1IR_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::And>(
      "nppiAnd_8u_C1IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_C1R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::And>(
      "nppiAnd_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_C3IR_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::And>(
      "nppiAnd_8u_C3IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_C3R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::And>(
      "nppiAnd_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_C4IR_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::And>(
      "nppiAnd_8u_C4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiAnd_8u_C4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::And>(
      "nppiAnd_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiAnd_8u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiDivC_16f_C1IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivC_16f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, true);
}

TEST(NumGen, nppiDivC_16f_C1R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, true);
}

TEST(NumGen, nppiDivC_16f_C3IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivC_16f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, true);
}

TEST(NumGen, nppiDivC_16f_C3R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, true);
}

TEST(NumGen, nppiDivC_16f_C4IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivC_16f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, true);
}

TEST(NumGen, nppiDivC_16f_C4R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, true);
}

TEST(NumGen, nppiDivC_16s_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 3, num::Op::DivC>(
      "nppiDivC_16s_AC4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16s_AC4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 3, num::Op::DivC>(
      "nppiDivC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 1, 1, num::Op::DivC>(
      "nppiDivC_16s_C1IRSfs_Ctx",
      [](const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_16s_C1RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 1, 1, num::Op::DivC>(
      "nppiDivC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_16s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 3, 3, num::Op::DivC>(
      "nppiDivC_16s_C3IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16s_C3RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 3, 3, num::Op::DivC>(
      "nppiDivC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16s_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 4, num::Op::DivC>(
      "nppiDivC_16s_C4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16s_C4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 4, num::Op::DivC>(
      "nppiDivC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16s_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 4, 3, num::Op::DivC>(
      "nppiDivC_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16sc_AC4RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 4, 3, num::Op::DivC>(
      "nppiDivC_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 1, 1, num::Op::DivC>(
      "nppiDivC_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_16sc_C1RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 1, 1, num::Op::DivC>(
      "nppiDivC_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_16sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 3, 3, num::Op::DivC>(
      "nppiDivC_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16sc_C3RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 3, 3, num::Op::DivC>(
      "nppiDivC_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::DivC>(
      "nppiDivC_16u_AC4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16u_AC4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::DivC>(
      "nppiDivC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::DivC>(
      "nppiDivC_16u_C1IRSfs_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_16u_C1RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::DivC>(
      "nppiDivC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_16u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::DivC>(
      "nppiDivC_16u_C3IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16u_C3RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::DivC>(
      "nppiDivC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::DivC>(
      "nppiDivC_16u_C4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_16u_C4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::DivC>(
      "nppiDivC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_16u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32f_AC4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 3, num::Op::DivC>(
      "nppiDivC_32f_AC4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, true);
}

TEST(NumGen, nppiDivC_32f_AC4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 3, num::Op::DivC>(
      "nppiDivC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, true);
}

TEST(NumGen, nppiDivC_32f_C1IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivC_32f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, true);
}

TEST(NumGen, nppiDivC_32f_C1R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, true);
}

TEST(NumGen, nppiDivC_32f_C3IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivC_32f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, true);
}

TEST(NumGen, nppiDivC_32f_C3R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, true);
}

TEST(NumGen, nppiDivC_32f_C4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivC_32f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, true);
}

TEST(NumGen, nppiDivC_32f_C4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, true);
}

TEST(NumGen, nppiDivC_32fc_AC4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 3, num::Op::DivC>(
      "nppiDivC_32fc_AC4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32fc_AC4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 3, num::Op::DivC>(
      "nppiDivC_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32fc_C1IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 1, 1, num::Op::DivC>(
      "nppiDivC_32fc_C1IR_Ctx",
      [](const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_32fc_C1R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 1, 1, num::Op::DivC>(
      "nppiDivC_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_32fc_C3IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 3, 3, num::Op::DivC>(
      "nppiDivC_32fc_C3IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32fc_C3R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 3, 3, num::Op::DivC>(
      "nppiDivC_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32fc_C4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 4, num::Op::DivC>(
      "nppiDivC_32fc_C4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32fc_C4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 4, num::Op::DivC>(
      "nppiDivC_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32fc_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::DivC>(
      "nppiDivC_32s_C1IRSfs_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_32s_C1RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::DivC>(
      "nppiDivC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_32s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::DivC>(
      "nppiDivC_32s_C3IRSfs_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32s_C3RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::DivC>(
      "nppiDivC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 4, 3, num::Op::DivC>(
      "nppiDivC_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32sc_AC4RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 4, 3, num::Op::DivC>(
      "nppiDivC_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 1, 1, num::Op::DivC>(
      "nppiDivC_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_32sc_C1RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 1, 1, num::Op::DivC>(
      "nppiDivC_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_32sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 3, 3, num::Op::DivC>(
      "nppiDivC_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_32sc_C3RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 3, 3, num::Op::DivC>(
      "nppiDivC_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_32sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_8u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::DivC>(
      "nppiDivC_8u_AC4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_8u_AC4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::DivC>(
      "nppiDivC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_8u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::DivC>(
      "nppiDivC_8u_C1IRSfs_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_8u_C1RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::DivC>(
      "nppiDivC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, true);
}

TEST(NumGen, nppiDivC_8u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::DivC>(
      "nppiDivC_8u_C3IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_8u_C3RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::DivC>(
      "nppiDivC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_8u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::DivC>(
      "nppiDivC_8u_C4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivC_8u_C4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::DivC>(
      "nppiDivC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDivC_8u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, true);
}

TEST(NumGen, nppiDivDeviceC_16f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_16f_C1IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiDivDeviceC_16f_C1R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiDivDeviceC_16f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_16f_C3IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiDivDeviceC_16f_C3R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiDivDeviceC_16f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_16f_C4IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiDivDeviceC_16f_C4R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiDivDeviceC_16s_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_16s_AC4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_AC4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_16s_C1IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_C1RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_16s_C3IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_C3RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_16s_C4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16s_C4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16s_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_16u_AC4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_AC4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_16u_C1IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_C1RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_16u_C3IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_C3RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_16u_C4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_16u_C4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_16u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_32f_AC4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_32f_AC4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_AC4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_AC4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_AC4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_32f_C1IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_C1R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_32f_C3IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_C3R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_32f_C4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32f_C4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiDivDeviceC_32s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_32s_C1IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_32s_C1RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_32s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_32s_C3IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_32s_C3RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_32s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_8u_AC4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_AC4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 3, num::Op::DivC>(
      "nppiDivDeviceC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_8u_C1IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_C1RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 1, 1, num::Op::DivC>(
      "nppiDivDeviceC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_8u_C3IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_C3RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 3, 3, num::Op::DivC>(
      "nppiDivDeviceC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_8u_C4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDivDeviceC_8u_C4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 4, num::Op::DivC>(
      "nppiDivDeviceC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiDivDeviceC_8u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiDiv_16f_C1IR_Ctx) {
  num::run_binary_inplace<Npp16f, 1, 1, num::Op::Div>(
      "nppiDiv_16f_C1IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, true);
}

TEST(NumGen, nppiDiv_16f_C1R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 1, 1, num::Op::Div>(
      "nppiDiv_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, true);
}

TEST(NumGen, nppiDiv_16f_C3IR_Ctx) {
  num::run_binary_inplace<Npp16f, 3, 3, num::Op::Div>(
      "nppiDiv_16f_C3IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, true);
}

TEST(NumGen, nppiDiv_16f_C3R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 3, 3, num::Op::Div>(
      "nppiDiv_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, true);
}

TEST(NumGen, nppiDiv_16f_C4IR_Ctx) {
  num::run_binary_inplace<Npp16f, 4, 4, num::Op::Div>(
      "nppiDiv_16f_C4IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, true);
}

TEST(NumGen, nppiDiv_16f_C4R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 4, 4, num::Op::Div>(
      "nppiDiv_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, true);
}

TEST(NumGen, nppiDiv_16s_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 3, num::Op::Div>(
      "nppiDiv_16s_AC4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_AC4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 3, num::Op::Div>(
      "nppiDiv_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 1, 1, num::Op::Div>(
      "nppiDiv_16s_C1IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_C1RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 1, 1, num::Op::Div>(
      "nppiDiv_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 3, 3, num::Op::Div>(
      "nppiDiv_16s_C3IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_C3RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 3, 3, num::Op::Div>(
      "nppiDiv_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 4, num::Op::Div>(
      "nppiDiv_16s_C4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16s_C4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 4, num::Op::Div>(
      "nppiDiv_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16s_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 4, 3, num::Op::Div>(
      "nppiDiv_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16sc_AC4RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 4, 3, num::Op::Div>(
      "nppiDiv_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 1, 1, num::Op::Div>(
      "nppiDiv_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16sc_C1RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 1, 1, num::Op::Div>(
      "nppiDiv_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 3, 3, num::Op::Div>(
      "nppiDiv_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16sc_C3RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 3, 3, num::Op::Div>(
      "nppiDiv_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::Div>(
      "nppiDiv_16u_AC4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_AC4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::Div>(
      "nppiDiv_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::Div>(
      "nppiDiv_16u_C1IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_C1RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::Div>(
      "nppiDiv_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::Div>(
      "nppiDiv_16u_C3IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_C3RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::Div>(
      "nppiDiv_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::Div>(
      "nppiDiv_16u_C4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_16u_C4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::Div>(
      "nppiDiv_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_16u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32f_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 3, num::Op::Div>(
      "nppiDiv_32f_AC4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_AC4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 3, num::Op::Div>(
      "nppiDiv_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_C1IR_Ctx) {
  num::run_binary_inplace<Npp32f, 1, 1, num::Op::Div>(
      "nppiDiv_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_C1R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 1, 1, num::Op::Div>(
      "nppiDiv_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_C3IR_Ctx) {
  num::run_binary_inplace<Npp32f, 3, 3, num::Op::Div>(
      "nppiDiv_32f_C3IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_C3R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 3, 3, num::Op::Div>(
      "nppiDiv_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_C4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 4, num::Op::Div>(
      "nppiDiv_32f_C4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32f_C4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 4, num::Op::Div>(
      "nppiDiv_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true);
}

TEST(NumGen, nppiDiv_32fc_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 3, num::Op::Div>(
      "nppiDiv_32fc_AC4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_AC4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 3, num::Op::Div>(
      "nppiDiv_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_C1IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 1, 1, num::Op::Div>(
      "nppiDiv_32fc_C1IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_C1R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 1, 1, num::Op::Div>(
      "nppiDiv_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_C3IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 3, 3, num::Op::Div>(
      "nppiDiv_32fc_C3IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_C3R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 3, 3, num::Op::Div>(
      "nppiDiv_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_C4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 4, num::Op::Div>(
      "nppiDiv_32fc_C4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32fc_C4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 4, num::Op::Div>(
      "nppiDiv_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32fc_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::Div>(
      "nppiDiv_32s_C1IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32s_C1RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Div>(
      "nppiDiv_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Div>(
      "nppiDiv_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::Div>(
      "nppiDiv_32s_C3IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32s_C3RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::Div>(
      "nppiDiv_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 4, 3, num::Op::Div>(
      "nppiDiv_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32sc_AC4RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 4, 3, num::Op::Div>(
      "nppiDiv_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 1, 1, num::Op::Div>(
      "nppiDiv_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32sc_C1RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 1, 1, num::Op::Div>(
      "nppiDiv_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 3, 3, num::Op::Div>(
      "nppiDiv_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_32sc_C3RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 3, 3, num::Op::Div>(
      "nppiDiv_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_32sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::Div>(
      "nppiDiv_8u_AC4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_AC4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::Div>(
      "nppiDiv_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::Div>(
      "nppiDiv_8u_C1IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_C1RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::Div>(
      "nppiDiv_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::Div>(
      "nppiDiv_8u_C3IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_C3RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::Div>(
      "nppiDiv_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::Div>(
      "nppiDiv_8u_C4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_8u_C4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::Div>(
      "nppiDiv_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiDiv_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true);
}

TEST(NumGen, nppiDiv_Round_16s_AC4IRSfs_Ctx) {
  num::run_div_round2d<Npp16s, 4, 3, true>(
      "nppiDiv_Round_16s_AC4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_AC4IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_AC4RSfs_Ctx) {
  num::run_div_round2d<Npp16s, 4, 3, false>(
      "nppiDiv_Round_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_C1IRSfs_Ctx) {
  num::run_div_round2d<Npp16s, 1, 1, true>(
      "nppiDiv_Round_16s_C1IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_C1IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_C1RSfs_Ctx) {
  num::run_div_round2d<Npp16s, 1, 1, false>(
      "nppiDiv_Round_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_C3IRSfs_Ctx) {
  num::run_div_round2d<Npp16s, 3, 3, true>(
      "nppiDiv_Round_16s_C3IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_C3IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_C3RSfs_Ctx) {
  num::run_div_round2d<Npp16s, 3, 3, false>(
      "nppiDiv_Round_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_C4IRSfs_Ctx) {
  num::run_div_round2d<Npp16s, 4, 4, true>(
      "nppiDiv_Round_16s_C4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_C4IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16s_C4RSfs_Ctx) {
  num::run_div_round2d<Npp16s, 4, 4, false>(
      "nppiDiv_Round_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16s_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_AC4IRSfs_Ctx) {
  num::run_div_round2d<Npp16u, 4, 3, true>(
      "nppiDiv_Round_16u_AC4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_AC4IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_AC4RSfs_Ctx) {
  num::run_div_round2d<Npp16u, 4, 3, false>(
      "nppiDiv_Round_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_C1IRSfs_Ctx) {
  num::run_div_round2d<Npp16u, 1, 1, true>(
      "nppiDiv_Round_16u_C1IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_C1IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_C1RSfs_Ctx) {
  num::run_div_round2d<Npp16u, 1, 1, false>(
      "nppiDiv_Round_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_C3IRSfs_Ctx) {
  num::run_div_round2d<Npp16u, 3, 3, true>(
      "nppiDiv_Round_16u_C3IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_C3IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_C3RSfs_Ctx) {
  num::run_div_round2d<Npp16u, 3, 3, false>(
      "nppiDiv_Round_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_C4IRSfs_Ctx) {
  num::run_div_round2d<Npp16u, 4, 4, true>(
      "nppiDiv_Round_16u_C4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_C4IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_16u_C4RSfs_Ctx) {
  num::run_div_round2d<Npp16u, 4, 4, false>(
      "nppiDiv_Round_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_16u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_AC4IRSfs_Ctx) {
  num::run_div_round2d<Npp8u, 4, 3, true>(
      "nppiDiv_Round_8u_AC4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_AC4IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_AC4RSfs_Ctx) {
  num::run_div_round2d<Npp8u, 4, 3, false>(
      "nppiDiv_Round_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_C1IRSfs_Ctx) {
  num::run_div_round2d<Npp8u, 1, 1, true>(
      "nppiDiv_Round_8u_C1IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_C1IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_C1RSfs_Ctx) {
  num::run_div_round2d<Npp8u, 1, 1, false>(
      "nppiDiv_Round_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_C3IRSfs_Ctx) {
  num::run_div_round2d<Npp8u, 3, 3, true>(
      "nppiDiv_Round_8u_C3IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_C3IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_C3RSfs_Ctx) {
  num::run_div_round2d<Npp8u, 3, 3, false>(
      "nppiDiv_Round_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_C4IRSfs_Ctx) {
  num::run_div_round2d<Npp8u, 4, 4, true>(
      "nppiDiv_Round_8u_C4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_C4IRSfs_Ctx(a, sa, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiDiv_Round_8u_C4RSfs_Ctx) {
  num::run_div_round2d<Npp8u, 4, 4, false>(
      "nppiDiv_Round_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int mode, int param, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)mode; (void)param; (void)c; return nppiDiv_Round_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, static_cast<NppRoundMode>(mode), param, c); });
}

TEST(NumGen, nppiExp_16s_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 1, 1, num::Op::Exp>(
      "nppiExp_16s_C1IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16s_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16s_C1RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 1, 1, num::Op::Exp>(
      "nppiExp_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16s_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16s_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 3, 3, num::Op::Exp>(
      "nppiExp_16s_C3IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16s_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16s_C3RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 3, 3, num::Op::Exp>(
      "nppiExp_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16s_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 1, 1, num::Op::Exp>(
      "nppiExp_16u_C1IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16u_C1RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 1, 1, num::Op::Exp>(
      "nppiExp_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 3, 3, num::Op::Exp>(
      "nppiExp_16u_C3IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_16u_C3RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 3, 3, num::Op::Exp>(
      "nppiExp_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_16u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_32f_C1IR_Ctx) {
  num::run_unary_inplace<Npp32f, 1, 1, num::Op::Exp>(
      "nppiExp_32f_C1IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_32f_C1IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiExp_32f_C1R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 1, 1, num::Op::Exp>(
      "nppiExp_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_32f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiExp_32f_C3IR_Ctx) {
  num::run_unary_inplace<Npp32f, 3, 3, num::Op::Exp>(
      "nppiExp_32f_C3IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_32f_C3IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiExp_32f_C3R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 3, 3, num::Op::Exp>(
      "nppiExp_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_32f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiExp_8u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 1, 1, num::Op::Exp>(
      "nppiExp_8u_C1IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_8u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_8u_C1RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 1, 1, num::Op::Exp>(
      "nppiExp_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_8u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_8u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 3, 3, num::Op::Exp>(
      "nppiExp_8u_C3IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_8u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiExp_8u_C3RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 3, 3, num::Op::Exp>(
      "nppiExp_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiExp_8u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLShiftC_16u_AC4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 4, 3, num::Op::LShiftC>(
      "nppiLShiftC_16u_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_16u_AC4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 4, 3, num::Op::LShiftC>(
      "nppiLShiftC_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_16u_C1IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 1, 1, num::Op::LShiftC>(
      "nppiLShiftC_16u_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiLShiftC_16u_C1R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 1, 1, num::Op::LShiftC>(
      "nppiLShiftC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *, Npp32u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiLShiftC_16u_C3IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 3, 3, num::Op::LShiftC>(
      "nppiLShiftC_16u_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_16u_C3R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 3, 3, num::Op::LShiftC>(
      "nppiLShiftC_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_16u_C4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 4, 4, num::Op::LShiftC>(
      "nppiLShiftC_16u_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_16u_C4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 4, 4, num::Op::LShiftC>(
      "nppiLShiftC_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_16u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_32s_AC4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 4, 3, num::Op::LShiftC>(
      "nppiLShiftC_32s_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_32s_AC4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 4, 3, num::Op::LShiftC>(
      "nppiLShiftC_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_32s_C1IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 1, 1, num::Op::LShiftC>(
      "nppiLShiftC_32s_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiLShiftC_32s_C1R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 1, 1, num::Op::LShiftC>(
      "nppiLShiftC_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *, Npp32u scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiLShiftC_32s_C3IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 3, 3, num::Op::LShiftC>(
      "nppiLShiftC_32s_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_32s_C3R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 3, 3, num::Op::LShiftC>(
      "nppiLShiftC_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_32s_C4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 4, 4, num::Op::LShiftC>(
      "nppiLShiftC_32s_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_32s_C4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 4, 4, num::Op::LShiftC>(
      "nppiLShiftC_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_32s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_8u_AC4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 4, 3, num::Op::LShiftC>(
      "nppiLShiftC_8u_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_8u_AC4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 4, 3, num::Op::LShiftC>(
      "nppiLShiftC_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_8u_C1IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 1, 1, num::Op::LShiftC>(
      "nppiLShiftC_8u_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiLShiftC_8u_C1R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 1, 1, num::Op::LShiftC>(
      "nppiLShiftC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *, Npp32u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiLShiftC_8u_C3IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 3, 3, num::Op::LShiftC>(
      "nppiLShiftC_8u_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_8u_C3R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 3, 3, num::Op::LShiftC>(
      "nppiLShiftC_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_8u_C4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 4, 4, num::Op::LShiftC>(
      "nppiLShiftC_8u_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLShiftC_8u_C4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 4, 4, num::Op::LShiftC>(
      "nppiLShiftC_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLShiftC_8u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiLn_16f_C1IR_Ctx) {
  num::run_unary_inplace<Npp16f, 1, 1, num::Op::Ln>(
      "nppiLn_16f_C1IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16f_C1IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiLn_16f_C1R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 1, 1, num::Op::Ln>(
      "nppiLn_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiLn_16f_C3IR_Ctx) {
  num::run_unary_inplace<Npp16f, 3, 3, num::Op::Ln>(
      "nppiLn_16f_C3IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16f_C3IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiLn_16f_C3R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 3, 3, num::Op::Ln>(
      "nppiLn_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiLn_16s_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 1, 1, num::Op::Ln>(
      "nppiLn_16s_C1IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16s_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16s_C1RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 1, 1, num::Op::Ln>(
      "nppiLn_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16s_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16s_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 3, 3, num::Op::Ln>(
      "nppiLn_16s_C3IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16s_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16s_C3RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 3, 3, num::Op::Ln>(
      "nppiLn_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16s_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 1, 1, num::Op::Ln>(
      "nppiLn_16u_C1IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16u_C1RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 1, 1, num::Op::Ln>(
      "nppiLn_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 3, 3, num::Op::Ln>(
      "nppiLn_16u_C3IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_16u_C3RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 3, 3, num::Op::Ln>(
      "nppiLn_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_16u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_32f_C1IR_Ctx) {
  num::run_unary_inplace<Npp32f, 1, 1, num::Op::Ln>(
      "nppiLn_32f_C1IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_32f_C1IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiLn_32f_C1R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 1, 1, num::Op::Ln>(
      "nppiLn_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_32f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiLn_32f_C3IR_Ctx) {
  num::run_unary_inplace<Npp32f, 3, 3, num::Op::Ln>(
      "nppiLn_32f_C3IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_32f_C3IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiLn_32f_C3R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 3, 3, num::Op::Ln>(
      "nppiLn_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_32f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiLn_8u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 1, 1, num::Op::Ln>(
      "nppiLn_8u_C1IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_8u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_8u_C1RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 1, 1, num::Op::Ln>(
      "nppiLn_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_8u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_8u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 3, 3, num::Op::Ln>(
      "nppiLn_8u_C3IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_8u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiLn_8u_C3RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 3, 3, num::Op::Ln>(
      "nppiLn_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiLn_8u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulCScale_16u_AC4IR_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 3, true, false>(
      "nppiMulCScale_16u_AC4IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_16u_AC4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_AC4R_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 3, false, false>(
      "nppiMulCScale_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_16u_AC4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_C1IR_Ctx) {
  num::run_mul_cscale<Npp16u, 1, 1, true, false>(
      "nppiMulCScale_16u_C1IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_16u_C1IR_Ctx(cv[0], d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_C1R_Ctx) {
  num::run_mul_cscale<Npp16u, 1, 1, false, false>(
      "nppiMulCScale_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_16u_C1R_Ctx(a, sa, cv[0], d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_C3IR_Ctx) {
  num::run_mul_cscale<Npp16u, 3, 3, true, false>(
      "nppiMulCScale_16u_C3IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_16u_C3IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_C3R_Ctx) {
  num::run_mul_cscale<Npp16u, 3, 3, false, false>(
      "nppiMulCScale_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_16u_C3R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_C4IR_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 4, true, false>(
      "nppiMulCScale_16u_C4IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_16u_C4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_16u_C4R_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 4, false, false>(
      "nppiMulCScale_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_16u_C4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_AC4IR_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 3, true, false>(
      "nppiMulCScale_8u_AC4IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_8u_AC4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_AC4R_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 3, false, false>(
      "nppiMulCScale_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_8u_AC4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_C1IR_Ctx) {
  num::run_mul_cscale<Npp8u, 1, 1, true, false>(
      "nppiMulCScale_8u_C1IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_8u_C1IR_Ctx(cv[0], d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_C1R_Ctx) {
  num::run_mul_cscale<Npp8u, 1, 1, false, false>(
      "nppiMulCScale_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_8u_C1R_Ctx(a, sa, cv[0], d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_C3IR_Ctx) {
  num::run_mul_cscale<Npp8u, 3, 3, true, false>(
      "nppiMulCScale_8u_C3IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_8u_C3IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_C3R_Ctx) {
  num::run_mul_cscale<Npp8u, 3, 3, false, false>(
      "nppiMulCScale_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_8u_C3R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_C4IR_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 4, true, false>(
      "nppiMulCScale_8u_C4IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulCScale_8u_C4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulCScale_8u_C4R_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 4, false, false>(
      "nppiMulCScale_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulCScale_8u_C4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulC_16f_C1IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulC_16f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, false);
}

TEST(NumGen, nppiMulC_16f_C1R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, false);
}

TEST(NumGen, nppiMulC_16f_C3IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulC_16f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiMulC_16f_C3R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiMulC_16f_C4IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulC_16f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiMulC_16f_C4R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiMulC_16s_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 3, num::Op::MulC>(
      "nppiMulC_16s_AC4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16s_AC4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 3, num::Op::MulC>(
      "nppiMulC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 1, 1, num::Op::MulC>(
      "nppiMulC_16s_C1IRSfs_Ctx",
      [](const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_16s_C1RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 1, 1, num::Op::MulC>(
      "nppiMulC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_16s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 3, 3, num::Op::MulC>(
      "nppiMulC_16s_C3IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16s_C3RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 3, 3, num::Op::MulC>(
      "nppiMulC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16s_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 4, num::Op::MulC>(
      "nppiMulC_16s_C4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16s_C4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 4, num::Op::MulC>(
      "nppiMulC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16s_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 4, 3, num::Op::MulC>(
      "nppiMulC_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16sc_AC4RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 4, 3, num::Op::MulC>(
      "nppiMulC_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 1, 1, num::Op::MulC>(
      "nppiMulC_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_16sc_C1RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 1, 1, num::Op::MulC>(
      "nppiMulC_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_16sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 3, 3, num::Op::MulC>(
      "nppiMulC_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16sc_C3RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 3, 3, num::Op::MulC>(
      "nppiMulC_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::MulC>(
      "nppiMulC_16u_AC4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16u_AC4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::MulC>(
      "nppiMulC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::MulC>(
      "nppiMulC_16u_C1IRSfs_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_16u_C1RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::MulC>(
      "nppiMulC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_16u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::MulC>(
      "nppiMulC_16u_C3IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16u_C3RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::MulC>(
      "nppiMulC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::MulC>(
      "nppiMulC_16u_C4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_16u_C4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::MulC>(
      "nppiMulC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_16u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32f_AC4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 3, num::Op::MulC>(
      "nppiMulC_32f_AC4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiMulC_32f_AC4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 3, num::Op::MulC>(
      "nppiMulC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiMulC_32f_C1IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulC_32f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiMulC_32f_C1R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiMulC_32f_C3IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulC_32f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiMulC_32f_C3R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiMulC_32f_C4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulC_32f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiMulC_32f_C4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiMulC_32fc_AC4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 3, num::Op::MulC>(
      "nppiMulC_32fc_AC4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32fc_AC4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 3, num::Op::MulC>(
      "nppiMulC_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32fc_C1IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 1, 1, num::Op::MulC>(
      "nppiMulC_32fc_C1IR_Ctx",
      [](const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_32fc_C1R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 1, 1, num::Op::MulC>(
      "nppiMulC_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_32fc_C3IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 3, 3, num::Op::MulC>(
      "nppiMulC_32fc_C3IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32fc_C3R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 3, 3, num::Op::MulC>(
      "nppiMulC_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32fc_C4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 4, num::Op::MulC>(
      "nppiMulC_32fc_C4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32fc_C4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 4, num::Op::MulC>(
      "nppiMulC_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32fc_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::MulC>(
      "nppiMulC_32s_C1IRSfs_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_32s_C1RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::MulC>(
      "nppiMulC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_32s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::MulC>(
      "nppiMulC_32s_C3IRSfs_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32s_C3RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::MulC>(
      "nppiMulC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 4, 3, num::Op::MulC>(
      "nppiMulC_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32sc_AC4RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 4, 3, num::Op::MulC>(
      "nppiMulC_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 1, 1, num::Op::MulC>(
      "nppiMulC_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_32sc_C1RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 1, 1, num::Op::MulC>(
      "nppiMulC_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_32sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 3, 3, num::Op::MulC>(
      "nppiMulC_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_32sc_C3RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 3, 3, num::Op::MulC>(
      "nppiMulC_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_32sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_8u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::MulC>(
      "nppiMulC_8u_AC4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_8u_AC4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::MulC>(
      "nppiMulC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_8u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::MulC>(
      "nppiMulC_8u_C1IRSfs_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_8u_C1RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::MulC>(
      "nppiMulC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiMulC_8u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::MulC>(
      "nppiMulC_8u_C3IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_8u_C3RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::MulC>(
      "nppiMulC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_8u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::MulC>(
      "nppiMulC_8u_C4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulC_8u_C4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::MulC>(
      "nppiMulC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMulC_8u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiMulDeviceCScale_16u_AC4IR_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 3, true, true>(
      "nppiMulDeviceCScale_16u_AC4IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_AC4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_AC4R_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 3, false, true>(
      "nppiMulDeviceCScale_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_AC4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_C1IR_Ctx) {
  num::run_mul_cscale<Npp16u, 1, 1, true, true>(
      "nppiMulDeviceCScale_16u_C1IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_C1IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_C1R_Ctx) {
  num::run_mul_cscale<Npp16u, 1, 1, false, true>(
      "nppiMulDeviceCScale_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_C1R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_C3IR_Ctx) {
  num::run_mul_cscale<Npp16u, 3, 3, true, true>(
      "nppiMulDeviceCScale_16u_C3IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_C3IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_C3R_Ctx) {
  num::run_mul_cscale<Npp16u, 3, 3, false, true>(
      "nppiMulDeviceCScale_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_C3R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_C4IR_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 4, true, true>(
      "nppiMulDeviceCScale_16u_C4IR_Ctx",
      [](const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_C4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_16u_C4R_Ctx) {
  num::run_mul_cscale<Npp16u, 4, 4, false, true>(
      "nppiMulDeviceCScale_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *cv, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_16u_C4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_8u_AC4IR_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 3, true, true>(
      "nppiMulDeviceCScale_8u_AC4IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_AC4IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_8u_AC4R_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 3, false, true>(
      "nppiMulDeviceCScale_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_AC4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_8u_C1IR_Ctx) {
  num::run_mul_cscale<Npp8u, 1, 1, true, true>(
      "nppiMulDeviceCScale_8u_C1IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_C1IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_8u_C1R_Ctx) {
  num::run_mul_cscale<Npp8u, 1, 1, false, true>(
      "nppiMulDeviceCScale_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_C1R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_8u_C3IR_Ctx) {
  num::run_mul_cscale<Npp8u, 3, 3, true, true>(
      "nppiMulDeviceCScale_8u_C3IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_C3IR_Ctx(cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceCScale_8u_C3R_Ctx) {
  num::run_mul_cscale<Npp8u, 3, 3, false, true>(
      "nppiMulDeviceCScale_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_C3R_Ctx(a, sa, cv, d, sd, r, c); });
}

#ifndef USE_NVIDIA_NPP_TESTS
TEST(NumGen, nppiMulDeviceCScale_8u_C4IR_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 4, true, true>(
      "nppiMulDeviceCScale_8u_C4IR_Ctx",
      [](const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_C4IR_Ctx(cv, d, sd, r, c); });
}
#endif

TEST(NumGen, nppiMulDeviceCScale_8u_C4R_Ctx) {
  num::run_mul_cscale<Npp8u, 4, 4, false, true>(
      "nppiMulDeviceCScale_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *cv, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulDeviceCScale_8u_C4R_Ctx(a, sa, cv, d, sd, r, c); });
}

TEST(NumGen, nppiMulDeviceC_16f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_16f_C1IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiMulDeviceC_16f_C1R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiMulDeviceC_16f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_16f_C3IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiMulDeviceC_16f_C3R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiMulDeviceC_16f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_16f_C4IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiMulDeviceC_16f_C4R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiMulDeviceC_16s_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_16s_AC4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_AC4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_16s_C1IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_C1RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_16s_C3IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_C3RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_16s_C4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16s_C4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16s_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_16u_AC4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_AC4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_16u_C1IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_C1RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_16u_C3IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_C3RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_16u_C4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_16u_C4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_16u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_32f_AC4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_32f_AC4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_AC4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_AC4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_AC4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_32f_C1IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_C1R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_32f_C3IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_C3R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_32f_C4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32f_C4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiMulDeviceC_32s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_32s_C1IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_32s_C1RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_32s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_32s_C3IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_32s_C3RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_32s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_8u_AC4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_AC4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 3, num::Op::MulC>(
      "nppiMulDeviceC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_8u_C1IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_C1RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 1, 1, num::Op::MulC>(
      "nppiMulDeviceC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_8u_C3IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_C3RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 3, 3, num::Op::MulC>(
      "nppiMulDeviceC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_8u_C4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulDeviceC_8u_C4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 4, num::Op::MulC>(
      "nppiMulDeviceC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiMulDeviceC_8u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiMulScale_16u_AC4IR_Ctx) {
  num::run_mul_scale<Npp16u, 4, 3, true>(
      "nppiMulScale_16u_AC4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_16u_AC4IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_AC4R_Ctx) {
  num::run_mul_scale<Npp16u, 4, 3, false>(
      "nppiMulScale_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_16u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_C1IR_Ctx) {
  num::run_mul_scale<Npp16u, 1, 1, true>(
      "nppiMulScale_16u_C1IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_16u_C1IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_C1R_Ctx) {
  num::run_mul_scale<Npp16u, 1, 1, false>(
      "nppiMulScale_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_16u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_C3IR_Ctx) {
  num::run_mul_scale<Npp16u, 3, 3, true>(
      "nppiMulScale_16u_C3IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_16u_C3IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_C3R_Ctx) {
  num::run_mul_scale<Npp16u, 3, 3, false>(
      "nppiMulScale_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_16u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_C4IR_Ctx) {
  num::run_mul_scale<Npp16u, 4, 4, true>(
      "nppiMulScale_16u_C4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_16u_C4IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_16u_C4R_Ctx) {
  num::run_mul_scale<Npp16u, 4, 4, false>(
      "nppiMulScale_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_16u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_AC4IR_Ctx) {
  num::run_mul_scale<Npp8u, 4, 3, true>(
      "nppiMulScale_8u_AC4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_8u_AC4IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_AC4R_Ctx) {
  num::run_mul_scale<Npp8u, 4, 3, false>(
      "nppiMulScale_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_8u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_C1IR_Ctx) {
  num::run_mul_scale<Npp8u, 1, 1, true>(
      "nppiMulScale_8u_C1IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_8u_C1IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_C1R_Ctx) {
  num::run_mul_scale<Npp8u, 1, 1, false>(
      "nppiMulScale_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_8u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_C3IR_Ctx) {
  num::run_mul_scale<Npp8u, 3, 3, true>(
      "nppiMulScale_8u_C3IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_8u_C3IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_C3R_Ctx) {
  num::run_mul_scale<Npp8u, 3, 3, false>(
      "nppiMulScale_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_8u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_C4IR_Ctx) {
  num::run_mul_scale<Npp8u, 4, 4, true>(
      "nppiMulScale_8u_C4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)c; return nppiMulScale_8u_C4IR_Ctx(a, sa, d, sd, r, c); });
}

TEST(NumGen, nppiMulScale_8u_C4R_Ctx) {
  num::run_mul_scale<Npp8u, 4, 4, false>(
      "nppiMulScale_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, NppStreamContext c) { (void)sa; (void)sb; (void)sd; (void)r; (void)c; return nppiMulScale_8u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); });
}

TEST(NumGen, nppiMul_16f_C1IR_Ctx) {
  num::run_binary_inplace<Npp16f, 1, 1, num::Op::Mul>(
      "nppiMul_16f_C1IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiMul_16f_C1R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 1, 1, num::Op::Mul>(
      "nppiMul_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiMul_16f_C3IR_Ctx) {
  num::run_binary_inplace<Npp16f, 3, 3, num::Op::Mul>(
      "nppiMul_16f_C3IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiMul_16f_C3R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 3, 3, num::Op::Mul>(
      "nppiMul_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiMul_16f_C4IR_Ctx) {
  num::run_binary_inplace<Npp16f, 4, 4, num::Op::Mul>(
      "nppiMul_16f_C4IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiMul_16f_C4R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 4, 4, num::Op::Mul>(
      "nppiMul_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiMul_16s_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 3, num::Op::Mul>(
      "nppiMul_16s_AC4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_AC4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 3, num::Op::Mul>(
      "nppiMul_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 1, 1, num::Op::Mul>(
      "nppiMul_16s_C1IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_C1RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 1, 1, num::Op::Mul>(
      "nppiMul_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 3, 3, num::Op::Mul>(
      "nppiMul_16s_C3IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_C3RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 3, 3, num::Op::Mul>(
      "nppiMul_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 4, num::Op::Mul>(
      "nppiMul_16s_C4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16s_C4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 4, num::Op::Mul>(
      "nppiMul_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16s_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 4, 3, num::Op::Mul>(
      "nppiMul_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16sc_AC4RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 4, 3, num::Op::Mul>(
      "nppiMul_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 1, 1, num::Op::Mul>(
      "nppiMul_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16sc_C1RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 1, 1, num::Op::Mul>(
      "nppiMul_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 3, 3, num::Op::Mul>(
      "nppiMul_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16sc_C3RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 3, 3, num::Op::Mul>(
      "nppiMul_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::Mul>(
      "nppiMul_16u_AC4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_AC4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::Mul>(
      "nppiMul_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::Mul>(
      "nppiMul_16u_C1IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_C1RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::Mul>(
      "nppiMul_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::Mul>(
      "nppiMul_16u_C3IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_C3RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::Mul>(
      "nppiMul_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::Mul>(
      "nppiMul_16u_C4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_16u_C4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::Mul>(
      "nppiMul_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_16u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32f_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 3, num::Op::Mul>(
      "nppiMul_32f_AC4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_AC4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 3, num::Op::Mul>(
      "nppiMul_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_C1IR_Ctx) {
  num::run_binary_inplace<Npp32f, 1, 1, num::Op::Mul>(
      "nppiMul_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_C1R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 1, 1, num::Op::Mul>(
      "nppiMul_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_C3IR_Ctx) {
  num::run_binary_inplace<Npp32f, 3, 3, num::Op::Mul>(
      "nppiMul_32f_C3IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_C3R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 3, 3, num::Op::Mul>(
      "nppiMul_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_C4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 4, num::Op::Mul>(
      "nppiMul_32f_C4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32f_C4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 4, num::Op::Mul>(
      "nppiMul_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiMul_32fc_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 3, num::Op::Mul>(
      "nppiMul_32fc_AC4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_AC4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 3, num::Op::Mul>(
      "nppiMul_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_C1IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 1, 1, num::Op::Mul>(
      "nppiMul_32fc_C1IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_C1R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 1, 1, num::Op::Mul>(
      "nppiMul_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_C3IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 3, 3, num::Op::Mul>(
      "nppiMul_32fc_C3IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_C3R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 3, 3, num::Op::Mul>(
      "nppiMul_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_C4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 4, num::Op::Mul>(
      "nppiMul_32fc_C4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32fc_C4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 4, num::Op::Mul>(
      "nppiMul_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32fc_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::Mul>(
      "nppiMul_32s_C1IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32s_C1RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Mul>(
      "nppiMul_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Mul>(
      "nppiMul_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::Mul>(
      "nppiMul_32s_C3IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32s_C3RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::Mul>(
      "nppiMul_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 4, 3, num::Op::Mul>(
      "nppiMul_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32sc_AC4RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 4, 3, num::Op::Mul>(
      "nppiMul_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 1, 1, num::Op::Mul>(
      "nppiMul_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32sc_C1RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 1, 1, num::Op::Mul>(
      "nppiMul_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 3, 3, num::Op::Mul>(
      "nppiMul_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_32sc_C3RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 3, 3, num::Op::Mul>(
      "nppiMul_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_32sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::Mul>(
      "nppiMul_8u_AC4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_AC4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::Mul>(
      "nppiMul_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::Mul>(
      "nppiMul_8u_C1IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_C1RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::Mul>(
      "nppiMul_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::Mul>(
      "nppiMul_8u_C3IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_C3RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::Mul>(
      "nppiMul_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::Mul>(
      "nppiMul_8u_C4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiMul_8u_C4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::Mul>(
      "nppiMul_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiMul_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiNot_8u_AC4IR_Ctx) {
  num::run_unary_inplace<Npp8u, 4, 3, num::Op::Not>(
      "nppiNot_8u_AC4IR_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_AC4IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_AC4R_Ctx) {
  num::run_unary<Npp8u, Npp8u, 4, 3, num::Op::Not>(
      "nppiNot_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_AC4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_C1IR_Ctx) {
  num::run_unary_inplace<Npp8u, 1, 1, num::Op::Not>(
      "nppiNot_8u_C1IR_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_C1IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_C1R_Ctx) {
  num::run_unary<Npp8u, Npp8u, 1, 1, num::Op::Not>(
      "nppiNot_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_C3IR_Ctx) {
  num::run_unary_inplace<Npp8u, 3, 3, num::Op::Not>(
      "nppiNot_8u_C3IR_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_C3IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_C3R_Ctx) {
  num::run_unary<Npp8u, Npp8u, 3, 3, num::Op::Not>(
      "nppiNot_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_C4IR_Ctx) {
  num::run_unary_inplace<Npp8u, 4, 4, num::Op::Not>(
      "nppiNot_8u_C4IR_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_C4IR_Ctx(d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiNot_8u_C4R_Ctx) {
  num::run_unary<Npp8u, Npp8u, 4, 4, num::Op::Not>(
      "nppiNot_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiNot_8u_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0);
}

TEST(NumGen, nppiOrC_16u_AC4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::OrC>(
      "nppiOrC_16u_AC4IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_16u_AC4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::OrC>(
      "nppiOrC_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_16u_C1IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::OrC>(
      "nppiOrC_16u_C1IR_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiOrC_16u_C1R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::OrC>(
      "nppiOrC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiOrC_16u_C3IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::OrC>(
      "nppiOrC_16u_C3IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_16u_C3R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::OrC>(
      "nppiOrC_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_16u_C4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::OrC>(
      "nppiOrC_16u_C4IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_16u_C4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::OrC>(
      "nppiOrC_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_16u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_32s_AC4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 4, 3, num::Op::OrC>(
      "nppiOrC_32s_AC4IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_32s_AC4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 4, 3, num::Op::OrC>(
      "nppiOrC_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_32s_C1IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::OrC>(
      "nppiOrC_32s_C1IR_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiOrC_32s_C1R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::OrC>(
      "nppiOrC_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiOrC_32s_C3IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::OrC>(
      "nppiOrC_32s_C3IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_32s_C3R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::OrC>(
      "nppiOrC_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_32s_C4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 4, 4, num::Op::OrC>(
      "nppiOrC_32s_C4IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_32s_C4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 4, 4, num::Op::OrC>(
      "nppiOrC_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_32s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_8u_AC4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::OrC>(
      "nppiOrC_8u_AC4IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_8u_AC4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::OrC>(
      "nppiOrC_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_8u_C1IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::OrC>(
      "nppiOrC_8u_C1IR_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiOrC_8u_C1R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::OrC>(
      "nppiOrC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiOrC_8u_C3IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::OrC>(
      "nppiOrC_8u_C3IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_8u_C3R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::OrC>(
      "nppiOrC_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_8u_C4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::OrC>(
      "nppiOrC_8u_C4IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOrC_8u_C4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::OrC>(
      "nppiOrC_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOrC_8u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiOr_16u_AC4IR_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::Or>(
      "nppiOr_16u_AC4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_AC4R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::Or>(
      "nppiOr_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_C1IR_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::Or>(
      "nppiOr_16u_C1IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_C1R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::Or>(
      "nppiOr_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_C3IR_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::Or>(
      "nppiOr_16u_C3IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_C3R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::Or>(
      "nppiOr_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_C4IR_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::Or>(
      "nppiOr_16u_C4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_16u_C4R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::Or>(
      "nppiOr_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_16u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 3, num::Op::Or>(
      "nppiOr_32s_AC4IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_AC4R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 3, num::Op::Or>(
      "nppiOr_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_C1IR_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::Or>(
      "nppiOr_32s_C1IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Or>(
      "nppiOr_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_C3IR_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::Or>(
      "nppiOr_32s_C3IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_C3R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::Or>(
      "nppiOr_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_C4IR_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 4, num::Op::Or>(
      "nppiOr_32s_C4IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_32s_C4R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 4, num::Op::Or>(
      "nppiOr_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_32s_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_AC4IR_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::Or>(
      "nppiOr_8u_AC4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_AC4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::Or>(
      "nppiOr_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_C1IR_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::Or>(
      "nppiOr_8u_C1IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_C1R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::Or>(
      "nppiOr_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_C3IR_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::Or>(
      "nppiOr_8u_C3IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_C3R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::Or>(
      "nppiOr_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_C4IR_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::Or>(
      "nppiOr_8u_C4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiOr_8u_C4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::Or>(
      "nppiOr_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiOr_8u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiRShiftC_16s_AC4IR_Ctx) {
  num::run_const_inplace<Npp16s, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_16s_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16s_AC4R_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_16s_AC4R_Ctx",
      [](const Npp16s *a, int sa, const Npp32u *arr, Npp32u, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16s_C1IR_Ctx) {
  num::run_const_inplace<Npp16s, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_16s_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_16s_C1R_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_16s_C1R_Ctx",
      [](const Npp16s *a, int sa, const Npp32u *, Npp32u scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_16s_C3IR_Ctx) {
  num::run_const_inplace<Npp16s, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_16s_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16s_C3R_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_16s_C3R_Ctx",
      [](const Npp16s *a, int sa, const Npp32u *arr, Npp32u, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16s_C4IR_Ctx) {
  num::run_const_inplace<Npp16s, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_16s_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16s_C4R_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_16s_C4R_Ctx",
      [](const Npp16s *a, int sa, const Npp32u *arr, Npp32u, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16u_AC4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_16u_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16u_AC4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16u_C1IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_16u_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_16u_C1R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *, Npp32u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_16u_C3IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_16u_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16u_C3R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16u_C4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_16u_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_16u_C4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp32u *arr, Npp32u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_16u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_32s_AC4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_32s_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_32s_AC4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_32s_C1IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_32s_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_32s_C1R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *, Npp32u scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_32s_C3IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_32s_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_32s_C3R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_32s_C4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_32s_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_32s_C4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32u *arr, Npp32u, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_32s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_8s_AC4IR_Ctx) {
  num::run_const_inplace<Npp8s, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_8s_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, true, false);
}

TEST(NumGen, nppiRShiftC_8s_AC4R_Ctx) {
  num::run_const<Npp8s, Npp8s, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_8s_AC4R_Ctx",
      [](const Npp8s *a, int sa, const Npp32u *arr, Npp32u, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, true, false);
}

TEST(NumGen, nppiRShiftC_8s_C1IR_Ctx) {
  num::run_const_inplace<Npp8s, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_8s_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_8s_C1R_Ctx) {
  num::run_const<Npp8s, Npp8s, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_8s_C1R_Ctx",
      [](const Npp8s *a, int sa, const Npp32u *, Npp32u scalar, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_8s_C3IR_Ctx) {
  num::run_const_inplace<Npp8s, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_8s_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, true, false);
}

TEST(NumGen, nppiRShiftC_8s_C3R_Ctx) {
  num::run_const<Npp8s, Npp8s, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_8s_C3R_Ctx",
      [](const Npp8s *a, int sa, const Npp32u *arr, Npp32u, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, true, false);
}

TEST(NumGen, nppiRShiftC_8s_C4IR_Ctx) {
  num::run_const_inplace<Npp8s, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_8s_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, true, false);
}

TEST(NumGen, nppiRShiftC_8s_C4R_Ctx) {
  num::run_const<Npp8s, Npp8s, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_8s_C4R_Ctx",
      [](const Npp8s *a, int sa, const Npp32u *arr, Npp32u, Npp8s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, true, false);
}

TEST(NumGen, nppiRShiftC_8u_AC4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_8u_AC4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_8u_AC4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 4, 3, num::Op::RShiftC>(
      "nppiRShiftC_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_8u_C1IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_8u_C1IR_Ctx",
      [](const Npp32u *, Npp32u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_8u_C1R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 1, 1, num::Op::RShiftC>(
      "nppiRShiftC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *, Npp32u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiRShiftC_8u_C3IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_8u_C3IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_8u_C3R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 3, 3, num::Op::RShiftC>(
      "nppiRShiftC_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_8u_C4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_8u_C4IR_Ctx",
      [](const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiRShiftC_8u_C4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp32u, 4, 4, num::Op::RShiftC>(
      "nppiRShiftC_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp32u *arr, Npp32u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiRShiftC_8u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSqr_16f_C1IR_Ctx) {
  num::run_unary_inplace<Npp16f, 1, 1, num::Op::Sqr>(
      "nppiSqr_16f_C1IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16f_C1IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqr_16f_C1R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 1, 1, num::Op::Sqr>(
      "nppiSqr_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqr_16f_C3IR_Ctx) {
  num::run_unary_inplace<Npp16f, 3, 3, num::Op::Sqr>(
      "nppiSqr_16f_C3IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16f_C3IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqr_16f_C3R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 3, 3, num::Op::Sqr>(
      "nppiSqr_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqr_16f_C4IR_Ctx) {
  num::run_unary_inplace<Npp16f, 4, 4, num::Op::Sqr>(
      "nppiSqr_16f_C4IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16f_C4IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqr_16f_C4R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 4, 4, num::Op::Sqr>(
      "nppiSqr_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16f_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqr_16s_AC4IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 4, 3, num::Op::Sqr>(
      "nppiSqr_16s_AC4IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_AC4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_AC4RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 4, 3, num::Op::Sqr>(
      "nppiSqr_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_AC4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 1, 1, num::Op::Sqr>(
      "nppiSqr_16s_C1IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_C1RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 1, 1, num::Op::Sqr>(
      "nppiSqr_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 3, 3, num::Op::Sqr>(
      "nppiSqr_16s_C3IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_C3RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 3, 3, num::Op::Sqr>(
      "nppiSqr_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_C4IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 4, 4, num::Op::Sqr>(
      "nppiSqr_16s_C4IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_C4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16s_C4RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 4, 4, num::Op::Sqr>(
      "nppiSqr_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16s_C4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_AC4IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 4, 3, num::Op::Sqr>(
      "nppiSqr_16u_AC4IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_AC4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_AC4RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 4, 3, num::Op::Sqr>(
      "nppiSqr_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_AC4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 1, 1, num::Op::Sqr>(
      "nppiSqr_16u_C1IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_C1RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 1, 1, num::Op::Sqr>(
      "nppiSqr_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 3, 3, num::Op::Sqr>(
      "nppiSqr_16u_C3IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_C3RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 3, 3, num::Op::Sqr>(
      "nppiSqr_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_C4IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 4, 4, num::Op::Sqr>(
      "nppiSqr_16u_C4IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_C4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_16u_C4RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 4, 4, num::Op::Sqr>(
      "nppiSqr_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_16u_C4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_32f_AC4IR_Ctx) {
  num::run_unary_inplace<Npp32f, 4, 3, num::Op::Sqr>(
      "nppiSqr_32f_AC4IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_AC4IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_AC4R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 4, 3, num::Op::Sqr>(
      "nppiSqr_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_AC4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_C1IR_Ctx) {
  num::run_unary_inplace<Npp32f, 1, 1, num::Op::Sqr>(
      "nppiSqr_32f_C1IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_C1IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_C1R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 1, 1, num::Op::Sqr>(
      "nppiSqr_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_C3IR_Ctx) {
  num::run_unary_inplace<Npp32f, 3, 3, num::Op::Sqr>(
      "nppiSqr_32f_C3IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_C3IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_C3R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 3, 3, num::Op::Sqr>(
      "nppiSqr_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_C4IR_Ctx) {
  num::run_unary_inplace<Npp32f, 4, 4, num::Op::Sqr>(
      "nppiSqr_32f_C4IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_C4IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_32f_C4R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 4, 4, num::Op::Sqr>(
      "nppiSqr_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_32f_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqr_8u_AC4IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 4, 3, num::Op::Sqr>(
      "nppiSqr_8u_AC4IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_AC4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_AC4RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 4, 3, num::Op::Sqr>(
      "nppiSqr_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_AC4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 1, 1, num::Op::Sqr>(
      "nppiSqr_8u_C1IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_C1RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 1, 1, num::Op::Sqr>(
      "nppiSqr_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 3, 3, num::Op::Sqr>(
      "nppiSqr_8u_C3IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_C3RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 3, 3, num::Op::Sqr>(
      "nppiSqr_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_C4IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 4, 4, num::Op::Sqr>(
      "nppiSqr_8u_C4IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_C4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqr_8u_C4RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 4, 4, num::Op::Sqr>(
      "nppiSqr_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqr_8u_C4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16f_C1IR_Ctx) {
  num::run_unary_inplace<Npp16f, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_16f_C1IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16f_C1IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqrt_16f_C1R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqrt_16f_C3IR_Ctx) {
  num::run_unary_inplace<Npp16f, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_16f_C3IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16f_C3IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqrt_16f_C3R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqrt_16f_C4IR_Ctx) {
  num::run_unary_inplace<Npp16f, 4, 4, num::Op::Sqrt>(
      "nppiSqrt_16f_C4IR_Ctx",
      [](Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16f_C4IR_Ctx(d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqrt_16f_C4R_Ctx) {
  num::run_unary<Npp16f, Npp16f, 4, 4, num::Op::Sqrt>(
      "nppiSqrt_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16f_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSqrt_16s_AC4IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_16s_AC4IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16s_AC4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16s_AC4RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16s_AC4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16s_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_16s_C1IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16s_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16s_C1RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16s_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16s_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16s, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_16s_C3IRSfs_Ctx",
      [](Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16s_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16s_C3RSfs_Ctx) {
  num::run_unary<Npp16s, Npp16s, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16s_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16u_AC4IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_16u_AC4IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16u_AC4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16u_AC4RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16u_AC4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_16u_C1IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16u_C1RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp16u, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_16u_C3IRSfs_Ctx",
      [](Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_16u_C3RSfs_Ctx) {
  num::run_unary<Npp16u, Npp16u, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_16u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_32f_AC4IR_Ctx) {
  num::run_unary_inplace<Npp32f, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_32f_AC4IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_AC4IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_AC4R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_AC4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_C1IR_Ctx) {
  num::run_unary_inplace<Npp32f, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_32f_C1IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_C1IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_C1R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_C1R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_C3IR_Ctx) {
  num::run_unary_inplace<Npp32f, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_32f_C3IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_C3IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_C3R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_C3R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_C4IR_Ctx) {
  num::run_unary_inplace<Npp32f, 4, 4, num::Op::Sqrt>(
      "nppiSqrt_32f_C4IR_Ctx",
      [](Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_C4IR_Ctx(d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_32f_C4R_Ctx) {
  num::run_unary<Npp32f, Npp32f, 4, 4, num::Op::Sqrt>(
      "nppiSqrt_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_32f_C4R_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSqrt_8u_AC4IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_8u_AC4IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_8u_AC4IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_8u_AC4RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 4, 3, num::Op::Sqrt>(
      "nppiSqrt_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_8u_AC4RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_8u_C1IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_8u_C1IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_8u_C1IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_8u_C1RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 1, 1, num::Op::Sqrt>(
      "nppiSqrt_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_8u_C1RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_8u_C3IRSfs_Ctx) {
  num::run_unary_inplace<Npp8u, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_8u_C3IRSfs_Ctx",
      [](Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_8u_C3IRSfs_Ctx(d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSqrt_8u_C3RSfs_Ctx) {
  num::run_unary<Npp8u, Npp8u, 3, 3, num::Op::Sqrt>(
      "nppiSqrt_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSqrt_8u_C3RSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubC_16f_C1IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubC_16f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, false);
}

TEST(NumGen, nppiSubC_16f_C1R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *, Npp32f scalar, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.002, 0.002, false, false, false);
}

TEST(NumGen, nppiSubC_16f_C3IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubC_16f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiSubC_16f_C3R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiSubC_16f_C4IR_Ctx) {
  num::run_const_inplace<Npp16f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubC_16f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiSubC_16f_C4R_Ctx) {
  num::run_const<Npp16f, Npp16f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp32f *arr, Npp32f, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.002, 0.002, true, false, false);
}

TEST(NumGen, nppiSubC_16s_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 3, num::Op::SubC>(
      "nppiSubC_16s_AC4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16s_AC4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 3, num::Op::SubC>(
      "nppiSubC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 1, 1, num::Op::SubC>(
      "nppiSubC_16s_C1IRSfs_Ctx",
      [](const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_16s_C1RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 1, 1, num::Op::SubC>(
      "nppiSubC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *, Npp16s scalar, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_16s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 3, 3, num::Op::SubC>(
      "nppiSubC_16s_C3IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16s_C3RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 3, 3, num::Op::SubC>(
      "nppiSubC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16s_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16s, Npp16s, 4, 4, num::Op::SubC>(
      "nppiSubC_16s_C4IRSfs_Ctx",
      [](const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16s_C4RSfs_Ctx) {
  num::run_const<Npp16s, Npp16s, Npp16s, 4, 4, num::Op::SubC>(
      "nppiSubC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *arr, Npp16s, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16s_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 4, 3, num::Op::SubC>(
      "nppiSubC_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16sc_AC4RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 4, 3, num::Op::SubC>(
      "nppiSubC_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 1, 1, num::Op::SubC>(
      "nppiSubC_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_16sc_C1RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 1, 1, num::Op::SubC>(
      "nppiSubC_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *, Npp16sc scalar, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_16sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16sc, Npp16sc, 3, 3, num::Op::SubC>(
      "nppiSubC_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16sc_C3RSfs_Ctx) {
  num::run_const<Npp16sc, Npp16sc, Npp16sc, 3, 3, num::Op::SubC>(
      "nppiSubC_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *arr, Npp16sc, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::SubC>(
      "nppiSubC_16u_AC4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16u_AC4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::SubC>(
      "nppiSubC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::SubC>(
      "nppiSubC_16u_C1IRSfs_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_16u_C1RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::SubC>(
      "nppiSubC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_16u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::SubC>(
      "nppiSubC_16u_C3IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16u_C3RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::SubC>(
      "nppiSubC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::SubC>(
      "nppiSubC_16u_C4IRSfs_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_16u_C4RSfs_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::SubC>(
      "nppiSubC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_16u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32f_AC4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 3, num::Op::SubC>(
      "nppiSubC_32f_AC4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiSubC_32f_AC4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 3, num::Op::SubC>(
      "nppiSubC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiSubC_32f_C1IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubC_32f_C1IR_Ctx",
      [](const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiSubC_32f_C1R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *, Npp32f scalar, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false, false, false);
}

TEST(NumGen, nppiSubC_32f_C3IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubC_32f_C3IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiSubC_32f_C3R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiSubC_32f_C4IR_Ctx) {
  num::run_const_inplace<Npp32f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubC_32f_C4IR_Ctx",
      [](const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiSubC_32f_C4R_Ctx) {
  num::run_const<Npp32f, Npp32f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *arr, Npp32f, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32f_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 1e-05, 1e-06, true, false, false);
}

TEST(NumGen, nppiSubC_32fc_AC4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 3, num::Op::SubC>(
      "nppiSubC_32fc_AC4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32fc_AC4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 3, num::Op::SubC>(
      "nppiSubC_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32fc_C1IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 1, 1, num::Op::SubC>(
      "nppiSubC_32fc_C1IR_Ctx",
      [](const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_32fc_C1R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 1, 1, num::Op::SubC>(
      "nppiSubC_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *, Npp32fc scalar, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_32fc_C3IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 3, 3, num::Op::SubC>(
      "nppiSubC_32fc_C3IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32fc_C3R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 3, 3, num::Op::SubC>(
      "nppiSubC_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32fc_C4IR_Ctx) {
  num::run_const_inplace<Npp32fc, Npp32fc, 4, 4, num::Op::SubC>(
      "nppiSubC_32fc_C4IR_Ctx",
      [](const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32fc_C4R_Ctx) {
  num::run_const<Npp32fc, Npp32fc, Npp32fc, 4, 4, num::Op::SubC>(
      "nppiSubC_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *arr, Npp32fc, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32fc_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32s_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::SubC>(
      "nppiSubC_32s_C1IRSfs_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32s_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_32s_C1RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::SubC>(
      "nppiSubC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32s_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_32s_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::SubC>(
      "nppiSubC_32s_C3IRSfs_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32s_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32s_C3RSfs_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::SubC>(
      "nppiSubC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32s_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32sc_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 4, 3, num::Op::SubC>(
      "nppiSubC_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32sc_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32sc_AC4RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 4, 3, num::Op::SubC>(
      "nppiSubC_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32sc_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32sc_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 1, 1, num::Op::SubC>(
      "nppiSubC_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32sc_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_32sc_C1RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 1, 1, num::Op::SubC>(
      "nppiSubC_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *, Npp32sc scalar, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32sc_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_32sc_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp32sc, Npp32sc, 3, 3, num::Op::SubC>(
      "nppiSubC_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32sc_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_32sc_C3RSfs_Ctx) {
  num::run_const<Npp32sc, Npp32sc, Npp32sc, 3, 3, num::Op::SubC>(
      "nppiSubC_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *arr, Npp32sc, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_32sc_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_8u_AC4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::SubC>(
      "nppiSubC_8u_AC4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_AC4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_8u_AC4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::SubC>(
      "nppiSubC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_AC4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_8u_C1IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::SubC>(
      "nppiSubC_8u_C1IRSfs_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_C1IRSfs_Ctx(scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_8u_C1RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::SubC>(
      "nppiSubC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_C1RSfs_Ctx(a, sa, scalar, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiSubC_8u_C3IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::SubC>(
      "nppiSubC_8u_C3IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_C3IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_8u_C3RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::SubC>(
      "nppiSubC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_C3RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_8u_C4IRSfs_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::SubC>(
      "nppiSubC_8u_C4IRSfs_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_C4IRSfs_Ctx(arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubC_8u_C4RSfs_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::SubC>(
      "nppiSubC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSubC_8u_C4RSfs_Ctx(a, sa, arr, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiSubDeviceC_16f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_16f_C1IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSubDeviceC_16f_C1R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSubDeviceC_16f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_16f_C3IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSubDeviceC_16f_C3R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSubDeviceC_16f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp16f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_16f_C4IR_Ctx",
      [](Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSubDeviceC_16f_C4R_Ctx) {
  num::run_devicec<Npp16f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, Npp32f *cv, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 0.002, 0.002);
}

TEST(NumGen, nppiSubDeviceC_16s_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_16s_AC4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_AC4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_16s_C1IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_C1RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_16s_C3IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_C3RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16s, Npp16s, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_16s_C4IRSfs_Ctx",
      [](Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16s_C4RSfs_Ctx) {
  num::run_devicec<Npp16s, Npp16s, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *cv, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16s_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_16u_AC4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_AC4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_16u_C1IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_C1RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_16u_C3IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_C3RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp16u, Npp16u, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_16u_C4IRSfs_Ctx",
      [](Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_16u_C4RSfs_Ctx) {
  num::run_devicec<Npp16u, Npp16u, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *cv, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_16u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_32f_AC4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_32f_AC4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_AC4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_AC4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_AC4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_C1IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_32f_C1IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_C1IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_C1R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_C1R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_C3IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_32f_C3IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_C3IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_C3R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_C3R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_C4IR_Ctx) {
  num::run_devicec_inplace<Npp32f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_32f_C4IR_Ctx",
      [](Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_C4IR_Ctx(cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32f_C4R_Ctx) {
  num::run_devicec<Npp32f, Npp32f, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, Npp32f *cv, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32f_C4R_Ctx(a, sa, cv, d, sd, r, c); },
      {0}, 1e-05, 1e-06);
}

TEST(NumGen, nppiSubDeviceC_32s_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_32s_C1IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32s_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_32s_C1RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32s_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_32s_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp32s, Npp32s, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_32s_C3IRSfs_Ctx",
      [](Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32s_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_32s_C3RSfs_Ctx) {
  num::run_devicec<Npp32s, Npp32s, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *cv, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_32s_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_AC4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_8u_AC4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_AC4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_AC4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 3, num::Op::SubC>(
      "nppiSubDeviceC_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_AC4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_C1IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_8u_C1IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_C1IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_C1RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 1, 1, num::Op::SubC>(
      "nppiSubDeviceC_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_C1RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_C3IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_8u_C3IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_C3IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_C3RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 3, 3, num::Op::SubC>(
      "nppiSubDeviceC_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_C3RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_C4IRSfs_Ctx) {
  num::run_devicec_inplace<Npp8u, Npp8u, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_8u_C4IRSfs_Ctx",
      [](Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_C4IRSfs_Ctx(cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSubDeviceC_8u_C4RSfs_Ctx) {
  num::run_devicec<Npp8u, Npp8u, 4, 4, num::Op::SubC>(
      "nppiSubDeviceC_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *cv, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)sa; (void)sd; (void)r; (void)p; (void)c; return nppiSubDeviceC_8u_C4RSfs_Ctx(a, sa, cv, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0);
}

TEST(NumGen, nppiSub_16f_C1IR_Ctx) {
  num::run_binary_inplace<Npp16f, 1, 1, num::Op::Sub>(
      "nppiSub_16f_C1IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiSub_16f_C1R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 1, 1, num::Op::Sub>(
      "nppiSub_16f_C1R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiSub_16f_C3IR_Ctx) {
  num::run_binary_inplace<Npp16f, 3, 3, num::Op::Sub>(
      "nppiSub_16f_C3IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiSub_16f_C3R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 3, 3, num::Op::Sub>(
      "nppiSub_16f_C3R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiSub_16f_C4IR_Ctx) {
  num::run_binary_inplace<Npp16f, 4, 4, num::Op::Sub>(
      "nppiSub_16f_C4IR_Ctx",
      [](const Npp16f *a, int sa, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiSub_16f_C4R_Ctx) {
  num::run_binary<Npp16f, Npp16f, 4, 4, num::Op::Sub>(
      "nppiSub_16f_C4R_Ctx",
      [](const Npp16f *a, int sa, const Npp16f *b, int sb, Npp16f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.002, 0.002, false);
}

TEST(NumGen, nppiSub_16s_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 3, num::Op::Sub>(
      "nppiSub_16s_AC4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_AC4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 3, num::Op::Sub>(
      "nppiSub_16s_AC4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 1, 1, num::Op::Sub>(
      "nppiSub_16s_C1IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_C1RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 1, 1, num::Op::Sub>(
      "nppiSub_16s_C1RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 3, 3, num::Op::Sub>(
      "nppiSub_16s_C3IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_C3RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 3, 3, num::Op::Sub>(
      "nppiSub_16s_C3RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16s, 4, 4, num::Op::Sub>(
      "nppiSub_16s_C4IRSfs_Ctx",
      [](const Npp16s *a, int sa, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16s_C4RSfs_Ctx) {
  num::run_binary<Npp16s, Npp16s, 4, 4, num::Op::Sub>(
      "nppiSub_16s_C4RSfs_Ctx",
      [](const Npp16s *a, int sa, const Npp16s *b, int sb, Npp16s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16s_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 4, 3, num::Op::Sub>(
      "nppiSub_16sc_AC4IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16sc_AC4RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 4, 3, num::Op::Sub>(
      "nppiSub_16sc_AC4RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 1, 1, num::Op::Sub>(
      "nppiSub_16sc_C1IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16sc_C1RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 1, 1, num::Op::Sub>(
      "nppiSub_16sc_C1RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16sc, 3, 3, num::Op::Sub>(
      "nppiSub_16sc_C3IRSfs_Ctx",
      [](const Npp16sc *a, int sa, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16sc_C3RSfs_Ctx) {
  num::run_binary<Npp16sc, Npp16sc, 3, 3, num::Op::Sub>(
      "nppiSub_16sc_C3RSfs_Ctx",
      [](const Npp16sc *a, int sa, const Npp16sc *b, int sb, Npp16sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::Sub>(
      "nppiSub_16u_AC4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_AC4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::Sub>(
      "nppiSub_16u_AC4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::Sub>(
      "nppiSub_16u_C1IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_C1RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::Sub>(
      "nppiSub_16u_C1RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::Sub>(
      "nppiSub_16u_C3IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_C3RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::Sub>(
      "nppiSub_16u_C3RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::Sub>(
      "nppiSub_16u_C4IRSfs_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_16u_C4RSfs_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::Sub>(
      "nppiSub_16u_C4RSfs_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_16u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32f_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 3, num::Op::Sub>(
      "nppiSub_32f_AC4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_AC4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 3, num::Op::Sub>(
      "nppiSub_32f_AC4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_C1IR_Ctx) {
  num::run_binary_inplace<Npp32f, 1, 1, num::Op::Sub>(
      "nppiSub_32f_C1IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_C1R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 1, 1, num::Op::Sub>(
      "nppiSub_32f_C1R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_C3IR_Ctx) {
  num::run_binary_inplace<Npp32f, 3, 3, num::Op::Sub>(
      "nppiSub_32f_C3IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_C3R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 3, 3, num::Op::Sub>(
      "nppiSub_32f_C3R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_C4IR_Ctx) {
  num::run_binary_inplace<Npp32f, 4, 4, num::Op::Sub>(
      "nppiSub_32f_C4IR_Ctx",
      [](const Npp32f *a, int sa, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32f_C4R_Ctx) {
  num::run_binary<Npp32f, Npp32f, 4, 4, num::Op::Sub>(
      "nppiSub_32f_C4R_Ctx",
      [](const Npp32f *a, int sa, const Npp32f *b, int sb, Npp32f *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32f_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 1e-05, 1e-06, false);
}

TEST(NumGen, nppiSub_32fc_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 3, num::Op::Sub>(
      "nppiSub_32fc_AC4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_AC4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 3, num::Op::Sub>(
      "nppiSub_32fc_AC4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_C1IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 1, 1, num::Op::Sub>(
      "nppiSub_32fc_C1IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_C1R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 1, 1, num::Op::Sub>(
      "nppiSub_32fc_C1R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_C3IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 3, 3, num::Op::Sub>(
      "nppiSub_32fc_C3IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_C3R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 3, 3, num::Op::Sub>(
      "nppiSub_32fc_C3R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_C4IR_Ctx) {
  num::run_binary_inplace<Npp32fc, 4, 4, num::Op::Sub>(
      "nppiSub_32fc_C4IR_Ctx",
      [](const Npp32fc *a, int sa, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32fc_C4R_Ctx) {
  num::run_binary<Npp32fc, Npp32fc, 4, 4, num::Op::Sub>(
      "nppiSub_32fc_C4R_Ctx",
      [](const Npp32fc *a, int sa, const Npp32fc *b, int sb, Npp32fc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32fc_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32s_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::Sub>(
      "nppiSub_32s_C1IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32s_C1RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Sub>(
      "nppiSub_32s_C1RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Sub>(
      "nppiSub_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32s_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::Sub>(
      "nppiSub_32s_C3IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32s_C3RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::Sub>(
      "nppiSub_32s_C3RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

#ifndef USE_NVIDIA_NPP_TESTS
TEST(NumGen, nppiSub_32s_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 4, num::Op::Sub>(
      "nppiSub_32s_C4IRSfs_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}
#endif

#ifndef USE_NVIDIA_NPP_TESTS
TEST(NumGen, nppiSub_32s_C4RSfs_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 4, num::Op::Sub>(
      "nppiSub_32s_C4RSfs_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32s_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}
#endif

TEST(NumGen, nppiSub_32sc_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 4, 3, num::Op::Sub>(
      "nppiSub_32sc_AC4IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32sc_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32sc_AC4RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 4, 3, num::Op::Sub>(
      "nppiSub_32sc_AC4RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32sc_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32sc_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 1, 1, num::Op::Sub>(
      "nppiSub_32sc_C1IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32sc_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32sc_C1RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 1, 1, num::Op::Sub>(
      "nppiSub_32sc_C1RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32sc_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32sc_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp32sc, 3, 3, num::Op::Sub>(
      "nppiSub_32sc_C3IRSfs_Ctx",
      [](const Npp32sc *a, int sa, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32sc_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_32sc_C3RSfs_Ctx) {
  num::run_binary<Npp32sc, Npp32sc, 3, 3, num::Op::Sub>(
      "nppiSub_32sc_C3RSfs_Ctx",
      [](const Npp32sc *a, int sa, const Npp32sc *b, int sb, Npp32sc *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_32sc_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_AC4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::Sub>(
      "nppiSub_8u_AC4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_AC4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_AC4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::Sub>(
      "nppiSub_8u_AC4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_C1IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::Sub>(
      "nppiSub_8u_C1IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_C1IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_C1RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::Sub>(
      "nppiSub_8u_C1RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_C3IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::Sub>(
      "nppiSub_8u_C3IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_C3IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_C3RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::Sub>(
      "nppiSub_8u_C3RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_C4IRSfs_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::Sub>(
      "nppiSub_8u_C4IRSfs_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_C4IRSfs_Ctx(a, sa, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiSub_8u_C4RSfs_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::Sub>(
      "nppiSub_8u_C4RSfs_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiSub_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c); },
      {0, 1, 2}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXorC_16u_AC4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 3, num::Op::XorC>(
      "nppiXorC_16u_AC4IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_16u_AC4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 3, num::Op::XorC>(
      "nppiXorC_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_16u_C1IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 1, 1, num::Op::XorC>(
      "nppiXorC_16u_C1IR_Ctx",
      [](const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiXorC_16u_C1R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 1, 1, num::Op::XorC>(
      "nppiXorC_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *, Npp16u scalar, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiXorC_16u_C3IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 3, 3, num::Op::XorC>(
      "nppiXorC_16u_C3IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_16u_C3R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 3, 3, num::Op::XorC>(
      "nppiXorC_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_16u_C4IR_Ctx) {
  num::run_const_inplace<Npp16u, Npp16u, 4, 4, num::Op::XorC>(
      "nppiXorC_16u_C4IR_Ctx",
      [](const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_16u_C4R_Ctx) {
  num::run_const<Npp16u, Npp16u, Npp16u, 4, 4, num::Op::XorC>(
      "nppiXorC_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *arr, Npp16u, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_16u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_32s_AC4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 4, 3, num::Op::XorC>(
      "nppiXorC_32s_AC4IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_32s_AC4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 4, 3, num::Op::XorC>(
      "nppiXorC_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_32s_C1IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 1, 1, num::Op::XorC>(
      "nppiXorC_32s_C1IR_Ctx",
      [](const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiXorC_32s_C1R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 1, 1, num::Op::XorC>(
      "nppiXorC_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *, Npp32s scalar, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiXorC_32s_C3IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 3, 3, num::Op::XorC>(
      "nppiXorC_32s_C3IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_32s_C3R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 3, 3, num::Op::XorC>(
      "nppiXorC_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_32s_C4IR_Ctx) {
  num::run_const_inplace<Npp32s, Npp32s, 4, 4, num::Op::XorC>(
      "nppiXorC_32s_C4IR_Ctx",
      [](const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_32s_C4R_Ctx) {
  num::run_const<Npp32s, Npp32s, Npp32s, 4, 4, num::Op::XorC>(
      "nppiXorC_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *arr, Npp32s, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_32s_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_8u_AC4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 3, num::Op::XorC>(
      "nppiXorC_8u_AC4IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_AC4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_8u_AC4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 3, num::Op::XorC>(
      "nppiXorC_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_AC4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_8u_C1IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 1, 1, num::Op::XorC>(
      "nppiXorC_8u_C1IR_Ctx",
      [](const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_C1IR_Ctx(scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiXorC_8u_C1R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 1, 1, num::Op::XorC>(
      "nppiXorC_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *, Npp8u scalar, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_C1R_Ctx(a, sa, scalar, d, sd, r, c); },
      {0}, 0.0, 0.0, false, false, false);
}

TEST(NumGen, nppiXorC_8u_C3IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 3, 3, num::Op::XorC>(
      "nppiXorC_8u_C3IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_C3IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_8u_C3R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 3, 3, num::Op::XorC>(
      "nppiXorC_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_C3R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_8u_C4IR_Ctx) {
  num::run_const_inplace<Npp8u, Npp8u, 4, 4, num::Op::XorC>(
      "nppiXorC_8u_C4IR_Ctx",
      [](const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_C4IR_Ctx(arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXorC_8u_C4R_Ctx) {
  num::run_const<Npp8u, Npp8u, Npp8u, 4, 4, num::Op::XorC>(
      "nppiXorC_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *arr, Npp8u, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXorC_8u_C4R_Ctx(a, sa, arr, d, sd, r, c); },
      {0}, 0.0, 0.0, true, false, false);
}

TEST(NumGen, nppiXor_16u_AC4IR_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 3, num::Op::Xor>(
      "nppiXor_16u_AC4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_AC4R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 3, num::Op::Xor>(
      "nppiXor_16u_AC4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_C1IR_Ctx) {
  num::run_binary_inplace<Npp16u, 1, 1, num::Op::Xor>(
      "nppiXor_16u_C1IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_C1R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 1, 1, num::Op::Xor>(
      "nppiXor_16u_C1R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_C3IR_Ctx) {
  num::run_binary_inplace<Npp16u, 3, 3, num::Op::Xor>(
      "nppiXor_16u_C3IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_C3R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 3, 3, num::Op::Xor>(
      "nppiXor_16u_C3R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_C4IR_Ctx) {
  num::run_binary_inplace<Npp16u, 4, 4, num::Op::Xor>(
      "nppiXor_16u_C4IR_Ctx",
      [](const Npp16u *a, int sa, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_16u_C4R_Ctx) {
  num::run_binary<Npp16u, Npp16u, 4, 4, num::Op::Xor>(
      "nppiXor_16u_C4R_Ctx",
      [](const Npp16u *a, int sa, const Npp16u *b, int sb, Npp16u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_16u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_AC4IR_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 3, num::Op::Xor>(
      "nppiXor_32s_AC4IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_AC4R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 3, num::Op::Xor>(
      "nppiXor_32s_AC4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_C1IR_Ctx) {
  num::run_binary_inplace<Npp32s, 1, 1, num::Op::Xor>(
      "nppiXor_32s_C1IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_C1R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 1, 1, num::Op::Xor>(
      "nppiXor_32s_C1R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_C3IR_Ctx) {
  num::run_binary_inplace<Npp32s, 3, 3, num::Op::Xor>(
      "nppiXor_32s_C3IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_C3R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 3, 3, num::Op::Xor>(
      "nppiXor_32s_C3R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_C4IR_Ctx) {
  num::run_binary_inplace<Npp32s, 4, 4, num::Op::Xor>(
      "nppiXor_32s_C4IR_Ctx",
      [](const Npp32s *a, int sa, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_32s_C4R_Ctx) {
  num::run_binary<Npp32s, Npp32s, 4, 4, num::Op::Xor>(
      "nppiXor_32s_C4R_Ctx",
      [](const Npp32s *a, int sa, const Npp32s *b, int sb, Npp32s *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_32s_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_AC4IR_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 3, num::Op::Xor>(
      "nppiXor_8u_AC4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_AC4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_AC4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 3, num::Op::Xor>(
      "nppiXor_8u_AC4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_AC4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_C1IR_Ctx) {
  num::run_binary_inplace<Npp8u, 1, 1, num::Op::Xor>(
      "nppiXor_8u_C1IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_C1IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_C1R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 1, 1, num::Op::Xor>(
      "nppiXor_8u_C1R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_C1R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_C3IR_Ctx) {
  num::run_binary_inplace<Npp8u, 3, 3, num::Op::Xor>(
      "nppiXor_8u_C3IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_C3IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_C3R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 3, 3, num::Op::Xor>(
      "nppiXor_8u_C3R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_C3R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_C4IR_Ctx) {
  num::run_binary_inplace<Npp8u, 4, 4, num::Op::Xor>(
      "nppiXor_8u_C4IR_Ctx",
      [](const Npp8u *a, int sa, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_C4IR_Ctx(a, sa, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}

TEST(NumGen, nppiXor_8u_C4R_Ctx) {
  num::run_binary<Npp8u, Npp8u, 4, 4, num::Op::Xor>(
      "nppiXor_8u_C4R_Ctx",
      [](const Npp8u *a, int sa, const Npp8u *b, int sb, Npp8u *d, int sd, NppiSize r, int p, NppStreamContext c) { (void)p; return nppiXor_8u_C4R_Ctx(a, sa, b, sb, d, sd, r, c); },
      {0}, 0.0, 0.0, false);
}
