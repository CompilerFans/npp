// Numerical conformance: nppiAdd/Sub/Mul/Div elementwise families.
//
// Expected values come from the existing CPU reference helpers in
// test/unit/nppi/nppi_arithmetic_operations/nppi_arithmetic_test_framework.h
// (namespace npp_arithmetic_test::expect).  There are no status assertions:
// a non-success DUT status skips that subcase.  The same tests build against
// NVIDIA NPP 13.4 in the reference implementation (the gate that validates the references)
// and against the in-tree MPP library locally.

#include "num_test_utils.h"

#include "nppi_arithmetic_test_framework.h"

using npp_arithmetic_test::expect::add;
using npp_arithmetic_test::expect::add_sfs;
using npp_arithmetic_test::expect::div;
// NPP 13.4 reference semantics, calibrated by the reference gate:
//   Add:  dst = rint_half_even((a + b) / 2^sf)
//   Sub:  dst = rint_half_even((b - a) / 2^sf)   (src2 - src1)
//   Mul:  dst = rint_half_even((a * b) / 2^sf)   (64-bit intermediate)
//   Div:  dst = rint_half_even((b / a) / 2^sf), src1 == 0 -> type maximum
// Integer results saturate to the destination type.
template <typename T> T npp_sat_ll(long long v) {
  using L = std::numeric_limits<T>;
  if constexpr (std::is_unsigned_v<T>) {
    if (v < 0) return T(0);
    if ((unsigned long long)v > (unsigned long long)L::max()) return L::max();
    return (T)v;
  } else {
    if (v < (long long)L::lowest()) return L::lowest();
    if (v > (long long)L::max()) return L::max();
    return (T)v;
  }
}

// exact / 2^sf with round-half-even, then saturate.
template <typename T> T npp_scale(long long exact, int sf) {
  if (sf <= 0) return npp_sat_ll<T>(exact);
  const double q = (double)exact / (double)(1ll << sf);
  return npp_sat_ll<T>((long long)std::rint(q));
}

template <typename T> T add_npp(T a, T b, int sf) {
  if constexpr (std::is_floating_point_v<T>) return (a + b) / (T)(1ll << sf);
  return npp_scale<T>((long long)a + (long long)b, sf);
}
template <typename T> T sub_npp(T a, T b, int sf) {
  if constexpr (std::is_floating_point_v<T>) return (b - a) / (T)(1ll << sf);
  return npp_scale<T>((long long)b - (long long)a, sf);
}
template <typename T> T mul_npp(T a, T b, int sf) {
  if constexpr (std::is_floating_point_v<T>) return (a * b) / (T)(1ll << sf);
  return npp_scale<T>((long long)a * (long long)b, sf);
}
template <typename T> T div_npp(T a, T b, int sf) {
  if constexpr (std::is_floating_point_v<T>) {
    return (b / a) / (T)(1ll << sf);  // IEEE inf/nan semantics, like NPP
  } else {
    if (a == 0) {
      if constexpr (std::is_signed_v<T>) return b < T(0) ? std::numeric_limits<T>::lowest(): std::numeric_limits<T>::max();
      else return std::numeric_limits<T>::max();
    }
    const double q = ((double)b / (double)a) / (double)(1ll << sf);
    return npp_sat_ll<T>((long long)std::rint(q));
  }
}
template <typename T> T div_float_npp(T a, T b) { return b / a; }

using npp_arithmetic_test::expect::add;
using npp_arithmetic_test::expect::add_sfs;
using npp_arithmetic_test::expect::div;
// Binary div-with-scale reference: rounded quotient (calibrated against NPP
// by the reference gate), then >> scale factor with round-half-up, saturating.
template <typename T> T div_sfs_ref(T a, T b, int sf) {
  if (b == 0) return T(0);
  long long q = ((long long)a + (long long)b / 2) / (long long)b;
  if (sf > 0) q = (q + (1ll << (sf - 1))) >> sf;
  if constexpr (std::is_same_v<T, Npp8u>) return (T)std::max(0ll, std::min(255ll, q));
  else if constexpr (std::is_same_v<T, Npp16u>) return (T)std::max(0ll, std::min(65535ll, q));
  else if constexpr (std::is_same_v<T, Npp16s>) return (T)std::max(-32768ll, std::min(32767ll, q));
  else return (T)q;
}
using npp_arithmetic_test::expect::mul;
using npp_arithmetic_test::expect::mul_sfs;
using npp_arithmetic_test::expect::sub;
using npp_arithmetic_test::expect::sub_sfs;

// One TEST per API: the macro supplies the _Ctx call expression and the CPU
// reference expression.  Commas inside the expressions are protected by the
// surrounding parentheses.
#define NUM_BIN(SUITE, NAME, S, D, C, A, RTOL, ATOL, PARAMS, CALL, REF)                     \
  TEST(SUITE, NAME) {                                                                       \
    num::run_binary<S, D, C, A>(                                                            \
        #NAME,                                                                              \
        [](const S *a, int sa, const S *b, int sb, D *d, int sd, NppiSize r, int p,          \
           NppStreamContext c) { (void)p; return CALL; },                                             \
        [](S a, S b, int p) { return REF; }, (RTOL), (ATOL), PARAMS);                      \
  }

// ------------------------------- Add --------------------------------------
NUM_BIN(NumArithmetic, Add_8u_C1RSfs, Npp8u, Npp8u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiAdd_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (add_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Add_8u_C3RSfs, Npp8u, Npp8u, 3, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiAdd_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (add_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Add_8u_C4RSfs, Npp8u, Npp8u, 4, 4, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiAdd_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (add_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Add_8u_AC4RSfs, Npp8u, Npp8u, 4, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiAdd_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (add_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Add_16u_C1RSfs, Npp16u, Npp16u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiAdd_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (add_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Add_16s_C1RSfs, Npp16s, Npp16s, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiAdd_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (add_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Add_32f_C1R, Npp32f, Npp32f, 1, 1, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiAdd_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c)), (add<decltype(a)>(a, b)))
NUM_BIN(NumArithmetic, Add_32f_C3R, Npp32f, Npp32f, 3, 3, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiAdd_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c)), (add<decltype(a)>(a, b)))

// ------------------------------- Sub --------------------------------------
NUM_BIN(NumArithmetic, Sub_8u_C1RSfs, Npp8u, Npp8u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiSub_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (sub_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Sub_8u_C3RSfs, Npp8u, Npp8u, 3, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiSub_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (sub_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Sub_8u_C4RSfs, Npp8u, Npp8u, 4, 4, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiSub_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (sub_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Sub_8u_AC4RSfs, Npp8u, Npp8u, 4, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiSub_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (sub_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Sub_16u_C1RSfs, Npp16u, Npp16u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiSub_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (sub_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Sub_16s_C1RSfs, Npp16s, Npp16s, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiSub_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (sub_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Sub_32f_C1R, Npp32f, Npp32f, 1, 1, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiSub_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c)), (sub_npp<decltype(a)>(a, b, 0)))
NUM_BIN(NumArithmetic, Sub_32f_C3R, Npp32f, Npp32f, 3, 3, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiSub_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c)), (sub_npp<decltype(a)>(a, b, 0)))

// ------------------------------- Mul --------------------------------------
NUM_BIN(NumArithmetic, Mul_8u_C1RSfs, Npp8u, Npp8u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiMul_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (mul_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Mul_8u_C3RSfs, Npp8u, Npp8u, 3, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiMul_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (mul_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Mul_8u_C4RSfs, Npp8u, Npp8u, 4, 4, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiMul_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (mul_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Mul_8u_AC4RSfs, Npp8u, Npp8u, 4, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiMul_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (mul_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Mul_16u_C1RSfs, Npp16u, Npp16u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiMul_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (mul_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Mul_16s_C1RSfs, Npp16s, Npp16s, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiMul_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (mul_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Mul_32f_C1R, Npp32f, Npp32f, 1, 1, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiMul_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c)), (mul<decltype(a)>(a, b)))
NUM_BIN(NumArithmetic, Mul_32f_C3R, Npp32f, Npp32f, 3, 3, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiMul_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c)), (mul<decltype(a)>(a, b)))

// ------------------------------- Div --------------------------------------
NUM_BIN(NumArithmetic, Div_8u_C1RSfs, Npp8u, Npp8u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiDiv_8u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (div_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Div_8u_C3RSfs, Npp8u, Npp8u, 3, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiDiv_8u_C3RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (div_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Div_8u_C4RSfs, Npp8u, Npp8u, 4, 4, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiDiv_8u_C4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (div_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Div_8u_AC4RSfs, Npp8u, Npp8u, 4, 3, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiDiv_8u_AC4RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (div_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Div_16u_C1RSfs, Npp16u, Npp16u, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiDiv_16u_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (div_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Div_16s_C1RSfs, Npp16s, Npp16s, 1, 1, 0.0, 0.0, (std::vector<int>{0, 1, 2}),
        (nppiDiv_16s_C1RSfs_Ctx(a, sa, b, sb, d, sd, r, p, c)), (div_npp<decltype(a)>(a, b, p)))
NUM_BIN(NumArithmetic, Div_32f_C1R, Npp32f, Npp32f, 1, 1, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiDiv_32f_C1R_Ctx(a, sa, b, sb, d, sd, r, c)), (div_float_npp<decltype(a)>(a, b)))
NUM_BIN(NumArithmetic, Div_32f_C3R, Npp32f, Npp32f, 3, 3, 1e-6, 1e-6, (std::vector<int>{0}),
        (nppiDiv_32f_C3R_Ctx(a, sa, b, sb, d, sd, r, c)), (div_float_npp<decltype(a)>(a, b)))
