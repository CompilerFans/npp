// Differential tests for the npps statistics families (packages p2.4/p2.4b):
// DotProd, Average/Maximum(Relative)Error, Mean/StdDev/MeanStdDev, Min/Max
// (+Indx/Abs), Norm/NormDiff (L1/L2/Inf/L2Sqr), Sum, MinEvery/MaxEvery,
// CountInRange, ZeroCrossing and the GetBufferSize contracts.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppsStatisticsDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppsStatisticsDiff.*'
//
// Case ids live under stats/... so they cannot collide with the pilot
// NppsSumDiff suite.  A Ctx case and its plain entry point share one id: the
// 5090 records the Ctx golden once, and the MPP build verifies both entry
// points against it.  Integer outputs are compared exactly, complex-integer
// scalars as their two component integers, floats with a per-kind tolerance.
//
// Contracts pinned by these cases (measured NVIDIA NPP 13.4 on RTX 5090):
//   * nLength == 0  -> NPP_SIZE_ERROR (no kernel launched);
//   * negative nLength (huge size_t on the reference build) -> the launch
//     fails gracefully with NPP_CUDA_KERNEL_EXECUTION_ERROR;
//   * null src/dst/buffer pointers -> NPP_NULL_POINTER_ERROR;
//   * StdDev/MeanStdDev with nLength == 1 -> NPP_SIZE_ERROR (sample stdev);
//   * GetBufferSize never validates nLength, only a null hpBufferSize.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

// ---- type traits -----------------------------------------------------------

template <typename T> struct is_cplx : std::false_type {};
template <> struct is_cplx<Npp16sc> : std::true_type {};
template <> struct is_cplx<Npp32sc> : std::true_type {};
template <> struct is_cplx<Npp64sc> : std::true_type {};
template <> struct is_cplx<Npp32fc> : std::true_type {};
template <> struct is_cplx<Npp64fc> : std::true_type {};

template <typename T> constexpr ValueKind kind_of() {
  if constexpr (std::is_same_v<T, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<T, Npp8s>) return ValueKind::I8;
  else if constexpr (std::is_same_v<T, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<T, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<T, Npp32u>) return ValueKind::U32;
  else if constexpr (std::is_same_v<T, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<T, Npp64s>) return ValueKind::I64;
  else if constexpr (std::is_same_v<T, Npp32f>) return ValueKind::F32;
  else if constexpr (std::is_same_v<T, Npp64f>) return ValueKind::F64;
  else if constexpr (std::is_same_v<T, Npp16sc>) return ValueKind::I16;  // 2 x i16
  else if constexpr (std::is_same_v<T, Npp32sc>) return ValueKind::I32;  // 2 x i32
  else if constexpr (std::is_same_v<T, Npp64sc>) return ValueKind::I64;  // 2 x i64
  else if constexpr (std::is_same_v<T, Npp32fc>) return ValueKind::CF32;
  else return ValueKind::CF64;
}

// complex-integer scalars are recorded as their two component integers.
template <typename T> constexpr size_t out_cnt() {
  if constexpr (std::is_same_v<T, Npp16sc> || std::is_same_v<T, Npp32sc> || std::is_same_v<T, Npp64sc>) {
    return 2;
  } else {
    return 1;
  }
}

template <typename T> void tol_of(double &rtol, double &atol) {
  if constexpr (std::is_same_v<T, Npp32f> || std::is_same_v<T, Npp32fc>) {
    rtol = 1e-4;
    atol = 1e-4;
  } else if constexpr (std::is_same_v<T, Npp64f> || std::is_same_v<T, Npp64fc>) {
    rtol = 1e-8;
    atol = 1e-8;
  } else {
    rtol = 0;
    atol = 0;
  }
}

// ---- deterministic bounded inputs ------------------------------------------

template <typename T> std::vector<T> gen(size_t n, uint64_t seed) {
  std::vector<T> v(n);
  Rng r(seed);
  for (auto &x : v) {
    if constexpr (is_cplx<T>::value) {
      using C = decltype(x.re);
      if constexpr (std::is_same_v<C, Npp32f>) {
        x.re = r.unit_float() * 4.0f - 2.0f;
        x.im = r.unit_float() * 4.0f - 2.0f;
      } else if constexpr (std::is_same_v<C, Npp64f>) {
        x.re = r.unit_double() * 4.0 - 2.0;
        x.im = r.unit_double() * 4.0 - 2.0;
      } else if constexpr (std::is_same_v<C, Npp16s>) {
        x.re = r.integer<Npp16s>(-1000, 1000);
        x.im = r.integer<Npp16s>(-1000, 1000);
      } else if constexpr (std::is_same_v<C, Npp32s>) {
        x.re = r.integer<Npp32s>(-100000, 100000);
        x.im = r.integer<Npp32s>(-100000, 100000);
      } else {
        x.re = r.integer<Npp64s>(-1000000, 1000000);
        x.im = r.integer<Npp64s>(-1000000, 1000000);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      x = r.unit_float() * 4.0f - 2.0f;
    } else if constexpr (std::is_same_v<T, Npp64f>) {
      x = r.unit_double() * 4.0 - 2.0;
    } else if constexpr (std::is_same_v<T, Npp8u>) {
      x = r.integer<Npp8u>(0, 255);
    } else if constexpr (std::is_same_v<T, Npp8s>) {
      x = r.integer<Npp8s>(-100, 100);
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      x = r.integer<Npp16u>(0, 60000);
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      x = r.integer<Npp16s>(-1000, 1000);
    } else if constexpr (std::is_same_v<T, Npp32u>) {
      x = r.integer<Npp32u>(0, 1000000);
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      x = r.integer<Npp32s>(-100000, 100000);
    } else {
      x = r.integer<Npp64s>(-1000000, 1000000);
    }
  }
  return v;
}

// ---- scratch + status-only helper ------------------------------------------

using CtxSizeFn = NppStatus (*)(NppSignalLength, SIZE_TYPE *, NppStreamContext);
using PlainSizeFn = NppStatus (*)(NppSignalLength, SIZE_TYPE *);

SIZE_TYPE query_size(CtxSizeFn fn, NppSignalLength n, const NppStreamContext &ctx) {
  SIZE_TYPE bytes = 0;
  EXPECT_EQ(fn(n, &bytes, ctx), NPP_NO_ERROR);
  return bytes;
}

#if NPP_DIFF_HAS_NONCTX
SIZE_TYPE query_size_nc(PlainSizeFn fn, NppSignalLength n) {
  SIZE_TYPE bytes = 0;
  EXPECT_EQ(fn(n, &bytes), NPP_NO_ERROR);
  return bytes;
}
#endif

// Status-only case (error paths, buffer-size contracts).
[[maybe_unused]] void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

// ---- shape helpers ---------------------------------------------------------
// Each helper runs one golden case.  nLength is the input length; error paths
// are separate hand-written tests at the end of the file.

// (src, n) -> O                                [Sum/Mean/StdDev/Norm/Min/Max/Abs]
template <typename T, typename O>
void one_src_case(const char *fname, const std::string &id, std::vector<T> src,
                  NppStatus (*f)(const T *, NppSignalLength, O *, Npp8u *, NppStreamContext), CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<O> od(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, od.get(), buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T, typename O>
void one_src_case_nc(const char *fname, const std::string &id, std::vector<T> src,
                     NppStatus (*f)(const T *, NppSignalLength, O *, Npp8u *), PlainSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<O> od(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, od.get(), buf.get());
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}
#endif

// (src, n, sf) -> O                           [Sum/Mean/StdDev/Norm Sfs]
template <typename T, typename O>
void one_src_sfs_case(const char *fname, const std::string &id, std::vector<T> src, int sf,
                      NppStatus (*f)(const T *, NppSignalLength, O *, int, Npp8u *, NppStreamContext),
                      CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<O> od(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, od.get(), sf, buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T, typename O>
void one_src_sfs_case_nc(const char *fname, const std::string &id, std::vector<T> src, int sf,
                         NppStatus (*f)(const T *, NppSignalLength, O *, int, Npp8u *), PlainSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<O> od(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, od.get(), sf, buf.get());
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}
#endif

// (a, b, n) -> O                              [DotProd/NormDiff plain, error metrics]
template <typename A, typename B, typename O>
void two_src_case(const char *fname, const std::string &id, std::vector<A> a, std::vector<B> b,
                  NppStatus (*f)(const A *, const B *, NppSignalLength, O *, Npp8u *, NppStreamContext),
                  CtxSizeFn szfn, double ortol = -1.0, double oatol = -1.0) {
  ASSERT_EQ(a.size(), b.size());
  const NppSignalLength n = (NppSignalLength)a.size();
  DeviceMemory<A> ad(a.size());
  ad.copyFromHost(a);
  DeviceMemory<B> bd(b.size());
  bd.copyFromHost(b);
  DeviceMemory<O> od(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(ad.get(), bd.get(), n, od.get(), buf.get(), ctx);
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(A));
  c.input(b.data(), b.size() * sizeof(B));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  if (ortol >= 0) rtol = ortol;
  if (oatol >= 0) atol = oatol;
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename A, typename B, typename O>
void two_src_case_nc(const char *fname, const std::string &id, std::vector<A> a, std::vector<B> b,
                     NppStatus (*f)(const A *, const B *, NppSignalLength, O *, Npp8u *), PlainSizeFn szfn) {
  ASSERT_EQ(a.size(), b.size());
  const NppSignalLength n = (NppSignalLength)a.size();
  DeviceMemory<A> ad(a.size());
  ad.copyFromHost(a);
  DeviceMemory<B> bd(b.size());
  bd.copyFromHost(b);
  DeviceMemory<O> od(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(ad.get(), bd.get(), n, od.get(), buf.get());
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(A));
  c.input(b.data(), b.size() * sizeof(B));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}
#endif

// (a, b, n, sf) -> O                          [DotProd/NormDiff Sfs]
template <typename A, typename B, typename O>
void two_src_sfs_case(const char *fname, const std::string &id, std::vector<A> a, std::vector<B> b, int sf,
                      NppStatus (*f)(const A *, const B *, NppSignalLength, O *, int, Npp8u *, NppStreamContext),
                      CtxSizeFn szfn) {
  ASSERT_EQ(a.size(), b.size());
  const NppSignalLength n = (NppSignalLength)a.size();
  DeviceMemory<A> ad(a.size());
  ad.copyFromHost(a);
  DeviceMemory<B> bd(b.size());
  bd.copyFromHost(b);
  DeviceMemory<O> od(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(ad.get(), bd.get(), n, od.get(), sf, buf.get(), ctx);
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(A));
  c.input(b.data(), b.size() * sizeof(B));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename A, typename B, typename O>
void two_src_sfs_case_nc(const char *fname, const std::string &id, std::vector<A> a, std::vector<B> b, int sf,
                         NppStatus (*f)(const A *, const B *, NppSignalLength, O *, int, Npp8u *),
                         PlainSizeFn szfn) {
  ASSERT_EQ(a.size(), b.size());
  const NppSignalLength n = (NppSignalLength)a.size();
  DeviceMemory<A> ad(a.size());
  ad.copyFromHost(a);
  DeviceMemory<B> bd(b.size());
  bd.copyFromHost(b);
  DeviceMemory<O> od(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(ad.get(), bd.get(), n, od.get(), sf, buf.get());
  Case c(id, fname);
  c.input(a.data(), a.size() * sizeof(A));
  c.input(b.data(), b.size() * sizeof(B));
  c.status(st);
  double rtol, atol;
  tol_of<O>(rtol, atol);
  c.device_output(od.get(), out_cnt<O>(), kind_of<O>(), rtol, atol);
  c.finish();
}
#endif

// (src, n) -> (mean, stddev)                [MeanStdDev]
template <typename T, typename OM, typename OS>
void meanstd_case(const char *fname, const std::string &id, std::vector<T> src,
                  NppStatus (*f)(const T *, NppSignalLength, OM *, OS *, Npp8u *, NppStreamContext),
                  CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<OM> md(1);
  DeviceMemory<OS> dd(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, md.get(), dd.get(), buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<OM>(rtol, atol);
  c.device_output(md.get(), out_cnt<OM>(), kind_of<OM>(), rtol, atol);
  tol_of<OS>(rtol, atol);
  c.device_output(dd.get(), out_cnt<OS>(), kind_of<OS>(), rtol, atol);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T, typename OM, typename OS>
void meanstd_case_nc(const char *fname, const std::string &id, std::vector<T> src,
                     NppStatus (*f)(const T *, NppSignalLength, OM *, OS *, Npp8u *), PlainSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<OM> md(1);
  DeviceMemory<OS> dd(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, md.get(), dd.get(), buf.get());
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<OM>(rtol, atol);
  c.device_output(md.get(), out_cnt<OM>(), kind_of<OM>(), rtol, atol);
  tol_of<OS>(rtol, atol);
  c.device_output(dd.get(), out_cnt<OS>(), kind_of<OS>(), rtol, atol);
  c.finish();
}
#endif

// (src, n, sf) -> (mean, stddev)             [MeanStdDev Sfs]
template <typename T, typename OM, typename OS>
void meanstd_sfs_case(const char *fname, const std::string &id, std::vector<T> src, int sf,
                      NppStatus (*f)(const T *, NppSignalLength, OM *, OS *, int, Npp8u *, NppStreamContext),
                      CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<OM> md(1);
  DeviceMemory<OS> dd(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, md.get(), dd.get(), sf, buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<OM>(rtol, atol);
  c.device_output(md.get(), out_cnt<OM>(), kind_of<OM>(), rtol, atol);
  tol_of<OS>(rtol, atol);
  c.device_output(dd.get(), out_cnt<OS>(), kind_of<OS>(), rtol, atol);
  c.finish();
}

// (src, n) -> (min, max)                    [MinMax]
template <typename T>
void minmax_case(const char *fname, const std::string &id, std::vector<T> src,
                 NppStatus (*f)(const T *, NppSignalLength, T *, T *, Npp8u *, NppStreamContext),
                 CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> mind(1);
  DeviceMemory<T> maxd(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, mind.get(), maxd.get(), buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(mind.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(maxd.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T>
void minmax_case_nc(const char *fname, const std::string &id, std::vector<T> src,
                    NppStatus (*f)(const T *, NppSignalLength, T *, T *, Npp8u *), PlainSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> mind(1);
  DeviceMemory<T> maxd(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, mind.get(), maxd.get(), buf.get());
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(mind.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(maxd.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.finish();
}
#endif

// (src, n) -> (min, minIndx, max, maxIndx)  [MinMaxIndx]
template <typename T>
void minmax_indx_case(const char *fname, const std::string &id, std::vector<T> src,
                      NppStatus (*f)(const T *, NppSignalLength, T *, int *, T *, int *, Npp8u *, NppStreamContext),
                      CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> mind(1);
  DeviceMemory<int> mindx(1);
  DeviceMemory<T> maxd(1);
  DeviceMemory<int> maxdx(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, mind.get(), mindx.get(), maxd.get(), maxdx.get(), buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(mind.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(mindx.get(), 1, ValueKind::I32, 0, 0);
  c.device_output(maxd.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(maxdx.get(), 1, ValueKind::I32, 0, 0);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T>
void minmax_indx_case_nc(const char *fname, const std::string &id, std::vector<T> src,
                         NppStatus (*f)(const T *, NppSignalLength, T *, int *, T *, int *, Npp8u *),
                         PlainSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> mind(1);
  DeviceMemory<int> mindx(1);
  DeviceMemory<T> maxd(1);
  DeviceMemory<int> maxdx(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, mind.get(), mindx.get(), maxd.get(), maxdx.get(), buf.get());
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(mind.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(mindx.get(), 1, ValueKind::I32, 0, 0);
  c.device_output(maxd.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(maxdx.get(), 1, ValueKind::I32, 0, 0);
  c.finish();
}
#endif

// (src, n) -> (val, indx)                   [Min/MaxIndx, Min/MaxAbsIndx]
template <typename T>
void val_indx_case(const char *fname, const std::string &id, std::vector<T> src,
                   NppStatus (*f)(const T *, NppSignalLength, T *, int *, Npp8u *, NppStreamContext),
                   CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> vd(1);
  DeviceMemory<int> idx(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, vd.get(), idx.get(), buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(vd.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(idx.get(), 1, ValueKind::I32, 0, 0);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
template <typename T>
void val_indx_case_nc(const char *fname, const std::string &id, std::vector<T> src,
                      NppStatus (*f)(const T *, NppSignalLength, T *, int *, Npp8u *), PlainSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> vd(1);
  DeviceMemory<int> idx(1);
  SIZE_TYPE bytes = query_size_nc(szfn, n);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, vd.get(), idx.get(), buf.get());
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(vd.get(), out_cnt<T>(), kind_of<T>(), rtol, atol);
  c.device_output(idx.get(), 1, ValueKind::I32, 0, 0);
  c.finish();
}
#endif

// (src, initial dst) -> dst                  [MinEvery/MaxEvery, in place]
template <typename T>
void every_case(const char *fname, const std::string &id, std::vector<T> src, std::vector<T> dst0,
                NppStatus (*f)(const T *, T *, NppSignalLength, NppStreamContext)) {
  ASSERT_EQ(src.size(), dst0.size());
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<T> dd(dst0.size());
  dd.copyFromHost(dst0);
  NppStreamContext ctx = current_context();
  clear_cuda_error();
  NppStatus st = f(sd.get(), dd.get(), n, ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.input(dst0.data(), dst0.size() * sizeof(T));
  c.status(st);
  double rtol, atol;
  tol_of<T>(rtol, atol);
  c.device_output(dd.get(), dst0.size(), kind_of<T>(), rtol, atol);
  c.finish();
}

// (src, lo, hi) -> count                     [CountInRange]
inline void count_case(const char *fname, const std::string &id, std::vector<Npp32s> src,
                       Npp32s lo, Npp32s hi,
                       NppStatus (*f)(const Npp32s *, NppSignalLength, int *, Npp32s, Npp32s,
                                      Npp8u *, NppStreamContext),
                       CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<Npp32s> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<int> cd(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, cd.get(), lo, hi, buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(Npp32s));
  c.status(st);
  c.device_output(cd.get(), 1, ValueKind::I32, 0, 0);
  c.finish();
}

// (src, mode) -> value                       [ZeroCrossing]
template <typename T>
void zc_case(const char *fname, const std::string &id, std::vector<T> src, NppsZCType mode,
             NppStatus (*f)(const T *, NppSignalLength, Npp32f *, NppsZCType, Npp8u *, NppStreamContext),
             CtxSizeFn szfn) {
  const NppSignalLength n = (NppSignalLength)src.size();
  DeviceMemory<T> sd(src.size());
  sd.copyFromHost(src);
  DeviceMemory<Npp32f> vd(1);
  NppStreamContext ctx = current_context();
  SIZE_TYPE bytes = query_size(szfn, n, ctx);
  DeviceMemory<Npp8u> buf(bytes > 0 ? (size_t)bytes : 1);
  clear_cuda_error();
  NppStatus st = f(sd.get(), n, vd.get(), mode, buf.get(), ctx);
  Case c(id, fname);
  c.input(src.data(), src.size() * sizeof(T));
  c.status(st);
  c.device_output(vd.get(), 1, ValueKind::F32, 1e-4, 1e-4);
  c.finish();
}

} // namespace

class NppsStatisticsDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===========================================================================
// DotProd: (a, b) -> dot product (complex = plain, unconjugated)
// ===========================================================================

// plain 32f: entry-point twins + length variation
TEST_F(NppsStatisticsDiff, DotProd_32f_n1) {
  two_src_case<Npp32f, Npp32f, Npp32f>("nppsDotProd_32f", "stats/DotProd/32f/n1",
      gen<Npp32f>(1, 0x101), gen<Npp32f>(1, 0x102),
      &nppsDotProd_32f_Ctx, &nppsDotProdGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f_n2) {
  two_src_case<Npp32f, Npp32f, Npp32f>("nppsDotProd_32f", "stats/DotProd/32f/n2",
      gen<Npp32f>(2, 0x103), gen<Npp32f>(2, 0x104),
      &nppsDotProd_32f_Ctx, &nppsDotProdGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f_n17) {
  two_src_case<Npp32f, Npp32f, Npp32f>("nppsDotProd_32f", "stats/DotProd/32f/n17",
      gen<Npp32f>(17, 0x105), gen<Npp32f>(17, 0x106),
      &nppsDotProd_32f_Ctx, &nppsDotProdGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f_n257) {
  two_src_case<Npp32f, Npp32f, Npp32f>("nppsDotProd_32f", "stats/DotProd/32f/n257",
      gen<Npp32f>(257, 0x107), gen<Npp32f>(257, 0x108),
      &nppsDotProd_32f_Ctx, &nppsDotProdGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f_n1000) {
  two_src_case<Npp32f, Npp32f, Npp32f>("nppsDotProd_32f", "stats/DotProd/32f/n1000",
      gen<Npp32f>(1000, 0x109), gen<Npp32f>(1000, 0x10A),
      &nppsDotProd_32f_Ctx, &nppsDotProdGetBufferSize_32f_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, DotProd_32f_Plain_n257) {
  two_src_case_nc<Npp32f, Npp32f, Npp32f>("nppsDotProd_32f", "stats/DotProd/32f/n257",
      gen<Npp32f>(257, 0x107), gen<Npp32f>(257, 0x108),
      &nppsDotProd_32f, &nppsDotProdGetBufferSize_32f);
}
#endif

TEST_F(NppsStatisticsDiff, DotProd_64f_n257) {
  two_src_case<Npp64f, Npp64f, Npp64f>("nppsDotProd_64f", "stats/DotProd/64f/n257",
      gen<Npp64f>(257, 0x10B), gen<Npp64f>(257, 0x10C),
      &nppsDotProd_64f_Ctx, &nppsDotProdGetBufferSize_64f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s32f_n257) {
  two_src_case<Npp16s, Npp16s, Npp32f>("nppsDotProd_16s32f", "stats/DotProd/16s32f/n257",
      gen<Npp16s>(257, 0x10D), gen<Npp16s>(257, 0x10E),
      &nppsDotProd_16s32f_Ctx, &nppsDotProdGetBufferSize_16s32f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s64s_n257) {
  two_src_case<Npp16s, Npp16s, Npp64s>("nppsDotProd_16s64s", "stats/DotProd/16s64s/n257",
      gen<Npp16s>(257, 0x10F), gen<Npp16s>(257, 0x110),
      &nppsDotProd_16s64s_Ctx, &nppsDotProdGetBufferSize_16s64s_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s_Sfs_n257) {
  two_src_sfs_case<Npp16s, Npp16s, Npp16s>("nppsDotProd_16s_Sfs", "stats/DotProd/16s_Sfs/n257",
      gen<Npp16s>(257, 0x111), gen<Npp16s>(257, 0x112), 4,
      &nppsDotProd_16s_Sfs_Ctx, &nppsDotProdGetBufferSize_16s_Sfs_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, DotProd_16s_Sfs_Plain_n257) {
  two_src_sfs_case_nc<Npp16s, Npp16s, Npp16s>("nppsDotProd_16s_Sfs", "stats/DotProd/16s_Sfs/n257",
      gen<Npp16s>(257, 0x111), gen<Npp16s>(257, 0x112), 4,
      &nppsDotProd_16s_Sfs, &nppsDotProdGetBufferSize_16s_Sfs);
}
#endif

TEST_F(NppsStatisticsDiff, DotProd_16s32s_Sfs_n257) {
  two_src_sfs_case<Npp16s, Npp16s, Npp32s>("nppsDotProd_16s32s_Sfs", "stats/DotProd/16s32s_Sfs/n257",
      gen<Npp16s>(257, 0x113), gen<Npp16s>(257, 0x114), 2,
      &nppsDotProd_16s32s_Sfs_Ctx, &nppsDotProdGetBufferSize_16s32s_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s32s32s_Sfs_n257) {
  two_src_sfs_case<Npp16s, Npp32s, Npp32s>("nppsDotProd_16s32s32s_Sfs", "stats/DotProd/16s32s32s_Sfs/n257",
      gen<Npp16s>(257, 0x115), gen<Npp32s>(257, 0x116), 5,
      &nppsDotProd_16s32s32s_Sfs_Ctx, &nppsDotProdGetBufferSize_16s32s32s_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32s_Sfs_n257) {
  two_src_sfs_case<Npp32s, Npp32s, Npp32s>("nppsDotProd_32s_Sfs", "stats/DotProd/32s_Sfs/n257",
      gen<Npp32s>(257, 0x117), gen<Npp32s>(257, 0x118), 3,
      &nppsDotProd_32s_Sfs_Ctx, &nppsDotProdGetBufferSize_32s_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16sc_Sfs_n257) {
  two_src_sfs_case<Npp16sc, Npp16sc, Npp16sc>("nppsDotProd_16sc_Sfs", "stats/DotProd/16sc_Sfs/n257",
      gen<Npp16sc>(257, 0x11B), gen<Npp16sc>(257, 0x11C), 2,
      &nppsDotProd_16sc_Sfs_Ctx, &nppsDotProdGetBufferSize_16sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16sc64sc_n257) {
  two_src_case<Npp16sc, Npp16sc, Npp64sc>("nppsDotProd_16sc64sc", "stats/DotProd/16sc64sc/n257",
      gen<Npp16sc>(257, 0x11D), gen<Npp16sc>(257, 0x11E),
      &nppsDotProd_16sc64sc_Ctx, &nppsDotProdGetBufferSize_16sc64sc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16sc32sc_Sfs_n257) {
  two_src_sfs_case<Npp16sc, Npp16sc, Npp32sc>("nppsDotProd_16sc32sc_Sfs", "stats/DotProd/16sc32sc_Sfs/n257",
      gen<Npp16sc>(257, 0x11F), gen<Npp16sc>(257, 0x120), 1,
      &nppsDotProd_16sc32sc_Sfs_Ctx, &nppsDotProdGetBufferSize_16sc32sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32fc_n257) {
  two_src_case<Npp32fc, Npp32fc, Npp32fc>("nppsDotProd_32fc", "stats/DotProd/32fc/n257",
      gen<Npp32fc>(257, 0x121), gen<Npp32fc>(257, 0x122),
      &nppsDotProd_32fc_Ctx, &nppsDotProdGetBufferSize_32fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32fc64fc_n257) {
  two_src_case<Npp32fc, Npp32fc, Npp64fc>("nppsDotProd_32fc64fc", "stats/DotProd/32fc64fc/n257",
      gen<Npp32fc>(257, 0x123), gen<Npp32fc>(257, 0x124),
      &nppsDotProd_32fc64fc_Ctx, &nppsDotProdGetBufferSize_32fc64fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_64fc_n257) {
  two_src_case<Npp64fc, Npp64fc, Npp64fc>("nppsDotProd_64fc", "stats/DotProd/64fc/n257",
      gen<Npp64fc>(257, 0x125), gen<Npp64fc>(257, 0x126),
      &nppsDotProd_64fc_Ctx, &nppsDotProdGetBufferSize_64fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32sc_Sfs_n257) {
  two_src_sfs_case<Npp32sc, Npp32sc, Npp32sc>("nppsDotProd_32sc_Sfs", "stats/DotProd/32sc_Sfs/n257",
      gen<Npp32sc>(257, 0x127), gen<Npp32sc>(257, 0x128), 3,
      &nppsDotProd_32sc_Sfs_Ctx, &nppsDotProdGetBufferSize_32sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32s32sc_Sfs_n257) {
  two_src_sfs_case<Npp32s, Npp32sc, Npp32sc>("nppsDotProd_32s32sc_Sfs", "stats/DotProd/32s32sc_Sfs/n257",
      gen<Npp32s>(257, 0x129), gen<Npp32sc>(257, 0x12A), 4,
      &nppsDotProd_32s32sc_Sfs_Ctx, &nppsDotProdGetBufferSize_32s32sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s16sc_Sfs_n257) {
  two_src_sfs_case<Npp16s, Npp16sc, Npp16sc>("nppsDotProd_16s16sc_Sfs", "stats/DotProd/16s16sc_Sfs/n257",
      gen<Npp16s>(257, 0x12B), gen<Npp16sc>(257, 0x12C), 1,
      &nppsDotProd_16s16sc_Sfs_Ctx, &nppsDotProdGetBufferSize_16s16sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s16sc32sc_Sfs_n257) {
  two_src_sfs_case<Npp16s, Npp16sc, Npp32sc>("nppsDotProd_16s16sc32sc_Sfs", "stats/DotProd/16s16sc32sc_Sfs/n257",
      gen<Npp16s>(257, 0x12D), gen<Npp16sc>(257, 0x12E), 2,
      &nppsDotProd_16s16sc32sc_Sfs_Ctx, &nppsDotProdGetBufferSize_16s16sc32sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s16sc32fc_n257) {
  two_src_case<Npp16s, Npp16sc, Npp32fc>("nppsDotProd_16s16sc32fc", "stats/DotProd/16s16sc32fc/n257",
      gen<Npp16s>(257, 0x131), gen<Npp16sc>(257, 0x132),
      &nppsDotProd_16s16sc32fc_Ctx, &nppsDotProdGetBufferSize_16s16sc32fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16s16sc64sc_n257) {
  two_src_case<Npp16s, Npp16sc, Npp64sc>("nppsDotProd_16s16sc64sc", "stats/DotProd/16s16sc64sc/n257",
      gen<Npp16s>(257, 0x135), gen<Npp16sc>(257, 0x136),
      &nppsDotProd_16s16sc64sc_Ctx, &nppsDotProdGetBufferSize_16s16sc64sc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f32fc_n257) {
  two_src_case<Npp32f, Npp32fc, Npp32fc>("nppsDotProd_32f32fc", "stats/DotProd/32f32fc/n257",
      gen<Npp32f>(257, 0x137), gen<Npp32fc>(257, 0x138),
      &nppsDotProd_32f32fc_Ctx, &nppsDotProdGetBufferSize_32f32fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f32fc64fc_n257) {
  two_src_case<Npp32f, Npp32fc, Npp64fc>("nppsDotProd_32f32fc64fc", "stats/DotProd/32f32fc64fc/n257",
      gen<Npp32f>(257, 0x139), gen<Npp32fc>(257, 0x13A),
      &nppsDotProd_32f32fc64fc_Ctx, &nppsDotProdGetBufferSize_32f32fc64fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_32f64f_n257) {
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsDotProd_32f64f", "stats/DotProd/32f64f/n257",
      gen<Npp32f>(257, 0x13B), gen<Npp32f>(257, 0x13C),
      &nppsDotProd_32f64f_Ctx, &nppsDotProdGetBufferSize_32f64f_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_16sc32fc_n257) {
  two_src_case<Npp16sc, Npp16sc, Npp32fc>("nppsDotProd_16sc32fc", "stats/DotProd/16sc32fc/n257",
      gen<Npp16sc>(257, 0x13F), gen<Npp16sc>(257, 0x140),
      &nppsDotProd_16sc32fc_Ctx, &nppsDotProdGetBufferSize_16sc32fc_Ctx);
}

TEST_F(NppsStatisticsDiff, DotProd_64f64fc_n257) {
  two_src_case<Npp64f, Npp64fc, Npp64fc>("nppsDotProd_64f64fc", "stats/DotProd/64f64fc/n257",
      gen<Npp64f>(257, 0x145), gen<Npp64fc>(257, 0x146),
      &nppsDotProd_64f64fc_Ctx, &nppsDotProdGetBufferSize_64f64fc_Ctx);
}

// ===========================================================================
// Average/Maximum (Relative)Error: (a, b) -> Npp64f
// ===========================================================================

#define STATS_ERR_METRIC_CASE(KIND, DT)                                                   \
  TEST_F(NppsStatisticsDiff, KIND##_##DT##_n257) {                                        \
    two_src_case<Npp##DT, Npp##DT, Npp64f>("npps" #KIND "_" #DT,                          \
        "stats/" #KIND "/" #DT "/n257", gen<Npp##DT>(257, 0x201), gen<Npp##DT>(257, 0x202), \
        &npps##KIND##_##DT##_Ctx, &npps##KIND##GetBufferSize_##DT##_Ctx);                 \
  }

STATS_ERR_METRIC_CASE(AverageError, 8s)
STATS_ERR_METRIC_CASE(AverageError, 8u)
STATS_ERR_METRIC_CASE(AverageError, 16s)
STATS_ERR_METRIC_CASE(AverageError, 16sc)
STATS_ERR_METRIC_CASE(AverageError, 16u)
STATS_ERR_METRIC_CASE(AverageError, 32f)
STATS_ERR_METRIC_CASE(AverageError, 32fc)
STATS_ERR_METRIC_CASE(AverageError, 32s)
STATS_ERR_METRIC_CASE(AverageError, 32sc)
STATS_ERR_METRIC_CASE(AverageError, 32u)
STATS_ERR_METRIC_CASE(AverageError, 64f)
STATS_ERR_METRIC_CASE(AverageError, 64fc)
STATS_ERR_METRIC_CASE(AverageError, 64s)
STATS_ERR_METRIC_CASE(AverageError, 64sc)
STATS_ERR_METRIC_CASE(AverageRelativeError, 8s)
STATS_ERR_METRIC_CASE(AverageRelativeError, 8u)
STATS_ERR_METRIC_CASE(AverageRelativeError, 16s)
STATS_ERR_METRIC_CASE(AverageRelativeError, 16sc)
STATS_ERR_METRIC_CASE(AverageRelativeError, 16u)
STATS_ERR_METRIC_CASE(AverageRelativeError, 32f)
STATS_ERR_METRIC_CASE(AverageRelativeError, 32fc)
STATS_ERR_METRIC_CASE(AverageRelativeError, 32s)
STATS_ERR_METRIC_CASE(AverageRelativeError, 32sc)
STATS_ERR_METRIC_CASE(AverageRelativeError, 32u)
STATS_ERR_METRIC_CASE(AverageRelativeError, 64f)
STATS_ERR_METRIC_CASE(AverageRelativeError, 64fc)
STATS_ERR_METRIC_CASE(AverageRelativeError, 64s)
STATS_ERR_METRIC_CASE(AverageRelativeError, 64sc)
STATS_ERR_METRIC_CASE(MaximumError, 8s)
STATS_ERR_METRIC_CASE(MaximumError, 8u)
STATS_ERR_METRIC_CASE(MaximumError, 16s)
STATS_ERR_METRIC_CASE(MaximumError, 16sc)
STATS_ERR_METRIC_CASE(MaximumError, 16u)
STATS_ERR_METRIC_CASE(MaximumError, 32f)
STATS_ERR_METRIC_CASE(MaximumError, 32fc)
STATS_ERR_METRIC_CASE(MaximumError, 32s)
STATS_ERR_METRIC_CASE(MaximumError, 32sc)
STATS_ERR_METRIC_CASE(MaximumError, 32u)
STATS_ERR_METRIC_CASE(MaximumError, 64f)
STATS_ERR_METRIC_CASE(MaximumError, 64fc)
STATS_ERR_METRIC_CASE(MaximumError, 64s)
STATS_ERR_METRIC_CASE(MaximumError, 64sc)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 8s)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 8u)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 16s)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 16sc)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 16u)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 32f)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 32fc)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 32s)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 32sc)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 32u)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 64f)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 64fc)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 64s)
STATS_ERR_METRIC_CASE(MaximumRelativeError, 64sc)
#undef STATS_ERR_METRIC_CASE

// length variation
TEST_F(NppsStatisticsDiff, AverageError_32f_n1) {
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsAverageError_32f", "stats/AverageError/32f/n1",
      gen<Npp32f>(1, 0x203), gen<Npp32f>(1, 0x204),
      &nppsAverageError_32f_Ctx, &nppsAverageErrorGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, AverageError_32f_n17) {
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsAverageError_32f", "stats/AverageError/32f/n17",
      gen<Npp32f>(17, 0x205), gen<Npp32f>(17, 0x206),
      &nppsAverageError_32f_Ctx, &nppsAverageErrorGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, AverageError_32f_n1000) {
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsAverageError_32f", "stats/AverageError/32f/n1000",
      gen<Npp32f>(1000, 0x207), gen<Npp32f>(1000, 0x208),
      &nppsAverageError_32f_Ctx, &nppsAverageErrorGetBufferSize_32f_Ctx);
}

// crafted probes: relative error is |a-b|/max(|a|,|b|), 0 when both are 0
TEST_F(NppsStatisticsDiff, MaximumRelativeError_32f_Identical) {
  std::vector<Npp32f> a = gen<Npp32f>(257, 0x209);
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsMaximumRelativeError_32f",
      "stats/MaximumRelativeError/32f/identical", a, a,
      &nppsMaximumRelativeError_32f_Ctx, &nppsMaximumRelativeErrorGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MaximumRelativeError_32f_BothZero) {
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsMaximumRelativeError_32f",
      "stats/MaximumRelativeError/32f/both_zero", std::vector<Npp32f>(257, 0.0f),
      std::vector<Npp32f>(257, 0.0f),
      &nppsMaximumRelativeError_32f_Ctx, &nppsMaximumRelativeErrorGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MaximumRelativeError_32f_ZeroVsValue) {
  std::vector<Npp32f> b = gen<Npp32f>(64, 0x20A);
  two_src_case<Npp32f, Npp32f, Npp64f>("nppsMaximumRelativeError_32f",
      "stats/MaximumRelativeError/32f/zero_vs_value", std::vector<Npp32f>(64, 0.0f), b,
      &nppsMaximumRelativeError_32f_Ctx, &nppsMaximumRelativeErrorGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MaximumError_8u_Identical) {
  std::vector<Npp8u> a = gen<Npp8u>(257, 0x20B);
  two_src_case<Npp8u, Npp8u, Npp64f>("nppsMaximumError_8u", "stats/MaximumError/8u/identical",
      a, a, &nppsMaximumError_8u_Ctx, &nppsMaximumErrorGetBufferSize_8u_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, AverageError_16s_Plain_n257) {
  two_src_case_nc<Npp16s, Npp16s, Npp64f>("nppsAverageError_16s", "stats/AverageError/16s/n257",
      gen<Npp16s>(257, 0x201), gen<Npp16s>(257, 0x202),
      &nppsAverageError_16s, &nppsAverageErrorGetBufferSize_16s);
}
#endif

// ===========================================================================
// Sum: (src) -> sum
// ===========================================================================

#define STATS_SUM_CASE(VARIANT, T, O, SF)                                                 \
  TEST_F(NppsStatisticsDiff, Sum_##VARIANT##_n257) {                                      \
    one_src_sfs_case<T, O>("nppsSum_" #VARIANT, "stats/Sum/" #VARIANT "/n257",            \
        gen<T>(257, 0x301), SF, &nppsSum_##VARIANT##_Ctx,                                 \
        &nppsSumGetBufferSize_##VARIANT##_Ctx);                                           \
  }

#define STATS_SUM_PLAIN_CASE(VARIANT, T, O)                                               \
  TEST_F(NppsStatisticsDiff, Sum_##VARIANT##_n257) {                                      \
    one_src_case<T, O>("nppsSum_" #VARIANT, "stats/Sum/" #VARIANT "/n257",                \
        gen<T>(257, 0x301), &nppsSum_##VARIANT##_Ctx,                                     \
        &nppsSumGetBufferSize_##VARIANT##_Ctx);                                           \
  }

STATS_SUM_PLAIN_CASE(32f, Npp32f, Npp32f)
STATS_SUM_PLAIN_CASE(32fc, Npp32fc, Npp32fc)
STATS_SUM_PLAIN_CASE(64f, Npp64f, Npp64f)
STATS_SUM_PLAIN_CASE(64fc, Npp64fc, Npp64fc)
STATS_SUM_CASE(16s_Sfs, Npp16s, Npp16s, 4)
STATS_SUM_CASE(16s32s_Sfs, Npp16s, Npp32s, 2)
STATS_SUM_CASE(16sc_Sfs, Npp16sc, Npp16sc, 1)
STATS_SUM_CASE(16sc32sc_Sfs, Npp16sc, Npp32sc, 2)
STATS_SUM_CASE(32s_Sfs, Npp32s, Npp32s, 3)
#undef STATS_SUM_CASE
#undef STATS_SUM_PLAIN_CASE

TEST_F(NppsStatisticsDiff, Sum_32f_n1) {
  one_src_case<Npp32f, Npp32f>("nppsSum_32f", "stats/Sum/32f/n1",
      gen<Npp32f>(1, 0x302), &nppsSum_32f_Ctx, &nppsSumGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, Sum_32f_n1000) {
  one_src_case<Npp32f, Npp32f>("nppsSum_32f", "stats/Sum/32f/n1000",
      gen<Npp32f>(1000, 0x303), &nppsSum_32f_Ctx, &nppsSumGetBufferSize_32f_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, Sum_32f_Plain_n257) {
  one_src_case_nc<Npp32f, Npp32f>("nppsSum_32f", "stats/Sum/32f/n257",
      gen<Npp32f>(257, 0x301), &nppsSum_32f, &nppsSumGetBufferSize_32f);
}
#endif

// ===========================================================================
// Mean: (src) -> mean
// ===========================================================================

TEST_F(NppsStatisticsDiff, Mean_32f_n257) {
  one_src_case<Npp32f, Npp32f>("nppsMean_32f", "stats/Mean/32f/n257",
      gen<Npp32f>(257, 0x311), &nppsMean_32f_Ctx, &nppsMeanGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_32f_n1) {
  one_src_case<Npp32f, Npp32f>("nppsMean_32f", "stats/Mean/32f/n1",
      gen<Npp32f>(1, 0x312), &nppsMean_32f_Ctx, &nppsMeanGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_32fc_n257) {
  one_src_case<Npp32fc, Npp32fc>("nppsMean_32fc", "stats/Mean/32fc/n257",
      gen<Npp32fc>(257, 0x313), &nppsMean_32fc_Ctx, &nppsMeanGetBufferSize_32fc_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_64f_n257) {
  one_src_case<Npp64f, Npp64f>("nppsMean_64f", "stats/Mean/64f/n257",
      gen<Npp64f>(257, 0x314), &nppsMean_64f_Ctx, &nppsMeanGetBufferSize_64f_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_64fc_n257) {
  one_src_case<Npp64fc, Npp64fc>("nppsMean_64fc", "stats/Mean/64fc/n257",
      gen<Npp64fc>(257, 0x315), &nppsMean_64fc_Ctx, &nppsMeanGetBufferSize_64fc_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_16s_Sfs_n257) {
  one_src_sfs_case<Npp16s, Npp16s>("nppsMean_16s_Sfs", "stats/Mean/16s_Sfs/n257",
      gen<Npp16s>(257, 0x316), 2, &nppsMean_16s_Sfs_Ctx, &nppsMeanGetBufferSize_16s_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_16sc_Sfs_n257) {
  one_src_sfs_case<Npp16sc, Npp16sc>("nppsMean_16sc_Sfs", "stats/Mean/16sc_Sfs/n257",
      gen<Npp16sc>(257, 0x317), 1, &nppsMean_16sc_Sfs_Ctx, &nppsMeanGetBufferSize_16sc_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, Mean_32s_Sfs_n257) {
  one_src_sfs_case<Npp32s, Npp32s>("nppsMean_32s_Sfs", "stats/Mean/32s_Sfs/n257",
      gen<Npp32s>(257, 0x318), 2, &nppsMean_32s_Sfs_Ctx, &nppsMeanGetBufferSize_32s_Sfs_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, Mean_32f_Plain_n257) {
  one_src_case_nc<Npp32f, Npp32f>("nppsMean_32f", "stats/Mean/32f/n257",
      gen<Npp32f>(257, 0x311), &nppsMean_32f, &nppsMeanGetBufferSize_32f);
}
#endif

// ===========================================================================
// StdDev: (src) -> sample standard deviation (n-1 normalization)
// ===========================================================================

TEST_F(NppsStatisticsDiff, StdDev_32f_n257) {
  one_src_case<Npp32f, Npp32f>("nppsStdDev_32f", "stats/StdDev/32f/n257",
      gen<Npp32f>(257, 0x321), &nppsStdDev_32f_Ctx, &nppsStdDevGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, StdDev_32f_n2) {
  one_src_case<Npp32f, Npp32f>("nppsStdDev_32f", "stats/StdDev/32f/n2",
      gen<Npp32f>(2, 0x322), &nppsStdDev_32f_Ctx, &nppsStdDevGetBufferSize_32f_Ctx);
}

// n == 1 -> NPP_SIZE_ERROR (sample stdev needs n >= 2)
TEST_F(NppsStatisticsDiff, StdDev_32f_n1) {
  one_src_case<Npp32f, Npp32f>("nppsStdDev_32f", "stats/StdDev/32f/n1",
      gen<Npp32f>(1, 0x323), &nppsStdDev_32f_Ctx, &nppsStdDevGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, StdDev_64f_n257) {
  one_src_case<Npp64f, Npp64f>("nppsStdDev_64f", "stats/StdDev/64f/n257",
      gen<Npp64f>(257, 0x324), &nppsStdDev_64f_Ctx, &nppsStdDevGetBufferSize_64f_Ctx);
}

TEST_F(NppsStatisticsDiff, StdDev_16s_Sfs_n257) {
  one_src_sfs_case<Npp16s, Npp16s>("nppsStdDev_16s_Sfs", "stats/StdDev/16s_Sfs/n257",
      gen<Npp16s>(257, 0x325), 3, &nppsStdDev_16s_Sfs_Ctx, &nppsStdDevGetBufferSize_16s_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, StdDev_16s32s_Sfs_n257) {
  one_src_sfs_case<Npp16s, Npp32s>("nppsStdDev_16s32s_Sfs", "stats/StdDev/16s32s_Sfs/n257",
      gen<Npp16s>(257, 0x326), 2, &nppsStdDev_16s32s_Sfs_Ctx, &nppsStdDevGetBufferSize_16s32s_Sfs_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, StdDev_32f_Plain_n257) {
  one_src_case_nc<Npp32f, Npp32f>("nppsStdDev_32f", "stats/StdDev/32f/n257",
      gen<Npp32f>(257, 0x321), &nppsStdDev_32f, &nppsStdDevGetBufferSize_32f);
}
#endif

// ===========================================================================
// MeanStdDev: (src) -> (mean, sample stddev)
// ===========================================================================

TEST_F(NppsStatisticsDiff, MeanStdDev_32f_n257) {
  meanstd_case<Npp32f, Npp32f, Npp32f>("nppsMeanStdDev_32f", "stats/MeanStdDev/32f/n257",
      gen<Npp32f>(257, 0x331), &nppsMeanStdDev_32f_Ctx, &nppsMeanStdDevGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MeanStdDev_32f_n2) {
  meanstd_case<Npp32f, Npp32f, Npp32f>("nppsMeanStdDev_32f", "stats/MeanStdDev/32f/n2",
      gen<Npp32f>(2, 0x332), &nppsMeanStdDev_32f_Ctx, &nppsMeanStdDevGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MeanStdDev_32f_n1) {
  meanstd_case<Npp32f, Npp32f, Npp32f>("nppsMeanStdDev_32f", "stats/MeanStdDev/32f/n1",
      gen<Npp32f>(1, 0x333), &nppsMeanStdDev_32f_Ctx, &nppsMeanStdDevGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MeanStdDev_64f_n257) {
  meanstd_case<Npp64f, Npp64f, Npp64f>("nppsMeanStdDev_64f", "stats/MeanStdDev/64f/n257",
      gen<Npp64f>(257, 0x334), &nppsMeanStdDev_64f_Ctx, &nppsMeanStdDevGetBufferSize_64f_Ctx);
}

TEST_F(NppsStatisticsDiff, MeanStdDev_16s_Sfs_n257) {
  meanstd_sfs_case<Npp16s, Npp16s, Npp16s>("nppsMeanStdDev_16s_Sfs", "stats/MeanStdDev/16s_Sfs/n257",
      gen<Npp16s>(257, 0x335), 3, &nppsMeanStdDev_16s_Sfs_Ctx,
      &nppsMeanStdDevGetBufferSize_16s_Sfs_Ctx);
}

TEST_F(NppsStatisticsDiff, MeanStdDev_16s32s_Sfs_n257) {
  meanstd_sfs_case<Npp16s, Npp32s, Npp32s>("nppsMeanStdDev_16s32s_Sfs", "stats/MeanStdDev/16s32s_Sfs/n257",
      gen<Npp16s>(257, 0x336), 2, &nppsMeanStdDev_16s32s_Sfs_Ctx,
      &nppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, MeanStdDev_32f_Plain_n257) {
  meanstd_case_nc<Npp32f, Npp32f, Npp32f>("nppsMeanStdDev_32f", "stats/MeanStdDev/32f/n257",
      gen<Npp32f>(257, 0x331), &nppsMeanStdDev_32f, &nppsMeanStdDevGetBufferSize_32f);
}
#endif

// ===========================================================================
// Norm / NormDiff: (src) -> O; (src1, src2) -> O
// ===========================================================================

#define STATS_NORM_PLAIN(NAME, T, O, GBHS)                                                \
  TEST_F(NppsStatisticsDiff, Norm_##NAME##_n257) {                                        \
    one_src_case<T, O>("nppsNorm_" #NAME, "stats/Norm/" #NAME "/n257",                    \
        gen<T>(257, 0x501), &nppsNorm_##NAME##_Ctx, &GBHS);                               \
  }
#define STATS_NORM_SFS(NAME, T, O, GBHS)                                                  \
  TEST_F(NppsStatisticsDiff, Norm_##NAME##_n257) {                                        \
    one_src_sfs_case<T, O>("nppsNorm_" #NAME, "stats/Norm/" #NAME "/n257",                \
        gen<T>(257, 0x501), 2, &nppsNorm_##NAME##_Ctx, &GBHS);                            \
  }
#define STATS_NORMDIFF_PLAIN(NAME, T, O, GBHS)                                            \
  TEST_F(NppsStatisticsDiff, NormDiff_##NAME##_n257) {                                    \
    two_src_case<T, T, O>("nppsNormDiff_" #NAME, "stats/NormDiff/" #NAME "/n257",         \
        gen<T>(257, 0x502), gen<T>(257, 0x503), &nppsNormDiff_##NAME##_Ctx, &GBHS);       \
  }
#define STATS_NORMDIFF_SFS(NAME, T, O, GBHS)                                              \
  TEST_F(NppsStatisticsDiff, NormDiff_##NAME##_n257) {                                    \
    two_src_sfs_case<T, T, O>("nppsNormDiff_" #NAME, "stats/NormDiff/" #NAME "/n257",     \
        gen<T>(257, 0x502), gen<T>(257, 0x503), 2, &nppsNormDiff_##NAME##_Ctx, &GBHS);    \
  }

STATS_NORM_PLAIN(Inf_16s32f, Npp16s, Npp32f, nppsNormInfGetBufferSize_16s32f_Ctx)
STATS_NORM_PLAIN(Inf_32f, Npp32f, Npp32f, nppsNormInfGetBufferSize_32f_Ctx)
STATS_NORM_PLAIN(Inf_32fc32f, Npp32fc, Npp32f, nppsNormInfGetBufferSize_32fc32f_Ctx)
STATS_NORM_PLAIN(Inf_64f, Npp64f, Npp64f, nppsNormInfGetBufferSize_64f_Ctx)
STATS_NORM_PLAIN(Inf_64fc64f, Npp64fc, Npp64f, nppsNormInfGetBufferSize_64fc64f_Ctx)
STATS_NORM_SFS(Inf_16s32s_Sfs, Npp16s, Npp32s, nppsNormInfGetBufferSize_16s32s_Sfs_Ctx)

STATS_NORM_PLAIN(L1_16s32f, Npp16s, Npp32f, nppsNormL1GetBufferSize_16s32f_Ctx)
STATS_NORM_PLAIN(L1_32f, Npp32f, Npp32f, nppsNormL1GetBufferSize_32f_Ctx)
STATS_NORM_PLAIN(L1_32fc64f, Npp32fc, Npp64f, nppsNormL1GetBufferSize_32fc64f_Ctx)
STATS_NORM_PLAIN(L1_64f, Npp64f, Npp64f, nppsNormL1GetBufferSize_64f_Ctx)
STATS_NORM_PLAIN(L1_64fc64f, Npp64fc, Npp64f, nppsNormL1GetBufferSize_64fc64f_Ctx)
STATS_NORM_SFS(L1_16s32s_Sfs, Npp16s, Npp32s, nppsNormL1GetBufferSize_16s32s_Sfs_Ctx)
STATS_NORM_SFS(L1_16s64s_Sfs, Npp16s, Npp64s, nppsNormL1GetBufferSize_16s64s_Sfs_Ctx)

STATS_NORM_PLAIN(L2_16s32f, Npp16s, Npp32f, nppsNormL2GetBufferSize_16s32f_Ctx)
STATS_NORM_PLAIN(L2_32f, Npp32f, Npp32f, nppsNormL2GetBufferSize_32f_Ctx)
STATS_NORM_PLAIN(L2_32fc64f, Npp32fc, Npp64f, nppsNormL2GetBufferSize_32fc64f_Ctx)
STATS_NORM_PLAIN(L2_64f, Npp64f, Npp64f, nppsNormL2GetBufferSize_64f_Ctx)
STATS_NORM_PLAIN(L2_64fc64f, Npp64fc, Npp64f, nppsNormL2GetBufferSize_64fc64f_Ctx)
STATS_NORM_SFS(L2_16s32s_Sfs, Npp16s, Npp32s, nppsNormL2GetBufferSize_16s32s_Sfs_Ctx)

STATS_NORM_SFS(L2Sqr_16s64s_Sfs, Npp16s, Npp64s, nppsNormL2SqrGetBufferSize_16s64s_Sfs_Ctx)

STATS_NORMDIFF_PLAIN(Inf_16s32f, Npp16s, Npp32f, nppsNormDiffInfGetBufferSize_16s32f_Ctx)
STATS_NORMDIFF_PLAIN(Inf_32f, Npp32f, Npp32f, nppsNormDiffInfGetBufferSize_32f_Ctx)
STATS_NORMDIFF_PLAIN(Inf_32fc32f, Npp32fc, Npp32f, nppsNormDiffInfGetBufferSize_32fc32f_Ctx)
STATS_NORMDIFF_PLAIN(Inf_64f, Npp64f, Npp64f, nppsNormDiffInfGetBufferSize_64f_Ctx)
STATS_NORMDIFF_PLAIN(Inf_64fc64f, Npp64fc, Npp64f, nppsNormDiffInfGetBufferSize_64fc64f_Ctx)
STATS_NORMDIFF_SFS(Inf_16s32s_Sfs, Npp16s, Npp32s, nppsNormDiffInfGetBufferSize_16s32s_Sfs_Ctx)

STATS_NORMDIFF_PLAIN(L1_16s32f, Npp16s, Npp32f, nppsNormDiffL1GetBufferSize_16s32f_Ctx)
STATS_NORMDIFF_PLAIN(L1_32f, Npp32f, Npp32f, nppsNormDiffL1GetBufferSize_32f_Ctx)
STATS_NORMDIFF_PLAIN(L1_32fc64f, Npp32fc, Npp64f, nppsNormDiffL1GetBufferSize_32fc64f_Ctx)
STATS_NORMDIFF_PLAIN(L1_64f, Npp64f, Npp64f, nppsNormDiffL1GetBufferSize_64f_Ctx)
STATS_NORMDIFF_PLAIN(L1_64fc64f, Npp64fc, Npp64f, nppsNormDiffL1GetBufferSize_64fc64f_Ctx)
STATS_NORMDIFF_SFS(L1_16s32s_Sfs, Npp16s, Npp32s, nppsNormDiffL1GetBufferSize_16s32s_Sfs_Ctx)
STATS_NORMDIFF_SFS(L1_16s64s_Sfs, Npp16s, Npp64s, nppsNormDiffL1GetBufferSize_16s64s_Sfs_Ctx)

STATS_NORMDIFF_PLAIN(L2_16s32f, Npp16s, Npp32f, nppsNormDiffL2GetBufferSize_16s32f_Ctx)
STATS_NORMDIFF_PLAIN(L2_32f, Npp32f, Npp32f, nppsNormDiffL2GetBufferSize_32f_Ctx)
STATS_NORMDIFF_PLAIN(L2_32fc64f, Npp32fc, Npp64f, nppsNormDiffL2GetBufferSize_32fc64f_Ctx)
STATS_NORMDIFF_PLAIN(L2_64f, Npp64f, Npp64f, nppsNormDiffL2GetBufferSize_64f_Ctx)
STATS_NORMDIFF_PLAIN(L2_64fc64f, Npp64fc, Npp64f, nppsNormDiffL2GetBufferSize_64fc64f_Ctx)
STATS_NORMDIFF_SFS(L2_16s32s_Sfs, Npp16s, Npp32s, nppsNormDiffL2GetBufferSize_16s32s_Sfs_Ctx)

STATS_NORMDIFF_SFS(L2Sqr_16s64s_Sfs, Npp16s, Npp64s, nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs_Ctx)

#undef STATS_NORM_PLAIN
#undef STATS_NORM_SFS
#undef STATS_NORMDIFF_PLAIN
#undef STATS_NORMDIFF_SFS

// ===========================================================================
// MinEvery / MaxEvery: pSrcDst = op(pSrcDst, pSrc)
// ===========================================================================

#define STATS_EVERY_CASE(KIND, DT)                                                        \
  TEST_F(NppsStatisticsDiff, KIND##_##DT##_n257) {                                        \
    every_case<Npp##DT>("npps" #KIND "_" #DT "_I", "stats/" #KIND "/" #DT "/n257",        \
        gen<Npp##DT>(257, 0x511), gen<Npp##DT>(257, 0x512),                               \
        &npps##KIND##_##DT##_I_Ctx);                                                      \
  }

STATS_EVERY_CASE(MinEvery, 8u)
STATS_EVERY_CASE(MinEvery, 16s)
STATS_EVERY_CASE(MinEvery, 16u)
STATS_EVERY_CASE(MinEvery, 32f)
STATS_EVERY_CASE(MinEvery, 32s)
STATS_EVERY_CASE(MinEvery, 64f)
STATS_EVERY_CASE(MaxEvery, 8u)
STATS_EVERY_CASE(MaxEvery, 16s)
STATS_EVERY_CASE(MaxEvery, 16u)
STATS_EVERY_CASE(MaxEvery, 32f)
STATS_EVERY_CASE(MaxEvery, 32s)

#undef STATS_EVERY_CASE

// ===========================================================================
// CountInRange: (src, lo, hi) -> count
// ===========================================================================

TEST_F(NppsStatisticsDiff, CountInRange_32s_n257) {
  count_case("nppsCountInRange_32s", "stats/CountInRange/32s/n257",
      gen<Npp32s>(257, 0x531), -50000, 50000,
      &nppsCountInRange_32s_Ctx, &nppsCountInRangeGetBufferSize_32s_Ctx);
}

TEST_F(NppsStatisticsDiff, CountInRange_32s_Full) {
  count_case("nppsCountInRange_32s", "stats/CountInRange/32s/full",
      gen<Npp32s>(257, 0x531), std::numeric_limits<Npp32s>::min(),
      std::numeric_limits<Npp32s>::max(),
      &nppsCountInRange_32s_Ctx, &nppsCountInRangeGetBufferSize_32s_Ctx);
}

TEST_F(NppsStatisticsDiff, CountInRange_32s_Empty) {
  count_case("nppsCountInRange_32s", "stats/CountInRange/32s/empty",
      gen<Npp32s>(257, 0x531), 10, 10,
      &nppsCountInRange_32s_Ctx, &nppsCountInRangeGetBufferSize_32s_Ctx);
}

TEST_F(NppsStatisticsDiff, CountInRange_32s_Inverted) {
  count_case("nppsCountInRange_32s", "stats/CountInRange/32s/inverted",
      gen<Npp32s>(257, 0x531), 50000, -50000,
      &nppsCountInRange_32s_Ctx, &nppsCountInRangeGetBufferSize_32s_Ctx);
}

TEST_F(NppsStatisticsDiff, CountInRange_32s_n1) {
  count_case("nppsCountInRange_32s", "stats/CountInRange/32s/n1",
      gen<Npp32s>(1, 0x532), -100000, 100000,
      &nppsCountInRange_32s_Ctx, &nppsCountInRangeGetBufferSize_32s_Ctx);
}

// Range bounds are exclusive (probed: values 10 and 20 are not counted).
TEST_F(NppsStatisticsDiff, CountInRange_32s_Bounds) {
  count_case("nppsCountInRange_32s", "stats/CountInRange/32s/bounds",
      {10, 11, 19, 20, 21}, 10, 20,
      &nppsCountInRange_32s_Ctx, &nppsCountInRangeGetBufferSize_32s_Ctx);
}

// ===========================================================================
// ZeroCrossing: (src, mode) -> value
// ===========================================================================

#define STATS_ZC_CASE(DT, T, TAG, MODE, SEED)                                             \
  TEST_F(NppsStatisticsDiff, ZeroCrossing_##DT##_##TAG) {                                 \
    zc_case<T>("nppsZeroCrossing_" #DT, "stats/ZeroCrossing/" #DT "/" #TAG,               \
        gen<T>(257, SEED), MODE, &nppsZeroCrossing_##DT##_Ctx,                            \
        &nppsZeroCrossingGetBufferSize_##DT##_Ctx);                                       \
  }

STATS_ZC_CASE(16s32f, Npp16s, ZCR, nppZCR, 0x521)
STATS_ZC_CASE(16s32f, Npp16s, ZCXor, nppZCXor, 0x521)
STATS_ZC_CASE(16s32f, Npp16s, ZCC, nppZCC, 0x521)
STATS_ZC_CASE(32f, Npp32f, ZCR, nppZCR, 0x522)
STATS_ZC_CASE(32f, Npp32f, ZCXor, nppZCXor, 0x522)
STATS_ZC_CASE(32f, Npp32f, ZCC, nppZCC, 0x522)

#undef STATS_ZC_CASE

// ===========================================================================
// Min / Max / MinIndx / MaxIndx / MinMax / MinMaxIndx / MinAbs / MaxAbs
// ties resolve to the lowest index
// ===========================================================================

#define STATS_MINMAX_CASE(KIND, DT)                                                       \
  TEST_F(NppsStatisticsDiff, KIND##_##DT##_n257) {                                        \
    one_src_case<Npp##DT, Npp##DT>("npps" #KIND "_" #DT, "stats/" #KIND "/" #DT "/n257",  \
        gen<Npp##DT>(257, 0x401), &npps##KIND##_##DT##_Ctx,                               \
        &npps##KIND##GetBufferSize_##DT##_Ctx);                                           \
  }

#define STATS_VAL_INDX_CASE(KIND, DT)                                                     \
  TEST_F(NppsStatisticsDiff, KIND##_##DT##_n257) {                                        \
    val_indx_case<Npp##DT>("npps" #KIND "_" #DT, "stats/" #KIND "/" #DT "/n257",          \
        gen<Npp##DT>(257, 0x402), &npps##KIND##_##DT##_Ctx,                               \
        &npps##KIND##GetBufferSize_##DT##_Ctx);                                           \
  }

#define STATS_MINMAX2_CASE(KIND, DT)                                                      \
  TEST_F(NppsStatisticsDiff, KIND##_##DT##_n257) {                                        \
    minmax_case<Npp##DT>("npps" #KIND "_" #DT, "stats/" #KIND "/" #DT "/n257",            \
        gen<Npp##DT>(257, 0x403), &npps##KIND##_##DT##_Ctx,                               \
        &npps##KIND##GetBufferSize_##DT##_Ctx);                                           \
  }

#define STATS_MINMAX2_INDX_CASE(KIND, DT)                                                 \
  TEST_F(NppsStatisticsDiff, KIND##_##DT##_n257) {                                        \
    minmax_indx_case<Npp##DT>("npps" #KIND "_" #DT, "stats/" #KIND "/" #DT "/n257",       \
        gen<Npp##DT>(257, 0x404), &npps##KIND##_##DT##_Ctx,                               \
        &npps##KIND##GetBufferSize_##DT##_Ctx);                                           \
  }

STATS_MINMAX_CASE(Min, 16s)
STATS_MINMAX_CASE(Min, 32f)
STATS_MINMAX_CASE(Min, 32s)
STATS_MINMAX_CASE(Min, 64f)
STATS_MINMAX_CASE(Max, 16s)
STATS_MINMAX_CASE(Max, 32f)
STATS_MINMAX_CASE(Max, 32s)
STATS_MINMAX_CASE(Max, 64f)
STATS_VAL_INDX_CASE(MinIndx, 16s)
STATS_VAL_INDX_CASE(MinIndx, 32f)
STATS_VAL_INDX_CASE(MinIndx, 32s)
STATS_VAL_INDX_CASE(MinIndx, 64f)
STATS_VAL_INDX_CASE(MaxIndx, 16s)
STATS_VAL_INDX_CASE(MaxIndx, 32f)
STATS_VAL_INDX_CASE(MaxIndx, 32s)
STATS_VAL_INDX_CASE(MaxIndx, 64f)
STATS_MINMAX_CASE(MinAbs, 16s)
STATS_MINMAX_CASE(MinAbs, 32s)
STATS_MINMAX_CASE(MaxAbs, 16s)
STATS_MINMAX_CASE(MaxAbs, 32s)
STATS_VAL_INDX_CASE(MinAbsIndx, 16s)
STATS_VAL_INDX_CASE(MinAbsIndx, 32s)
STATS_VAL_INDX_CASE(MaxAbsIndx, 16s)
STATS_VAL_INDX_CASE(MaxAbsIndx, 32s)
STATS_MINMAX2_CASE(MinMax, 8u)
STATS_MINMAX2_CASE(MinMax, 16s)
STATS_MINMAX2_CASE(MinMax, 16u)
STATS_MINMAX2_CASE(MinMax, 32f)
STATS_MINMAX2_CASE(MinMax, 32s)
STATS_MINMAX2_CASE(MinMax, 32u)
STATS_MINMAX2_CASE(MinMax, 64f)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 8u)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 16s)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 16u)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 32f)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 32s)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 32u)
STATS_MINMAX2_INDX_CASE(MinMaxIndx, 64f)
#undef STATS_MINMAX_CASE
#undef STATS_VAL_INDX_CASE
#undef STATS_MINMAX2_CASE
#undef STATS_MINMAX2_INDX_CASE

// crafted tie: minimum -7 at indices 5 and 200, maximum 9 at 0 and 256
TEST_F(NppsStatisticsDiff, MinMaxIndx_32f_Tie) {
  std::vector<Npp32f> v(257, 0.0f);
  v[0] = 9.0f;
  v[5] = -7.0f;
  v[200] = -7.0f;
  v[256] = 9.0f;
  minmax_indx_case<Npp32f>("nppsMinMaxIndx_32f", "stats/MinMaxIndx/32f/tie", v,
      &nppsMinMaxIndx_32f_Ctx, &nppsMinMaxGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MinIndx_32f_Tie) {
  std::vector<Npp32f> v(257, 1.0f);
  v[10] = -3.0f;
  v[150] = -3.0f;
  val_indx_case<Npp32f>("nppsMinIndx_32f", "stats/MinIndx/32f/tie", v,
      &nppsMinIndx_32f_Ctx, &nppsMinIndxGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MinIndx_32f_Nan) {
  std::vector<Npp32f> v = gen<Npp32f>(257, 0x405);
  v[13] = std::nanf("1");
  val_indx_case<Npp32f>("nppsMinIndx_32f", "stats/MinIndx/32f/nan", v,
      &nppsMinIndx_32f_Ctx, &nppsMinIndxGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MaxIndx_32f_Nan) {
  std::vector<Npp32f> v = gen<Npp32f>(257, 0x406);
  v[13] = std::nanf("1");
  val_indx_case<Npp32f>("nppsMaxIndx_32f", "stats/MaxIndx/32f/nan", v,
      &nppsMaxIndx_32f_Ctx, &nppsMaxIndxGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, Min_32f_n1) {
  one_src_case<Npp32f, Npp32f>("nppsMin_32f", "stats/Min/32f/n1",
      gen<Npp32f>(1, 0x407), &nppsMin_32f_Ctx, &nppsMinGetBufferSize_32f_Ctx);
}

TEST_F(NppsStatisticsDiff, MinMax_16s_n1) {
  minmax_case<Npp16s>("nppsMinMax_16s", "stats/MinMax/16s/n1",
      gen<Npp16s>(1, 0x408), &nppsMinMax_16s_Ctx, &nppsMinMaxGetBufferSize_16s_Ctx);
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsStatisticsDiff, Min_32f_Plain_n257) {
  one_src_case_nc<Npp32f, Npp32f>("nppsMin_32f", "stats/Min/32f/n257",
      gen<Npp32f>(257, 0x401), &nppsMin_32f, &nppsMinGetBufferSize_32f);
}

TEST_F(NppsStatisticsDiff, MinMaxIndx_16s_Plain_n257) {
  minmax_indx_case_nc<Npp16s>("nppsMinMaxIndx_16s", "stats/MinMaxIndx/16s/n257",
      gen<Npp16s>(257, 0x404), &nppsMinMaxIndx_16s, &nppsMinMaxIndxGetBufferSize_16s);
}
#endif
