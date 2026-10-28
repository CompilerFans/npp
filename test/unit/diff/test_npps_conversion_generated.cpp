// Differential tests for the npps conversion + threshold families (p1.2b).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppsConvertDiff*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppsConvertDiff*'
//
// The comprehensive matrix exercises the `_Ctx` entry points (the reference
// NPP 13 build only exports those).  A small set of NPP_DIFF_HAS_NONCTX-guarded
// cases additionally proves the non-Ctx adapter forwards to the same golden.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

// ---- boundary input builders ------------------------------------------------
std::vector<Npp8s>  b8s()  { return {-128, -2, -1, 0, 1, 64, 127}; }
std::vector<Npp8u>  b8u()  { return {0, 1, 128, 255}; }
std::vector<Npp16s> b16s() { return {-32768, -2, -1, 0, 1, 2, 32767}; }
std::vector<Npp16u> b16u() { return {0, 1, 32768, 65535}; }
std::vector<Npp32s> b32s() { return {-2147483647 - 1, -1, 0, 1, 32768, 2147483647}; }
std::vector<Npp64s> b64s() { return {(-9223372036854775807LL - 1), -1, 0, 1, 32768, 9223372036854775807LL}; }
std::vector<Npp32f> b32f() { return {-1e6f, -3.75f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 3.75f, 1e6f}; }
std::vector<Npp64f> b64f() { return {-1e6, -3.75, -1.5, -0.5, 0.0, 0.5, 1.5, 3.75, 1e6}; }

template <typename D>
ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8s>)  return ValueKind::I8;
  else if constexpr (std::is_same_v<D, Npp8u>)   return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16s>)  return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp16u>)  return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp32s>)  return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp64s>)  return ValueKind::I64;
  else if constexpr (std::is_same_v<D, Npp32f>)  return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp64f>)  return ValueKind::F64;
  else if constexpr (std::is_same_v<D, Npp32fc>) return ValueKind::CF32;
  else if constexpr (std::is_same_v<D, Npp64fc>) return ValueKind::CF64;
  else return ValueKind::Bytes;  // Npp16sc -> exact byte comparison
}

template <typename T>
std::vector<T> th_data() {
  std::vector<T> v;
  for (int k = -6; k <= 6; ++k) v.push_back(static_cast<T>(k));
  return v;
}

}  // namespace

class NppsConvertDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ---------------------------------------------------------------------------
// Convert helpers (Ctx only)
// ---------------------------------------------------------------------------
namespace {

template <typename A, typename D>
void convert_plain(const char *fname, const std::string &idBase, const std::vector<A> &hsrc,
                   NppStatus (*f)(const A *, D *, NppSignalLength, NppStreamContext)) {
  const std::string id = "nppsConvert/" + idBase + "/basic";
  DeviceMemory<A> s(hsrc.size());
  DeviceMemory<D> d(hsrc.size());
  s.copyFromHost(hsrc);
  clear_cuda_error();
  NppStatus st = f(s.get(), d.get(), hsrc.size(), current_context());
  Case c(id, fname);
  c.input(hsrc.data(), hsrc.size() * sizeof(A));
  c.status(st);
  c.device_output<D>(d.get(), hsrc.size(), out_kind<D>(), 1e-5, 1e-6);
  c.finish();
}

template <typename A, typename D>
void convert_scale(const char *fname, const std::string &idBase, const std::vector<A> &hsrc, int sf,
                   NppStatus (*f)(const A *, D *, NppSignalLength, int, NppStreamContext)) {
  const std::string id = "nppsConvert/" + idBase + "/scale";
  DeviceMemory<A> s(hsrc.size());
  DeviceMemory<D> d(hsrc.size());
  s.copyFromHost(hsrc);
  clear_cuda_error();
  NppStatus st = f(s.get(), d.get(), hsrc.size(), sf, current_context());
  Case c(id, fname);
  c.input(hsrc.data(), hsrc.size() * sizeof(A));
  c.status(st);
  c.device_output<D>(d.get(), hsrc.size(), out_kind<D>(), 1e-5, 1e-6);
  c.finish();
}

template <typename A, typename D>
void convert_round(const char *fname, const std::string &idBase, const std::vector<A> &hsrc,
                   NppRoundMode mode, int sf,
                   NppStatus (*f)(const A *, D *, NppSignalLength, NppRoundMode, int, NppStreamContext)) {
  const std::string id = "nppsConvert/" + idBase + "/round";
  DeviceMemory<A> s(hsrc.size());
  DeviceMemory<D> d(hsrc.size());
  s.copyFromHost(hsrc);
  clear_cuda_error();
  NppStatus st = f(s.get(), d.get(), hsrc.size(), mode, sf, current_context());
  Case c(id, fname);
  c.input(hsrc.data(), hsrc.size() * sizeof(A));
  c.status(st);
  c.device_output<D>(d.get(), hsrc.size(), out_kind<D>(), 1e-5, 1e-6);
  c.finish();
}

// 16s8s_Sfs uses Npp32u nLength.
void convert_16s8s(const std::vector<Npp16s> &hsrc) {
  const std::string id = "nppsConvert/16s8s_sfs/round";
  DeviceMemory<Npp16s> s(hsrc.size());
  DeviceMemory<Npp8s> d(hsrc.size());
  s.copyFromHost(hsrc);
  clear_cuda_error();
  NppStatus st = nppsConvert_16s8s_Sfs_Ctx(s.get(), d.get(), static_cast<Npp32u>(hsrc.size()),
                                           NPP_RND_FINANCIAL, 1, current_context());
  Case c(id, "nppsConvert_16s8s_Sfs");
  c.input(hsrc.data(), hsrc.size() * sizeof(Npp16s));
  c.status(st);
  c.device_output<Npp8s>(d.get(), hsrc.size(), ValueKind::I8, 1e-5, 1e-6);
  c.finish();
}

}  // namespace

TEST_F(NppsConvertDiff, Convert_8s16s)   { convert_plain("nppsConvert_8s16s", "8s16s", b8s(), &nppsConvert_8s16s_Ctx); }
TEST_F(NppsConvertDiff, Convert_8s32f)   { convert_plain("nppsConvert_8s32f", "8s32f", b8s(), &nppsConvert_8s32f_Ctx); }
TEST_F(NppsConvertDiff, Convert_8u32f)   { convert_plain("nppsConvert_8u32f", "8u32f", b8u(), &nppsConvert_8u32f_Ctx); }
TEST_F(NppsConvertDiff, Convert_16s32f)  { convert_plain("nppsConvert_16s32f", "16s32f", b16s(), &nppsConvert_16s32f_Ctx); }
TEST_F(NppsConvertDiff, Convert_16s32s)  { convert_plain("nppsConvert_16s32s", "16s32s", b16s(), &nppsConvert_16s32s_Ctx); }
TEST_F(NppsConvertDiff, Convert_16u32f)  { convert_plain("nppsConvert_16u32f", "16u32f", b16u(), &nppsConvert_16u32f_Ctx); }
TEST_F(NppsConvertDiff, Convert_32s16s)  { convert_plain("nppsConvert_32s16s", "32s16s", b32s(), &nppsConvert_32s16s_Ctx); }
TEST_F(NppsConvertDiff, Convert_32s32f)  { convert_plain("nppsConvert_32s32f", "32s32f", b32s(), &nppsConvert_32s32f_Ctx); }
TEST_F(NppsConvertDiff, Convert_32s64f)  { convert_plain("nppsConvert_32s64f", "32s64f", b32s(), &nppsConvert_32s64f_Ctx); }
TEST_F(NppsConvertDiff, Convert_64s64f)  { convert_plain("nppsConvert_64s64f", "64s64f", b64s(), &nppsConvert_64s64f_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f64f)  { convert_plain("nppsConvert_32f64f", "32f64f", b32f(), &nppsConvert_32f64f_Ctx); }
TEST_F(NppsConvertDiff, Convert_64f32f)  { convert_plain("nppsConvert_64f32f", "64f32f", b64f(), &nppsConvert_64f32f_Ctx); }

TEST_F(NppsConvertDiff, Convert_16s32f_Sfs){ convert_scale("nppsConvert_16s32f_Sfs","16s32f_ss",b16s(),1,&nppsConvert_16s32f_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_16s64f_Sfs){ convert_scale("nppsConvert_16s64f_Sfs","16s64f_ss",b16s(),2,&nppsConvert_16s64f_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32s16s_Sfs){ convert_scale("nppsConvert_32s16s_Sfs","32s16s_ss",b32s(),1,&nppsConvert_32s16s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32s32f_Sfs){ convert_scale("nppsConvert_32s32f_Sfs","32s32f_ss",b32s(),1,&nppsConvert_32s32f_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32s64f_Sfs){ convert_scale("nppsConvert_32s64f_Sfs","32s64f_ss",b32s(),1,&nppsConvert_32s64f_Sfs_Ctx); }

TEST_F(NppsConvertDiff, Convert_32f16s_Sfs){ convert_round("nppsConvert_32f16s_Sfs","32f16s_ss",b32f(),NPP_RND_NEAR,1,&nppsConvert_32f16s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f16u_Sfs){ convert_round("nppsConvert_32f16u_Sfs","32f16u_ss",b32f(),NPP_RND_NEAR,1,&nppsConvert_32f16u_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f8s_Sfs_Near){ convert_round("nppsConvert_32f8s_Sfs","32f8s_ss_near",b32f(),NPP_RND_NEAR,1,&nppsConvert_32f8s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f8s_Sfs_Fin){ convert_round("nppsConvert_32f8s_Sfs","32f8s_ss_fin",b32f(),NPP_RND_FINANCIAL,1,&nppsConvert_32f8s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f8s_Sfs_Zero){ convert_round("nppsConvert_32f8s_Sfs","32f8s_ss_zero",b32f(),NPP_RND_ZERO,1,&nppsConvert_32f8s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f8u_Sfs){ convert_round("nppsConvert_32f8u_Sfs","32f8u_ss",b32f(),NPP_RND_NEAR,1,&nppsConvert_32f8u_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_32f32s_Sfs){ convert_round("nppsConvert_32f32s_Sfs","32f32s_ss",b32f(),NPP_RND_NEAR,1,&nppsConvert_32f32s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_64f16s_Sfs){ convert_round("nppsConvert_64f16s_Sfs","64f16s_ss",b64f(),NPP_RND_NEAR,1,&nppsConvert_64f16s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_64f32s_Sfs){ convert_round("nppsConvert_64f32s_Sfs","64f32s_ss",b64f(),NPP_RND_NEAR,1,&nppsConvert_64f32s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_64f64s_Sfs){ convert_round("nppsConvert_64f64s_Sfs","64f64s_ss",b64f(),NPP_RND_FINANCIAL,2,&nppsConvert_64f64s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_64s32s_Sfs){ convert_round("nppsConvert_64s32s_Sfs","64s32s_ss",b64s(),NPP_RND_NEAR,1,&nppsConvert_64s32s_Sfs_Ctx); }
TEST_F(NppsConvertDiff, Convert_16s8s_Sfs){ convert_16s8s(b16s()); }

TEST_F(NppsConvertDiff, Contract_Null_16s32f) {
  clear_cuda_error();
  NppStatus st = nppsConvert_16s32f_Ctx(nullptr, nullptr, 4, current_context());
  Case c("nppsConvert/16s32f/null", "nppsConvert_16s32f_Ctx");
  c.status(st);
  c.finish();
}
TEST_F(NppsConvertDiff, Contract_ZeroLen_8u32f) {
  clear_cuda_error();
  NppStatus st = nppsConvert_8u32f_Ctx(nullptr, nullptr, 0, current_context());
  Case c("nppsConvert/8u32f/zerolen", "nppsConvert_8u32f_Ctx");
  c.status(st);
  c.finish();
}

// ---------------------------------------------------------------------------
// Threshold helpers (Ctx only)
// ---------------------------------------------------------------------------
namespace {

template <typename T, typename L>
void th_op(const char *fname, const std::string &idBase, const std::vector<T> &hsrc, L level,
           NppCmpOp op, NppStatus (*f)(const T *, T *, NppSignalLength, L, NppCmpOp, NppStreamContext),
           NppStatus (*fi)(T *, NppSignalLength, L, NppCmpOp, NppStreamContext)) {
  const std::string opSuffix = (op == NPP_CMP_GREATER) ? "GT" : "LT";
  const std::string id = "nppsThreshold/" + idBase + "/" + opSuffix;
  {  // out of place
    DeviceMemory<T> s(hsrc.size());
    DeviceMemory<T> d(hsrc.size());
    s.copyFromHost(hsrc);
    clear_cuda_error();
    NppStatus st = f(s.get(), d.get(), hsrc.size(), level, op, current_context());
    Case c(id, fname);
    c.input(hsrc.data(), hsrc.size() * sizeof(T));
    c.status(st);
    c.device_output<T>(d.get(), hsrc.size(), out_kind<T>(), 1e-5, 1e-6);
    c.finish();
  }
  {  // in place
    DeviceMemory<T> sd(hsrc.size());
    sd.copyFromHost(hsrc);
    clear_cuda_error();
    NppStatus st = fi(sd.get(), hsrc.size(), level, op, current_context());
    Case c(id + "/In", fname);
    c.input(hsrc.data(), hsrc.size() * sizeof(T));
    c.status(st);
    c.device_output<T>(sd.get(), hsrc.size(), out_kind<T>(), 1e-5, 1e-6);
    c.finish();
  }
}

template <typename T, typename L>
void th_fixed(const char *fname, const std::string &idBase, const std::vector<T> &hsrc, L level,
              NppStatus (*f)(const T *, T *, NppSignalLength, L, NppStreamContext),
              NppStatus (*fi)(T *, NppSignalLength, L, NppStreamContext)) {
  const std::string id = "nppsThreshold/" + idBase + "/basic";
  {
    DeviceMemory<T> s(hsrc.size());
    DeviceMemory<T> d(hsrc.size());
    s.copyFromHost(hsrc);
    clear_cuda_error();
    NppStatus st = f(s.get(), d.get(), hsrc.size(), level, current_context());
    Case c(id, fname);
    c.input(hsrc.data(), hsrc.size() * sizeof(T));
    c.status(st);
    c.device_output<T>(d.get(), hsrc.size(), out_kind<T>(), 1e-5, 1e-6);
    c.finish();
  }
  {
    DeviceMemory<T> sd(hsrc.size());
    sd.copyFromHost(hsrc);
    clear_cuda_error();
    NppStatus st = fi(sd.get(), hsrc.size(), level, current_context());
    Case c(id + "/In", fname);
    c.input(hsrc.data(), hsrc.size() * sizeof(T));
    c.status(st);
    c.device_output<T>(sd.get(), hsrc.size(), out_kind<T>(), 1e-5, 1e-6);
    c.finish();
  }
}

template <typename T, typename L>
void th_val(const char *fname, const std::string &idBase, const std::vector<T> &hsrc, L level, T value,
            NppStatus (*f)(const T *, T *, NppSignalLength, L, T, NppStreamContext),
            NppStatus (*fi)(T *, NppSignalLength, L, T, NppStreamContext)) {
  const std::string id = "nppsThreshold/" + idBase + "/basic";
  {
    DeviceMemory<T> s(hsrc.size());
    DeviceMemory<T> d(hsrc.size());
    s.copyFromHost(hsrc);
    clear_cuda_error();
    NppStatus st = f(s.get(), d.get(), hsrc.size(), level, value, current_context());
    Case c(id, fname);
    c.input(hsrc.data(), hsrc.size() * sizeof(T));
    c.status(st);
    c.device_output<T>(d.get(), hsrc.size(), out_kind<T>(), 1e-5, 1e-6);
    c.finish();
  }
  {
    DeviceMemory<T> sd(hsrc.size());
    sd.copyFromHost(hsrc);
    clear_cuda_error();
    NppStatus st = fi(sd.get(), hsrc.size(), level, value, current_context());
    Case c(id + "/In", fname);
    c.input(hsrc.data(), hsrc.size() * sizeof(T));
    c.status(st);
    c.device_output<T>(sd.get(), hsrc.size(), out_kind<T>(), 1e-5, 1e-6);
    c.finish();
  }
}

}  // namespace

TEST_F(NppsConvertDiff, Threshold_16s) {
  th_op<Npp16s, Npp16s>("nppsThreshold_16s", "16s", th_data<Npp16s>(), (Npp16s)0, NPP_CMP_LESS,
                        &nppsThreshold_16s_Ctx, &nppsThreshold_16s_I_Ctx);
  th_op<Npp16s, Npp16s>("nppsThreshold_16s", "16s", th_data<Npp16s>(), (Npp16s)0, NPP_CMP_GREATER,
                         &nppsThreshold_16s_Ctx, &nppsThreshold_16s_I_Ctx);
}
TEST_F(NppsConvertDiff, Threshold_32f) {
  th_op<Npp32f, Npp32f>("nppsThreshold_32f", "32f", th_data<Npp32f>(), 0.0f, NPP_CMP_LESS,
                        &nppsThreshold_32f_Ctx, &nppsThreshold_32f_I_Ctx);
  th_op<Npp32f, Npp32f>("nppsThreshold_32f", "32f", th_data<Npp32f>(), 0.0f, NPP_CMP_GREATER,
                        &nppsThreshold_32f_Ctx, &nppsThreshold_32f_I_Ctx);
}
TEST_F(NppsConvertDiff, Threshold_64f) {
  th_op<Npp64f, Npp64f>("nppsThreshold_64f", "64f", th_data<Npp64f>(), 0.0, NPP_CMP_LESS,
                        &nppsThreshold_64f_Ctx, &nppsThreshold_64f_I_Ctx);
  th_op<Npp64f, Npp64f>("nppsThreshold_64f", "64f", th_data<Npp64f>(), 0.0, NPP_CMP_GREATER,
                        &nppsThreshold_64f_Ctx, &nppsThreshold_64f_I_Ctx);
}
TEST_F(NppsConvertDiff, Threshold_16sc) {
  std::vector<Npp16sc> v;
  for (int k = -4; k <= 4; ++k) { Npp16sc t; t.re = (Npp16s)(k * 2); t.im = (Npp16s)(k + 1); v.push_back(t); }
  th_op<Npp16sc, Npp16s>("nppsThreshold_16sc", "16sc", v, (Npp16s)2, NPP_CMP_LESS,
                         &nppsThreshold_16sc_Ctx, &nppsThreshold_16sc_I_Ctx);
  th_op<Npp16sc, Npp16s>("nppsThreshold_16sc", "16sc", v, (Npp16s)2, NPP_CMP_GREATER,
                         &nppsThreshold_16sc_Ctx, &nppsThreshold_16sc_I_Ctx);
}
TEST_F(NppsConvertDiff, Threshold_32fc) {
  std::vector<Npp32fc> v;
  for (auto k = -4; k <= 4; ++k) { Npp32fc t; t.re = (Npp32f)(k * 2); t.im = (Npp32f)(k + 1); v.push_back(t); }
  th_op<Npp32fc, Npp32f>("nppsThreshold_32fc", "32fc", v, 2.0f, NPP_CMP_LESS,
                         &nppsThreshold_32fc_Ctx, &nppsThreshold_32fc_I_Ctx);
  th_op<Npp32fc, Npp32f>("nppsThreshold_32fc", "32fc", v, 2.0f, NPP_CMP_GREATER,
                         &nppsThreshold_32fc_Ctx, &nppsThreshold_32fc_I_Ctx);
}
TEST_F(NppsConvertDiff, Threshold_64fc) {
  std::vector<Npp64fc> v;
  for (auto k = -4; k <= 4; ++k) { Npp64fc t; t.re = (Npp64f)(k * 2); t.im = (Npp64f)(k + 1); v.push_back(t); }
  th_op<Npp64fc, Npp64f>("nppsThreshold_64fc", "64fc", v, 2.0, NPP_CMP_LESS,
                         &nppsThreshold_64fc_Ctx, &nppsThreshold_64fc_I_Ctx);
  th_op<Npp64fc, Npp64f>("nppsThreshold_64fc", "64fc", v, 2.0, NPP_CMP_GREATER,
                         &nppsThreshold_64fc_Ctx, &nppsThreshold_64fc_I_Ctx);
}

// GT / LT
TEST_F(NppsConvertDiff, Threshold_GT_16s)  { th_fixed<Npp16s, Npp16s>("nppsThreshold_GT_16s","GT_16s",th_data<Npp16s>(),(Npp16s)1,&nppsThreshold_GT_16s_Ctx,&nppsThreshold_GT_16s_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LT_16s)  { th_fixed<Npp16s, Npp16s>("nppsThreshold_LT_16s","LT_16s",th_data<Npp16s>(),(Npp16s)1,&nppsThreshold_LT_16s_Ctx,&nppsThreshold_LT_16s_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_GT_32f)  { th_fixed<Npp32f, Npp32f>("nppsThreshold_GT_32f","GT_32f",th_data<Npp32f>(),0.0f,&nppsThreshold_GT_32f_Ctx,&nppsThreshold_GT_32f_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LT_32f)  { th_fixed<Npp32f, Npp32f>("nppsThreshold_LT_32f","LT_32f",th_data<Npp32f>(),0.0f,&nppsThreshold_LT_32f_Ctx,&nppsThreshold_LT_32f_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_GT_64f)  { th_fixed<Npp64f, Npp64f>("nppsThreshold_GT_64f","GT_64f",th_data<Npp64f>(),0.0,&nppsThreshold_GT_64f_Ctx,&nppsThreshold_GT_64f_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LT_64f)  { th_fixed<Npp64f, Npp64f>("nppsThreshold_LT_64f","LT_64f",th_data<Npp64f>(),0.0,&nppsThreshold_LT_64f_Ctx,&nppsThreshold_LT_64f_I_Ctx); }

namespace {
template <typename T>
std::vector<T> cplx() {
  std::vector<T> v;
  for (auto k = -4; k <= 4; ++k) {
    T t;
    if constexpr (std::is_same_v<T, Npp16sc>) { t.re = (Npp16s)(k * 2); t.im = (Npp16s)(k + 1); }
    else if constexpr (std::is_same_v<T, Npp32fc>) { t.re = (Npp32f)(k * 2); t.im = (Npp32f)(k + 1); }
    else { t.re = (Npp64f)(k * 2); t.im = (Npp64f)(k + 1); }
    v.push_back(t);
  }
  return v;
}
}  // namespace

TEST_F(NppsConvertDiff, Threshold_GT_16sc) { th_fixed<Npp16sc, Npp16s>("nppsThreshold_GT_16sc","GT_16sc",cplx<Npp16sc>(),(Npp16s)2,&nppsThreshold_GT_16sc_Ctx,&nppsThreshold_GT_16sc_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LT_16sc) { th_fixed<Npp16sc, Npp16s>("nppsThreshold_LT_16sc","LT_16sc",cplx<Npp16sc>(),(Npp16s)2,&nppsThreshold_LT_16sc_Ctx,&nppsThreshold_LT_16sc_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_GT_32fc) { th_fixed<Npp32fc, Npp32f>("nppsThreshold_GT_32fc","GT_32fc",cplx<Npp32fc>(),2.0f,&nppsThreshold_GT_32fc_Ctx,&nppsThreshold_GT_32fc_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LT_32fc) { th_fixed<Npp32fc, Npp32f>("nppsThreshold_LT_32fc","LT_32fc",cplx<Npp32fc>(),2.0f,&nppsThreshold_LT_32fc_Ctx,&nppsThreshold_LT_32fc_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_GT_64fc) { th_fixed<Npp64fc, Npp64f>("nppsThreshold_GT_64fc","GT_64fc",cplx<Npp64fc>(),2.0,&nppsThreshold_GT_64fc_Ctx,&nppsThreshold_GT_64fc_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LT_64fc) { th_fixed<Npp64fc, Npp64f>("nppsThreshold_LT_64fc","LT_64fc",cplx<Npp64fc>(),2.0,&nppsThreshold_LT_64fc_Ctx,&nppsThreshold_LT_64fc_I_Ctx); }

// GTVal/LTVal
TEST_F(NppsConvertDiff, Threshold_GTVal_16s) { th_val<Npp16s, Npp16s>("nppsThreshold_GTVal_16s","GTVal_16s",th_data<Npp16s>(),(Npp16s)0,(Npp16s)-5,&nppsThreshold_GTVal_16s_Ctx,&nppsThreshold_GTVal_16s_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LTVal_16s) { th_val<Npp16s, Npp16s>("nppsThreshold_LTVal_16s","LTVal_16s",th_data<Npp16s>(),(Npp16s)0,(Npp16s)9,&nppsThreshold_LTVal_16s_Ctx,&nppsThreshold_LTVal_16s_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_GTVal_32f) { th_val<Npp32f, Npp32f>("nppsThreshold_GTVal_32f","GTVal_32f",th_data<Npp32f>(),0.0f,5.5f,&nppsThreshold_GTVal_32f_Ctx,&nppsThreshold_GTVal_32f_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LTVal_32f) { th_val<Npp32f, Npp32f>("nppsThreshold_LTVal_32f","LTVal_32f",th_data<Npp32f>(),0.0f,-5.5f,&nppsThreshold_LTVal_32f_Ctx,&nppsThreshold_LTVal_32f_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_GTVal_64f) { th_val<Npp64f, Npp64f>("nppsThreshold_GTVal_64f","GTVal_64f",th_data<Npp64f>(),0.0,5.5,&nppsThreshold_GTVal_64f_Ctx,&nppsThreshold_GTVal_64f_I_Ctx); }
TEST_F(NppsConvertDiff, Threshold_LTVal_64f) { th_val<Npp64f, Npp64f>("nppsThreshold_LTVal_64f","LTVal_64f",th_data<Npp64f>(),0.0,-5.5,&nppsThreshold_LTVal_64f_Ctx,&nppsThreshold_LTVal_64f_I_Ctx); }

TEST_F(NppsConvertDiff, Threshold_Null_32f) {
  clear_cuda_error();
  NppStatus st = nppsThreshold_32f_Ctx(nullptr, nullptr, 4, 0.0f, NPP_CMP_GREATER, current_context());
  Case c("nppsThreshold/32f/null", "nppsThreshold_32f_Ctx");
  c.status(st);
  c.finish();
}
TEST_F(NppsConvertDiff, Threshold_Zero_32f) {
  clear_cuda_error();
  NppStatus st = nppsThreshold_32f_Ctx(nullptr, nullptr, 0, 0.0f, NPP_CMP_GREATER, current_context());
  Case c("nppsThreshold/32f/zerolen", "nppsThreshold_32f_Ctx");
  c.status(st);
  c.finish();
}

// ---------------------------------------------------------------------------
// Non-Ctx adapter spot checks (MPP only): same golden id as the Ctx case.
// ---------------------------------------------------------------------------
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsConvertDiff, Convert_16s32f_NonCtx) {
  std::vector<Npp16s> hsrc = b16s();
  DeviceMemory<Npp16s> s(hsrc.size());
  DeviceMemory<Npp32f> d(hsrc.size());
  s.copyFromHost(hsrc);
  clear_cuda_error();
  NppStatus st = nppsConvert_16s32f(s.get(), d.get(), static_cast<int>(hsrc.size()));
  Case c("nppsConvert/16s32f/basic", "nppsConvert_16s32f");
  c.status(st);
  c.device_output<Npp32f>(d.get(), hsrc.size(), ValueKind::F32, 1e-5, 1e-6);
  c.finish();
}
TEST_F(NppsConvertDiff, Threshold_32f_NonCtx) {
  std::vector<Npp32f> hsrc = th_data<Npp32f>();
  DeviceMemory<Npp32f> s(hsrc.size());
  DeviceMemory<Npp32f> d(hsrc.size());
  s.copyFromHost(hsrc);
  clear_cuda_error();
  NppStatus st = nppsThreshold_32f(s.get(), d.get(), static_cast<int>(hsrc.size()), 0.0f, NPP_CMP_GREATER);
  Case c("nppsThreshold/32f/GT", "nppsThreshold_32f");
  c.status(st);
  c.device_output<Npp32f>(d.get(), hsrc.size(), ValueKind::F32, 1e-5, 1e-6);
  c.finish();
}
#endif
