// CPU-golden coverage for the runtime-typed fused absdiff threshold functions
// (nppi_threshold_and_compare_operations.h):
//   nppiFusedAbsDiff_Threshold_GTVal_Ctx and the in-place variant.
//
// Measured NPP 13.4 semantics (RTX 5090):
//   * supported source types are 8u, 16u, 16s and 32f (8s/32s/64f are
//     rejected by the reference);
//   * NPP_CH_1 uses one threshold/value; NPP_CH_3 and NPP_CH_A4 use three
//     per-channel values;
//   * the A4 alpha element is left untouched (the destination keeps it);
//   * out = |a-b|, replaced by the value when |a-b| > threshold (strict).
#pragma once

#include "num_nppi_geom.h"

namespace num {
namespace fusedabs {

template <typename T, int PHY, int ARR>
inline bool check_case(const char *id, const GenCase &gc, const uint8_t *s1, int ss1, const uint8_t *s2, int ss2,
                       const uint8_t *pd, int sd, const void *thr, const void *val, int w, int h,
                       const std::vector<uint8_t> &pre, RunCounts &cnt, std::string &first) {
  std::vector<uint8_t> want = pre;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < PHY; ++c) {
        if (PHY == 4 && c == 3) continue;
        const size_t offA = (size_t)y * ss1 + (size_t)(x * PHY + c) * sizeof(T);
        const size_t offB = (size_t)y * ss2 + (size_t)(x * PHY + c) * sizeof(T);
        const size_t offD = (size_t)y * sd + (size_t)(x * PHY + c) * sizeof(T);
        const T a = *(const T *)(s1 + offA);
        const T b = *(const T *)(s2 + offB);
        const T ad = (T)(a >= b ? a - b : b - a);
        const T th = ((const T *)thr)[c < ARR ? c : 0];
        const T vv = ((const T *)val)[c < ARR ? c : 0];
        const T out = (ad > th) ? vv : ad;
        std::memcpy(want.data() + offD, &out, sizeof(T));
      }
  const size_t rowBytes = (size_t)w * PHY * sizeof(T);
  bool ok = true;
  for (int y = 0; y < h; ++y)
    for (size_t i = 0; i < rowBytes; i += sizeof(T)) {
      ++cnt.compared;
      if (std::memcmp(pd + (size_t)y * sd + i, want.data() + (size_t)y * sd + i, sizeof(T)) == 0) continue;
      char b[320];
      std::snprintf(b, sizeof(b), "[%s] %s row=%d elem=%zu got=%s want=%s", id, gc.id().c_str(), y,
                    i / sizeof(T), dbg_elem<T>(pd + (size_t)y * sd + i).c_str(),
                    dbg_elem<T>(want.data() + (size_t)y * sd + i).c_str());
      if (first.empty()) first = b;
      EXPECT_TRUE(false) << b;
      ++cnt.mismatched;
      ok = false;
      if (cnt.mismatched >= 8) return false;
    }
  return ok;
}

template <bool Inplace, class Call>
void run_fused(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  RunCounts cnt;
  std::string first;
  bool failed = false;
  struct Cfg {
    int w, h;
  };
  static const Cfg cfgs[] = {{64, 64}, {96, 64}, {17, 13}};
  // NppDataType values for 8u, 16u, 16s and 32f (the enum is not dense).
  static const int kDt[4] = {NPP_8U, NPP_16U, NPP_16S, NPP_32F};
  // NppiChannels values for CH_1, CH_3 and CH_A4 (the enum skips CH_2 in use).
  static const int kCh[3] = {NPP_CH_1, NPP_CH_3, NPP_CH_A4};
  for (const auto &cf : cfgs) {
    for (int ci = 0; ci < 3 && !failed; ++ci) {
      for (int di = 0; di < 4 && !failed; ++di) {
        uint8_t thr[16], val[16];
        std::memset(thr, 0, sizeof(thr));
        std::memset(val, 0, sizeof(val));
        if (di == 0) {
          Npp8u t[4] = {40, 70, 100, 0}, v[4] = {200, 210, 220, 0};
          std::memcpy(thr, t, 4);
          std::memcpy(val, v, 4);
        } else if (di == 1) {
          Npp16u t[4] = {1000, 2000, 3000, 0}, v[4] = {40000, 41000, 42000, 0};
          std::memcpy(thr, t, 8);
          std::memcpy(val, v, 8);
        } else if (di == 2) {
          Npp16s t[4] = {-100, 200, 1000, 0}, v[4] = {3000, 3100, 3200, 0};
          std::memcpy(thr, t, 8);
          std::memcpy(val, v, 8);
        } else {
          Npp32f t[4] = {0.25f, 0.5f, 0.75f, 0.0f}, v[4] = {100.0f, 200.0f, 300.0f, 0.0f};
          std::memcpy(thr, t, 16);
          std::memcpy(val, v, 16);
        }
        const GenCase gc{cf.w, cf.h, ci, 0, 0, kSmall, di};
        bool ok = false;
        auto invoke = [&](auto typeTag) {
          using T = decltype(typeTag);
          const int PHY = ci == 0 ? 1 : (ci == 1 ? 3 : 4);
          const int rowBytes = cf.w * PHY * (int)sizeof(T);
          Buf<uint8_t> src1(rowBytes, cf.h, 1, 0, 0x5A);
          Buf<uint8_t> src2(rowBytes, cf.h, 1, 0, 0xA5);
          for (int y = 0; y < cf.h; ++y)
            for (int x = 0; x < rowBytes / (int)sizeof(T); ++x) {
              const T v1 = (T)((x * 37 + y * 11) % 197);
              const T v2 = (T)((x * 13 + y * 29 + 7) % 191);
              std::memcpy(src1.host.data() + (size_t)y * src1.step + (size_t)x * sizeof(T), &v1, sizeof(T));
              std::memcpy(src2.host.data() + (size_t)y * src2.step + (size_t)x * sizeof(T), &v2, sizeof(T));
            }
          src1.dev.copyFromHost(src1.host);
          src2.dev.copyFromHost(src2.host);
          Buf<uint8_t> dst(rowBytes, cf.h, 1, 0, 0x5A);
          dst.dev.copyFromHost(dst.host);
          const std::vector<uint8_t> pre1 = src1.host;
          const std::vector<uint8_t> preD = dst.host;
          npp_diff::clear_cuda_error();
          NppStatus st;
          if constexpr (Inplace) {
            st = call(ctx, kDt[di], kCh[ci], src1.dptr(), src1.step, src2.dptr(), src2.step, NppiSize{cf.w, cf.h}, thr, val);
          } else {
            st = call(ctx, kDt[di], kCh[ci], src1.dptr(), src1.step, src2.dptr(), src2.step, dst.dptr(), dst.step,
                      NppiSize{cf.w, cf.h}, thr, val);
          }
          if (st != NPP_NO_ERROR) {
            ++cnt.skipped;
            if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
            ok = true;
            return;
          }
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "[" << id << "] sync";
          ++cnt.executed;
          Buf<uint8_t> &res = Inplace ? src1 : dst;
          res.dev.copyToHost(res.host);
          const std::vector<uint8_t> &pre = Inplace ? pre1 : preD;
          if (ci == 0)
            ok = check_case<T, 1, 1>(id, gc, pre1.data(), src1.step, src2.host.data(), src2.step, res.host.data(),
                                     res.step, thr, val, cf.w, cf.h, pre, cnt, first);
          else if (ci == 1)
            ok = check_case<T, 3, 3>(id, gc, pre1.data(), src1.step, src2.host.data(), src2.step, res.host.data(),
                                     res.step, thr, val, cf.w, cf.h, pre, cnt, first);
          else
            ok = check_case<T, 4, 3>(id, gc, pre1.data(), src1.step, src2.host.data(), src2.step, res.host.data(),
                                     res.step, thr, val, cf.w, cf.h, pre, cnt, first);
        };
        if (di == 0) invoke(Npp8u{});
        else if (di == 1) invoke(Npp16u{});
        else if (di == 2) invoke(Npp16s{});
        else invoke(Npp32f{});
        if (!ok) failed = true;
        if (cnt.mismatched >= 8) failed = true;
      }
    }
  }
  finish(id, cnt, first);
}

}  // namespace fusedabs
}  // namespace num
