#pragma once

// Shared helpers for the WarpAffine family differential tests (package g1).
// Suites: NppiWarpAffineDiff / NppiWarpAffineBackDiff / NppiWarpAffineQuadDiff
//          NppiWarpAffineBatchDiff / NppiWarpAffineBatchInitDiff
//
// Every case allocates a canvas that covers both ROIs, fills src and dst with
// deterministic random data (so untouched padding / alpha channels are still
// compared), runs the _Ctx entry point and records status + dst buffer(s).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace warp_affine_diff {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

inline NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

inline int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp16f>) return ValueKind::F16;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp64f>) return ValueKind::F64;
  else return ValueKind::Bytes;
}

// Shrink random integer inputs into a small magnitude range (keeps 32s linear
// interpolation out of the int32 overflow region).
template <typename T> void shrink_mod(std::vector<T> &data, int mod) {
  if (mod <= 0) {
    return;
  }
  if constexpr (std::is_integral_v<T>) {
    for (auto &v : data) {
      v = static_cast<T>(static_cast<long long>(v) % mod);
    }
  }
}

// ---- Packed single-image warp (nppiWarpAffine / Back: shared signature) ----

template <typename T, typename F>
void warp_packed(const char *fname, const std::string &id, int width, int height, int chans, int pad,
                 const double coeff[2][3], int interp, NppiRect srcRoi, NppiRect dstRoi, F fn, uint64_t seed,
                 int mod = 0, double rtol = 1e-5, double atol = 1e-6) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(width * chans * es + pad, es);
  const int dStep = align_step(width * chans * es + pad, es);
  const size_t se = (size_t)height * sStep / es;
  const size_t de = (size_t)height * dStep / es;
  std::vector<T> hs = make_uniform<T>(se, seed + 11);
  std::vector<T> hd = make_uniform<T>(de, seed + 29);
  shrink_mod(hs, mod);
  DeviceMemory<T> &s = arena<T>(se);
  DeviceMemory<T> &d = arena<T>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st =
      fn(s.get(), NppiSize{width, height}, sStep, srcRoi, d.get(), dStep, dstRoi, coeff, interp, current_ctx());
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), de, out_kind<T>(), rtol, atol);
  c.finish();
}

// Error-path variant: optionally null src/dst pointers.
template <typename T, typename F>
void warp_packed_err(const char *fname, const std::string &id, int width, int height, int chans,
                     const double coeff[2][3], int interp, NppiRect srcRoi, NppiRect dstRoi, bool nullSrc,
                     bool nullDst, F fn, uint64_t seed) {
  const int es = (int)sizeof(T);
  const int step = align_step(width * chans * es, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<T> hs = make_uniform<T>(ne, seed + 11);
  DeviceMemory<T> &s = arena<T>(ne);
  DeviceMemory<T> &d = arena<T>(ne);
  s.copyFromHost(hs);
  d.copyFromHost(std::vector<T>(ne, T(0)));
  clear_cuda_error();
  NppStatus st = fn(nullSrc ? nullptr : s.get(), NppiSize{width, height}, step, srcRoi,
                    nullDst ? nullptr : d.get(), step, dstRoi, coeff, interp, current_ctx());
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(nullDst ? nullptr : d.get(), ne, out_kind<T>());
  c.finish();
}

// ---- Planar single-image warp ----

template <typename T, typename F>
void warp_planar(const char *fname, const std::string &id, int width, int height, int nplanes, int pad,
                 const double coeff[2][3], int interp, NppiRect srcRoi, NppiRect dstRoi, F fn, uint64_t seed,
                 int mod = 0, double rtol = 1e-5, double atol = 1e-6) {
  const int es = (int)sizeof(T);
  const int step = align_step(width * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  const T *srcP[4] = {nullptr, nullptr, nullptr, nullptr};
  T *dstP[4] = {nullptr, nullptr, nullptr, nullptr};
  std::vector<std::vector<T>> hs(nplanes);
  for (int p = 0; p < nplanes; ++p) {
    hs[p] = make_uniform<T>(ne, seed + 11 + 7 * p);
    shrink_mod(hs[p], mod);
    DeviceMemory<T> &s = arena<T>(ne);
    s.copyFromHost(hs[p]);
    srcP[p] = s.get();
    std::vector<T> hd = make_uniform<T>(ne, seed + 29 + 7 * p);
    DeviceMemory<T> &d = arena<T>(ne);
    d.copyFromHost(hd);
    dstP[p] = d.get();
  }
  clear_cuda_error();
  NppStatus st =
      fn(srcP, NppiSize{width, height}, step, srcRoi, dstP, step, dstRoi, coeff, interp, current_ctx());
  Case c(id, fname);
  for (int p = 0; p < nplanes; ++p) {
    c.input(hs[p].data(), hs[p].size() * sizeof(T));
  }
  c.status(st);
  for (int p = 0; p < nplanes; ++p) {
    c.device_output<T>(dstP[p], ne, out_kind<T>(), rtol, atol);
  }
  c.finish();
}

// ---- Packed planar-quad warp (nppiWarpAffineQuad) ----

template <typename T, typename F>
void quad_packed(const char *fname, const std::string &id, int width, int height, int chans, int pad,
                 const double srcQuad[4][2], const double dstQuad[4][2], int interp, NppiRect srcRoi,
                 NppiRect dstRoi, F fn, uint64_t seed, int mod = 0, double rtol = 1e-5, double atol = 1e-6) {
  const int es = (int)sizeof(T);
  const int step = align_step(width * chans * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<T> hs = make_uniform<T>(ne, seed + 11);
  std::vector<T> hd = make_uniform<T>(ne, seed + 29);
  shrink_mod(hs, mod);
  DeviceMemory<T> &s = arena<T>(ne);
  DeviceMemory<T> &d = arena<T>(ne);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = fn(s.get(), NppiSize{width, height}, step, srcRoi, srcQuad, d.get(), step, dstRoi, dstQuad, interp,
                    current_ctx());
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), ne, out_kind<T>(), rtol, atol);
  c.finish();
}

// ---- Planar quad warp ----

template <typename T, typename F>
void quad_planar(const char *fname, const std::string &id, int width, int height, int nplanes, int pad,
                 const double srcQuad[4][2], const double dstQuad[4][2], int interp, NppiRect srcRoi,
                 NppiRect dstRoi, F fn, uint64_t seed, int mod = 0, double rtol = 1e-5, double atol = 1e-6) {
  const int es = (int)sizeof(T);
  const int step = align_step(width * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  const T *srcP[4] = {nullptr, nullptr, nullptr, nullptr};
  T *dstP[4] = {nullptr, nullptr, nullptr, nullptr};
  std::vector<std::vector<T>> hs(nplanes);
  for (int p = 0; p < nplanes; ++p) {
    hs[p] = make_uniform<T>(ne, seed + 11 + 7 * p);
    shrink_mod(hs[p], mod);
    DeviceMemory<T> &s = arena<T>(ne);
    s.copyFromHost(hs[p]);
    srcP[p] = s.get();
    std::vector<T> hd = make_uniform<T>(ne, seed + 29 + 7 * p);
    DeviceMemory<T> &d = arena<T>(ne);
    d.copyFromHost(hd);
    dstP[p] = d.get();
  }
  clear_cuda_error();
  NppStatus st = fn(srcP, NppiSize{width, height}, step, srcRoi, srcQuad, dstP, step, dstRoi, dstQuad, interp,
                    current_ctx());
  Case c(id, fname);
  for (int p = 0; p < nplanes; ++p) {
    c.input(hs[p].data(), hs[p].size() * sizeof(T));
  }
  c.status(st);
  for (int p = 0; p < nplanes; ++p) {
    c.device_output<T>(dstP[p], ne, out_kind<T>(), rtol, atol);
  }
  c.finish();
}

// ---- nppiWarpAffineBatch: per-image forward (src -> dst) coefficients ----

inline void batch_entry(NppiWarpAffineBatchCXR &e, const void *src, int srcStep, void *dst, int dstStep,
                        Npp64f *coeffs) {
  e.pSrc = src;
  e.nSrcStep = srcStep;
  e.pDst = dst;
  e.nDstStep = dstStep;
  e.pCoeffs = coeffs;
  for (int r = 0; r < 2; ++r) {
    for (int k = 0; k < 3; ++k) {
      e.aTransformedCoeffs[r][k] = -12345.0;  // sentinel: detect "left untouched"
    }
  }
}

template <typename T, typename F>
void warp_batch(const char *fname, const std::string &id, int width, int height, int chans, int pad,
                const std::vector<std::array<double, 6>> &coeffs, int interp, NppiRect srcRoi, NppiRect dstRoi,
                F fn, uint64_t seed, int mod = 0, double rtol = 1e-5, double atol = 1e-6) {
  const int n = (int)coeffs.size();
  const int es = (int)sizeof(T);
  const int step = align_step(width * chans * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<const T *> srcP(n);
  std::vector<T *> dstP(n);
  std::vector<std::vector<T>> hs(n);
  for (int i = 0; i < n; ++i) {
    hs[i] = make_uniform<T>(ne, seed + 11 + 7 * i);
    shrink_mod(hs[i], mod);
    DeviceMemory<T> &s = arena<T>(ne);
    s.copyFromHost(hs[i]);
    srcP[i] = s.get();
    std::vector<T> hd = make_uniform<T>(ne, seed + 29 + 7 * i);
    DeviceMemory<T> &d = arena<T>(ne);
    d.copyFromHost(hd);
    dstP[i] = d.get();
  }
  std::vector<Npp64f> hc(6 * n);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 6; ++k) {
      hc[6 * i + k] = coeffs[i][k];
    }
  }
  DeviceMemory<Npp64f> &co = arena<Npp64f>(6 * n);
  co.copyFromHost(hc);
  std::vector<NppiWarpAffineBatchCXR> hl(n);
  for (int i = 0; i < n; ++i) {
    batch_entry(hl[i], srcP[i], step, dstP[i], step, co.get() + 6 * i);
  }
  DeviceMemory<NppiWarpAffineBatchCXR> &bl = arena<NppiWarpAffineBatchCXR>(n);
  bl.copyFromHost(hl);
  NppStreamContext ctx = current_ctx();
  clear_cuda_error();
  NppStatus stInit = nppiWarpAffineBatchInit_Ctx(bl.get(), (unsigned)n, ctx);
  EXPECT_EQ(stInit, NPP_NO_ERROR);
  NppStatus st = fn(NppiSize{width, height}, srcRoi, dstRoi, interp, bl.get(), (unsigned)n, ctx);
  Case c(id, fname);
  for (int i = 0; i < n; ++i) {
    c.input(hs[i].data(), hs[i].size() * sizeof(T));
  }
  c.status(st);
  for (int i = 0; i < n; ++i) {
    c.device_output<T>(dstP[i], ne, out_kind<T>(), rtol, atol);
  }
  c.finish();
}

// ---- nppiWarpAffineBatchInit: record the transformed coefficients ----

inline void batch_init_case(const char *fname, const std::string &id,
                            const std::vector<std::array<double, 6>> &coeffs, uint64_t seed) {
  const int n = (int)coeffs.size();
  std::vector<Npp64f> hc = make_uniform<Npp64f>(6 * n, seed + 11);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 6; ++k) {
      hc[6 * i + k] = coeffs[i][k];
    }
  }
  DeviceMemory<Npp64f> &co = arena<Npp64f>(6 * n);
  co.copyFromHost(hc);
  std::vector<NppiWarpAffineBatchCXR> hl(n);
  for (int i = 0; i < n; ++i) {
    batch_entry(hl[i], nullptr, 0, nullptr, 0, co.get() + 6 * i);
  }
  DeviceMemory<NppiWarpAffineBatchCXR> &bl = arena<NppiWarpAffineBatchCXR>(n);
  bl.copyFromHost(hl);
  clear_cuda_error();
  NppStatus st = nppiWarpAffineBatchInit_Ctx(bl.get(), (unsigned)n, current_ctx());
  std::vector<NppiWarpAffineBatchCXR> back(n);
  bl.copyToHost(back);
  Case c(id, fname);
  c.status(st);
  for (int i = 0; i < n; ++i) {
    c.host_output<double>(&back[i].aTransformedCoeffs[0][0], 6, ValueKind::F64, 1e-12, 1e-12);
  }
  c.finish();
}

}  // namespace warp_affine_diff
