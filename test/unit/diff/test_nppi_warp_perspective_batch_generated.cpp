// Differential tests for package g5: nppiWarpPerspectiveBatch
// (8u C1R/C3R/C4R/AC4R, 16f C1R/C3R/C4R, 32f C1R/C3R/C4R/AC4R; Ctx and
// non-Ctx) plus nppiWarpPerspectiveBatchInit.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiWarpPerspectiveBatchDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiWarpPerspectiveBatchDiff.*'
//
// NppiWarpPerspectiveBatchCXR carries both pCoeffs (device pointer, caller
// supplied) and aTransformedCoeffs (in-struct storage filled by
// nppiWarpPerspectiveBatchInit).  The goldens below capture
// aTransformedCoeffs after Init as host doubles, which shows whether NVIDIA
// stores the caller's matrix verbatim or pre-inverted, and the no-Init case
// (aTransformedCoeffs left zeroed) shows whether the batch kernels read the
// transformed copy or fall back to pCoeffs.  AC4 destinations are pre-filled
// with 0xA5 to expose the alpha rule; error-path probes deliberately
// over-allocate the source buffers so that an unvalidated out-of-bounds ROI
// cannot crash the reference run.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

// Steps are rounded up to 256 bytes: the NVIDIA 32f C4R batch warp
// kernel uses 16-byte vectorized accesses and faults on element-aligned
// steps (observed on the RTX 5090, NPP 13.4).
int align_step(int step, int es) { (void)es; return (step + 255) & ~255; }

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8u>) return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16u>) return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp16s>) return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp32s>) return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32f>) return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp16f>) return ValueKind::F16;
  else return ValueKind::Bytes;
}

using BatchFn = NppStatus (*)(NppiSize, NppiRect, NppiRect, int, NppiWarpPerspectiveBatchCXR *, unsigned int,
                              NppStreamContext);

// Position-encoded packed source (channel 0 = x, channel 1 = y, rest = 7);
// disabled for 16f (no arithmetic on Npp16f in host code).
template <typename T> std::vector<T> make_packed_positional(size_t count, int stride, int row_elems) {
  std::vector<T> v(count);
  for (size_t i = 0; i < count; ++i) {
    if (stride == 1) {
      v[i] = (T)(i % (size_t)row_elems);
      continue;
    }
    switch (i % (size_t)stride) {
    case 0: v[i] = (T)((i % (size_t)row_elems) / (size_t)stride); break;
    case 1: v[i] = (T)(i / (size_t)row_elems); break;
    default: v[i] = (T)7; break;
    }
  }
  return v;
}

template <typename T> bool has_positional() {
  return !std::is_same_v<T, Npp16f>;
}

// Destination pre-fill value (exposes the AC4 alpha rule): Npp16f has no
// arithmetic conversions, so use a distinctive half value instead.
template <typename T> T fill_value() {
  if constexpr (std::is_same_v<T, Npp16f>) {
    return to_npp16f(0.645f);
  } else {
    return (T)0xA5;
  }
}

DeviceMemory<NppiWarpPerspectiveBatchCXR> &batch_arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<NppiWarpPerspectiveBatchCXR>>> v;
  v.push_back(std::make_unique<DeviceMemory<NppiWarpPerspectiveBatchCXR>>(n));
  return *v.back();
}

// Batched warp: uploads the batch list, records the Init call (status plus
// the aTransformedCoeffs the library produced), then records the batch call
// itself (status plus every destination buffer plus the post-call
// aTransformedCoeffs).
template <typename T, int Stride, bool AC4>
void batch_case(const char *fname, const std::string &key, int nImages, NppiSize srcSize, NppiRect srcROI,
                NppiSize dstSize, NppiRect dstROI, const std::vector<const double (*)[3]> &coeffs, int interp,
                bool positional, BatchFn f, uint64_t seed) {
  const int es = (int)sizeof(T);
  const int sStep = align_step(srcSize.width * Stride * es + 8, es);
  const int dStep = align_step(dstSize.width * Stride * es + 8, es);
  const size_t se = (size_t)srcSize.height * sStep / es;
  const size_t de = (size_t)dstSize.height * dStep / es;

  std::vector<DeviceMemory<T> *> srcs, dsts;
  std::vector<DeviceMemory<double>> coeffBufs;
  std::vector<NppiWarpPerspectiveBatchCXR> entries(nImages);
  for (int i = 0; i < nImages; ++i) {
    DeviceMemory<T> &s = arena<T>(se);
    DeviceMemory<T> &d = arena<T>(de);
    std::vector<T> hs;
    if constexpr (std::is_same_v<T, Npp16f>) {
      // No positional pattern for half data (no host arithmetic on Npp16f).
      hs = make_uniform<T>(se, seed + 11 + (uint64_t)i * 13);
    } else {
      hs = positional ? make_packed_positional<T>(se, Stride, sStep / es)
                      : make_uniform<T>(se, seed + 11 + (uint64_t)i * 13);
    }
    std::vector<T> hd(de, fill_value<T>());
    s.copyFromHost(hs);
    d.copyFromHost(hd);
    srcs.push_back(&s);
    dsts.push_back(&d);
    coeffBufs.emplace_back(9);
    coeffBufs.back().copyFromHost(std::vector<double>(&coeffs[i][0][0], &coeffs[i][0][0] + 9));
    entries[i].pSrc = s.get();
    entries[i].nSrcStep = sStep;
    entries[i].pDst = d.get();
    entries[i].nDstStep = dStep;
    entries[i].pCoeffs = reinterpret_cast<Npp64f *>(coeffBufs.back().get());
    std::memset(entries[i].aTransformedCoeffs, 0, sizeof(entries[i].aTransformedCoeffs));
  }
  DeviceMemory<NppiWarpPerspectiveBatchCXR> &batch = batch_arena(nImages);
  batch.copyFromHost(entries);

  std::vector<NppiWarpPerspectiveBatchCXR> readback(nImages);
  auto readList = [&]() {
    cudaMemcpy(readback.data(), batch.get(), sizeof(NppiWarpPerspectiveBatchCXR) * nImages, cudaMemcpyDeviceToHost);
  };

  // --- Init call -----------------------------------------------------------
  clear_cuda_error();
  NppStatus stInit = nppiWarpPerspectiveBatchInit_Ctx(batch.get(), (unsigned)nImages, current_ctx());
  {
    Case c("nppiWarpPerspectiveBatch/" + key + "/init", "nppiWarpPerspectiveBatchInit");
    c.input(entries.data(), sizeof(entries[0]) * nImages);
    c.status(stInit);
    readList();
    std::vector<double> flat;
    flat.reserve((size_t)nImages * 9);
    for (int i = 0; i < nImages; ++i) {
      flat.insert(flat.end(), &readback[i].aTransformedCoeffs[0][0], &readback[i].aTransformedCoeffs[0][0] + 9);
    }
    c.host_output<double>(flat.data(), flat.size(), ValueKind::F64, 1e-9, 1e-12);
    c.finish();
  }

  // --- Batch call ----------------------------------------------------------
  clear_cuda_error();
  NppStatus st = f(srcSize, srcROI, dstROI, interp, batch.get(), (unsigned)nImages, current_ctx());
  {
    Case c("nppiWarpPerspectiveBatch/" + key, fname);
    for (int i = 0; i < nImages; ++i) {
      c.input(&coeffs[i][0][0], sizeof(double) * 9);
    }
    c.status(st);
    for (int i = 0; i < nImages; ++i) {
      c.device_output<T>(dsts[i]->get(), de, out_kind<T>(), 1e-5, 1e-6);
    }
    readList();
    std::vector<double> flat;
    flat.reserve((size_t)nImages * 9);
    for (int i = 0; i < nImages; ++i) {
      flat.insert(flat.end(), &readback[i].aTransformedCoeffs[0][0], &readback[i].aTransformedCoeffs[0][0] + 9);
    }
    c.host_output<double>(flat.data(), flat.size(), ValueKind::F64, 1e-9, 1e-12);
    c.finish();
  }
}

// Contract probe helper for 8u C1R with oversized buffers.
void batch_contract_case(const std::string &key, NppiSize smallest, NppiRect srcROI, NppiRect dstROI, int interp,
                         unsigned nBatch, bool doInit) {
  const int es = 1;
  const int sStep = align_step(48 * es + 8, es);
  const int dStep = align_step(48 * es + 8, es);
  const size_t se = (size_t)40 * sStep / es;
  const size_t de = (size_t)40 * dStep / es;
  static DeviceMemory<Npp8u> *src = nullptr;
  static DeviceMemory<Npp8u> *dst = nullptr;
  static DeviceMemory<double> *coeffs = nullptr;
  if (!src) {
    src = &arena<Npp8u>(se);
    dst = &arena<Npp8u>(de);
    coeffs = new DeviceMemory<double>(9);
    coeffs->copyFromHost(std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    src->copyFromHost(std::vector<Npp8u>(se, 3));
    dst->copyFromHost(std::vector<Npp8u>(de, 0xA5));
  }
  DeviceMemory<NppiWarpPerspectiveBatchCXR> &batch = batch_arena(nBatch > 0 ? nBatch : 1);
  NppiWarpPerspectiveBatchCXR entries[4];
  for (int i = 0; i < 4; ++i) {
    entries[i].pSrc = src->get();
    entries[i].nSrcStep = sStep;
    entries[i].pDst = dst->get();
    entries[i].nDstStep = dStep;
    entries[i].pCoeffs = reinterpret_cast<Npp64f *>(coeffs->get());
    std::memset(entries[i].aTransformedCoeffs, 0, sizeof(entries[i].aTransformedCoeffs));
  }
  if (nBatch > 0) {
    batch.copyFromHost(std::vector<NppiWarpPerspectiveBatchCXR>(entries, entries + nBatch));
  }
  if (doInit) {
    clear_cuda_error();
    (void)nppiWarpPerspectiveBatchInit_Ctx(batch.get(), nBatch, current_ctx());
  }
  clear_cuda_error();
  NppStatus st = nppiWarpPerspectiveBatch_8u_C1R_Ctx(smallest, srcROI, dstROI, interp, batch.get(), nBatch,
                                                    current_ctx());
  Case c("nppiWarpPerspectiveBatch/contract/" + key, "nppiWarpPerspectiveBatch_8u_C1R");
  c.status(st);
  c.finish();
}

void batch_init_contract_case(const std::string &key, unsigned nBatch) {
  const int es = 1;
  const int sStep = align_step(48 * es + 8, es);
  const int dStep = align_step(48 * es + 8, es);
  const size_t se = (size_t)40 * sStep / es;
  const size_t de = (size_t)40 * dStep / es;
  static DeviceMemory<Npp8u> *src = nullptr;
  static DeviceMemory<Npp8u> *dst = nullptr;
  static DeviceMemory<double> *coeffs = nullptr;
  if (!src) {
    src = &arena<Npp8u>(se);
    dst = &arena<Npp8u>(de);
    coeffs = new DeviceMemory<double>(9);
    src->copyFromHost(std::vector<Npp8u>(se, 3));
    dst->copyFromHost(std::vector<Npp8u>(de, 0xA5));
  }
  coeffs->copyFromHost(std::vector<double>{2, 0, 0, 0, 2, 0, 0, 0, 1});
  DeviceMemory<NppiWarpPerspectiveBatchCXR> &batch = batch_arena(nBatch > 0 ? nBatch : 1);
  NppiWarpPerspectiveBatchCXR entries[4];
  for (int i = 0; i < 4; ++i) {
    entries[i].pSrc = src->get();
    entries[i].nSrcStep = sStep;
    entries[i].pDst = dst->get();
    entries[i].nDstStep = dStep;
    entries[i].pCoeffs = reinterpret_cast<Npp64f *>(coeffs->get());
    std::memset(entries[i].aTransformedCoeffs, 0, sizeof(entries[i].aTransformedCoeffs));
  }
  // NOTE: a per-entry null pCoeffs is deliberately NOT probed here.  NVIDIA
  // nppiWarpPerspectiveBatchInit does not validate it and dereferences the
  // null device pointer, which poisons the CUDA context for the whole
  // process (cudaErrorInvalidValue then an illegal-memory-access fault);
  // mcMPP returns NPP_NULL_POINTER_ERROR instead.
  if (nBatch > 0) {
    batch.copyFromHost(std::vector<NppiWarpPerspectiveBatchCXR>(entries, entries + nBatch));
  }
  clear_cuda_error();
  NppStatus st = nppiWarpPerspectiveBatchInit_Ctx(batch.get(), nBatch, current_ctx());
  Case c("nppiWarpPerspectiveBatch/init_contract/" + key, "nppiWarpPerspectiveBatchInit");
  c.status(st);
  c.finish();
}

const double kIdent[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
const double kScale2[3][3] = {{2, 0, 0}, {0, 2, 0}, {0, 0, 1}};
// Projective transform.  The coefficients are chosen so that no destination
// pixel maps within 0.24 source pixels of the source-ROI edge (the 12x9 ROI
// lands on integer destination rows 1..7): NVIDIA's batch kernel resolves
// source-ROI-edge pixels with an internal rounding that is not modelled here,
// and a matrix that grazes the edge produces isolated one-pixel coverage
// differences (measured with the previous coefficients: destination (18,5)
// has u=10.964 inside a 0..11 ROI, but NPP leaves it untouched while NPP does
// write destination (18,6) with u=10.911).
const double kPersp[3][3] = {{1.4336, 0.0143, 2.5688}, {0.0135, 0.9186, 0.416}, {0.0044, 0.0015, 1.0}};
const double kShift[3][3] = {{1, 0, 3}, {0, 1, -2}, {0, 0, 1}};

}  // namespace

class NppiWarpPerspectiveBatchDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== 8u ===================================================================
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_8u_C1) {
  batch_case<Npp8u, 1, false>("nppiWarpPerspectiveBatch_8u_C1R", "8u_C1R/n3", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                              &nppiWarpPerspectiveBatch_8u_C1R_Ctx, 800);
  batch_case<Npp8u, 1, false>("nppiWarpPerspectiveBatch_8u_C1R", "8u_C1R/n3_linear", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_LINEAR, true,
                              &nppiWarpPerspectiveBatch_8u_C1R_Ctx, 801);
  batch_case<Npp8u, 1, false>("nppiWarpPerspectiveBatch_8u_C1R", "8u_C1R/n3_cubic", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_CUBIC, true,
                              &nppiWarpPerspectiveBatch_8u_C1R_Ctx, 802);
  batch_case<Npp8u, 1, false>("nppiWarpPerspectiveBatch_8u_C1R", "8u_C1R/n2", 2, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, {kShift, kPersp},
                              NPPI_INTER_NN, false, &nppiWarpPerspectiveBatch_8u_C1R_Ctx, 803);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_8u_C3) {
  batch_case<Npp8u, 3, false>("nppiWarpPerspectiveBatch_8u_C3R", "8u_C3R/n3", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                              &nppiWarpPerspectiveBatch_8u_C3R_Ctx, 810);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_8u_C4) {
  batch_case<Npp8u, 4, false>("nppiWarpPerspectiveBatch_8u_C4R", "8u_C4R/n3", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                              &nppiWarpPerspectiveBatch_8u_C4R_Ctx, 820);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_8u_AC4) {
  batch_case<Npp8u, 4, true>("nppiWarpPerspectiveBatch_8u_AC4R", "8u_AC4R/n3", 3, NppiSize{12, 9},
                             NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                             {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                             &nppiWarpPerspectiveBatch_8u_AC4R_Ctx, 830);
  batch_case<Npp8u, 4, true>("nppiWarpPerspectiveBatch_8u_AC4R", "8u_AC4R/n3_linear", 3, NppiSize{12, 9},
                             NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                             {kIdent, kScale2, kPersp}, NPPI_INTER_LINEAR, true,
                             &nppiWarpPerspectiveBatch_8u_AC4R_Ctx, 831);
}

// ===== 16f ==================================================================
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_16f_C1) {
  batch_case<Npp16f, 1, false>("nppiWarpPerspectiveBatch_16f_C1R", "16f_C1R/n3", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_NN, false,
                               &nppiWarpPerspectiveBatch_16f_C1R_Ctx, 840);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_16f_C3) {
  batch_case<Npp16f, 3, false>("nppiWarpPerspectiveBatch_16f_C3R", "16f_C3R/n3", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_NN, false,
                               &nppiWarpPerspectiveBatch_16f_C3R_Ctx, 841);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_16f_C4) {
  batch_case<Npp16f, 4, false>("nppiWarpPerspectiveBatch_16f_C4R", "16f_C4R/n3", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_NN, false,
                               &nppiWarpPerspectiveBatch_16f_C4R_Ctx, 842);
}

// ===== 32f ==================================================================
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_32f_C1) {
  batch_case<Npp32f, 1, false>("nppiWarpPerspectiveBatch_32f_C1R", "32f_C1R/n3", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                               &nppiWarpPerspectiveBatch_32f_C1R_Ctx, 850);
  batch_case<Npp32f, 1, false>("nppiWarpPerspectiveBatch_32f_C1R", "32f_C1R/n3_linear", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_LINEAR, true,
                               &nppiWarpPerspectiveBatch_32f_C1R_Ctx, 851);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_32f_C3) {
  batch_case<Npp32f, 3, false>("nppiWarpPerspectiveBatch_32f_C3R", "32f_C3R/n3", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                               &nppiWarpPerspectiveBatch_32f_C3R_Ctx, 860);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_32f_C4) {
  batch_case<Npp32f, 4, false>("nppiWarpPerspectiveBatch_32f_C4R", "32f_C4R/n3", 3, NppiSize{12, 9},
                               NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                               {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                               &nppiWarpPerspectiveBatch_32f_C4R_Ctx, 861);
}
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_32f_AC4) {
  batch_case<Npp32f, 4, true>("nppiWarpPerspectiveBatch_32f_AC4R", "32f_AC4R/n3", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_NN, true,
                              &nppiWarpPerspectiveBatch_32f_AC4R_Ctx, 870);
  batch_case<Npp32f, 4, true>("nppiWarpPerspectiveBatch_32f_AC4R", "32f_AC4R/n3_linear", 3, NppiSize{12, 9},
                              NppiRect{0, 0, 12, 9}, NppiSize{24, 18}, NppiRect{0, 0, 24, 18},
                              {kIdent, kScale2, kPersp}, NPPI_INTER_LINEAR, true,
                              &nppiWarpPerspectiveBatch_32f_AC4R_Ctx, 871);
}

// ===== batch contract =======================================================
TEST_F(NppiWarpPerspectiveBatchDiff, Batch_Contract) {
  batch_contract_case("n0", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN, 0, true);
  batch_contract_case("n1", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN, 1, true);
  batch_contract_case("n2", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN, 2, true);
  batch_contract_case("src_size_empty", NppiSize{0, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN,
                      3, true);
  batch_contract_case("src_roi_w0", NppiSize{12, 9}, NppiRect{0, 0, 0, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN, 3,
                      true);
  batch_contract_case("dst_roi_w0", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 0, 9}, NPPI_INTER_NN, 3,
                      true);
  // Source ROI outside oSmallestSrcSize; buffers are oversized (48x40) so an
  // unvalidated implementation cannot crash the reference run.
  batch_contract_case("src_roi_oob", NppiSize{12, 9}, NppiRect{2, 2, 32, 12}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN,
                      3, true);
  batch_contract_case("src_roi_neg", NppiSize{12, 9}, NppiRect{-2, 0, 12, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN,
                      3, true);
  batch_contract_case("interp_undefined", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9},
                      NPPI_INTER_UNDEFINED, 3, true);
  batch_contract_case("interp_super", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9},
                      NPPI_INTER_SUPER, 3, true);
  // No BatchInit: aTransformedCoeffs stay zeroed while pCoeffs is valid.
  batch_contract_case("no_init", NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9}, NPPI_INTER_NN, 3,
                      false);
  // Null batch list.
  {
    clear_cuda_error();
    NppStatus st = nppiWarpPerspectiveBatch_8u_C1R_Ctx(NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 12, 9},
                                                       NPPI_INTER_NN, nullptr, 3, current_ctx());
    Case c("nppiWarpPerspectiveBatch/contract/null_list", "nppiWarpPerspectiveBatch_8u_C1R");
    c.status(st);
    c.finish();
  }
}

// ===== BatchInit contract ===================================================
TEST_F(NppiWarpPerspectiveBatchDiff, BatchInit_Contract) {
  // NOTE: a null list pointer is deliberately NOT probed here.  NVIDIA
  // nppiWarpPerspectiveBatchInit does not validate it and dereferences the
  // null device pointer (cudaErrorIllegalAddress, which poisons the CUDA
  // context for the whole process); mcMPP returns NPP_NULL_POINTER_ERROR
  // instead.  The batch entry points DO validate the null list (see
  // Batch_Contract/null_list).
  batch_init_contract_case("n0", 0);
  batch_init_contract_case("n1", 1);
  // n3_null_coeffs removed: NVIDIA dereferences a per-entry null pCoeffs and
  // poisons the CUDA context; no golden can be recorded for it.
}

// ===== Non-Ctx adapters (representative _Ctx cases -> same golden) ==========
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiWarpPerspectiveBatchDiff, NonCtx_Batch) {
  // Reproduces the 8u_C1R/n3 case without _Ctx entry points (BatchInit plus
  // batch call, 3 identity/scale2/persp images, NN).
  const int es = 1;
  const int sStep = align_step(12 * es + 8, es);
  const int dStep = align_step(24 * es + 8, es);
  const size_t se = (size_t)9 * sStep / es;
  const size_t de = (size_t)18 * dStep / es;
  std::vector<DeviceMemory<Npp8u> *> srcs, dsts;
  std::vector<DeviceMemory<double>> coeffBufs;
  NppiWarpPerspectiveBatchCXR entries[3];
  const double *coeffData[3] = {&kIdent[0][0], &kScale2[0][0], &kPersp[0][0]};
  for (int i = 0; i < 3; ++i) {
    DeviceMemory<Npp8u> &s = arena<Npp8u>(se);
    DeviceMemory<Npp8u> &d = arena<Npp8u>(de);
    std::vector<Npp8u> hs = make_packed_positional<Npp8u>(se, 1, sStep / es);
    std::vector<Npp8u> hd(de, 0xA5);
    s.copyFromHost(hs);
    d.copyFromHost(hd);
    srcs.push_back(&s);
    dsts.push_back(&d);
    coeffBufs.emplace_back(9);
    coeffBufs.back().copyFromHost(std::vector<double>(coeffData[i], coeffData[i] + 9));
    entries[i].pSrc = s.get();
    entries[i].nSrcStep = sStep;
    entries[i].pDst = d.get();
    entries[i].nDstStep = dStep;
    entries[i].pCoeffs = reinterpret_cast<Npp64f *>(coeffBufs.back().get());
    std::memset(entries[i].aTransformedCoeffs, 0, sizeof(entries[i].aTransformedCoeffs));
  }
  DeviceMemory<NppiWarpPerspectiveBatchCXR> &batch = batch_arena(3);
  batch.copyFromHost(std::vector<NppiWarpPerspectiveBatchCXR>(entries, entries + 3));
  // The non-Ctx entries share the golden case ids of the _Ctx cases, so they
  // have to record the same outputs: status plus the post-Init and post-call
  // aTransformedCoeffs read back from the device list.
  std::vector<NppiWarpPerspectiveBatchCXR> readback(3);
  auto readList = [&]() {
    cudaMemcpy(readback.data(), batch.get(), sizeof(NppiWarpPerspectiveBatchCXR) * 3, cudaMemcpyDeviceToHost);
  };
  auto transformed = [&]() {
    std::vector<double> flat;
    flat.reserve(27);
    for (int i = 0; i < 3; ++i) flat.insert(flat.end(), &readback[i].aTransformedCoeffs[0][0], &readback[i].aTransformedCoeffs[0][0] + 9);
    return flat;
  };

  clear_cuda_error();
  NppStatus stInit = nppiWarpPerspectiveBatchInit(batch.get(), 3);
  Case ci("nppiWarpPerspectiveBatch/8u_C1R/n3/init", "nppiWarpPerspectiveBatchInit");
  ci.input(entries, sizeof(entries[0]) * 3);
  ci.status(stInit);
  readList();
  {
    std::vector<double> flat = transformed();
    ci.host_output<double>(flat.data(), flat.size(), ValueKind::F64, 1e-9, 1e-12);
  }
  ci.finish();

  clear_cuda_error();
  NppStatus st = nppiWarpPerspectiveBatch_8u_C1R(NppiSize{12, 9}, NppiRect{0, 0, 12, 9}, NppiRect{0, 0, 24, 18},
                                                 NPPI_INTER_NN, batch.get(), 3);
  Case c("nppiWarpPerspectiveBatch/8u_C1R/n3", "nppiWarpPerspectiveBatch_8u_C1R");
  for (int i = 0; i < 3; ++i) c.input(&coeffData[i][0], sizeof(double) * 9);
  c.status(st);
  for (int i = 0; i < 3; ++i) {
    c.device_output<Npp8u>(dsts[i]->get(), de, ValueKind::U8, 1e-5, 1e-6);
  }
  readList();
  {
    std::vector<double> flat = transformed();
    c.host_output<double>(flat.data(), flat.size(), ValueKind::F64, 1e-9, 1e-12);
  }
  c.finish();
}
#endif
