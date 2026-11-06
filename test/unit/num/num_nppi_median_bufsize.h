// CPU-golden coverage for the median scratch-size helpers
// (nppi_filtering_functions.h):
//   nppiFilterMedianGetBufferSize and nppiFilterMedianBorderGetBufferSize.
//
// Measured NPP 13.4 behaviour (RTX 5090):
//   * the size is ROI-independent and equals 49152 * elementSize * maskW *
//     maskH on the "buffered" kernel path, 0 otherwise;
//   * the border variant is buffered iff maskW*maskH*elementSize > 192;
//   * the plain variant's dispatch is the 16-config probe table copied into
//     num_nppi_median_bufsize_table.h (bits (w-1) of row (h-1));
//   * the value is written only when the call succeeds (the sentinel below
//     catches a missing write).
#pragma once

#include "num_nppi_geom.h"
#include "num_nppi_median_bufsize_table.h"

namespace num {
namespace mediabufdrv {

// DT: 0 = 8u, 1 = 16u, 2 = 16s, 3 = 32f.  A == 3 with C == 4 selects AC4.
template <int DT, int C, int A, bool Border, class Call>
void run_median_bufsize(const char *id, Call call) {
  NppStreamContext ctx{};
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    GTEST_SKIP() << "[" << id << "] no stream context";
    return;
  }
  const int layout = (C == 4 && A == 3) ? 3 : (C == 4 ? 2 : (C == 3 ? 1 : 0));
  const int idx = DT * 4 + layout;
  const int esize = DT == 0 ? 1 : (DT == 3 ? 4 : 2);
  const int dim = num::mediabuf::kMedianBufMaxDim;
  RunCounts cnt;
  std::string first;
  const NppiSize rois[] = {{33, 25}, {7, 7}};
  for (const NppiSize &roi : rois) {
    for (int mw = 1; mw <= dim; ++mw)
      for (int mh = 1; mh <= dim; ++mh) {
        const bool buffered =
            Border ? ((long long)mw * mh * esize > 192)
                   : (((num::mediabuf::kMedianPlainBufMask[idx][mh - 1] >> (mw - 1)) & 1ull) != 0);
        const long long expVal = buffered ? 49152ll * esize * mw * mh : 0;
        Npp32u size = 0xDEADBEEFu;
        const NppStatus st = call(roi, NppiSize{mw, mh}, &size,
                                  Border ? (int)NPP_BORDER_REPLICATE : 0, ctx);
        if (st != NPP_NO_ERROR) {
          ++cnt.skipped;
          if (st == NPP_NOT_IMPLEMENTED_ERROR) ++cnt.notimpl;
          continue;
        }
        ++cnt.executed;
        ++cnt.compared;
        if (size != (Npp32u)expVal) {
          char b[256];
          std::snprintf(b, sizeof(b), "[%s] roi=%dx%d mask=%dx%d got=%u want=%lld", id, roi.width,
                        roi.height, mw, mh, size, expVal);
          if (first.empty()) first = b;
          EXPECT_TRUE(false) << b;
          ++cnt.mismatched;
          if (cnt.mismatched >= 8) break;
        }
      }
  }
  finish(id, cnt, first);
}

}  // namespace mediabufdrv
}  // namespace num
