/**
 * @file test_npp_stream_sync.cpp
 * @brief Repro for mppSetStream()/mppGetStream() desync (ZTE frame-skew issue).
 *
 * Customer flow (fd_zte.cpp):
 *   mcStreamCreateWithPriority(&stream, 1, 4); // flags=1: non-blocking stream
 *   mppSetStream(stream);
 *   mppGetStreamContext(&mppStreamCtx);   // <-- mppStreamCtx.hStream must equal stream
 *   mppiSet_32f_C1R_Ctx(..., mppStreamCtx);
 *   mppiCopy_32f_C3P3R_Ctx(..., mppStreamCtx);
 *   mcStreamSynchronize(stream);          // must wait for the MPP pre-processing
 *   session.Run(...);                     // inference on the same stream
 *
 * When mppSetStream() is a no-op and mppGetStreamContext() always reports the
 * NULL stream, the MPP kernels are queued on the legacy stream.  The customer
 * only synchronizes its own (non-blocking) stream, so inference starts before
 * pre-processing of the current frame is done and consumes the previous
 * frame's buffer -> boxes lag by one frame.
 *
 * The ordering tests below reproduce that deterministically: the legacy stream
 * is gated with an event that is only recorded after we have probed the buffer.
 * If the MPP kernel was submitted to the managed (customer) stream, it runs
 * before the probe copy; if it leaked onto the legacy stream, the probe copy
 * observes the untouched sentinel value.
 */

#include "../framework/npp_test_base.h"

#include <cuda_runtime.h>

#include <vector>

using namespace npp_functional_test;

namespace {

// Restores the process-wide NPP managed stream even when an ASSERT returns early.
struct ManagedStreamGuard {
  ~ManagedStreamGuard() { nppSetStream(nullptr); }
};

// Blocks the legacy (NULL) stream until release() is called.  release() uses a
// private stream so that recording the event can never deadlock against the
// wait itself.
class LegacyStreamGate {
public:
  LegacyStreamGate() {
    valid_ = cudaStreamCreateWithFlags(&releaseStream_, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) == cudaSuccess;
    // Work submitted later to the legacy stream must wait for event_.
    valid_ = valid_ && cudaStreamWaitEvent(nullptr, event_, 0) == cudaSuccess;
  }

  ~LegacyStreamGate() {
    // Never leave the legacy stream blocked, even if EXPECT_* failed early.
    if (!released_ && event_ != nullptr) {
      cudaEventRecord(event_, releaseStream_);
    }
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
    if (releaseStream_ != nullptr) {
      cudaStreamDestroy(releaseStream_);
    }
  }

  bool valid() const { return valid_; }

  void release() {
    if (!released_) {
      cudaEventRecord(event_, releaseStream_);
      released_ = true;
    }
  }

private:
  cudaStream_t releaseStream_ = nullptr;
  cudaEvent_t event_ = nullptr;
  bool released_ = false;
  bool valid_ = false;
};

struct TestBuffers {
  Npp32f *src = nullptr;
  Npp32f *d = nullptr;
  Npp32f *h = nullptr;
  cudaStream_t stream = nullptr;

  ~TestBuffers() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (h != nullptr) {
      cudaFreeHost(h);
    }
    if (d != nullptr) {
      cudaFree(d);
    }
    if (src != nullptr) {
      cudaFree(src);
    }
  }
};

struct U8Buffers {
  Npp8u *src = nullptr;
  Npp8u *mask = nullptr;
  Npp8u *dst = nullptr;
  Npp8u *h = nullptr;
  cudaStream_t stream = nullptr;

  ~U8Buffers() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (h != nullptr) {
      cudaFreeHost(h);
    }
    if (dst != nullptr) {
      cudaFree(dst);
    }
    if (mask != nullptr) {
      cudaFree(mask);
    }
    if (src != nullptr) {
      cudaFree(src);
    }
  }
};

struct WarpBatchBuffers {
  Npp64f *coeffs[2] = {nullptr, nullptr};
  NppiWarpPerspectiveBatchCXR *batch = nullptr;
  void *scratch = nullptr;
  cudaStream_t stream = nullptr;

  ~WarpBatchBuffers() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (scratch != nullptr) {
      cudaFree(scratch);
    }
    if (batch != nullptr) {
      cudaFree(batch);
    }
    for (Npp64f *&coeff : coeffs) {
      if (coeff != nullptr) {
        cudaFree(coeff);
      }
    }
  }
};

// Captures a Ctx entry point on a fresh non-blocking stream and checks that
// it neither blocks the host nor falls back to a non-capturable legacy-stream
// operation.
template <typename Fn>
static void expectGraphCapturable(const char *opName, const Fn &fn) {
  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess) << opName;

  NppStreamContext ctx{};
  ctx.hStream = stream;
  cudaGraph_t graph = nullptr;
  cudaError_t begin = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  cudaError_t end = begin;
  NppStatus status = NPP_ERROR;
  if (begin == cudaSuccess) {
    status = fn(ctx);
    end = cudaStreamEndCapture(stream, &graph);
  }

  size_t nodes = 0;
  if (graph != nullptr) {
    cudaGraphGetNodes(graph, nullptr, &nodes);
    cudaGraphDestroy(graph);
  }

  EXPECT_EQ(begin, cudaSuccess) << opName;
  EXPECT_EQ(status, NPP_SUCCESS) << opName << " cannot be captured";
  EXPECT_EQ(end, cudaSuccess) << opName;
  EXPECT_GT(nodes, 0u) << opName;
  cudaStreamDestroy(stream);
}

constexpr float kSentinel = -123.0f;
constexpr float kFillValue = 42.0f;

constexpr int kWidth = 1024;
constexpr int kHeight = 1024;
constexpr size_t kBytes = static_cast<size_t>(kWidth) * kHeight * sizeof(Npp32f);

} // namespace

class NppStreamSyncTest : public NppTestBase {};

// The managed stream must be observable through both nppGetStream() and
// nppGetStreamContext().  Before the fix both returned the default stream.
TEST_F(NppStreamSyncTest, SetStream_GetStreamAndContextStayInSync) {
  ManagedStreamGuard restoreStream;
  TestBuffers buffers;
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  ASSERT_EQ(nppSetStream(buffers.stream), NPP_NO_ERROR);
  EXPECT_EQ(nppGetStream(), buffers.stream)
      << "nppGetStream() did not return the stream passed to nppSetStream()";

  NppStreamContext ctx{};
  ASSERT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR);
  EXPECT_EQ(ctx.hStream, buffers.stream)
      << "nppGetStreamContext().hStream does not match the stream set by nppSetStream()";

  // Restore the default stream and verify the restore is observable too.
  EXPECT_EQ(nppSetStream(nullptr), NPP_NO_ERROR);
  EXPECT_EQ(nppGetStream(), static_cast<cudaStream_t>(nullptr));
}

// Mirrors the customer chain: set stream -> get context -> Ctx variant.
TEST_F(NppStreamSyncTest, GetStreamContext_CtxOperationIsOrderedWithUserStream) {
  ManagedStreamGuard restoreStream;
  TestBuffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.d), kBytes), cudaSuccess);
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void **>(&buffers.h), kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(buffers.d, 0, kBytes), cudaSuccess);
  buffers.h[0] = kSentinel;
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  LegacyStreamGate gate;
  ASSERT_TRUE(gate.valid()) << "failed to gate the legacy stream";

  ASSERT_EQ(nppSetStream(buffers.stream), NPP_NO_ERROR);

  NppStreamContext ctx{};
  ASSERT_EQ(nppGetStreamContext(&ctx), NPP_NO_ERROR);

  NppiSize roi = {kWidth, kHeight};
  NppStatus status = nppiSet_32f_C1R_Ctx(kFillValue, buffers.d, kWidth * sizeof(Npp32f), roi, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Probe the result through the customer stream only - exactly what the
  // demo does before handing the tensor to ORT.
  ASSERT_EQ(cudaMemcpyAsync(buffers.h, buffers.d, kBytes, cudaMemcpyDeviceToHost, buffers.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(buffers.stream), cudaSuccess);
  const float observed = buffers.h[0];

  // Unblock the legacy stream before asserting so a failure cannot hang the GPU.
  gate.release();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(observed, kFillValue)
      << "MPP kernel was not ordered with the stream returned by nppGetStreamContext(); "
         "inference would read the previous frame (observed sentinel " << kSentinel << ")";
}

// The NPP-managed-stream API (no Ctx argument) must honour nppSetStream() too.
// NPP 13 removed the non-Ctx entry points, so this only runs for our
// implementation (and older reference NPP releases).
#ifndef USE_NVIDIA_NPP_TESTS
TEST_F(NppStreamSyncTest, ManagedStream_NonCtxOperationIsOrderedWithUserStream) {
  ManagedStreamGuard restoreStream;
  TestBuffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.d), kBytes), cudaSuccess);
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void **>(&buffers.h), kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(buffers.d, 0, kBytes), cudaSuccess);
  buffers.h[0] = kSentinel;
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  LegacyStreamGate gate;
  ASSERT_TRUE(gate.valid()) << "failed to gate the legacy stream";

  ASSERT_EQ(nppSetStream(buffers.stream), NPP_NO_ERROR);

  NppiSize roi = {kWidth, kHeight};
  NppStatus status = nppiSet_32f_C1R(kFillValue, buffers.d, kWidth * sizeof(Npp32f), roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  ASSERT_EQ(cudaMemcpyAsync(buffers.h, buffers.d, kBytes, cudaMemcpyDeviceToHost, buffers.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(buffers.stream), cudaSuccess);
  const float observed = buffers.h[0];

  gate.release();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(observed, kFillValue)
      << "non-Ctx MPP entry point ignored the stream set by nppSetStream(); "
         "inference would read the previous frame (observed sentinel " << kSentinel << ")";
}
#endif // !USE_NVIDIA_NPP_TESTS

// Regression: the 3x3 fast paths of the morphology Ctx entry points launched
// kernels without a stream argument and therefore ignored the context stream.
TEST_F(NppStreamSyncTest, Morphology3x3_CtxOperationsAreOrderedWithUserStream) {
  constexpr int kSize = 32;
  constexpr int kCenter = (kSize / 2) * kSize + kSize / 2;
  constexpr Npp8u kSrcValue = 7;

  ManagedStreamGuard restoreStream;
  U8Buffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.src), kSize * kSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.mask), 9), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.dst), kSize * kSize), cudaSuccess);
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void **>(&buffers.h), 1), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  std::vector<Npp8u> src(kSize * kSize, kSrcValue);
  std::vector<Npp8u> mask(9, 1);
  ASSERT_EQ(cudaMemcpy(buffers.src, src.data(), src.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(buffers.mask, mask.data(), mask.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(buffers.dst, 0, kSize * kSize), cudaSuccess);

  LegacyStreamGate gate;
  ASSERT_TRUE(gate.valid()) << "failed to gate the legacy stream";

  NppStreamContext ctx{};
  ctx.hStream = buffers.stream;
  NppiSize roi = {kSize, kSize};
  NppiSize maskSize = {3, 3};
  NppiPoint anchor = {1, 1};

  ASSERT_EQ(nppiErode_8u_C1R_Ctx(buffers.src, kSize, buffers.dst, kSize, roi, buffers.mask, maskSize, anchor, ctx),
            NPP_SUCCESS);
  ASSERT_EQ(cudaMemcpyAsync(buffers.h, buffers.dst + kCenter, 1, cudaMemcpyDeviceToHost, buffers.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(buffers.stream), cudaSuccess);
  const Npp8u observedErode = buffers.h[0];

  // Reset on the same stream, then exercise the dilate 3x3 fast path.
  ASSERT_EQ(cudaMemsetAsync(buffers.dst, 0, kSize * kSize, buffers.stream), cudaSuccess);
  ASSERT_EQ(nppiDilate_8u_C1R_Ctx(buffers.src, kSize, buffers.dst, kSize, roi, buffers.mask, maskSize, anchor, ctx),
            NPP_SUCCESS);
  ASSERT_EQ(cudaMemcpyAsync(buffers.h, buffers.dst + kCenter, 1, cudaMemcpyDeviceToHost, buffers.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(buffers.stream), cudaSuccess);
  const Npp8u observedDilate = buffers.h[0];

  gate.release();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(observedErode, kSrcValue) << "nppiErode_8u_C1R_Ctx did not run on the context stream";
  EXPECT_EQ(observedDilate, kSrcValue) << "nppiDilate_8u_C1R_Ctx did not run on the context stream";
}

// Regression: the 3x3/5x5/7x7 fast paths of nppiFilterBox_32f_C1R_Ctx
// launched kernels without a stream argument and therefore ignored the
// context stream.
TEST_F(NppStreamSyncTest, FilterBox32f3x3_CtxOperationIsOrderedWithUserStream) {
  constexpr int kSize = 32;
  constexpr int kCenter = (kSize / 2) * kSize + kSize / 2;
  constexpr float kSrcValue = 5.0f;

  // The non-border FilterBox variant expects the source window to be
  // in-bounds; provide a one-pixel halo around the ROI.
  constexpr int kPadded = kSize + 2;

  ManagedStreamGuard restoreStream;
  TestBuffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.src), kPadded * kPadded * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.d), kSize * kSize * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void **>(&buffers.h), sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  std::vector<Npp32f> src(kPadded * kPadded, kSrcValue);
  ASSERT_EQ(cudaMemcpy(buffers.src, src.data(), src.size() * sizeof(Npp32f), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(buffers.d, 0, kSize * kSize * sizeof(Npp32f)), cudaSuccess);

  LegacyStreamGate gate;
  ASSERT_TRUE(gate.valid()) << "failed to gate the legacy stream";

  NppStreamContext ctx{};
  ctx.hStream = buffers.stream;
  NppiSize roi = {kSize, kSize};
  NppiSize maskSize = {3, 3};
  NppiPoint anchor = {1, 1};

  Npp32f *srcRoi = buffers.src + kPadded + 1;
  ASSERT_EQ(nppiFilterBox_32f_C1R_Ctx(srcRoi, kPadded * sizeof(Npp32f), buffers.d, kSize * sizeof(Npp32f), roi,
                                      maskSize, anchor, ctx),
            NPP_SUCCESS);
  ASSERT_EQ(cudaMemcpyAsync(buffers.h, buffers.d + kCenter, sizeof(Npp32f), cudaMemcpyDeviceToHost, buffers.stream),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(buffers.stream), cudaSuccess);
  const float observed = buffers.h[0];

  gate.release();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(observed, kSrcValue) << "nppiFilterBox_32f_C1R_Ctx 3x3 fast path did not run on the context stream";
}

// Regression: pure-kernel Ctx entry points used to synchronize the stream
// before returning, which blocks the host and makes the ops unusable inside a
// CUDA graph capture.
TEST_F(NppStreamSyncTest, ColorToGray_CtxIsGraphCapturable) {
  constexpr int kW = 256;
  constexpr int kH = 256;

  U8Buffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.src), static_cast<size_t>(kW) * kH * 3), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.dst), static_cast<size_t>(kW) * kH), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  NppStreamContext ctx{};
  ctx.hStream = buffers.stream;
  const NppiSize roi = {kW, kH};
  const Npp32f coeffs[3] = {0.299f, 0.587f, 0.114f};

  cudaGraph_t graph = nullptr;
  cudaError_t begin = cudaStreamBeginCapture(buffers.stream, cudaStreamCaptureModeThreadLocal);
  NppStatus status = NPP_ERROR;
  cudaError_t end = begin;
  if (begin == cudaSuccess) {
    status = nppiColorToGray_8u_C3C1R_Ctx(buffers.src, kW * 3, buffers.dst, kW, roi, coeffs, ctx);
    end = cudaStreamEndCapture(buffers.stream, &graph);
  }

  size_t nodes = 0;
  if (graph != nullptr) {
    cudaGraphGetNodes(graph, nullptr, &nodes);
    cudaGraphDestroy(graph);
  }

  EXPECT_EQ(begin, cudaSuccess);
  EXPECT_EQ(status, NPP_SUCCESS) << "Ctx entry point still synchronizes internally and cannot be captured";
  EXPECT_EQ(end, cudaSuccess);
  EXPECT_GT(nodes, 0u);
}

// Regression: the multi-channel Set implementations allocated a device buffer
// and synchronized the stream, making them synchronous and non-capturable.
TEST_F(NppStreamSyncTest, Set32fC3R_CtxIsGraphCapturable) {
  constexpr int kW = 256;
  constexpr int kH = 256;

  TestBuffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.d),
                       static_cast<size_t>(kW) * kH * 3 * sizeof(Npp32f)),
            cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);

  NppStreamContext ctx{};
  ctx.hStream = buffers.stream;
  const NppiSize roi = {kW, kH};
  const Npp32f values[3] = {1.0f, 2.0f, 3.0f};

  cudaGraph_t graph = nullptr;
  cudaError_t begin = cudaStreamBeginCapture(buffers.stream, cudaStreamCaptureModeThreadLocal);
  NppStatus status = NPP_ERROR;
  cudaError_t end = begin;
  if (begin == cudaSuccess) {
    status = nppiSet_32f_C3R_Ctx(values, buffers.d, kW * 3 * sizeof(Npp32f), roi, ctx);
    end = cudaStreamEndCapture(buffers.stream, &graph);
  }

  size_t nodes = 0;
  if (graph != nullptr) {
    cudaGraphGetNodes(graph, nullptr, &nodes);
    cudaGraphDestroy(graph);
  }

  EXPECT_EQ(begin, cudaSuccess);
  EXPECT_EQ(status, NPP_SUCCESS) << "nppiSet_32f_C3R_Ctx still allocates/synchronizes inside the call";
  EXPECT_EQ(end, cudaSuccess);
  EXPECT_GT(nodes, 0u);
}

// Regression: ColorTwist, Threshold and SwapChannels used to allocate a
// temporary device buffer and synchronize the stream inside the Ctx call.
TEST_F(NppStreamSyncTest, SmallParamCtxOps_AreGraphCapturable) {
  constexpr int kW = 256;
  constexpr int kH = 256;

  U8Buffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.src), static_cast<size_t>(kW) * kH * 3), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.dst), static_cast<size_t>(kW) * kH * 3), cudaSuccess);

  const NppiSize roi = {kW, kH};
  const Npp32f twist[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  const Npp8u thresholds[3] = {10, 20, 30};
  const Npp8u values[3] = {1, 2, 3};
  const int order[3] = {2, 1, 0};

  expectGraphCapturable("ColorTwist32f_8u_C3R", [&](const NppStreamContext &ctx) {
    return nppiColorTwist32f_8u_C3R_Ctx(buffers.src, kW * 3, buffers.dst, kW * 3, roi, twist, ctx);
  });
#ifndef USE_NVIDIA_NPP_TESTS
  // NVIDIA NPP 13's threshold Ctx implementation waits on an internal event
  // and is not reliably graph-capturable, so only assert this for our code.
  expectGraphCapturable("Threshold_LTVal_8u_C3R", [&](const NppStreamContext &ctx) {
    return nppiThreshold_LTVal_8u_C3R_Ctx(buffers.src, kW * 3, buffers.dst, kW * 3, roi, thresholds, values, ctx);
  });
#endif
  expectGraphCapturable("SwapChannels_8u_C3R", [&](const NppStreamContext &ctx) {
    return nppiSwapChannels_8u_C3R_Ctx(buffers.src, kW * 3, buffers.dst, kW * 3, roi, order, ctx);
  });
}

// Regression: these Ctx entry points allocate temporary device memory; it is
// now stream-ordered so the calls stay asynchronous and capturable.
TEST_F(NppStreamSyncTest, AllocBasedCtxOps_AreGraphCapturable) {
  constexpr int kW = 256;
  constexpr int kH = 256;

  U8Buffers buffers;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.src), static_cast<size_t>(kW) * kH * 3), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.dst), static_cast<size_t>(kW) * kH * 3), cudaSuccess);

  const NppiSize roi = {kW, kH};
  const int order[3] = {2, 1, 0};
  Npp32s lutValues[256] = {0};
  Npp32s lutLevels[256] = {0};
  Npp32s kernel[9] = {0};

  expectGraphCapturable("SwapChannels_8u_C3IR", [&](const NppStreamContext &ctx) {
    return nppiSwapChannels_8u_C3IR_Ctx(buffers.src, kW * 3, roi, order, ctx);
  });
  expectGraphCapturable("Mirror_8u_C3IR", [&](const NppStreamContext &ctx) {
    return nppiMirror_8u_C3IR_Ctx(buffers.src, kW * 3, roi, NPP_VERTICAL_AXIS, ctx);
  });
  expectGraphCapturable("LUT_8u_C1R", [&](const NppStreamContext &ctx) {
    return nppiLUT_8u_C1R_Ctx(buffers.src, kW * 3, buffers.dst, kW * 3, roi, lutValues, lutLevels, 256, ctx);
  });
  expectGraphCapturable("Filter_8u_C1R", [&](const NppStreamContext &ctx) {
    return nppiFilter_8u_C1R_Ctx(buffers.src, kW * 3, buffers.dst, kW * 3, roi, kernel, {3, 3}, {1, 1}, 1, ctx);
  });

  Npp32s *histDev = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&histDev), 255 * sizeof(Npp32s)), cudaSuccess);
  expectGraphCapturable("HistogramEven_8u_C1R", [&](const NppStreamContext &ctx) {
    return nppiHistogramEven_8u_C1R_Ctx(buffers.src, kW * 3, roi, histDev, 256, 0, 255, buffers.dst, ctx);
  });
  cudaFree(histDev);
}

// Regression: nppiWarpPerspectiveBatchInit_Ctx used the legacy stream for its
// host-side device copies, so batch descriptors produced on the context stream
// could be read before they were written.  NPP computes a transformed matrix
// (not a plain copy), so the check compares the delayed run against a
// synchronized reference run instead of hard-coding the expected coefficients.
TEST_F(NppStreamSyncTest, WarpPerspectiveBatchInit_CtxIsOrderedWithContextStream) {
  constexpr unsigned int kBatchSize = 2;
  constexpr size_t kCoeffBytes = 3 * 3 * sizeof(Npp64f);
  constexpr size_t kScratchBytes = 512ull * 1024 * 1024;

  static const Npp64f kStale[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
  static const Npp64f kFresh[3][3] = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};

  ManagedStreamGuard restoreStream;
  WarpBatchBuffers buffers;
  ASSERT_EQ(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.batch),
                       sizeof(NppiWarpPerspectiveBatchCXR) * kBatchSize),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&buffers.scratch, kScratchBytes), cudaSuccess);

  std::vector<NppiWarpPerspectiveBatchCXR> batchHost(kBatchSize);
  for (unsigned int i = 0; i < kBatchSize; ++i) {
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&buffers.coeffs[i]), kCoeffBytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(buffers.coeffs[i], kStale, kCoeffBytes, cudaMemcpyHostToDevice), cudaSuccess);
    batchHost[i].pSrc = buffers.coeffs[i];
    batchHost[i].pDst = buffers.coeffs[i];
    batchHost[i].pCoeffs = buffers.coeffs[i];
  }
  ASSERT_EQ(cudaMemcpy(buffers.batch, batchHost.data(), sizeof(NppiWarpPerspectiveBatchCXR) * kBatchSize,
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  auto runBatchInit = [&](bool delayStream, std::vector<NppiWarpPerspectiveBatchCXR> &output) -> bool {
    // Restore the stale coefficients and a pristine batch list.
    if (cudaStreamSynchronize(buffers.stream) != cudaSuccess) {
      return false;
    }
    for (unsigned int i = 0; i < kBatchSize; ++i) {
      if (cudaMemcpy(buffers.coeffs[i], kStale, kCoeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
      }
    }
    if (cudaMemcpy(buffers.batch, batchHost.data(), sizeof(NppiWarpPerspectiveBatchCXR) * kBatchSize,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }

    if (delayStream) {
      // Keep the context stream busy so a legacy-stream implementation would
      // read the coefficients before the updates below execute.
      if (cudaMemsetAsync(buffers.scratch, 0, kScratchBytes, buffers.stream) != cudaSuccess) {
        return false;
      }
    }
    for (unsigned int i = 0; i < kBatchSize; ++i) {
      if (cudaMemcpyAsync(buffers.coeffs[i], kFresh, kCoeffBytes, cudaMemcpyHostToDevice, buffers.stream) !=
          cudaSuccess) {
        return false;
      }
    }
    if (!delayStream) {
      // Reference run: make the fresh coefficients visible before the call.
      if (cudaStreamSynchronize(buffers.stream) != cudaSuccess) {
        return false;
      }
    }

    NppStreamContext ctx{};
    ctx.hStream = buffers.stream;
    if (nppiWarpPerspectiveBatchInit_Ctx(buffers.batch, kBatchSize, ctx) != NPP_SUCCESS) {
      return false;
    }
    if (cudaStreamSynchronize(buffers.stream) != cudaSuccess) {
      return false;
    }

    output.resize(kBatchSize);
    return cudaMemcpy(output.data(), buffers.batch, sizeof(NppiWarpPerspectiveBatchCXR) * kBatchSize,
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  };

  std::vector<NppiWarpPerspectiveBatchCXR> reference;
  std::vector<NppiWarpPerspectiveBatchCXR> delayed;
  ASSERT_TRUE(runBatchInit(false, reference));
  ASSERT_TRUE(runBatchInit(true, delayed));

  for (unsigned int i = 0; i < kBatchSize; ++i) {
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        EXPECT_DOUBLE_EQ(delayed[i].aTransformedCoeffs[row][col], reference[i].aTransformedCoeffs[row][col])
            << "batch " << i << " coefficient (" << row << "," << col
            << ") was read before the context stream updated it";
      }
    }
  }
}
