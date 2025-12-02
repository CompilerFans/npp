#pragma once

#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

namespace nppi {
namespace arithmetic {

// ============================================================================
// Template-based API Generation Framework
// ============================================================================
// This framework uses C++ templates to generate NPP API functions with minimal
// code duplication. No macros are used.
// ============================================================================

// ============================================================================
// Traits for operation categories
// ============================================================================

// Whether an operation uses scale factor
template <typename Op> struct HasScaleFactor : std::false_type {};

// Integer types typically use scale factor
template <> struct HasScaleFactor<AddOp<Npp8u>> : std::true_type {};
template <> struct HasScaleFactor<AddOp<Npp16u>> : std::true_type {};
template <> struct HasScaleFactor<AddOp<Npp16s>> : std::true_type {};
template <> struct HasScaleFactor<SubOp<Npp8u>> : std::true_type {};
template <> struct HasScaleFactor<SubOp<Npp16u>> : std::true_type {};
template <> struct HasScaleFactor<SubOp<Npp16s>> : std::true_type {};
template <> struct HasScaleFactor<MulOp<Npp8u>> : std::true_type {};
template <> struct HasScaleFactor<MulOp<Npp16u>> : std::true_type {};
template <> struct HasScaleFactor<MulOp<Npp16s>> : std::true_type {};
template <> struct HasScaleFactor<DivOp<Npp8u>> : std::true_type {};
template <> struct HasScaleFactor<DivOp<Npp16u>> : std::true_type {};
template <> struct HasScaleFactor<DivOp<Npp16s>> : std::true_type {};
template <> struct HasScaleFactor<SqrOp<Npp8u>> : std::true_type {};
template <> struct HasScaleFactor<SqrOp<Npp16u>> : std::true_type {};
template <> struct HasScaleFactor<SqrOp<Npp16s>> : std::true_type {};
template <> struct HasScaleFactor<SqrtOp<Npp8u>> : std::true_type {};
template <> struct HasScaleFactor<SqrtOp<Npp16u>> : std::true_type {};
template <> struct HasScaleFactor<SqrtOp<Npp16s>> : std::true_type {};

// ============================================================================
// Binary Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class BinaryOpAPI {
public:
  using Op = OpTemplate<T>;

  // Standard binary operation
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<T, Channels, Op>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                             oSizeROI, nScaleFactor, nppStreamCtx.hStream);
  }

  // Without scale factor (for float types)
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 0, nppStreamCtx);
  }

  // In-place with scale factor
  static NppStatus executeInplace(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor, NppStreamContext nppStreamCtx) {
    return execute(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
  }

  // In-place without scale factor
  static NppStatus executeInplace(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return executeInplace(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx);
  }
};

// ============================================================================
// Unary Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class UnaryOpAPI {
public:
  using Op = OpTemplate<T>;

  // Standard unary operation with scale factor
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                           NppStreamContext nppStreamCtx) {
    return UnaryOperationExecutor<T, Channels, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                            nppStreamCtx.hStream);
  }

  // Without scale factor
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                           NppStreamContext nppStreamCtx) {
    return execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx);
  }

  // In-place with scale factor
  static NppStatus executeInplace(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
  }

  // In-place without scale factor
  static NppStatus executeInplace(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return executeInplace(pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx);
  }
};

// ============================================================================
// Constant Operation API Generator (single constant)
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class ConstOpAPI {
public:
  using Op = OpTemplate<T>;

  // With scale factor - single constant
  static NppStatus execute(const T *pSrc, int nSrcStep, T nConstant, T *pDst, int nDstStep, NppiSize oSizeROI,
                           int nScaleFactor, NppStreamContext nppStreamCtx) {
    Op op(nConstant);
    return ConstOperationExecutor<T, Channels, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                            nppStreamCtx.hStream, op);
  }

  // Without scale factor - single constant
  static NppStatus execute(const T *pSrc, int nSrcStep, T nConstant, T *pDst, int nDstStep, NppiSize oSizeROI,
                           NppStreamContext nppStreamCtx) {
    return execute(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, 0, nppStreamCtx);
  }

  // In-place with scale factor
  static NppStatus executeInplace(T nConstant, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
  }

  // In-place without scale factor
  static NppStatus executeInplace(T nConstant, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return executeInplace(nConstant, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx);
  }
};

// ============================================================================
// Multi-channel Constant Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class MultiConstOpAPI {
public:
  using Op = OpTemplate<T>;

  // With scale factor - array of constants
  static NppStatus execute(const T *pSrc, int nSrcStep, const T *aConstants, T *pDst, int nDstStep, NppiSize oSizeROI,
                           int nScaleFactor, NppStreamContext nppStreamCtx) {
    return MultiConstOperationExecutor<T, Channels, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                                 nppStreamCtx.hStream, aConstants);
  }

  // Without scale factor
  static NppStatus execute(const T *pSrc, int nSrcStep, const T *aConstants, T *pDst, int nDstStep, NppiSize oSizeROI,
                           NppStreamContext nppStreamCtx) {
    return execute(pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, 0, nppStreamCtx);
  }

  // In-place with scale factor
  static NppStatus executeInplace(const T *aConstants, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
  }

  // In-place without scale factor
  static NppStatus executeInplace(const T *aConstants, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return executeInplace(aConstants, pSrcDst, nSrcDstStep, oSizeROI, 0, nppStreamCtx);
  }
};

// ============================================================================
// Logical Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class LogicalOpAPI {
public:
  using Op = OpTemplate<T>;

  // Standard logical operation (no scale factor)
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return BinaryOperationExecutor<T, Channels, Op>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                             oSizeROI, 0, nppStreamCtx.hStream);
  }

  // In-place
  static NppStatus executeInplace(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
  }
};

// ============================================================================
// Logical Constant Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class LogicalConstOpAPI {
public:
  using Op = OpTemplate<T>;

  // Single constant
  static NppStatus execute(const T *pSrc, int nSrcStep, T nConstant, T *pDst, int nDstStep, NppiSize oSizeROI,
                           NppStreamContext nppStreamCtx) {
    Op op(nConstant);
    return ConstOperationExecutor<T, Channels, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, op);
  }

  // In-place single constant
  static NppStatus executeInplace(T nConstant, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
  }

  // Multi-channel constants
  static NppStatus executeMulti(const T *pSrc, int nSrcStep, const T *aConstants, T *pDst, int nDstStep,
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return MultiConstOperationExecutor<T, Channels, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                 nppStreamCtx.hStream, aConstants);
  }

  // In-place multi-channel constants
  static NppStatus executeMultiInplace(const T *aConstants, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
    return executeMulti(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
  }
};

// ============================================================================
// Shift Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class ShiftOpAPI {
public:
  // Single shift count
  static NppStatus execute(const T *pSrc, int nSrcStep, Npp32u nConstant, T *pDst, int nDstStep, NppiSize oSizeROI,
                           NppStreamContext nppStreamCtx) {
    OpTemplate<T> op(static_cast<int>(nConstant));
    return ConstOperationExecutor<T, Channels, OpTemplate<T>>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                                       nppStreamCtx.hStream, op);
  }

  // In-place single shift count
  static NppStatus executeInplace(Npp32u nConstant, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, nConstant, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
  }
};

// ============================================================================
// Helper: Get default stream context
// ============================================================================

inline NppStreamContext getDefaultStreamContext() {
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  return ctx;
}

} // namespace arithmetic
} // namespace nppi
