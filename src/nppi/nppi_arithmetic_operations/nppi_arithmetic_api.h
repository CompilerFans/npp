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
// AC4 Unary Operation API Generator (processes 3 channels, preserves alpha)
// ============================================================================

template <typename T, template <typename> class OpTemplate> class UnaryOpAC4API {
public:
  using Op = OpTemplate<T>;

  // Standard unary operation with scale factor
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                           NppStreamContext nppStreamCtx) {
    return UnaryAC4OperationExecutor<T, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor,
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
// AC4 Binary Operation API Generator (processes 3 channels, preserves alpha)
// ============================================================================

template <typename T, template <typename> class OpTemplate> class BinaryOpAC4API {
public:
  using Op = OpTemplate<T>;

  // Standard binary operation
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    return BinaryAC4OperationExecutor<T, Op>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI,
                                                      nScaleFactor, nppStreamCtx.hStream);
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
// AC4 Multi-channel Constant Operation API Generator (processes 3 channels, preserves alpha)
// ============================================================================

template <typename T, template <typename> class OpTemplate> class MultiConstOpAC4API {
public:
  using Op = OpTemplate<T>;

  // With scale factor - array of constants (3 constants for RGB channels)
  static NppStatus execute(const T *pSrc, int nSrcStep, const T *aConstants, T *pDst, int nDstStep, NppiSize oSizeROI,
                           int nScaleFactor, NppStreamContext nppStreamCtx) {
    return MultiConstAC4OperationExecutor<T, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nScaleFactor,
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
// AC4 Logical Operation API Generator (processes 3 channels, preserves alpha)
// ============================================================================

template <typename T, template <typename> class OpTemplate> class LogicalOpAC4API {
public:
  using Op = OpTemplate<T>;

  // Standard logical operation (no scale factor)
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return BinaryAC4OperationExecutor<T, Op>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 0,
                                                      nppStreamCtx.hStream);
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
// AC4 Logical Constant Operation API Generator (processes 3 channels, preserves alpha)
// ============================================================================

template <typename T, template <typename> class OpTemplate> class LogicalConstOpAC4API {
public:
  using Op = OpTemplate<T>;

  // Multi-channel constants (3 constants for RGB channels)
  static NppStatus executeMulti(const T *pSrc, int nSrcStep, const T *aConstants, T *pDst, int nDstStep,
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return MultiConstAC4OperationExecutor<T, Op>::execute(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
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
// Multi-Channel Shift Operation API Generator
// ============================================================================

template <typename T, int Channels, template <typename, int> class MultiOpTemplate> class ShiftMultiOpAPI {
public:
  // Multi-channel shift counts
  static NppStatus execute(const T *pSrc, int nSrcStep, const Npp32u *aConstants, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return ShiftMultiOperationExecutor<T, Channels, MultiOpTemplate<T, Channels>>::execute(
        pSrc, nSrcStep, aConstants, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  }

  // In-place multi-channel shift counts
  static NppStatus executeInplace(const Npp32u *aConstants, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
  }
};

// ============================================================================
// AC4 Multi-Channel Shift Operation API Generator (processes 3 channels, preserves alpha)
// ============================================================================

template <typename T, template <typename, int> class MultiOpTemplate> class ShiftMultiOpAC4API {
public:
  // Multi-channel shift counts (3 constants for RGB channels)
  static NppStatus execute(const T *pSrc, int nSrcStep, const Npp32u *aConstants, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return ShiftMultiAC4OperationExecutor<T, MultiOpTemplate<T, 4>>::execute(pSrc, nSrcStep, aConstants, pDst, nDstStep,
                                                                              oSizeROI, nppStreamCtx.hStream);
  }

  // In-place multi-channel shift counts
  static NppStatus executeInplace(const Npp32u *aConstants, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return execute(pSrcDst, nSrcDstStep, aConstants, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
  }
};

// Convenience aliases for shift operations
template <typename T, int C> using LShift = ShiftOpAPI<T, C, LShiftConstOp>;
template <typename T, int C> using RShift = ShiftOpAPI<T, C, RShiftConstOp>;
template <typename T, int C> using LShiftMulti = ShiftMultiOpAPI<T, C, LShiftConstMultiOp>;
template <typename T, int C> using RShiftMulti = ShiftMultiOpAPI<T, C, RShiftConstMultiOp>;
template <typename T> using LShiftMultiAC4 = ShiftMultiOpAC4API<T, LShiftConstMultiOp>;
template <typename T> using RShiftMultiAC4 = ShiftMultiOpAC4API<T, RShiftConstMultiOp>;

// ============================================================================
// DivRound Operation API Generator (with rounding mode)
// ============================================================================

template <typename T, int Channels> class DivRoundOpAPI {
public:
  // Standard division with rounding mode
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
    return DivRoundOperationExecutor<T, Channels>::execute(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI,
                                                           rndMode, nScaleFactor, nppStreamCtx.hStream);
  }

  // In-place with rounding mode
  static NppStatus executeInplace(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
    return execute(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, rndMode, nScaleFactor,
                   nppStreamCtx);
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
