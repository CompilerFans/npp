// Generated API wrappers -- do not edit by hand.
// family: nppsDiv_Round


#include "npp.h"
#include "npps_elementwise_executor.h"

NppStatus nppsDiv_Round_16s_ISfs_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return npps::elementwise::binaryRoundInplace<npps::elementwise::DivRoundOp>(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppsDiv_Round_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return npps::elementwise::binaryRound<npps::elementwise::DivRoundOp>(pSrc2, pSrc1, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppsDiv_Round_16u_ISfs_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return npps::elementwise::binaryRoundInplace<npps::elementwise::DivRoundOp>(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppsDiv_Round_16u_Sfs_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return npps::elementwise::binaryRound<npps::elementwise::DivRoundOp>(pSrc2, pSrc1, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppsDiv_Round_8u_ISfs_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return npps::elementwise::binaryRoundInplace<npps::elementwise::DivRoundOp>(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppsDiv_Round_8u_Sfs_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return npps::elementwise::binaryRound<npps::elementwise::DivRoundOp>(pSrc2, pSrc1, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppsDiv_Round_16s_ISfs(const Npp16s * pSrc, Npp16s * pSrcDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsDiv_Round_16s_ISfs_Ctx(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx);
}

NppStatus nppsDiv_Round_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsDiv_Round_16s_Sfs_Ctx(pSrc1, pSrc2, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx);
}

NppStatus nppsDiv_Round_16u_ISfs(const Npp16u * pSrc, Npp16u * pSrcDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsDiv_Round_16u_ISfs_Ctx(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx);
}

NppStatus nppsDiv_Round_16u_Sfs(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsDiv_Round_16u_Sfs_Ctx(pSrc1, pSrc2, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx);
}

NppStatus nppsDiv_Round_8u_ISfs(const Npp8u * pSrc, Npp8u * pSrcDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsDiv_Round_8u_ISfs_Ctx(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx);
}

NppStatus nppsDiv_Round_8u_Sfs(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsDiv_Round_8u_Sfs_Ctx(pSrc1, pSrc2, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx);
}
