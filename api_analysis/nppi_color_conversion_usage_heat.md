# nppi_color_conversion.h usage heat (repo scan)

Date: 2026-01-31

Assumption: usage heat is measured by symbol mentions in test/ and examples/ only (to avoid counting implementations).
Scan scope: directories = test, examples, extensions = .c, .cc, .cpp, .cu, .cuh, .cxx, .h, .hpp, .hxx, .inl, .md, .txt.
Files scanned: 270
APIs in nppi_color_conversion.h: 1040

## Usage tiers
- High (>=5): 1
- Medium (2-4): 23
- Low (1): 144
- Zero (0): 872

## Top usage (up to 30)
- nppiNV12ToRGB_8u_P2C3R: 5
- nppiLUT_Linear_8u_C1R: 4
- nppiLUT_Linear_8u_C1R_Ctx: 4
- nppiNV12ToBGR_8u_P2C3R: 3
- nppiRGBToHLS_8u_C3R: 3
- nppiRGBToYUV_8u_C3R: 3
- nppiYUV420ToRGB_8u_P3C3R: 3
- nppiCFAToRGB_8u_C1C3R_Ctx: 2
- nppiColorTwist32f_8u_C3R_Ctx: 2
- nppiGammaFwd_8u_AC4IR: 2
- nppiGammaFwd_8u_AC4R: 2
- nppiGammaFwd_8u_C3IR: 2
- nppiGammaInv_8u_AC4IR: 2
- nppiGammaInv_8u_AC4IR_Ctx: 2
- nppiGammaInv_8u_C3IR: 2
- nppiGammaInv_8u_C3IR_Ctx: 2
- nppiGammaInv_8u_C3R: 2
- nppiNV12ToRGB_709CSC_8u_P2C3R: 2
- nppiRGBToGray_8u_C3C1R: 2
- nppiRGBToHLS_8u_C3R_Ctx: 2
- nppiRGBToHSV_8u_C3R: 2
- nppiYUV420ToRGB_8u_P3AC4R: 2
- nppiYUV420ToRGB_8u_P3C4R: 2
- nppiYUVToRGB_8u_C3R: 2
- nppiBGRToHLS_8u_AC4R: 1
- nppiBGRToHLS_8u_AC4R_Ctx: 1
- nppiBGRToHLS_8u_C3P3R: 1
- nppiBGRToHLS_8u_C3P3R_Ctx: 1
- nppiBGRToHLS_8u_P3R: 1
- nppiBGRToHLS_8u_P3R_Ctx: 1
