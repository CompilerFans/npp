# MPP vs NVIDIA NPP 验证报告

**生成时间**: 2025-12-18 16:39:12

---

## 基础图像处理

| 示例程序 | 输出文件 | 状态 | 差异比例 | PSNR |
|----------|----------|------|----------|------|
| boxFilterNPP | teapot512_boxFilter.pgm | [PASS] 完全匹配 | 0.00% | 100.0dB |
| cannyEdgeDetectorNPP | teapot512_cannyEdgeDetection.pgm | [DIFF] 有明显差异 | 0.30% | 25.2dB |
| histEqualizationNPP | teapot512_histEqualization.pgm | [PASS] 完全匹配 | 0.00% | 100.0dB |
| freeImageInteropNPP | teapot512_boxFilterFII.pgm | [WARN] 可接受 (PSNR>=30dB) | 0.92% | 39.7dB |

## FilterBorderControl

| 示例程序 | 输出文件 | 状态 | 差异比例 | PSNR |
|----------|----------|------|----------|------|
| FilterBorderControlNPP | teapot512_...PrewittBorderY_Horizontal.pgm | [PASS] 完全匹配 | 0.00% | 100.0dB |
| FilterBorderControlNPP | teapot512_...PrewittBorderX_Vertical.pgm | [DIFF] 有明显差异 | 2.90% | 15.4dB |

## BatchedLabelMarkers

| 图像 | 类型 | 状态 | 差异比例 |
|------|------|------|----------|
| batchedLabelMarkersNPP | teapot_LabelMarkersUF | [PASS] 完全匹配 | 0.00% |
| batchedLabelMarkersNPP | teapot_LabelMarkersUFBatch | [PASS] 完全匹配 | 0.00% |
| batchedLabelMarkersNPP | teapot_CompressedMarkerLabelsUF | [OK] 标签数一致 (155343) | 100.00% |
| batchedLabelMarkersNPP | teapot_CompressedMarkerLabelsUFBatch | [OK] 标签数一致 (155343) | 61.91% |
| batchedLabelMarkersNPP | CT_skull_LabelMarkersUF | [DIFF] 标签数: REF=439 MPP=407 | 12.25% |
| batchedLabelMarkersNPP | CT_skull_LabelMarkersUFBatch | [DIFF] 标签数: REF=417 MPP=407 | 0.21% |
| batchedLabelMarkersNPP | CT_skull_CompressedMarkerLabelsUF | [DIFF] 标签数: REF=439 MPP=407 | 99.65% |
| batchedLabelMarkersNPP | CT_skull_CompressedMarkerLabelsUFBatch | [DIFF] 标签数: REF=417 MPP=407 | 86.87% |
| batchedLabelMarkersNPP | PCB_METAL_LabelMarkersUF | [PASS] 完全匹配 | 0.00% |
| batchedLabelMarkersNPP | PCB_METAL_LabelMarkersUFBatch | [PASS] 完全匹配 | 0.00% |
| batchedLabelMarkersNPP | PCB_METAL_CompressedMarkerLabelsUF | [OK] 标签数一致 (3732) | 100.00% |
| batchedLabelMarkersNPP | PCB_METAL_CompressedMarkerLabelsUFBatch | [OK] 标签数一致 (3732) | 90.55% |
| batchedLabelMarkersNPP | PCB_LabelMarkersUF | [DIFF] 标签数: REF=1467 MPP=1436 | 25.97% |
| batchedLabelMarkersNPP | PCB_LabelMarkersUFBatch | [DIFF] 标签数: REF=1457 MPP=1436 | 17.63% |
| batchedLabelMarkersNPP | PCB_CompressedMarkerLabelsUF | [DIFF] 标签数: REF=1467 MPP=1436 | 99.49% |
| batchedLabelMarkersNPP | PCB_CompressedMarkerLabelsUFBatch | [DIFF] 标签数: REF=1457 MPP=1436 | 99.61% |
| batchedLabelMarkersNPP | PCB2_LabelMarkersUF | [DIFF] 标签数: REF=1272 MPP=1202 | 0.45% |
| batchedLabelMarkersNPP | PCB2_LabelMarkersUFBatch | [DIFF] 标签数: REF=1222 MPP=1202 | 0.30% |
| batchedLabelMarkersNPP | PCB2_CompressedMarkerLabelsUF | [DIFF] 标签数: REF=1272 MPP=1202 | 99.75% |
| batchedLabelMarkersNPP | PCB2_CompressedMarkerLabelsUFBatch | [DIFF] 标签数: REF=1222 MPP=1202 | 13.66% |

---

## 总结

### 基础图像处理

| 指标 | 结果 |
|------|------|
| 完全匹配 | 3/6 |
| 高质量 (PSNR>=40dB) | 0/6 |

### BatchedLabelMarkers

| 指标 | 结果 |
|------|------|
| 完全匹配 | 4/20 |
| 标签数一致 | 4/20 |
| 总测试数 | 20 |

### 说明

- **完全匹配**: 所有像素值完全相同，实现正确
- **高质量**: PSNR >= 40dB，微小差异可能由浮点精度导致
- **可接受**: PSNR >= 30dB，存在一定差异，建议排查
- **有明显差异**: PSNR < 30dB，很可能存在实现问题，需要修复

### 差异判定标准

| 差异程度 | 判定 | 建议 |
|----------|------|------|
| 0% | 实现正确 | 无需处理 |
| <1% 且 PSNR>40dB | 可能是精度差异 | 可接受，建议观察 |
| 1%-5% | 可能有问题 | 建议排查实现 |
| >5% | 很可能有bug | 需要修复 |

### 可能的差异原因

微小差异 (<1%) 可能由以下原因导致：
- 浮点运算顺序不同
- 舍入方式差异 (round/floor/ceil)

明显差异 (>1%) 通常表示：
- 算法实现有误
- 边界处理方式不同
- 参数解释不一致
