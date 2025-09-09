#!/bin/bash

echo "OpenNPP与NVIDIA NPP结果对比..."

# 创建OpenNPP输出目录（如果不存在）
mkdir -p opennpp_results

# 检查是否有构建输出
if [ ! -d "build" ]; then
    echo "❌ 未找到build目录，请先运行 ./build.sh"
    exit 1
fi

echo ""
echo "=== 运行OpenNPP样本程序 ==="

cd build/bin

echo "运行boxFilterNPP..."
./boxFilterNPP

echo "运行cannyEdgeDetectorNPP..."
./cannyEdgeDetectorNPP

echo "运行histEqualizationNPP..."
./histEqualizationNPP

echo "运行FilterBorderControlNPP..."
./FilterBorderControlNPP

echo "watershedSegmentationNPP已运行（.raw文件已在bin目录中）"

cd ../..

echo ""
echo "=== 收集OpenNPP输出结果 ==="

# 从Samples目录复制OpenNPP的输出结果
if [ -f "Samples/4_CUDA_Libraries/boxFilterNPP/teapot512_boxFilter.pgm" ]; then
    cp "Samples/4_CUDA_Libraries/boxFilterNPP/teapot512_boxFilter.pgm" opennpp_results/
    echo "✅ boxFilter结果已收集"
else
    echo "❌ boxFilter结果未找到"
fi

if [ -f "Samples/4_CUDA_Libraries/cannyEdgeDetectorNPP/teapot512_cannyEdgeDetection.pgm" ]; then
    cp "Samples/4_CUDA_Libraries/cannyEdgeDetectorNPP/teapot512_cannyEdgeDetection.pgm" opennpp_results/
    echo "✅ Canny边缘检测结果已收集"
else
    echo "❌ Canny边缘检测结果未找到"
fi

if [ -f "Common/data/teapot512_histEqualization.pgm" ]; then
    cp "Common/data/teapot512_histEqualization.pgm" opennpp_results/
    echo "✅ 直方图均衡化结果已收集"
else
    echo "❌ 直方图均衡化结果未找到"
fi

# FilterBorderControl结果
filter_results=(
    "teapot512.pgm_gradientVectorPrewittBorderX_Vertical.pgm"
    "teapot512_gradientVectorPrewittBorderY_Horizontal.pgm"
    "teapot512_gradientVectorPrewittBorderX_Vertical_WithNoSourceBorders.pgm"
    "teapot512_gradientVectorPrewittBorderY_Horizontal_WithNoSourceBorders.pgm"
    "teapot512_gradientVectorPrewittBorderX_Vertical_BorderDiffs.pgm"
    "teapot512_gradientVectorPrewittBorderY_Horizontal_BorderDiffs.pgm"
)

mkdir -p opennpp_results/FilterBorderControl
filter_count=0
for result in "${filter_results[@]}"; do
    src_path="Samples/4_CUDA_Libraries/FilterBorderControlNPP/data/$result"
    if [ -f "$src_path" ]; then
        cp "$src_path" "opennpp_results/FilterBorderControl/"
        ((filter_count++))
    fi
done
echo "✅ FilterBorderControl: $filter_count/6 个结果已收集"

# Watershed结果
watershed_files=(
    "teapot_Segments_8Way_512x512_8u.raw"
    "teapot_CompressedSegmentLabels_8Way_512x512_32u.raw"
    "teapot_SegmentBoundaries_8Way_512x512_8u.raw"
    "teapot_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw"
    "CT_skull_Segments_8Way_512x512_8u.raw"
    "CT_skull_CompressedSegmentLabels_8Way_512x512_32u.raw"
    "CT_skull_SegmentBoundaries_8Way_512x512_8u.raw"
    "CT_skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw"
    "Rocks_Segments_8Way_512x512_8u.raw"
    "Rocks_CompressedSegmentLabels_8Way_512x512_32u.raw"
    "Rocks_SegmentBoundaries_8Way_512x512_8u.raw"
    "Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw"
)

mkdir -p opennpp_results/watershed
watershed_count=0

# OpenNPP的watershed文件名稍有不同，需要映射
watershed_mapping=(
    "teapot_Segments_8Way_512x512_8u.raw:teapot_Segments_8Way_512x512_8u.raw"
    "teapot_CompressedSegmentLabels_8Way_512x512_32u.raw:teapot_SegmentLabels_8Way_512x512_32u.raw"
    "teapot_SegmentBoundaries_8Way_512x512_8u.raw:teapot_SegmentBoundaries_8Way_512x512_8u.raw"
    "teapot_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw:teapot_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw"
    "CT_skull_Segments_8Way_512x512_8u.raw:CT_skull_Segments_8Way_512x512_8u.raw"
    "CT_skull_CompressedSegmentLabels_8Way_512x512_32u.raw:CT_skull_SegmentLabels_8Way_512x512_32u.raw"
    "CT_skull_SegmentBoundaries_8Way_512x512_8u.raw:CT_skull_SegmentBoundaries_8Way_512x512_8u.raw"
    "CT_skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw:CT_skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw"
    "Rocks_Segments_8Way_512x512_8u.raw:Rocks_Segments_8Way_512x512_8u.raw"
    "Rocks_CompressedSegmentLabels_8Way_512x512_32u.raw:Rocks_SegmentLabels_8Way_512x512_32u.raw"
    "Rocks_SegmentBoundaries_8Way_512x512_8u.raw:Rocks_SegmentBoundaries_8Way_512x512_8u.raw"
    "Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw:Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw"
)

for mapping in "${watershed_mapping[@]}"; do
    nvidia_name="${mapping%:*}"
    opennpp_name="${mapping#*:}"
    
    if [ -f "build/bin/$opennpp_name" ]; then
        cp "build/bin/$opennpp_name" "opennpp_results/watershed/$nvidia_name"
        ((watershed_count++))
    fi
done
echo "✅ Watershed: $watershed_count/12 个结果已收集"

echo ""
echo "=== 结果文件大小对比 ==="

compare_files() {
    local category=$1
    local opennpp_dir=$2
    local nvidia_dir=$3
    
    echo ""
    echo "--- $category ---"
    for file in "$nvidia_dir"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            nvidia_size=$(stat -c%s "$file")
            
            if [ -f "$opennpp_dir/$filename" ]; then
                opennpp_size=$(stat -c%s "$opennpp_dir/$filename")
                if [ "$nvidia_size" -eq "$opennpp_size" ]; then
                    echo "✅ $filename: 大小匹配 ($nvidia_size bytes)"
                else
                    echo "⚠️  $filename: 大小不匹配 (NVIDIA: $nvidia_size, OpenNPP: $opennpp_size)"
                fi
            else
                echo "❌ $filename: OpenNPP结果缺失"
            fi
        fi
    done
}

# 对比基础结果
echo "基础处理结果:"
for file in reference_nvidia_npp/*.pgm; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        nvidia_size=$(stat -c%s "$file")
        
        if [ -f "opennpp_results/$filename" ]; then
            opennpp_size=$(stat -c%s "opennpp_results/$filename")
            if [ "$nvidia_size" -eq "$opennpp_size" ]; then
                echo "✅ $filename: 大小匹配 ($nvidia_size bytes)"
            else
                echo "⚠️  $filename: 大小不匹配 (NVIDIA: $nvidia_size, OpenNPP: $opennpp_size)"
            fi
        else
            echo "❌ $filename: OpenNPP结果缺失"
        fi
    fi
done

# 对比FilterBorderControl结果
compare_files "FilterBorderControl" "opennpp_results/FilterBorderControl" "reference_nvidia_npp/FilterBorderControl"

# 对比Watershed结果
compare_files "Watershed" "opennpp_results/watershed" "reference_nvidia_npp/watershed"

echo ""
echo "=== 对比总结 ==="
echo "NVIDIA NPP参考结果位于: reference_nvidia_npp/"
echo "OpenNPP输出结果位于: opennpp_results/"
echo ""
echo "如需详细像素级对比，请使用verify_results程序进行数值验证。"