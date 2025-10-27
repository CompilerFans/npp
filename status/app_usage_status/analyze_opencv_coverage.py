#!/usr/bin/env python3
"""
Analyze MPP coverage for OpenCV NPPI APIs
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def load_opencv_apis(opencv_api_file):
    """Load OpenCV used APIs"""
    apis = set()
    with open(opencv_api_file, 'r') as f:
        for line in f:
            api = line.strip()
            if api:
                apis.add(api)
    return apis

def load_mpp_coverage(coverage_csv):
    """Load MPP implementation coverage"""
    implemented = set()
    tested = set()

    with open(coverage_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            func_name = row['function_name']
            if row['implemented'] == 'yes':
                implemented.add(func_name)
            if row['tested'] == 'yes':
                tested.add(func_name)

    return implemented, tested

def categorize_apis(apis):
    """Categorize APIs by function group"""
    categories = defaultdict(list)

    for api in apis:
        # Remove _Ctx suffix for categorization
        base = api.replace('_Ctx', '')

        # Extract category from function name
        if 'Histogram' in base:
            cat = 'Histogram'
        elif 'Filter' in base or 'Erode' in base or 'Dilate' in base:
            cat = 'Filtering'
        elif 'Warp' in base or 'Resize' in base or 'Remap' in base or 'Rotate' in base:
            cat = 'Geometry'
        elif 'Transpose' in base:
            cat = 'Transpose'
        elif 'Integral' in base or 'SumWindow' in base:
            cat = 'Integral'
        elif 'Alpha' in base or 'Gamma' in base or 'SwapChannels' in base:
            cat = 'Color'
        elif 'Mean' in base or 'StdDev' in base:
            cat = 'Statistics'
        elif 'Threshold' in base:
            cat = 'Threshold'
        elif 'AndC' in base or 'XorC' in base or 'LShiftC' in base or 'RShiftC' in base:
            cat = 'Logical'
        elif 'Magnitude' in base:
            cat = 'Arithmetic'
        elif 'Graphcut' in base:
            cat = 'Segmentation'
        elif 'St' in base or 'ncv' in base:
            cat = 'Legacy'
        else:
            cat = 'Other'

        categories[cat].append(api)

    return categories

def main():
    opencv_api_file = Path('nppi_apis_unique.txt')
    coverage_csv = Path('../../api_analysis/coverage.csv')

    if not opencv_api_file.exists():
        print(f"Error: {opencv_api_file} not found")
        return 1

    if not coverage_csv.exists():
        print(f"Error: {coverage_csv} not found")
        return 1

    # Load data
    opencv_apis = load_opencv_apis(opencv_api_file)
    implemented, tested = load_mpp_coverage(coverage_csv)

    # Analyze coverage
    opencv_implemented = opencv_apis & implemented
    opencv_tested = opencv_apis & tested
    opencv_missing = opencv_apis - implemented

    # Categorize
    all_categories = categorize_apis(opencv_apis)
    missing_categories = categorize_apis(opencv_missing)

    # Generate report
    print("=" * 60)
    print("OpenCV NPPI API Coverage Analysis")
    print("=" * 60)
    print()

    print(f"Total OpenCV APIs: {len(opencv_apis)}")
    print(f"Implemented: {len(opencv_implemented)} ({len(opencv_implemented)*100/len(opencv_apis):.1f}%)")
    print(f"Tested: {len(opencv_tested)} ({len(opencv_tested)*100/len(opencv_apis):.1f}%)")
    print(f"Missing: {len(opencv_missing)} ({len(opencv_missing)*100/len(opencv_apis):.1f}%)")
    print()

    print("=" * 60)
    print("Coverage by Category")
    print("=" * 60)

    for cat in sorted(all_categories.keys()):
        total = len(all_categories[cat])
        missing = len(missing_categories.get(cat, []))
        impl = total - missing
        print(f"{cat:15s}: {impl:3d}/{total:3d} ({impl*100/total:5.1f}%)")

    print()
    print("=" * 60)
    print("Missing APIs by Priority")
    print("=" * 60)
    print()

    # Priority order
    priority_cats = [
        'Filtering',
        'Geometry',
        'Color',
        'Arithmetic',
        'Histogram',
        'Statistics',
        'Threshold',
        'Logical',
        'Integral',
        'Transpose',
        'Segmentation',
        'Legacy',
        'Other'
    ]

    for cat in priority_cats:
        if cat in missing_categories and missing_categories[cat]:
            print(f"\n{cat}:")
            for api in sorted(missing_categories[cat])[:10]:
                print(f"  - {api}")
            if len(missing_categories[cat]) > 10:
                print(f"  ... and {len(missing_categories[cat]) - 10} more")

    # Write summary CSV
    output_csv = Path('opencv_coverage_summary.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'total', 'implemented', 'missing', 'coverage_%'])

        for cat in sorted(all_categories.keys()):
            total = len(all_categories[cat])
            missing = len(missing_categories.get(cat, []))
            impl = total - missing
            coverage = impl * 100 / total
            writer.writerow([cat, total, impl, missing, f"{coverage:.1f}"])

    print(f"\nDetailed summary saved to: {output_csv}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
