#!/usr/bin/env python3
"""
Analyze OpenCV NPPI API support in MPP and generate implementation plan
"""

import csv
from pathlib import Path
from collections import defaultdict

def load_opencv_apis(api_file):
    """Load OpenCV used APIs"""
    apis = set()
    with open(api_file, 'r') as f:
        for line in f:
            api = line.strip()
            if api:
                apis.add(api)
    return apis

def load_coverage(coverage_csv):
    """Load implementation and test coverage"""
    status = {}
    with open(coverage_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            func_name = row['function_name']
            status[func_name] = {
                'implemented': row['implemented'] == 'yes',
                'impl_file': row['impl_file'],
                'tested': row['tested'] == 'yes',
                'test_file': row['test_file'],
                'module': row['module']
            }
    return status

def categorize_api(api):
    """Categorize API by function type"""
    base = api.replace('_Ctx', '')

    if 'Histogram' in base:
        return 'Histogram', 'P1'
    elif 'FilterMax' in base or 'FilterMin' in base:
        return 'Filtering/MinMax', 'P1'
    elif 'FilterBox' in base or 'Erode' in base or 'Dilate' in base:
        return 'Filtering/Morphology', 'P0'
    elif 'Rotate' in base:
        return 'Geometry/Rotate', 'P0'
    elif 'WarpAffine' in base or 'WarpPerspective' in base:
        return 'Geometry/Warp', 'P0'
    elif 'Resize' in base or 'Remap' in base:
        return 'Geometry/Resize', 'P0'
    elif 'Transpose' in base:
        return 'Transpose', 'P1'
    elif 'Integral' in base or 'SumWindow' in base:
        return 'Integral', 'P1'
    elif 'AlphaComp' in base or 'AlphaPremul' in base:
        return 'Color/Alpha', 'P1'
    elif 'Gamma' in base or 'SwapChannels' in base:
        return 'Color/Gamma', 'P1'
    elif 'Mean' in base or 'StdDev' in base or 'RectStdDev' in base:
        return 'Statistics', 'P1'
    elif 'Threshold' in base:
        return 'Threshold', 'P0'
    elif 'AndC' in base or 'XorC' in base or 'LShiftC' in base or 'RShiftC' in base:
        return 'Logical/Bitwise', 'P1'
    elif 'Magnitude' in base:
        return 'Arithmetic/Magnitude', 'P1'
    elif 'Graphcut' in base:
        return 'Segmentation/Graphcut', 'P2'
    elif 'Mirror' in base:
        return 'Geometry/Mirror', 'P1'
    elif 'St' in base or 'ncv' in base:
        return 'Legacy', 'P3'
    else:
        return 'Other', 'P2'

def main():
    opencv_api_file = Path('nppi_apis_unique.txt')
    coverage_csv = Path('../../api_analysis/coverage.csv')

    # Load data
    opencv_apis = load_opencv_apis(opencv_api_file)
    coverage = load_coverage(coverage_csv)

    # Analyze
    implemented = set()
    tested = set()
    impl_not_tested = set()
    missing = set()

    for api in opencv_apis:
        if api in coverage:
            if coverage[api]['implemented']:
                implemented.add(api)
                if coverage[api]['tested']:
                    tested.add(api)
                else:
                    impl_not_tested.add(api)
            else:
                missing.add(api)
        else:
            missing.add(api)

    # Categorize missing APIs by priority
    missing_by_priority = defaultdict(lambda: defaultdict(list))
    for api in missing:
        category, priority = categorize_api(api)
        missing_by_priority[priority][category].append(api)

    # Categorize implemented but not tested
    impl_not_tested_by_cat = defaultdict(list)
    for api in impl_not_tested:
        category, _ = categorize_api(api)
        impl_not_tested_by_cat[category].append(api)

    # Generate report
    print("=" * 70)
    print("OpenCV NPPI API Implementation Status in MPP")
    print("=" * 70)
    print()

    print(f"Total OpenCV APIs: {len(opencv_apis)}")
    print(f"Implemented: {len(implemented)} ({len(implemented)*100/len(opencv_apis):.1f}%)")
    print(f"  - With tests: {len(tested)} ({len(tested)*100/len(opencv_apis):.1f}%)")
    print(f"  - Without tests: {len(impl_not_tested)} ({len(impl_not_tested)*100/len(opencv_apis):.1f}%)")
    print(f"Not implemented: {len(missing)} ({len(missing)*100/len(opencv_apis):.1f}%)")
    print()

    # Implemented but not tested
    if impl_not_tested:
        print("=" * 70)
        print(f"Implemented but Not Tested ({len(impl_not_tested)} APIs)")
        print("=" * 70)
        for cat in sorted(impl_not_tested_by_cat.keys()):
            apis = impl_not_tested_by_cat[cat]
            print(f"\n{cat} ({len(apis)}):")
            for api in sorted(apis)[:5]:
                print(f"  - {api}")
            if len(apis) > 5:
                print(f"  ... and {len(apis) - 5} more")

    # Missing APIs by priority
    print()
    print("=" * 70)
    print(f"Not Implemented ({len(missing)} APIs)")
    print("=" * 70)

    for priority in ['P0', 'P1', 'P2', 'P3']:
        if priority in missing_by_priority:
            print(f"\n{priority} Priority:")
            for cat in sorted(missing_by_priority[priority].keys()):
                apis = missing_by_priority[priority][cat]
                print(f"\n  {cat} ({len(apis)}):")
                for api in sorted(apis)[:8]:
                    print(f"    - {api}")
                if len(apis) > 8:
                    print(f"    ... and {len(apis) - 8} more")

    # Generate implementation plan
    print()
    print("=" * 70)
    print("IMPLEMENTATION PLAN")
    print("=" * 70)
    print()

    # Calculate priorities
    p0_count = sum(len(apis) for apis in missing_by_priority['P0'].values())
    p1_count = sum(len(apis) for apis in missing_by_priority['P1'].values())
    p2_count = sum(len(apis) for apis in missing_by_priority['P2'].values())

    print("Phase 1: Add tests for implemented APIs")
    print(f"  - {len(impl_not_tested)} APIs need tests")
    print("  - Focus: Filtering, Geometry, Histogram")
    print()

    print("Phase 2: Implement P0 (Critical for OpenCV core)")
    print(f"  - {p0_count} APIs")
    for cat in sorted(missing_by_priority['P0'].keys()):
        print(f"    - {cat}: {len(missing_by_priority['P0'][cat])} APIs")
    print()

    print("Phase 3: Implement P1 (High priority)")
    print(f"  - {p1_count} APIs")
    for cat in sorted(missing_by_priority['P1'].keys()):
        count = len(missing_by_priority['P1'][cat])
        if count > 5:
            print(f"    - {cat}: {count} APIs")
    print()

    print("Phase 4: Implement P2 (Medium priority)")
    print(f"  - {p2_count} APIs")
    print()

    # Generate detailed CSV
    output_csv = Path('opencv_implementation_plan.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['api', 'category', 'priority', 'implemented', 'tested', 'action'])

        for api in sorted(opencv_apis):
            cat, priority = categorize_api(api)
            impl = api in implemented
            test = api in tested

            if impl and not test:
                action = 'add_test'
            elif not impl:
                action = 'implement'
            else:
                action = 'done'

            writer.writerow([api, cat, priority, 'yes' if impl else 'no',
                           'yes' if test else 'no', action])

    print(f"Detailed plan saved to: {output_csv}")
    print()

if __name__ == '__main__':
    main()
