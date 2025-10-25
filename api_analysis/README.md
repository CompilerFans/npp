# API Analysis

API analysis and coverage tracking for NPP library implementation.

## Files

### API Inventory

- `api_functions_11.4.csv` - Complete function list for CUDA 11.4
- `api_functions_12.2.csv` - Complete function list for CUDA 12.2
- `api_functions_12.8.csv` - Complete function list for CUDA 12.8
- `api_inventory.json` - Complete inventory with version comparisons
- `api_report.md` - Summary report with statistics

### Coverage Reports

- `coverage.csv` - Detailed coverage status for each function
- `coverage_summary.md` - Summary statistics by module

## Usage

### Generate API Inventory

Analyze API headers and generate function lists:

```bash
# Analyze single version
python3 tools/analyze_api.py --versions 12.2

# Analyze multiple versions with comparison
python3 tools/analyze_api.py --versions 11.4 12.2 12.8
```

Outputs:
- `api_functions_{version}.csv` - Function list for each version
- `api_inventory.json` - Complete inventory with version differences
- `api_report.md` - Statistics and comparison report

### Update Coverage

Analyze implementation and test coverage:

```bash
./update_coverage.sh
```

Or manually:

```bash
python3 tools/check_coverage.py
```

Outputs:
- `coverage.csv` - Implementation and test status for each function
- `coverage_summary.md` - Coverage statistics by module

## CSV Formats

### api_functions_{version}.csv

Columns:
- `function_name` - API function name
- `module` - Header file name
- `return_type` - Return type
- `params` - Parameter list
- `signature` - Complete function signature

### coverage.csv

Columns:
- `function_name` - API function name
- `module` - Header file name
- `implemented` - yes/no
- `impl_file` - Implementation file path
- `tested` - yes/no
- `test_file` - Test file path

## Version Switching

Switch between CUDA SDK versions:

```bash
./switch_cuda_version.sh 11.4
./switch_cuda_version.sh 12.2
./switch_cuda_version.sh 12.8
```

Note: Rebuild project after switching versions.
