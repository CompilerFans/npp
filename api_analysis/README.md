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

- `coverage.csv` - Legacy-compatible coverage status for each function
  (`implemented=yes` only counts real code, not `NPP_NOT_IMPLEMENTED_ERROR`
  stubs)
- `coverage_summary.md` - Summary statistics by module
- `api_catalog.csv` - Public API contract parsed from `API/` with family,
  dtype, channel/layout, in-place and Ctx axes plus a `real`/`stub`/`missing`
  status
- `backlog.csv` - Family-level backlog (remaining functions split into Ctx /
  non-Ctx) used to define implementation work packages

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
python3 tools/npp_api_catalog.py --update
```

Useful filters while working on a family:

```bash
python3 tools/npp_api_catalog.py --group nppsSum --list
python3 tools/npp_api_catalog.py --module nppi_statistics --status missing
```

`tools/check_coverage.py` is kept for the legacy 12.2 CSV view, but it cannot
distinguish a real implementation from a `NPP_NOT_IMPLEMENTED_ERROR` stub and
is superseded by `npp_api_catalog.py`.

Outputs:
- `coverage.csv` - Implementation and test status for each function
- `coverage_summary.md` - Coverage statistics by module
- `api_catalog.csv` - Public contract + family axes + corrected status
- `backlog.csv` - Family-level remaining work

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
