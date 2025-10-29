# MPP Documentation

Documentation for MPP (Modern Performance Primitives) library.

## Directory Structure

### 📐 [resize/](resize/)
Image resize and geometric transform documentation
- `resize_refactoring_analysis.md` - Resize implementation refactoring analysis
- `test_resize_refactoring_analysis.md` - Resize test refactoring details

### 🔍 [super_sampling/](super_sampling/)
Super sampling algorithm documentation and analysis
- `supersampling_analysis.md` - Super sampling algorithm analysis
- `supersampling_summary.md` - Quick summary of super sampling approach
- `super_sampling_v2_usage.md` - V2 implementation usage guide
- `kunzmi_mpp_super_analysis.md` - Analysis of kunzmi/mpp super sampling reference
- `cpu_vs_cuda_implementation_analysis.md` - Detailed comparison of CPU reference vs CUDA implementation
- `border_test_readme.md` - Border handling test documentation

### 🎨 [filtering/](filtering/)
Image filtering operations and boundary handling
- `filter_box_boundary_modes.md` - Box filter boundary mode documentation
- `npp_filterbox_algorithm_analysis.md` - Box filter algorithm analysis
- `npp_filterbox_boundary_analysis.md` - Boundary handling analysis for box filter
- `NVIDIA_NPP_BEHAVIOR_ANALYSIS.md` - NVIDIA NPP behavior analysis

### 🔲 [morphology/](morphology/)
Morphological operations documentation
- `morphology_boundary_modes.md` - Morphology boundary mode handling
- `README_Morphology_Parameterized_Tests.md` - Parameterized test suite for morphology

### 🏗️ [project/](project/)
Project architecture and goals
- `project_architecture.md` - Overall project architecture
- `project_goal.md` - Project goals and objectives
- `opencv_used_npp.md` - OpenCV's usage of NPP functions

### 🧪 [testing/](testing/)
Testing strategies and compatibility
- `nvidia_npp_compatibility_testing.md` - NVIDIA NPP compatibility testing guide

### 📚 [ref_code/super_sampling_cpu_reference/](../ref_code/super_sampling_cpu_reference/)
CPU reference implementation with extensive documentation (see that directory for details)

## Quick Links

### For New Contributors
1. Start with [project/project_goal.md](project/project_goal.md)
2. Read [project/project_architecture.md](project/project_architecture.md)
3. Review [testing/nvidia_npp_compatibility_testing.md](testing/nvidia_npp_compatibility_testing.md)

### For Super Sampling Development
1. [super_sampling/supersampling_summary.md](super_sampling/supersampling_summary.md) - Quick overview
2. [super_sampling/cpu_vs_cuda_implementation_analysis.md](super_sampling/cpu_vs_cuda_implementation_analysis.md) - Implementation comparison
3. [ref_code/super_sampling_cpu_reference/](../ref_code/super_sampling_cpu_reference/) - CPU reference code and tests

### For Algorithm Understanding
- Box Filter: [filtering/npp_filterbox_algorithm_analysis.md](filtering/npp_filterbox_algorithm_analysis.md)
- Super Sampling: [super_sampling/supersampling_analysis.md](super_sampling/supersampling_analysis.md)
- Morphology: [morphology/README_Morphology_Parameterized_Tests.md](morphology/README_Morphology_Parameterized_Tests.md)

## Other Resources

- `NPP_API.xlsx` - NVIDIA NPP API reference spreadsheet
