This script automates the complete MPP setup process:
1. Create wrapper directory structure
2. Generate function mappings (from original NPP API)
3. Generate wrapper files (from original NPP API)
4. Convert NPP source code to MPP naming
5. Convert CUDA Runtime API to MACA Runtime API
6. Copy MPP headers to wrapper/include
7. Build MPP library
8. Deploy to MACA SDK and cu-bridge (optional)

IMPORTANT: Steps 1-3 must run BEFORE step 4, because wrapper generation
needs to scan the original NPP API headers to create npp->mpp mappings.

Usage:
  ./setup_mpp.sh [options]

Options:
  --maca            Path to MACA SDK directory
  --cu-bridge       Path to cu-bridge directory
  --skip-convert    Skip NPP to MPP conversion
  --skip-build      Skip library build
  --skip-deploy     Skip deployment
  --dry-run         Preview changes without applying
  -h, --help        Show this help message

Environment Variables (used by deploy step):
  MACA_PATH     If set, will prompt to use as default MACA SDK path
  CUCC_PATH     If set, will prompt to use as default cu-bridge path

============================================================================
Individual Scripts (can be used standalone)
============================================================================

1. convert_npp2mpp.py - Convert NPP naming to MPP naming (Generic Rules)
   Uses generic prefix replacement rules (npp*->mpp*, Npp*->Mpp*, NPP_*->MPP_*)
   Scope: src/, test/, cmake/, examples/, API/ directories and root files
   Usage:
     python3 mpp_scripts/convert_npp2mpp.py --preview  # Preview changes
     python3 mpp_scripts/convert_npp2mpp.py --backup   # Create backup in .npp_backup/
     python3 mpp_scripts/convert_npp2mpp.py --apply    # Apply conversion
     python3 mpp_scripts/convert_npp2mpp.py --restore  # Restore from backup

2. generate_function_mappings.py - Generate function name mappings
   Scans API headers and generates npp2mpp_functions.h with #define mappings.
   IMPORTANT: Must run BEFORE convert_npp2mpp.py (needs original NPP headers)
   Usage:
     python3 mpp_scripts/generate_function_mappings.py
   Output: wrapper/npp2mpp_functions.h

3. generate_npp_wrapper.py - Generate NPP compatibility wrapper files
   Creates type/enum/constant mappings and user-facing NPP headers.
   Usage:
     python3 mpp_scripts/generate_npp_wrapper.py
   Output:
     wrapper/*_wrapper.h        - Bridge headers (NPP types -> MPP types)
     wrapper/user_include/*.h   - User-facing NPP headers

4. deploy_mpp.sh - Deploy MPP to MACA SDK and cu-bridge
   Copies built libraries (libmppc + stubs) and headers to target directories.
   Usage:
     ./mpp_scripts/deploy_mpp.sh --maca <maca_sdk_path> --cu-bridge <cu_bridge_path>
     ./mpp_scripts/deploy_mpp.sh --maca /opt/maca-3.3.0 --cu-bridge ~/cu-bridge --dry-run

5. convert_cuda2maca.py - Convert CUDA Runtime API to MACA Runtime API (Generic Rules)
   Uses generic prefix replacement (cuda*->mc*) with exceptions for special cases.
   Scope: src/, test/, examples/, API/ directories (wrapper/ excluded for compatibility)
   Usage:
     python3 mpp_scripts/convert_cuda2maca.py --check    # Preview changes
     python3 mpp_scripts/convert_cuda2maca.py --backup   # Create backup in .cuda_backup/
     python3 mpp_scripts/convert_cuda2maca.py --apply    # Apply conversion
     python3 mpp_scripts/convert_cuda2maca.py --restore  # Restore from backup

6. update_cubridge_link.sh - Update cu-bridge link.json with NPP->MPP mappings
   Automatically adds library path mappings for cu-bridge.
   Usage:
     ./mpp_scripts/update_cubridge_link.sh              # Update link.json
     ./mpp_scripts/update_cubridge_link.sh --dry-run    # Preview changes
     ./mpp_scripts/update_cubridge_link.sh --restore    # Restore backup