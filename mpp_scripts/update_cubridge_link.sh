#!/bin/bash
#
# Update cu-bridge link.json to add NPP to MPP library mappings
#
# Usage:
#   ./update_cubridge_link.sh [--cu-bridge <path>] [--dry-run] [--restore]
#
# Options:
#   --cu-bridge <path>  Path to cu-bridge directory (default: $CUCC_PATH)
#   --dry-run           Show what would be done without making changes
#   --restore           Restore from backup file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
CU_BRIDGE="${CUCC_PATH:-}"
DRY_RUN=false
RESTORE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cu-bridge)
            CU_BRIDGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --restore)
            RESTORE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--cu-bridge <path>] [--dry-run] [--restore]"
            echo ""
            echo "Options:"
            echo "  --cu-bridge <path>  Path to cu-bridge directory (default: \$CUCC_PATH)"
            echo "  --dry-run           Show what would be done without making changes"
            echo "  --restore           Restore from backup file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find cu-bridge path
if [ -z "$CU_BRIDGE" ]; then
    echo "Error: cu-bridge path not specified."
    echo "Please set CUCC_PATH environment variable or use --cu-bridge option."
    exit 1
fi

LINK_JSON="$CU_BRIDGE/tools/link.json"
BACKUP_FILE="$LINK_JSON.mpp_backup"

if [ ! -f "$LINK_JSON" ]; then
    echo "Error: link.json not found at: $LINK_JSON"
    exit 1
fi

# Restore mode
if [ "$RESTORE" = true ]; then
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "Error: Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would restore $LINK_JSON from $BACKUP_FILE"
    else
        cp "$BACKUP_FILE" "$LINK_JSON"
        echo "Restored $LINK_JSON from backup"
    fi
    exit 0
fi

# Check if NPP mappings already exist
if grep -q '"libnppc.so"' "$LINK_JSON" 2>/dev/null; then
    echo "NPP library mappings already exist in $LINK_JSON"
    echo "Use --restore to revert to backup, then re-run this script if needed."
    exit 0
fi

# NPP to MPP library mappings
NPP_MAPPINGS='
        {"dir":"lib64",  "ln":"libnppc.so",    "src":"$MACA_PATH/lib/libmppc.so"},
        {"dir":"lib64",  "ln":"libnppial.so",  "src":"$MACA_PATH/lib/libmppial.so"},
        {"dir":"lib64",  "ln":"libnppicc.so",  "src":"$MACA_PATH/lib/libmppicc.so"},
        {"dir":"lib64",  "ln":"libnppidei.so", "src":"$MACA_PATH/lib/libmppidei.so"},
        {"dir":"lib64",  "ln":"libnppif.so",   "src":"$MACA_PATH/lib/libmppif.so"},
        {"dir":"lib64",  "ln":"libnppig.so",   "src":"$MACA_PATH/lib/libmppig.so"},
        {"dir":"lib64",  "ln":"libnppim.so",   "src":"$MACA_PATH/lib/libmppim.so"},
        {"dir":"lib64",  "ln":"libnppist.so",  "src":"$MACA_PATH/lib/libmppist.so"},
        {"dir":"lib64",  "ln":"libnppisu.so",  "src":"$MACA_PATH/lib/libmppisu.so"},
        {"dir":"lib64",  "ln":"libnppitc.so",  "src":"$MACA_PATH/lib/libmppitc.so"},
        {"dir":"lib64",  "ln":"libnpps.so",    "src":"$MACA_PATH/lib/libmpps.so"},
'

echo "=== Update cu-bridge link.json for NPP/MPP mapping ==="
echo "Target file: $LINK_JSON"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would add the following mappings:"
    echo "$NPP_MAPPINGS"
    echo ""
    echo "[DRY-RUN] No changes made."
    exit 0
fi

# Create backup
cp "$LINK_JSON" "$BACKUP_FILE"
echo "Created backup: $BACKUP_FILE"

# Find insertion point: after libnvToolsExt.so line
# Use Python for reliable JSON-aware insertion
python3 << EOF
import json
import re

link_json_path = "$LINK_JSON"

# Read the file as text to preserve formatting
with open(link_json_path, 'r') as f:
    content = f.read()

# NPP mappings to insert in lib64
npp_mappings_lib64 = '''
        {"dir":"lib64",  "ln":"libnppc.so",    "src":"\$MACA_PATH/lib/libmppc.so"},
        {"dir":"lib64",  "ln":"libnppial.so",  "src":"\$MACA_PATH/lib/libmppial.so"},
        {"dir":"lib64",  "ln":"libnppicc.so",  "src":"\$MACA_PATH/lib/libmppicc.so"},
        {"dir":"lib64",  "ln":"libnppidei.so", "src":"\$MACA_PATH/lib/libmppidei.so"},
        {"dir":"lib64",  "ln":"libnppif.so",   "src":"\$MACA_PATH/lib/libmppif.so"},
        {"dir":"lib64",  "ln":"libnppig.so",   "src":"\$MACA_PATH/lib/libmppig.so"},
        {"dir":"lib64",  "ln":"libnppim.so",   "src":"\$MACA_PATH/lib/libmppim.so"},
        {"dir":"lib64",  "ln":"libnppist.so",  "src":"\$MACA_PATH/lib/libmppist.so"},
        {"dir":"lib64",  "ln":"libnppisu.so",  "src":"\$MACA_PATH/lib/libmppisu.so"},
        {"dir":"lib64",  "ln":"libnppitc.so",  "src":"\$MACA_PATH/lib/libmppitc.so"},
        {"dir":"lib64",  "ln":"libnpps.so",    "src":"\$MACA_PATH/lib/libmpps.so"},
'''

# NPP mappings to insert in lib64/stubs
npp_mappings_stubs = '''
        {"dir":"lib64/stubs",  "ln":"libnppc.so",    "src":"\$MACA_PATH/lib/libmppc.so"},
        {"dir":"lib64/stubs",  "ln":"libnppial.so",  "src":"\$MACA_PATH/lib/libmppial.so"},
        {"dir":"lib64/stubs",  "ln":"libnppicc.so",  "src":"\$MACA_PATH/lib/libmppicc.so"},
        {"dir":"lib64/stubs",  "ln":"libnppidei.so", "src":"\$MACA_PATH/lib/libmppidei.so"},
        {"dir":"lib64/stubs",  "ln":"libnppif.so",   "src":"\$MACA_PATH/lib/libmppif.so"},
        {"dir":"lib64/stubs",  "ln":"libnppig.so",   "src":"\$MACA_PATH/lib/libmppig.so"},
        {"dir":"lib64/stubs",  "ln":"libnppim.so",   "src":"\$MACA_PATH/lib/libmppim.so"},
        {"dir":"lib64/stubs",  "ln":"libnppist.so",  "src":"\$MACA_PATH/lib/libmppist.so"},
        {"dir":"lib64/stubs",  "ln":"libnppisu.so",  "src":"\$MACA_PATH/lib/libmppisu.so"},
        {"dir":"lib64/stubs",  "ln":"libnppitc.so",  "src":"\$MACA_PATH/lib/libmppitc.so"},
        {"dir":"lib64/stubs",  "ln":"libnpps.so",    "src":"\$MACA_PATH/lib/libmpps.so"}
'''

# Step 1: Insert lib64 mappings after libnvToolsExt.so
pattern = r'(\{"dir":"lib64"[^}]*"libnvToolsExt\.so"[^}]*\},?\s*\n)'
match = re.search(pattern, content)

if match:
    insert_pos = match.end()
    content = content[:insert_pos] + "\n" + npp_mappings_lib64 + content[insert_pos:]
else:
    # Fallback: find lib64/stubs section and insert before it
    pattern = r'(\s*\{"dir":"lib64/stubs")'
    match = re.search(pattern, content)
    if match:
        insert_pos = match.start()
        content = content[:insert_pos] + "\n" + npp_mappings_lib64 + content[insert_pos:]
    else:
        print("Error: Could not find insertion point for lib64 mappings")
        exit(1)

# Step 2: Insert lib64/stubs mappings after libnvrtc.so in stubs section
pattern = r'(\{"dir":"lib64/stubs"[^}]*"libnvrtc\.so"[^}]*\})'
match = re.search(pattern, content)

if match:
    insert_pos = match.end()
    content = content[:insert_pos] + "," + npp_mappings_stubs + content[insert_pos:]
else:
    # Fallback: find end of lib64/stubs section (before pre_cmake)
    pattern = r'(\{"dir":"lib64/stubs"[^}]*\})\s*\n\s*\]'
    match = re.search(pattern, content)
    if match:
        insert_pos = match.end(1)
        content = content[:insert_pos] + "," + npp_mappings_stubs + content[insert_pos:]
    else:
        print("Warning: Could not find insertion point for lib64/stubs mappings")

with open(link_json_path, 'w') as f:
    f.write(content)

print("Successfully added NPP library mappings")
EOF

echo ""
echo "=== NPP/MPP library mappings added ==="
echo ""
echo "Mapping summary (lib64 and lib64/stubs):"
echo "  libnppc.so    -> libmppc.so     (Core)"
echo "  libnppial.so  -> libmppial.so   (Image Arithmetic and Logical)"
echo "  libnppicc.so  -> libmppicc.so   (Image Color Conversion)"
echo "  libnppidei.so -> libmppidei.so  (Image Data Exchange and Initialization)"
echo "  libnppif.so   -> libmppif.so    (Image Filtering)"
echo "  libnppig.so   -> libmppig.so    (Image Geometry Transforms)"
echo "  libnppim.so   -> libmppim.so    (Image Morphological)"
echo "  libnppist.so  -> libmppist.so   (Image Statistics)"
echo "  libnppisu.so  -> libmppisu.so   (Image Support/Utility)"
echo "  libnppitc.so  -> libmppitc.so   (Image Threshold and Compare)"
echo "  libnpps.so    -> libmpps.so     (Signal Processing)"
echo ""
echo "Directories updated:"
echo "  - lib64/"
echo "  - lib64/stubs/"
echo ""
echo "To restore original: $0 --cu-bridge $CU_BRIDGE --restore"
