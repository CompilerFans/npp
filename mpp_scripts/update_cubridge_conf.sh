#!/bin/bash
#
# Update cu-bridge conf.json to add NPP to MPP library parameter substitution
#
# Usage:
#   ./update_cubridge_conf.sh [--cu-bridge <path>] [--dry-run] [--restore]
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

CONF_JSON="$CU_BRIDGE/bin/conf.json"
BACKUP_FILE="$CONF_JSON.mpp_backup"

if [ ! -f "$CONF_JSON" ]; then
    echo "Error: conf.json not found at: $CONF_JSON"
    exit 1
fi

# Restore mode
if [ "$RESTORE" = true ]; then
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "Error: Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would restore $CONF_JSON from $BACKUP_FILE"
    else
        cp "$BACKUP_FILE" "$CONF_JSON"
        echo "Restored $CONF_JSON from backup"
    fi
    exit 0
fi

# Check if NPP mappings already exist
if grep -q '"-lnppc"' "$CONF_JSON" 2>/dev/null; then
    echo "NPP library mappings already exist in $CONF_JSON"
    echo "Use --restore to revert to backup, then re-run this script if needed."
    exit 0
fi

echo "=== Update cu-bridge conf.json for NPP/MPP parameter substitution ==="
echo "Target file: $CONF_JSON"
echo ""

# NPP library mappings to add
NPP_MAPPINGS='"-lnppc","-lmppc",
			"-lnppial","-lmppial",
			"-lnppicc","-lmppicc",
			"-lnppidei","-lmppidei",
			"-lnppif","-lmppif",
			"-lnppig","-lmppig",
			"-lnppim","-lmppim",
			"-lnppist","-lmppist",
			"-lnppisu","-lmppisu",
			"-lnppitc","-lmppitc",
			"-lnpps","-lmpps",'

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would add the following mappings to 'all.replacer' section:"
    echo "$NPP_MAPPINGS"
    echo ""
    echo "[DRY-RUN] No changes made."
    exit 0
fi

# Create backup
cp "$CONF_JSON" "$BACKUP_FILE"
echo "Created backup: $BACKUP_FILE"

# Use Python for reliable JSON modification
python3 << EOF
import sys

conf_path = "$CONF_JSON"

with open(conf_path, 'r') as f:
    content = f.read()

# NPP library mappings to insert in replacer section
npp_mappings = [
    '"-lnppc","-lmppc"',
    '"-lnppial","-lmppial"',
    '"-lnppicc","-lmppicc"',
    '"-lnppidei","-lmppidei"',
    '"-lnppif","-lmppif"',
    '"-lnppig","-lmppig"',
    '"-lnppim","-lmppim"',
    '"-lnppist","-lmppist"',
    '"-lnppisu","-lmppisu"',
    '"-lnppitc","-lmppitc"',
    '"-lnpps","-lmpps"',
]

# Step 1: Add library mappings to replacer section
insert_marker = '"-lnvjpeg","-lmcjpeg"'

if insert_marker in content:
    marker_pos = content.find(insert_marker)
    end_of_line = content.find('\n', marker_pos)
    if end_of_line != -1:
        new_mappings = ""
        for mapping in npp_mappings:
            new_mappings += "\n\t\t\t" + mapping + ","

        content = content[:end_of_line+1] + new_mappings + content[end_of_line+1:]
        print("Added NPP library mappings to replacer section")
    else:
        print("Error: Could not find line ending for replacer")
        sys.exit(1)
else:
    print(f"Error: Could not find marker in conf.json")
    sys.exit(1)

# Step 2: Add include path to adder section
include_marker = '"-I \${MACA_PATH}/include/mchotspot"'
mcmpp_include = '"-I \${MACA_PATH}/include/mcmpp"'

if mcmpp_include not in content:
    if include_marker in content:
        marker_pos = content.find(include_marker)
        end_of_line = content.find('\n', marker_pos)
        if end_of_line != -1:
            content = content[:end_of_line+1] + '\t\t\t"-I \${MACA_PATH}/include/mcmpp",\n' + content[end_of_line+1:]
            print("Added mcmpp include path to adder section")
    else:
        print("Warning: Could not find adder insertion point for mcmpp include")
else:
    print("mcmpp include path already exists")

with open(conf_path, 'w') as f:
    f.write(content)

print("Successfully updated conf.json")
EOF

echo ""
echo "=== NPP/MPP configuration added ==="
echo ""
echo "Library substitution (replacer section):"
echo "  -lnppc    -> -lmppc     (Core)"
echo "  -lnppial  -> -lmppial   (Image Arithmetic and Logical)"
echo "  -lnppicc  -> -lmppicc   (Image Color Conversion)"
echo "  -lnppidei -> -lmppidei  (Image Data Exchange and Initialization)"
echo "  -lnppif   -> -lmppif    (Image Filtering)"
echo "  -lnppig   -> -lmppig    (Image Geometry Transforms)"
echo "  -lnppim   -> -lmppim    (Image Morphological)"
echo "  -lnppist  -> -lmppist   (Image Statistics)"
echo "  -lnppisu  -> -lmppisu   (Image Support/Utility)"
echo "  -lnppitc  -> -lmppitc   (Image Threshold and Compare)"
echo "  -lnpps    -> -lmpps     (Signal Processing)"
echo ""
echo "Include path (adder section):"
echo "  -I \${MACA_PATH}/include/mcmpp"
echo ""
echo "To restore original: $0 --cu-bridge $CU_BRIDGE --restore"
