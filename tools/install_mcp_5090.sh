#!/usr/bin/env bash
# Install the 5090 remote-execution MCP server for the current user.
#
#   tools/install_mcp_5090.sh
#
# Installs a stable copy under ~/.local/share/mcpp-5090/, writes a reusable
# config at ~/.config/mcpp-5090-mcp.json (used by headless subagents via
# --mcp-config), and registers the server at user scope for interactive
# Claude Code sessions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/mcpp-5090"
DEST="$DEST_DIR/mcp_5090_server.py"
CONFIG="$HOME/.config/mcpp-5090-mcp.json"
CREDS="${MCPP_5090_ENV:-$HOME/.config/mcpp-5090.env}"

mkdir -p "$DEST_DIR" "$(dirname "$CONFIG")"
install -m 0755 "$ROOT/tools/mcp_5090_server.py" "$DEST"

python3 - "$CONFIG" "$DEST" "$CREDS" <<'PY'
import json, sys
config, server, creds = sys.argv[1:4]
data = {
    "mcpServers": {
        "mcpp-5090": {
            "command": "python3",
            "args": [server],
            "env": {"MCPP_5090_ENV": creds},
        }
    }
}
with open(config, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY

# Register at user scope (idempotent).
claude mcp remove -s user mcpp-5090 >/dev/null 2>&1 || true
claude mcp add -s user mcpp-5090 -e "MCPP_5090_ENV=$CREDS" -- python3 "$DEST" >/dev/null

echo "installed: $DEST"
echo "config   : $CONFIG"
echo "scope    : user"
echo
claude mcp list
