#!/usr/bin/env python3
"""Dependency-free MCP stdio server for the RTX 5090 reference machine.

Tools (all scoped to the repository checkout that started the server):
  sync     rsync the local checkout to a remote working directory
  build    configure + build the NVIDIA NPP test tree on the 5090
  test     run a gtest target on the 5090
  record   run a target with NPP_RECORD_DIR on the 5090
  pull     rsync a remote directory back into the checkout
  status   GPU / disk / CPU summary
  run      arbitrary remote shell command (escape hatch)

Parallel agents should pass a distinct `remote_dir` (for example
~/mcMPP-ref-<agent>) so their builds and records do not collide.

Credentials and host defaults come from ~/.config/mcpp-5090.env:
  export SSHPASS='...'
  export MCPP_5090_HOST='metax@10.2.120.6'
  export MCPP_5090_DIR='~/mcMPP'
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SERVER_NAME = "mcpp-5090"
SERVER_VERSION = "1.0.0"
ROOT = Path(os.environ.get("MCPP_REPO_ROOT") or Path.cwd()).resolve()
ENV_FILE = Path(os.environ.get("MCPP_5090_ENV", Path.home() / ".config" / "mcpp-5090.env"))
DEFAULT_TIMEOUT = 1800
MAX_OUTPUT = 24000


def log(msg: str) -> None:
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


def load_env() -> None:
    if not ENV_FILE.is_file():
        log(f"warning: {ENV_FILE} not found; relying on the process environment")
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def host() -> str:
    return os.environ.get("MCPP_5090_HOST", "metax@10.2.120.6")


def remote_default() -> str:
    return os.environ.get("MCPP_5090_DIR", "~/mcMPP")


SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]


def ssh_cmd(remote_dir: Optional[str], command: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    remote_dir = remote_dir or remote_default()
    path = "export PATH=$HOME/tools/cmake/bin:$HOME/tools:/usr/local/cuda/bin:$PATH"
    full = f"{path}; cd {shlex.quote(remote_dir)} && {command}"
    argv = ["sshpass", "-e", "ssh", *SSH_OPTS, host(), full]
    return run(argv, timeout)


def run(argv: List[str], timeout: int) -> Dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n... (truncated, {len(out)} chars total)"
        return {"exit_code": proc.returncode, "output": out}
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return {"exit_code": 124, "output": out + f"\n... timed out after {timeout}s"}


def tool_sync(args: Dict[str, Any]) -> Dict[str, Any]:
    remote_dir = args.get("remote_dir") or remote_default()
    local = Path(args.get("repo_path") or ROOT)
    delete = bool(args.get("delete", False))
    exclude = [
        "--exclude=.git", "--exclude=build", "--exclude=build-*",
        "--exclude=build_maca*", "--exclude=.claude", "--exclude=artifacts",
        "--exclude=tools/__pycache__", "--exclude=*.pyc", "--exclude=.npp_backup_full",
    ]
    argv = ["sshpass", "-e", "rsync", "-az"]
    if delete:
        argv.append("--delete")
    argv += exclude + ["-e", "ssh " + " ".join(SSH_OPTS), str(local) + "/", f"{host()}:{remote_dir}/"]
    return run(argv, DEFAULT_TIMEOUT)


def tool_build(args: Dict[str, Any]) -> Dict[str, Any]:
    remote_dir = args.get("remote_dir") or remote_default()
    build_dir = args.get("build_dir") or "build-nvidia"
    jobs = int(args.get("jobs", 8))
    cfg = (
        f"cmake -S . -B {shlex.quote(build_dir)} -G Ninja -DCMAKE_BUILD_TYPE=Release "
        "-DBUILD_TESTS=ON -DBUILD_BENCHMARK=OFF -DNPP_WARNINGS_AS_ERRORS=OFF "
        "-DUSE_NVIDIA_NPP=ON >/tmp/mcp_cfg.log 2>&1"
    )
    cmd = f"{cfg} && cmake --build {shlex.quote(build_dir)} -j{jobs}"
    return ssh_cmd(remote_dir, cmd, int(args.get("timeout", DEFAULT_TIMEOUT)))


def tool_test(args: Dict[str, Any]) -> Dict[str, Any]:
    remote_dir = args.get("remote_dir") or remote_default()
    build_dir = args.get("build_dir") or "build-nvidia"
    target = args.get("target") or "diff_tests"
    filt = args.get("filter")
    env = args.get("env") or {}
    env_str = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    cmd = f"cd {shlex.quote(build_dir)} && {env_str} ./{shlex.quote(target)}"
    if filt:
        cmd += f" --gtest_filter={shlex.quote(str(filt))}"
    return ssh_cmd(remote_dir, cmd, int(args.get("timeout", DEFAULT_TIMEOUT)))


def tool_record(args: Dict[str, Any]) -> Dict[str, Any]:
    remote_dir = args.get("remote_dir") or remote_default()
    build_dir = args.get("build_dir") or "build-nvidia"
    out = args.get("out_dir") or f"{remote_dir.rstrip('/')}/{build_dir}/npp_records"
    filt = args.get("filter")
    cmd = f"mkdir -p {shlex.quote(out)} && cd {shlex.quote(build_dir)} && NPP_RECORD_DIR={shlex.quote(out)} ./diff_tests"
    if filt:
        cmd += f" --gtest_filter={shlex.quote(str(filt))}"
    result = ssh_cmd(remote_dir, cmd, int(args.get("timeout", DEFAULT_TIMEOUT)))
    result["record_dir"] = out
    return result


def tool_pull(args: Dict[str, Any]) -> Dict[str, Any]:
    remote_dir = args.get("remote_dir") or remote_default()
    build_dir = args.get("build_dir") or "build-nvidia"
    src = args.get("src") or f"{remote_dir.rstrip('/')}/{build_dir}/npp_records"
    dest = Path(args.get("dest") or (ROOT / "artifacts" / "5090" / "npp_records"))
    dest.mkdir(parents=True, exist_ok=True)
    argv = ["sshpass", "-e", "rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS),
            f"{host()}:{src}/", str(dest) + "/"]
    return run(argv, DEFAULT_TIMEOUT)


def tool_status(args: Dict[str, Any]) -> Dict[str, Any]:
    cmd = (
        "hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; "
        "nproc; df -h ~ | tail -1; uptime"
    )
    return run(["sshpass", "-e", "ssh", *SSH_OPTS, host(), cmd], 120)


def tool_run(args: Dict[str, Any]) -> Dict[str, Any]:
    command = args.get("command")
    if not command:
        return {"exit_code": 2, "output": "missing command"}
    return ssh_cmd(args.get("remote_dir"), command, int(args.get("timeout", DEFAULT_TIMEOUT)))


TOOLS = [
    {
        "name": "sync",
        "description": "rsync the local checkout to the 5090 (excludes build dirs, .git, artifacts).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "local checkout (default: server cwd)"},
                "remote_dir": {"type": "string", "description": "remote directory (default ~/mcMPP)"},
                "delete": {"type": "boolean", "description": "delete remote files not present locally"},
            },
        },
    },
    {
        "name": "build",
        "description": "configure and build the NVIDIA NPP test tree on the 5090.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "remote_dir": {"type": "string"},
                "build_dir": {"type": "string", "description": "default build-nvidia"},
                "jobs": {"type": "integer", "description": "parallel build jobs (default 8)"},
                "timeout": {"type": "integer"},
            },
        },
    },
    {
        "name": "test",
        "description": "run a gtest target (default diff_tests) on the 5090.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "remote_dir": {"type": "string"},
                "build_dir": {"type": "string"},
                "target": {"type": "string"},
                "filter": {"type": "string"},
                "env": {"type": "object", "description": "extra environment variables"},
                "timeout": {"type": "integer"},
            },
        },
    },
    {
        "name": "record",
        "description": "run diff_tests with NPP_RECORD_DIR to capture NVIDIA NPP goldens.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "remote_dir": {"type": "string"},
                "build_dir": {"type": "string"},
                "filter": {"type": "string"},
                "out_dir": {"type": "string"},
                "timeout": {"type": "integer"},
            },
        },
    },
    {
        "name": "pull",
        "description": "rsync recorded artifacts from the 5090 into the local checkout.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "remote_dir": {"type": "string"},
                "build_dir": {"type": "string"},
                "src": {"type": "string"},
                "dest": {"type": "string"},
            },
        },
    },
    {
        "name": "status",
        "description": "GPU, CPU and disk status of the 5090.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["summary"],
                           "description": "status target (always 'summary')"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "run",
        "description": "run an arbitrary shell command in the remote checkout.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "remote_dir": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
]

for _tool in TOOLS:
    _tool["inputSchema"].setdefault("additionalProperties", False)

HANDLERS = {
    "sync": tool_sync,
    "build": tool_build,
    "test": tool_test,
    "record": tool_record,
    "pull": tool_pull,
    "status": tool_status,
    "run": tool_run,
}


def result_text(result: Dict[str, Any]) -> str:
    return f"exit_code={result.get('exit_code')}\n{result.get('output', '')}"


def handle(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = msg.get("method")
    msg_id = msg.get("id")
    if msg_id is None:
        return None  # notification
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        handler = HANDLERS.get(name)
        if handler is None:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32602, "message": f"unknown tool {name}"}}
        try:
            result = handler(params.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": result_text(result)}],
                               "isError": result.get("exit_code") not in (0, None)}}
        except Exception as exc:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"error: {exc}"}],
                               "isError": True}}
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> int:
    load_env()
    log(f"serving host={host()} default_dir={remote_default()}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log(f"invalid json: {line[:200]}")
            continue
        response = handle(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
