#!/usr/bin/env python3
"""
QEMU Swarm Health Oracle (ADR-062)

Validates collective health of a multi-node ESP32-S3 QEMU swarm.
Checks cross-node assertions like TDM ordering, inter-node communication,
and swarm-level frame rates.

Usage:
    python3 swarm_health.py --config swarm_config.yaml --log-dir build/swarm_logs/
    python3 swarm_health.py --log-dir build/swarm_logs/ --assertions all_nodes_boot no_crashes
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ANSI helpers (disabled when not a TTY)
# ---------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t: str) -> str:
    return _color(t, "32")


def yellow(t: str) -> str:
    return _color(t, "33")


def red(t: str) -> str:
    return _color(t, "1;31")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AssertionResult:
    """Result of a single swarm-level assertion."""
    name: str
    passed: bool
    message: str
    severity: int  # 0 = pass, 1 = warn, 2 = fail


@dataclass
class NodeLog:
    """Parsed log for a single QEMU node."""
    node_id: int
    lines: List[str]
    text: str


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_logs(log_dir: Path, node_count: int) -> List[NodeLog]:
    """Load qemu_node{i}.log (or node_{i}.log fallback) from *log_dir*."""
    logs: List[NodeLog] = []
    for i in range(node_count):
        path = log_dir / f"qemu_node{i}.log"
        if not path.exists():
            path = log_dir / f"node_{i}.log"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = ""
        logs.append(NodeLog(node_id=i, lines=text.splitlines(), text=text))
    return logs


def _node_count_from_dir(log_dir: Path) -> int:
    """Auto-detect node count by scanning for qemu_node*.log (or node_*.log) files."""
    count = 0
    while (log_dir / f"qemu_node{count}.log").exists() or (log_dir / f"node_{count}.log").exists():
        count += 1
    return count


# ---------------------------------------------------------------------------
# Individual assertions
# ---------------------------------------------------------------------------

_BOOT_PATTERNS = [
    r"app_main\(\)", r"main_task:", r"main:", r"ESP32-S3 CSI Node",
]

_CRASH_PATTERNS = [
    r"Guru Meditation", r"assert failed", r"abort\(\)", r"panic",
    r"LoadProhibited", r"StoreProhibited", r"InstrFetchProhibited",
    r"IllegalInstruction", r"Unhandled debug exception", r"Fatal exception",
]

_HEAP_PATTERNS = [
    r"HEAP_ERROR", r"out of memory", r"heap_caps_alloc.*failed",
    r"malloc.*fail", r"heap corruption", r"CORRUPT HEAP",
    r"multi_heap", r"heap_lock",
]

_FRAME_PATTERNS = [
    r"frame", r"CSI", r"mock_csi", r"iq_data", r"subcarrier",
    r"csi_collector", r"enqueue",
]

_FALL_PATTERNS = [r"fall[=: ]+1", r"fall detected", r"fall_event"]


def assert_all_nodes_boot(logs: List[NodeLog], timeout_s: float = 10.0) -> AssertionResult:
    """Check each node's log for boot patterns."""
    missing: List[int] = []
    for nl in logs:
        found = any(
            re.search(p, nl.text) for p in _BOOT_PATTERNS
        )
        if not found:
            missing.append(nl.node_id)

    if not missing:
        return AssertionResult(
            name="all_nodes_boot", passed=True,
            message=f"All {len(logs)} nodes booted (timeout={timeout_s}s)",
            severity=0,
        )
    return AssertionResult(
        name="all_nodes_boot", passed=False,
        message=f"Nodes missing boot indicator: {missing}",
        severity=2,
    )


def assert_no_crashes(logs: List[NodeLog]) -> AssertionResult:
    """Check no node has crash patterns."""
    crashed: List[str] = []
    for nl in logs:
        for line in nl.lines:
            for pat in _CRASH_PATTERNS:
                if re.search(pat, line):
                    crashed.append(f"node_{nl.node_id}: {line.strip()[:100]}")
                    break
            if crashed and crashed[-1].startswith(f"node_{nl.node_id}:"):
                break  # one crash per node is enough

    if not crashed:
        return AssertionResult(
            name="no_crashes", passed=True,
            message="No crash indicators in any node",
            severity=0,
        )
    return AssertionResult(
        name="no_crashes", passed=False,
        message=f"Crashes found: {crashed[0]}" + (
            f" (+{len(crashed)-1} more)" if len(crashed) > 1 else ""
        ),
        severity=2,
    )


def assert_tdm_no_collision(logs: List[NodeLog]) -> AssertionResult:
    """Parse TDM slot assignments from logs, verify uniqueness."""
    slot_map: Dict[int, List[int]] = {}  # slot -> [node_ids]
    tdm_pat = re.compile(r"tdm[_ ]?slot[=: ]+(\d+)", re.IGNORECASE)

    for nl in logs:
        for line in nl.lines:
            m = tdm_pat.search(line)
            if m:
                slot = int(m.group(1))
                slot_map.setdefault(slot, [])
                if nl.node_id not in slot_map[slot]:
                    slot_map[slot].append(nl.node_id)
                break  # first occurrence per node

    collisions = {s: nids for s, nids in slot_map.items() if len(nids) > 1}

    if not slot_map:
        return AssertionResult(
            name="tdm_no_collision", passed=True,
            message="No TDM slot assignments found (may be N/A)",
            severity=0,
        )
    if not collisions:
        return AssertionResult(
            name="tdm_no_collision", passed=True,
            message=f"TDM slots unique across {len(slot_map)} assignments",
            severity=0,
        )
    return AssertionResult(
        name="tdm_no_collision", passed=False,
        message=f"TDM collisions: {collisions}",
        severity=2,
    )


def assert_all_nodes_produce_frames(
    logs: List[NodeLog],
    sensor_ids: Optional[List[int]] = None,
) -> AssertionResult:
    """Each sensor node has CSI frame output.

    Args:
        logs: Parsed node logs.
        sensor_ids: If provided, only check these node IDs (skip coordinators).
                    If None, check all nodes (legacy behavior).
    """
    silent: List[int] = []
    for nl in logs:
        if sensor_ids is not None and nl.node_id not in sensor_ids:
            continue
        found = any(
            re.search(p, line, re.IGNORECASE)
            for line in nl.lines for p in _FRAME_PATTERNS
        )
        if not found:
            silent.append(nl.node_id)

    checked = len(sensor_ids) if sensor_ids is not None else len(logs)
    if not silent:
        return AssertionResult(
            name="all_nodes_produce_frames", passed=True,
            message=f"All {checked} checked nodes show frame activity",
            severity=0,
        )
    return AssertionResult(
        name="all_nodes_produce_frames", passed=False,
        message=f"Nodes with no frame activity: {silent}",
        severity=1,
    )


def assert_coordinator_receives_from_all(
    logs: List[NodeLog],
    coordinator_id: int = 0,
    sensor_ids: Optional[List[int]] = None,
) -> AssertionResult:
    """Coordinator log shows frames from each sensor's node_id."""
    coord_log = None
    for nl in logs:
        if nl.node_id == coordinator_id:
            coord_log = nl
            break

    if coord_log is None:
        return AssertionResult(
            name="coordinator_receives_from_all", passed=False,
            message=f"Coordinator node_{coordinator_id} log not found",
            severity=2,
        )

    if sensor_ids is None:
        sensor_ids = [nl.node_id for nl in logs if nl.node_id != coordinator_id]

    missing: List[int] = []
    recv_pat = re.compile(r"(from|node_id|src)[=: ]+(\d+)", re.IGNORECASE)
    received_ids: set = set()
    for line in coord_log.lines:
        m = recv_pat.search(line)
        if m:
            received_ids.add(int(m.group(2)))

    for sid in sensor_ids:
        if sid not in received_ids:
            missing.append(sid)

    if not missing:
        return AssertionResult(
            name="coordinator_receives_from_all", passed=True,
            message=f"Coordinator received from all sensors: {sensor_ids}",
            severity=0,
        )
    return AssertionResult(
        name="coordinator_receives_from_all", passed=False,
        message=f"Coordinator missing frames from nodes: {missing}",
        severity=1,
    )


def assert_fall_detected(logs: List[NodeLog], node_id: int) -> AssertionResult:
    """Specific node reports fall detection."""
    for nl in logs:
        if nl.node_id == node_id:
            found = any(
                re.search(p, line, re.IGNORECASE)
                for line in nl.lines for p in _FALL_PATTERNS
            )
            if found:
                return AssertionResult(
                    name=f"fall_detected_node_{node_id}", passed=True,
                    message=f"Node {node_id} reported fall event",
                    severity=0,
                )
            return AssertionResult(
                name=f"fall_detected_node_{node_id}", passed=False,
                message=f"Node {node_id} did not report fall event",
                severity=1,
            )

    return AssertionResult(
        name=f"fall_detected_node_{node_id}", passed=False,
        message=f"Node {node_id} log not found",
        severity=2,
    )


def assert_frame_rate_above(logs: List[NodeLog], min_fps: float = 10.0) -> AssertionResult:
    """Each node meets minimum frame rate."""
    fps_pat = re.compile(r"(?:fps|frame.?rate)[=: ]+([0-9.]+)", re.IGNORECASE)
    count_pat = re.compile(r"(?:frame[_ ]?count|frames)[=: ]+(\d+)", re.IGNORECASE)
    below: List[str] = []

    for nl in logs:
        best_fps: Optional[float] = None
        # Try explicit FPS
        for line in nl.lines:
            m = fps_pat.search(line)
            if m:
                try:
                    best_fps = max(best_fps or 0.0, float(m.group(1)))
                except ValueError:
                    pass
        # Fallback: estimate from frame count (assume 1-second intervals)
        if best_fps is None:
            counts = []
            for line in nl.lines:
                m = count_pat.search(line)
                if m:
                    try:
                        counts.append(int(m.group(1)))
                    except ValueError:
                        pass
            if len(counts) >= 2:
                best_fps = float(counts[-1] - counts[0]) / max(len(counts) - 1, 1)

        if best_fps is not None and best_fps < min_fps:
            below.append(f"node_{nl.node_id}={best_fps:.1f}")

    if not below:
        return AssertionResult(
            name="frame_rate_above", passed=True,
            message=f"All nodes meet minimum {min_fps} fps",
            severity=0,
        )
    return AssertionResult(
        name="frame_rate_above", passed=False,
        message=f"Nodes below {min_fps} fps: {', '.join(below)}",
        severity=1,
    )


def assert_max_boot_time(logs: List[NodeLog], max_seconds: float = 10.0) -> AssertionResult:
    """All nodes boot within N seconds (based on timestamp in log)."""
    boot_time_pat = re.compile(r"\((\d+)\)\s", re.IGNORECASE)
    slow: List[str] = []

    for nl in logs:
        boot_found = False
        for line in nl.lines:
            if any(re.search(p, line) for p in _BOOT_PATTERNS):
                boot_found = True
                m = boot_time_pat.search(line)
                if m:
                    ms = int(m.group(1))
                    if ms > max_seconds * 1000:
                        slow.append(f"node_{nl.node_id}={ms}ms")
                break
        if not boot_found:
            slow.append(f"node_{nl.node_id}=no_boot")

    if not slow:
        return AssertionResult(
            name="max_boot_time", passed=True,
            message=f"All nodes booted within {max_seconds}s",
            severity=0,
        )
    return AssertionResult(
        name="max_boot_time", passed=False,
        message=f"Slow/missing boot: {', '.join(slow)}",
        severity=1,
    )


def assert_no_heap_errors(logs: List[NodeLog]) -> AssertionResult:
    """No OOM/heap errors in any log."""
    errors: List[str] = []
    for nl in logs:
        for line in nl.lines:
            for pat in _HEAP_PATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    errors.append(f"node_{nl.node_id}: {line.strip()[:100]}")
                    break
            if errors and errors[-1].startswith(f"node_{nl.node_id}:"):
                break

    if not errors:
        return AssertionResult(
            name="no_heap_errors", passed=True,
            message="No heap errors in any node",
            severity=0,
        )
    return AssertionResult(
        name="no_heap_errors", passed=False,
        message=f"Heap errors: {errors[0]}" + (
            f" (+{len(errors)-1} more)" if len(errors) > 1 else ""
        ),
        severity=2,
    )


# ---------------------------------------------------------------------------
# Assertion registry & dispatcher
# ---------------------------------------------------------------------------

ASSERTION_REGISTRY: Dict[str, Any] = {
    "all_nodes_boot": assert_all_nodes_boot,
    "no_crashes": assert_no_crashes,
    "tdm_no_collision": assert_tdm_no_collision,
    "all_nodes_produce_frames": assert_all_nodes_produce_frames,
    "coordinator_receives_from_all": assert_coordinator_receives_from_all,
    "frame_rate_above": assert_frame_rate_above,
    "max_boot_time": assert_max_boot_time,
    "no_heap_errors": assert_no_heap_errors,
    # fall_detected is parameterized, handled separately
}


def _parse_assertion_spec(spec: Any) -> tuple:
    """Parse a YAML assertion entry into (name, kwargs).

    Supported forms:
        - "all_nodes_boot"                      -> ("all_nodes_boot", {})
        - {"frame_rate_above": 15}              -> ("frame_rate_above", {"min_fps": 15})
        - "fall_detected_by_node_2"             -> ("fall_detected", {"node_id": 2})
        - {"max_boot_time_s": 10}               -> ("max_boot_time", {"max_seconds": 10})
    """
    if isinstance(spec, str):
        # Check for fall_detected_by_node_N pattern
        m = re.match(r"fall_detected_by_node_(\d+)", spec)
        if m:
            return ("fall_detected", {"node_id": int(m.group(1))})
        return (spec, {})

    if isinstance(spec, dict):
        for key, val in spec.items():
            m = re.match(r"fall_detected_by_node_(\d+)", str(key))
            if m:
                return ("fall_detected", {"node_id": int(m.group(1))})
            if key == "frame_rate_above":
                return ("frame_rate_above", {"min_fps": float(val)})
            if key == "max_boot_time_s":
                return ("max_boot_time", {"max_seconds": float(val)})
            if key == "coordinator_receives_from_all":
                return ("coordinator_receives_from_all", {})
            return (str(key), {})

    return (str(spec), {})


def run_assertions(
    logs: List[NodeLog],
    assertion_specs: List[Any],
    config: Optional[Dict] = None,
) -> List[AssertionResult]:
    """Run all requested assertions against loaded logs."""
    results: List[AssertionResult] = []

    # Derive coordinator/sensor IDs from config if available
    coordinator_id = 0
    sensor_ids: Optional[List[int]] = None
    if config and "nodes" in config:
        for node_def in config["nodes"]:
            if node_def.get("role") == "coordinator":
                coordinator_id = node_def.get("node_id", 0)
        sensor_ids = [
            n["node_id"] for n in config["nodes"]
            if n.get("role") == "sensor"
        ]

    for spec in assertion_specs:
        name, kwargs = _parse_assertion_spec(spec)

        if name == "fall_detected":
            results.append(assert_fall_detected(logs, **kwargs))
        elif name == "coordinator_receives_from_all":
            results.append(assert_coordinator_receives_from_all(
                logs, coordinator_id=coordinator_id, sensor_ids=sensor_ids,
            ))
        elif name == "all_nodes_produce_frames":
            results.append(assert_all_nodes_produce_frames(
                logs, sensor_ids=sensor_ids, **kwargs,
            ))
        elif name in ASSERTION_REGISTRY:
            fn = ASSERTION_REGISTRY[name]
            results.append(fn(logs, **kwargs))
        else:
            results.append(AssertionResult(
                name=name, passed=False,
                message=f"Unknown assertion: {name}",
                severity=1,
            ))

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: List[AssertionResult], swarm_name: str = "") -> int:
    """Print the assertion report and return max severity."""
    header = "QEMU Swarm Health Report (ADR-062)"
    if swarm_name:
        header += f" - {swarm_name}"

    print()
    print("=" * 60)
    print(f"  {header}")
    print("=" * 60)
    print()

    max_sev = 0
    for r in results:
        if r.severity == 0:
            icon = green("PASS")
        elif r.severity == 1:
            icon = yellow("WARN")
        else:
            icon = red("FAIL")

        print(f"  [{icon}] {r.name}: {r.message}")
        max_sev = max(max_sev, r.severity)

    print()
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    summary = f"  {passed}/{total} assertions passed"

    if max_sev == 0:
        print(green(summary))
    elif max_sev == 1:
        print(yellow(summary + " (with warnings)"))
    else:
        print(red(summary + " (with failures)"))

    print()
    return max_sev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QEMU Swarm Health Oracle (ADR-062)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 swarm_health.py --config scripts/swarm_presets/standard.yaml \\\n"
            "                          --log-dir build/swarm_logs/\n"
            "\n"
            "  python3 swarm_health.py --log-dir build/swarm_logs/ \\\n"
            "                          --assertions all_nodes_boot no_crashes\n"
            "\n"
            "Example output:\n"
            "  ============================================================\n"
            "    QEMU Swarm Health Report (ADR-062) - standard\n"
            "  ============================================================\n"
            "\n"
            "    [PASS] all_nodes_boot: All 3 nodes booted (timeout=10.0s)\n"
            "    [PASS] no_crashes: No crash indicators in any node\n"
            "    [PASS] tdm_no_collision: TDM slots unique across 3 assignments\n"
            "    [PASS] all_nodes_produce_frames: All 3 nodes show frame activity\n"
            "    [PASS] coordinator_receives_from_all: Coordinator received from all\n"
            "    [WARN] fall_detected_node_2: Node 2 did not report fall event\n"
            "    [PASS] frame_rate_above: All nodes meet minimum 15.0 fps\n"
            "\n"
            "    6/7 assertions passed (with warnings)\n"
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to swarm YAML config (defines nodes and assertions)",
    )
    parser.add_argument(
        "--log-dir", type=str, required=True,
        help="Directory containing node_0.log, node_1.log, etc.",
    )
    parser.add_argument(
        "--assertions", nargs="*", default=None,
        help="Override assertions (space-separated). Ignores YAML assertion list.",
    )
    parser.add_argument(
        "--node-count", type=int, default=None,
        help="Number of nodes (auto-detected from log files if omitted)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(2)

    # Load YAML config if provided
    config: Optional[Dict] = None
    swarm_name = ""
    yaml_assertions: List[Any] = []

    if args.config:
        if yaml is None:
            print("ERROR: PyYAML is required for --config. Install with: pip install pyyaml",
                  file=sys.stderr)
            sys.exit(2)
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(2)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        swarm_name = config.get("swarm", {}).get("name", "")
        yaml_assertions = config.get("assertions", [])

    # Determine node count
    if args.node_count is not None:
        node_count = args.node_count
    elif config and "nodes" in config:
        node_count = len(config["nodes"])
    else:
        node_count = _node_count_from_dir(log_dir)

    if node_count == 0:
        print("ERROR: No node logs found and node count not specified.", file=sys.stderr)
        sys.exit(2)

    # Load logs
    logs = load_logs(log_dir, node_count)

    # Determine which assertions to run
    if args.assertions is not None:
        assertion_specs = args.assertions
    elif yaml_assertions:
        assertion_specs = yaml_assertions
    else:
        # Default set
        assertion_specs = ["all_nodes_boot", "no_crashes", "no_heap_errors"]

    # Run assertions
    results = run_assertions(logs, assertion_specs, config)

    # Print report and exit
    max_sev = print_report(results, swarm_name)
    sys.exit(max_sev)


if __name__ == "__main__":
    main()
