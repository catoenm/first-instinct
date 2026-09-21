"""Supervise one CPU data job on macOS; never load a model or kill other apps.

RSS and pressure polling are coarse safeguards, not a GPU allocation guarantee.
Local foundation-model inference remains paused after the memory incident.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def swap_bytes(text):
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGT])", text)
    if not match:
        raise ValueError("Cannot read swap usage")
    return int(float(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1))


def host_state():
    pressure = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=5)
    swap = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True, timeout=5)
    return {"pressure": int(pressure.strip()), "swap_bytes": swap_bytes(swap)}


def group_rss(group):
    listing = subprocess.check_output(["/bin/ps", "-axo", "pgid=,rss="], text=True, timeout=5)
    return sum(int(parts[1]) * 1024 for line in listing.splitlines()
               if len(parts := line.split()) == 2 and int(parts[0]) == group)


def stop_reason(state, initial_swap, resident, elapsed, max_rss, max_swap_growth, max_seconds):
    if state["pressure"] != 1:
        return "system_memory_pressure"
    if state["swap_bytes"] - initial_swap > max_swap_growth:
        return "system_swap_growth"
    if resident > max_rss:
        return "owned_process_group_rss"
    if elapsed >= max_seconds:
        return "wall_clock_deadline"
    return None


def terminate_group(group):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # Some desktop runtimes forbid killpg even for our own fresh session.
        # Signal only the individually witnessed members of that owned group.
        listing = subprocess.check_output(["/bin/ps", "-axo", "pid=,pgid="], text=True, timeout=5)
        members = [int(parts[0]) for line in listing.splitlines()
                   if len(parts := line.split()) == 2 and int(parts[1]) == group]
        if not members:
            return
        for pid in members:
            try:
                if os.getpgid(pid) == group:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(0.5)


def run(command, output, *, max_rss=4*1024**3, max_swap_growth=512*1024**2, max_seconds=1800):
    if sys.platform != "darwin":
        raise ValueError("This local CPU guard requires macOS pressure telemetry")
    if not command or min(max_rss, max_swap_growth, max_seconds) <= 0:
        raise ValueError("A command and positive limits are required")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="preflight", scope="one CPU data job; not a model-memory qualification",
                  limits=dict(rss_bytes=max_rss, swap_growth_bytes=max_swap_growth, seconds=max_seconds),
                  peak_owned_group_rss_bytes=0, started_at=time.time())
    def save():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary.replace(output / "status.json")
    child = None
    start = time.monotonic()
    previous_termination = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        initial = host_state()
        record["initial_host"] = initial
        if initial["pressure"] != 1:
            record.update(status="refused", reason="system_memory_pressure")
            save()
            return 1
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0"}
        with (output / "stdout.log").open("x") as log, (output / "stderr.log").open("x") as errors:
            child = subprocess.Popen(command, stdout=log, stderr=errors, env=env, start_new_session=True)
            record.update(status="running", pid=child.pid)
            save()
            while child.poll() is None:
                state, resident = host_state(), group_rss(child.pid)
                elapsed = time.monotonic() - start
                record.update(last_host=state, elapsed_seconds=elapsed,
                              peak_owned_group_rss_bytes=max(record["peak_owned_group_rss_bytes"], resident))
                reason = stop_reason(state, initial["swap_bytes"], resident, elapsed,
                                     max_rss, max_swap_growth, max_seconds)
                if reason:
                    record.update(status="stopped", reason=reason)
                    terminate_group(child.pid)
                    break
                save()
                time.sleep(1)
            code = child.wait(timeout=10)
            if record["status"] == "running":
                record["status"] = "completed" if code == 0 else "child_failed"
            record["exit_code"] = code
    except BaseException as exc:
        record.update(status="failed_closed", reason=type(exc).__name__)
        raise
    finally:
        if child is not None:
            terminate_group(child.pid)
            child.wait(timeout=10)
        record["elapsed_seconds"] = time.monotonic() - start
        save()
        signal.signal(signal.SIGTERM, previous_termination)
    return 0 if record["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(run(command, args.output, max_seconds=args.max_seconds,
                         max_rss=args.max_rss_mib*1024**2))


if __name__ == "__main__":
    main()
