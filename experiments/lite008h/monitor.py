"""Bounded child-process monitoring for LITE-008H.

Child stdout/stderr are redirected so verbose native decoder diagnostics cannot
fill a pipe and create a false resource-monitor hang.
"""
from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path


def run_monitored(command: list[str], timeout_seconds: float) -> dict:
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    peak_rss = 0
    peak_pss = 0
    timed_out = False
    while True:
        rc = child.poll()
        status = Path(f"/proc/{child.pid}/status")
        if status.exists():
            try:
                text = status.read_text()
                peak_rss = max(peak_rss, int(text.split("VmRSS:")[1].split()[0]) * 1024)
            except (OSError, IndexError, ValueError):
                pass
        rollup = Path(f"/proc/{child.pid}/smaps_rollup")
        if rollup.exists():
            try:
                text = rollup.read_text()
                peak_pss = max(peak_pss, int(text.split("Pss:")[1].split()[0]) * 1024)
            except (OSError, IndexError, ValueError):
                pass
        if rc is not None:
            break
        if time.monotonic() - started > timeout_seconds:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=1.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            rc = child.returncode
            break
        time.sleep(0.01)
    return {
        "exit_status": rc,
        "timed_out": timed_out,
        "peak_rss_bytes": peak_rss,
        "peak_pss_bytes": peak_pss,
        "wall_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    import sys
    result = run_monitored([sys.executable, "-c", "import time; time.sleep(.1)"], 2.0)
    print(result)
